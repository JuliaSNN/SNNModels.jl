# Performance notes

## Event-driven STDP (branch event-stdp, 2026-10-04)

### What was done

- `STDPGerstner`, `STDPConfavreux2025`, the new `STDPWeightDependent` and the new
  `STDPTriplet` share Auryn-style event-driven loops (`sparse_plasticity/STDP_kernels.jl`):
  1. presynaptic spikes walk their outgoing synapses (`colptr`);
  2. postsynaptic spikes walk their incoming synapses (`rowptr`/`index`);
  3. only touched weights are clamped (plus one clamp of all weights on the first step);
  4. traces are incremented by 1 after the weight passes and multiplied by a precomputed
     `exp(-dt/τ)`, with no per-neuron `exp`.

  Cost per step is O(N) multiplies plus O(spikes x fan-out). Before, it was O(nnz) plus
  2N `exp` calls.
- `Threads.@threads` was removed from STDPGerstner/STDPConfavreux2025. The new loops are
  serial. A naive split over presynaptic spikes would race on shared postsynaptic rows.
- STDPMexicanHat: event-driven passes and touched-only clamp. The rule is unchanged.
- STDPSymmetric/STDPAntiSymmetric: touched-only clamp instead of clamping all of `W`
  every step.
- Float32: `sparse_matrix` always returns `SparseMatrixCSC{Float32}`. `SpikingSynapse`
  builds `ρ`, delays and delay queues in Float32, and `RateSynapse` converts its matrix.
  The `@snn_kw` structs already converted fields to Float32, so this mainly removes
  Float64 temporaries and the Float64 return type of `sparse_matrix`.

### Measured (seroquel, Julia 1.12.6, `bench/stdp_event_bench.jl`)

4000-neuron `Identity` population, recurrent p = 0.02 (320,000 synapses), dt = 0.125 ms,
forced Bernoulli spikes, 10^4 steps, best of 3, cost of copying the spike mask subtracted.
Times are microseconds per step for `plasticity!` alone.

| rule | rate | old, 1 thread | old, 24 threads | new, 1 thread | speedup vs old 1 thread / 24 threads |
|---|---|---|---|---|---|
| STDPGerstner | 5 Hz | 535.9 | 189.2 | 6.4 | 83x / 29x |
| STDPGerstner | 10 Hz | 543.2 | 190.8 | 7.5 | 72x / 25x |
| STDPGerstner | 20 Hz | 552.4 | 193.3 | 9.6 | 58x / 20x |
| STDPMexicanHat | 5 Hz | 698.3 | - | 17.0 | 41x |
| STDPMexicanHat | 10 Hz | 710.8 | - | 25.3 | 28x |
| STDPMexicanHat | 20 Hz | 744.8 | - | 41.0 | 18x |
| STDPTriplet (new) | 5 / 10 / 20 Hz | - | - | 9.8 / 11.0 / 13.6 | |
| STDPWeightDependent (new) | 5 / 10 / 20 Hz | - | - | 14.6 / 17.1 / 22.9 | |

Full `train!` step on a 4000-neuron Poisson population projecting to a 4000-neuron IF
population with STDPGerstner (320,000 synapses), 1 s of simulated time:

- old, 1 thread: 4.59 s
- old, 24 threads: 1.85 s
- new, 1 thread: 0.21 s, against 0.17 s for the same network without plasticity

Plasticity overhead in this network went from about 4.4 s to about 0.05 s per simulated
second. The IF population's own rate was not measured; the input rate barely matters for
the new code.

Correctness:
- `test/syn/stdp_event.jl` runs the new kernels in lockstep with the old ones, kept in
  `test/syn/stdp_reference.jl` with the two quirks turned off. Relative weight differences
  are 2e-6 (Gerstner), 1e-6 (Confavreux) and 1.5e-7 (MexicanHat).
- Cross-simulator validation is in the umbrella repo,
  `papers/JuliaSNN_publication/validation/stdp/`: agreement within 2e-7 with Auryn and
  2e-6 with Brian2 (float64).

### Behaviour changes, deliberate

- STDPGerstner: traces now increment by 1 and A is applied once. Before, the effective
  amplitude was A^2 and its sign was lost, so a negative `A_post` produced potentiation.
  The default `A_post` is now -1e-4 (LTD).
- Same-step pre/post spikes: traces are read before this step's increments, as in Auryn
  and Brian2. The old code ignored the whole trace of any neuron that fired in the current
  step, which made a 6 % weight difference at 160 Hz forced rates in the lockstep test.

### Remaining known performance issues

- Delay queue in `forward!(::SpikingSynapse, ::SpikingSynapseDelayParameter)`: per-neuron
  sorted `Vector`s with `findlast(.<(spike), times)` (allocating broadcast) and
  `insert!`/`popfirst!` for every delayed spike. A ring buffer of `Npost x max_delay_steps`
  would make this O(1) per spike and allocation-free.
- ~~`sparse_matrix` builds a dense `Npost x Npre` matrix~~: fixed on core-clock-sparse,
  see below.
- Plasticity runs only under `train!`; `sim!` never calls `update_traces!`/`plasticity!`.
- Lazy-trace `exp` per neuron per step: removed from the four Scheme-A rules. Euler traces
  remain in iSTDP, vSTDP, STDPMexicanHat and the structured rules (O(N) per step, cheap).
- The core `sim!`/`train!` loop and `forward!` are single-threaded. Parallel plasticity
  needs a partition by postsynaptic neuron for the post pass and by presynaptic neuron for
  the pre pass, or atomic-free colouring. This is the next step.
- The time accumulator `T.t[1] += dt` is Float32 and drifts. At dt = 0.1 ms it is off by
  more than half a step after about 0.8 s, and by much more over long runs. Time-based
  stimuli (`SpikeTimeStimulus`) then land on the wrong step. Work in progress on the
  local branch `wip-clock` (integer clock).

## Dense-free `sparse_matrix` (branch core-clock-sparse, 2026-10-04)

### What was done

- `sparse_matrix(Npre, Npost; ...)` builds `SparseMatrixCSC{Float32,Int}` directly:
  - `:Bernoulli`: geometric skip sampling over the column-major flattened index space,
    gap = floor(log(U) / log1p(-p)). O(nnz) draws, rows sorted by construction.
  - `:Fixed`/`:FixedIn`: K presynaptic indices per postsynaptic row, then a counting sort
    on the column to scatter into CSC.
  - `:FixedOut`/`:PowerLaw`: per-column distinct targets, then sorted.
  - Distinct sampling is a partial Fisher-Yates on a reused permutation buffer: O(K) per
    row/column and no per-call allocation (StatsBase `sample!(...; replace=false)` allocated
    a length-n index vector per call: 3.7 GB at 2e4 x 2e4, p = 0.1).
  - Weights: one Float32 draw per stored entry from `dist(|μ|, σ)`; draws <= 0 removed by
    in-place CSC compaction (`_filter_csc!`), sign applied afterwards. Same treatment as
    the dense path.
- Autapses: `remove_autapses!` removes the diagonal structurally. The old
  `w[diagind(w)] .= 0` left stored zeros, i.e. zero-weight synapses that plasticity could
  grow.
- `SpikeTimeStimulusIdentity` used `Matrix(I(N))` (dense N x N Bool); now a sparse identity.
- PoissonStimulusLayer had no dense matrix of its own; it only went through `sparse_matrix`.
- The old generator is kept as `SNNModels.sparse_matrix_dense_legacy` (not exported) for
  the statistical comparison tests (`test/utils/sparse_matrix_gen_test.jl`).

### Measured (seroquel, Julia 1.12.6, single thread, first call after warm-up)

| Npre = Npost | rule, p, (μ, σ) = (1, 0.2) | nnz   | old time | old alloc | new time | new alloc | result size |
|--------------|----------------------------|-------|----------|-----------|----------|-----------|-------------|
| 2e4          | Bernoulli, 1e-3            | 4.0e5 | 13.5 s   | 19.9 GiB  | 0.01 s   | 5 MiB     | 4.8 MiB     |
| 2e4          | Fixed, 0.1                 | 4.0e7 | 11.4 s   | 13.0 GiB  | 0.80 s   | 0.75 GiB  | 458 MiB     |
| 1e5          | Bernoulli, 1e-3            | 1.0e7 | not run (dense 80 GB) | > 80 GB | 0.22 s | 0.11 GiB | 115 MiB |
| 1e5          | Fixed, 1e-3                | 1.0e7 | not run (dense 80 GB) | > 80 GB | 0.26 s | 0.19 GiB | 115 MiB |

The new allocation is the result itself plus, for the fixed in-degree rule, one transient
column-index vector of length nnz.
