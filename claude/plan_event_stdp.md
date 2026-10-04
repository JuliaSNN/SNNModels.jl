# Plan: event-driven STDP (Auryn style), triplet and weight-dependent STDP, Float32 synaptic data

Branch: `event-stdp` (new, from `dev`), SNNModels.jl submodule. Local commits only, no push.
This file is a working document and is not committed.

## Goal

Rewrite the trace-based pair STDP rules of SNNModels so that the cost of a plasticity step is
O(spikes x fan-out) rather than O(nnz): weights are updated only along the outgoing row of a
presynaptic spike (CSC `colptr`) and the incoming row of a postsynaptic spike (transposed
`rowptr`/`index`), and only the touched weights are clamped (the Auryn `STDPConnection`
algorithm). Fix the STDPGerstner amplitude bug (trace incremented by A and weight multiplied by A
again, effective amplitude A^2 with sign lost). Add two new rules in the same structure: the
Pfister-Gerstner (2006) triplet rule and a weight-dependent (soft-bound) pair rule. Make Float32 the
only element type produced by the connectivity constructors. Validate against analytic results,
Brian2 2.9.0 and Auryn with identical forced spike trains. Single-threaded; parallelisation is a
later step.

## Findings from inspection (no codegraph index for this submodule; inspected with Read/Grep)

1. `STDPGerstner` / `STDPConfavreux2025` (`src/connections/sparse_plasticity/STDP_traces.jl`):
   traces are lazy (`tpre`, `last_pre`), but every step recomputes `Δpre`/`Δpost` for all neurons
   with one `exp` each, then scans all nnz synapses under `Threads.@threads` and clamps every weight.
2. Amplitude bug (confirmed): STDPGerstner increments `tpre += A_pre`, `tpost += A_post` and then
   does `W += A_pre * Δpre`, `W += A_post * Δpost`. Effective amplitudes are `A_pre^2` and `A_post^2`;
   with a negative `A_post` (classical LTD) the depression branch becomes potentiation, so the old
   rule can never depress. STDPConfavreux2025 increments by 1 and multiplies by η once: no bug.
3. Same-step coincidence quirk (new finding, to be reported): the old code sets `Δpre[j] = 0` for
   any presynaptic neuron that fires in the current step (and `Δpost[i] = 0` likewise), so when pre
   j and post i fire in the same step, the LTP and LTD contributions of their *earlier* spikes are
   also dropped for that step. Auryn and Brian2 instead read the trace value before this step's
   increment (earlier history kept, same-step pair ignored).
4. `STDPMexicanHat`: weight update loops scan all rows and all columns (O(nnz)) testing `fireJ`/
   `fireI` inside, then clamp all W. `MexicanHat(x)` returns `Int 0` on NaN (type instability).
5. `STDPSymmetric`/`STDPAntiSymmetric` (`STDP_structured.jl`): weight updates already event-driven,
   but a full `clamp` over all W every step (O(nnz)).
6. `iSTDPRate/iSTDPTime/iSTDPPotential`: already event-driven with touched-only clamp (traces Euler
   per step). Not modified. Suspicious but out of scope, to be reported: `@turbo for st = rowptr...;
   st = index[st]` rebinds the loop variable inside `@turbo`.
7. `vSTDPParameter` (Clopath): LTP depends on the continuous post voltage every step; it cannot be
   event-driven. Not modified.
8. `CaRule.jl`, `dump.jl` and `_new_feats/WIPtriplet.jl` are NOT included by the package (they
   define a duplicate `STDPVariables`). Not touched.
9. Plasticity runs only in `train!` (order per connection: `update_traces!` -> `forward!` ->
   `plasticity!`, after all populations integrated, so `fireI`/`fireJ` hold this step's spikes);
   `sim!` never calls plasticity. Unchanged, documented in performance.md.
10. Float32: the `@snn_kw` keyword constructors already convert every field to the default
    parameter type (`VFT = Vector{Float32}`), so `SpikingSynapse.W`, `ρ`, `delaytime`, and all STDP
    variables are already Float32 at runtime, even with `conn = (p=0.4, μ=0.5, σ=0.1)` or a Float64
    `conn` matrix (verified). Float64 survives only in: the return value of the exported
    `sparse_matrix` (Float64 when μ/σ are Float64/Int; the dense Npost x Npre work matrix is then
    Float64, double memory), `rand(delay_dist, n)` (Float64 temporary), `ρ .= 1.0`, `spike_time =
    [[] ...]` (Vector{Any} temporary), `RateSynapse` (`sprandn` Float64). Those are the places to fix.
11. Baseline: `test/syn/plasticity_params.jl` + `test/syn/with_plasticity.jl` pass 88/88
    (Julia 1.12, env `papers/JuliaSNN_publication`, `-t 1`).

## Usages of STDPGerstner whose behaviour changes with the amplitude fix (not re-tuned)

| Location | A values | Old effective (LTP, LTD) | New (LTP, LTD) |
|---|---|---|---|
| defaults (`STDPGerstner()`): test/syn/with_plasticity.jl, plasticity_params.jl, test/pop/spiketime.jl | A_pre = A_post = 1e-4 | +1e-8, +1e-8 | +1e-4, +1e-4 (both potentiating, see note) |
| `SpikingNeuralNetworks.jl/examples/tutorials/STDP_kernel.jl` (1st) | 5e-2, -5e-2 | +2.5e-3, **+2.5e-3** | +5e-2, -5e-2 |
| same file (2nd, rate heatmap) | 1, -1 | +1, **+1** | +1, -1 |
| `SNNPlots.jl/src/stdp_plots.jl` `stdp_test`/`stdp_kernel` | whatever is passed | kernel plotted as A^2 | kernel plotted as A |

No uses in SNNUtils.jl or the paper folder code. Note: the default `A_post` is positive, so even
after the fix the default STDPGerstner has no depression; I will NOT change the default, only flag
it in the docstring and the report (decision for you, see D6).

## New module/subsystem boundary

All inside `src/connections/sparse_plasticity/`. `STDP_traces.jl` is rewritten in place (same
types, same `plasticity!(c, param, vars, dt, T)` dispatch, same exports); two new files add the new
rules and are included from `sparse_plasticity.jl` after `STDP_traces.jl`. A small shared kernel
file holds the event loops so the three pair-type rules do not duplicate them.

## Algorithm and ordering convention (documented in a comment block at the top of the new kernel file)

Per call of `plasticity!` at step n (time t_n), with `fireJ`/`fireI` the spikes emitted at t_n:

1. LTD / pre-spike pass: for each j with `fireJ[j]`, for s in `colptr[j]:colptr[j+1]-1`, i = I[s]:
   `W[s] += f_pre(W[s], post traces of i)`; clamp `W[s]`.
2. LTP / post-spike pass: for each i with `fireI[i]`, for st in `rowptr[i]:rowptr[i+1]-1`,
   s = index[st], j = J[s]: `W[s] += f_post(W[s], pre traces of j, post traces of i)`; clamp `W[s]`.
3. Trace increment: `x_pre[j] += 1` for firing j, `x_post[i] += 1` for firing i
   (and `last_pre[j] = t`, `last_post[i] = t`, kept as informational spike times).
4. Trace decay: `x .*= exp(-dt/τ)` for every trace (one multiply per neuron; the two/four scalar
   `exp` factors are computed once per call, no per-neuron `exp`).

Traces are read in 1-2 *before* this step's increment, so a trace read at step n equals
`sum_{m<n} exp(-(t_n - t_m)/τ)` over earlier spikes m, i.e. exactly the old lazy formula, and the
same-step pre/post pair contributes nothing (Auryn/Brian2 semantics, see D1). This is Auryn's
`System::run` order: evolve neurons -> propagate (forward + plasticity) -> evolve_traces.
Trace representation is the Auryn `EulerTrace` one (decayed vector, multiplicative exact factor).
Presynaptic timing is the somatic spike (`fireJ`), also for `SpikingSynapseDelayParameter`
(unchanged from current behaviour; Auryn uses the delayed spike, mapped in validation).
On the first call after `plasticityvariables` is created, all weights are clamped once (flag in the
variables), reproducing the old "clamp everything" behaviour for initial weights outside bounds
without paying O(nnz) per step.

## Rules (exact formulae, also in the docstrings)

- `STDPGerstner{FT}` (unchanged fields `A_pre, A_post, τpre, τpost, Wmax, Wmin`), additive:
  pre spike: `w += A_post * x_post[i]`; post spike: `w += A_pre * x_pre[j]`; traces +1.
  Kernel for one pair, Δt = t_post - t_pre: `A_pre*exp(-Δt/τpre)` for Δt > 0,
  `A_post*exp(Δt/τpost)` for Δt < 0, 0 for Δt = 0. A_post must be negative for LTD.
- `STDPConfavreux2025{FT}` (unchanged fields): pre: `w += η(κ x_post[i] + α)`;
  post: `w += η(γ x_pre[j] + β)`. Same event loops.
- `STDPWeightDependent{FT}` (new; Gütig et al. 2003 / Morrison 2008, identical to Auryn
  `STDPwdConnection` when Wmin = 0): fields `η, α, μ_plus, μ_minus, τpre, τpost, Wmax, Wmin`.
  With `w̃ = W - Wmin`, `W̃max = Wmax - Wmin`:
  pre spike (LTD): `w -= η α W̃max^(1-μ_minus) * w̃^μ_minus * x_post[i]`;
  post spike (LTP): `w += η W̃max^(1-μ_plus) * (Wmax - w)^μ_plus * x_pre[j]`; then clamp.
  μ = 0 gives additive STDP with hard bounds, μ = 1 multiplicative (LTD ∝ w, LTP ∝ Wmax - w).
  Defaults as Auryn: τ = 20 ms, α = 1, μ_plus = μ_minus = 1. Uses `STDPVariables`.
- `STDPTriplet{FT}` (new; Pfister & Gerstner 2006, all-to-all; Auryn `MinimalTripletConnection` when
  A3_minus = 0): fields `A2_plus, A3_plus, A2_minus, A3_minus, τ_plus, τ_minus, τ_x, τ_y, Wmax, Wmin`,
  amplitudes positive, signs explicit:
  pre spike: `w -= o1[i] * (A2_minus + A3_minus * r2[j])`;
  post spike: `w += r1[j] * (A2_plus + A3_plus * o2[i])`; all four traces read before increment
  (r2 and o2 "at t - ε"). Variables `STDPTripletVariables`: `r1, r2` (Npre), `o1, o2` (Npost),
  `last_pre, last_post, active`. Defaults: Pfister-Gerstner visual-cortex all-to-all minimal set
  (A2_plus = 5e-10, A3_plus = 6.2e-3, A2_minus = 7e-3, A3_minus = 0, τ_plus = 16.8 ms, τ_minus = 33.7 ms,
  τ_x = 101 ms, τ_y = 125 ms); I will check the numbers against Table 4 of the paper (Zotero) before
  committing and state the source in the docstring.
- `STDPMexicanHat`: same formula and Euler trace update order (its definition), but the weight
  passes run only over spiking neurons (colptr for pre spikes, rowptr/index for post spikes), both
  passes applied before clamping the touched synapses (preserves the old double-update semantics
  bit-for-bit); `isnan ? 0f0`. Clamp touched only.
- `STDPSymmetric`/`STDPAntiSymmetric`: only replace the full-W clamp by a touched-only clamp (loops
  over all neurons testing `fire` kept, they are O(N) not O(nnz)). Formulae unchanged.

`Threads.@threads` is removed from STDPGerstner/STDPConfavreux2025 (the new loops are event-driven
and serial; a thread split over pre-spikes would race on shared post rows, deferred to the
parallelisation step). No threading is added anywhere.

## Float32 enforcement

- `sparse_matrix(Npre, Npost; ...)`: allocate the work matrix as `Matrix{Float32}` and fill with
  `rand!(dist(μ, σ), w)` (or `Float32.(...)` if a distribution does not support it), return
  `SparseMatrixCSC{Float32}` for any μ/σ type. `sparse_matrix(Npre, Npost, conn::AbstractMatrix)`
  returns `sparse(Float32.(conn))`. Blast radius checked: callers are SpikingSynapse, HetRec,
  SpikeTimeStimulus (timed.jl), PoissonLayer (two methods); all store into Float32 `VFT` fields
  already, so only the transient type changes. `matrix(c)` already returns c.W's Float32.
- `SpikingSynapse`: `delaytime = Float32.(rand(delay_dist, n))`, `ρ = ones(Float32, n)`,
  `spike_time/spike_w = [Float32[] for _ ...]`.
- `RateSynapse`: `Float32` conversion of the `sprandn` matrix.
- `dsparse` stays generic. Parameter structs already default to `FT = Float32` and convert.

## Files

- `src/connections/sparse_plasticity/STDP_kernels.jl` — new — ordering comment block, shared
  helpers: `_decay!(x, factor)`, `_increment!(x, last, fire, t)`, `_clamp_touched_pre!`,
  `_clamp_touched_post!`, `_initial_clamp!`.
- `src/connections/sparse_plasticity/STDP_traces.jl` — modified — STDPVariables (fields `tpre,
  tpost` now decayed traces, `last_pre, last_post` kept, `Δpre, Δpost` removed, `initialized` flag
  added), event-driven STDPGerstner (amplitude fixed), STDPConfavreux2025, STDPMexicanHat.
- `src/connections/sparse_plasticity/STDP_weight_dependent.jl` — new — STDPWeightDependent.
- `src/connections/sparse_plasticity/STDP_triplet.jl` — new — STDPTriplet, STDPTripletVariables.
- `src/connections/sparse_plasticity/STDP_structured.jl` — modified — touched-only clamp.
- `src/connections/sparse_plasticity.jl` — modified — includes.
- `src/utils/sparse_matrix.jl`, `src/connections/spiking_synapse.jl`,
  `src/connections/rate_synapse.jl` — modified — Float32.
- `test/syn/stdp_reference.jl` — new, not exported, test-only — verbatim copy of the old
  clock-driven STDPGerstner / STDPConfavreux2025 / STDPMexicanHat kernels operating on a plain
  NamedTuple of vectors, with two switches `amplitude_quirk::Bool` and `coincidence_quirk::Bool`
  (both `true` = the old code exactly).
- `test/syn/stdp_event.jl` — new — tests (below).
- `test/syn/float32.jl` — new — eltype tests.
- `test/runtests.jl` — modified — register the two new test files.
- `test/syn/plasticity_params.jl` — unchanged (fields it checks are kept).
- `claude/LIBRARY_MAP.md`, `docs/stdp_rules_memo.md` — modified — the lines describing STDPVariables
  / Δpre / threaded scan. (The repo's notes folder is `claude/` lowercase; the brief said
  `CLAUDE/performance.md`. I will create `claude/performance.md` to stay in the existing folder
  unless you prefer a new `CLAUDE/`, see D5.)
- `claude/performance.md` — new — what was done, measured numbers, remaining issues.
- `bench/stdp_event_bench.jl` (or `test/simulation_speed`-style script, not run in the suite) — new.

Outside the submodule (validation, see below):
`/home/aquaresi/Documents/Research/projects/JuliaSNN/papers/JuliaSNN_publication/validation/stdp/`
and `.../sims/auryn/`. `papers/` is untracked in the umbrella repo, so these are written but not
committed anywhere (D7).

## Tests (SNNModels, run with `-t 1` in env papers/JuliaSNN_publication)

1. Equivalence: 300-neuron recurrent IF network (p = 0.1) driven to fire, stepped manually with the
   single-step `train!(P, C, S, dt, T)`; after each step the reference kernel is applied to a copy
   `W_ref` with the same `fireI/fireJ`. Spikes are identical because `W_ref` is not fed back.
   For STDPGerstner and STDPConfavreux2025: new vs reference with `amplitude_quirk = false,
   coincidence_quirk = false`: `isapprox(W, W_ref; rtol = 1e-4)` (residual = decayed-vector vs
   lazy exp in Float32). For STDPMexicanHat: new vs reference unchanged code: exact up to 1e-6.
   Plus a check that the reference with both quirks true differs (documents the fix).
2. Fixed amplitude: two Identity neurons with forced spikes (SpikeTimeStimulusIdentity), single pair
   Δt = ±10 ms, A_pre = 0.01, A_post = -0.012: Δw = A_pre e^{-10/τpre} and A_post e^{-10/τpost}
   (rtol 1e-5), sign of LTD negative; Δt = 0 gives 0.
3. Kernel sweep: Δt in -100:2.5:100 ms (grid multiples), compare with the analytic kernel.
4. Triplet: pre-post-post (pre at 0, posts at Δ and Δ+T) and post-pre-post analytic predictions:
   pre-post-post: Δw = A2_plus e^{-Δ/τ+} + e^{-(Δ+T)/τ+}(A2_plus + A3_plus e^{-T/τy});
   post-pre-post (post at 0, pre at Δ1, post at Δ1+Δ2): Δw = -A2_minus e^{-Δ1/τ-} +
   e^{-Δ2/τ+}(A2_plus + A3_plus e^{-(Δ1+Δ2)/τy}); with A3_minus ≠ 0 also a post-pre-pre case.
5. Weight-dependent: single pair at several initial w, Δw matches the formula; μ = 0 equals
   STDPGerstner with A_pre = η, A_post = -ηα; weights stay in bounds under long random driving.
6. Float32: for μ/σ given as Int, Float64, Float32, and a Float64 `conn` matrix and a Float64
   `Uniform` delay distribution: `eltype(sparse_matrix(...)) == Float32`,
   `eltype(syn.W) == eltype(syn.ρ) == eltype(syn.param.delaytime) == Float32`,
   eltypes of all STDP variable vectors Float32.
7. Existing `with_plasticity.jl` and `plasticity_params.jl` rerun (must stay 88/88; plus whatever
   triplet/weight-dependent testsets I add to with_plasticity.jl).
8. Benchmark (script, numbers into performance.md): 4000 neurons, p = 0.02 (3.2e5 synapses),
   dt = 0.125 ms, Bernoulli-forced fire vectors at 5, 10, 20 Hz, 10^4 steps, `-t 1`; time of
   `plasticity!` alone (old reference vs new) and of a full `train!` step; median of several runs
   with `@elapsed` after warm-up (BenchmarkTools if present in the env).

## Validation (cross-simulator), `papers/JuliaSNN_publication/validation/stdp/`

Common protocol, generated once by `make_spikes.py` (numpy, seed 1234) and written to
`spikes_pre.txt`, `spikes_post.txt` (`time_s neuron_id`, 0-based ids), `params.json`:
20 pre neurons, 10 post neurons, all-to-all (200 synapses, no connectivity file needed), 20 s at
dt = 0.1 ms (Auryn's fixed timestep, so all three run on the same grid), ~2000 spikes total, spike
times drawn as integer step indices (no rounding ambiguity). Pre and post spikes never share a step
(enforced at generation) for the Brian2 comparison; a second protocol `*_coinc` with coincidences
is used only for analytic/Auryn comparison. Initial w = Wmax/2 and amplitudes chosen so no weight
hits a bound (clipping is tested separately), except one run with bounds active.

(a) Analytic: `analytic.jl` — pair kernel sweep, triplet configurations, weight-dependent single
    pairs (same as tests 2-5 but at the validation dt), plotted to `analytic_kernel.png`.
(b) Brian2: `brian2_stdp.py` run with `.venv/bin/python`: two `SpikeGeneratorGroup`s, `Synapses`
    with event-driven traces `dapre/dt = -apre/taupre (event-driven)` etc.; pair:
    `on_pre: w = clip(w + A_post*apost, wmin, wmax); apre += 1`,
    `on_post: w = clip(w + A_pre*apre, wmin, wmax); apost += 1`; triplet with `r1, r2, o1, o2`
    and the "before increment" ordering written explicitly; weight-dependent with the formula above.
    `StateMonitor(w)` every 10 ms. Outputs `brian2_<rule>_w.npy`, `brian2_<rule>_traj.npy`.
(c) Auryn: `sims/auryn/sim_stdp_validation.cpp`: two `FileInputGroup`s reading the same ras files,
    `STDPConnection`, `MinimalTripletConnection`, `STDPwdConnection` (selected by CLI flag), all-to-all
    with w0, `WeightMonitor` of all 200 synapses every 10 ms, final `write_to_file`. Built against
    the existing `auryn/src/build/release/src/libauryn.a` with a small Makefile in `sims/auryn/`
    using the include/link flags from the existing CMakeCache (no edits inside the auryn tree).
    Time mapping: Auryn delivers presynaptic spikes to plasticity after the group delay (MINDELAY =
    8 steps = 0.8 ms) and post spikes immediately; traces are incremented and decayed after
    propagate. Therefore the Auryn pre ras file contains the pre times shifted by -8 steps (all
    protocol pre times start after 1 ms), which makes Auryn's effective pre times identical to
    JuliaSNN/Brian2's. FileInputGroup rounds `t/dt`; integer-step times make it exact. Auryn
    parameter mapping: STDPConnection has A = -η, B = η (so A_pre = η, A_post = -η, Wmin = 0);
    MinimalTriplet has A2_plus = 5.3e-3 η, A3_plus = 8e-3 η, A2_minus = 3.5e-3 η, τ = 16.8/33.7/40 ms,
    A3_minus = 0 (our struct is set to those values); STDPwd τ = 20 ms, α, μ±, Wmin = 0.
(d) JuliaSNN: `juliasnn_stdp.jl` drives `Identity` populations with `SpikeTimeStimulusIdentity`
    from the same files (a measured constant step offset between stimulus time and `fire` is applied
    identically to pre and post, so relative timing is exact), `train!` with dt = 0.1 ms, records W
    every 10 ms, writes `julia_<rule>_w.txt`/`_traj.txt`.
(e) `compare.py`: per-synapse final weights and trajectories, max abs and max rel difference, one
    figure per rule. Expected: JuliaSNN vs Auryn agree to Float32 rounding (both float32, same
    multiplicative decay); vs Brian2 (float64, exact exponentials) to ~1e-5 relative; any residual of
    order one dt (e.g. coincidence handling, delay mapping) is reported, not hidden.
(f) `README.md`: what each script does, exact commands to rerun (generate -> brian2 -> auryn
    build+run -> julia -> compare), versions used.

## Checkpoints (commits on `event-stdp`)

1. `feat: Float32 at the connectivity constructor boundary` — sparse_matrix, SpikingSynapse,
   RateSynapse + test/syn/float32.jl. Baseline tests rerun.
2. `test: keep old clock-driven STDP kernels as reference` — test/syn/stdp_reference.jl.
3. `feat: event-driven pair STDP (Auryn ordering), fix STDPGerstner A^2 amplitude` — kernel file,
   STDP_traces.jl (Gerstner, Confavreux, MexicanHat), STDP_structured clamp, equivalence + kernel
   tests.
4. `feat: weight-dependent and triplet STDP` — two new files + tests.
5. `perf/docs: benchmark script, claude/performance.md, LIBRARY_MAP and memo updates`.
Validation scripts written and run after checkpoint 4 (outside the repo); results go into the
report and into performance.md/README.

## Dependencies

All present: SNNModels deps (Distributions, SparseArrays, UnPack), Brian2 2.9.0 + numpy 1.26 in the
paper venv, Auryn static lib already built (needs Boost MPI/serialization/filesystem/program_options
as in its CMakeCache; to be confirmed at build time). No new Julia package added to SNNModels.
BenchmarkTools only if already in the paper env; otherwise `@elapsed` loops.

## Out of scope

Threading/parallel plasticity; plasticity under `sim!`; the delay queue in `forward!`
(insert!/findlast allocations); the dense Npost x Npre work matrix in `sparse_matrix` (only made
Float32 here, still dense); iSTDP/vSTDP/STP rewrites; Zenke homeostatic triplet; `CaRule.jl`,
`dump.jl`, `_new_feats/WIPtriplet.jl`; changing any default parameter value; pushing.

## Decisions for you (my recommendation first)

- D1 Coincidence semantics: adopt read-before-increment (Auryn/Brian2; same-step pair contributes 0,
  earlier history kept) for STDPGerstner and STDPConfavreux2025. Alternative: keep the old
  "zero the whole trace of a neuron that fires this step". Recommend D1 = Auryn.
- D2 Remove `Δpre`/`Δpost` from `STDPVariables` (only used as per-step scratch). Recommend remove.
- D3 Trace representation: decayed vectors (Auryn EulerTrace, one multiply per neuron per step,
  no exp) instead of lazy per-neuron exp. `tpre/tpost` then hold the current trace value.
  Recommend decayed vectors.
- D4 Weight-dependent rule = Auryn STDPwd / Gütig form above (generalised to Wmin ≠ 0).
- D5 performance notes in existing `claude/performance.md` (lowercase folder) rather than a new
  `CLAUDE/`. Recommend `claude/`.
- D6 Leave STDPGerstner default `A_post = +1e-4` (no depression by default) unchanged and only flag it.
- D7 Validation and Auryn sim files live in the untracked `papers/` tree, not committed.
