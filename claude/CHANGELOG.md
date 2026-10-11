# SNNModels changelog (branch notes)

## 1.10 (unreleased): sparse spike lists, ring-buffer delays, always-lazy trace decay
Faster event-driven STDP and delayed transmission: fewer scans of the spike flags, type-stable
plasticity calls, no per-step passes over the neurons when a `SpikeTimeStimulus` drives the network.
Weights of the event-driven STDP rules are no longer bit-identical to 1.9.1 (lazy trace decay, differences up
to about 1e-5 relative, see "Behaviour notes"); the other models checked are, except delayed transmission at a
`dt` that is not exactly representable in Float32. Breaking changes in the types (listed with their migration below);
saved models from 1.9.1 or earlier cannot be used (`load_model` throws an `ArgumentError` at load time, see "Saved models and parameters"). `vSTDPParameter` is now the event-driven voltage-based rule (the dense one
is `vSTDPDenseParameter`) and its weights are read through `plasticity_value(syn, :W)`; the library runs in Float64 when the
populations are built with `Float64` state (see "Precision from the population type").

### Performance
- Library micro-benchmark (`bench/stdp_event_bench.jl`, pair STDP `train!` step, us/step): 1e4 synapses
  1.571 to 0.373, 200 synapses 0.42 to 0.14 (spike lists, inlined `record!`, concrete types); with the
  sparse stimulus hand-off 0.15 and 0.08.
- Validation task (`papers/JuliaSNN_publication/validation/stdp`, pair rule, 20 s of simulated time, one thread
  active, median of 3): 200 synapses 0.093 s to 0.0136 s, 1e4 synapses 0.353 s to
  0.020 s (pinned core; traces always lazy, 0.017 s and 0.032 s with the previous dense decay). Same task against compiled Brian2 and Auryn: see the README of that directory.
- Delayed synapses (audit measurement, 1000 x 1000 neurons, p = 0.1, `delay_dist = Uniform(0.5, 5)`, `sim!` of 1 s):
  0.119 s to 0.046 s.
- The pair-STDP `train!` step with `Identity` populations driven by `SpikeTimeStimulus` allocates 0 B (was 336 B). Allocation per `train!`
  step measured in the independent review (bytes, 1.9.1 to 1.10): delayed `IF` network 17000 to 240; STP event 272 to 240;
  `vSTDPParameter` 848 to 240; balanced `sim!` 596 to 384; `AdEx` 0 to 0; `Tripod` 560 to 560 (unchanged: per-step allocations remain
  in the network step, 240 B with a `PoissonStimulus`).
- Sparse Poisson stimuli (`PoissonStimulus`, `PoissonStimulusLayer` with `PoissonLayer` and `PoissonLayerHet`): the
  spikes are sampled without one random number per input neuron and step, and the stimuli use the sparse hand-off
  of `SpikeTimeStimulus`. Microbenchmarks (`julia -t 1`, median of 3, `dt = 0.1`, us per `train!` step, whole network):
  1000 `PoissonStimulus` inputs at 10 Hz on an `Identity` -> pair STDP -> 10 `Identity` neurons (post driven by a
  `PoissonStimulus` at 10 Hz) 7.43 to 0.23 (33x); the same with a 1000-neuron `PoissonLayer` (one-to-one) 1.28 to 0.25;
  Vogels-Abbott-like 3200 E + 800 I `IF` network (p = 0.02) with a `PoissonStimulus` of 20 Hz per neuron
  86.6 to 56.0, with two 1000-neuron `PoissonLayer`s at 10 Hz (p = 0.1) 63.9 to 56.5 (the rest of these two is the
  integration of 4000 `IF` neurons and the recurrent `forward!`). Allocation of `stimulate!` of a `PoissonStimulus`: 0 B
  was not verified in isolation; an `IF` network with a `PoissonStimulus` allocates 240 B per `train!` step in 1.10 and in 1.9.1 (measured in the
  independent review, whole step, source not isolated).
  - Sampler. `PoissonStimulus` draws a Poisson count per neuron (as before). A neuron has at least one spike with
    probability `q = 1 - exp(-rate dt)`, independently, so the neurons that spike form a Bernoulli(`q`) sequence along
    `neurons`: the gap to the next one is `floor(-log(1 - U) / (rate dt))` (geometric), and the count of a visited neuron
    is drawn from the zero-truncated Poisson law (inversion; rejection for `rate dt >= 1`). The marginal law of every count is
    exactly Poisson(`rate dt`), counts are independent, and the cost of a step is `O(1 + rate dt n)`: one logarithm and one
    uniform per input spike. `PoissonLayer` (Bernoulli `p = rate dt` per neuron and step, at most one spike per step as
    before) skips with `floor(log(1 - U) / log(1 - p))`. `PoissonLayerHet` (one rate per neuron) uses rejection against
    the largest probability: candidates are skipped with `pmax = max(rates) dt` and neuron `j` is kept with probability
    `rates[j] dt / pmax`; this is exact for every `rates`, also if they change between steps (`set_variable!`), and needs
    no per-neuron state. `pmax` is recomputed each step by one read-only pass over `rates` (no random numbers). A bucket
    or heap of per-neuron next-spike steps was not used: it needs state per neuron and is invalidated by every change of a rate.
    Above `rate dt` (`pmax`) of 0.2 for the layers every neuron is tested with one uniform (the skipping sampler is slower there);
    `PoissonStimulus` stays exact and approaches the dense cost as `rate dt` approaches 1.
  - Hand-off. Both stimuli log the neurons of an `Identity` target they write (`inbox`, `attach_driver!`), so an `Identity`
    whose writers are all such stimuli keeps its sparse update; on any other population (`IF`, `AdEx`, `Tripod`, ...)
    only the targets of the neurons that spike are written (`inbox` is `nothing`). `PoissonStimulusLayer` clears its `fire`
    flags through its ascending spike list `spikes` (a fed `SpikeIndex`, returned by `spike_index(stim)`) instead of `fill!`.
    Stimuli are not populations and no connection reads their spikes, so nothing else scans `fire`.
- Where the time went and what was done:
  - Spike index lists (`SpikeIndex`): `SpikingSynapse` has two new fields, `pre_spikes` and `post_spikes`
    (scratch lists of `Int32` indices). `train!`/`sim!` call `prepare_spikes!(c)` before and
    `release_spikes!(c)` after each connection; the lists are built from `fireJ`/`fireI` by one scan on first
    use in the step (8 flags per `UInt64` load) and shared by `forward!` (including the delay method),
    `update_traces!` of the Markram STP rules, and `_pair_stdp!`/`STDPTriplet` (`_pre_spike_pass!`,
    `_post_spike_pass!`, `_spike_increment!` have list methods taking `AbstractVector{Int32}`). Populations and
    stimuli still write the `Bool` `fire` vector. Outside `prepare_spikes!`/`release_spikes!` (direct calls of
    `forward!`, `plasticity!`) every call rescans `fire`, so nothing is stale. The dense kernel methods and the
    generic `presynaptic_spikes`/`postsynaptic_spikes` fallbacks keep external synapse types working.
  - Concrete types: the plasticity rule and state of `SpikingSynapse`, and the `param` of `SpikeTimeStimulus`, are
    type parameters (see Breaking changes): no dynamic dispatch on `any(c.STPVars.active)` or
    `plasticity!(c, c.LTPParam, c.LTPVars, ...)` in the step.
  - `record!`: an unmonitored object costs one inlined `isempty(obj.records)` test (the body is `_record!`: no
    call, no hashed lookup, no allocation); `record_zero!` is inlined the same way. Was 70 ns of the 170 ns per
    step of the 200-synapse validation task.
  - Sparse hand-off, stimulus to `Identity` population. A `SpikeTimeStimulus` on an `Identity` logs the
    neurons whose input it increments (`inbox`, a `TouchList`; `attach_driver!`) and clears its own `fire`
    flags where it set them (`spiking`), not with `fill!`. While all writers of `Identity.g` are such stimuli
    (`nwriters == ndrivers`, counted by `synaptic_target`), `integrate!` updates only the logged neurons and the
    neurons that fired last step (`_integrate_sparse!`), keeps the ascending list `spikes` of the neurons that fire,
    and sets `spikes.fed`; `SpikingSynapse(pre, post, ...)` shares `pre.spikes`/`post.spikes` (`spike_index(pop)`)
    as its `pre_spikes`/`post_spikes`, so `forward!` and the event-driven rules never scan `fire`. `fire`,
    `spikecount`, `h` and `g` equal the dense update at every step (`test/syn/spike_handoff.jl`). The dense update
    (and one scan to rebuild the list) is used for any other writer (a connection, a stimulus without a log), after
    an overflow of the inbox (more writes than neurons in one step), and for the first step after a change of
    writers. Per step, the sparse path removes the dense `integrate!` (105 ns at 1e4), the flag scans (57 ns) and
    the `fill!` of the stimulus (6 ns). `SpikeIndex` (and `spike_list!`, `_arm!`, `_disarm!`) live in
    `src/populations/spike_index.jl` and have a `fed` field.
  - `Identity` `integrate!` is branchless and vectorised (same values).
  - `IF` `update_neuron!` (and its adaptation loop) is branchless, `@inbounds @simd`, with `round(Int, τabs/dt)` hoisted out
    of the loop and a length check up front (`DimensionMismatch` if a state vector is shorter than `N`, instead of an
    out-of-bounds access). Results are bit-identical (Float32 and Float64, both synapse models, `τw > 0`, refractory neurons;
    `test/pop/if_update_loop.jl` against the old loop kept as reference; benchmark rasters byte-identical). Balanced E/I network
    (`validation/scaling_balanced`, k100, one pinned core, 10 s at dt 0.1 ms): 5.0 to 1.56 ns per neuron and step at N = 10000
    (0.488 s to 0.151 s per simulated second), 4.7 to 1.5 ns at N = 500; now 8-12 % faster than Auryn (`-O3 -ffast-math`) and
    within 20 % of Brian2 standalone. `@turbo` was not used: it contracts to FMA and changes the spikes. A non-finite `τabs`
    now throws at the first step instead of at the first spike. The other populations follow in the next bullet.
  - Neuron and synapse loops of the other populations (same recipe as `IF`: up-front length check that throws `DimensionMismatch`,
    `@inbounds @simd`, loop invariants hoisted, branchless where a data-dependent branch blocked vectorisation). Results are
    bit-identical to 1.9.1 (Float32; every converted method has a test against a copy of the old loop in `test/pop/*_update_loop.jl`
    and `test/utils/inline_math_test.jl`: random states, refractory neurons, spikes, N = 0, 1, odd N, scalar and per-neuron parameters),
    except the Poisson populations (see Behaviour notes). Whole `integrate!` of the population (its synapse stage included), ns per
    neuron and step, one pinned core, `dt = 0.1`, N = 1000 / 10000, regime with spikes / without (after the change the two regimes differ by
    less than 5 %, the loops no longer branch on the spikes; the Poisson populations cost depends on the rate), before -> after:
    `AdEx` 12.5 -> 3.1 (4.0x), `AdEx` with per-neuron parameters 15.8 -> 3.2 (4.9x), `ExtendedIF` 3.3 -> 0.83 (4.0x),
    `HH` 36.7 -> 8.3 (4.4x), `IZ` 0.87 -> 0.69 (1.3x), `MorrisLecar` 24-31 -> 8.1 (3.0-3.8x), `Rate` 4.9-6.0 -> 0.65 (7.5-9x; 2.7x in the quiet regime),
    `WilsonCowan` 5.2-6.0 -> 0.67 (7.7-9x; 2.7x quiet), `Tripod` 54.6 -> 21.7 (2.5x), `BallAndStick` 36.3 -> 15.1 (2.4x),
    synapse stage of `ReceptorSynapse` 5.8 -> 0.96 (6x at N = 1000, 4.3x at 10000) and of `MultiReceptorSynapse` 11.5 -> 1.3 (8-9x),
    Poisson populations: see below. `IF` (1.4), the synapse stages of `DoubleExpSynapse`, `DoubleExpCurrentSynapse`, `SingleExpSynapse`,
    `DeltaSynapse`, `CurrentSynapse` and `Confavreux2025Synapse` (0.2-0.9) were already vectorised and are unchanged.
    Whole networks (pinned core, `sim!`, no monitors, spike rasters and final states hash-identical to 1.9.1): the balanced E/I network of
    `validation/scaling_balanced` with `AdEx` neurons (`adex_balanced.jl`, k100, 10 s at `dt = 0.1`, 11 Hz) N = 2000 2.55 s -> 0.668 s (3.8x),
    N = 10000 13.2 s -> 3.83 s (3.45x; 13.2 -> 3.8 ns per neuron and step); a network of 400 `Tripod` + 100 `IF` neurons with Poisson inputs
    (43.7 Hz) 24.0 -> 10.7 us per step (2.2x), 4000 + 1000 neurons 229 -> 96.7 us per step (2.4x).
    - `AdEx` (scalar and per-neuron parameters): the loop was branching (`continue` after the refractory counter) and had an
      exponential term, which is an out-of-line call that stops `@simd`; the exponential is now inlined at the call site
      (`@inline exp`, the algorithm of `Base.exp` for `Float32`, so the same bits) and the update is branchless. `ExtendedIF`: branchless, hoisted
      `round(Int, τabs / dt)`. `HH`: inlined exponentials and `x^4`, which `Base` evaluates in `Float64` through an out-of-line
      `power_by_squaring` (`_pow4`: the same arithmetic written out). `IZ`: the three passes over the neurons are one loop.
      `MorrisLecar`, `Rate`, `WilsonCowan`: `tanh` and `cosh` of `Float32` with the exponential inlined (`_tanh_inl`, `_cosh_inl`, copies of
      the `Base` algorithms, checked bit by bit against `Base.tanh`/`Base.cosh` on 4e6 random arguments in `test/utils/inline_math_test.jl`).
      They rely on internals of `Base.Math` (`tanh_kernel`, `TANH_LARGE_X`, ...), so a new Julia version that changes them fails that test.
    - `Tripod`, `BallAndStick`: the refractory/backpropagation/spike/Heun loop of `integrate!` is branchless (the `Float64` arithmetic of the old
      `0.5 * dt * (...)` is kept), the dendritic and somatic synaptic currents of `ReceptorSynapse` (`synaptic_current!`, scalar, bounds-checked, an `exp256`
      and a division for every receptor, also for the receptors without a magnesium block) are vectorised with the block evaluated only for the NMDA
      receptors, and `update_receptor!` of `MultiReceptorSynapse` has `exp64` of the loop-invariant argument hoisted. Not changed: the two `@fastmath` Heun
      loops (`update_neuron!`) keep the axial currents `ic` in memory, because holding them in registers lets LLVM vectorise and reassociate the fast-math
      expression and changes the derivatives by 1 ulp: that variant measured 10.7 (Tripod) and 6.7 ns (BallAndStick) against 21.7 and 15.1, but is not
      bit-identical, so it was not kept. These two loops are now about half of the `Tripod` step.
    - `HetRec` is unchanged (273 ns per neuron and step at N = 1000: a gather over about `N overlap Nd` weights per neuron, 0.27 ns per weight, and
      the dense `rand!` is a small part). `IF_CANAHP`, `AdExMultiTimescale` and `Multipod` are not included in the module (commented out in `populations.jl`)
      and were not touched. The `@fastmath` synapse stages of `SingleExpSynapse` and `DeltaSynapse` are already vectorised by LLVM (checked in the IR).
    - Poisson populations (`Poisson` with one rate or one rate per neuron, `InhomogeneousPoisson`, `VariablePoisson`): sparse sampling with a fed spike list
      (see Behaviour notes). ns per neuron and step, N = 1000 / 10000: `Poisson` at 50 Hz 0.26 -> 0.14 and 0.19 -> 0.09 (1.9-2.1x), at 1 Hz 0.26 -> 0.03 and 0.19 -> 0.006
      (10x and 32x), heterogeneous rates 0.26 -> 0.28 and 0.22 -> 0.22 at 50 Hz (no gain: 7-10 candidates per step cost as much as the vectorised
      dense draw) and 0.25 -> 0.07, 0.23 -> 0.04 at 1 Hz (3.5x, 5x), `InhomogeneousPoisson` (50 Hz, `β = 2`) 6.9 -> 1.4 (4.8x; the per-neuron rate
      update and the exponential are vectorised, the spike draw was a scalar `rand` and an exponential per neuron), `VariablePoisson` 1.0 -> 0.17 at 50 Hz (6x) and
      0.96 -> 0.07 at 1 Hz (14x; 84x at N = 10000). Above 3 % per step (300 Hz at `dt = 0.1`) skipping is slower than testing every neuron: the population
      then does what 1.9.1 did, one vectorised uniform draw per neuron and the `fire` flags, and neither builds the spike list nor offers it (`spikes.fed = false`),
      so the connections scan `fire` as before. Pinned, median of 3, ns per neuron and step, `Poisson` at 500 Hz, N = 1000 / 10000: 1.9.1 0.30 / 0.20, here 0.31 / 0.22
      (parity within the 5-10 % noise; an earlier version that built the list by compaction cost 0.57 / 0.4); heterogeneous rates 0.31 / 0.28 against 0.34 / 0.27
      (the largest rate is read in the same pass as the draw while the population stays dense; +4 % on the loop at N = 1000). `InhomogeneousPoisson` and
      `VariablePoisson` are 3-4x and 2.7-3x faster than 1.9.1 at the rates measured above (50 Hz); their dense regime now skips the list as well (not timed separately).
  - Delay ring buffer: see Breaking changes. Per spike the queue insertion (`findlast` with an allocating
    broadcast, `insert!`, `popfirst!`) is replaced by one add into a matrix.
  - Trace decay: the factors `exp(-dt/τ)` are cached in the rule variables (`decay` field of `STDPVariables`,
    `STDPTripletVariables`) and recomputed only if `dt` or a `τ` changes. `_rescale!` (was `_decay!`) sets values
    below `floatmin(Float32)` to zero: the trace of a silent neuron used to run through the subnormal range at
    about 4 us per 1000 elements per step (now 0.06 us). Weight changes from this are below 1e-38.
  - Lazy trace decay, always on (see New API; the dense per-step decay of the event-driven rules and the
    `lazy_decay` option are gone): the step cost no longer grows with the number of neurons.
  - The Euler-trace rules are lazy too: `iSTDPRate`, `STDPMexicanHat`, `STDPSymmetric`, `STDPAntiSymmetric` (they used
    `x += -dt x/τ` on every neuron at every step, plus allocating `findall` scans of the spike flags). Same scale mechanism
    with the Euler one-step factor `1 - dt/τ` (kept in `Float64`), spike lists for the weight passes and the increments.
    A step costs O(spikes x fan-out), independent of the number of neurons apart from the one scan of the spike flags
    that a population without a spike list needs. Weights agree with the dense Float32 kernels to a few Float32 ulps of
    the largest weight (see Behaviour notes).

### Opt-in thread parallelism of the population updates (off by default)
- `neurons_per_thread` (keyword of `sim!` and `train!`, and `set_neurons_per_thread!`, `neurons_per_thread()`, `with_neurons_per_thread`,
  `set_optimal_neurons_per_thread!`): the per-neuron stages (`integrate!` of `IF`, `AdEx`, `Identity` with `CurrentSynapse`, `DoubleExpSynapse`,
  `SingleExpSynapse`, and the synapse and plasticity stages that have a chunked method) run in chunks of whole 128-neuron batches on Polyester
  workers (`@batch`). The default is `0`: threading is off and nothing changes. With one Julia thread, with `0` or `Inf`, or with a single chunk the
  serial loop runs and `@batch` is not called. The number of chunks depends on `N` and the setting only (at most 8), never on the number of threads, and a
  per-neuron update has no reduction, so the result is bit-identical for every setting and thread count. New dependency: Polyester 0.7.
  `Tripod`, `BallAndStick`, `ReceptorSynapse`, `DeltaSynapse`, `ExtendedIF` and the Poisson populations stay serial. Measurements and the prototype are in
  `papers/JuliaSNN_publication/validation/threading`.

### Breaking changes
Each item gives the migration.
- `SpikingSynapse` field types. `LTPParam`, `STPParam`, `LTPVars`, `STPVars` are type parameters (`LTPP`, `STPP`,
  `LTPV`, `STPV`), so a rule of another type can no longer be assigned in place. `update_plasticity!` and
  `change_plasticity!` accept only a rule of the same type as the current one (new parameters, fresh state) and
  throw an `ArgumentError` otherwise (for example, the documented `change_plasticity!(syn; LTP = STDPGerstner())`
  on a synapse built without LTP). Direct assignment `c.LTPParam = rule` of another type is a `MethodError`
  (cannot `convert`).
  Migration: build the synapse with the rule (`LTPParam = ...`), or `c2 = with_plasticity(c; LTP = rule)` and
  replace `c` by `c2` when the model is composed (`compose(; ..., EE = c2, ...)`); `c2` shares all arrays with `c`,
  so do not keep both in one model. Same-type changes (new parameters) still work with `update_plasticity!`.
  `update_plasticity!(c; LTP, STP)` checks LTP before STP and writes LTP before checking STP: if the STP rule has
  another type, the LTP change has already been applied.
- `SpikeTimeStimulus.param` is a type parameter (`PT <: SpikeTimeStimulusParameter`) and the struct has a new type
  parameter `TLT` and fields `spiking`, `inbox`. The `fire` flags are cleared where they were set, not with
  `fill!` (a flag written by hand into `stim.fire` is no longer cleared). Migration: keyword construction is
  unchanged; code that constructs the struct positionally or annotates the full parameter list must be updated;
  `update_spikes!`, `shift_spikes!` act in place and are unchanged.
- `PoissonStimulus` has type parameters `PT <: PoissonStimulusParameter` (the `param` field was abstractly typed) and
  `TLT`, and a field `inbox`. `PoissonStimulusLayer` has a new type parameter `TLT` and fields `spikes`, `inbox`;
  its field `randcache` is removed (the spikes are no longer drawn from a cache of one uniform number per neuron).
  Migration: keyword construction and `Stimulus(param, post, sym; ...)` are unchanged; code that read or wrote
  `randcache`, or that defined its own `stimulate!` for a `PoissonStimulus`/`PoissonStimulusLayer` using it (for example
  the refractory Poisson tutorial in `SpikingNeuralNetworks/examples/tutorials/extensions`, already out of date), must
  draw its own random numbers. A flag written by hand into `stim.fire` of a `PoissonStimulusLayer` is no longer cleared.
  A `PoissonStimulus` with a user-defined `PoissonStimulusParameter` (own `stimulate!`) does not log the neurons it writes: it is not attached as a list-aware
  driver, so an `Identity` target keeps its dense update and receives all writes (the refractory Poisson example of the `SpikingNeuralNetworks` docs).
- Delays use a ring buffer. `SpikingSynapseDelayParameter` (internal) loses `spike_time` and `spike_w` and has
  `delaytime`, `delaysteps`, `buffer` (`Npost x (max delay steps + 1)`), `dt`, `pos`; it is now `mutable`.
  Delays are rounded UP to whole steps, `ceil(delay / dt)` (1e-4 steps of tolerance for the quotient): 0.8 ms at
  `dt = 0.125` ms arrive 7 steps (0.875 ms) after the spike. The steps are computed at the first `forward!` and
  recomputed when `dt` or `delaytime` changes (`_remap_synapse_state!` invalidates them); if the buffer must be
  resized, queued increments are dropped. The delay is counted in calls of `forward!` of the connection.
  Memory is `Npost x (max delay / dt + 1)` Float32 per delayed connection (1000 targets, 5 ms, `dt = 0.125`: 160 kB).
  Migration: code that read `spike_time`/`spike_w` has no equivalent; read `buffer`. After editing
  `param.delaytime` in place, set `param.dt[1] = NaN32` so that the steps are recomputed (only the reordering of
  synapses by `_remap_synapse_state!` does it for you). Results at `dt = 0.125` are bit-identical to 1.9.1; at other
  `dt` see "Behaviour notes".
- `Identity` has new fields `spikes`, `inbox`, `nwriters`, `ndrivers`, `tracked`; `synaptic_target` for an
  `Identity` increments `nwriters`; `SpikeIndex` has a field `fed`. Migration: keyword construction is unchanged.
  Do not write `g`, `fire` or `h` of an `Identity` that is driven by a `SpikeTimeStimulus` by hand; if a connection
  built with the `Identity` as target is redirected to another `g` vector after construction, call
  `release_input!(pop)`. (Writing `pop.g` by hand is fine for an `Identity` with no stimulus: it integrates densely.)
- Lazily stored plasticity state. The event-driven STDP rules (`STDPGerstner`, `STDPConfavreux2025`,
  `STDPWeightDependent`, `STDPTriplet`) and the Euler-trace rules `STDPMexicanHat`, `iSTDPRate`, `STDPSymmetric`,
  `STDPAntiSymmetric` always store a trace as `stored * scale`; `STDPVariables`, `STDPTripletVariables`,
  `iSTDPVariables`, `STDPStructuredVariables` have fields `scale`, `decay` (no `lazy` field, and the rules have no
  `lazy_decay` field: `STDPGerstner(lazy_decay = ...)` is a `MethodError`). `STDPStructuredVariables` is now a
  `LTPVariables`. Reading `syn.LTPVars.tpre` (`tpost`, `r1`, `r2`, `o1`, `o2`, `to_x`, `to_y`, `tr_x`, `tr_y`) directly
  returns the STORED values, not the traces, and `syn.STPVars.u`, `.x` of the event-driven Markram
  rules are the values at the last presynaptic spike, not relaxed to the current time. Recorded `u` and `x` of the event Markram
  rules (and recorded traces) therefore change meaning: 1.9.1 recorded the stored values at the last spike, 1.10 records the true
  values; the recordings of the two versions differ by up to 0.135 (`u`) and 0.77 (`x`) (independent review, event and `Het`
  rules), and the new event recordings agree with the new `Timestep` rule to 1.1e-4 (`u`) and 5.1e-4 (`x`). The `rho` is
  bit-identical. Comparisons of recordings across versions break for these variables. Migration: read through
  `trace_values(syn, :tpre)` / `plasticity_value(syn, :LTPVars, :tpre, t)`, or call `sync_traces!(syn)` first;
  `monitor!`, `getvariable` and `record` already return values. The internal `_decay!(x, f)` is now `_rescale!(x, s)`.
- Saved models and parameters. A model file saved with SNNModels 1.9.1 or earlier by `save_model`/`SNNsave` (JLD2) cannot be
  used. JLD2 does not fail at load for most of them: for population-only, plain, STDP and delayed-synapse models alike it
  returns placeholder objects (`JLD2.ReconstructedMutable`: the clock `Time` of the model, `SpikingSynapse`, `SpikeTimeStimulus`
  and the STDP rule objects changed layout), and `train!` on them fails with an unrelated "Invalid argument type" error
  (independent review; 1.9.1 files of a plain, a plastic and a delayed synapse, a population only; an `Identity` file raises a
  `MethodError` inside JLD2 at load). `load_model`, `load_data` and `SNNload` now check the loaded object recursively
  (`check_loaded_model`, exported) and throw an `ArgumentError` at load time that names the offending type and states that the file was
  saved with an older SNNModels (<= 1.9.1) layout. No migration is attempted. Parameter objects of
  `STDPGerstner`, `STDPTriplet`, `STDPWeightDependent`, `STDPConfavreux2025` saved in any JLD2 file (a
  configuration, an Optuna artifact) are also reconstructed, together with the container that holds them; the check is only
  applied by the SNNModels load functions (`OptunaLoader` skips the stripping of such files). The parameters `iSTDPRate`,
  `MarkramSTPParameter*`, `IFParameter`, `STDPMexicanHat`, `NoLTP` load. Migration: re-create the model from its configuration and
  save it again with 1.10, or load the old file with 1.9.1; the raw arrays of an old file stay readable with `JLD2.load` (fields of
  the reconstructed objects). No JLD2 upgrade (typemap) is provided.
- `vSTDPParameter` is the event-driven voltage-based rule, `vSTDPDenseParameter` the dense one (breaking). Up to 1.9 and in the
  1.10 development tree before this change `vSTDPParameter` was the clock-driven rule (now `vSTDPDenseParameter`,
  `vSTDPDenseVariables`: same fields, same equations, same numbers) and the event-driven rule was `vSTDPEventParameter` /
  `vSTDPEventVariables` (removed: no alias; they are `vSTDPParameter` / `vSTDPVariables`, same fields). Consequences for existing
  code: (1) the numbers of a run with `vSTDPParameter` agree with the previous ones to the Float32 rounding of the dense rule (1e-5
  relative at 1e4 synapses), they are not bit-identical; (2) `syn.W` of a synapse with `vSTDPParameter` is NOT the current weight
  vector between materialisations: read it with `plasticity_value(syn, :W)` or call `sync_traces!(syn)` first (`monitor!(syn, :W)`,
  `matrix`, `save_model` and the other library readers do it for you, see "Reading the weights"); (3) the state is `vSTDPVariables` (fields
  `u`, `v`, `x` = stored amplitudes, `K`, `snap`, ...), the old `vSTDPVariables` fields are those of `vSTDPDenseVariables`;
  (4) `A_LTP >= 0` and `0 < epoch <= 25` are enforced; a negative `A_LTP` needs `vSTDPDenseParameter`; (5) `dt < τx` is required.
  To keep the previous behaviour exactly use `vSTDPDenseParameter` (a drop-in replacement of the old `vSTDPParameter`).
  `current_weights` and `sync_weights!` are no longer exported (see "Reading the weights").
- Tests changed: `test/syn/with_plasticity.jl` ("change_plasticity!": same type in place, other type via
  `with_plasticity`), `test/syn/float32.jl` (delay state: `eltype(buffer)` instead of `spike_time`/`spike_w`).

### New API
- `with_plasticity(c::SpikingSynapse; LTP, STP)`: synapse that shares all arrays with `c` and has the new rule
  with fresh state (exported).
- Lazy trace decay (no option): a trace is stored as `x` and one `Float64` scale per trace type
  (`STDPVariables.scale`, `STDPTripletVariables.scale`), its value is `x * scale`; a step multiplies the scale by the
  Float32 one-step factor, a spike adds `1/scale`, reads multiply by `scale`, and only when the scale is below 1e-6
  the stored values absorb it (one pass over the neurons every `log(1e-6)/log(f)` steps, flushing subnormals). A
  change of `dt` or of a `τ` syncs first.
- `plasticity_value(obj, group, key, T)` (exported): the true value of any plasticity variable (`group` = `:LTPVars`
  or `:STPVars`) at time `T`, state unchanged. STDP traces: stored times scale. `u`, `x` of the event-driven Markram
  rules (`MarkramSTPParameterEvent`, `Het`), which are updated only at presynaptic spikes: exact relaxation from
  `last_spike`. Everything else (`_ρ`, `last_spike`, `MarkramSTPParameterTimestep`, iSTDP, vSTDP, structured rules,
  `STDPMexicanHat`: dense or event quantities) as stored. Extension points: `current_value(vars, param, key, t)` and
  `lazy_variables(vars)` (a declared lazy field without a `current_value` method throws an `ArgumentError`).
- `sync_traces!(syn)` (rescales the stored traces so that `syn.LTPVars.tpre` etc. hold the values; rounds once,
  relative error about 1e-7) and `trace_values(syn, :tpre)` (values, state unchanged).
- Recording. `monitor!(syn, keys, :LTPVars)` / `:STPVars` read every key through `plasticity_value` at each sampling
  step (and only then: `record!` no longer reads the field on non-sampling steps), so `:tpre`, `:tpost`, `:r1`..`:o2`,
  `:u`, `:x` are recorded as values. The recorder never changes the state (no dense mode when monitoring): a
  monitored and an unmonitored run give bit-identical weights (`test/syn/lazy_traces.jl`). `monitor!` of a lazily
  stored variable without accessor throws at `monitor!` time. Before this, recording `:u`, `:x` of the event-driven STP
  rules stored the values at the last spike.
- `release_input!(pop::Identity)` (exported), `attach_driver!`, `spike_index`, `spike_list!`, `prepare_spikes!`,
  `release_spikes!`, `presynaptic_spikes`, `postsynaptic_spikes` (not exported; extension points for new synapse and
  population types: a population with a sparse update returns its own `SpikeIndex` from `spike_index`).
- `fed_spike_index(N)` and `spike_index(::PoissonStimulusLayer)` (not exported): the list of the layer neurons that fired.
- Tests: `test/syn/spike_index.jl`, `test/syn/spike_handoff.jl`, `test/syn/lazy_traces.jl`, `test/syn/lazy_traces_euler.jl`, `test/stim/poisson_sparse.jl`.

### Behaviour notes
- Weights of `STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent`, `STDPTriplet` differ from 1.9.1 (lazy decay,
  see below). Bit-identical to 1.9.1 in the seeded audit of the dev tree before the lazy decay became the only path
  (Poisson and `SpikeTimeStimulus` inputs, `Identity` relay and two-layer networks, 4 s to 2 s of `train!`):
  `STDPMexicanHat`, `iSTDPRate`, `vSTDPDenseParameter` (then called `vSTDPParameter`), Markram STP (`rho`, `u`, `x`), a synapse with delays plus
  STDP with the dense decay, the `SNNPlots.stdp_kernel` construction. The suite of SNNModels (29719 tests: 10382 with
  `test/syn/lazy_traces.jl` rewritten, the audit measured 10278 before, plus `test/stim/poisson_sparse.jl` with 19201 and the tests of the concurrent plasticity work) passes; SNNUtils, SNNPlots and
  SpikingNeuralNetworks suites and doc examples were not rerun after the change to always-lazy traces.
- Poisson stimuli: seeded results change (breaking behaviour). `PoissonStimulus`, `PoissonLayer` and `PoissonLayerHet`
  consume the global random number stream differently (geometric skipping and thinning instead of one draw per neuron and
  step), so a run with the same `Random.seed!` no longer reproduces 1.9 spike for spike, nor does any network that contains
  one of them (the random numbers that follow are shifted too). The process is the same: Poisson counts of mean
  `rate dt` per neuron and step for `PoissonStimulus`, Bernoulli(`rate dt`) for the layers (at most one spike per step),
  independent across neurons and steps, with rates that may change in time. Within 1.10 a run is bit-reproducible for a fixed
  seed, and a monitored run equals an unmonitored one. Statistics checked against the 1.9 dense sampler in
  `test/stim/poisson_sparse.jl` (rates 1, 10, 50, 200 Hz at `dt = 0.1`, `N` = 10 and 1000, heterogeneous rates 1 to 3000 Hz,
  rates that switch or oscillate): counts per neuron (mean, Fano factor, Poisson/binomial dispersion), law of the per-step
  count, inter-spike intervals (chi-square and KS against the geometric law), neighbour correlations, lag-1 autocorrelation,
  two-sample KS against the dense sampler on per-neuron counts and on intervals.
- Poisson populations: seeded results change (breaking behaviour), as for the stimuli. `Poisson` (both parameter types),
  `InhomogeneousPoisson` and `VariablePoisson` sample their spikes sparsely: the gap to the next firing neuron is geometric (one probability), or
  candidates are generated with the largest per-step probability and thinned (per-neuron probabilities), so a step costs one random draw per spike
  and the spike list `spikes` (a fed `SpikeIndex`, returned by `spike_index(pop)`) is built by the population, so the connections that read it never
  scan `fire`. Above 3 % per step every neuron is tested with one vectorised uniform number (`randcache`, `randcache_β`) as in 1.9, the list is not built and
  `spikes.fed` is `false` (the connections then scan `fire`, `spike_list!`; the regime can change at runtime and `spike_index(pop)` stays valid). The random number
  stream differs from 1.9: a run with the same `Random.seed!` no longer reproduces 1.9.1 spike for spike, nor does a network that contains one of
  these populations. The statistics are those of 1.9 (Bernoulli(`rate dt`) per neuron and step, `1 - exp(-ν dt)` for `InhomogeneousPoisson`, `E dt / N`
  for `VariablePoisson`), tested as for the stimuli in `test/stim/poisson_pop_sparse.jl` (33563 tests: counts per neuron, dispersion, law of the
  per-step count, inter-spike intervals (chi-square, KS), neighbour correlations, lag-1 autocorrelation, KS against the dense sampler, heterogeneous
  rates 1 to 3000 Hz, rates changed at runtime, time-varying `InhomogeneousPoisson` and `VariablePoisson` (spikes against the probabilities of the
  steps, in 10 probability classes), the dense regime, the switch between the two regimes and of the fed list at runtime for the four populations, lists and flags, connections reading the list).
  New fields: `InhomogeneousPoisson.pspike` (probability of the last step), `VariablePoisson.Erate` (total rate of the last step), `spikes` on the three
  populations; `Poisson` and `VariablePoisson` keep `randcache` (used only in the dense regime). The noise and rate offsets of `InhomogeneousPoisson` and the
  total rate of `VariablePoisson` follow the same equations (the noise and offsets are the same to 1e-5 for the same random numbers).
- Other populations (`AdEx`, `ExtendedIF`, `HH`, `IZ`, `MorrisLecar`, `Rate`, `WilsonCowan`, `Tripod`, `BallAndStick`, `ReceptorSynapse`,
  `MultiReceptorSynapse`) are bit-identical to 1.9.1; a state vector (or `Δv`, `is`, `v_pred`, a receptor input buffer, a per-neuron parameter vector of
  `AdEx`) shorter than `N` is now a `DimensionMismatch` at the start of the step instead of an `@inbounds` out-of-bounds access (or a `BoundsError`).
  A non-finite `τabs` makes `AdEx` and `ExtendedIF` throw at the first step instead of at the first spike.
- Delays at a `dt` that is not exactly representable in Float32 (0.1, 0.05, 0.025 ms) are not bit-identical to
  1.9.1: 1.9.1 compared the arrival time with the Float32 clock, which accumulates `dt` and drifts, so a delay that
  is an exact multiple of `dt` arrived one step late in 59.6 % of the events at `dt = 0.1`, 80 % at 0.05, 39 % at 0.025 and
  22 % at 0.2 (measured in the independent review, delays 0.8, 1.0, 2.5 ms and Dirac; fraction of events delivered one step earlier
  by 1.10 than by 1.9.1). A continuous delay (`Uniform(0.5, 5)`) arrived a step late in 0.08 % of the events at `dt = 0.1`; at
  `dt = 0.05` 0.04 % late and 0.24 % early (1.9.1 also delivered some events a step early). The ring buffer counts steps and is exact.
  Trajectories of chaotic networks differ from 1.9.1 accordingly. Statistics: balanced E/I network, 10 s, `dt = 0.1`, delay 0.8:
  N = 500 unchanged (rate 6.11 to 6.22 Hz, KS p = 0.22, CV p = 0.93); N = 2000 the rate distribution shifts by 1.6 %
  (6.51 to 6.61 Hz, KS D = 0.064, p < 1e-3, CV p = 0.11). The shift is the 1.9.1 delay bias: 1.10 is closer to Auryn (6.61 Hz) and
  Brian2 (6.73 Hz) than 1.9.1 (KS against Auryn: D = 0.022, p = 0.71 for 1.10, D = 0.056, p = 0.004 for 1.9.1). At `dt = 0.125` (a power of two) results are
  identical.
- Lazy decay is not bit-identical to the 1.9.1 dense decay (the dense Float32 product chain and `x * scale` round
  differently): weights differ from 1.9.1 by a few Float32 rounding steps, up to about 1e-5 relative in long chaotic
  runs, 1.2e-7 to 7.2e-7 absolute on the validation task. Agreement with Auryn stays at 1.2e-7 to 1.4e-7 for pair,
  triplet and weight-dependent, but the 40x amplitude `pair_bounds` run deviates by 4.8e-7 (4 Float32 steps of a weight
  near 1) instead of 1.8e-7. With Brian2 the agreement stays 2.0e-6. Analytic kernel errors are unchanged.
- Euler-trace rules (`iSTDPRate`, `STDPMexicanHat`, `STDPSymmetric`, `STDPAntiSymmetric`) keep their numerics: the
  lazy factor is the Euler factor `1 - dt/τ` (not `exp(-dt/τ)`) and the order of decay and increment of each rule is
  unchanged (`x f + 1` for these rules, `(x + 1) f` for the exponential pair rules; `iSTDPRate` reads `tpost` of the
  previous step at a presynaptic spike and `tpre` of this step, spike included, at a postsynaptic spike; the
  structured rules read the traces of the previous step; `STDPMexicanHat` reads traces that contain this step's spikes).
  Against the dense Float32 kernels of 1.9.1 (1500 steps, 50 x 40 synapses, 5 % spikes per step; `test/syn/lazy_traces_euler.jl`):
  max|dW| 9.5e-6 (`iSTDPRate`, weights up to 32), 3.8e-6 (`STDPMexicanHat`), 7.6e-6 (`STDPSymmetric`), 8.6e-6
  (`STDPAntiSymmetric`), i.e. 3e-7 to 8e-7 of the largest weight (a few ulps). The independent review (random trains, 10 s, weights
  15 to 25) measured 3.5e-5 absolute for `STDPSymmetric` (2.3e-6 relative to the largest weight), 9.5e-6 for `STDPMexicanHat`, 7.6e-6 for
  `iSTDPRate`, at most 3.8e-6 for the others: all below 1e-4 but `STDPSymmetric` is 3 to 5 times the figure above. Against a Float64 run of the dense
  kernel the lazy weights are as close as the Float32 dense ones (1.6e-5 / 8.9e-6 / 9.2e-6 against 1.5e-5 / 9.9e-6 /
  9.5e-6 for `iSTDPRate` / `STDPSymmetric` / `STDPAntiSymmetric`). The factor is kept in `Float64`; a Float32 factor
  biased the effective time constant by 1e-5 and the weights by 1e-4 (measured, rejected). Recorded traces equal an
  independent dense Float64 recursion to rtol 1e-5 and monitored/unmonitored weights are bit-identical. `iSTDPPotential`
  is not converted (its postsynaptic trace follows the membrane potential at every step); its variables keep `scale = 1`.
- Plasticity still runs only under `train!`; `sim!` calls `prepare_spikes!`/`release_spikes!` around `forward!`.
- `update_plasticity!`/`change_plasticity!` reset the state of the new rule (traces, STP variables) as before.
- First `train!` of a model compiles more (one specialisation per combination of rule types): 0.36 s against 0.08 s
  for a six-synapse E/I network in a warm session.

### Lazy classification of the plasticity rules
A = exactly lazy-able (pure exponential decay between events, events at spikes only; Euler decay `x (1 - dt/τ)` counts,
with that factor); B = lazy-able only with an approximation; C = a state is driven by a continuous quantity (membrane
potential), per-step evaluation of that state. State after this change:

| rule | state and update | class | status |
|---|---|---|---|
| `STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent` | `x_pre`, `x_post`: exp. decay, +1 at spikes; weight at spikes | A | lazy |
| `STDPTriplet` | `r1`, `r2`, `o1`, `o2` as above | A | lazy |
| `STDPMexicanHat` | Euler traces `x f + spike`; weight from `log(x_pre/x_post)` at spikes | A | lazy (this change) |
| `STDPSymmetric`, `STDPAntiSymmetric` | Euler traces `to_x`, `to_y`, `tr_x`, `tr_y`; weight at spikes | A | lazy (this change) |
| `iSTDPRate` | Euler traces `tpre`, `tpost`; weight at spikes | A | lazy (this change) |
| `iSTDPPotential` | `tpre` as above (A); `tpost` low-pass filter of `v_post`, read at presynaptic spikes | C | dense (O(Npre + Npost) per step); the weight updates are event-driven |
| `vSTDPDenseParameter` | `x` trace (A); `u`, `v` filters of `v_post`; LTD at presynaptic spikes from `u`; LTP `dt A x [v-θ-]+ [V-θ+]+` on every synapse at every step | C | dense (reference). The weight term is exactly event-drivable (see below): `vSTDPParameter` |
| `vSTDPParameter` | same rule; filters and accumulator `K` of the postsynaptic neurons per step, weights materialised at presynaptic spikes and once per epoch | C (per neuron) | event-driven, O(Npost) per step, lazy `W` (accessor `plasticity_value(syn, :W)`) |
| `MarkramSTPParameterEvent`, `Het` | `u`, `x` relax exponentially, updated at presynaptic spikes | A | event-stored, accessor |
| `MarkramSTPParameterTimestep` | `u`, `x` Euler relaxation of all neurons at every step | A | dense (reference of the event rule) |
| `SynapseNormalization`, `RandomTurnover` | one pass over the synapses every `τ/dt` steps | periodic | no per-step cost |
| `ActivityDependentTurnover` | Euler traces of spikes, read once per period | A | dense |
| `AggregateScaling` | rate trace `y` (A); `WT` driven by `y` at every step; rescale every `τ/dt` steps | C (per neuron) | dense, O(N) per step |
| `FLSynapse`, `FLSparseSynapse`, `PINning*`, `RateSynapse` | rate-based, error-driven RLS / rate updates at every step | C | dense |
| `iSTDPTime`, `CaRule.jl` | no `plasticity!` method / file not included | - | not usable |

vSTDP weight term, exact event-driven form (implemented as `vSTDPParameter`, see "Event-driven vSTDP"). With `g_i(n) = [v_i(n) - θ_LTD]_+ [V_i(n) - θ_LTP]_+`
(a function of the postsynaptic neuron only), `q = 1 - dt/τx` and `x_j(n) = q x_j(n-1) + s_j(n)/τx`, the LTP weight is
`dt A_LTP sum_n x_j(n) g_i(n)`. With the per-postsynaptic-neuron accumulator `K_i(n) = sum_{m <= n} q^m g_i(m)` (updated
at every step, O(Npost)) the contribution of the presynaptic spike at step `n_k` up to step `n` is
`(dt A_LTP/τx) q^{-n_k} (K_i(n) - K_i(n_k - 1))`, so the weight is materialised from one snapshot of `K_i` per synapse
taken at the previous presynaptic spike of `j`. The Wmax clamp commutes with the accumulation (the LTP increments are
non-negative and, in a step, the LTD at presynaptic spikes comes before the LTP), so the result is that of the per-step
update up to rounding, not an approximation (checked in float64 on the validation input, one synapse, 10 s, with the bound
active: difference 1e-15). `K` is dominated by its oldest terms, so the difference `K(n) - K(n_k - 1)` loses all digits
after about 35 `τx`: the accumulator is restarted every few `τx` (about 10) by materialising all weights (one pass over the
synapses per epoch, amortised nnz / 1500 per step at `τx = 15 ms`, `dt = 0.1 ms`). Cost: O(Npost) per step for `u`, `v`,
`g`, `K`; O(fan-out) per presynaptic spike; no per-synapse work per step otherwise. `forward!` needs the weights of the spiking neurons only, so materialising in
`update_traces!` is enough; monitors and other readers of `W` need an explicit materialisation.

### Event-driven vSTDP (`vSTDPParameter`, default) and threading removed from the step path
- Rule `vSTDPParameter` / `vSTDPVariables` (event-driven, the default; the dense rule is `vSTDPDenseParameter` / `vSTDPDenseVariables`, the
  former `vSTDPParameter`; same fields plus `epoch = 10`, in units of `tau_x`). The text of this section was written when the event-driven rule was
  called `vSTDPEventParameter`. Same equations and step order; the LTP term is not evaluated
  per synapse per step: per-postsynaptic-neuron accumulator `K_i(n) = sum_m q^m g_i(m)` (`q = 1 - dt/tau_x`), per-presynaptic
  amplitude `a_j` (`x_j(n) = a_j q^n`, a spike adds `q^-n/tau_x`), per-synapse snapshot of `K_i` at the previous presynaptic spike.
  Weights are materialised at presynaptic spikes (in `update_traces!`, before `forward!`; LTD then acts on the materialised
  weight), once per epoch for all synapses (the Float64 accumulator loses its digits after about 35 `tau_x`; error of an increment
  about 2e-12 at the default 10 `tau_x`; `K`, snapshots and `a_j` are Float64, with Float32 the same cancellation would cost 1e-3),
  and on demand. A first spike of a synapse has `a_j = 0` before it (nothing pending). Exactness of the bounds: LTP increments are
  non-negative (`A_LTP >= 0` is enforced), LTD acts first in the step and is clamped at `Wmin` at the spike, so one clamp at `Wmax`
  after the accumulation equals the per-step clamp, also when LTD has pushed a weight to `Wmin`; the only corner is an initial
  weight above `Wmax` (dense: clamped at step 1, event: at the first materialisation or read; every read sees the clamped weight).
  A change of `dt`, `tau_x`, `A_LTP`, `Wmax` or `epoch` (checked every step) first materialises everything with the old values.
- Readers of `W`: `syn.W` of this rule holds the weights as of each synapse's last materialisation; read it through the same
  accessor as the lazily decayed traces, see "Reading the weights" below (this replaces the first version of this item, which had
  `current_weights(syn)` and `sync_weights!(syn)` as exported functions). The trace `x` is stored as the
  amplitude `a_j` and is declared lazy (`plasticity_value`/`monitor!` return the value). Recording does not change the state:
  monitored and unmonitored runs are bit-identical (tested).
- Validation (`papers/JuliaSNN_publication/validation/stdp_continuous`): on the protocol inputs the final weights differ from the
  dense rule by 2.5e-6 (200 synapses, 8e-6 of the largest change 0.306) and 2.9e-6 (1e4 synapses, 1.3e-5 of 0.222), and from Brian2
  (float64) by 1.0e-6 and 7.8e-7, against 2.3e-6 and 2.8e-6 for the dense rule: the difference is the Float32 accumulation error of
  the dense rule. Tests: `test/syn/vstdp.jl` (was `vstdp_event.jl`; 55 tests: dense and Float64 reference, trajectory, tiny/large epoch, silent
  synapses longer than an epoch, Wmax and Wmin bounds, dt and `A_LTP` changed mid-run, records, bit-identical monitoring, 0 B per step).
- Timing, 20 s, dt 0.1 ms, one pinned thread, median of 3 (us per step; 200 / 1e4 synapses): dense with `Threads.@threads` 1.16 / 10.4,
  dense without threads 0.258 / 9.40, event-driven 0.130 / 0.181, Brian2 cpp_standalone 0.452 / 13.0 as reported
  (Brian2 is compared as reported only). The first 2.7x gap to Brian2 at 200 synapses was the threading dispatch (0.74 us per step with one
  thread), not the LTP arithmetic (JuliaSNN's loop is 1.0 vs 1.16 ns per synapse at 200 and 0.79 vs 1.14 at 1e4); see the README
  of that directory.
- Threading removed from the step path pending the `neurons_per_thread` design; behaviour-preserving. The `Threads.@threads`
  loops of `vSTDPDenseParameter` (chunks of presynaptic neurons), `AggregateScaling` and `SynapseNormalization` (two each) are plain
  loops with the same arithmetic and order; `src/analysis/**` is untouched. Checked: `vSTDPDenseParameter` final weights bit-identical
  before and after on the validation tasks (200 and 1e4 synapses) and on a seeded Poisson-driven run; `SynapseNormalization`
  (multiplicative and additive) bit-identical on a seeded run; `AggregateScaling` differs by one Float32 unit in the last place in
  178 of 960 weights (max 4.8e-7, 2.3e-7 relative; accepted, recorded here, no further work): the row sum `Wt[i] += ...` is a `@fastmath @simd` reduction whose summation
  order depends on how the compiler vectorises the loop, which changed with the loop leaving the threaded closure. The suite
  passes (29774 tests, previously 29719 plus the 55 new; 30052 with the weight accessor and precision tests: `lazy_weights.jl` 41, `precision.jl` 229, the vSTDP tests of `with_plasticity.jl` and `sweep_fixes.jl` run for both rules).

### Reading the weights: one accessor for traces and for deferred weights
A rule that defers its weight update (the event-driven `vSTDPParameter`) now uses the mechanism of the lazily decayed traces:
- `lazy_variables(vars)` may list `:W` (`vSTDPVariables` lists `(:x, :W)`). `plasticity_value(syn, :W)` (also `trace_values(syn, :W)`) returns the
  current weights, state unchanged: `syn.W` itself for every other rule, a scratch vector (no allocation after the first call) with the pending
  potentiation added and clamped for a deferred rule. `sync_traces!(syn)` materialises `syn.W` in place (restarts the accumulation epoch; the
  rounding level of a materialisation is 1e-12 relative in Float64, 1e-7 in Float32 for the STDP traces) and is the one entry point for code that
  needs `syn.W` itself. The extension points are `current_weights(c, vars)` and `sync_weights!(c, vars)` (a rule that lists `:W` defines both).
  `current_weights(syn)` and `sync_weights!(syn)` remain as thin non-exported wrappers (they were exported in the development tree).
- Fail loudly: a variables type that lists `:W` in `lazy_variables` without `current_weights` / `sync_weights!` methods throws an `ArgumentError` from
  `plasticity_value(syn, :W)`, `trace_values`, `sync_traces!`, `matrix`, and from `monitor!(syn, [:W])` at `monitor!` time (not in the middle of a run),
  exactly as a lazily stored field without `current_value` does. `monitor!(syn, [:W])` reads through the accessor (`get_model_field`).
- Readers of `syn.W` in the library, and what was done: `matrix(c)` reads through the accessor (a new sparse matrix; was stale);
  `update_weights!` (both forms), `update_sparse_matrix!` (both forms, hence `connect!`) and `synaptic_turnover!` materialise first (a value set
  by the user must not receive the pending potentiation, and the per-synapse snapshot is not reordered by a rewiring); `SynapseNormalization`,
  `AggregateScaling`, `Turnover` (immediate `plasticity!` methods; and the constructors of `SynapseNormalization` and `AggregateScaling`, which sum the weights into `W0`/`WT`) materialise the synapses before they rescale (the pending increment was not
  rescaled before: a limitation of the first version, now removed); `with_plasticity` and `update_plasticity!` materialise before the state of the old rule is dropped;
  `save_model`/`SNNsave` (both files) materialise every sparse synapse of the model before writing, so a saved model never stores a stale `W` (the epoch
  restarts: a perturbation of the later trajectory at the rounding level; the saved state is otherwise the live one); `modelcopy` copies the deferred
  state as it is (a copy continues the run exactly; its `W` is as stale as the original's, read it through the accessor). Outside SNNModels:
  `SNNPlots.stdp_test` (single pair kernel measurement: stale with `vSTDPParameter`), `SNNPlots.plot_weights` and `mutual_EI_connections`
  (iSTDP synapses, not deferred), `SNNUtils.average_weight` and the tutorial `SpikingNeuralNetworks/examples/tutorials/plasticity/AdEx_plasticity.jl`
  (weight histogram after training) now read through `plasticity_value`. Not changed: `record`ed `:W` (the recorder reads through the accessor, so
  recorded weights are values), `SNNUtils` functions that take a recording, `SNNPlots` functions that take `record(syn, :W)`. Code of other
  projects that reads `syn.W` after a run with `vSTDPParameter` must use the accessor or `sync_traces!`. After these changes no direct read of `.W` of a
  synapse remains in SNNPlots, SNNUtils (included files) or SpikingNeuralNetworks `src` except commented code and the legacy file
  `SNNUtils/src/analysis/performance.jl`, which is not included (it also writes `W` in `update_weight!`); `SNNUtils.average_weight_dynamics`
  takes a record. Uses of `vSTDPParameter` in the other packages (now the event-driven rule): `SNNUtils` parameter sets `quaresima2023`
  and `dendrite_STM` (`vstdp = vSTDPParameter(...)`, `A_LTP >= 0`, they work unchanged; `dt < τx` holds), the `SpikingNeuralNetworks` tutorial
  `examples/tutorials/plasticity/AdEx_plasticity.jl` (updated), its docs (`plasticity.md`, `catalogue/plasticity_rules.md`, updated),
  `SpikingNeuralNetworks` explicit export list (`vSTDPDenseParameter` added), `docs/src/catalogue/snnutils_models.md` (describes the parameter sets), commented
  uses in `examples/utils/{load_cluster,get_graph,load_model}.jl`, and the validation scripts in `papers/JuliaSNN_publication/validation/sweep_bugs`
  (`measure_other.jl`, a one-synapse dt-independence check: now reads `W` and `x` through `plasticity_value`; rerun with 1.10: dW of the
  one-pre-spike test 0.30327 / 0.30306 / 0.30293 at dt 0.125 / 0.0625 / 0.025, peak x 0.0667, dW at rest 0, i.e. the dt independence holds, as in
  the 1.9.x data).
- Tests: `test/syn/lazy_weights.jl` (41).

### Precision from the population type (Float32 default, Float64 by the type of the populations)
The precision of a model is chosen once, on the populations, with the type parameters they already have, and everything else follows. No keyword,
no global flag, no separate precision argument.
- Selecting it: `Identity(N = 10, VFT = Vector{Float64})`, `IF(N = 10, VFT = Vector{Float64})` (its `param`, `synapse` (`DoubleExpSynapse`), `spike`,
  the synaptic variables and the receptor buffers default to `Float64` from `VFT`; passing a `Float32` object to a `Float64` IF is an `ArgumentError`,
  the same the other way). `float_type(x)` (exported) is the element type of a population, connection, stimulus, rule, clock or type: the first
  type parameter that is a float or a vector of floats.
- A connection takes the type of the variable it writes into (`SpikingSynapse`: `eltype` of the target `g`): `W`, `ρ`, delay times, delay
  ring buffer, the state of the plasticity rule (`plasticityvariables` takes the type of the rule, `float_type(rule)`), the rules `LTPParam`/`STPParam`
  themselves (`convert_precision(rule, FT)`: a copy with its floating fields and type parameters converted; the object itself if it already has
  the type, so a `Float32` model uses the very objects the user passed), `sparse_matrix(...; FT)` (draws the weights in `FT`; the matrix of an explicit
  `conn` is converted). Stimuli (`SpikeTimeStimulus`, `PoissonStimulus`, `PoissonStimulusLayer`) take the type of their target the same way
  (weights, spike times, `next_spike`, parameter). `AggregateScaling` and `SynapseNormalization` take the type of their synapses. Mixed precision:
  a Float32 presynaptic population onto a Float64 one gives a Float64 synapse (the synapse follows its target); a model of populations of both
  precisions runs, each component in its own precision (the loops pass `dt` converted to the precision of the component) with a `Float64` clock;
  a population that mixes precisions, or a connection onto a target variable whose element type differs from its membrane potential, is an
  `ArgumentError`. Components that have no Float64 model refuse a Float64 target at construction with an `ArgumentError` (`CurrentStimulus`,
  `BalancedStimulus`, `Turnover`) or when stepped (`integrate!` of a population without a Float64 method, e.g. `AdEx(VFT = Vector{Float64})`)
  instead of copying the target variable and silently stopping to write into it.
- The clock is `Time{FT}` (`Time()` is `Time{Float32}`, `Time{Float64}()`); `compose` creates `Time{FT}` with `FT` the promotion of the
  populations' types (`time = nothing` by default; was `Time()`), `sim!`/`train!` convert `dt` to the clock type; `get_time` returns the type of the clock.
  The lazy trace scale, the vSTDP accumulators (`K`, snapshots, amplitudes) and the cache vectors stay `Float64` in both precisions (they were).
- Kernels and signatures generic over the type: `dt::Float32` became `dt::AbstractFloat` in the connection, plasticity, metaplasticity, stimulus
  and `Identity`/`IF` methods (about 70 signatures; the Float32 methods of the components that are not converted keep `dt::Float32`),
  `_rescale!`, `_advance_scale!`, `_read_scale`, `_write_inc`, `_decay_factors!` (the factors `exp(-dt/τ)` are computed in the precision of `dt`),
  `_spike_increment!`, `MexicanHat`, the Markram relaxation, the delay steps; literals `0.0f0` -> `zero(T)`. The default of a field of a `@snn_kw`
  struct may now refer to the type parameters (the type parameters are the first keyword arguments of the generated constructor).
- Float32 is unchanged and bit-identical: the dense `vSTDPDenseParameter` on both validation tasks (200 and 1e4 synapses) equals the stored weights
  of the previous implementation bit for bit, the event-driven rule equals the previous implementation (run side by side) bit for bit on both tasks,
  and the suite (30052 tests) passes; the arithmetic of every other rule is the same expression with `Float32` literals replaced by
  `zero(T)`/`oftype`. The dense rule keeps `clamp(x, z, Inf)` (its Float32 arithmetic already promotes through the `Float64` `Inf`: kept for
  bit-identity; in Float64 everything is Float64).
- Recording: `Float64` variables are recorded in the legacy (exact) mode; the dense buffers are `Float32`.
- Defaults and units: the unit constants (`ms`, `pA`, `Hz`, ...) and the default values of the rules are `Float32` numbers. A `Float32` rule used on a
  `Float64` population keeps its `Float32` values (8e-4 is 8.0000002e-4); build the rule with `FT = Float64` and `Float64` values to have them exactly
  (`vSTDPParameter(FT = Float64, A_LTD = 8e-4)`). `SpikeTimeParameter(times, neurons)` rounds the times to `Float32`: for exact `Float64` times use the
  positional `SpikeTimeStimulusParameter(times, neurons)` or the keyword form of `SpikeTimeParameter` (sorted times).
- Done / not done. Stage 1 (done): sparse synapses and all their plasticity rules (STDP pair/triplet/weight-dependent/Confavreux/MexicanHat/structured,
  iSTDP, vSTDP both rules, Markram STP event, Het, Timestep), delays, `SpikeTimeStimulus`, `PoissonStimulus`, `PoissonStimulusLayer`, `Identity`,
  `AggregateScaling`, `SynapseNormalization`, clock, `sim!`/`train!`, records. Stage 2 (done for): `IF` with `DoubleExpSynapse`. Not converted
  (refuse Float64 at construction or when stepped): the other populations (`AdEx`, `Tripod`, `BallAndStick`, `Poisson`, `HH`, `IZ`, `MorrisLecar`, `Rate`, ...), the
  other synapse models of `IF` (`SingleExp`, `Delta`, `Current`, `Receptor`, `Confavreux2025`), `Turnover`, the rate/FORCE synapses, `CurrentStimulus`,
  `BalancedStimulus`. Estimate: about 25 population/synapse-model methods and their `VFT` defaults (`zeros(Float32, N)` defaults in about 20
  structs, `dt::Float32` in about 75 methods), the same recipe (type-parameter defaults from `VFT`, `dt::AbstractFloat`, `zero(T)`), one population at a time.
- Tests: `test/syn/precision.jl` (229): the same small network in Float32 and Float64 (STDPGerstner pair, triplet, event vSTDP, dense vSTDP, STP event, Poisson
  stimulus, `SpikeTimeStimulus` onto `Identity`, `PoissonStimulusLayer`, delay synapse, normalization, aggregate scaling) with the eltypes of every state
  array, the clock, `@inferred` on a step of each kind of component, an analytic pair-STDP check in Float64 (1e-12), mixed precision and refusals.
- Float64 reconciliation of the two vSTDP rules (validation, `papers/JuliaSNN_publication/validation/stdp_continuous/reconcile_float64.jl`): in Float64 the
  event-driven and the dense rule agree to 3.6e-14 (200 synapses) and 8.2e-14 (1e4 synapses), and with Brian2 (float64) to 3e-15 (dense) and 3e-14 to 9e-14
  (event), against 2.4e-6 / 2.9e-6 between them in Float32: the Float32 difference is the accumulation error of the dense rule (hypothesis confirmed).
  Float64 costs 13 % on the dense rule at 1e4 synapses (2.09 s against 1.88 s, unpinned) and nothing measurable on the event-driven one.

### Dependencies: unused packages removed (branch dev-diag, pre-1.10 diagnostics, job 1)
- Removed from `[deps]`/`[compat]` of `Project.toml` (13): `Documenter` (no docs build in this package; the
  SpikingNeuralNetworks docs environment has its own), `BenchmarkTools` and `Revise` (only in standalone scripts
  `test/simulation_speed.jl`, `test/pop/multipod.jl`, `_new_feats/`, not run by `runtests.jl`), `Printf` (test only, already
  in `test/Project.toml`), `ColorSchemes`, `IterTools`, `Parameters` (`@with_kw` only in `sparse_plasticity/dump.jl`, which is
  not included; `@unpack` comes from `UnPack`), `Profile`, `Requires` (no `@require`), `Roots` (only before commented-out code
  in `utils/sparse_matrix.jl`), `SGtSNEpi`, `StaticArrays`, `TimerOutputs`. The matching `using` lines were removed from
  `src/SNNModels.jl`, `src/utils/spatial.jl`, `src/utils/sparse_matrix.jl`, `src/connections/sparse_plasticity/CaRule.jl`.
  No code path changes; 20 runtime dependencies remain.
- `test/Project.toml`: `SHA` added (`test/threading_synapses.jl` does `using SHA`; the testset errored with "Package SHA not
  found" on 6b7addc as well).
- Measured (`papers/JuliaSNN_publication/validation/profilation_v1.10/perf/pre110_diag/job1_load.csv`, `job1_load.jl`;
  core 8, `-t 1`, median of 3, precompiled): `julia -e 'using SNNModels'` wall 2.58 s to 2.03 s (bare Julia 0.19 s),
  VmRSS after `using` 582 MB to 515 MB (bare 221 MB). Packages in the resolved manifest 190 to 138 (`job1_manifest_counts.csv`).
  Most of the gain is the Documenter tree (Documenter, REPL, `Pkg -> REPLExt`, JSON, Parsers, Git: about 360 of the 392 ms of
  `@time_imports` entries that disappear, `job1_time_imports_diff.csv`).
- Per direct dependency (fresh process `using Dep`, `job1_deps.csv`; trees overlap, so these do not add up): load time
  over bare Documenter 0.68 s, SGtSNEpi 0.60 (was never loaded by SNNModels), DrWatson 0.52, JLD2 0.50, DSP 0.43,
  MetaGraphs/Graphs 0.39, Interpolations 0.34, Distributions 0.33, LoopVectorization 0.28 s; RSS over bare Documenter
  +204 MB, DrWatson +162 (pulls in Pkg), DSP +128, JLD2 +127, MetaGraphs +119, LoopVectorization +111, Distributions
  +106, Interpolations +95 MB. Heavy runtime dependencies left for the maintainer to decide on: DrWatson (`savename`, `@strdict`,
  `dict2ntuple`, `save`/`load` in `utils/io.jl`), DSP (`conv`, `xcorr` in the analysis), Graphs/MetaGraphs (`graph(model)` only), JLD2.
- Other packages, `Project.toml` only, each on its own `dev-diag` branch (worktrees `src/snnplots_diag`, `src/snnutils_diag`,
  `src/snnrl_diag`): SNNPlots drops `Plots` (only in files that are not included: `old_plot.jl`, `backend/plots.jl`), `DrWatson`
  and `Colors` (not used), and moves `CairoMakie` (used only by the tests and in docstring examples) to `[extras]`/`[targets]`;
  SNNUtils drops `Dates`, `Interpolations`, `JLD2`, `Requires` (used only in files that are not included, or not at all;
  `HDF5` stays: `DrWatson.save` of `.h5` files loads it through FileIO); SNNRL moves `ReinforcementLearning` and
  `ReinforcementLearningTrajectories` (used only by `examples/1d_RW.jl`) to a new `examples/Project.toml`. SpikingNeuralNetworks,
  CalciumSpike, OptunaLoader: no change (all dependencies used by the loaded source).

### Known limitations
- `SNNModels.compose` (`utils/util.jl:212-214`) calls `sort(::Dict; by)`, a deprecated method that OrderedCollections adds
  to `Base.sort` (warning "`sort(d::Dict; args...)` is deprecated" in the SNNPlots/SNNUtils test runs); SNNModels does not
  depend on OrderedCollections directly, so a release of OrderedCollections that drops the method breaks `compose`.
- `iSTDPPotential` and the dense vSTDP are not converted (the event-driven vSTDP keeps the postsynaptic filters per step, O(Npost)): their postsynaptic filters follow the membrane potential at every step
  (class C, see "Lazy classification of the plasticity rules"); `SpikeRateSynapse` keeps its dense scans. The clock-driven
  `MarkramSTPParameterTimestep` (lazy-able with the Euler factor per elapsed step, but that needs a dt-aware elapsed step
  count and it is the dense reference of the event rule) and the traces of `ActivityDependentTurnover` (read once per
  turnover period) are lazy-able and not converted.
- The sparse hand-off needs every writer of an `Identity` to be a `SpikeTimeStimulus` (any other writer, such as a connection or
  a stimulus without a log, makes the population use the dense update; results are the same, only slower). Other populations (`Poisson`, `IF`, ...) do not keep a spike list: their synapses scan `fire` once per step.
- Models and JLD2 files saved with 1.9.1 or earlier are refused by `load_model`/`load_data`/`SNNload` with an `ArgumentError`
  (see Breaking changes); direct `JLD2.load` of such a file returns placeholder objects.
- If the delay buffer is resized (`dt` or delays changed while spikes are in flight), the queued increments are
  dropped; the in-flight increments are not rescaled when `dt` changes without a size change.
- The Float32 clock `T.t[1] += dt` still drifts (work in progress on the local branch `wip-clock`): stimuli that
  compare spike times with the clock (`SpikeTimeStimulus`) are affected at `dt` values that are not exactly
  representable; delays are not any more (counted in steps).

### Not yet done
- `iSTDPPotential`
  `tpre`; `SpikeRateSynapse` to the spike lists. Benchmark inputs for the comparison with Brian2:
  `papers/JuliaSNN_publication/validation/stdp_continuous/`.
- JLD2 upgrade (typemap `Upgrade`) so that models saved with 1.9.1 load.
- Parallel plasticity (partition by postsynaptic neuron for the post pass, by presynaptic neuron for the pre pass).
- Integer clock (`wip-clock`).
- Documentation pages of SpikingNeuralNetworks that still describe `change_plasticity!`/`update_plasticity!` as
  replacing a rule of any type (`docs/src/plasticity.md`, `docs/src/catalogue/connections.md`) and the delay queue entry
  of `claude/performance.md` ("Remaining known performance issues"); release notes entry (with the breaking-changes
  table, as for 1.9.0) when 1.10 is released; Project.toml is still 1.9.1.

## 1.9.1: recording buffers across chunks
- `_extend_fire!` and `_extend_dense!` (called when a new `sim!`/`train!` chunk needs more room)
  grew by exactly the missing amount, so a run of many chunks reallocated and copied the whole
  buffer at almost every chunk (quadratic copying, garbage per chunk). They now grow
  geometrically (`max(old + extra, 2 old)`), and warn once per recording, sharing the latch of
  the mid-loop overflow path. Reported by SNN2AC.

## fix/sweep-bugs: membrane pair rule (1.9.0)
- `AdExParameter`, `IFParameter`: `C`, `gl`, `R`, `τm` always consistent. Construction from a
  pair (or the default pair); a single value is an error; more values must agree within 0.1 %.
  One changed value follows a fixed rule (τm keeps gl; C keeps gl; gl/R keeps C) in `@update!`,
  `setproperty!` (mutable AdEx), `with_membrane`, `make_heterogeneous`.
- Reason: AdEx/IF integrate with τm, R; Tripod/BallAndStick with C, gl. `@update! adex.τm` had no
  effect on Tripod; `IFParameter(τm = x)` silently combined with the default R.
- `@snn_kw` keyword constructors pass their fields through `snn_kw_finalize(T, nt)` (identity by
  default) before building the struct.

## core-clock-sparse

### `sparse_matrix` without a dense matrix
- All NamedTuple rules (`:Fixed`, `:FixedIn`, `:FixedOut`, `:Bernoulli`, `:PowerLaw`) build
  CSC directly in O(nnz) memory; return type `SparseMatrixCSC{Float32,Int}` as before.
- Behaviour change: seeded networks do not reproduce the old realisations (different random
  stream). Statistics are the same (tested against `sparse_matrix_dense_legacy`).
- Behaviour change: recurrent `SpikingSynapse(pre, pre, ...)` no longer stores zero-weight
  autapses (they were removed only by value before, and could be potentiated).
- `SpikeTimeStimulusIdentity` uses a sparse identity.

### iSTDP kernels
- iSTDPRate/iSTDPPotential: the postsynaptic pass reassigned its loop variable
  (`st = index[st]`) inside `@turbo`, which LoopVectorization does not support; iSTDPRate
  diverged from a plain loop by up to 22 pF. Now plain `@inbounds @simd` loops.

The integer-clock work lives on the local branch `wip-clock`.
