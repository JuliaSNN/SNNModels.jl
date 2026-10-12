# SNNModels 2.0.0: in-depth changelog

Changes from 1.9.1 (commit 72dd815) to 2.0.0. The release summary is `CHANGELOG.md`; the raw development log, with every
measurement, is `claude/CHANGELOG.md`. Part 1 changes how spikes move and how results are computed, Part 2 lists the
struct, function and file changes, Part 3 the changes that only affect speed and memory.

## What to do when upgrading

- Require SNNModels 2 in `[compat]`; use SNNUtils 2, SNNPlots 2, SpikingNeuralNetworks 2 and CalciumSpike 2.
- Rebuild saved models from their configuration: files of 1.9.1 and earlier are refused at load (2.6).
- Install a rule of another type with `with_plasticity`, `without_plasticity` or `change_plasticity`;
  `update_plasticity!` / `change_plasticity!` and field assignment accept only the current rule type (1.6).
- Read plasticity state through the accessors: `plasticity_value(syn, :W)` for the weights of `vSTDPParameter`,
  `trace_values` / `plasticity_value` / `sync_traces!` for traces. For the 1.9.1 voltage rule use `vSTDPDenseParameter` (1.5).
- Before reading `syn.J` / `syn.index` call `SNNModels.ensure_transpose!(syn)`; read `syn.ρ` with `SNNModels.efficacy`,
  call `SNNModels.materialize_efficacy!` before writing it; `param.delaytime` may have length 1 (2.1).
- Regenerate seeded reference results that involve Poisson inputs, delays at a `dt` such as 0.1 ms, or STDP weights (1.3-1.5).
- Replace positional constructors of the changed structs by keyword constructors (2.1).
- For Float64 build the populations with `VFT = Vector{Float64}` and the rules with `FT = Float64` (1.7).
- Pass `flush_subnormals = false` to `sim!` / `train!` if a result depends on subnormal numbers (1.8).
- Custom components must not write `pop.fire` of an integrate-and-fire population between its `integrate!` and the end of
  the step (1.1).

---

## Part 1. Semantics

### 1.1 Spike lists

**Index lists instead of flag scans**
- Change: connections, event-driven rules and the `:fire` recorder read the spikes of a step as an ascending `Int32` list
  (`SpikeIndex`) built from the `Bool` vector `fire`, which populations and stimuli still write. Three kinds of list:
  scratch (`SpikeIndex(N)`, private to a connection; `prepare_spikes!` opens and `release_spikes!` closes it around the
  step of that connection, one scan per connection), fed (written by its owner, no scan: `Identity` under list-aware
  stimuli, Poisson populations in the sparse regime, `PoissonStimulusLayer`), owned (one per population of the
  integrate-and-fire family, below).
- Consequence: a list equals `findall(fire)` of the step, so the spikes delivered are unchanged. Outside the step window
  (direct calls of `forward!`, `plasticity!`, `record!`; manual loops) every call rescans `fire`. For fed-list owners, flags
  written by hand into `fire` are ignored by connections and recorder while the list is fed.
- Where: `src/populations/spike_index.jl` (`SpikeIndex`, `spike_list!`, `fed_spike_index`, `spike_index`, `_arm!`,
  `_disarm!`); `src/connections/spiking_synapse.jl` (`presynaptic_spikes`, `postsynaptic_spikes`, `prepare_spikes!`,
  `release_spikes!`); `src/connections/connections.jl` (generic `presynaptic_spikes` fallback).
- Migration: none for keyword-built models. A custom connection keeps working through the generic fallbacks; a custom
  population with its own list returns it from `spike_index`.

**One list per integrate-and-fire population: publish/retire window**
- Change: `IF`, `AdEx`, `ExtendedIF`, `HetRec`, `Tripod`, `BallAndStick`, `HH`, `IZ`, `MorrisLecar`
  (`_OWNED_LIST_POPULATIONS`) own `spikes = owned_spike_index(N)`; `spike_index(pop)` returns it, so every `SpikingSynapse`
  built on the population and the recorder share it. `sim!` / `train!` call `_publish_spikes!(p)` right after
  `integrate!(p)` (O(1): window open, list not built); the first consumer scans `fire`, later consumers reuse the list;
  `_retire_all_spikes!(P)` closes every window at the end of the step. `with_plasticity`, `modelcopy`, `save_model` /
  `load_model` keep the sharing.
- Contract: no component may write `pop.fire` between `integrate!(pop)` and the end of the step; flags set there by a
  custom component are not seen by the consumers after the first. If a population appears twice in `P`, its second
  `integrate!` publishes again. `spike_list!` on a shared list must not be called inside a `foreach_chunk` body.
- Failure: a step that throws is a fatal error and the run is discarded. Nothing closes the windows after an exception
  (no `try/finally`), so a list may stay open and stale; the next `sim!` / `train!` step publishes afresh. For debugging,
  call `SNNModels._retire_all_spikes!(P)` before calling `forward!`, `plasticity!` or `record!` by hand.
- Where: `src/populations/populations.jl` (`_OWNED_LIST_POPULATIONS`, `spike_index`, `_publish_spikes!`,
  `_retire_spikes!`); `src/populations/spike_index.jl` (`owned_spike_index`, `_retire_all_spikes!`); `src/utils/main.jl`
  (`_sim_component!`, `_train_component!`, `_sim_step!`, `_train_step!`); `src/utils/record.jl` (`_recording_spike_index`,
  `_fire_step_list`, `_has_step_list`).
- Migration: positional constructors of the nine types change arity (new field `spikes` after `fire`).

### 1.2 Sparse hand-off, stimulus to `Identity`

**Input logged by the stimuli**
- Change: `SpikeTimeStimulus`, `PoissonStimulus` (built-in parameters `PoissonFixed`, `PoissonVariable`, `PoissonInterval`)
  and `PoissonStimulusLayer` log the neurons of an `Identity` target they write (`inbox`, a `TouchList` returned by
  `attach_driver!`). `Identity` counts its writers (`nwriters`, incremented by `synaptic_target`) and its logging drivers
  (`ndrivers`). While the two agree, `integrate!` visits only the logged neurons and those that fired the step before
  (`_integrate_sparse!`) and feeds `spikes`; any other writer, an inbox overflow, or the first step after a change of
  writers uses the dense update. Stimuli clear their own `fire` flags where they set them, not with `fill!`.
- Consequence: `fire`, `spikecount`, `h`, `g` equal the dense update at every step. A flag written by hand into `stim.fire`
  of a `SpikeTimeStimulus` or `PoissonStimulusLayer` is no longer cleared. A `PoissonStimulus` with a user-defined
  `PoissonStimulusParameter` does not log; its `Identity` target integrates densely.
- Where: `src/populations/identity.jl` (`integrate!`, `_integrate_sparse!`, `release_input!`, `attach_driver!`),
  `src/populations/spike_index.jl` (`TouchList`, `attach_driver!`), `src/stimuli/timed.jl`, `src/stimuli/poisson.jl`
  (`_logs_writes`), `src/stimuli/poisson_layer.jl`.
- Migration: do not write `g`, `h` or `fire` of a stimulus-driven `Identity` by hand. If a connection built onto an
  `Identity` is redirected to another vector after construction, call `release_input!(pop)`.

### 1.3 Delays

**Ring buffer, delays rounded up to whole steps**
- Change: the per-synapse queue is replaced by a ring buffer of `Npost x (max delay steps + 1)` increments. A delay
  becomes `ceil(delay / dt)` steps (1e-4 step tolerance), computed at the first `forward!` and again when `dt` or the
  length of `delaytime` changes; the delay is counted in calls of `forward!`.
- Consequence: 0.8 ms at `dt = 0.125` arrives after 7 steps (0.875 ms). At `dt = 0.125` buffered delays give the results
  of 1.9.1. At `dt` not exactly representable in Float32 (0.1, 0.05, 0.025 ms) 1.9.1 compared arrival times with the
  drifting Float32 clock and delivered a delay that is an exact multiple of `dt` one step late in 22 to 80 % of the
  events; 2.0 is exact, so chaotic trajectories differ from 1.9.1. If the buffer must be resized (`dt` or delays changed),
  queued increments are dropped.
- Where: `src/connections/spiking_synapse.jl` (`SpikingSynapseDelayParameter`, `_delay_steps!`, delayed `forward!`),
  `src/connections/delay_kernels.jl`, `src/utils/sparse_matrix.jl` (`_remap_synapse_state!`).
- Migration: `spike_time` / `spike_w` have no equivalent (read `param.buffer`). After editing `param.delaytime` in place
  set `param.dt[1] = NaN` so that the steps are recomputed.

**Common delay stored once; without plasticity delivered from the spike list**
- Change: if all drawn delays are equal (`Dirac(d)`), `delaytime` and `delaysteps` have length 1 (same draw, random stream
  unchanged). Such a synapse with `NoLTP` and `NoSTP` delays the spike list (`param.spike_ring`, the lists of the last
  `d + 1` steps) and scatters the list emitted `d` steps ago into `g`; no buffer is allocated.
- Consequence: on this path `W` is read at arrival, not at emission (a weight changed in between, also by
  `SynapseNormalization` or `AggregateScaling`, is delivered with its new value), and the increments of a step are added
  one at a time (few-ulp differences; chaotic runs diverge, statistics unchanged). A network with a `Dirac` delay and the
  same network with per-synapse `delaytime` therefore give different bits. After `update_sparse_matrix!` spikes in flight
  on this path are delivered through the new connectivity. Delays that differ, and synapses with LTP or STP, keep the
  buffer (increment at emission).
- Spikes in flight across a plasticity switch (`with_plasticity` adds or removes the rules; both synapses share `param`)
  are delivered once: towards the buffer the pending lists are scattered with the current `W` and no efficacy; towards
  the spike list the buffer is drained over `d` steps (`param.drain`) and then released. A `dt` change that keeps the
  delay in steps keeps them; one that changes it drops them.
- Where: `src/connections/spiking_synapse.jl` (constructor), `src/connections/delay_kernels.jl` (`_use_spike_ring`,
  `_forward_spike_ring!`, `_delay_buffer!`, `_forward_delay!`).
- Migration: code that indexes `delaytime` / `delaysteps` per synapse must check their length.

### 1.4 Poisson sampling

**Sparse samplers, new random streams**
- Change: stimuli (`PoissonStimulus`, `PoissonStimulusLayer` with `PoissonLayer` and `PoissonLayerHet`) and populations
  (`Poisson`, `InhomogeneousPoisson`, `VariablePoisson`) draw one random number per spike: geometric skipping to the
  next firing neuron, or thinning at the largest per-step probability for per-neuron rates. When that probability reaches
  0.03 for the populations (`_POP_DENSE_SPIKE_P`) or 0.2 for the layers (`_DENSE_SPIKE_P`), every neuron is tested with
  one uniform number; the populations then do not feed their list (`spikes.fed = false`) and connections scan `fire`.
- Consequence: the laws are those of 1.9.1 (Poisson counts per neuron and step for `PoissonStimulus`, Bernoulli for the
  layers and populations, independent; tested in `test/stim/poisson_sparse.jl` and `test/stim/poisson_pop_sparse.jl`), but
  the global random stream is consumed differently: a seeded run of 1.9.1 is not reproduced, neither are the random
  numbers drawn after these components. Runs of 2.0 are reproducible for a fixed seed.
- Where: `src/stimuli/poisson.jl`, `src/stimuli/poisson_layer.jl`, `src/populations/poisson.jl`,
  `src/populations/inhomogeneous_poisson.jl`.
- Migration: code that used `randcache` of `PoissonStimulusLayer`, or a custom `stimulate!` relying on it, draws its own
  numbers.

### 1.5 Lazily stored plasticity state

**Traces stored as value times scale**
- Change: the exponential-trace rules (`STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent`, `STDPTriplet`) and the
  Euler-trace rules (`iSTDPRate`, `STDPMexicanHat`, `STDPSymmetric`, `STDPAntiSymmetric`) store each trace as
  `stored * scale` (one Float64 `scale` per trace type; a step multiplies the scale, a spike adds `1/scale`, the stored
  values absorb the scale when it falls below 1e-6). Euler rules keep the factor `1 - dt/τ` and their order of decay and
  increment. `iSTDPPotential` is not converted (`scale = 1`).
- Consequence: a direct read of `syn.LTPVars.tpre` (`tpost`, `r1`, `r2`, `o1`, `o2`, `to_x`, `to_y`, `tr_x`, `tr_y`) returns
  stored values, not traces. Weights differ from 1.9.1 by Float32 rounding: up to about 1e-5 relative in long chaotic runs
  for the exponential rules, 3e-7 to 8e-7 of the largest weight for the Euler rules. Monitored and unmonitored runs are
  bit-identical (the recorder never changes the state).
- Migration: read through `trace_values(syn, :tpre)`, `plasticity_value(syn, :LTPVars, :tpre, t)`, or call
  `sync_traces!(syn)` first; `monitor!`, `record`, `getvariable` return values. A custom variables type declares its lazy
  fields in `lazy_variables(vars)` and reads them in `current_value(vars, param, key, t)`; a lazy field without accessor
  is an `ArgumentError` at `monitor!` time.

**Markram STP variables of the event rules**
- Change: `syn.STPVars.u`, `.x` of `MarkramSTPParameterEvent` and `Het` are the values at the last presynaptic spike;
  `plasticity_value` relaxes them exactly to the requested time.
- Consequence: recorded `u` and `x` are now values at the sampling time, where 1.9.1 recorded the stored values; the two
  versions differ by up to 0.135 (`u`) and 0.77 (`x`). `ρ` is unchanged. Comparisons of these recordings across versions break.

**`vSTDPParameter` is event-driven; deferred weights**
- Change: `vSTDPParameter` / `vSTDPVariables` are the event-driven voltage rule; the 1.9.1 clock-driven rule is
  `vSTDPDenseParameter` / `vSTDPDenseVariables` (same fields, same numbers). The event rule accumulates the LTP term per
  postsynaptic neuron and writes `W` at presynaptic spikes, once per epoch (`epoch = 10` in units of `τx`) and on demand.
- Consequence: between materialisations `syn.W` is stale. Weights agree with the dense rule to its Float32 accumulation
  error (about 1e-5 relative at 1e4 synapses). Enforced at construction or first step (`ArgumentError`): `A_LTP >= 0`,
  `0 < epoch <= 25`, `dt < τx`; a negative `A_LTP` needs `vSTDPDenseParameter`.
- Readers that materialise or read through the accessor: `matrix`, `monitor!(syn, [:W])`, `update_weights!`,
  `update_sparse_matrix!` (hence `connect!`), `synaptic_turnover!`, `SynapseNormalization`, `AggregateScaling`, `Turnover`
  (constructors and every application), `with_plasticity`, `update_plasticity!`, `save_model` / `SNNsave`. `modelcopy`
  copies the deferred state as is. In the sibling packages `SNNUtils.average_weight` and `SNNPlots.stdp_test` were changed.
- Where: `src/connections/sparse_plasticity/vSTDP.jl`; `src/connections/sparse_plasticity.jl` (`plasticity_value`,
  `trace_values`, `sync_traces!`, `lazy_variables`, `current_value`, `current_weights`, `sync_weights!`);
  `STDP_kernels.jl`, `STDP_traces.jl`, `STDP_triplet.jl`, `STDP_weight_dependent.jl`, `STDP_structured.jl`, `iSTDP.jl`,
  `STP.jl` in `src/connections/sparse_plasticity/`; `src/utils/sparse_matrix.jl`, `src/utils/io.jl`, `src/utils/record.jl`.
- Migration: read `plasticity_value(syn, :W)` or call `sync_traces!(syn)` before using `syn.W`; to keep 1.9.1 numbers
  exactly, replace `vSTDPParameter` by `vSTDPDenseParameter`.

### 1.6 Switching plasticity

**`with_plasticity`, `without_plasticity`, `change_plasticity`**
- Change: rules and their state are type parameters of `SpikingSynapse`. `with_plasticity(c; LTP, STP)` returns a new
  synapse; a rule left to `nothing` is kept with its state, `NoLTP()` / `NoSTP()` switch that part off, a new rule gets
  fresh state. `without_plasticity(c)` switches both off. `change_plasticity(model, :name; LTP, STP)` returns a new model
  with that synapse replaced (unknown name: `ArgumentError` listing the synapses); `compose(; model..., EE = new)` is a
  duplicate-key error that points to it.
- Shared: `W`, `I`, `colptr`, `rowptr`, `g`, `v_post`, `fireI`, `fireJ`, `param` (delays), `records`, `id`, `name`, the spike
  lists, the chunk cache and a kept rule with its state. `J` / `index` are shared only if the new rules need them and `c`
  has them, else built fresh or left empty; `ρ` is shared with `STP = nothing`. Use only one of the two synapses in a model.
- Consequence: the source synapse is no longer modified (its vectors are never resized or refilled). `STP = NoSTP()` resets
  the efficacy to 1 (to freeze `ρ`, keep the rule and call `set_STP!(c, false)`). Records of a rule variable that the new
  rule lacks are closed (data kept and readable); the source stops sampling them too, since `records` is shared. A
  switched-off synapse has the type, memory and work of a never-plastic one and gives the same run.
- `update_plasticity!` / `change_plasticity!` accept only a rule of the current type (new parameters, fresh state) and throw
  `ArgumentError` otherwise; LTP is checked and written before STP is checked. Assigning `c.LTPParam` of another type is
  a `MethodError`.
- Where: `src/connections/spiking_synapse.jl` (`with_plasticity`, `without_plasticity`, `update_plasticity!`,
  `_close_rule_records!`), `src/connections/sparse_plasticity.jl` (`change_plasticity!`), `src/utils/util.jl`
  (`change_plasticity`, `_dup_msg`).
- Migration: `model = change_plasticity(model, :EE; LTP = STDPGerstner())`, or build the synapse with the rule.

### 1.7 Precision from the population type

**Float64 by `VFT = Vector{Float64}`**
- Change: a population built with `VFT = Vector{Float64}` runs in Float64 with all its defaults (parameters, spike, synapse
  model, receptor buffers, dendrites). Connections and stimuli take the precision of the variable they write; their
  rules are converted with `convert_precision(rule, FT)` (the object itself if already of that type). `float_type(x)`
  returns the precision of any component. The clock is `Time{FT}`; `compose` builds it from the promotion of the population
  types; `sim!` / `train!` pass `dt` to each component in its own precision, so mixed-precision models run.
- Consequence: Float32 results are unchanged. Unit constants and rule defaults are Float32: a Float32 rule on a Float64
  population keeps Float32-rounded values (build it with `FT = Float64`). One precision per population: a Float32 parameter
  object on a Float64 population, or a target whose variable and membrane potential differ, is an `ArgumentError`. Float32
  only: `HetRec` (refuses a Float64 step), `CurrentStimulus`, `BalancedStimulus`, `Turnover` (refuse a Float64 target at
  construction), the rate and FORCE connections. Float64 costs 1.4 to 4.3 times Float32 by model (2.3 on the E/I network).
- Where: `src/utils/structs.jl` (`float_type`, `Time{FT}`, `_require_float32`), `src/connections/connections.jl`
  (`convert_precision`), `src/populations/populations.jl` (`_check_precision`); every population and synapse model;
  `src/utils/util.jl` (`compose`); `src/utils/main.jl` (`_step_dt`).
- Migration: none for Float32 models (`Time()` is `Time{Float32}`).

### 1.8 Flush-to-zero during `sim!` / `train!`

**`flush_subnormals = true` by default**
- Change: the keyword forms of `sim!` / `train!` run the loop inside `with_flush_subnormals`, which sets
  `set_zero_subnormals(true)` on the stepping thread and, with threading on, on every Polyester worker, and restores the
  previous mode afterwards (also on an exception).
- Consequence: values below `floatmin` become 0; a decaying variable stops at a small normal value
  (`|g| <= floatmin τ/dt`) instead of a subnormal one. Spike times were bit-identical in the networks checked. Not affected:
  direct calls of `integrate!` / `forward!` and the five-argument per-step `sim!` / `train!(P, C, S, dt, T)`.
- Where: `src/utils/subnormals.jl`, `src/utils/main.jl` (`_run!`).
- Migration: `flush_subnormals = false` for IEEE subnormals.

### 1.9 Behaviour fixes and stricter checks

- **Dendrite physiology.** `Tripod` and `BallAndStick` build their dendrites with `param.physiology`; before, they always
  used `human_dend`. Only users who passed another physiology (e.g. `mouse_dend`) see a change; defaults are bit-identical.
  Where: `src/populations/multicompartment/tripod.jl`, `ballandstick.jl`, `dendrite.jl` (`create_dendrite`).
- **Short state vectors.** A state or parameter vector shorter than `N` is a `DimensionMismatch` at the start of the step
  instead of an out-of-bounds access (`_check_state_lengths`, `src/populations/generalized_if/if.jl`, used by the
  point-neuron, multicompartment and Poisson populations).
- **Refractory time.** A non-finite `τabs` makes `IF`, `AdEx`, `ExtendedIF` throw at the first step instead of at the first spike.
- **Recorder subsets.** An entry of `indices[:fire]` outside `1:N` is an `ArgumentError` at the first `sim!` / `train!`
  (`_fire_mask!`, `src/utils/record.jl`).
- **`compose` and OrderedCollections 2.0.** `compose`, `population_indices`, `filter_items`, `subpopulations` no longer use
  `sort(::Dict)` (removed in OrderedCollections 2.0); same key order and values (`sorted_ntuple`, `src/utils/util.jl`).

---

## Part 2. Structs, functions and files

### 2.1 Struct layout

Keyword construction is unchanged for every type below; positional constructors and code that names the full type
parameter list must be updated.

| type | change |
|---|---|
| `SpikingSynapse` | type parameters `LTPP`, `STPP`, `LTPV`, `STPV` (rules and state, were abstract fields); new fields `pre_spikes`, `post_spikes` (`SpikeIndex`), `chunkptr` (`ChunkPtr`). `J`, `index` empty unless a rule `needs_transpose` (all except `NoLTP`, `NoSTP`, the Markram rules, both vSTDP rules) or a reader builds them (`presynaptic`, `presynaptic_idxs`, `SynapseNormalization`, `AggregateScaling`, `ensure_transpose!`); `ρ` empty without a short-term rule (ρ = 1; `efficacy`, `materialize_efficacy!`) |
| `SpikingSynapseDelayParameter` | now `mutable`; `spike_time`, `spike_w` removed; fields `delaytime`, `delaysteps`, `buffer`, `dt`, `pos`, `spike_ring`, `drain`; `delaytime` of length 1 for a common delay |
| `SpikeTimeStimulus` | `param` typed by `PT <: SpikeTimeStimulusParameter`; type parameter `TLT`; fields `spiking`, `inbox` |
| `PoissonStimulus` | type parameters `PT <: PoissonStimulusParameter`, `TLT`; field `inbox` |
| `PoissonStimulusLayer` | type parameter `TLT`; fields `spikes` (fed list), `inbox`; `randcache` removed |
| `Identity` | fields `spikes`, `inbox`, `nwriters`, `ndrivers`, `tracked` |
| `IF`, `AdEx`, `ExtendedIF`, `HetRec`, `Tripod`, `BallAndStick`, `HH`, `IZ`, `MorrisLecar` | field `spikes::SpikeIndex` after `fire` |
| `Poisson`, `VariablePoisson`, `InhomogeneousPoisson` | field `spikes`; `InhomogeneousPoisson.pspike`; `VariablePoisson.Erate` is a vector, `re`, `cc` removed; `randcache` used only in the dense regime |
| `STDPVariables`, `STDPTripletVariables`, `iSTDPVariables`, `STDPStructuredVariables` | fields `scale` and `decay` (cached factors) |
| `STDPStructuredVariables` | subtype of `LTPVariables` (was `PlasticityVariables`) |
| `vSTDPVariables` | now the state of the event rule (`u`, `v`, `x` stored amplitudes, `K`, `snap`, ...); the old fields are in `vSTDPDenseVariables` |
| `SpikeIndex` | new type: `idx`, `n`, `valid`, `active`, `fed`, `owned` |
| `Time` | `Time{FT}`: `t::Vector{FT}`, `dt::FT` (was Float32) |

### 2.2 Renames and removals

- `iSTDPTime` removed (it had no `plasticity!` kernel and could not run); use `iSTDPRate` or `iSTDPPotential`.
- Internal: `_decay!(x, f)` is `_rescale!(x, s)`.
- Names of 2.0 development builds that are not in the release (no alias): `vSTDPEventParameter` / `vSTDPEventVariables`
  (now `vSTDPParameter` / `vSTDPVariables`), the `lazy_decay` field of the STDP rules, the exports of `current_weights` /
  `sync_weights!` (kept as non-exported extension points), `_has_fed_list` (now `_has_step_list`).
- The export of `convolve`, which SNNModels never defined, is dropped.
- Deleted source files, none of which was loaded in 1.x: `src/populations/multicompartment/multipod.jl` (Multipod),
  `src/populations/generalized_if/if_CANAHP.jl`, `src/populations/adex/adex_multitimescale.jl` (and `test/pop/multipod.jl`),
  restorable with `git show 72dd815:<path>`; `src/stimuli/variable_inputs.jl` (`VariableInputParameter`),
  `src/connections/sparse_plasticity/CaRule.jl`, `src/connections/sparse_plasticity/dump.jl`, restorable with
  `git show 72dd815:<path>`.

### 2.3 Exported names (`names(SNNModels)`, 1.9.1 against 2.0.0)

- Added: `@chunks`, `ChunkPtr`, `change_plasticity`, `check_loaded_model`, `chunk_ptr!`, `convert_precision`, `float_type`,
  `foreach_chunk`, `foreach_chunk_c`, `invalidate_chunk_ptr!`, `neurons_per_thread`, `plasticity_value`, `release_input!`,
  `set_neurons_per_thread!`, `set_optimal_neurons_per_thread!`, `sync_traces!`, `trace_values`, `vSTDPDenseParameter`,
  `vSTDPDenseVariables`, `with_neurons_per_thread`, `with_plasticity`, `without_plasticity`.
- Removed: `convolve`, `iSTDPTime`.
- New internal extension points (not exported): `SpikeIndex`, `spike_index`, `spike_list!`, `prepare_spikes!`,
  `release_spikes!`, `presynaptic_spikes`, `postsynaptic_spikes`, `attach_driver!`, `fed_spike_index`, `owned_spike_index`,
  `ensure_transpose!`, `has_transpose`, `needs_transpose`, `efficacy`, `materialize_efficacy!`, `lazy_variables`,
  `current_value`, `current_weights`, `sync_weights!`, `with_flush_subnormals`.

### 2.4 Step loop (`src/utils/main.jl`)

- The keyword forms of `sim!` / `train!` share `_run!` (records allocated, `with_neurons_per_thread`,
  `with_flush_subnormals`), `_run_loop!` and `_run_pbar!` (progress bar; the firing-rate `Dict` is built only there).
- One step is `_sim_step!` / `_train_step!`: one function barrier per component (`_sim_component!` /
  `_train_component!`, taking the component and its `param`), then `_retire_all_spikes!(P)`. Order of operations unchanged.
- `dt` reaches the step as a value or in a `Ref`: the keyword forms wrap it once per call; the five-argument
  `sim!` / `train!(P, C, S, dt, T)` (kept) pass it as is when the three vectors are concretely typed and in a `Ref` otherwise
  (`_dtarg`, read by `_dtval`), so that dynamic calls do not box it.
- New keywords of `sim!` / `train!`: `neurons_per_thread = nothing` (keep the current setting), `flush_subnormals = true`.

### 2.5 Threading removed from step paths

The `Threads.@threads` loops of `vSTDPDenseParameter`, `AggregateScaling` and `SynapseNormalization` are serial loops with
the same arithmetic. `vSTDPDenseParameter` and `SynapseNormalization` are bit-identical; `AggregateScaling` differs by one
Float32 ulp in 178 of 960 weights of the check (a `@fastmath @simd` row sum vectorised differently; accepted).
`src/analysis/` keeps its `Threads.@threads`.

### 2.6 Saved models

- Change: `load_model`, `load_data` and `SNNload` pass the loaded object to `check_loaded_model` (exported,
  `src/utils/io.jl`), which searches it recursively for JLD2 placeholders (`ReconstructedMutable`, `ReconstructedStatic`)
  and throws an `ArgumentError` naming the type.
- Consequence: files of 1.9.1 and earlier, and of 2.0 development builds before the owned spike lists, are refused (JLD2
  alone returns placeholders that fail later in `train!`). STDP parameter objects saved in any JLD2 file are refused the
  same way when loaded through these functions. `save_model` / `SNNsave` materialise deferred weights before writing.
- Migration: rebuild the model from its configuration and save it again, or open the file with SNNModels 1.9.1. No JLD2
  upgrade path is provided.

### 2.7 Dependencies

- Removed (13): BenchmarkTools, ColorSchemes, Documenter, IterTools, Parameters, Printf, Profile, Requires, Revise, Roots,
  SGtSNEpi, StaticArrays, TimerOutputs. Added: Polyester 0.7. No code path depended on the removed packages.
- Test environment: SHA added (used by `test/threading_synapses.jl`).

### 2.8 Tests

- New: `test/pop/{_loop_equiv,adex_update_loop,extendedif_update_loop,hh_iz_update_loop,if_update_loop,ml_rate_update_loop,
  multicompartment_update_loop,precision_populations}.jl`, `test/sim/{allocations,subnormals}.jl`,
  `test/stim/{_poisson_stats,poisson_pop_sparse,poisson_sparse}.jl`, `test/syn/{change_plasticity,delay_uniform,lazy_traces,
  lazy_traces_euler,lazy_weights,plasticity_switch,population_spike_list,precision,spike_handoff,spike_index,
  synapse_footprint,vstdp}.jl`, `test/threading.jl`, `test/threading_synapses.jl`, `test/utils/inline_math_test.jl`,
  `test/utils/record_spikeindex_test.jl`.
- Changed: `test/pop/dendrite.jl`, `test/sim/sweep_fixes.jl`, `test/syn/{float32,istdp_kernel,metaplasticity,
  plasticity_params,with_plasticity}.jl`, `test/utils/io_test.jl`, `test/runtests.jl`. Removed: `test/pop/multipod.jl`.

---

## Part 3. Performance

Changes in this part do not alter results unless stated. The spike lists (1.1) also remove most scans of `fire`: one per
population and step for the integrate-and-fire family, instead of one per connection and recorder.

**Vectorised neuron and synapse loops**
- Change: branchless `@inbounds @simd` loops with hoisted invariants for `IF`, `AdEx`, `ExtendedIF`, `HH`, `IZ`
  (three passes merged), `MorrisLecar`, `Rate`, `WilsonCowan`, `Identity`, the Heun stages of `Tripod` and `BallAndStick`,
  and the synapse stages of `ReceptorSynapse` (magnesium block only for NMDA receptors) and `MultiReceptorSynapse`.
  Out-of-line calls that blocked vectorisation are inlined: `exp` (same algorithm as `Base.exp`), `exp64`, `exp256`, `_pow4`
  (`x^4`), `_tanh_inl`, `_cosh_inl` (copies of the `Base` algorithms). The `Tripod` / `BallAndStick` loops have no
  `@fastmath` and use `muladd` in the order the 1.9.1 `@fastmath` loop compiled to on x86-64 with FMA.
- Consequence: bit-identical to 1.9.1 in Float32 (Tripod and BallAndStick on x86-64 with FMA, Julia 1.12.6). Speed-up per
  neuron and step: IF 3.2x (5.0 to 1.56 ns at N = 1e4), AdEx 4.0x, ExtendedIF 4.0x, HH 4.4x, IZ 1.3x, MorrisLecar 3.0-3.8x,
  Rate and WilsonCowan 7.5-9x, Tripod 2.0-2.1x, BallAndStick 2.1-2.3x; synapse stages ReceptorSynapse 4.3-6x,
  MultiReceptorSynapse 8-9x.
- Where: the population files under `src/populations/`, `src/populations/synapse/synapses/ReceptorSynapse.jl`,
  `src/utils/util.jl` (inlined helpers).

**Event-driven and lazy plasticity, sparse Poisson samplers**
- Change: the lazy traces (1.5) and the event-driven `vSTDPParameter` cost O(spikes x fan-out) per step instead of a pass
  over all neurons or synapses; the Poisson samplers (1.4) cost one draw per spike. Trace-decay factors `exp(-dt/τ)` are
  cached in the rule variables (`decay`), and `_rescale!` sets values below `floatmin` to zero, so a silent trace no longer
  runs through the subnormal range.
- Consequence: event vSTDP is 50x faster than the dense rule at 1e4 synapses. `Poisson` at 1 Hz is 10-32x faster per
  neuron and step.

**Delayed transmission**
- Change: one add into the ring buffer per synaptic event instead of a sorted queue insertion; for a common delay the
  arrival column is computed once per step and the scatter runs with `@simd ivdep` (`src/connections/delay_kernels.jl`).
- Consequence: Poisson-driven network (N = 1e4, 500 Hz inputs) 1.854 to 0.947 s per simulated second, together with the
  spike-list delivery of 1.3 and the footprint change below.

**Synapse memory footprint**
- `SpikingSynapse` arrays 29.2 to 9.2 B per synapse in the balanced network without plasticity (lazy `J` / `index` / `ρ`,
  common delay stored once, Int32 storage built without a transposed temporary); peak RSS slope 49.6 to 34.7 B per synapse.

**Recording and step overhead**
- `record!` of an unmonitored object is one inlined `isempty` test; the `:fire` recorder writes the spike list of the step
  instead of visiting every neuron; `_allocate_records!` has a function barrier per component group.
- No allocation per step through the step loop of 2.4; before, 16 B per component and step.
- The five-argument per-step entry `train!(P, C, S, dt, T)` had become 12 to 30 ns per call dearer during development;
  7cb36e7 removed the `try/finally` and the `Ref` for concretely typed vectors. ms per 2e5 steps, 395412d / b25eb27 / 2.0.0:
  pair STDP 200 synapses 14.5 / 17.9 / 15.2, 1e4 synapses 20.3 / 24.8 / 20.0, event vSTDP 1e4 37.7 / 39.1 / 35.5.

**Subnormals and start-up**
- Flush-to-zero (1.8): balanced network N = 500, p = 0.02, 0.0512 to 0.0090 s per simulated second; quiet `Rate` through
  `sim!` 40.8 to 0.86 ns per neuron and step.
- Dependency removal (2.7): `using SNNModels` 2.58 to 2.03 s, resolved packages 190 to 138.

**Opt-in threading**
- `neurons_per_thread` (keyword of `sim!` / `train!`, `set_neurons_per_thread!`, `with_neurons_per_thread`,
  `set_optimal_neurons_per_thread!`) runs the per-neuron stages, and `forward!` and the event-driven STDP, iSTDP and STP
  updates of `SpikingSynapse` (partitioned by postsynaptic range), in chunks of whole 128-neuron batches (at most 8 chunks)
  on Polyester workers (`@batch`). Off by default (0). Chunks depend on `N` and the setting only, so results are
  bit-identical for every setting and thread count. Serial: `DeltaSynapse`, `ExtendedIF`, the Poisson populations,
  `vSTDPParameter`, `iSTDPPotential`, synapses below `SNNModels.SYN_MIN_OPS[]` operations per step. AdEx N = 2e4: 2.21x at
  8 threads. Files: `src/utils/threading.jl`, `src/utils/threading_tuner.jl`.

**Headline: balanced E/I LIF network, K = 100, one pinned core**

Seconds per simulated second, Float32, Julia 1.12.6, Xeon w5-3425 (commit b25eb27; Auryn 0.8.3, Brian2 2.9.0 cpp_standalone
as reported):

| N | 1.9.1 | 2.0 | Auryn | Brian2 |
|---|---|---|---|---|
| 500 | 0.0614 | 0.00697 | 0.00728 | 0.00916 |
| 2000 | 0.283 | 0.0218 | 0.0301 | 0.0350 |
| 10000 | 1.79 | 0.109 | 0.166 | 0.169 |
| 20000 | - | 0.235 | 0.359 | 0.313 |
| 50000 | - | 0.684 | 1.048 | 0.961 |

On one core JuliaSNN 2.0 is at the level of Auryn and Brian2 in this benchmark. Recording all spikes costs 1.35x at
N = 500 and 1.06x at N = 1e4. Peak RSS of the Julia process is about 800 MB at every N (runtime and compiled code).

---

## Known limitations

- Per-step dense work remains in `iSTDPPotential`, `vSTDPDenseParameter`, the postsynaptic filters of `vSTDPParameter`,
  `MarkramSTPParameterTimestep`, `ActivityDependentTurnover`, `AggregateScaling`; `SpikeRateSynapse` scans `fire`.
- `fire` is still scanned once per connection for the Poisson populations in the dense regime and for an `Identity` with
  a writer that does not log.
- A record closed by a plasticity switch is not reopened by a later switch (`clear_records!` / `clear_monitor!` first).
- In-flight buffer increments are not rescaled when `dt` changes without a change of the buffer size.
- The Float32 clock `T.t[1] += dt` drifts: `SpikeTimeStimulus` compares spike times with it, so at `dt` not exactly
  representable a spike can be delivered one step off.
- Bit identity of `Tripod` / `BallAndStick` to 1.9.1 depends on the LLVM version and holds on x86-64 with FMA; aarch64 is
  not verified. `_tanh_inl` / `_cosh_inl` rely on internals of `Base.Math`; a Julia version that changes them fails
  `test/utils/inline_math_test.jl`.
- The first `train!` compiles one specialisation per combination of rule types (0.36 s against 0.08 s for a six-synapse
  E/I network in a warm session).

## Verification

- Suites at 2.0.0 (Julia 1.12.6): SNNModels 77273 pass, 0 fail; SNNUtils 17, SNNPlots 14, SpikingNeuralNetworks 3,
  OptunaLoader 15 pass; CalciumSpike has no tests.
- Bit identity to 1.9.1 (Float32): every converted neuron and synapse loop against a copy of the old loop
  (`test/pop/*_update_loop.jl`, `test/utils/inline_math_test.jl`); the `Tripod` / `BallAndStick` bitwise test runs only
  without `--check-bounds=yes`. A single LIF neuron gives byte-identical spikes and `v` traces.
- Evidence for the bit-identity statements within 2.0: owned spike lists against per-connection scans (state and raster
  hashes, 40 runs at `-t 1` and `-t 8`); Float32 state hashes before and after the Float64 work (36 cases);
  `test/threading.jl`, `test/threading_synapses.jl`, `test/syn/plasticity_switch.jl`.
- Final STDP weight extrema agree within 1e-6 between 1.9.1, 2.0 Float32, 2.0 Float64 and Brian2.
