# SNNModels changelog

## 2.0.2

Compact storage of perturbation records (same change as 1.9.2).

- `perturbation_test(...; add_records = cond)` stored every window as its own `Dict("interval", "data")` under
  `records[:perturbation][var][cond][n]`. With 1,320 presentations x 13 conditions x several variables that is 17,160
  small objects per population and variable, JLD2 files of 35-41 GB and a load that peaked at 83 GB (SNN2AC_optim).
  The windows of one `(variable, condition)` now go to one ragged store: `PerturbationStore` (dense: `N x sum(T)`
  Float32 kept as a flat vector, CSR `offsets` with `length + 1` entries, `tstart`, `tend`, `ids`, `next_id`) or
  `PerturbationSpikeStore` (`:fire`: flat `times` and `neuron`, CSR `offsets`, `tstart`, `tend`, `ids`, `next_id`).
  Both are mutable structs with a `version` field (1) and the same fields in 1.9.2 and 2.0.2. See the docstrings.
- `perturbation_record` returns bit-identical values to the old layout (same window order, clipping, interpolation).
- New exports: `perturbation_windows(obj, var, cond)` (named tuples `(tstart, tend, data)`, data a read-only view,
  insertion order), `compact_perturbation_records!(model_or_obj)`, `PerturbationStore`, `PerturbationSpikeStore`.
  Code that read `entry["interval"]` / `entry["data"]` should call `perturbation_windows`; `store[n]` still returns
  such a dictionary (a copy).
- The old and the new layout never coexist in a working model. Records of the old layout (Dict per window) are not
  read, appended to or edited window by window: a flush, `perturbation_windows`, `perturbation_record` or
  `clear_perturbation_records!(...; n)` that meets one throws an `ArgumentError` naming object, variable and
  condition ("... are in the pre-1.9.2 layout (one Dict per window); convert them once with
  compact_perturbation_records! (then save the model again) ..."). Clearing a whole condition, variable or all
  records still works on old records. Nothing is converted automatically.
- `compact_perturbation_records!(model_or_obj)` is the required one-time migration of existing files: in place, one
  object and one variable at a time with `GC.gc()` between, each store checked bit by bit against the old
  dictionaries before they are dropped, idempotent, returns windows and bytes converted per variable. The old keys
  `n` (gaps allowed) become the `ids`, `next_id` becomes `maximum(n) + 1`; conditions it cannot store (3-D data,
  differing N) throw an `ArgumentError` naming object, variable, condition and `n`. Save the model again afterwards.
- Dense windows with more than two dimensions (matrix-valued variables, size `(N, D, T)`) are refused by the flush
  with an `ArgumentError` naming object, variable, condition and the number of dimensions. They could never be read
  before either (the old reader asserted `Matrix{Float32}`).
- Bug fix: window numbers were `length(dict) + 1`, so after `clear_perturbation_records!(...; n = 2)` of `{1,2,3}`
  the next flush overwrote window 3. Window numbers (`ids`) are now never reused, not even after every window was
  deleted (the empty store stays and keeps its `next_id`); `n` in `clear_*` is that number. Deleting window `n`
  physically removes its columns (dense) or spikes (`:fire`) and shifts the later `offsets`.
- Compatibility: files holding compact records need SNNModels >= 2.0.2 in every process that reads them
  (`perturbation_record` of 2.0.1 and earlier cannot read them). Stores carry a `version` field (1); readers and
  writers throw an `ArgumentError` for another value. Objects with 0 rows and windows with 0 columns work.
- A `:data` file written by 1.9.2 is still refused by `load_data` in 2.0 (its populations use the 1.9 layout); a store
  loaded without SNNModels makes JLD2 reconstruct it with the arrays accessible by field name.
- Not changed: a window of one sample (`T = 1`, e.g. `perturbation_test(model, 1ms)` at 1 kHz) makes
  `perturbation_record` throw an `ArgumentError` from Interpolations, as before.

## 2.0.1

Bug-fix release; STTC values change.

- `STTC` (#86): both trains are restricted to `[t0, t1]` before computing P and T, and T is the length of the union of the
  `[s-Δt, s+Δt]` windows clipped to `[t0, t1]` divided by `t1 - t0` (Cutts & Eglen 2014), without the `2Δt` extension.
  With 800 independent Poisson trains recorded for 6 s and `interval = 3s:5s`, `Δt = 50ms`, 2.0.0 gave |STTC| > 1 for 1-2 %
  of the pairs at 25 Hz and about 25 % at 50 Hz, with -Inf and NaN matrix means; now all values are finite and in [-1, 1].
  Independent 50 Hz trains over 10 s score 0.18 instead of 0.34; at 25 Hz over 60 s the mean is 0.002. A term with
  `P T = 1` is 1 (Elephant convention), so a train against itself gives 1 instead of NaN; trains without spikes in the
  interval give 0. STTC saturates when `Δt` times the rate is about 1 or more.
- `monitor!`: `monitor_rate` carries units like the other rates (`monitor_rate = 50Hz` sizes the spike buffer for 50 Hz;
  in 2.0.0 it sized it for 0.05 Hz). A plain number (above `1kHz` in library units) is an `ArgumentError`.

No other change.

## 2.0.0

Major release with breaking changes in the types of synapses, populations and stimuli, in random streams and in how
plasticity state is read. Packages that depend on SNNModels must move to 2: SNNUtils 1, SNNPlots 1,
SpikingNeuralNetworks 2, CalciumSpike 1. Models saved with 1.9.1 or earlier are refused at load time. Every change, its
consequences, the files it touches and its migration: `CHANGELOG_2.0.0.md`.

### Semantics
- Spikes move as index lists (`SpikeIndex`): scratch per connection, fed by `Identity` and the Poisson populations, and one
  list per integrate-and-fire population, opened after `integrate!` and closed at the end of the step. Custom components
  must not write `pop.fire` inside that window. A failed step is fatal: nothing closes the window after an exception
  (debugging: `SNNModels._retire_all_spikes!(P)`).
- Stimuli log the `Identity` neurons they write, so a stimulus-driven `Identity` updates sparsely (`release_input!`).
- Delays use a ring buffer and are rounded up to whole steps (`ceil(delay / dt)`); exact at `dt` such as 0.1 ms, where
  1.9.1 was often one step late. A common delay is stored once; without plasticity it delays the spike list, so `W` is
  read at arrival.
- Poisson stimuli and populations sample one random number per spike: same laws, different random streams.
- Plasticity state is stored lazily: direct reads of traces return stored values; use `plasticity_value`,
  `trace_values`, `sync_traces!`. Recorded `u`, `x` of the event Markram rules are now values at the sampling time.
- `vSTDPParameter` is event-driven with deferred weights (read `plasticity_value(syn, :W)`); the 1.9.1 rule is
  `vSTDPDenseParameter`. STDP weights differ from 1.9.1 by Float32 rounding.
- Plasticity rules are type parameters: `with_plasticity`, `without_plasticity`, `change_plasticity` install a rule of
  another type; `update_plasticity!` / `change_plasticity!` accept only the same type. `NoSTP()` resets the efficacy to 1.
- Precision follows the population type (`VFT = Vector{Float64}`); `float_type`, `convert_precision`, `Time{FT}`.
- `sim!` / `train!` flush subnormals to zero (`flush_subnormals = true`).
- Fixes: `Tripod` / `BallAndStick` use `param.physiology`; short state vectors raise `DimensionMismatch`; an out-of-range
  `indices[:fire]` is an `ArgumentError`; `compose` works with OrderedCollections 2.0.

### Structs, functions and files
- New type parameters and fields in `SpikingSynapse` (`J`, `index`, `ρ` empty until needed: `ensure_transpose!`,
  `efficacy`, `materialize_efficacy!`), `SpikingSynapseDelayParameter`, `SpikeTimeStimulus`, `PoissonStimulus`,
  `PoissonStimulusLayer`, `Identity`, the nine integrate-and-fire populations and the Poisson populations. Keyword
  construction is unchanged.
- Removed: `iSTDPTime` (could not run), the export of `convolve` (never defined), and files not loaded since 1.x
  (Multipod, IF_CANAHP, AdExMultiTimescale, `variable_inputs.jl`, `CaRule.jl`, `dump.jl`).
- New exports: `with_plasticity`, `without_plasticity`, `change_plasticity`, `plasticity_value`, `trace_values`,
  `sync_traces!`, `vSTDPDenseParameter`, `vSTDPDenseVariables`, `float_type`, `convert_precision`, `release_input!`,
  `check_loaded_model`, threading controls (`neurons_per_thread` and related, `foreach_chunk`, `ChunkPtr`).
- `load_model`, `load_data`, `SNNload` refuse files written with an older layout (`check_loaded_model`).
- 13 unused dependencies removed (Documenter among them); Polyester added.

### Performance (results unchanged unless stated above)
- Vectorised, branchless neuron and synapse loops, bit-identical to 1.9.1 in Float32: 1.3x to 9x per model.
- Event-driven and lazy plasticity at O(spikes); event vSTDP 50x faster than the dense rule at 1e4 synapses.
- Synapse arrays 29.2 to 9.2 bytes per synapse; no allocation per step; spike recording from the spike lists.
- Opt-in thread parallelism of the population updates (`neurons_per_thread`, off by default, bit-identical).
- Balanced E/I LIF network (K = 100, one core): 1.79 to 0.109 s per simulated second at N = 1e4, at the level of Auryn
  and Brian2 (cpp_standalone) in this benchmark. Peak RSS of the Julia process is about 800 MB.
