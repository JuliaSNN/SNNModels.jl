# SNNModels changelog

## 2.1.0

Tripod / BallAndStick: soma-clamp fix (RESULTS CHANGE), dendrite diameter, fixed paper configurations, mouse synapses, and
`TripodPaper`, the configuration that reproduces the published Tripod code.

**Results change.** During the somatic back-propagating spike (`up`) and the absolute refractory period (`τabs`) the dendrites of
`Tripod` and `BallAndStick` only relaxed axially towards the clamped soma (forward Euler), without leak and without any synaptic
current, NMDA included. One somatic spike could therefore end an NMDA plateau. Both periods now integrate the full dendritic
derivative (leak, all synaptic currents, axial current to the clamped soma, external current) with the same Heun scheme as free
integration; the soma is held at its clamp value (`AP_membrane` during the back-propagation, `adex.Vr` during the refractory
period) in both stages, `w_s` stays frozen, and the dendrites still hold on the spike step itself. Free integration is unchanged
(same arithmetic, same speed, bit-identical to the reference loops of the tests). Measured effects:
  - paper Fig. 8 protocol (one NMDA event of weight 100 on d1, both dendrites 250 um, original parameters): plateau (time with
    V_d1 >= -60 mV) 28.7 ms in 2.0.3, 54.2 ms in 2.1.0 (the original code gives 58.4 ms with its own exp32 Mg block); 400 um: 67.2
    ms unchanged (no spike); 150 um: 13.2 -> 13.7 ms.
  - default `Tripod(N = 200)` under Poisson noise on d1, d2, soma (5 s, two seeds, `rate_net.jl` of the audit): mean rate 25.7 ->
    26.1 Hz at the intermediate drive (+1.5 %), 54.7 -> 55.8 Hz at the strongest (+2.1 %); default `BallAndStick`: changes below
    0.3 %. The effect grows with the NMDA drive and with the length of the clamped window (`up + τabs`, 2 ms by default).
  Results of Tripod/BallAndStick simulations with spikes are not reproducible bit for bit across 2.0.x and 2.1.0.

**Results change (GABA defaults).** The default somatic and dendritic GABA_A of Tripod/BallAndStick changed to the original Tripod
code (`aquaresima/tripod_neuron` at 8698e1a, `src/TripodNeuron.jl/src/model/synapses.jl:119` and `:124`): E_rev -70 -> -75 mV (soma
and dendrite), somatic τr 0.1 -> 0.3 ms; results change for models using the default synapses (`MilesGabaSoma`, `MilesGabaDend`,
`TripodSomaSynapse`, `TripodDendSynapse`, `TripodDendSynapseMouse`). Unchanged: τd and g0 of both, dendritic GABA_A τr 4.8 / τd 29 /
g0 0.27, dendritic GABA_B (-90 mV, 30, 400, 0.006), all NMDA and the dendritic AMPA. The somatic AMPA default (`DuarteGluSoma`) also follows the
original: τr 0.26 -> 0.25 ms. `TripodPaper` is unchanged (`PaperGabaSoma`, `PaperGabaDend` are now aliases of the defaults).
Standard network (`rate_net.jl`, seeds 7 / 8, soma-clamp-fixed 2.1.0 before -> with the GABA change): Tripod mu_e 0.15: 0.046 / 0.079 ->
0 / 0 Hz; 0.2: 26.34 / 25.81 -> 23.78 / 23.19 Hz (-10 %); 0.3: 55.75 / 55.91 -> 55.31 / 55.43 Hz (-0.8 %); BallAndStick 0.3: 13.90 /
14.19 -> 14.05 / 14.34; 0.4: 22.72 / 22.92 -> 23.10 / 23.32; 0.6: 34.99 / 35.09 -> 35.41 / 35.49 Hz (about +1 %). Tests that encoded the old
values: `precision_populations.jl` (receptor table of the default synapses).

Other changes (no change of results unless stated):
- Dendrite diameter: `d` (default `4um`) in `DendNeuronParameter`, `TripodParameter`, `BallAndStickParameter`, passed to
  `create_dendrite` (`g_ax` ~ d^2, `g_m` ~ d, `C` ~ d). The default reproduces 2.0 exactly.
- Fixed configurations of the paper as `ds` presets: `distal_distal` (400, 400 um), `proximal_proximal` (150, 150), `proximal_distal`
  (150, 400), `distal_proximal` (400, 150: the default of the published code, whose protocols stimulate d1), `soma_only`
  (both dendrites disconnected). **Renamed presets (behaviour change for the old names):** `proximal_distal` (random 150-400 um) is now
  `random_proximal_distal`, `proximal_proximal` (random 150-300 um) is now `random_proximal_proximal`; the old names give the fixed
  lengths above. `proximal` and `all_lengths` are unchanged.
- `create_dendrite`: the 500 um length bound (added without a documented reason in the 2025 import, probably a guard against a length
  given without unit) is replaced by a guard at 1000 um with a message about units; paper Fig. 3 uses 550 um dendrites.
- Mouse synapses: `DuarteGluDend` (dendritic AMPA E 0, tau_r 0.25, tau_d 2 ms, g0 0.73 nS; NMDA tau_r 0.99, tau_d 100 ms, g0 0.159 nS,
  as `DuarteGluDend` of the published code), `TripodDendSynapseMouse` (selected with `dend_syn =`; the default `dend_syn` of `Tripod`
  is unchanged), `MouseNMDA` (b 3.57, k -0.062, the mouse values printed in the paper; the published code does not use them and runs the
  mouse with the human block 3.36 / -0.077, which `TripodDendSynapseMouse` follows).
- NMDA block option `NMDAVoltageDependency(approx = 256)` (default unchanged): `32` is the `exp32` of the published code (block
  0.0225 at -70 mV instead of 0.0160), `64`, or `0` for the exact exponential; `EyalNMDA32` is predefined. Receptor decay option
  `ReceptorSynapse(decay = 64)` (default unchanged): `32`, `256`, `0`. New internal `exp32o` (inputs below -10 give exactly 0 as in
  the original).
- `TripodPaper(; N, physiology = :human | :mouse, config = :distal_proximal, d, VFT, kwargs...)`: a `Tripod` with the parameters of
  `aquaresima/tripod_neuron` at 8698e1a written out in its docstring (AdEx 281 pF / 40 nS / -70.6 / 2 / -50.4 / 4 / 80.5 / 144; spike
  test `V_s >= -50.4 mV` on the potential at the start of the step; 21 mV spike step, 20 mV for 9 steps, 21 refractory steps
  (`up = 1 ms`, `τabs = 2.1 ms` at dt 0.1 ms); initial state at E_L; GABAa -75 mV, soma AMPA tau_r 0.25 / GABAa 0.3 ms; exp32 Mg
  block and synaptic decay; dendritic synaptic current clamped to +-1500 pA). The integration is the split scheme of that code
  (`scheme = PaperScheme()`, new `Tripod` field, default `HeunScheme()`): external current as a kick, axial currents frozen over
  the step, dendrites two-stage with frozen axial current, forward-Euler `w`, soma two-stage with the new `w`, dendrites integrated
  against the stored soma potential during the clamp. Use `dt = 0.1ms` (a warning is issued once for another step). Predefined
  receptor sets of the original: `TripodSomaSynapsePaper`, `TripodDendSynapsePaper`, `TripodDendSynapsePaperMouse`. Against the original
  code on the audit protocols (dt 0.1 ms, identical inputs): subthreshold step and EPSP max |dV| < 1e-3 mV, 1200 pA step 15/15 spikes
  at identical times, frozen Poisson input 35/35 spikes at identical times, Fig. 8 plateaus 71.1 / 58.4 / 28.7 ms (400 / 250 / 150 um)
  identical to the original. Not implemented: `soma_only` routing of dendritic inputs to the soma, `ampa_equivalent`, `BAP_gax`
  (default 1 in the original, 2 only in a revision-figure script).
- Documentation fixes: `ReceptorSynapse`/`MultiReceptorSynapse` (the decay is the `exp64` approximation, not exact; the current uses the
  stage potentials of the Heun method in `Tripod`/`BallAndStick`), `Tripod` (Eq. 1 of the preprint has no g_L on the exponential
  term while the code and the model have it; `Vspike = adex.Vt` does not reproduce the original spike rule), references of the cable
  constants (Eyal et al. 2016, Dasika et al. 2007), of the receptors (Eyal et al. 2018, Miles et al. 1996, Duarte and Morrison) as given
  in the original code.
- Tests: `test/pop/tripod_21.jl` (all of the above, with small reference traces of the original code in `test/data`); the TripodPaper
  case in `threading_synapses.jl`; the reference loops of `multicompartment_update_loop.jl` follow the new clamped period (they were
  the bug-for-bug reference of the axial-only Euler step).
- Validation: `papers/JuliaSNN_publication/validation/tripod_audit` of the JuliaSNN repository (audit, `tripodpaper_check.jl`).
- Test suite: 78323 pass, 1 broken (the `@test_skip` of 2.0.3; 78092 + 1 before: +181 in `tripod_21.jl`, +50 TripodPaper cases in the threading test).

**New option `El_dend`.** `DendNeuronParameter`, `TripodParameter` and `BallAndStickParameter` take `El_dend = nothing`: the leak
reversal potential of the dendrites. `nothing` (default) keeps the behaviour of 1.9.0 and later (dendrites use the somatic `adex.El`,
bit-identical results); a number (mV) gives `create_dendrite(...; El = El_dend)`, so optimisation projects that sample the somatic `El`
can keep the dendritic leak fixed (the original Tripod fixes it at -70.6 mV). The value is read when the population is constructed
(`d1.El`, `d2.El`, `d.El`); changing `adex.El` of an existing population (e.g. `@update!`) moves the somatic leak and the adaptation
coupling `a (V_s - El)` but not the dendritic leak, with or without `El_dend`. `TripodPaper` now passes `El_dend = -70.6mV`, so its
dendritic leak stays at -70.6 mV even if the somatic `El` is changed through `kwargs`.

**Results change (default spike event, all neuron types).** The defaults of `PostSpike` changed: `τabs` 1 -> 2 ms, `τA` 10 -> 30 ms,
`AP_membrane` 10 -> 20 mV (`At = 0`, `up = 1 ms` unchanged), and the default `Vspike` of `Tripod` and `BallAndStick` (predicted-potential spike
detection threshold) changed from -10 to 0 mV, so the spike event is the same everywhere: detection at 0 mV, 20 mV clamp for 1 ms, 2 ms
absolute refractory period. Affects every `IF`, `AdEx`, `Tripod` and `BallAndStick` that relies on the default `PostSpike()` (the refractory
period of IF/AdEx doubles; `τA` matters only for `AdEx`/`Tripod`/`BallAndStick` with `At != 0`; `AP_membrane` and `Vspike` only for the
multicompartment models). The predictive rule is unchanged (`V + dt k2 >= Vspike`); with 0 mV it fires at most one step later than with -10 mV, and
no overflow of the exponential term was found before detection (strongly driven Tripod/BallAndStick, Float32 and Float64, all potentials
finite). `TripodPaper` is unchanged (own `PostSpike`, `Vspike = -50.4`, `PaperScheme`). Code that passes `τabs` explicitly (the published benchmark
harness) is unaffected. Measured, 2.1.0 before -> after (200 neurons, 5 s after 0.5 s, seeds 7 / 8):
  - `Tripod` (`rate_net.jl`, Poisson on d1, d2, soma) mu_e 0.2: 23.79 / 23.20 -> 23.04 / 22.48 Hz; mu_e 0.3: 55.32 / 55.44 -> 52.21 / 52.32 Hz.
  - `BallAndStick` mu_e 0.3: 14.06 / 14.37 -> 13.78 / 14.08 Hz.
  - `IF` (default `IFParameter`, 500 Poisson 15 Hz + 200 Poisson 8 Hz inputs, p = 0.2, mu 3 / 3 on ge / gi): rate 314.5 / 315.1 -> 237.6 / 238.1 Hz;
    mean ISI 3.18 -> 4.21 ms, CV 0.196 -> 0.152.
  - `AdEx` (default `AdExParameter`, same inputs on he / hi, mu 6 / 3): rate 102.9 / 103.1 -> 92.3 / 92.5 Hz; mean ISI 9.71 -> 10.83 ms, CV 0.638 -> 0.571.

## 2.0.3

Smaller `:fire` records, one allocation for a declared duration, provenance from outside the repository.

- `:fire` COO record: neuron ids are stored as `Int32` (`records[:fire][:neurons_buf]`), 8 instead of 12 bytes per
  spike. Spike times, `spiketimes`, `firing_rate` and `record(obj, :fire)` are bit-identical to 2.0.2 for the same
  simulation (checked on a seeded 500-neuron E/I network, 154,824 spikes, against the 2.0.2 code). `_allocate_fire!`
  throws an `ArgumentError` if `N > typemax(Int32)`.
- Files saved with 2.0.2 (`Int64` ids) load unchanged and are read as they are (the readers index with any integer
  type). The next `sim!`/`train!` (or a buffer growth) narrows the loaded buffer to `Int32` before writing to it, so
  a loaded model can continue recording. Nothing is converted at load time.
- `monitor!(...; monitor_time = T)` now sizes the `:fire` buffer at the first allocation, like the dense records:
  `monitor_rate * N * T` spike slots plus a 10 % margin (`monitor_rate` default 20 Hz). In 2.0.2 `:fire` ignored
  `monitor_time` and was sized from the first `sim!` call. For both `:fire` and dense records a later chunk now
  only needs room for its own duration, so a run split into many `sim!`/`train!` chunks allocates once (2.0.2: dense
  records with `monitor_time` extended at every chunk, as less than the whole phase was left). Geometric growth
  with its one warning remains the fallback when no `monitor_time` is given or the estimate is exceeded.
  `clear_records!` starts a new phase.
- `clear_monitor!(::NamedTuple)` skips `Time{Float32}` (`obj isa Time`), no more "Could not clear monitor for Time".
- `get_git_commit_hash(; dir = pwd(), warn = true)` runs git in `dir`. `write_config` (so `save_model`/`SNNsave`)
  records the commit of `commit_dir` if given, else of the active project directory, else of the working directory,
  else `"unknown"` with the warning; processes started outside the repository no longer lose provenance.

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
