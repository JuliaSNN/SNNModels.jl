# SNNModels changelog (branch notes)

## 2.1.0: Tripod soma-clamp fix, diameter, configurations, mouse synapses, TripodPaper

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

## 2.0.3: smaller :fire records, monitor_time for :fire, git provenance

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

## 2.0.2: compact perturbation records
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

## 2.0.0 (was 1.10 during development): sparse spike lists, ring-buffer delays, always-lazy trace decay
Faster event-driven STDP and delayed transmission: fewer scans of the spike flags, type-stable
plasticity calls, no per-step passes over the neurons when a `SpikeTimeStimulus` drives the network.
Weights of the event-driven STDP rules are no longer bit-identical to 1.9.1 (lazy trace decay, differences up
to about 1e-5 relative, see "Behaviour notes"); the other models checked are, except delayed transmission at a
`dt` that is not exactly representable in Float32 and constant delays without plasticity (rounding of the per-step sum, see "Behaviour notes"). Breaking changes in the types (listed with their migration below);
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
  in the network step, 240 B with a `PoissonStimulus`; removed later, see "No allocation per `sim!`/`train!` step").
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
      receptors, and `update_receptor!` of `MultiReceptorSynapse` has `exp64` of the loop-invariant argument hoisted. The two Heun-stage loops
      (`update_neuron!`) hold the axial currents in registers instead of storing and reloading them through `p.ic` (which kept them scalar) and
      have no `@fastmath`; the arithmetic is written with `muladd` in the order the `@fastmath` loop of 1.9.1 compiled to on x86-64 with FMA
      (soma `muladd(gl, muladd(ΔT, e, El - vs), I - (((ws + is) + ic1) + ic2)) / C`, `ws = muladd(Δv4, dt, w)`, dendrites
      `(muladd(El - vd, gm, -is) + ic + Id) / Cd`, adaptation `muladd(a, vs - El, -ws) / τw`). They vectorise, and without fast-math LLVM
      keeps that order in the vector body and in the scalar remainder, so a neuron gets the same bits for every chunking and thread count
      (`threading_synapses.jl`, N up to 4097, 1, 4 and 8 threads). On x86-64 with FMA the result is bit-identical to 1.9.1 (verified with
      Julia 1.12.6; the identity also depends on the LLVM version, which may compile the `@fastmath` reference of 1.9.1 differently):
      `test/pop/multicompartment_update_loop.jl` compares both Heun stages and the whole step with the `@fastmath` reference bit for bit there
      (elsewhere within 4 ulp of the largest term of the derivative); the driven single neurons of `pre110_fixes/scripts/j2_single_neuron.jl`
      (Tripod and BallAndStick, N = 1 and 100, subthreshold, spiking and synaptic kicks, 500 ms) give byte-identical traces and spike steps, and
      the `Tripod` network of `tripod_net.jl` (800 + 200 neurons, 5 s, seed 1) identical spike trains. Bit identity on aarch64 (Apple M-series)
      is not verified: there LLVM may contract the `@fastmath` reference differently. `p.ic` is no longer written (field kept) and the
      threaded path no longer grows it. ns per neuron and step of the final `muladd` build (`pre110_fixes/out/j2r_t5.csv`, round r1: the
      arms before J2, first J2 build and final build run one after the other on one pinned core, median of 3 after the warm-up, N = 1000 / 10000,
      spiking and quiet): `Tripod` 22.2-23.3 -> 10.5-11.6 (2.0-2.1x), `BallAndStick` 15.0-15.5 -> 6.7-7.3 (2.1-2.3x).
      Note on the first attempt (same loop in source order, IEEE, no `muladd`): it was as fast but not bit-identical (derivatives within 3 ulp
      of the largest term, driven potentials up to 0.23 mV apart before spikes with the same spike steps, network statistics within the
      seed-to-seed spread); the failure analysis is in `pre110_fixes/README.md` ("J2 failure analysis"). With bit identity the per-case
      tolerances and nulls of that comparison (BallAndStick kick case at 2.2e-4 mV, the kick-case nulls) no longer apply.
    - `HetRec` is unchanged (273 ns per neuron and step at N = 1000: a gather over about `N overlap Nd` weights per neuron, 0.27 ns per weight, and
      the dense `rand!` is a small part). `IF_CANAHP`, `AdExMultiTimescale` and `Multipod` were not included in the module and were not touched
      (their files were deleted before the release, a99c647). The `@fastmath` synapse stages of `SingleExpSynapse` and `DeltaSynapse` are already vectorised by LLVM (checked in the IR).
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
  - Constant delays deliver through one ring-buffer column per step (`delay_kernels.jl`).
    Measured first (`validation/profilation_v1.10/perf/pre110_fixes`, `j1_fwd_micro.jl`: the T8 network, Poisson input at 500 Hz,
    2000 recorded steps replayed on the two delayed input connections, pinned core, median of 3; no hardware counters on the
    machine, `perf_event_paranoid = 4`, so the cost is decomposed by variants). Hypothesis "two touches of the target array"
    (buffer write, then arrival pass): REJECTED. At N = 10000, ns per synaptic event: whole `forward!` 2.770, of which the flag
    scan 0.267 and transmission 2.503; the same events scattered without delay into `g` 1.535, so the delay costs 0.968, of which
    the arrival pass explains 0.027 (1.0 % of `forward!`, 1.37 us per step for the 1e4 targets) and the extra read-modify-write of the
    buffer 0.032 (6 % together); the dominant term is the per-synapse `delaysteps` load, slot wrap and 2-D index, 0.827 (85 %;
    N = 2000: 0.541 of 0.700, 77 %). Change: when the delay is stored once for all synapses (`Dirac` delays or equal draws, see "Synapse memory
    footprint", step 4: `forward!` then passes it as an integer and `_forward_delay!` dispatches on it; per-synapse delays that only
    round to the same number of steps keep the general path), the arrival column is computed once per step and the scatter `buffer[col * Npost + I[s]] += W[s] * ρ[s]` runs with
    `@simd ivdep` (the rows of one CSC column are distinct; 16 % faster than without `ivdep` in the replay), serial and per chunk of
    the threaded path. Transmission 2.503 -> 1.365 ns per event at N = 10000 (`forward!` 138.4 -> 82.0 us per step), 1.628 -> 0.937
    at N = 2000. T8 (whole network, s per simulated second, 500 Hz) N = 10000 1.854 -> 1.232, N = 2000 0.261 -> 0.187; Auryn measured
    the same day 1.266 and 0.202, so 1.46x and 1.29x slower before, 0.97x and 0.93x after. 50 Hz N = 10000 0.429 -> 0.308; at 1 Hz
    the arrival pass is 55 % of `forward!` and this change gives nothing (see the next item). Bit-identical: test
    `test/syn/delay_uniform.jl` (every step against the general path, Float32 and Float64, delays 0, dt, 0.8 at dt 0.1 and 0.125,
    STDP and STP, a change of dt) and `threading_synapses.jl` (1, 4, 8 threads); spike rasters and final `v`, `g`, buffers SHA1-identical
    to the previous commit for T8 (N = 2000, 500 Hz), T4 (delay 0.8), T7 (delay 5 ms) and the Float64 DoubleExp network. This SHA1
    identity is that of the column path (the delays of these networks have no plasticity, so the next item now sends them through the
    spike list, which is not bit-identical). The numbers of this item and of the next were measured before "Synapse memory footprint";
    on top of it T8 N = 10000 500 Hz is 0.947 s per simulated second (last item of this group).
  - Constant delays without plasticity (`NoLTP` and `NoSTP`, a delay of `d` steps stored once) delay the spike list instead of the
    increments, so on this path the weight `W` is read at arrival, not at emission (`param.spike_ring`: the lists of the last `d + 1`
    steps, capacity kept, 0 B per step after warm-up): the list emitted `d`
    steps ago is scattered straight into `g` (one touch, `@simd ivdep`), with no arrival pass over the targets; the buffer is not
    allocated on this path (0 rows; its `Npost` rows are created when the buffer path runs, e.g. after `with_plasticity`). Adopted by
    the rule of the plan: after the previous item the arrival pass was still 55 % of `forward!` at 1 Hz (1.36 of 2.49 us per step at
    N = 10000). Replay at N = 10000: 1 Hz `forward!` 2.49 -> 1.07 us per step (what is left is the scan of the 1e4 spike flags, 1.0 us),
    500 Hz transmission 1.365 -> 1.301 ns per event. Whole networks, s per simulated second, base -> constant column -> spike list:
    T8 N = 10000 500 Hz 1.854 -> 1.232 -> 1.084 (0.86x Auryn, 1.266-1.308 the same day), 50 Hz 0.429 -> 0.308 -> 0.276, 1 Hz 0.214 ->
    0.192 -> 0.158 (Auryn 1.29 / 0.305 / 0.178 in the campaign); N = 2000 500 Hz 0.261 -> 0.187 -> 0.171 (0.85x Auryn); T4 (k100,
    N = 10000, delay 0.8 on the four recurrent connections) 0.159 -> 0.156 -> 0.125; T7 delay 1 ms 0.174 -> 0.151 -> 0.126, 10 ms 0.185 ->
    0.169 -> 0.129 (the recurrent connections at a few Hz were dominated by their arrival passes). Not bit-identical: the increments of
    one step reach `g[i]` one by one (`g + w1 + w2` instead of `g + (w1 + w2)`) and `W` is read at arrival (see Behaviour notes).
    Acceptance on T8 N = 2000 at 500 Hz against the buffer path from the same state and the same Poisson draws: over 100 ms max |dv| =
    1.5e-5 mV (3.3e-7 of max |v - El|; first difference at step 353, no spike differs), max |dg| 1.2e-4 (1e-7 of max |g|); over 2 s
    rates 6.587 vs 6.638 Hz (E, 0.8 %) and 6.506 vs 6.513 Hz (I), CV-ISI 0.951 vs 0.944 and 0.958 vs 0.971. Same spike counts as before in
    the four hash configurations above (rasters differ). Synapses with plasticity keep the bit-identical column path of the previous item.
  - On top of "Synapse memory footprint" (both items above were measured on the tree before it): its common delay stored once already
    removes the per-synapse delay load from the general loop, so most of the gain of the constant column now comes from that change.
    T8 N = 10000 500 Hz, same day, pinned, median of 3: 1.088 s per simulated second with the footprint change alone, 0.947 with the
    two items above as well (13 % less). With delays stored once the column path stays bit-identical to the general loop (SHA1 of
    rasters, `v`, `g`, buffers in the four hash configurations, buffer path forced). Script and data: `pre110_fixes/scripts/
    run_rebase.py`, `out/rb_t8.jsonl`, `out/rb_identity.jsonl`.
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
  `Tripod` and `BallAndStick` run in chunks too when their synapse models have chunk methods (`ReceptorSynapse`, `MultiReceptorSynapse`, the
  defaults; also `DoubleExpSynapse`, `SingleExpSynapse`, `CurrentSynapse`): synapses, both Heun stages and the state update of a chunk in one
  task, one fork/join per step; `IF` with `ReceptorSynapse` likewise. `DeltaSynapse`, `ExtendedIF` and the Poisson populations stay serial.
  Network of 800 / 4000 `Tripod` + 200 / 1000 `IF` (`tripod_net.jl` via `jl_thread.jl`, best setting, s per simulated second; cores 9, 7,9-11
  and 0,1,3,6,7,9-11 because 8 was taken by other pinned jobs): 1 / 4 / 8 threads 0.119 / 0.115 / 0.118 (NE = 800, no gain: the step is too
  short) and 0.586 / 0.417 / 0.379 (NE = 4000, 1.40x and 1.55x); before the vectorised Heun stage 0.212 / 0.154 / 0.149 and 1.019 / 0.520 /
  0.554 (1.96x at 4 threads): the serial part shrank more than the parallel one. Measurements and the prototype are in
  `papers/JuliaSNN_publication/validation/threading`.

### Breaking changes
Each item gives the migration.
- `iSTDPTime` removed (struct, docstring, export). It had no `plasticity!` kernel, so a synapse built with it errored under `train!` and it could not run. Migration: use `iSTDPRate` or `iSTDPPotential`.
- `SpikingSynapse` field types. `LTPParam`, `STPParam`, `LTPVars`, `STPVars` are type parameters (`LTPP`, `STPP`,
  `LTPV`, `STPV`), so a rule of another type can no longer be assigned in place. `update_plasticity!` and
  `change_plasticity!` accept only a rule of the same type as the current one (new parameters, fresh state) and
  throw an `ArgumentError` otherwise (for example, the documented `change_plasticity!(syn; LTP = STDPGerstner())`
  on a synapse built without LTP). Direct assignment `c.LTPParam = rule` of another type is a `MethodError`
  (cannot `convert`).
  Migration: build the synapse with the rule (`LTPParam = ...`), or `c2 = with_plasticity(c; LTP = rule)` and
  replace `c` by `c2` when the model is composed (`compose(; ..., EE = c2, ...)`, or `change_plasticity`); `c2` shares `W`,
  the connectivity, `param` and `records` with `c` (see "Switching plasticity on and off"), so do not keep both in one model. Same-type changes (new parameters) still work with `update_plasticity!`.
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
  Memory is `Npost x (max delay / dt + 1)` values of the synapse precision per delayed connection (Float32: 1000 targets, 5 ms,
  `dt = 0.125`: 160 kB; no buffer on the spike-list path of constant delays without plasticity).
  Migration: code that read `spike_time`/`spike_w` has no equivalent; read `buffer`. After editing
  `param.delaytime` in place, set `param.dt[1] = NaN32` so that the steps are recomputed (only the reordering of
  synapses by `_remap_synapse_state!` does it for you). Results at `dt = 0.125` are bit-identical to 1.9.1 on the buffer path
  (constant delays without plasticity: see "Behaviour notes"); at other `dt` see "Behaviour notes".
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
- `SpikingSynapse` stores only what it uses (see "Synapse memory footprint"). (1) `syn.ρ` is EMPTY for a synapse without a
  short-term rule (`STPParam = NoSTP()`), meaning ρ = 1 for every synapse: user code that wrote `syn.ρ .= x` on such a synapse now
  silently does nothing (an empty vector is broadcast to), and code that indexed `syn.ρ[s]` reads out of bounds. Migration: read
  with `SNNModels.efficacy(syn)` (the stored vector, or a new vector of ones); call `SNNModels.materialize_efficacy!(syn)` before
  writing to `syn.ρ` (after it the written values are used by `forward!`, as before). `monitor!(syn, [:ρ])`, `matrix(syn, :ρ)`,
  `with_plasticity(syn; STP = ...)`, `connect!`/`update_sparse_matrix!` and `synaptic_turnover!` handle this themselves.
  (2) `syn.J` and `syn.index` are EMPTY until a plasticity rule that walks the inputs of a neuron (every rule except `NoLTP`,
  `NoSTP`, the Markram STP rules and `vSTDPParameter`/`vSTDPDenseParameter`; user-defined rules count as needing them),
  `presynaptic`, `presynaptic_idxs`, `SynapseNormalization`, `AggregateScaling` or `SNNModels.ensure_transpose!(syn)` needs
  them. Code that reads `syn.J`/`syn.index` of a synapse without such a rule must call `SNNModels.ensure_transpose!(syn)` first
  (done in SNNUtils `average_weight`, `weights_indices`, `update_weight!` and SNNPlots `mutual_EI_connections`; their
  `[compat]` requires SNNModels 2). (3) A delay common to all synapses
  (`delay_dist = Dirac(d)`, or any draw that gives equal values) is stored once: `syn.param.delaytime` and
  `syn.param.delaysteps` have length 1; code that indexes them per synapse must check the length.
- Saved models and this change. Loading and saving compatibility with older models is NOT guaranteed: model files written by
  this version contain empty `ρ`/`J`/`index` and length-1 delays, and must not be opened with any older SNNModels (1.9.1 refuses
  them at load, but the earlier 1.10 development commits do not and read out of bounds under `@inbounds`). Files written by the
  earlier 1.10 development commits (full arrays; the struct layout is unchanged) were read and run by this version; since
  "One spike list per population" changed the layout of the populations they are refused by `load_model`. Breaking the reading of old files was accepted for this change but was not needed.
- One spike list per population (see that section). `SpikeIndex` gains a sixth field `owned::Bool`, and the nine
  integrate-and-fire population types (`IF`, `AdEx`, `ExtendedIF`, `HetRec`, `Tripod`, `BallAndStick`, `HH`, `IZ`,
  `MorrisLecar`) gain a field `spikes::SpikeIndex` after `fire`. Positional constructors of these structs change arity;
  keyword construction (`@snn_kw`) is unchanged and is the only path used in the package and its siblings. Because the JLD2
  layout of these types changed, JLD2 files saved by earlier 1.10 development builds load as reconstructed types and are
  refused by `load_model`/`load_data` (`check_loaded_model`). This is the same policy as for 1.9.1 files; 1.10 is unreleased.

### New API
- `with_plasticity(c::SpikingSynapse; LTP, STP)`: synapse that shares `W`, the connectivity, `param` and `records` with `c`
  (the lazy arrays only as described in "Switching plasticity on and off") and has the new rule
- `change_plasticity(model, syn_name; LTP, STP)` (exported): a new model with one named synapse replaced by `with_plasticity(model.syn[syn_name]; LTP, STP)`; everything else is shared with `model`. The old and the new synapse share `W`, `colptr` and `I` and must be treated as desynchronised from the start. An unknown name raises an `ArgumentError` that lists the synapses. `compose(; model..., EE = new)` collides on the duplicate key (the error now names the key and says to use `change_plasticity`).
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

### Bug fixes

- `Tripod` and `BallAndStick` now build their dendrites with `param.physiology` (`DendNeuronParameter.physiology`, `Ri`, `Rd`, `Cd`). Before they
  called `create_dendrite` without it and always used the default `human_dend`, whatever the parameter held (in Float32 and in the decimal Float64
  path alike). Behaviour change only for users who passed a non-default physiology (e.g. `mouse_dend`, `Rd = 1700 Ω cm²`, `Cd = 1 μF/cm²`): their
  dendrites now get the constants they asked for. The default is `human_dend` and is unchanged: the state of default `Tripod` and `BallAndStick`
  is bit-identical over 2000 steps (SHA1 of every array field, Float32 and Float64, `-t 1` and `-t 8`). No compatibility shim. Test:
  `test/pop/dendrite.jl` (dendrite arrays equal `create_dendrite` of the physiology, `mouse_dend` against `human_dend`, Float32 and Float64, still spikes).

### Behaviour notes
- Constant delays without plasticity (`NoLTP`, `NoSTP`, a delay stored once for all synapses, e.g. `Dirac`) deliver the spike list
  emitted `d` steps ago (see Performance): the weight `W` is read at arrival, not at emission, so a weight changed in between by
  the user or by metaplasticity (`SynapseNormalization`, `AggregateScaling`) is delivered with its new value; the increments of one
  step are added to `g` one at a time (rounding differences of a few ulp, chaotic trajectories diverge after some hundred steps,
  statistics unchanged). Delays that differ between synapses, and synapses with LTP or STP, keep the buffer (increment at emission).
  Switching between the two paths with spikes in flight (`with_plasticity` installing or removing LTP/STP; the old and the new
  connection share `param`) delivers each of them once: on the way to the buffer the pending lists are scattered into the columns of
  their arrival step with the current `W` (no efficacy: they were emitted before the rule existed); on the way to the spike list the
  buffer is kept and drained for `d` steps (`param.drain`) and then released. (Fix: the first 1.10 version dropped them.) A change of `dt` that keeps the delay in steps keeps the spikes in flight (as the buffer); one that
  changes it drops them (as the buffer, whose columns are rebuilt). After `update_sparse_matrix!` (rewiring) with the same delay in
  steps the spikes in flight are delivered through the new connectivity (targets and weights of the presynaptic neuron at arrival),
  where the buffer delivers the increments computed with the old one.
- Weights of `STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent`, `STDPTriplet` differ from 1.9.1 (lazy decay,
  see below). Bit-identical to 1.9.1 in the seeded audit of the dev tree before the lazy decay became the only path
  (Poisson and `SpikeTimeStimulus` inputs, `Identity` relay and two-layer networks, 4 s to 2 s of `train!`):
  `STDPMexicanHat`, `iSTDPRate`, `vSTDPDenseParameter` (then called `vSTDPParameter`), Markram STP (`rho`, `u`, `x`), a synapse with delays plus
  STDP with the dense decay, the `SNNPlots.stdp_kernel` construction. The suite of SNNModels (29719 tests: 10382 with
  `test/syn/lazy_traces.jl` rewritten, the audit measured 10278 before, plus `test/stim/poisson_sparse.jl` with 19201 and the tests of the concurrent plasticity work) passes; SNNUtils, SNNPlots and
  SpikingNeuralNetworks suites were not rerun at that point (they were later, see "Synapse memory footprint").
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
- One spike list per population (see that section). For the integrate-and-fire family, `spike_index(p)` returns the
  population's own list (`p.spikes`, the same object at every call) instead of a new list per call. Inside `sim!`/`train!`
  the list of a step is built by its first consumer. Flags set by hand in `fire` between `integrate!` of the population
  and the end of the step are therefore not seen by the later consumers; only a custom component can do this. Direct calls
  of `forward!`, `plasticity!` and `record!` outside `sim!`/`train!` (tests, manual loops, populations not passed in `P`)
  rescan `fire` at every call and see such flags, as before.

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
| `iSTDPTime`, `CaRule.jl` | no `plasticity!` method / file not included | - | removed in 2.0.0 |

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
  the same the other way), and since J3 every other population model the same way (see "Float64 for every population model" below). `float_type(x)` (exported) is the element type of a population, connection, stimulus, rule, clock or type: the first
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
  `BalancedStimulus`, `Turnover`) or when stepped (`integrate!` of a population without a Float64 method, since J3 only `HetRec`)
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
  `AggregateScaling`, `SynapseNormalization`, clock, `sim!`/`train!`, records. Stage 2 (done): `IF` with `DoubleExpSynapse`; then (J3, below) every
  population model and every population synapse model. Not converted (refuse Float64 at construction or when stepped): `HetRec`, `Turnover`, the
  rate/FORCE connections (`RateSynapse`, `SpikeRateSynapse`, `FL*`, `PINning*`), `CurrentStimulus`, `BalancedStimulus`;
  `VariableInputs`, `IF_CANAHP`, `AdExMultiTimescale` and `Multipod` were not included in the module (files deleted before the release).
- Tests: `test/syn/precision.jl` (229): the same small network in Float32 and Float64 (STDPGerstner pair, triplet, event vSTDP, dense vSTDP, STP event, Poisson
  stimulus, `SpikeTimeStimulus` onto `Identity`, `PoissonStimulusLayer`, delay synapse, normalization, aggregate scaling) with the eltypes of every state
  array, the clock, `@inferred` on a step of each kind of component, an analytic pair-STDP check in Float64 (1e-12), mixed precision and refusals.
- Float64 reconciliation of the two vSTDP rules (validation, `papers/JuliaSNN_publication/validation/stdp_continuous/reconcile_float64.jl`): in Float64 the
  event-driven and the dense rule agree to 3.6e-14 (200 synapses) and 8.2e-14 (1e4 synapses), and with Brian2 (float64) to 3e-15 (dense) and 3e-14 to 9e-14
  (event), against 2.4e-6 / 2.9e-6 between them in Float32: the Float32 difference is the accumulation error of the dense rule (hypothesis confirmed).
  Float64 costs 13 % on the dense rule at 1e4 synapses (2.09 s against 1.88 s, unpinned) and nothing measurable on the event-driven one.
- Float64 for every population model (J3). `AdEx` (scalar and per-neuron parameters), `ExtendedIF`, `HH`, `IZ`, `MorrisLecar`, `Rate`,
  `WilsonCowan`, `Tripod`, `BallAndStick`, `Poisson` (one rate or one per neuron), `VariablePoisson`, `InhomogeneousPoisson`, and on `IF`/`AdEx`
  every population synapse model (`SingleExpSynapse`, `DeltaSynapse`, `CurrentSynapse`, `DoubleExpCurrentSynapse`, `DoubleExpSynapse`,
  `ReceptorSynapse`, `MultiReceptorSynapse`, `Confavreux2025Synapse`) run in `Float64` when the population is built with `VFT = Vector{Float64}`
  and nothing else: the parameter, spike and synapse defaults derive from `eltype(VFT)` (`AdExParameter(FT = eltype(VFT))`, `PostSpike(FT = ...)`,
  `convert_precision(TripodSomaSynapse, eltype(VFT))`, `create_dendrite(...; FT)`), the synapse models allocate their state and receptor buffers
  in the population type (`synaptic_variables(s, N, FT)`, `synaptic_receptors(s, N, FT)`; the two-argument forms keep `float_type(s)`), and the
  kernels take any `dt::AbstractFloat`. `convert_precision` now covers the synapse models (`ReceptorSynapse`/`MultiReceptorSynapse` convert each
  `Receptor` and the NMDA block), `PostSpike` and the population parameters; `PoissonParameter(rate; FT)` and `VariablePoissonParameter{FT}` are new.
  One precision per population: a `Float32` parameter, spike or synapse object passed to a `Float64` population (or the reverse) is an
  `ArgumentError` from a shared check (`_check_precision`); objects without a float field (`DeltaSynapse()`, `RateParameter()`) and the
  morphology `param` of `Tripod`/`BallAndStick` are precision-agnostic. `snn_kw_finalize` now receives the struct type with its type
  parameters (e.g. `HH{Vector{Float64},Vector{Bool}}`), so the check reads `eltype(VFT)` and not the unconverted default of `v`. `exp64`/`exp256`
  are generic in the float type; `_pow4` has a `Float64` method; `HH`/`IZ`/`MorrisLecar` write their literals as `T(0.32)` (`=== 0.32f0`
  in Float32, checked for every literal). `Tripod.Vspike` stays a `Float32` field, compared after promotion; the dendritic geometry of a
  Float64 population is computed in Float64 from decimal values (see "Float64 parameter values" below). Still Float32 only (refused): `HetRec`, the rate/FORCE connections, `CurrentStimulus`, `BalancedStimulus`,
  `Turnover` (`VariableInputs`: file deleted before the release).
  - Float32 is bit-identical: SHA1 of the whole state trajectory (N = 257, 2000 steps, seeded synaptic drive) of 32 cases (every population
    model, the two Poisson regimes, heterogeneous AdEx, `Tripod` with `SingleExpSynapse`, `IF` and `AdEx` with each of the eight synapse
    models) equal before J3 (e265921) and after each of the four J3 commits (`pre110_fixes/out/j3_identity.csv`); all `*_update_loop.jl` and
    `inline_math_test.jl` tests pass unchanged. Review round 1 (`scripts/j3_f32_identity_r1.jl`, `out/j3r1_identity.csv`): 36 cases (the 32
    above, `BallAndStick` with `SingleExpSynapse`, `MorrisLecar` driven to spike, `Rate` and `WilsonCowan` with different drives so that
    their hashes differ), spike flags and the records (`monitor!` of `fire` and the main variable, read back with `spiketimes` and
    `getvariable`) in the hash, at `-t 1` and at `-t 8` with `neurons_per_thread = 128` (chunked and threaded paths, N = 257 in 3 chunks):
    j3pre (e265921) against the review commit d2a11f0, 72 / 72 pairs equal (36 cases x 2 configurations, state and records SHA1; the
    threaded hashes also equal the serial ones). Full suite `-t 1` 75918 / 75918; threading tests and `precision_populations.jl` at
    `-t 4` 8733 / 8733.
  - Float64 parameter values (review round 1). The library unit constants are `Float32` and some are not exact (`um`, `Ω`, `Hz`), so a
    default computed with them and converted to `Float64` carried the `Float32` rounding (`τr = 0.25999999046`). The `Float64` path now
    uses the decimal values, the `Float32` defaults are unchanged (each `Float32` object compared field by field with the one before):
    the predefined receptors and NMDA blocks of `SomaSynapse`, `TripodSomaSynapse`, `TripodDendSynapse` are built from their decimal
    keywords and `convert_precision` rebuilds them from those keywords in the target type (`τr = 0.26`, `g0 = 0.73`, `b = 3.36`, derived
    `gsyn`, `α`, `τr⁻`, `τd⁻` computed in Float64); the defaults `NMDAVoltageDependency` `b, k, mg`, `Confavreux2025Synapse.α = 0.23`,
    `MorrisLecarParameter.ϕ = 0.025`, `PoissonHomoParameter.rate = 0.001` are decimal literals (`Float32(parse(Float64, s)) ===
    parse(Float32, s)` checked for every literal); `AdExParameter`/`IFParameter` resolve the membrane pair in their precision
    (`τm = 281/40 = 7.025`, `R = 1/40`; IF `R = 0.06`, `gl = 1/0.06`, `C = 15 gl`); `HHParameter` `Cm, gl, gn, gk` are
    `200, 10, 20000, 6000` in Float64 and keep their Float32 values `199.99998, 9.999999, 19999.998, 5999.9995` (computed with the
    inexact `um`; `_unit_default`); the dendrites of a Float64 `Tripod`/`BallAndStick` (`create_dendrite(...; FT = Float64)`) draw the
    length exactly as Float32 does (`rand` of the same Float32 range `lmin:1um:lmax`: same index, same RNG state afterwards) and read
    the drawn length, fixed lengths and the diameter as the decimal multiple of `um` they denote (`200um` as `0.02`, `5um =
    0.00049999997f0` as `0.0005`), take the cable constants of any `Physiology` as the decimals in their units (`human_dend`
    `(2e-7, 3.8907e-5, 5e5)`, `mouse_dend` `(2e-7, 1.7e-6, 1e6)`), and compute `gm`, `gax`, `C` in Float64 (Float32 geometry
    unchanged). Review round 2: the first version sampled the decimal range `lo:1e-4:hi` built from the shortest decimals of the
    bounds, which differs in length from the Float32 range when a bound is not the correctly rounded Float32 of its decimal (1105
    integer-um ranges on a 5 um grid, e.g. `(0, 5um)`, `(10um, 500um)`, `(35um, 85um)`), and had `mouse_dend.Rd = 1.7e-5` instead
    of `1.7e-6`; test: every integer-um range on a 5 um grid in 0-500 um, two seeds, Float64 lengths are the decimals of the
    Float32 lengths and the RNG states are equal. Test: every default reachable from a Float64 population of every model, and from every Float64 synapse model and
    parameter type, is checked not to be a Float32-rounded value. Still `Float32`-rounded in a Float64 model: values the user writes
    with the inexact unit constants (`10Hz`, `150um` outside the dendrite lengths) and any non-predefined `Float32` object converted with
    `convert_precision` (converted value by value), the `Float32` field `Tripod.Vspike`.
  - Steps of another precision (review round 1). Every population model with a Float64 method converts a `dt` of another type to
    `float_type(p)` at the top of `integrate!` (as `IF`/`AdEx` through the generalized-IF `integrate!`, `Tripod`, `BallAndStick`, the point
    models and the Poisson family alike): `integrate!(p, p.param, 0.1)` on a Float32 population steps with `0.1f0`, a Float64 population
    given `0.1f0` steps with `Float64(0.1f0)`. `sim!`/`train!` already pass `dt` in the population type, so nothing changes under them
    (Float32 bits unchanged). `HetRec` (Float32 only) still refuses a Float64 step.
  - Synapse models written for 1.9 (third-party types). The populations now call the three-argument `synaptic_variables(s, N, FT)` and
    `synaptic_receptors(s, N, FT)`. A model that defines only the two-argument forms keeps working in Float32 populations: the generic
    three-argument methods call its own two-argument method when `FT` is the precision of the model (`float_type(s)`, Float32 for such a
    model); a Float64 population needs the three-argument methods, and the error says so. (Before this fallback a model with only a
    two-argument `synaptic_receptors` silently got the default `glu`/`gaba` buffers.)
  - Tests: `test/pop/precision_populations.jl` (596 after review round 1, 417 before): per model (also `Tripod` and `BallAndStick` with
    `SingleExpSynapse` on soma and dendrites), Float64 construction from `VFT` only with no `Float32` field left (state,
    parameters, spike, synapse, synaptic variables, receptor buffers, dendrites), `@inferred` step with `dt::Float64`, allocations no larger than
    in Float32 (0 where Float32 is 0), Float64 against Float32 from the same state: 200 quiet steps (no spike, main state within 1e-4
    relative) and 500 driven steps (1000 for HH; HH and MorrisLecar with 3x and 2x the current so that they fire): 20 to 200 spikes, total
    count within 2 %, per neuron count within 1 and first spike within 2 steps (observed: equal), state before the first spike within
    1e-4 relative (observed <= 3e-5); decimal defaults; `dt` of the other precision; two-argument synapse models;
    two Float64 runs identical, `dt` against `dt/2`, a `SpikingSynapse` onto the population is Float64; the synapse models on `IF`/`AdEx`;
    `convert_precision` round trips; refusals; a Float64 network of `Poisson`, `Tripod`, `IF` with `sim!`. Coverage of the J3-changed source lines
    by these and the update-loop tests: 183 / 188 (97 %).
  - Cost (`pre110_fixes/scripts/jl_neurons_prec.jl`, T5 single population, pinned core, median of 3, warm-up discarded; ratio Float64 / Float32,
    spikes / no spikes): N = 10000: `IF` 2.29 / 2.41, `AdEx` 1.70 / 1.71, `ExtendedIF` 2.65 / 2.73, `HH` 1.39 / 1.40, `IZ` 1.96 / 1.94,
    `MorrisLecar` 4.32 / 3.60, `Rate` 6.54 / 1.90, `WilsonCowan` 8.49 / 1.92, `Tripod` 2.99 / 2.96, `BallAndStick` 2.57 / 2.60, `Poisson`
    (500 Hz / 1 Hz) 1.97 / 0.95; N = 1000: `IF` 2.60 / 2.45, `AdEx` 1.75 / 1.77, `ExtendedIF` 2.72 / 2.73, `HH` 1.36 / 1.42, `IZ` 2.03 / 1.99,
    `MorrisLecar` 3.66 / 3.58, `Rate` 7.95 / 1.92, `WilsonCowan` 8.37 / 1.89, `Tripod` 2.08 / 2.04, `BallAndStick` 2.08 / 2.11, `Poisson`
    1.66 / 0.95 (the multicompartment models pay more at N = 10000; hypothesis, not verified: their Float64 state no longer fits the cache). `Rate`/`WilsonCowan` with input and `MorrisLecar` use `Base.tanh`/`cosh` in Float64 (out of line,
    the loop does not vectorise; the inlined copies exist for Float32 only). Float32 after J3 against before (same round): 0.946-1.019 for
    every model and regime except `Poisson` without spikes, 1.047 (N = 1000) and 1.092 (N = 10000) at 0.006-0.05 ns per neuron-step, the
    noise level of this measurement. The regression pairs the Float32 arms before and after J3 of one round (r2 for `Rate` and `Poisson`,
    re-measured because their r1 arms were noisy); the Float64/Float32 ratios pair the arms of round r1. Table (each column with its
    round): `pre110_fixes/out/tables_j3.md`, figure
    `pre110_fixes/figures/pre110_fixes.png` (d).

### Synapse memory footprint (`SpikingSynapse`, 29 to 9 bytes per synapse)
Live memory of the synapses of the T4 network of the performance campaign (IF E 8000 / I 2000, `CurrentSynapse`, K = 100,
1e6 synapses, `delay_dist = Dirac(0.8)`, no plasticity), `Base.summarysize` per field after 100 steps (arrays that alias
population state counted in the synapse); construction temporaries are `@allocated` of one more EE constructor. Results of the
reference worktree (6b7addc) and of the four steps, which were measured one after the other; Float32 results bit-identical to
6b7addc after every step (spike times, final `v`, synaptic variables, `W`, efficacy; T4 network, pair STDP + iSTDP, event vSTDP,
Markram STP and STP Het, weight-recorded network, heterogeneous delays, no delays; serial and threaded with
`neurons_per_thread = 64`, `SYN_MIN_OPS[] = 0`). Scripts and CSV: `papers/JuliaSNN_publication/validation/profilation_v1.10/perf/
synapse_footprint/` (`scripts/fp_fields.jl`, `scripts/fp_identity.jl`, `scripts/fp_summary.py`; `out/fields.csv`,
`out/construct.csv`, `out/summary_per_cut.csv`, `out/fields_EE_per_cut.csv`, `out/identity.csv`).

| step | network B/syn | EE | EI | IE | II | EE construction B/syn |
|---|---|---|---|---|---|---|
| before (6b7addc) | 29.18 | 28.77 | 29.12 | 30.77 | 31.10 | 64.5 |
| 1: Int32 storage without the transposed temporary | 29.18 | 28.77 | 29.12 | 30.77 | 31.10 | 36.4 |
| 2: `ρ` empty without STP | 25.18 | 24.77 | 25.12 | 26.77 | 27.10 | 32.4 |
| 3: `J`/`index` built only when read | 17.18 | 16.77 | 17.12 | 18.77 | 19.09 | 24.3 |
| 4: a common delay stored once | 9.18 | 8.77 | 9.12 | 10.77 | 11.09 | 24.3 |

EE per field (B/syn): `I` 4, `W` 4 (both kept); `J` 4, `index` 4, `ρ` 4, `param.delaytime` 4, `param.delaysteps` 4 (all 0 after
the change); ring buffer, pointers, aliased population state and spike lists 0.76. A synapse keeps 4 B/syn for `ρ` with a
short-term rule, 8 B/syn for `J`/`index` with a rule that needs them, 8 B/syn for heterogeneous delays.
- Step 1. The constructor builds `colptr`, `I`, `rowptr` in Int32 directly (`_csc_int32`, `_rowptr_int32`) instead of `dsparse`
  (`sparse(A')`, Int64 `J`/`index`, Int64 -> Int32 copies); `dsparse` is unchanged (exported, used by the other connections).
  `ensure_transpose!` builds `J` (from `colptr`) and `index` (one counting pass over the columns) in place, identical to `dsparse`.
- Step 2. `ρ = FT[]` without a short-term rule. `forward!` (both methods) passes `ρ` or `nothing` to its kernels through a
  function barrier (one branch per call, before any threaded fork); the increment is `W[s] * ρ[s]` or `W[s]` (`W * 1 == W`
  exactly; the undelayed kernel keeps the `@fastmath` product). New: `efficacy`, `materialize_efficacy!` (not exported).
- Step 3. `J`/`index` are built at construction only if a rule needs them (`needs_transpose(rule)`, `true` by default), by
  `with_plasticity`/`update_plasticity!`, by a guard at the top of `plasticity!`/`update_traces!` of the synapse (serial, before the rule
  kernels fork; covers synapses loaded from a file or built with the keyword constructor), by `presynaptic*`, by the normalization
  and scaling rules (constructor and every application), and are kept by `connect!`/`update_sparse_matrix!` if present. `matrix`,
  `matrix_record`, weight records, `update_weights!`, `indices`, `postsynaptic*` and the event vSTDP never read them (a temporary
  presynaptic index is used where needed, `_presyn_index`). New: `has_transpose`, `ensure_transpose!`, `needs_transpose`.
- Step 4. If all drawn delays are equal (after the draw and the conversion, so the random stream is unchanged) one value is kept;
  the delayed kernels take the delay in steps as a scalar or per synapse through the same function barrier. `update_sparse_matrix!`
  expands a common delay before remapping (a new synapse gets the same mean delay as with per-synapse storage) and stores it once
  again if all delays are equal. Since the delayed-transmission change (Performance, constant delays, and Behaviour notes) a delay
  stored once without plasticity (`NoLTP`, `NoSTP`) is delivered through the spike list, which is not bit-identical to the buffer
  path: a network built with a `Dirac` delay (stored once) and the same network whose `delaytime` is assigned per synapse (stored per
  synapse, buffer path) now give different bits. The bit-identity of common and per-synapse delays stated here and in the tests
  below holds for the buffer and column paths (with plasticity, or with the buffer path forced).
- Tests: `test/syn/synapse_footprint.jl` (memory bounds, lazy arrays empty/built exactly when expected for every rule type,
  `ρ` empty and its readers, in-process bit-identity of empty vs stored `ρ`, common vs per-synapse delays (buffer and column paths; see
  step 4 for the spike-list path), eager vs lazy transpose
  with pair STDP/STP, weight records and vSTDP, threaded vs serial when run with more than one thread, JLD2 round trip). Minimal
  edits of `syn/spike_index.jl`, `sim/sweep_fixes.jl`, `syn/lazy_weights.jl`, `syn/vstdp.jl`, `stim/poisson_pop_sparse.jl`.
- Test suites: SNNModels `-t 1` 74714 passed (0 failed, 0 errored); also passed with `--check-bounds=yes` and with `-t 4`.
  SNNUtils 17, SNNPlots 14, SpikingNeuralNetworks 3 passed.
- Measured time and memory, before (6b7addc) vs after (48c60eb). T4 network, K = 100, Float32, one thread pinned to core 8,
  10 s simulated, warm-up discarded. Before and after interleaved over 4 rounds; median of 12 repetitions; shared machine,
  load 4.6-8.5. Folder `perf/synapse_footprint/` (README.md, `out/time_summary.csv`, `out/mem_slopes.csv`, `out/plastic_summary.csv`,
  `fig/F_synapse_footprint.png`).

  | N | before ns/neuron-step | after | after/before (paired rounds) | Auryn, same session |
  |---|---|---|---|---|
  | 10000 | 1.647 | 1.486 | 0.88-0.93 | 1.73 |
  | 20000 | 1.863 | 1.664 | 0.84-0.94 | 1.94 |
  | 30000 | 1.966 | 1.732 | 0.86-0.93 | 1.96 |
  | 50000 | 2.43 (2.36-2.43 undisturbed) | 2.11 (2.10-2.12 undisturbed) | 0.87 | 2.16-2.18 (campaign 1.99) |

  Peak-RSS slope against synapse count (N = 1e4, 2e4, 5e4): 49.6 -> 34.7 B/syn (Auryn 8.4). RSS growth during the build:
  40.8 -> 25.1 B/syn.

  No plastic regression:
  - pair-STDP E/I N = 10000: 0.237 -> 0.216 s per simulated s, with rates and final weights identical;
  - vSTDP 1e4 synapses: dense 1.983 -> 1.989 s, event 0.0366 -> 0.0371 s;
  - pair STDP 1e4 synapses: 0.0203 -> 0.0213 s. Two of the 12 "after" processes ran at 2x, and later rounds did not
    reproduce it; without them the ratio is 1.02.

  The speed-up is close to a constant factor at every N, and the growth from 1e4 to 5e4 stays at +42 % (before +47 %, Auryn
  +26-39 %). The smaller arrays therefore help at all sizes, but they do not explain the steeper growth with N. The cache-footprint
  hypothesis of the perf README is only weakly supported. Splitting the gain into bytes and instructions per event would need
  hardware counters (not tested).

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

### Flush-to-zero during `sim!` / `train!` (branch dev-diag, pre-1.10 diagnostics, job 2)
- Cause of the p = 0.02 small-N cliff and of the slow quiet `Rate` population (perf README, disadvantage 8): subnormal
  floats. Every decay integrated as `g += -dt g / τ` (or `x += dt (-x)`) stops changing once `dt g / τ` is below half an ulp
  of `g`, so a variable with no input ends in the subnormal range and stays there (it never reaches 0); each operation on it
  takes a microcode assist. Counted (`job2_subnormal_counts.csv`, every 100 steps over 10 s): T4 p = 0.02 N = 500, `E.gi`
  up to 70 of 400 entries subnormal (mean 58), `I.gi` 17/100, `E.ge` 3, `E.syn_curr` 2; N = 1000 `E.gi` 62/800; N = 2000
  `E.gi` 34/1600; quiet `Rate` (N = 10000) all of `x` and `r` after 80 ms. Weights, delay buffers, `v` and every other array:
  none. The same happens with `SingleExpSynapse`, `DoubleExpSynapse`, `DoubleExpCurrentSynapse` and the default `AdEx` synapse
  (REPL probe, not in a CSV: every synaptic variable subnormal after 2 s without input).
- Fix: `sim!` and `train!` take `flush_subnormals::Bool = true` and run the step loop inside `with_flush_subnormals`
  (new file `src/utils/subnormals.jl`): `set_zero_subnormals(true)` on the calling thread and, when the population updates
  are threaded (`neurons_per_thread > 0`, more than one thread), on every Polyester worker (`@batch per = thread`); the
  previous mode of each thread is restored afterwards, also on an exception. One change covers every model, synapse and
  plasticity trace, and leaves the arithmetic of normal numbers, so no kernel and no generated loop changes (a per-loop
  `ifelse(abs(x) < floatmin)` would have touched about ten `@fastmath` loops, some of which are kept bit-identical to 1.9).
  `integrate!`/`forward!` called directly (outside `sim!`/`train!`) are not affected.
- Effect on results: a result that would be subnormal is 0 and a subnormal operand reads as 0, so a decaying variable
  stops at a small normal value (|g| <= floatmin τ / dt, 5.9e-37 for τ = 5 ms, dt = 0.1 ms) instead of a subnormal one.
  T4 p = 0.02 N = 500/1000/2000 over 10 s: spike times of E and I and the final `v` bit-identical to the run without it;
  `ge`/`gi` differ in 1-64 entries per array by at most 5.9e-37 (absolute); quiet `Rate` `x`/`r` by at most 1.2e-37
  (`job2_correctness.csv`). The multi-threaded path (`-t 4`, `neurons_per_thread = 128`) gives the same `v` with and
  without the flag and leaves every thread with FTZ off.
- Measured (`perf/pre110_diag/job2_subnormal.csv`, core 8, `-t 1`, median of 3, load1 4.6-8.3; before = ee48a1f): T4 p = 0.02
  s per simulated second, IEEE vs `sim!` default after the change: N = 500 0.0512 to 0.0090 (Auryn 0.0107), N = 1000 0.0473 to
  0.0160, N = 2000 0.0476 to 0.0291; quiet `Rate` through `sim!`, ns per neuron-step: N = 1000 40.8 to 0.86, N = 10000 39.4
  to 0.68 (driven: 0.67-0.85, unchanged). Setting FTZ by hand on the 1.9/ee48a1f code gives the same numbers.
- Test: `test/sim/subnormals.jl` (14 tests): a quiet `Rate` and a silent `CurrentSynapse` have no subnormal entries after
  `sim!`/`train!` (and do with `flush_subnormals = false`), spikes and `v` identical with and without the flag, thread mode
  restored after the run and after an exception. Full suite `-t 1`: 74620 / 74620 (with OrderedCollections 1.8.2, see below).
- Found while testing: a fresh resolve of the test environment picks OrderedCollections 2.0.2, which no longer has
  `sort(::Dict; by)`; `compose` then throws (`utils/util.jl:212-214`, 86 errored tests in `record_test.jl` and others). The
  known limitation below is now real; not fixed here (outside job 2; fixed in the next section). The suite was run with OrderedCollections pinned to 1.8.2
  in a scratch copy of the test environment.

### `compose` without `sort(::Dict)` (branch dev-diag, pre-1.10 diagnostics, job 3 task 0)
- `compose` (`utils/util.jl`) and `population_indices`, `filter_items`, `subpopulations` (`analysis/populations.jl`) built
  their `NamedTuple` with `dict2ntuple(sort(d; by))`, i.e. the `sort(::Dict)` method that OrderedCollections 1.x added to
  `Base.sort` (deprecated there, removed in OrderedCollections 2.0). A fresh resolve now picks OrderedCollections 2.0.2 and
  `compose` threw (86 errored tests). New internal helper `sorted_ntuple(d; by = identity)`: sorts the keys with `by`
  (applied to the keys, as before), converts `String` keys to `Symbol` and returns the `NamedTuple` in that order, so the
  returned objects are unchanged (same key order, same value types; ties of `by` keep the `Dict` iteration order as the
  stable `sortperm` of the old method did). No new dependency.
- Full suite `-t 1` in a freshly resolved copy of `test/Project.toml` (OrderedCollections 2.0.2): 74620 / 74620.

### Startup jump N = 10000 -> 20000: a full GC mark, no library change (branch dev-diag, pre-1.10 diagnostics, job 3)
- Perf README T11: construction of the balanced k100 network 0.064 s at N = 10000, 0.46 s at N = 20000. Measured stage by
  stage (`perf/pre110_diag/job3_startup.jl`, `job3_startup.csv`, `job3_stages.csv`; idle physical core, 2 or 5, because
  core 8 was in use by another job; `-t 1`, median of 3 after a warm-up; load1 5.9-8.7) at N = 10000/15000/20000/30000.
- Cause: one collection that re-marks the whole heap, ~0.27 s, independent of N. It lands inside the construction once
  the bytes allocated by the construction exceed the GC heap-growth trigger: 151/227 MB (N = 10000/15000) give one cheap
  young collection; 303/454 MB (N = 20000/30000) give a full sweep (0.04 s) and then a full re-mark (`mark_s` 0.27 s;
  inside the build in 1 of 6 reps at N = 20000 and 5 of 6 at N = 30000, with and without the 10 s simulation between reps;
  where it falls is set by the GC, which is why the campaign saw 0.46 s at N = 20000). The mark cost is the base heap: a full GC
  after `using SNNModels` and the warm-up takes 0.30 s (mark 0.26 s, 36 MB live) against 0.09 s in a bare Julia
  (`job3_gc_deps.csv`: DrWatson or JLD2 alone 0.16 s, DSP 0.14 s, LoopVectorization 0.13 s). Profile at N = 30000
  (`job3_profile_N30000.txt`): 56 % of the samples in `ijl_gc_collect`, 47 % in `gc_mark_loop`, entered from a small
  allocation in `dsparse`/`SpikingSynapse`, i.e. wherever the trigger happens to be crossed.
- No O(N^2) step, dense intermediate or push!/resize! growth: allocated bytes are linear (15.1 MB per 1000 neurons,
  ~1240 allocations at every N). With the GC paused during the build (`gcoff=1`) startup is 0.082/0.128/0.180/0.285 s;
  the library part (4 `SpikingSynapse`) 0.033/0.050/0.072/0.112 s and 62 B allocated per synapse (185 MB at N = 30000);
  the script part (graph read, masks, `sparse(I, J, V)`) 0.049/0.076/0.109/0.173 s and 89-268 MB.
- Benchmark-script fix (not applied to perf/scripts, shown in `job3_startup.jl script=lean`): read the Int32 file in place
  and fill the four CSC blocks in one pass (the file is sorted by (pre, post), i.e. CSC order; identical matrices,
  `mode=checklean`). Startup 0.073/0.069/0.095/0.158 s at N = 10000/15000/20000/30000 (from 0.073/0.126/0.231/0.537 s),
  163 MB at N = 20000, no full collection up to N = 30000.
- Library: nothing cheap to change. Allocation per synapse could drop from 62 B to roughly 40 B (the transposed copy in `dsparse`,
  the double delay vector and the copy in the keyword constructor of `SpikingSynapse`), which only moves the trigger; the
  re-mark cost belongs to the Julia runtime and the loaded dependencies.

### No allocation per `sim!`/`train!` step (branch dev-diag, pre-1.10 diagnostics, job 4)
- Cause of the residual per-step allocations (Performance above: 240 B with a `PoissonStimulus`, 384 B balanced, 560 B
  `Tripod` in the review measurements): the step loops over abstract component vectors, so `_step_dt(x, dt)` and
  `integrate!`/`stimulate!`/`forward!` are dynamic calls, and the `Float32` returned by `_step_dt` (and passed on) was
  boxed: one 16 B box per stimulus, population and connection per step (`Profile.Allocs`, `main.jl` lines of
  `stimulate!`, `integrate!`, `forward!`; nothing in the component methods). `with_flush_subnormals` and
  `with_neurons_per_thread` allocate nothing.
- Fix (`src/utils/main.jl` only): the keyword forms of `sim!`/`train!` share one loop (`_run!`, `_run_loop!`, and
  `_run_pbar!` for the progress bar) that wraps `dt` in a `Ref` once per call; the step (`_sim_step!`, `_train_step!`)
  calls one function barrier per component, `_sim_component!(x, param, dtr, T)` / `_train_component!`, with the component and
  its `param` read from the boxed vector element (some populations hold an abstractly typed `param`; a call dispatched on it
  from inside a method specialized on the population would box the population again, 176 B for an `IF`) and returns
  `nothing`, so no argument or result is boxed and everything inside the barrier is statically dispatched. The public
  per-step methods `sim!(P, C, S, dt, T)` / `train!(P, C, S, dt, T)` remain (16 B each: the `Ref`; 0 B since 7cb36e7 when the three vectors are concretely typed, see "Known limitations"). The firing-rate `Dict`
  is only built with `pbar = true`. Order of operations and arithmetic unchanged.
- Measured (`perf/pre110_diag/job4_alloc.csv`, `job4_alloc.jl`; pinned idle core, `-t 1`, `@allocated` after warm-up,
  model form, 1/10/100/1000 steps, least-squares slope and constant; before = 7c3d4ad): `IF` E/I with `CurrentSynapse` (2
  populations, 3 synapses) 80 B per step + 2192 B per call to 0 + 1936; `IF` with a `PoissonStimulus` 48 + 1008 to 0 + 752;
  `Tripod` with a `PoissonStimulus` 48 + 1376 to 0 + 1120 (same for `train!`). Per-call constant (`job4_percall_sites.csv`):
  `_args_model` (the component vectors and the boxing of the populations stored in the model NamedTuple), `_allocate_records!`
  (`utils/record.jl:278-281`, iteration tuples and `Float32` boxes, recording code not changed here), the `Ref` (16 B).
  Spike counts and final `v` bit-identical before/after for the three cases (`job4_correctness.csv`).
- Test: `test/sim/allocations.jl` (18 tests): for the three networks, `sim!` and `train!` bytes do not grow with the run
  length (< 1 B per step between 10 and 400 steps), the per-call constant is below 4 kB, and `_sim_step!`/`_train_step!`
  allocate 0 B.

### Switching plasticity on and off (`with_plasticity`, `without_plasticity`)
`with_plasticity(c; LTP = nothing, STP = nothing)` returns a new synapse with the connectivity, weights and delays of `c`; a rule
left to `nothing` is kept with its state, `NoLTP()`/`NoSTP()` switch that part off. The lazy arrays are handed over by one rule:
`J`/`index` go to the new synapse only if one of its rules `needs_transpose` (the source's vectors if it has them, else fresh vectors
built by `ensure_transpose!` on the new synapse; otherwise fresh empty vectors); `ρ` is the source's with `STP = nothing`, the
source's (if stored) or fresh ones with a new short-term rule, and a fresh empty vector (efficacy 1) with `STP = NoSTP()` (fresh
ones while `:ρ` is recorded, so that the record continues with the value 1). Everything else (`W`, `I`, `colptr`, `rowptr`, `g`,
`v_post`, `fireI`, `fireJ`, `param`, `targets`, `records`, `id`, `name`, spike lists, chunk cache) is shared, as before. The vectors
of the source are never resized or refilled. New: `without_plasticity(c) = with_plasticity(c; LTP = NoLTP(), STP = NoSTP())`
(exported). The switched-off synapse has the type of a never-plastic synapse (`NoLTP`, `NoSTP`, `NoVariables`), its memory and
its per-step work, and gives the same run as a synapse rebuilt by the constructor from the frozen weights (tested bit for bit).
- Behaviour changes. (1) The source is not modified any more: before, a switch-on materialised `ρ` and `J`/`index` in the source
  (aliased vectors resized in place: a never-plastic source grew from 8.73 to 20.73 B/syn and its `ρ` was then written by the new
  short-term rule). (2) `STP = NoSTP()` resets the efficacy to 1: before, the switched-off synapse kept the last STP efficacy in
  transmission (and the `ρ` kernel); to freeze `ρ` at its value keep the rule and call `set_STP!(c, false)`. (3) Records of a rule
  variable that the new rule does not have (`meta[:var_map]` entry of the replaced group whose field the new state lacks, e.g.
  `:LTPVars_tpre` after switch-off or Gerstner -> Triplet) are closed: removed from `records[:data]`, buffer, times and `var_map`
  kept and readable; before, the next sampling step threw a `FieldError` mid-run. A record whose field the new rule has
  (Gerstner -> WeightDependent `:tpre`) continues. `:W`, `:ρ`, `:fire` and population records are untouched.
- `plasticity!`/`update_traces!` of a sparse synapse return at once when it has no rule (`_has_rules`, a type-level constant for
  `SpikingSynapse`): the two `any(active)` reads per call of a never-plastic or switched-off synapse are gone; results unchanged.
- Memory (B/syn = `Base.summarysize / nnz`). Small test network (E 1600 / I 400, Dirac 0.8 ms, EE 127918 synapses), EE switched on
  (`STDPGerstner + MarkramSTPParameter`), trained, switched off: never-plastic 8.73; plastic 21.13; source after the switch-on 20.73
  before / 8.73 after; switched off 20.73 before / 8.73 after. T4 network (N = 10000, 1e6 synapses, k100, Dirac 0.8 ms,
  `CurrentSynapse`, STDPGerstner on EE): switched-off EE 16.77 before / 8.77 after (network 14.36 / 9.24), equal to never-plastic.
- Time, T4 network, pinned core 8 of seroquel (Xeon w5-3425), `-t 1`, 1 s pre-phase then 10 s timed (1e5 steps), median of 3
  processes x median of 3 reps, 1-min load average 5.6-8.9 (shared machine), commit e10795c. ns per neuron-step: never-plastic
  1.603 (`train!`) / 1.547 (`sim!`), plastic 2.123, switched off 1.525, rebuilt by the constructor 1.547; B/syn network 9.24 / EE
  8.77 for never, off and rebuilt, 14.49 / 16.97 plastic. Switched off and rebuilt fire identically (535904 E / 134448 I spikes in
  the timed window). Scripts, CSV, figure: `papers/JuliaSNN_publication/validation/profilation_v1.10/perf/plasticity_switch/README.md`.
- Tests: `test/syn/plasticity_switch.jl` (434): round trip on, off, on again for pair, triplet, weight-dependent, event and dense
  vSTDP, iSTDPRate, Markram event/timestep and pair STDP + Markram (no delay, Dirac, heterogeneous delays) against twins built by
  the constructor; the source of a switch still simulates as an untouched twin; records across the switch; Float64; threaded vs
  serial; normalization and `presynaptic` build `J`/`index` at use; the early return. `test/syn/synapse_footprint.jl`: one assertion
  that encoded the old aliasing of `ρ` updated. Full suite (fresh resolved test env, Julia 1.12.6): `-t 1` 75148 pass, 0 fail, 0 error (baseline 48c60eb, run from a git-archive snapshot: 74711 pass, 1 fail, 1 error, 74713 total; the fail is `io_test.jl:81`, `get_git_commit_hash` needs a git repo, and the error a vstdp data-path skip; the same tree run in place has 74714 tests; +434 new); `-t 4` 75150 pass (at 81874f3); the new file alone with `-t 4` and with `--check-bounds=yes` pass (-t 4 rerun at 3e3f19e: 434; --check-bounds run at 81874f3: 432).

### Spike recording from the spike lists
`record_fire_dense!` used to visit every neuron at every step and, for each spike, read `meta[:allocated][:fire]` through a
`Dict{Symbol,Any}` (35 % of the step of the T4 network with `:fire` recorded, `perf/out/profiles.txt`). It now writes the list
of the neurons that fired in the step (`src/utils/record.jl` only; population, stimulus and connection files untouched):
- Source of the list (`_recording_spike_index`, `_fire_step_list`): the owner's `SpikeIndex` when it is fed (`fed == true`)
  for `Poisson`, `VariablePoisson`, `InhomogeneousPoisson`, `Identity`, `PoissonStimulusLayer` (O(spikes)); otherwise a
  word scan of `fire` into a recorder scratch (`_flags_to_list!`, N/8 `UInt64` loads plus O(spikes)). The IF family (`IF`,
  `AdEx`, `Tripod`, `HH`, `IZ`, ...) now owns a list shared with its connections: inside the step the recorder reads it,
  and only outside the step window (`record_zero!`, direct `record!` calls) does it take the scan (see "One spike list per
  population"). `SpikeTimeStimulus` also takes the scan (its `TouchList` is in spike-time order and can
  hold duplicates). A test checks that every type with its own `spike_index` method has a `_recording_spike_index` method.
- Capacity growth checked once per step (`_ensure_fire_capacity!`, with `_grow_fire!` as its out-of-line slow path; same single "Fire COO buffer
  overflowed" warning). Subsets (`indices[:fire]`) through a prebuilt membership mask (`_fire_mask!`, rebuilt at every
  `sim!`/`train!` entry); unsorted or duplicated subsets keep the previous order-preserving loop. Typed, hoisted meta reads
  in `_record!`. The legacy jagged `record_fire!` and variable recording are unchanged in behaviour.
- `_allocate_records!`: function barrier per component group with one `(dt, duration)` holder instead of
  `Iterators.flatten` over a heterogeneous tuple; allocation per call from 0.9-5 kB (grew with the number of components) to
  16 B.
- Recorded spikes (times, ids, order) bit-identical to 48c60eb: `test/utils/record_spikeindex_test.jl` (every population and
  stimulus type, subsets, chunks, manual loops, threading, Float64, 0 B per step added by `:fire` recording) and the
  benchmark hashes below.

Benchmark (T4 balanced E/I LIF, K = 100, Float32, 10 s, one thread pinned with `taskset`, 2 processes x 3 repetitions per
cell alternated before/after, compilation excluded; host ld25-1001 (seroquel), load average 3.5-7.1). ns per neuron-step and
recording overhead rec1/rec0 (`:fire` of E and I / nothing recorded); Auryn, Brian2 and 1.9.1 ratios from the T4 campaign
(`perf/out/t4.csv`), Auryn at N = 5000 run with this benchmark:

| N | rec0 before | rec0 after | rec1 before | rec1 after | overhead before | overhead after | Auryn | Brian2 | 1.9.1 |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 1.542 | 1.567 | 2.991 | 1.940 | 1.94 | 1.24 | 1.06 | 1.00 | 1.10 |
| 2000 | 1.391 | 1.393 | 2.540 | 1.625 | 1.83 | 1.17 | 1.09 | 1.01 | 1.10 |
| 4000 | 1.420 | 1.431 | 2.541 | 1.569 | 1.79 | 1.10 | 1.09 | 1.00 | 1.07 |
| 5000 | 1.472 | 1.458 | 2.647 | 1.664 | 1.80 | 1.14 | 1.08 | - | - |
| 10000 | 1.488 | 1.495 | 2.617 | 1.608 | 1.76 | 1.08 | 1.08 | 0.99 | 1.08 |
| 20000 | 1.645 | 1.644 | 2.734 | 1.778 | 1.66 | 1.08 | 1.08 | 0.99 | 1.06 |

- What recording still costs (0.37 us per step at N = 1000 to 2.7 us at N = 20000, against 1.4-22 us before) is a fixed
  `_record!` cost of about 0.2 us per population per step (dominant at N <= 2000) plus the flag scan of about 0.1 ns per
  neuron (dominant at N >= 10000). Targets of the manifest: overhead <= 1.10 met at N = 4000, 10000, 20000; missed at
  N = 2000 (1.17) and 5000 (1.14); <= 1.20 at N = 1000 missed (1.24); stretch <= 1.05 at N >= 10000 not met (1.08).
- No recording: after/before 0.98-1.02 (no regression).
- Voltage recording (`:v` of E and I at 1 kHz): `:fire` + `:v` over rec0 went from 1.78-2.36 to 1.17-1.45; the cost `:v`
  adds on top of `:fire` is 0.08-0.21 of rec0 after against 0.11-0.42 before; `:v` alone is unchanged at N >= 10000
  (1.15-1.16 against 1.13-1.16) and faster at small N (1.28 against 1.48 at N = 1000).
- Spike counts, spike hashes and `:v` hashes identical before/after at every N (mode=check).
- Scripts, CSVs, README and figure: `papers/JuliaSNN_publication/validation/profilation_v1.10/perf/record_fix/`
  (`scripts/rf_balanced.jl`, `run_rf.py`, `rf_tables.py`, `plot_rf.jl`; `out/summary.csv`, `out/overhead.csv`,
  `out/check.csv`, `out/time.csv`; `fig/F_record_fix.png`/`.pdf`). N = 5000 data generated with the generator and seeds of
  `perf/scripts/make_data.py` (`scripts/rf_make_data.py`).
- Behaviour notes: an out-of-range entry of `indices[:fire]` now raises `ArgumentError` (naming the population) when the mask
  is built, at the first `sim!`/`train!` (previously undefined behaviour under `@inbounds`). For the populations that own a
  fed list (`Poisson` family, `Identity`, `PoissonStimulusLayer`), `fire` flags set by hand (not by the population's own
  update) are no longer recorded while `fed` is set; the connections reading those populations already ignored them.
- Not fixed here (pre-existing on 48c60eb): `test/Project.toml` lacks `SHA`, which `test/threading_synapses.jl` uses, so that
  file fails in a fresh test environment (fixed on `dev` by ee48a1f, which adds `SHA`).

### One spike list per population (integrate-and-fire family)
Branch `dev-poplist`, commits 94b1415..78165d3, manifest `~/.claude/code-manifests/snnmodels-poplist.md`.

Before this change, `spike_index(p)` returned a new `SpikeIndex` at every call for the IF family. Every `SpikingSynapse`
therefore kept two private lists (pre and post), and each list scanned the population's `fire` once per connection and step.
The `:fire` recorder made one more scan.

What changed:
- **Owned list (D1, D2).** The nine types `IF`, `AdEx`, `ExtendedIF`, `HetRec`, `Tripod`, `BallAndStick`, `HH`, `IZ` and
  `MorrisLecar` (`const _OWNED_LIST_POPULATIONS`, populations.jl) have a field
  `spikes::SpikeIndex = owned_spike_index(N)`. `SpikeIndex` has a new flag `owned`, and `_arm!`/`_disarm!` (called by
  `prepare_spikes!`/`release_spikes!` of each connection) leave owned lists alone.
- **Shared by the synapses.** `spike_index(p)` returns `p.spikes`, so `pre_spikes`/`post_spikes` of every `SpikingSynapse`
  built from the population are the same object. `spiking_synapse.jl` is not modified. `with_plasticity`,
  `without_plasticity`, `modelcopy` and `save_model`/`load_model` keep the sharing.
- **Lazy step window (D3).** `_publish_spikes!(p)` runs right after `integrate!` in `_sim_component!`/`_train_component!`.
  It sets `active = true, valid = false` (O(1)); the first consumer of the step (a connection or the recorder) scans `fire`
  once, and the others reuse the list. `_retire_all_spikes!(P)` closes the window at the end of `_sim_step!`/
  `_train_step!`. A failed step is a fatal error and the run is discarded: nothing closes the window after an exception (no `try/finally`),
  and the next step publishes afresh. Outside the window an owned list rescans at every call, exactly like an unarmed scratch list. A population that no
  synapse or recorder reads pays the O(1) publish/retire only.
- **Recorder (D5).** `_recording_spike_index` returns `p.spikes` for the owners. `_fire_step_list` uses `spike_list!` when
  the list is owned and active, and otherwise the recorder's scratch scan. `_has_step_list` (renamed from `_has_fed_list`)
  routes sorted subsets smaller than N/8 to the list path; both paths produce ascending lists, so the output is the same.
- **Unchanged:** `Identity`, the Poisson family, `PoissonStimulusLayer` and `SpikeTimeStimulus` (fed lists, or one scan per
  connection in the dense regime). A population type that is not in the Union falls back to a new list per connection.

Why the results are bit-identical (D3): no component writes a population's `fire` between its `integrate!` and the end of
the step. Stimuli write only their own `fire`; population `plasticity!` is a no-op; connections never write `fire`. Inside
the window the shared list is therefore `findall(fire)` of the flags `integrate!` left (ascending, `_flags_to_list!`). That is
exactly the list each connection used to build for itself. If the same population appears twice in `P`, its second
`integrate!` re-publishes the list and invalidates it.

Threading (D4): publish and retire are flag writes on the stepping thread after `integrate!` returns (the Polyester
`@batch` is synchronous). Every `spike_list!` call in the package sits at the top of `forward!`/`plasticity!`/kernels, before
any `foreach_chunk`/`@batch` fork. The single scan therefore runs on the stepping thread, and workers only read the list.
The lists are identical at any `-t` and `neurons_per_thread`. The `spike_list!` docstring says it must not be called on a
shared list from inside a `foreach_chunk` body.

Scans of `fire` per step, T4 network (E 8000 / I 2000; EE EI IE II):

| mode | before | after |
|---|---|---|
| `sim!` | 4 | 2 |
| `sim!`, `:fire` of E and I recorded | 6 | 2 |
| `train!`, STDP on EE | 5 | 2 |
| `train!`, STDP on EE, `:fire` recorded | 7 | 2 |

Memory: one `Int32` vector of capacity N per owning population, instead of `4 (Npre + Npost)` bytes per synapse. For T4 this
is 16 N -> 4 N bytes: 160 kB -> 40 kB at N = 1e4, and 320 kB -> 80 kB at N = 2e4. `Base.summarysize(syn)` still counts the
shared list, so the per-synapse footprint numbers do not change. The constructor allocates 8 N bytes less per synapse.

Bit-identity against c5aac91 (`perf/pop_spikelist/out/check.md`): 20 configurations, each run at `-t 1` and at `-t 8` with
`neurons_per_thread = 128` and `SYN_MIN_OPS[] = 0`. Each run hashes (SHA1) the final state of every population and synapse,
the `:fire` record buffers and the raster. The configurations:
- balanced E/I LIF at N = 2000: rec0, rec1, sorted subset, `hdelay`, STDP `train!` (rec0, rec1, with `hdelay`), Float64;
- balanced E/I LIF at N = 10000: rec0, rec1;
- AdEx at N = 2000 (rec0, rec1);
- Tripod (NE = 400), HH (1000) and IZ (1000), each rec0 and rec1.

Result: **40/40 identical** (after = ceb16d1, whose `src/` equals 78165d3).

Benchmark (T4, K = 100, Float32, `CurrentSynapse`, Dirac 0.8 ms):
- one thread pinned with `taskset -c 7` (core 7 and its sibling 19 were 95-100 % idle at the start and end of every cell);
- host ld25-1001 (seroquel), 2026-10-11 22:42-23:05; load average 2.8-4.4 (1-minute load per process 2.7-4.6);
- 3 processes x 3 repetitions per cell, with before/after processes alternated; IF N = 1e4 rec0 had 3 extra rounds because
  its before processes disagreed;
- simulated time 12-30 s, so each timed run lasts 3.4-4.5 s; compilation and build are excluded.

ns per neuron-step: median of the process medians [min-max of the process medians]. after/before: ratio of the medians
[range of the paired-round ratios].

| cell | before | after | after/before |
|---|---|---|---|
| IF N=1e4 rec0 (6 processes) | 1.199 [1.154-1.233] | 1.139 [1.105-1.147] | 0.950 [0.921-0.994] |
| IF N=1e4 rec1 | 1.425 [1.413-1.428] | 1.208 [1.195-1.215] | 0.848 [0.839-0.855] |
| IF N=2e4 rec0 | 1.302 [1.301-1.331] | 1.222 [1.201-1.239] | 0.938 [0.923-0.939] |
| IF N=2e4 rec1 | 1.365 [1.348-1.371] | 1.256 [1.256-1.286] | 0.920 [0.916-0.954] |
| AdEx N=1e4 rec0 | 3.202 [3.190-3.274] | 3.141 [3.116-3.148] | 0.981 [0.961-0.985] |
| AdEx N=1e4 rec1 | 3.276 [3.270-3.279] | 3.160 [3.149-3.207] | 0.965 [0.961-0.981] |
| IF N=1e4 STDP `train!` rec0 | 1.822 [1.815-1.879] | 1.743 [1.725-1.757] | 0.956 [0.935-0.956] |

Recording overhead rec1/rec0 (ratio of the medians [range of the per-round ratios]); Auryn and Brian2 values from the T4
campaign (`perf/out/t4.csv`):

| | before | after | Auryn | Brian2 |
|---|---|---|---|---|
| IF N=1e4 | 1.19 [1.15-1.23] | 1.06 [1.04-1.07] | 1.08 | 0.99 |
| IF N=2e4 | 1.05 [1.03-1.05] | 1.03 [1.01-1.05] | 1.08 | 0.99 |
| AdEx N=1e4 | 1.02 [1.00-1.03] | 1.01 [1.00-1.02] | - | - |

- All 24 paired before/after processes favour after; the largest paired ratio is 0.994. Every acceptance criterion of the
  manifest is met: rec0 <= 1.00, rec1 <= 0.97 at IF N = 1e4, no cell slower, rec1/rec0 after <= before.
- The gains are larger than the manifest estimate (3.5 % rec0 / 7 % rec1). How large they are is uncertain where the spread
  is large. IF N = 1e4 rec0 before has a 6.6 % spread between processes (the repetitions inside one process agree within
  0.2-1.8 %), so its 5 % gain could be anywhere from about 1 % to 8 %. The rec1/rec0 improvement at N = 2e4 and for AdEx is
  within the round spread. The clear effects are the absolute rec1 times (-8 % to -15 % for IF) and the IF rec0 times at
  N = 2e4 (-6 %).
- Absolute values are lower than in the record_fix benchmark (1.14-1.23 here against 1.49 for IF N = 1e4 rec0). That run
  was in a different session, at a higher load, on another core, with 10 s simulated. Compare ratios, not absolute values.
- Spike counts in the timed rec1 runs are identical before/after (IF N = 1e4 2028745, N = 2e4 2026592, AdEx 1319787).
- Scripts, CSVs, README and figure: `papers/JuliaSNN_publication/validation/profilation_v1.10/perf/pop_spikelist/`
  (`scripts/pl_balanced.jl`, `run_pl.py`, `pl_tables.py`, `plot_pl.jl`; bit-identity `run_pl_check.py`, `pl_check.jl`,
  `pl_hash.jl`; `out/summary.csv`, `out/overhead.csv`, `out/time.csv`, `out/uptime.txt`, `out/check.csv`/`check.md`;
  `fig/F_pop_spikelist.png`/`.pdf`).

Files:
- `src/populations/spike_index.jl`: `owned`, `owned_spike_index`, the window functions, docstrings;
- the nine population files: one field and one docstring line each;
- `src/populations/populations.jl`: the Union, `spike_index`, publish/retire for the owners;
- `src/utils/main.jl`: publish after `integrate!`, retire at the end of the step, docstrings;
- `src/utils/record.jl`: the owner method of `_recording_spike_index`, `_fire_step_list`, `_has_step_list`;
- tests: `test/syn/population_spike_list.jl` (new, L1-L14) and `test/utils/record_spikeindex_test.jl` (T9).

### Known limitations
- An exception inside a step leaves the owned spike lists open and possibly stale: a failed step is a fatal error and the run is
  discarded, so no `try/finally` closes them (the `finally` cost 5 to 8 ns per call of the five-argument `sim!`/`train!`, a
  measurable share of the event-driven STDP benchmarks). The next step publishes the lists afresh; for debugging call
  `SNNModels._retire_all_spikes!(P)` before using `forward!`, `plasticity!` or `record!` by hand (test L10).
- The five-argument `sim!(P, C, S, dt, T)` / `train!(P, C, S, dt, T)` (one call per step) passes `dt` as is when the three
  vectors have concrete element types, and in a `Ref` (16 B per call) when they do not, where the calls are dynamic and an
  unboxed `dt` would be boxed per component; the function barrier of `_train_step!` stays (0 to 4 ns per call). Timing of the
  event-driven STDP benchmarks against the 1.10-dev campaign (395412d), 2e5 steps, median of 3: pair 200 synapses 15.2 vs 14.5 ms
  (b25eb27: 17.9), pair 1e4 20.0 vs 20.3 (24.8), event vSTDP 1e4 35.5 vs 37.7 (39.1); `final_v2/out/step_fix_check.txt`.
- A record closed by a switch (rule variable the new rule does not have) is not reopened by a later switch; re-monitoring the
  same key needs `clear_records!`/`clear_monitor!` first. `records` is shared, so the source synapse stops sampling it too.
- `compose` cannot replace a component by name (`compose(; model..., EE = new)` fails with a duplicate-key error that points to
  `change_plasticity`); use `change_plasticity(model, :EE; LTP, STP)`, or `compose(; pop = model.pop, syn = merge(model.syn,
  (; EE = new)), stim = model.stim, time = model.time)`, or a `Vector{AbstractConnection}`.
- `iSTDPPotential` and the dense vSTDP are not converted (the event-driven vSTDP keeps the postsynaptic filters per step, O(Npost)): their postsynaptic filters follow the membrane potential at every step
  (class C, see "Lazy classification of the plasticity rules"); `SpikeRateSynapse` keeps its dense scans. The clock-driven
  `MarkramSTPParameterTimestep` (lazy-able with the Euler factor per elapsed step, but that needs a dt-aware elapsed step
  count and it is the dense reference of the event rule) and the traces of `ActivityDependentTurnover` (read once per
  turnover period) are lazy-able and not converted.
- The sparse hand-off needs every writer of an `Identity` to be a stimulus that logs its writes (`SpikeTimeStimulus`,
  `PoissonStimulus` with a built-in parameter, `PoissonStimulusLayer`) (any other writer, such as a connection or
  a stimulus without a log, makes the population use the dense update; results are the same, only slower). Cases that still scan `fire` once per connection and step: the Poisson family in the dense regime (`fed = false`) and an
  `Identity` that is not driven by list-aware stimuli. The integrate-and-fire family scans once per step (see "One spike list
  per population").
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
- Optional: owned lists for the Poisson family in the dense regime and for an undriven `Identity`. These still scan once per
  connection.
- The delay queue entry of `claude/performance.md` ("Remaining known performance issues"). (Done: the SpikingNeuralNetworks
  docs of `change_plasticity!`/`update_plasticity!`, the `SpikingSynapse` docstring of `pre_spikes`/`post_spikes`, the release
  notes in `CHANGELOG.md` and `CHANGELOG_2.0.0.md`, Project.toml at 2.0.0.)

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
