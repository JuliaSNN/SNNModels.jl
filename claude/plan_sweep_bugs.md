# Plan: fix the code bugs found by the documentation sweep

Branch `fix/sweep-bugs` from `docs/sweep` in SNNModels, SNNUtils, SNNPlots, SpikingNeuralNetworks
(local worktrees, never pushed). Source of the bug list: `claude/doc_sweep_report.md`, section
"Code bugs noticed, not fixed" and the top-12 list. One commit per bug or tightly related group;
each commit carries the fix, the docstring/docs text of the symbol, and a regression test.
Unfixed reference = `docs/sweep` commit, checked out as detached worktrees in
`scratchpad/unfixed/` and loaded through `scratchpad/envs/unfixed`.

Baseline: SNNModels `Pkg.test()` on `docs/sweep`: 1261/1261 pass (Julia 1.12.6).

## Evidence for the Tripod gl question (decided: fix)

- Published equation (Quaresima et al. 2023, J Physiol 601:3265, bioRxiv 2022.09.05.506197,
  Eq. 1): `C dV/dt = -g_m [(V - V_r) + ΔT exp((V - V_T)/ΔT)] - Σ g_k (V - E_k) - w + I_d`,
  i.e. the exponential term is multiplied by the leak conductance (the sign inside the bracket in
  the printed equation is a typo; the code is unambiguous).
- Published code (github.com/aquaresima/tripod_neuron, `TripodNeuron.jl/src/simulator/equations.jl:110-113`):
  `C⁻ * (gl * (-v + Er + ΔT * exp(ΔT⁻ * (v - θ))) - w - syn - axial)`.
- SNNModels `AdEx` (`adex.jl`, gl factored into τm/R) and the unloaded `multipod.jl` both apply gl.
- The `gl`-less form entered with the SpikingNeuralNetworks.jl port (first SNNModels commit
  `4cc5a5c`, `+ gl * (-v_s + Δv*dt + Er) + ΔT * exp64(...)`), is dimensionally inconsistent
  (mV added to pA) and was never in the published model.

## Bugs, root cause, fix, behaviour change, test, comparison

| # | Bug | Root cause | Fix | Changes numbers | Test | Comparison |
|--:|:--|:--|:--|:--|:--|:--|
| 1 | Tripod/BallAndStick Heun stage of `w` | predicted state `w + Δw` and `v + Δv` lack `dt`; stage-1 `w` derivative reads the freshly overwritten `Δv[:,1]`; `w` not predicted in the V equation; synaptic currents not at predicted V | compute the full predicted state `x + dt k1` once, evaluate all derivatives (incl. synaptic currents) there | yes | dt-convergence test: spike times at dt = 0.05/0.025 agree; one-step check against hand-computed Heun | full Tripod/BAS suite, dt study |
| 2 | Tripod/BallAndStick exponential term lacks `gl` | port error (see evidence) | `gl ΔT exp((V-θ)/ΔT)` | yes | derivative at V = θ equals `gl ΔT / C` contribution | full suite, per-fix decomposition |
| 3 | Spike detection hard-coded at -10 mV | literal in code | population field `Vspike = -10mV` (Tripod, BallAndStick) | no (default) | custom `Vspike` changes spike count; default identical | none |
| 4 | Dendritic leak uses somatic `El`; `Dendrite.El` unused | literal `El` | use `d.El[i]` (default -70.6 mV = adex default) | no with defaults | dendrite relaxes to `d.El` | none |
| 5 | BallAndStick ignores `Is`/`Id` | line commented out | add `Is` to soma, `Id` to dendrite | no (zero default) | current step depolarises | f-I curve uses it |
| 6 | `synaptic_target` of dendritic models: invalid target gives `UndefVarError: v_post` | conditional assignment | explicit `ArgumentError` | no | `@test_throws` | none |
| 7 | `DeltaSynapse` with Tripod fails at first step with MethodError | no 5-arg method | clear `ArgumentError` at construction of Tripod/BAS | no | `@test_throws` | none |
| 8 | Multipod | file not loaded, depends on removed types; `(-v + Δv dt + El)` sign error | keep unloaded (re-enabling is a rewrite, out of scope); fix the sign in the file; header note | no (not loaded) | none | none |
| 9 | `train!` fails for IZ, HH, MorrisLecar and without connections | parameter types not `<: AbstractPopulationParameter`; `EmptySynapse` has no `update_traces!`/`plasticity!` | subtype; add no-op methods | no (enables code paths) | `train!` runs | none |
| 10 | FLSynapse/PINningSynapse cannot run; FLSparseSynapse constructor | parameter types not `<: AbstractConnectionParameter`; `2rand(N) - 1`; `colptr` not unpacked | subtype, broadcasting, unpack | no (enables) | `sim!`/`train!` FORCE run, error decreases | none |
| 11 | MorrisLecar, ExtendedIF, WilsonCowan cannot receive connections | no `synaptic_target` | add methods | no (enables) | connection built and drives target | none |
| 12 | HH/MorrisLecar flag several spikes per AP | level test | upward crossing (`v_prev <= θ < v`) | yes | 1 flag per AP | spikes per AP, f-I |
| 13 | `MorrisLecar_w_nullcline` sign; unreachable returns | typo | `n_ss(v)` | yes (helper only) | nullcline value | none |
| 14 | `Rate.g` never reset under `RateSynapse` | `fill!` commented out | reset `g` in `Rate` after integration | yes | g equals instantaneous input | small check |
| 15 | `RateSynapse(p = 0)` NaN weights; `SpikeRateSynapse` `rJ`, `name`, `targets` | `μ/√(p N)` with p = 0; missing fields | dense normal weights scaled by √N when p = 0 ... (see report); add fields | yes for p = 0 only | no NaN; train! runs | none |
| 16 | HetRec trace lacks `dt`; soma leak applied once per dendrite | missing factor; leak inside loop | `dt/τrate`; leak once, inputs summed | yes | dt invariance at two dt | HetRec at two dt |
| 17 | Confavreux2025Synapse input enters as `dt w` | `dt *` | increment by `w` | yes (x 1/dt) | jump equals w | two dt |
| 18 | vSTDP LTP without `dt`, `x` jumps by `dt/τx` | the two factors cancel (rule is dt-independent) but `x` itself scales with dt | Clopath form: `x += 1/τx` at spikes, `ΔW = dt A_LTP ...`; weights identical, `x` dt-independent | weights no, `x` yes | weights equal at two dt; x jump | two dt |
| 19 | vSTDP / iSTDPPotential traces start at 0 mV | zero init | initialise traces to `v_post` at the first plasticity step | yes (first tens of ms) | no spurious LTD on a silent target | small check |
| 20 | AggregateScaling per-step time constants, units, `N = 0`, `Wmin` default | missing `dt`, spike count vs Hz | `dt/τ`, rate trace in Hz units, pass `N`, `Wmin = 0.5pF` | yes | dt invariance | two dt |
| 21 | AdditiveNorm offset; default `MultiplicativeNorm()`; τ < dt division by zero | wrong formula; missing default | `(W0 - W1)/n_i`; default τ; `max(1, ...)` | yes for AdditiveNorm | row sum restored | none |
| 22 | Turnover: RandomTurnover has no `plasticity!`; defaults (`RandomTurnover(0)`, `SpikingSynapse()`, `p_new` arity); negative new weights; `sample` failure | missing methods/defaults | add method, fix defaults, `abs`, cap sample size | yes (enables) | train! runs | none |
| 23 | BalancedStimulus unusable | undefined `randcache`, `i = 1`, rate x N, `BSParam`, `sym` for both e and i | rewrite `stimulate!` per documented equations | yes (enables) | rates match the parameter | small check |
| 24 | Stimuli: PoissonLayer default `param`, `active` ignored; `next_neuron`; `shift_spikes!`/`update_spikes!` empty; `neurons(::StimulusGroup)`; group `comp` keyword; `set_variable!` scalar | various | fix each | partly | per item | none |
| 25 | `interpolated_record` time axis misaligned when monitoring starts after 0 / non-multiple durations | `range(start, end, n)` | sample times from recorded steps | yes (time axis) | sample times exact | alignment figure |
| 26 | `monitor!(obj, [(:fire, idx)])` ignores idx; `interpolated_record(:fire, τ)` ignores τ; `r_v` undefined | branches | fix | yes (selection) | per item | none |
| 27 | `set_plasticity!`/`has_plasticity` FieldError; `connect!`/`update_sparse_matrix!` do not resize ρ/delays/plasticity (out of bounds); shrinking matrix; `sparse_matrix` swallows unknown keys; `NoPlasticityVariables` | various | fix, resize state, keep dims, warn | no for existing paths | per item | none |
| 28 | Analysis: cross-correlogram, covariance density, `average_firing_rate(pops)`, `st_order`, `FanoFactor` default, `sample_inputs`, `KDE`, dead check | missing kwargs / undefined names | fix | enables | per item | none |
| 29 | IO/graph/util/spatial/macros: `SNNload` varargs, `load_data`, `load_or_run`, `data2model`, precedence, `read_folder`, `filter_edge_props`, `find_key_graph`, `periodic_distance` L1, gaussian rule `i == j`, time windows, `@update`/`@update!` single forms, `@snn_kw` `KwStrSentinel`, `compose` dead code | various | fix | partly (`periodic_distance`) | per item | none |
| 30 | 36 exported-but-undefined names | stale exports | remove; `raster!` exported by SNNPlots; `make_copy`: removed (use `modelcopy`), docs updated | no | `isdefined` for every export | none |
| 31 | SNNPlots `stdp_test`/`stdp_kernel` shifted kernel | post neuron driven by the measured synapse | post spike driven by its own `SpikeTimeStimulus`, measured synapse weight 0 effect isolated | yes | kernel sign for Gerstner | kernel figure |
| 32 | SNNPlots raster/vecplot defects (ms label, `ylims!` axis, names, labels, Plots calls, `SNN` undefined, `factor(neurons, :)`) | various | fix | plotting only | smoke tests | none |
| 33 | SNNUtils always-throwing functions (`compute_kei`, `get_model`, `residual_current`, `optimal_kei`, `sym_features`, `store_activity_data`, `MultinomialLogisticRegression`) and other defects (`score_spikes`, `trial_average`, `merge_intervals`, `step_input` default, bioseq paths/labels, `@show`, stale exports, `dist = Normal`) | removed APIs | port to SNNModels 1.8.4 API or fix | enables | per item | none |

Not changed (design or physiology choices, documented, listed in the report): STP `u^-`/`u^+`
order, `MarkramSTPParameterTimestep` first-spike efficacy, `STDPMexicanHat` same-step double
update, Poisson `R(noise β, 1)` rate rule, HH/IZ initial conductances (COBA benchmark), clamp
limits of the dendritic synaptic currents (±1500/±1000 pA), `CurrentNoise` per-step amplitude,
IF heterogeneous parameters (feature), Multipod (not loaded).

## Comparisons (fixed vs unfixed)

Scripts, JLD2 data and PNG figures in
`papers/JuliaSNN_publication/validation/sweep_bugs/` of the umbrella repo (not committed).
Each measurement script takes the environment (fixed / unfixed / variant) on the command line and
writes one JLD2 file; one plotting script per figure reads the JLD2 files and uses
`SNNPlots.@makie_default`. Tripod variants: `unfixed`, `dt-only` (commit 1), `gl-only`
(commit 2 cherry-picked on `docs/sweep`, scratch worktree), `fixed` (both).

- Tripod and BallAndStick: f-I curve, rheobase, spike waveform and spike timing, adaptation
  current and ISI adaptation, dendritic EPSP attenuation and backpropagation, NMDA plateau,
  dt = 0.05/0.1/0.125 ms convergence, network check (SNNUtils/SpikingNeuralNetworks Tripod
  network, same seeds: rates, CV, E/I).
- HH and MorrisLecar spikes per AP and f-I; vSTDP, AggregateScaling, HetRec, Confavreux at two
  dt; `interpolated_record` alignment; `stdp_kernel` shift.
