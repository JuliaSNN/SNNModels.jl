# Fixes of the documentation-sweep bugs: report

Branches `fix/sweep-bugs` created from `docs/sweep` in the four local worktrees
(`scratchpad/docsweep/{SNNModels.jl,SNNUtils.jl,SNNPlots.jl,SpikingNeuralNetworks.jl}`).
Nothing pushed; no other branch touched. "Unfixed" = the `docs/sweep` commits (code identical to
SNNModels 1.8.4, SNNPlots 0.2.10, SNNUtils 0.2.9, SpikingNeuralNetworks 1.2.1). Plan:
`claude/plan_sweep_bugs.md`.

Validation scripts, data (JLD2), summaries and figures:
`papers/JuliaSNN_publication/validation/sweep_bugs/` (umbrella repo, not committed). `run_all.sh`
regenerates everything; `data/*_summary.txt` hold all numbers quoted below.

## Tripod: the gl question

Evidence (decision: fix, the published model uses gl):
- Quaresima et al. (2023) J Physiol 601:3265 (bioRxiv 2022.09.05.506197), Eq. 1:
  `C dV/dt = -g_m [(V - V_r) + ΔT exp((V - V_T)/ΔT)] - Σ g_k (V - E_k) - w + I_d`. g_m multiplies
  the exponential term. (The printed bracket has a sign typo: with `+` the term would be
  hyperpolarising; the published code has the correct sign.)
- Published code (github.com/aquaresima/tripod_neuron, `TripodNeuron.jl/src/simulator/equations.jl:110-113`):
  `C⁻ * (gl * (-v + Er + ΔT * exp(ΔT⁻ * (v - θ))) - w - syn_current - axial)`.
- SNNModels AdEx (adex.jl: `dt/τm * [-(v-El) + ΔT exp(...) - R w + R I]`, gl factored out) and the
  unloaded multipod.jl both include gl.
- History: the gl-less form `gl * (-v_s + Δv dt + Er) + ΔT * exp64(...)` is already in the first
  SNNModels commit (4cc5a5c, port from SpikingNeuralNetworks); it is dimensionally inconsistent
  (mV added to pA) and is not in the published model.

Final somatic equation (docstrings of `Tripod`/`BallAndStick` and catalogue page):
`C dV_s/dt = gl (El - V_s) + gl ΔT exp((V_s - θ)/ΔT) - w_s - I_syn,s - Σ_k gax_k (V_s - V_d,k) + I`.

Note: the published simulator detects spikes at `V_s >= V_T` (-50.4 mV); SNNModels uses -10 mV on
the predicted potential. Kept (behaviour-preserving default) and exposed as `Vspike`
(`Vspike = adex.Vt` reproduces the published rule).

## Per-bug summary (SNNModels unless stated)

| # | Bug | Root cause | Fix | Commit | Test | Changes numbers | Measured effect |
|--:|:--|:--|:--|:--|:--|:--|:--|
| 1 | Tripod/BAS Heun w stage | `v_s+Δv`, `w_s+Δv` without dt; stage-1 w used overwritten Δv; w and synaptic currents not at predicted state | all derivatives at `x + dt k1` (`v_pred`) | b042299 | multicompartment_numerics.jl (one Heun step vs hand reference; dt convergence) | yes | spike times shift up to 7 ms / 500 ms (gl-only vs fixed); see Tripod section |
| 2 | Tripod/BAS exp term lacks gl | port error | `gl ΔT exp` | 1e5c478 | Heun reference with gl | yes | rheobase -29 %, rates x1.5-2, network E rate x2 |
| 3 | spike threshold literal -10 mV | literal | field `Vspike = -10mV` | 7dc284a | default and custom Vspike | no | - |
| 4 | dendritic leak used somatic El | literal | `d.El`, set from `adex.El` by default (`create_dendrite(N, l; El)`) | 7dc284a | dendrite relaxes to its El | no (default) | - |
| 5 | BAS ignored Is/Id | commented out | applied | 7dc284a | current step depolarises | only if currents used | BAS f-I now possible |
| 6 | invalid compartment -> UndefVarError | conditional assignment | ArgumentError | 7dc284a | @test_throws | no | - |
| 7 | DeltaSynapse in Tripod -> MethodError | no 5-arg method | ArgumentError with explanation | 7dc284a | @test_throws | no | - |
| 8 | Multipod | unloaded, depends on removed API; sign error `(-v + Δv dt + El)` | stays unloaded (re-enabling = port to current API); sign fixed; header note; its export line is inert, nothing exported | 7dc284a | - | no | - |
| 9 | Tripod/BAS refractory counter (new, found by the dt study) | `round((up+τabs)/dt)` vs `tabs > τabs/dt`: up = τabs = 0.1 ms at dt = 0.125 skipped the reset | `n_up = max(1, round(up/dt))`, `n_abs = max(1, round(τabs/dt))` | cb10b33 | rate at dt 0.1 vs 0.125 | only when up/τabs not multiples of dt | paper parameters at dt = 0.125: 1126 Hz -> 32 Hz at 1500 pA |
| 10 | train! fails for IZ/HH/ML and without connections | param types not `<: AbstractPopulationParameter`; EmptySynapse lacked update_traces!/plasticity! | subtype; no-op methods | c25d688 | sweep_fixes.jl | no | - |
| 11 | FLSynapse/PINning cannot run; FLSparse ctor | param types not `<: AbstractConnectionParameter`; `2rand(N)-1`; colptr | subtype, broadcast, unpack, p check | 59bb3fd | FORCE learning error drops (0.019 -> 0.0012 in a 1 s run) | enables | FORCE tutorial runs again |
| 12 | ML/ExtendedIF/WilsonCowan cannot receive connections | no synaptic_target | methods with symbol mapping | 32fd7e2 | conductances driven | enables | - |
| 13 | HH/ML several flags per AP; ML w-nullcline sign | level tests; typo | upward crossing; `n_ss` | 7bbd80e | 1 flag per AP | yes | HH: 60.3 flags per AP at dt 0.01 (4657 vs 77 Hz at 500 pA); ML: 1596 flags for 3 APs |
| 14 | Rate g never reset; RateSynapse p=0 NaN; SpikeRateSynapse | fill commented out; Inf scale; missing fields | reset g in Rate/WilsonCowan; ArgumentError for p outside (0,1]; delta input W/dt, name/targets, no-op plasticity | a496eb8, a8cc64e (docs) | g = W r; x jumps by W | yes | mean x after 220 ms: 1137 -> ~0 (unfixed saturated r in 100 % of units) |
| 15 | HetRec trace without dt; soma per-dendrite sequential | missing dt | dt; one Euler step of the documented ODE | 924ad1c | dt invariance | yes | baseline at 300 ms: 4.35 (dt 0.125) vs 4.64 (dt 0.05) -> 2.42 vs 2.42 |
| 16 | Confavreux synapse input w dt | dt inside the Euler term | increment by w | 26ec741 | jump = w at two dt | yes (x 1/dt) | gAMPA after a w=1 spike: 0.125/0.05 -> 1.0/1.0; PSP 0.029/0.012 -> 0.228/0.229 mV |
| 17 | vSTDP LTP without dt; x jumps dt/τx | the two factors cancel (weights dt-independent) but x ∝ dt | Clopath form (x += 1/τx, LTP × dt) | 46f0246 | dW equal at two dt; x | x only | dW 0.4296/0.4303/0.4306 at dt 0.125/0.0625/0.025 (unfixed): dt-independent already; x peak 0.0083/0.0042/0.0017 -> 0.0667 constant |
| 18 | vSTDP/iSTDPPotential traces start at 0 mV | zero init | set to v_post at first call | 46f0246 | no LTD at rest | yes (first τ) | dW for a pre spike with post at rest: -0.0557 -> 0; silent AdEx target 20 Hz, 50 ms: -0.0183 -> -0.0007 |
| 19 | AggregateScaling per-step constants, units, sum, N, Wmin | missing dt; spike count vs rate; μ formula | dt, rate estimate, μ = (WT - n Wmin)/Wt, N set, Wmin 0.5pF, WT under train! only | cadce7c | y ≈ rate, dt invariance | yes | post 100 Hz: y = 1.81/1.15 (dt 0.125/0.05, spike counts) -> 0.105/0.105 /ms; WT(2 s) -9.7e8/-3.9e8 -> -7.8/-7.8 |
| 20 | AdditiveNorm offset, default param, τ<dt | wrong formula; unconstructible default | (W0-W1)/n; required param; max(1, ...) | 974e5d5 | sums restored | yes | weights x1.5: sum 25.6 -> 19.2 (W0 = 19.2) |
| 21 | Turnover: RandomTurnover train!, defaults, p_new arity, negative weights, sample error | missing method/defaults | method, required fields, `(post, pre) -> 1.0`, truncated normal, cap | 974e5d5 | train! runs, W > 0 | enables | - |
| 22 | connect!/update_sparse_matrix! did not resize/reorder ρ, delays; shrinking dims; set_plasticity!/has_plasticity FieldError; unknown conn keys silent; dist type rejected | storage rebuild ignored per-synapse arrays | remap per-synapse state; keep dims; SpikingSynapse methods; warning; accept types | 974e5d5 | lengths, values | fixes out-of-bounds reads | - |
| 23 | BalancedStimulus unusable | randcache, N draws, neuron 1, sym for E and I, BSParam | rewrite per the documented equations | 2cfc30c | per-neuron rates | enables | default: UndefVarError -> exc 1.29/ms (relaxing from 1.5 r0), inh 0.507/ms (kIE r0 = 0.5), 50/50 neurons, separate buffers |
| 24 | stimuli: PoissonLayer active and defaults, next_neuron, empty spike lists, update_spikes! sort, neurons(group), group comp/param types, set_variable! scalars | various | fixed; Poisson params mutable | 36757df, a7f264b | sweep_fixes.jl | partly | - |
| 25 | interpolated_record axis; sampling period; :fire indices; τ for :fire; r_v | record times of first record! call; floor; ignored indices | times of first/last sample; round; indices recorded with original ids | 67ce2e5 | exact axes | time axis only | :v at 10 Hz from 2 s: axis 2000.125:105.26:4000 -> 2100:100:4000; 10.5 ms at 1 kHz: 0:1.05:10.5 -> 0:1:10. The old floor made 10 Hz sample every 99.875 ms |
| 26 | analysis: correlograms, covariance density, average_firing_rate(pops), st_order, FanoFactor default, KDE, sample_inputs dead check | missing kwargs / undefined names | fixed | b655c21 | sweep_fixes.jl | enables | - |
| 27 | @update/@update! single forms; @snn_kw outside SNNModels | undefined variable / unescaped; unqualified sentinel | fixed | 9ff2e45 | sweep_fixes.jl | no | - |
| 28 | IO/graph/util/spatial | see commit | SNNload kwargs, load_data, load_or_run folder, data2model layout, write_config precedence, read_folder filter, filter_edge_props, find_key_graph, dead code, periodic_distance L1 -> Euclidean, gaussian i == j, windows | 57fbc1d | sweep_fixes.jl | periodic_distance (vector grid), spatial_activity windows | (0.1, 0.1) from origin: 0.2 -> 0.141 |
| 29 | Time(time) InexactError; non-const NMDA globals | Int32 of a float; globals | `Time(time; dt)` with round; const Float32 | 6f3016a | sweep_fixes.jl | no | - |
| 30 | undefined exports (35) | stale exports | removed: 19 SNNModels (incl. BSParam), 11 SNNPlots, 1 SNNUtils (pca), 4 SNN; `raster!` now exported by SNNPlots; `make_copy` removed, SNN exports `modelcopy` instead (make_copy never existed in SNNModels history; docs/tutorial updated) | 3a6efe9, SNNPlots 6a7dfdb/676700f, SNNUtils 9fc1f41, SNN aeaa3a4 | tests: every export defined (all 4 packages) | no | - |
| 31 | SNNPlots stdp_test/stdp_kernel shifted kernel | post neuron driven by the measured synapse | separate pre/post Identity, synapse transmits into a dummy buffer | SNNPlots 676700f | antisymmetric Gerstner kernel | yes | ΔT = -10 ms: +3.9e-5 -> -6.07e-5 (pair rule -6.07e-5); +10 ms: +1.6e-4 -> +6.07e-5 |
| 32 | SNNPlots raster/vecplot | ms label, ylims axis, names, kwargs, Plots calls, `SNN` undefined, factor matrix | fixed | SNNPlots 6a7dfdb | SNNPlots test suite (new) | plots only | - |
| 33 | SNNUtils always-throwing functions and defects | removed APIs; residual_current self defaults and leak sign | ported/fixed (see commits) | SNNUtils 239da0a, 9fc1f41, 3b74d34, 4538bac | SNNUtils test suite (new) | residual_current sign | 1.8.4: all four functions threw; with the old leak sign the residual at the compute_kei ratio was 34.3 (now 0.0) |

Docs/docstrings: every commit updates the docstrings of its symbols; the SpikingNeuralNetworks
pages are updated in 9fc948b and 7d3d4e3 (release notes with the behaviour changes). A lost
backslash in three LaTeX `\right` commands (spatial.jl, compute_kei.jl) was repaired.

## Tripod / BallAndStick comparison in detail

Variants: unfixed (1.8.4); dt-only (Heun fix, commit b042299); gl-only (1.8.4 + gl, scratch);
all fixes (branch head). BallAndStick protocols with somatic current use scratch variants in
which only Is/Id is applied (the 1.8.4 BAS ignores currents). Configurations: "default"
(AdExParameter(), PostSpike(), dendrites 300 um) and "paper" (parameter set of the SNNModels Tripod
tests: Vr -55.6, Vt -50.4, At 10 mV, τA 30 ms, up = τabs = 0.1 ms, dendrites 160/200 um).
dt = 0.1 ms unless stated.

Single neuron (`data/mc_summary.txt`; unfixed -> dt-only -> gl-only -> all fixes):

| | rheobase (pA) | rate 1000 pA | rate 1500 pA | rate 2500 pA | first ISI 1500 pA (ms) | adaptation index | bAP in d1 (mV) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Tripod default | 1134/1134/814/814 | 0/0/14/14 | 24/25/44/44 | 70/71/88/89 | 18.65/18.60/10.90/10.90 | 2.47/2.45/2.31/2.29 | 32.4/32.4/37.3/37.3 |
| Tripod paper | 1097/1096/782/782 | 0/0/12/12 | 19/19/32/32 | 55/55/66/67 | 23.52/23.45/10.80/10.77 | 2.39/2.39/3.09/3.08 | 10.6/10.6/12.0/12.0 |
| BAS default | 1107/1107/794/794 | 0/0/15/15 | 26/26/45/45 | 72/73/90/90 | 17.38/17.33/10.58/10.58 | 2.46/2.43/2.31/2.29 | 32.4/32.4/37.3/37.3 |
| BAS paper | 1082/1081/771/771 | 0/0/13/13 | 20/20/33/33 | 56/56/67/68 | 22.25/22.15/10.47/10.45 | 2.44/2.44/3.13/3.12 | 7.2/7.2/8.1/8.1 |

- The gl fix dominates: rheobase -28 to -29 %, rates at 1500 pA +65 to +85 %, first ISI halved,
  first spike earlier (Tripod default 1500 pA: 14.95 -> 8.70 ms), adaptation current about 1.5
  times larger in steady state (more spikes), ISI adaptation index changes from 2.47 to 2.29
  (default) and 2.39 to 3.08 (paper). The somatic spike shape is clamped (AP_membrane), so the
  waveform differs mainly in the upstroke and in the timing (figure mc_traces.png).
- Subthreshold dendritic integration is unchanged: single-input EPSPs (weights 0.5-64) differ by
  < 0.4 mV in the dendrite and < 0.04 mV at the soma; somato-dendritic attenuation ratio for w=1:
  6.62 (Tripod default), 4.47 (paper), 6.33/5.05 (BAS), identical in all variants; NMDA
  plateau duration above -50 mV identical (4.9 ms at w=32, 31-32 ms at w=64 in the default
  Tripod). The Heun fix changes dendritic peaks by 0.1-0.4 mV at large weights.
- The Heun fix alone changes spike times by up to 1.5 ms in the first 200 ms at 1500 pA
  (Tripod default: 145.95 vs 144.45 ms for the 6th spike) and, combined with gl, by 2.5-7 ms over
  500 ms (gl-only vs all fixes at dt = 0.0125: 7.35 ms Tripod default, 2.52 paper, 7.14 BAS default,
  2.59 BAS paper).

dt dependence (500 ms steps at 800-2500 pA, figure mc_dt.png): max spike-time change against
the same code at dt = 0.0125 ms, for dt = 0.025/0.05/0.1:
- Tripod default: unfixed 4.62/5.38/1.48 ms (non-monotone: the 1.8.4 scheme does not converge,
  the error stays at the ms level as dt shrinks); all fixes 0.21/0.56/1.46 ms (first-order
  convergence, limited by spike detection on the grid).
- Other configurations: unfixed 0.14-0.81 ms, fixed 0.09-1.9 ms. Against the fixed code at
  dt = 0.0125, the unfixed code is off by 240-340 ms (spikes added) at every dt, gl-only by
  2.5-7 ms at every dt (the 1.8.4 Heun error does not vanish with dt), the fixed code by
  0.1-2 ms.
- Rheobase does not depend on dt (identical at dt 0.05/0.1/0.125 in every variant).
- dt = 0.125 ms with up = τabs = 0.1 ms ("paper"): unfixed, dt-only and gl-only fire at
  1104-1128 Hz at 1500 pA (refractory rounding bug #9); all fixes: 32-33 Hz, as at smaller dt.

Network (`data/net_summary.txt`, figure network.png; 400 Tripod E, 50+50 IF, 5 s, sim!, seed
1234; second seed 4321 for unfixed and fixed):

| dt = 0.1 ms | E (Hz) | I1 (Hz) | I2 (Hz) | CV E | Fano E (5 ms) | active E | g_exc/g_inh |
|:--|:--|:--|:--|:--|:--|:--|:--|
| unfixed | 12.6 (13.0) | 50.3 (53.0) | 25.7 (26.5) | 0.23 (0.20) | 55 (62) | 0.74 (0.79) | 187 (179) |
| dt-only | 13.7 | 55.2 | 27.9 | 0.23 | 53 | 0.74 | 172 |
| gl-only | 23.9 | 112.2 | 45.0 | 0.29 | 58 | 0.93 | 96 |
| all fixes | 24.8 (25.4) | 118.0 (121.9) | 46.7 (46.1) | 0.27 (0.26) | 59 (64) | 0.92 (0.93) | 92 (91) |

(values in parentheses: second seed). The fix doubles all population rates (seed-to-seed
variation 0.5 Hz for E), raises the fraction of active E cells from 0.74 to 0.92 and halves the
excitatory/inhibitory conductance ratio of the E cells (more inhibition from the faster
interneurons). gl accounts for about 90 % of the change, the Heun fix for the rest (+1.1 Hz E,
+5 Hz I1). At dt = 0.125 all rates are 5-10 % lower than at dt = 0.1 in every variant (Poisson
input and synaptic discretisation), so this dt effect is not a Tripod-specific artefact.

### Assessment for published Tripod results

- Results produced with the original TripodNeuron.jl code (the 2023 J Physiol paper and its
  companions) are not affected: that code includes gl in the exponential term and uses forward
  Euler with its own spike rule.
- Results produced with SpikingNeuralNetworks.jl/SNNModels Tripod or BallAndStick populations
  (every version since the port, 4cc5a5c; SNNModels up to 1.8.4) used a soma whose
  spike-initiation current was 40 times too small: neurons need about 40 % more current (or
  input) to fire, fire at roughly half the rate for the same drive, and adapt differently.
  Network rates in such models were tuned on this behaviour; with the fixed code the same
  parameters give about twice the activity, so tuned models need recalibration (input weights or
  rates) rather than reinterpretation. Qualitative dendritic results (EPSP attenuation, NMDA
  plateaus, dendritic nonlinearity, bAP shape) are essentially unchanged; anything that depends on
  the somatic input-output function (rates, rheobase, f-I gain, E/I balance, timing) changes.
- Models run with `up` or `τabs` not a multiple of `dt` (e.g. 0.1 ms at the default
  dt = 0.125 ms, as in the SNNModels Tripod tests and possibly in user configurations) produced
  kHz bursting after every spike in 1.8.4; such results are invalid and must be rerun.
- The Heun fix is a numerical correction (ms-level spike-time shifts, < 10 % rate changes); its
  main effect is that results now converge with dt.

## Figures

All in `papers/JuliaSNN_publication/validation/sweep_bugs/figures/`:
- `mc_fI.png` (plot_multicompartment.jl): f-I curves and rheobases, 4 variants x Tripod/BAS x
  2 configurations.
- `mc_traces.png`: soma/dendrite voltage, adaptation current, ISI sequence at 1500 pA.
- `mc_epsp.png`: EPSP amplitudes vs weight (soma, dendrite), NMDA plateau traces.
- `mc_dt.png`: spike-time error vs dt against the fixed code and against the same code.
- `network.png` (plot_network.jl): network rates, CV, g_exc/g_inh at two dt and two seeds,
  rasters unfixed vs fixed.
- `other_fixes.png` (plot_other.jl): HH spike flags, stdp_kernel, AggregateScaling, vSTDP trace,
  Confavreux PSP, recording time axis.

## Tests and docs

- SNNModels `Pkg.test()` (Julia 1.12.6): baseline `docs/sweep` 1261/1261; final 1402/1402.
  New test files: `test/pop/multicompartment_numerics.jl`, `test/sim/sweep_fixes.jl`; extended
  `test/pop/hetrec.jl`, `test/stim/balanced.jl`, `test/syn/istdp_kernel.jl` (reference updated for
  the trace initialisation), `test/ctors.jl`.
- SNNUtils, SNNPlots, SpikingNeuralNetworks: new test suites (they had none), run with
  `Pkg.test` from an environment that dev-links the four worktrees: 17/17, 14/14, 3/3.
- Docs: `docs/make.jl` builds; only the three environment warnings of the baseline (no git
  remote, deployment skipped). Page examples: 88 OK, 0 FAIL, 7 skipped (was 86/0/8: the FORCE
  tutorial runs again; a BalancedStimulus example was added). Docstring examples: 130 OK, 0 FAIL.

## Not fixed (and why)

- Design or physiology choices, documented: STP `u^-`/`u^+` order and the first-spike efficacy
  of `MarkramSTPParameterTimestep`; `STDPMexicanHat` same-step double update; the Poisson rate
  rule `R(β η, 1)` (also in InhomogeneousPoisson and BalancedStimulus); HH/IZ random initial
  conductances (COBA benchmark initialisation); clamps of the dendritic synaptic currents
  (±1500 / ±1000 pA); `CurrentNoise` amplitude per step; DeltaSynapse voltage jump `R w dt/τm`.
- Features rather than bugs: IF with heterogeneous (vector) parameters; ExtendedIF unused
  fields; InhomogeneousPoisson using the global RNG for the spike draw; `sample_inputs` ignoring
  `N` for count matrices.
- Multipod: not loaded (needs a port to the current API).
- SNNUtils model files other than stp_het.jl remain unloaded (duarte2019, lkd2014,
  dendrite_STM, mongillo use removed types; porting them is a rewrite).
- The STDP_structured `@turbo` scatter loops were not tested against a plain loop.
- `raster_firing` (SNNPlots vecplot.jl, not exported) still uses the Plots.jl API.
- SNNUtils `step_input` with point neurons needs the SNNModels fix a7f264b (new SNNModels
  release); the compat bounds of SNNUtils/SNNPlots/SpikingNeuralNetworks were not changed.
- SNNPlots commit 676700f contains the full new test file, whose raster/vecplot tests pass only
  with the next commit 6a7dfdb.
