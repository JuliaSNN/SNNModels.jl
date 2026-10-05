# Documentation sweep report (JuliaSNN stack, SNNModels 1.8.4)

Branch `docs/sweep` in four local worktrees, created from `dev` (SNNModels, SNNPlots,
SpikingNeuralNetworks) and `main` (SNNUtils). Nothing was pushed and no other branch was touched.
Every change is documentation or comments. For each of the 103 changed source files, the parsed
code with docstrings, comments and line numbers removed is identical to the base branch. The check
script is `scratchpad/codeid.jl`.

## Commits

| Repository | Base | Commits on `docs/sweep` |
|:--|:--|:--|
| SNNModels.jl | `dev` `bea4269` | `cfa3745` docstrings for every public symbol (77 files); the next commit adds `claude/doc_inventory.md` and this report |
| SNNUtils.jl | `main` `8910518` | `049e8ae` docstrings for every exported symbol, status comments in `src/models` (20 files) |
| SNNPlots.jl | `dev` `861b97e` | `80d4cdd` docstrings for the plotting API (6 files) |
| SpikingNeuralNetworks.jl | `dev` `1633ecd` | `cf8fc76` model catalogue, rewritten user pages, `make.jl`, module docstring (25 files) |

## What was done

- **Inventory.** I listed every exported name of SNNModels, SNNUtils, SNNPlots and
  SpikingNeuralNetworks with the Julia doc system, before and after the sweep. Each defining file
  was reviewed against the code, and each name was checked against the rendered site (baseline
  build of `dev` and final build of `docs/sweep`). The per-symbol table is in
  `claude/doc_inventory.md`.
- **Docstrings.** Every public symbol now has a docstring that was checked against the code:
  - signature;
  - behaviour;
  - every field with its default and unit;
  - equations in LaTeX (`@doc raw"""..."""`);
  - integration scheme and update order;
  - references: only those cited in the code or the canonical paper of the model, otherwise
    marked "reference not given in the code";
  - a runnable example.
- **Docs site.** New "Model catalogue" section (`docs/src/catalogue/`) with one page per family:
  - integrate-and-fire (`IF`, `AdEx`, `ExtendedIF`, `PostSpike`, the generalized-IF pipeline);
  - Izhikevich, Hodgkin-Huxley, Morris-Lecar;
  - rate models (`Rate`, `WilsonCowan`, `HetRec`);
  - spike sources (`Poisson` variants, `InhomogeneousPoisson`, `Identity`);
  - multicompartment (`Tripod`, `BallAndStick`, dendrite geometry);
  - synapse and receptor models;
  - connections and connectivity rules;
  - plasticity rules;
  - metaplasticity;
  - stimuli;
  - SNNUtils models (every file of `SNNUtils/src/models`, with its load status);
  - SNNUtils tools.

  Each model has its equations, a parameter table with defaults and units, the integration
  scheme, a short runnable example and the `@autodocs` block of its source file.

  I rewrote these pages against the current API: index, tutorial (`examples.md`), populations,
  stimuli, recordings, plasticity, visualization (now the SNNPlots page), API reference, models
  extension and contributing. `release_notes.md` is unchanged and consistent with the new pages.

  Every source file of SNNModels, SNNUtils and SNNPlots is rendered by exactly one `@autodocs`
  block, selected with `Pages = [...]`. The `Filter`-based blocks are gone. SNNUtils and SNNPlots
  were added to `makedocs(modules = ...)`, so `checkdocs` covers them.
- **Not loaded.** These source files exist but their `include` is commented out:
  `if_CANAHP.jl`, `adex_multitimescale.jl`, `multipod.jl`, `CaRule.jl`, `dump.jl`,
  `stimuli/variable_inputs.jl`. (The commented-out includes of `synapses/MultiReceptorSynapse.jl` and
  `sparse_plasticity/longshortSP.jl` point to files that no longer exist.) They are described
  in the catalogue as "present in the source tree but not loaded by SNNModels 1.8.4", with no
  `@autodocs` and no example.

## Inventory summary

"Exported" rows are the names exported by each package. "All rows" also counts the public
non-exported types and functions that were documented. A docstring counts as "Wrong before" when
it did not match the code: wrong default, unit, field, signature or behaviour, or a docstring
attached to the wrong binding. For the `SNN` rows, which are re-exports, correctness is tracked
on the row of the defining package.

| Package | Rows | Exported but undefined | Documented before | Undocumented before | Wrong before | Documented after | Undocumented after | On site before | On site after |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SNNModels (exported) | 313 | 19 | 184 | 110 | 103 | 294 | 0 | 179 | 290 |
| SNNModels (all rows) | 367 | 19 | 214 | 134 | 117 | 347 | 1 | 207 | 332 |
| SNNUtils (exported) | 49 | 1 | 27 | 21 | 20 | 48 | 0 | 0 | 48 |
| SNNUtils (all rows) | 55 | 1 | 28 | 26 | 21 | 54 | 0 | 0 | 54 |
| SNNPlots (exported) | 29 | 11 | 6 | 12 | 3 | 13 | 5 | 6 | 13 |
| SNNPlots (all rows) | 31 | 11 | 6 | 14 | 3 | 15 | 5 | 6 | 15 |
| SNN (exported) | 89 | 5 | 53 | 31 | 6 | 84 | 0 | 52 | 83 |
| SNN (all rows) | 89 | 5 | 53 | 31 | 6 | 84 | 0 | 52 | 83 |


- **Documented after the sweep.** Every defined exported name is documented, except five
  third-party re-exports of SNNPlots: `plot` and `plot!` (Makie), `inch` and `pt` (Measures), and
  `cm` (the SNNModels unit constant re-exported by SNNPlots).
- **Not on the site.** SNNModels re-exports `convolve` (Distributions) and `load`, `save`,
  `savename` (DrWatson). Their docstrings live in those packages and are not rendered.
- **No docstring.** `Multipod`/`MultipodNeurons` (in `multipod.jl`, which is not loaded) has none.
- **Exported but undefined: 36 names.** None was removed, because that would be a code change.
  - SNNModels (19): `BSParam`, `HUMAN`, `MOUSE`, `MultiRecetorSynapse`, `NMDA_CANAHP`,
    `PSParam`, `SpikeTime`, `SpikingSynapseDelay`, `Synapse_CANAHP`, `autocorrelogram`,
    `filter_populations`, `gcamp6_kernel`, `get_path`, `get_synapse_symbols`, `isi_cv`,
    `no_PlasticityVariables`, `no_STDPParameter`, `record_plast!`, `synapsearray`.
  - SNNPlots (11): `default_colors`, `dendrite_gplot`, `nature_figure`, `plot_activity`,
    `plot_connections`, `plot_model`, `plot_stimulus`, `plot_weights`, `soma_gplot`,
    `stdp_weight_decorrelated`, `stp_plot`.
  - SNNUtils (1): `pca`.
  - SpikingNeuralNetworks (5): `LTPParam`, `STPParam`, `SNNModel`, `make_copy`, `raster!`.
    `LTPParam` and `STPParam` still work as keyword names of `SpikingSynapse`.

## Correctness checks

- **Page examples.** I ran every ```` ```julia ```` block of every page (Julia 1.12.6, scratch env
  dev-linking the four worktrees, `scratchpad/run_md_examples.jl`):
  - 86 OK, 0 FAIL, 8 skipped.
  - The skipped blocks are marked `<!-- norun -->`:
    - the `sim!`/`train!` loop pseudocode and the `]add` line (index.md);
    - the FORCE-learning tutorial block (examples.md), which cannot run because of the
      `FLSynapse` bug below and carries a warning box;
    - a Plots.jl tip (recordings.md);
    - signature sketches (models_ext.md, contributing.md);
    - two equation/signature listings (connections.md, metaplasticity.md).
- **Docstring examples.** I extracted and ran every ```` ```julia ```` block in the docstrings of
  the four packages: 138 OK, 0 FAIL, 1 skipped (a list of `trigger!` lambdas in
  `perturbation_test`).
- **Baseline build** (`dev`, unmodified): succeeded with 30 "duplicate docs found" warnings. They
  came from the `Filter`-based `@autodocs` in populations.md, stimuli.md and visualization.md;
  visualization.md was not in `pages` but was built anyway. There were also 3 environment
  warnings.
- **Final build** (`docs/sweep`, `docs/make.jl`, offline): succeeded. The first attempt stopped on
  one ambiguous `@ref` in plasticity.md, now fixed. After that there are no docstring,
  `@autodocs`, missing-docs or cross-reference warnings. Only three environment warnings remain:
  - `HTML(edit_link)` cannot be determined (no `git remote` in the sandbox);
  - `deploydocs(devbranch)` cannot be determined, for the same reason;
  - "could not auto-detect the building environment. Skipping deployment".

  A search-index size warning (613 KiB) is silenced by
  `search_size_threshold_warn = 1000 * 2^10` in `make.jl`. The site renders 436 docstring bindings,
  against 238 before.

## Code bugs with the largest user impact (details and line numbers below)

1. **Missing methods break `train!`.**
   - `train!` raises a MethodError for `IZ`, `HH` and `MorrisLecar` populations: their parameter
     types are not `AbstractPopulationParameter`, so there is no `update_traces!` fallback.
   - It does the same for any `train!([pop])` without connections: `EmptySynapse` has no
     `update_traces!`.
2. **FORCE learning cannot run.** `FLSynapse` and `PINningSynapse` fail under both `sim!` and
   `train!` (their parameter types are not `AbstractConnectionParameter`). The `FLSparseSynapse`
   constructor always fails.
3. **Three models cannot receive connections.** `MorrisLecar`, `ExtendedIF` and `WilsonCowan`
   have no `synaptic_target`.
4. **`BalancedStimulus` is unusable.** The default `same_input = false` throws `UndefVarError:
   randcache`; `same_input = true` drives only neuron 1.
5. **Tripod and BallAndStick numerics.**
   - The Heun second stage of `w` lacks `* dt`.
   - The exponential term lacks `gl`.
   - The spike threshold is hard-coded at -10 mV.
   - The dendritic leak uses the somatic `El`.
   - `BallAndStick` ignores `Is` and `Id`.
6. **Results depend on `dt`.**
   - vSTDP: the LTP term has no `dt`.
   - `Confavreux2025Synapse`: input enters as `dt * w`.
   - `AggregateScaling`: per-step time constants.
   - `HetRec`: the trace update lacks `dt`.
7. **Spike detection and state are wrong.**
   - HH and Morris-Lecar `fire` is a level test, so one action potential gives several spike
     flags.
   - The input `g` of `Rate` is never reset under `RateSynapse`.
8. **Recording.**
   - `interpolated_record`'s time axis is misaligned when monitoring starts after `t = 0`.
   - `monitor!(obj, [(:fire, idx)])` ignores `idx`.
9. **SNNPlots `stdp_test`/`stdp_kernel`** measure a shifted kernel: the test neuron is driven by
   the plastic synapse itself.
10. **SNNUtils `compute_kei`, `get_model`, `residual_current`, `optimal_kei`** always throw
    (`PostSpike(A = ...)`, `EyalEquivalentNAR`, `synapsearray`).
11. **Analysis and IO functions that always fail:**
    - `compute_cross_correlogram`, `compute_covariance_density`;
    - `average_firing_rate(populations)`;
    - `st_order`;
    - `FanoFactor` with its default `interval`;
    - `load_data(path, name, info)`.
12. **`@update` / `@update!`**: the single-assignment forms fail at expansion.

## Things I am unsure about

- **References not cited in the code.** Some docstrings and pages give the canonical citation
  and page numbers of a model:
  - Brette and Gerstner 2005, J. Neurophysiol. 94:3637-3642;
  - Hodgkin and Huxley 1952, J. Physiol. 117:500-544;
  - Morris and Lecar 1981, Biophys. J. 35:193-213;
  - Wilson and Cowan 1972, Biophys. J. 12:1-24;
  - Izhikevich 2003, IEEE TNN;
  - Sussillo and Abbott 2009, Neuron 63:544-557;
  - Rajan, Harvey and Tank 2016, Neuron 90:128-142;
  - Miles et al. 1996, Neuron 16:815-823 (the latter two are taken from SNNUtils comments).

  I believe they are correct, but they were checked from memory, not against the papers.
- **References inferred from file names** (marked as inferred on the pages): Duarte and Morrison
  2019, Mongillo, Barak and Tsodyks 2008, Campagnola et al. 2022, Quaresima et al. 2023 for the
  Tripod.
- **HH kinetics.** They are described as Traub-Miles (COBAHH benchmark, Brette et al. 2007),
  flagged as not in the code. `El = -65 mV` differs from that benchmark.
- **STDPTriplet defaults.** The code attributes them to Pfister and Gerstner 2006, Table 4; not
  checked against the paper. The vSTDP defaults may be swapped relative to Litwin-Kumar and
  Doiron 2014 (from memory, not stated in the docs).
- **Intent unclear.** The missing `gl` in the Tripod/BallAndStick exponential term may be
  intentional. The `BalancedParameter` equations are documented literally from code that does
  not run. `HetRec` is described as a "non-recurrent layer", as in the old docstring.
- **Unit of NMDA `b`.** It is written as mM; check against `NMDAVoltageDependency`.
- **Review coverage.** The catalogue pages (about 2,900 lines) were written area by area. Each
  area was checked against its source files and its examples were run. I spot-checked the IF/AdEx
  and STDP sections against the code myself, but I did not re-read every line of every page.

## Docstrings that were wrong (and how they were fixed)

### Point neurons, rate models, spike sources

- `IFParameter`: defaults claimed C = 281 pF, gl = 40 nS, τm = 20 ms and "based on the standard Izhikevich model"; code has sentinels C = -1pF, gl = -1nS, τm = C/gl if both > 0 else 15ms, R = 1nS/gl if gl > 0 else 0.06 GΩ; ΔT (unused by IF) was undocumented. -> rewritten with exact defaults, equations, adaptation rule.
- `IF`: listed fields `glu`, `gaba` (do not exist; they are in `receptors`), type parameters `VBT`/`GIFT`, `tabs` as `VFT` zeros (code: Vector{Int} ones). -> rewritten, including the integration order and the first-step skip caused by `tabs = 1`.
- `AdEx`: signature with non-existent type parameters; no equations; spike detection at `v >= 0mV`, 20 mV peak value, reset on the next step and adaptive threshold θ (from `PostSpike.At/τA`) were not described. -> rewritten.
- `AdExParameter`: `Vt` called "membrane potential threshold"; spikes are detected at 0 mV, `Vt` is the exponential-term threshold and resting value of θ. -> fixed; added the code-comment remark that Brette-Gerstner use gl = 30 nS.
- `PostSpike`: documented fields `A` (does not exist) and `τA`; `At`, `AP_membrane`, `τabs`, `up` undocumented; which model uses which field not stated. -> rewritten.
- `IZ`: spike condition written as v >= 30 (code: v > 30); the synaptic term `ge (Ee - v) + gi (Ei - v)` was missing from the equations. -> fixed.
- `HetRecParameter`: listed a field `N` that does not exist. -> fixed; units of `rate` (1/ms) given.
- `Identity`: listed the type parameters as fields; state fields undocumented. -> rewritten.
- Stray docstring `[Integrate-And-Fire Neuron](...)` attached to the AdEx `update_neuron!` method removed (replaced by a comment).
- Docstrings that were only a link (`HH`, `MorrisLecar`, `Poisson`) replaced by full docstrings.

### Multicompartment neurons, synapse and receptor models

- `CurrentSynapse`: listed fields `E_i`, `E_e` that do not exist -> removed; added equations
  (current-based, `I_syn = -(ge - gi)`), Euler scheme.
- `CurrentSynapseVars`, `DoubleExpCurrentSynapseVars`: `ge`/`gi` called conductances -> they are
  currents (pA).
- `DeltaSynapse`: signature `DeltaSynapse{FT}` (no type parameter exists) -> `DeltaSynapse()`;
  added that it only has the three-argument `synaptic_current!` and fails with Tripod/BallAndStick,
  and that the voltage jump is `R w dt / τm` (dt-dependent). `DeltaSynapseVars` body was indented
  (rendered as code) -> fixed.
- `SingleExpSynapse`: `τi` described as "rise time constant" -> it is the inhibitory decay time
  constant.
- `ReceptorSynapse`: signature listed non-existent type parameters `FT`, `VFT` -> real keyword and
  positional signatures; added receptor kinetics, normalisation, NMDA block, exponential-Euler scheme.
- `MultiReceptorSynapse`: verbatim copy of the `ReceptorSynapse` docstring (fields `glu_receptors`,
  `gaba_receptors` do not exist; `receptors` missing; per-target inputs not described; positional
  constructor of 1.8.4 undocumented) -> rewritten.
- `Confavreux2025Synapse`: header said `DoubleExpSynapse{FT} <: AbstractDoubleExpParameter`;
  `τAMPA`, `τGABA` called rise times (they are decay times); `α` called "NMDA voltage dependence
  parameter" (it is the AMPA fraction of the excitatory conductance; the model has no voltage
  dependence) -> rewritten with equations. `Confavreux2025SynapseVars` header said
  `DoubleExpSynapseVars` -> fixed.
- `Receptors`: described as a struct with fields AMPA/NMDA/GABAa/GABAb -> it is a function returning
  `Vector{Receptor{Float32}}`; all four methods documented.
- `Receptor`: fields `name`, `target` missing, no units, `gsyn` rule vague -> complete field list
  with computed defaults (`gsyn = g0 * norm_synapse(τr, τd)` if `g0 > 0`).
- `infer_receptors`: claimed to throw for `target == :none` (it only logs `@error`); example not
  runnable -> fixed.
- `Glutamatergic`/`GABAergic`: fields typed `::T` (they are `::Receptor`) -> fixed.
- `NMDAVoltageDependency`: no formula/units -> `B(V) = 1/(1 + mg/b exp(k V))`, b in mM,
  k in 1/mV, mg in mM. Verified against `nmda_gating` and the inline expression in
  `ReceptorSynapse`'s `synaptic_current!`: both use `1/(1 + (mg/b) * exp256(k*v))`, consistent.
- `AbstractSynapseParameter`: method list referenced the legacy `update_synapses!(p, synapse,
  glu, gaba, synvars, dt)` (error-only fallback); real interface uses `receptors::NamedTuple` ->
  rewritten with the interface and the per-step order.
- `AbstractSynapseVariable`: subtype list missed `DoubleExpCurrentSynapseVars` and
  `Confavreux2025SynapseVars` -> complete list with fields (verified with the code).
- `Dendrite`: fields documented as scalars with defaults (El = -70.6mV, C = 10pF, gax = 10nS,
  gm = 1nS, l = 150um, d = 4um); actually `Vector{Float32}` of length `N` with default zeros,
  plus field `N` -> fixed.
- `Tripod`: said the soma has "dynamic thresholds for spike generation"; the spike threshold is a
  hard-coded -10 mV on the predicted potential and `θ` only enters the exponential term ->
  fixed, full equations added; type parameters `RECTS`, `RECTD` added; `I_d` shared by both
  dendrites stated. Field list was otherwise correct.
- `BallAndStick`: listed non-existent fields `gaba_d`, `glu_d`, `gaba_s`, `glu_s` (actual:
  `receptors_s`, `receptors_d`); did not say `Is`/`Id` are unused -> fixed, equations added.
- Newly documented (no docstring before): `synaptic_current!`, `update_synapses!`,
  `synaptic_variables`, `synaptic_receptors`, `synaptic_target`, `get_synapse_symbol`,
  `nmda_gating`, `create_dendrite`, `Physiology`, `human_dend`, `mouse_dend`, `proximal`,
  `proximal_distal`, `proximal_proximal`, `all_lengths`, `TripodParameter`,
  `BallAndStickParameter`, `Population(::DendNeuronParameter)`, `integrate!(::Tripod)`,
  `integrate!(::BallAndStick)`, `ReceptorVoltage`, `ReceptorArray`, `EyalNMDA`, `SomaNMDA`,
  `SomaReceptors`, `SomaSynapse`, `TripodSomaSynapse`, `TripodDendSynapse`.
- References: GABA (Miles et al. 1996, Neuron 16(4):815-823) and dendritic AMPA/NMDA (Eyal et al.
  2018, Front. Cell. Neurosci. 12, doi:10.3389/fncel.2018.00181) taken from the citations in
  SNNUtils/src/models/quaresima2022.jl. Jahr and Stevens (1990, J. Neurosci.) cited without
  volume/pages as the canonical form of the Mg block. Tripod: Quaresima et al. (2023), J. Physiol.,
  marked "not given in the code". Confavreux 2025 and "Duarte" (somatic AMPA): reference not given.

### Connections, connectivity, metaplasticity

- `SpikingSynapseParameter`: header stated `<: AbstractConnectionParameter`; the actual supertype is `AbstractSpikingSynapseParameter` -> fixed, described role.
- `SpikingSynapse`: essentially correct. Added: transmission equation and units of `W`, delayed-queue semantics, `:ge/:he -> :glu`, `:gi/:hi -> :gaba` mapping, `dt` keyword is unused, `LTPParam`/`STPParam` are keyword names (types are `LTPParameter`/`STPParameter`), all fields, self-contained example (old example had no `using`/`@load_units`).
- `AbstractNormalization`: stated `<: AbstractConnection`; actual `<: AbstractMetaPlasticity` -> fixed.
- `RateSynapse`, `SpikeRateSynapse`: docstring was only a link to a Brian2 synapse tutorial, unrelated to the implemented weight rule -> replaced by signature, equations (transmission and the learning rule), defaults, the `p = 0` NaN pitfall, missing helper fields.
- `FLSynapse`, `FLSparseSynapse`, `PINningSynapse`, `PINningSparseSynapse`: link-only docstrings (the dense `PINningSynapse` was titled "PINing Sparse Receptors") -> full RLS equations, initialisation, defaults, references (Sussillo & Abbott 2009; Rajan, Harvey & Tank 2016, both already linked in the code), and the fact that `sim!`/`train!` cannot dispatch on them.
- `MultiplicativeNorm`: "`{FT = Int32}`" (is Float32) and "τ (default 0.0)" (τ has no default, it is required) -> fixed.
- `AdditiveNorm`: "τ (default 0.0)" -> required; documented that the additive offset does not restore the initial sum.
- `SynapseNormalization`: listed a non-existent type parameter `MFT` (actual `VST`), described the unused field `t` as "time points"; `plasticity!` docstring had a wrong signature (`param::AdditiveNorm`, 3 arguments) and called `μ` a "rate of change" -> rewritten with the exact update for both norms.
- `AggregateScaling`: its only docstring was copy-pasted from `SynapseNormalization` (title "SynapseNormalization(N; param, kwargs...)", "N: The number of synapses" - N is the number of postsynaptic neurons, or an object with field N); `plasticity!(::AggregateScaling, ...)` docstring was also the SynapseNormalization one -> rewritten with the trace/target/rescaling equations.
- `RandomTurnover`, `ActivityDependentTurnover`: header only; `ActivityDependentTurnover{VFT <: Vector{Float32}}` is wrong (scalar `FT`, no default) -> fields, defaults, behaviour.
- `synaptic_turnover!`: documented a non-existent keyword `p_pre`, documented `p_new` as a one-argument function (it is called as `p_new(post, pre)`), omitted `p_values`, restricted the argument to `SpikingSynapse` -> fixed.
- `sparse_matrix`: "dense generator used up to SNNModels 1.8" -> "before SNNModels 1.8.2"; added that draws `<= 0` lower the realised degree, unknown keywords are ignored, `w` is unused, LogNormal semantics, example.
- Newly documented (had no docstring): `forward!` (generic), `EmptySynapse`, `SpikingSynapseDelayParameter`, `update_plasticity!`, `RateSynapseParameter`, `FLSynapseParameter`, `PINningSynapseParameter`, `FLSparseSynapseParameter`, `PINningSparseSynapseParameter`, `MetaPlasticityParameter`, `AbstractMetaPlasticity`, `AggregateScalingParameter`, `Turnover`, `MetaPlasticity` (both methods), `dsparse`, `matrix`, `matrix_record`, `indices`, `update_weights!`, `presynaptic`, `postsynaptic`, `presynaptic_idxs`, `postsynaptic_idxs`, `connect!`, `set_plasticity!(c, ::Bool)`, `has_plasticity`, `update_sparse_matrix!`, `sparse_matrix(Npre, Npost, conn)`.

### Plasticity rules (LTP, STP)

- `vSTDPParameter`: header said `<: SpikingSynapseParameter`; it is `<: LTPParameter`. No defaults, no equations. Fixed: full signature with defaults (A_LTD = 8e-4, A_LTP = 1.4e-3, θ_LTD = -70 mV, θ_LTP = -49 mV, τu = 20, τv = 7, τx = 15 ms, Wmax = 30 pF, Wmin = 0.1 pF), Clopath-form equations, update order, notes on the per-step LTP and 0 mV initial traces.
- `plasticity!(::AbstractSparseSynapse, ::vSTDPParameter, ...)`: described a normalisation step (`c.normalize.param.operator`, period `τ`) that does not exist, and a 3-argument signature. Rewritten.
- `plasticity!(..., ::iSTDPPotential, ...)`: said traces decay "otherwise" (they decay every step; `tpost` follows `v_post`) and that the weight always increases at a presynaptic spike (it changes by `η (tpost - v0)`, usually negative). Rewritten.
- `MarkramSTPParameter`: its docstring was attached to `AbstractMarkramSTPParameter`; `MarkramSTPParameter`, `MarkramSTPParameterEvent/Timestep/Het` had none. `U` was called "maximum utilization" (it is the baseline utilisation). Now one docstring per type with the exact event-driven equations and where the update happens (`update_traces!`, before `forward!`).
- `MarkramSTPVariables`: docstring existed but was not attached (blank line + `@snn_kw`). Fixed.
- `STDPMexicanHat`: claimed a zero-integral kernel; the implemented kernel `(1-z) e^{-z/√2}`, `z = (Δt/τ)^2`, has integral `√(π√2)(1 - 1/√2) ≠ 0`. Claim removed (also in plasticity.md), notation `[ln(x_pre/x_post)]^2` made explicit, added same-step interaction and example.
- `STDPSymmetric`: plain `"""` docstring with `\frac`, `\tau` (rendered as form-feed / tab), unbalanced formula with `1/τ` instead of `2τ` in the denominator, no fields. Rewritten as `@doc raw` with the kernel `A_x/(2τ_x) e^{-|Δt|/τ_x} - A_y/(2τ_y) e^{-|Δt|/τ_y}`, fields and defaults.
- Examples of `STDPGerstner`, `STDPWeightDependent`, `STDPTriplet`, `iSTDPRate` were not self-contained (undefined `pre`, `post`, `SNN`); now runnable.
- Undocumented before, documented now: `NoLTP`, `NoSTP`, `NoVariables`, `LTP`, `STP`, `NoSTDP`, `LTPParameter`, `STPParameter`, `STDPParameter`, `plasticityvariables`, generic `plasticity!` and `update_traces!` for sparse synapses, `set_plasticity!`, `set_LTP!`, `set_STP!`, `change_plasticity!`, `AbstractMarkramSTPParameter`, `MarkramSTPParameterEvent`, `MarkramSTPParameterTimestep`, `MarkramSTPParameterHet`, `vSTDPVariables`, `STDPAntiSymmetric`, `STDPStructuredVariables`, `STDPStructuredParameter`.
- `STDPConfavreux2025`: correct but did not say that `κ` weights the post-before-pre term (read at a presynaptic spike) and `γ` the pre-before-post term; the struct comments label `α` "post" and `β` "pre", the opposite of where they are applied (α at presynaptic spikes). Comments left, docstring explicit.

### Stimuli, recording, simulation loop

- `PoissonFixed`: `rate` documented as `Vector{R}` (scalar in code), `μ` missing -> rewritten with all fields.
- `PoissonInterval`: text copied from the layer ("each neuron of the N Poisson population"), `rate` as vector, `μ` missing -> rewritten; interval test `int[1] < t < int[end]` stated.
- `PoissonVariable`: `μ` missing, rate-function signature `(t, variables)` not given -> fixed.
- `PoissonLayer`: documented a non-existent field `ϵ` and an input rate `rate*N*ϵ`; `rate` as vector -> rewritten; noted that `active` is not read.
- `stimulate!(::PoissonStimulusLayer, ::PoissonLayer, ...)`: docstring had signature `p::PoissonStimulus` and was separated from the method by a blank line (so not attached) -> replaced and attached.
- `CurrentStimulus`: field `param::CurrentStimulus` (wrong type); the `Stimulus` method was documented as `CurrentStimulus(param::CurrentStimulus, ...)` -> fixed.
- `SpikeTimeStimulusParameter`: claimed that the keyword constructor `SpikeTimeParameter(; spiketimes, neurons)` sorts (it does not) -> fixed; `SpikeTimeParameter` got its own docstring.
- `SpikeTimeStimulus`: listed keyword arguments `p, μ, σ, w, dist, rule` that do not exist (the API takes `conn`) -> rewritten with `conn`, `N`, timing rule.
- `SpikeTimeStimulusIdentity`: third argument documented as `target::AbstractCompartment` (it is an optional `comp`) -> fixed.
- `next_neuron`: behaviour at the end of the list misdescribed -> documented as implemented, with the defect.
- `BalancedParameter`: titled `BalancedStimulusParameter{VFT} <: AbstractParameter`, field `w` missing, no equations -> rewritten with the equations transcribed from the code.
- `BalancedStimulus`: listed non-existent fields (`neurons, colptr, rowptr, I, J, index, randcache`) and a non-existent constructor `(post, sym, r, neurons; N_pre, p_post, μ)` -> rewritten, with a warning box on the defects.
- `neurons(::StimulusGroup)`: claimed a concatenated vector; returns a vector of vectors -> fixed.
- `monitor!`: its docstring was attached to `const _RECORD_META_KEYS` (moved to the function); it said indexed recording uses the legacy path only (Vector fields with indices are dense) and omitted the `200Hz` default of the collection methods, `verbose`, and the positional `variables` form -> rewritten.
- `record`: example used `interval = (0.0, 1.0)` (MethodError, a range is required); `interpolate`, `variables` undocumented -> rewritten with a running example.
- `interpolated_record`: omitted `τ`; implied bracket indexing at arbitrary times (StackOverflowError; call syntax `y(i, t)` is required) -> rewritten, time-axis assumptions documented.
- `sim!`: default `dt = 0.1f0` (code `0.125f0`); `train!`: default `dt = 0.1ms` (code `0.125ms`); both lacked the stimulus argument, model/keyword forms, `perturbation!`, return value -> rewritten with the exact step order.
- `get_interval`: return type "StepRange" (it is a `Float32` range from `dt` to `t`) -> fixed.
- New docstrings: `AbstractStimulusParameter` (expanded), `PoissonStimulusParameter`, `PoissonLayerParameter`, `PoissonStimulus`, `PoissonLayerHet`, `get_poisson_rate`, `EmptyStimulus`, generic `stimulate!`, `neurons`, `set_variable!`, `set_intervals!`, `set_active!` (base methods), all `Stimulus` methods, `reset_time!`, `get_measure_interval`, `add_endtime!`, `add_starttime!`, `getrecord`, `clear_records!` methods.

### Utilities, IO, analysis, API reference, umbrella module

- `Time`: `tt` listed as `Vector{Int}` (is `Vector{Int32}`), no defaults, duplicated lines -> keyword signature with defaults; `Time(time)` documents the `InexactError` for non-multiples of 0.125 ms.
- `AbstractPopulation`, `AbstractConnection`, `AbstractStimulus`, `AbstractStimulusGroup`: placeholder type names, `forward!(c, param)` instead of `forward!(c, param, dt, T)`, `update_traces!` missing, train!-only plasticity not stated, groups said to be stimulated as a whole (they are expanded into elements) -> rewritten from `main.jl`/`validate_*`.
- `@load_units`: whole docstring indented (rendered as code); no values -> unit table with base units and values, constant list, note that `mF` is not loaded.
- `@snn_kw`: inference of type parameters without defaults undocumented; example fails outside SNNModels -> documented, example imports `KwStrSentinel`.
- `@update`: advertised `@update config b.c = 5`, which throws at expansion -> only the block form documented, limitation stated. `@update!` (no docstring) -> documented with the same limitation.
- `compose`: wrong signature `compose(kwargs...; syn, pop)`, claimed to return `(pop, syn)`, empty Example -> real signature, NamedTuple return, key prefixing, sorting, `time`, runnable example.
- `print_model` (missing `get_keys`), `remove_element` (claimed `ArgumentError`; only logs), `exp64`/`exp256` ("64/256 iterations", "clamps"; real: 6/8 squarings, x < -10 returns 0).
- `modelcopy`: copy of `Base.deepcopy` text -> documents that recorded data, `:start_time`/`:end_time` and `:perturbation` are emptied.
- `graph`: edge property list wrong/incomplete, dangling sentence -> full vertex/edge property list.
- `SNNload`/`SNNsave`: default `count` documented as 1 (is 0), `suffix` missing, `type = :data` documented for `SNNsave` (unsupported) -> fixed; `SNNfile` file-name pattern given.
- `load_or_run`, `data2model`, `read_folder`, `write_config`: behaviour mismatches now stated (see section 3).
- `place_populations`: documented as `place_populations(config)` -> `place_populations(Npop, grid_size)`.
- `periodic_distance`: documented as Euclidean for all methods; the vector-grid method returns the L1 sum -> stated, with formulas.
- `neurons_within_circle`: said to return indices (returns a Bool mask).
- `compute_connections`: docstring was for a non-existent `compute_long_short_connections(...; dc, pl, ϵ, grid_size, conn)` and wrong returns -> both rules (`:critical_distance`, `:gaussian`) with equations, returns `(L, W, P)`, example.
- `linear_network`: positional arguments documented (keywords), return type Float32 (Float64) -> equation and correct signature.
- `spatial_activity`: required keyword `T` missing, `N` described as time steps (it is the number of cells), example failed -> rewritten, runnable example.
- `perturbation_test`: documented a positional `condition!` argument that does not exist (hook is keyword `trigger!`), example would throw `MethodError`; `perturbation!` keyword missing; "original model never mutated" is false with `add_records` -> fixed, runnable example.
- `alpha_function`: signature `alpha_function(t; t0, τ)` -> `alpha_function(t, τ)`, peak at τ.
- `firing_rate`: claimed that a missing `interval` throws (it builds `tt0:20ms:last spike`); other methods undocumented; example used an undefined model -> fixed with runnable example.
- `bin_spiketimes`: documented `bin_spiketimes(spiketimes, interval, sr)` with a sampling-rate argument and bin centres -> real keyword signature, all methods, return types.
- `compute_covariance_density`, `compute_cross_correlogram` (docstring titled `autocor`): wrong signatures -> documented as non-functional (section 3).
- `FanoFactor`: `window = 100ms` keyword does not exist -> `interval::AbstractRange` required.
- `ISI_CV2`, `merge_spiketimes`, `spiketimes_split`, `spikecount`, `print_summary`: indented text rendered as code, missing methods -> reformatted and completed.
- `st_order` (empty), `relative_time!` (named `relative_time`), `spikes_in_interval` (unformatted).
- `population_indices`: non-existent `type` argument; `filter_items`: documented a regex argument (real: keyword `condition`), and the docstring was attached to `no_noise`; `subpopulations`: wrong return description.
- `asynchronous_state`: its docstring was a stale copy for `inter_spike_interval` -> documented `(cv, ff, si)`.
- `is_attractor_state`: signature `(spiketimes, interval, N)` and Bool return wrong -> real signature, `(width | false_value, kde)`.
- `STTC`: no formula, pairwise signature missing `interval` -> formula, reference (Cutts & Eglen 2014, the canonical STTC paper), example.
- `gaussian_kernel`, `gaussian_kernel_estimate`: non-existent `length` argument; only closed boundaries described (default is periodic).
- Added docstrings (none before): `AbstractComponent`, `AbstractGroup`, `NetworkModel`, `@update!`, `pretty_nt_print`, `merge_models`, `ISI_CV`, `alpha_kernel`, `isi`, `shift_spikes!`, `spikes_in_intervals`, `find_interval_indices`, `interval_standard_spikes!`, `resample_spikes`, `infer_spikes`, `infer_spiketimes`, `sample_inputs`, `average_firing_rate`, `get_maxima`, `is_unimodal`, `tile_interval`, `target_neurons`, `average_conn_strength`, module docstrings `SNNModels` and `SpikingNeuralNetworks`, `DOCS_ASSETS_PATH`.
- An orphan docstring for a commented-out `firing_rate(P, τ; dt)` (spikes.jl:879-898) was attached to nothing useful; replaced by a one-line comment.

### SNNUtils and SNNPlots

SNNUtils
- `step_input`: documented keyword `lexicon` (actual `inputs`), a nonexistent `start_rate`, `pop`
  default `:E` (actual `:Exc`), return value "PoissonStimulus objects" (actual
  `MultiCompartmentStimulusGroup`s). Rewritten; now states that the default `targets = [nothing]`
  raises a MethodError; example uses a Tripod with `[:d1, :d2]`.
- `set_stimuli!`: documented a `targets` argument that does not exist and a returned model
  (returns `nothing`). Fixed; stimulus names `w_<word>` / `p_<phoneme>` documented.
- `update_stimuli!`: documented a nonexistent `targets` argument. Fixed.
- `generate_lexicon`: returned field documented as `silence_symbol` (actual `silence`). Fixed.
- `generate_sequence`: wrong signature (`generate_sequence(lexicon, config, seed=nothing)`;
  actual `generate_sequence(seq_function; lexicon, seed = -1, kwargs...)`), no description of the
  6-row sequence matrix. Rewritten with the row layout and an example.
- `word_phonemes_sequence`: the docstring described a different function
  (`generate_random_word_sequence(sequence_length, dictionary, silence_symbol; ...)`). Rewritten
  (modes `:fixed`, `:random`, `:balanced`, `presentations`, `seed`).
- `symbol_names`: docstring header `symbolnames`, and claimed words are prefixed with `w_` (false;
  that is `stimuli_names`). Fixed.
- `getdictionary`: `insert` argument undocumented. `getphonemes`: silence symbol `:_` appended,
  not stated. Fixed.
- `compute_kei`: default `Vs` documented as -52 mV (code: -55 mV). Fixed. `get_model`,
  `residual_current`, `compute_kei`, `optimal_kei`: presented as working, but they cannot run
  with SNNModels 1.8.4. Added a Status section. `residual_current`: keywords shown as optional,
  they are required. `nmda_curr`: called "NMDA current", it is the dimensionless Mg-block factor.
- `do_pca`: claimed to return a PCA object; it returns the projected data. Fixed.
- `score_spikes`: `pop` default documented `:E` (actual `:Exc`), `delay` shown as positional
  (keyword), "uses parallel processing" (the loop is serial). Fixed.
- `spikecount_features`: return type `Float32` (actual `Float64`). Fixed.
- `sym_features`: described as working and thread-safe; it throws `UndefVarError: SNN`. Fixed.
- `MultinomialLogisticRegression`: described as working; always throws (`make_set_index`
  undefined); in-place modification of `X` not stated. Fixed.
- `trial_average`: claimed any `dim` works; only the last dimension is averaged correctly. Fixed.
- `average_weight_dynamics`: synapse described as a "Receptors object"; record row ordering not
  stated. Fixed.
- `SVCtrain`: mostly correct; added the `labels` keyword (unused), the z-scoring and the
  small-set fallback.

SNNPlots
- `stdp_kernel!`: the docstring had the header `stdp_kernel(stdp_param; ΔT = -97.5:5:100ms)` and
  return type `Plots.Plot`; actual `stdp_kernel!(ax, stdp_param; ΔTs = ...)` returning a Makie
  plot. Rewritten; separate docstrings added for `stdp_kernel` and `stdp_test`.
- `plot_spatial_connectivity`: `do_legend` keyword and return values (`ax` or
  `(ax, legend_info)`) undocumented. `plot_connection_distances`: `probability` keyword
  undocumented. Fixed.
- Previously undocumented, now documented: `raster`, `raster!`, `vecplot`, `vecplot!`,
  `stdp_kernel`, `stdp_test`, `@makie_default`, `okabe_ito_10`, module docstring.

### Tutorial, index, models extension, contributing pages

None edited (the Tutorial, index, models extension, contributing pages area edits no source files).


## Docs-site text that was wrong (fixed)

### Point neurons, rate models, spike sources

- `populations.md`: autodocs blocks with `Filter = t -> t <: SNN.IF` etc. (caused duplicate-docs warnings with other pages); stated equation dV/dt = -(V - El)/τm + R(-w + I - I_syn) (code: τm dv/dt = -(v - El) + R(I - w) - R I_syn, i.e. the R term is also divided by τm); AdEx equation had ΔT exp((V-θ)/ΔT) without the τm scaling being consistent and no adaptive threshold; `@contents Pages = ["models.md"]` pointed to a non-existent page; unfinished sentences ("A list of", "must implement the following"). -> page rewritten as an overview with links to the catalogue; no autodocs.

### Multicompartment neurons, synapse and receptor models

- populations.md (area: Point neurons, rate models, spike sources) shows `AbstractSynapseParameter` twice (duplicate-docs warning
  in the baseline build). The new catalogue/synapses.md owns all synapse @autodocs; the Point neurons, rate models, spike sources area should
  drop the synapse autodocs from populations.md.

### Connections, connectivity, metaplasticity

- The old site had no connections or metaplasticity page; `visualization.md` contained an `@autodocs` with `Filter = t -> t <: SNNModels.AbstractConnection` that dumped the normalization types under "visualization" (owned by another worker; superseded by catalogue/metaplasticity.md).
- `SpikingNeuralNetworks` exports `LTPParam` and `STPParam`, which do not exist (they are keyword names of `SpikingSynapse`); `SNNModels` exports `SpikingSynapseDelay`, which does not exist. Stated on catalogue/connections.md.

### Plasticity rules (LTP, STP)

- plasticity.md, "Rules at a glance": `STDPMexicanHat` listed as "zero-integral kernel" (false, see above). Fixed.
- plasticity.md: the pre-spike pass was described as "LTD" and the post-spike pass as "LTP"; that holds only for the default signs (`STDPConfavreux2025` with default parameters potentiates in both passes). Reworded.
- plasticity.md: "Rule reference" and "Inhibitory STDP" pointed to the API Reference page for the rule docstrings; they now live in catalogue/plasticity_rules.md. The "Heterosynaptic Plasticity" section had three `@autodocs` Filter blocks (duplicates of other pages, including `AbstractSpikingSynapseParameter` which is not heterosynaptic); replaced by a link to catalogue/metaplasticity.md (page owned by C1).
- plasticity.md table was missing `iSTDPTime` (unusable) and the STP rules; added.
- Release notes are consistent with the code; nothing changed there.

### Stimuli, recording, simulation loop

- `stimuli.md`: consisted only of `@autodocs` blocks with type `Filter`s (8 duplicate-docs warnings in the baseline build) and an `@contents` of a non-existent `models.md` -> replaced with a user guide (building, targets, subsets, runtime changes, multicompartment); API moved to `catalogue/stimuli.md` with one `@autodocs` per source file.
- `recordings.md`:
  - every `SpikingSynapse(E, E, :he; μ = 2, p = 0.02, ...)` call failed: the connectivity must be passed as `conn = (μ = 2, p = 0.02)`;
  - interpolated records were indexed as `v[1, 3.14s]`, `v[1:10, 2.4s:15ms:3.1s]`, `ρ[:, 6.5s]`, `x[...]`, `tpost[...]`: this throws `StackOverflowError`; call syntax `v(1, 3.14s)` is required;
  - `SNN.matrix(EE, :ρ, 6.5s)` and `SNN.matrix(EE, :ρ, 6.5s:10ms:7s)` do not exist (`matrix_record(c, sym, t)` does);
  - used `histogram` without loading a plotting package;
  - said the start time is the model time "when `monitor!` was called" (it is the first `record!` call of the following run), and said nothing about the 200 Hz default of collections or the time-axis misalignment;
  - the "multiple neurons" snippet assigned `presynaptic` to `Is` and `postsynaptic` to `Js` (names swapped).
  All examples were rewritten (network without stimulus produced no activity; Poisson input added), durations reduced to 2 s + 2 s, and the page marked `<!-- run: sequential -->`.

### Utilities, IO, analysis, API reference, umbrella module

- `api_reference.md`: used Filter-based `@autodocs` over all of SNNModels, which duplicated docstrings already included on other pages (baseline duplicate-docs warnings) and put model types in "Other types". (Its statement that rules are passed with `LTPParam` is correct: it is the keyword/field of `SpikingSynapse`; the exported umbrella name `LTPParam` itself is undefined.) Rewritten: index, topic sections by `Pages`, table of pointers to the catalogue, umbrella-module section noting the undefined exports.

### SNNUtils and SNNPlots

- `visualization.md` (title "Plots"): contained only `@autodocs` blocks with type filters over
  SNNModels populations, parameters, connections and plasticity types, which duplicated docstrings
  of other pages (21 of the 30 baseline duplicate-docs warnings came from this page) and no
  SNNPlots content at all; it was also not listed in `make.jl` pages. Rewritten as the SNNPlots
  page.
- SNNUtils had no page and none of its docstrings appeared on the site; new pages
  `catalogue/snnutils.md` and `catalogue/snnutils_models.md`.

### Tutorial, index, models extension, contributing pages

index.md
- The example `SNN.SpikingSynapse(E, E, :ge, w = rand(E.N, E.N))` failed with `UndefKeywordError: conn`;
  the constructor is `SpikingSynapse(pre, post, sym, comp = nothing; conn, ...)`. Now `conn = (p = 0.1, μ = 2nS)`.
- The model was described as a NamedTuple with keys `pop, syn, stim`; `compose` also returns `name` and `time`.
- The pseudocode loop was wrong: it called `forward!(c, c_type)` (the real call is `forward!(c, c.param, dt, T)`),
  used `getfield(t, :param)` (undefined `t`), and did not show `record_zero!`, `update_traces!` (train! only,
  before integrate!/forward!) or `plasticity!` of populations. Rewritten after utils/main.jl and marked norun.
- `SpikingSynapse<:AbstractSynapse`: no such type; it is `SpikingSynapse <: AbstractConnection`
  (via AbstractSpikingSynapse <: AbstractSparseSynapse).
- `[Populations](@ref)` / `[Model Extensions ](@ref)` heading refs replaced by page links; links to the
  catalogue added. Default dt (0.125 ms) and both `sim!` call forms stated.

examples.md (Tutorial)
- Whole page used the removed Plots interface (`Plots.default`, `plot(...)`, `savefig`, `Plots.mm`) and
  `ASSET_PATH` (undefined); SNNPlots 0.2.x is Makie-based. All figures now built with Makie
  (`Figure`/`Axis` + `SNN.vecplot!`, `SNNPlots.raster!`) and saved to `mktempdir()`.
- Setup block used DrWatson `quickactivate` and `import SNNPlots: Plots`; replaced.
- AdEx equations: `exp((V - θ)/ΔT)` uses the dynamic threshold θ (relaxes to Vt with τA, jumps by At),
  spike at V >= 0 mV, V set to 20 mV then reset to Vr; the page now states this. Removed DataFrames dependency.
  Fixed `[Generalized Integrate and Fire models](@ref)` (stale anchor).
- Noise current: referred to a non-existent `CurrentNoiseParameter`; type is `CurrentNoise`. Its "variance
  100pA" is the standard deviation of `Normal`. α semantics (exponential mixing) documented from code.
- Balanced input: targeted `:ge`/`:gi` with `monitor!(E, [:ge, :gi])` (not fields of AdEx) and used the
  removed `SNN.gplot`; now `:glu`/`:gaba` and `monitor!(...; variables = :receptors)`.
- Ball-and-stick: used `DendNeuronParameter(C=..., gl=..., soma_syn=..., dend_syn=..., NMDA=..., postspike=...)`
  (none of these are fields of DendNeuronParameter any more), `PoissonLayerParameter` (does not exist),
  the old `PoissonLayer(E, :glu, :d, param=...)` constructor, `synapsearray` (undefined), monitored `:g_s, :g_d`
  (not fields). Rewritten with `BallAndStick(adex = AdExParameter(...), param = BallAndStickParameter(...))` and
  `Stimulus(PoissonLayer(...), E, :glu, :d; conn)`.
- Recurrent EI network: used `IFSinExpParameter` (does not exist), `PoissonLayerParameter`, the old
  `PoissonLayer(E, :ge, param=...)` and `SpikingSynapse(E, E, :ge, p=..., μ=...)` keyword API. Rewritten with
  `IFParameter` + `SingleExpSynapse` + `PostSpike`, `conn` NamedTuples, reduced to 1000 neurons / 1 s.
- FORCE learning: does not run in SNNModels 1.8.4 (see code bugs). Block marked norun with a warning admonition.
- Empty sections "STDP with homeostatic plasticity", "Recurrent network with dendrites", "Working memory with
  synaptic plasticity" removed; added a short "Synaptic plasticity" section showing that `sim!` leaves `W`
  unchanged and `train!` applies `STDPGerstner`.

models_ext.md
- Said the abstract types are `AbstractPopulation, AbstractStimulus, AbstractSynapse` (no AbstractSynapse;
  it is AbstractConnection) and that populations implement `integrate(...)` (it is `integrate!`).
- The neuron example imported `CurrentNoiseParameter` (undefined) and called `CurrentStimulus(neuron, :I, param = ...)`
  with a non-existent parameter type; the stimulus example used `PoissonLayer(neuron, :g; param=...)` (old API),
  fields `randcache, colptr, W, I` of `PoissonStimulus` that do not exist, `ProtoStructs` and a 100 s simulation.
  Both rewritten and runnable.
- Added the real extension interface: required fields (validate_*_model), integrate!/synaptic_target/Population,
  forward!/plasticity! for connections (no generic plasticity! fallback; generic forward!/update_traces! only for
  `AbstractConnectionParameter` subtypes), stimulate!/Stimulus, and plasticityvariables/plasticity!/update_traces!
  for LTP/STP rules; the simulation loop.
- Documented two @snn_kw constraints found while testing: a docstring cannot precede `@snn_kw struct` directly
  (must be attached to the bare name), and in-place parametric field types (`Vector{Float32}`) are rejected.

contributing.md
- Signatures were wrong (`sim!(p, c, duration)`, `train!(p::Vector{AbstractConnection}, ...)`, `forward!(p, p.param)`,
  `plasticity!(c, c.param, dt)`), code was not fenced, and it referred to a SparseMatrices notebook. Rewritten.


## Code bugs noticed, not fixed

Line numbers refer to the `dev` heads before the sweep (SNNModels `bea4269`, SNNUtils `8910518` on `main`, SNNPlots `861b97e`, SpikingNeuralNetworks `1633ecd`); docstring insertions shift them on `docs/sweep`.

### Point neurons, rate models, spike sources

- `src/populations/iz.jl:20`, `src/populations/hh.jl:1`, `src/populations/morrislecar.jl:2`: `IZParameter`, `HHParameter`, `MorrisLecarParameter` are not subtypes of `AbstractPopulationParameter`, so no `update_traces!` (and for IZ/HH no `plasticity!`) fallback exists: `train!` with an IZ, HH or MorrisLecar population raises `MethodError: no method matching update_traces!(::IZ, ::IZParameter, ...)`.
- `src/connections/empty.jl`: `EmptySynapse` has no `update_traces!`/`plasticity!` methods; `train!([pop]; duration)` without connections (default `C = [EmptySynapse()]`) raises a MethodError for every population type (verified for IF, AdEx, Poisson, Rate, ...). (File (area: Connections, connectivity, metaplasticity))
- `src/populations/morrislecar.jl` and `src/populations/generalized_if/if_extended.jl`, `src/populations/wilsoncowan.jl`: no `synaptic_target` method, so `SpikingSynapse(pre, ::MorrisLecar|::ExtendedIF, ...)` and `RateSynapse(pre, ::WilsonCowan)` raise MethodError; these models cannot receive connections.
- `src/populations/morrislecar.jl:93`: `MorrisLecar_w_nullcline` returns `-n_ss`; the w-nullcline is `w = n_ss(v)` (sign error). `MorrisLecar_dv` and `MorrisLecar_v_nullcline` have unreachable `return dv` lines.
- `src/populations/hh.jl:85`, `src/populations/morrislecar.jl:68`: `fire` is a level test (`v > -20` / `v > 20`), so one action potential yields several consecutive spike flags; spike counts and rates computed from `fire` are inflated.
- `src/populations/rate.jl` (integrate!) and `src/connections/rate_synapse.jl:43` (`# fill!(g, ...)` commented out): the input `g` of `Rate` is never reset, so with a `RateSynapse` it accumulates the sum of all past inputs (verified: g grows ~10x between 20 ms and 220 ms).
- `src/populations/hetrec.jl:151`: the soma update `v_s += (W v_d - v_s) dt/τm` is applied once per connected dendrite, so the effective somatic time constant is τm/k and depends on the number of connected dendrites (likely intended a sum of inputs with a single leak). `hetrec.jl:157`: `trace += (v_s - trace)/τrate` lacks the `dt` factor (time-step dependent).
- `src/populations/poisson.jl:96`, `src/populations/inhomogeneous_poisson.jl:44`: `R(noise * β, 1.0f0)` returns 1 for non-positive arguments and the argument itself for positive ones; the rate is discontinuous at 0 and smaller for small positive noise than for negative noise. Probably intended `1 + β noise` or a rectification.
- `src/populations/inhomogeneous_poisson.jl:48`: the spike draw uses the global RNG `rand(Float32)` while the noise uses the cache; harmless but inconsistent.
- `src/populations/generalized_if/if.jl`: `IF` does not accept heterogeneous (vector) parameters from `make_heterogeneous` (`isless(::Float32, ::Vector{Float32})`); only `AdEx` has a vector method.
- `src/populations/generalized_if/if_extended.jl:28`: `tabs::VFT` (Float32) receives `round(Int, ...)`; works but the field `w`, `Δv`, `Δv_temp` are unused.
- `src/populations/iz.jl`, `src/populations/hh.jl`: initial `gi = (12randn(N) .+ 20) .* 10nS` can be negative, and both `ge`, `gi` start at large non-zero values (40 and 200 nS), which strongly perturbs the first tens of ms.
- `src/populations/wilsoncowan.jl`: `WilsonCowan` is a copy of `Rate`, not the Wilson-Cowan model (documented as such).
- `src/populations/generalized_if/if_CANAHP.jl`, `src/populations/adex/adex_multitimescale.jl`: not included; they reference `AbstractIFParameter`, `AbstractAdExParameter`, `AbstractAdEx`, `NMDA_CANAHP`, `synapsearray`, `Synapse_CANAHP` which do not exist any more. Several of these names are still exported elsewhere (see the undefined-exports list).

### Multicompartment neurons, synapse and receptor models

- src/populations/multicompartment/tripod.jl:228 and ballandstick.jl:201: Heun second stage of the
  adaptation derivative uses `v_s[i] + Δv[i, 1]` and `w_s[i] + Δv[i, 4]` (missing `* dt`), unlike the
  voltage equations (`v + Δv * dt`). The predicted state is wrong by a factor 1/dt.
- tripod.jl:222, ballandstick.jl:194: exponential term `ΔT * exp((v - θ)/ΔT)` is not multiplied by
  `gl` (standard AdEx: `gl ΔT exp(...)`); with gl = 40 nS the spike-initiation current is 40x smaller
  than in AdEx. Possibly intentional, undocumented.
- tripod.jl:175, ballandstick.jl:153: spike threshold hard-coded at -10 mV; `adex.Vt` is only the
  resting value of θ.
- tripod.jl:214-217, ballandstick.jl:200: dendritic leak uses the somatic `adex.El`; the
  `Dendrite.El` field (set to -70.6 mV) is never used.
- ballandstick.jl:197: external current `+ I[i]` commented out; fields `Is` and `Id` exist but are
  never used (no way to inject current into a BallAndStick).
- tripod.jl:208 vs ballandstick.jl:185: synaptic currents clamped to ±1500 pA vs ±1000 pA
  (inconsistent, undocumented hard limits).
- dendneuron_parameter.jl:85: `v_post` is only assigned if the population has a field `v_<target>`;
  an invalid target gives `UndefVarError: v_post` instead of a clear error.
- synapse/synapses/DeltaSynapse.jl: only the three-argument `synaptic_current!` method exists, so
  `Tripod(soma_syn = DeltaSynapse())` fails with a MethodError at the first step (verified).
- synapse/synapses/Confraveux2025.jl:67-68: input enters as `dt * glu[i]`, so a spike of weight w
  increments gAMPA by w*dt; every other synapse model increments by w. Results depend on dt.
- Exported but not defined: `get_synapse_symbols`, `MultiRecetorSynapse` (synapses.jl:93, :98),
  `synapsearray` (receptors.jl:253), `NMDA_CANAHP`, `Synapse_CANAHP` (receptor_types.jl:105-106),
  `HUMAN`, `MOUSE` (dendrite.jl:113-114). I added a `# NOTE` comment above each export.
- receptors.jl:220-222: `Mg_mM`, `nmda_b`, `nmda_k` are non-const untyped globals (`nmda_b = 3.36` is
  Float64), used as `NMDAVoltageDependency` defaults; harmless but non-const globals.
- multipod.jl (not loaded): depends on `AdExSoma`, `synapsearray`, `PostSpike(A = ...)`, which do not
  exist; it cannot be re-enabled without changes.

### Connections, connectivity, metaplasticity

- connections/fl_synapse.jl:1, connections/pinning_synapse.jl:1, connections/fl_sparse_synapse.jl:1, connections/pinning_sparse_synapse.jl:1: `FLSynapseParameter`, `PINningSynapseParameter`, ... are not subtypes of `AbstractConnectionParameter`, so the generic `forward!(c, param, dt, T)` (connections.jl:14) does not apply: `sim!`/`train!` with an exported `FLSynapse` or `PINningSynapse` raise `MethodError: no method matching forward!(::FLSynapse, ::FLSynapseParameter, ::Float32, ::Time)` (verified). `update_traces!` has the same restriction.
- connections/fl_sparse_synapse.jl:35-36: `2rand(post.N) - 1` (vector minus scalar) -> constructor always fails (verified). Lines 54 and 74: `forward!`/`plasticity!` use `colptr` without unpacking it.
- connections/spike_rate_synapse.jl:48: `plasticity!` unpacks `rJ`, which `SpikeRateSynapse` does not have -> `train!` fails. The struct (line 2) has no `name` field and the constructor does not set `targets`.
- connections/rate_synapse.jl:26 (also fl_sparse_synapse.jl, pinning_sparse_synapse.jl, spike_rate_synapse.jl): with the default `p = 0.0` the scale `μ / √(p * pre.N)` is `Inf`; `Inf * sprandn(...)` gives a dense matrix of `NaN` (verified: `RateSynapse(r, r)` with N = 10 has 100 NaN weights).
- connections/spiking_synapse.jl:23-24: field defaults `NoPlasticityVariables()` refer to an undefined name (only `NoVariables` exists). Harmless today because the constructor always passes `LTPVars`/`STPVars`, but the keyword constructor without them would throw `UndefVarError`.
- connections/spiking_synapse.jl:209: exports `SpikingSynapseDelay`, which is not defined.
- SpikingNeuralNetworks.jl/src/SpikingNeuralNetworks.jl:55: exports `LTPParam`, `STPParam` (and `SNNModel`), which are not defined anywhere.
- utils/sparse_matrix.jl:135-139: `set_plasticity!(synapse, ::Bool)` and `has_plasticity` read `synapse.param.active`; `SpikingSynapseParameter` has no fields, so both raise `FieldError` for every `SpikingSynapse` (verified). SpikingNeuralNetworks re-exports `set_plasticity!`.
- utils/sparse_matrix.jl:1-6 and 434-462: `connect!` / `update_sparse_matrix!(c, W)` change the number of synapses but do not resize `ρ`, delays or plasticity variables (verified: after `connect!` creating a synapse, `length(W) = 201`, `length(ρ) = 200`). `forward!` then reads `ρ[s]` out of bounds inside `@inbounds` (memory unsafety).
- utils/sparse_matrix.jl:465: `update_sparse_matrix!(c)` rebuilds from `sparse(c.I, c.J, c.W)` without dimensions, so if the highest-index pre/post neuron has no synapse the matrix shrinks and `rowptr`/`colptr` change length. It also re-sorts synapses within columns while `ρ` and plasticity variables keep the old order (relevant after `synaptic_turnover!`).
- utils/sparse_matrix.jl:490: `dsparse` FIXME "Breaks when A is empty" (not verified).
- utils/sparse_matrix.jl:198: `sparse_matrix` swallows unknown keyword arguments (`kwargs...`), so a typo in a `conn` key (e.g. `sigma`) is silently ignored; the `w` keyword is unused.
- connections/metaplasticity/normalization.jl:147: `AdditiveNorm` computes `μ = (W0 - W1) / W1` and adds it to every synapse; the offset should presumably be `(W0 - W1) / n_i` to restore the sum (verified: W0 = 20, row sum after normalization 30.7).
- connections/metaplasticity/normalization.jl:51: default `param = MultiplicativeNorm()` cannot be constructed (`τ` has no default) - only matters for the keyword constructor without `param`.
- connections/metaplasticity/normalization.jl:127 (and aggregate_scaling.jl, turnover.jl): `get_step(T) % round(Int, τ / dt)` divides by zero when `τ < dt / 2`.
- connections/metaplasticity/aggregate_scaling.jl:88-94: the trace `y` and target `WT` are updated per step without `dt` (effective time constants `τa*dt`, `τe*dt`), in `forward!`, so they also evolve under `sim!`; `y` is a spike count while `Y` is given in Hz (1/ms) -> the comparison `y / Y` mixes units. (A remote branch `fix/aggregate-scaling-dt` exists.)
- connections/metaplasticity/aggregate_scaling.jl:27: struct field `N` defaults to 0 and the constructor does not pass `N`, so `AggregateScaling.N == 0` always (verified).
- connections/metaplasticity/aggregate_scaling.jl (positional `AggregateScalingParameter`): `Wmin` default 0.05 versus `0.5pF` in the keyword constructor.
- connections/metaplasticity/turnover.jl:59: only `ActivityDependentTurnover` has a `plasticity!(c, param, dt, T)` method; a `Turnover` with `RandomTurnover` makes `train!` raise `MethodError` (verified). `RandomTurnover.threshold` (line 14) is unused, and `p_rewire` is never set for it.
- connections/metaplasticity/turnover.jl:40: default `param = RandomTurnover(0)` (positional) and `synapse = SpikingSynapse()` (no such zero-argument method) - only reachable through the keyword constructor.
- connections/metaplasticity/turnover.jl:105: default `p_new = x -> rand()` is one-argument but is called as `p_new(post, pre)` -> `MethodError` when the default is used (verified).
- connections/metaplasticity/turnover.jl:162: new weights `Normal(μ, sqrt(μ))` can be negative.
- connections/metaplasticity/turnover.jl (synaptic_turnover!): `sample(plausible_post, weights, post_n; replace = false)` fails if a presynaptic neuron has more synapses to rewire than free targets.

### Plasticity rules (LTP, STP)

- `connections/sparse_plasticity/vSTDP.jl:143` (LTP loop of `plasticity!(…, ::vSTDPParameter, …)`): the LTP increment `A_LTP * x * [v-θ_LTD]_+ * [V-θ_LTP]_+` is added at every step without a factor `dt`, so the potentiation rate scales with `1/dt` (Clopath 2010 LTP is a continuous rate). Possibly intended if `A_LTP` is calibrated for `dt = 0.125 ms`, but changing `dt` changes the rule.
- `vSTDP.jl:92-93` `vSTDPVariables` and `iSTDP.jl:154` `iSTDPVariables`: the voltage traces `u`, `v` (vSTDP) and `tpost` (iSTDPPotential) start at 0 mV instead of the membrane potential. Verified: with a silent AdEx target and 20 Hz Poisson input, mean weight drops by about 0.022 in the first 50 ms purely from the transient `[u - θ_LTD]_+ ≈ 70 mV`; for iSTDPPotential the initial `tpost - v0 > 0` potentiates inhibition at the start.
- `vSTDP.jl:124`: a presynaptic spike raises `x_j` by `dt/τx` (Euler step of `-x + fireJ`), not by `1/τx` or 1, so the effective LTP amplitude also depends on `dt`.
- `STP.jl:229-233, 259-263` `update_traces!` for `MarkramSTPParameterEvent`/`Het` (lines with `ρ[s] = _ρ[j]`, then `u[j] += U*(1-u[j])`, `x[j] += -u[j]*x[j]`): the efficacy uses the utilisation before the facilitation jump (`u^- x^-`) while the depletion uses the utilisation after it (`u^+ x^-`). Mongillo et al. 2008 use `u^+` for both; Markram et al. 1998 use the same `u_n` for both. Not necessarily a bug, but internally inconsistent; documented as implemented.
- `STP.jl` `plasticity!(…, ::MarkramSTPParameterTimestep, …)`: the spike of step n is transmitted with the efficacy of step n-1, and the first spike of a new synapse with `ρ = 1` (SpikingSynapse initialises `ρ = ones`) instead of `U`.
- `sparse_plasticity.jl` export list: `no_STDPParameter`, `no_PlasticityVariables` are exported but not defined.
- `spiking_synapse.jl:23-24` (C1 file): defaults `NoPlasticityVariables()` reference an undefined name (harmless because the constructor always passes LTPVars/STPVars).
- SpikingNeuralNetworks.jl exports `LTPParam` and `STPParam`, which are not defined anywhere (they are keyword names of `SpikingSynapse`, not types).
- `STDPMexicanHat`: if a pre and a post neuron fire in the same step, the synapse is updated in both passes (double update). Documented.
- `STDP_structured.jl`: the weight loops use `@turbo` with gathers `to_y[I[s]]`; fine as long as no index aliasing, but `W[s]` writes in the post pass go through `index[st]` (scatter) inside `@turbo`, the same pattern that broke iSTDPRate (there the loop variable was reassigned; here it is not, so it should be correct). Not tested against a plain loop.

### Stimuli, recording, simulation loop

- `src/stimuli/balanced.jl:182`: `rand!(randcache)` refers to an undefined variable; `stimulate!` of a `BalancedStimulus` with the default `same_input = false` throws `UndefVarError` at the first step (verified).
- `src/stimuli/balanced.jl:171-173`: `same_input = true` branch adds `N` Poisson draws to `ge[i]` with `i = 1` fixed: all excitatory input goes to neuron 1 (verified: 1 non-zero entry).
- `src/stimuli/balanced.jl:184-186`: per-neuron branch adds `N` draws (loop over `n`) to `ge[i]`, i.e. rate multiplied by `N`.
- `src/stimuli/balanced.jl:105-108`: `param::Real` branch calls the undefined `BSParam` and reads `param.kIE` from a number (FieldError, verified).
- `src/stimuli/balanced.jl:133`: `Stimulus(param::BalancedParameter, post, sym)` passes `sym` as both `sym_e` and `sym_i`: inhibition is added to the excitatory buffer (verified `ge === gi`).
- Exported but undefined names: `BSParam` (balanced.jl:192), `PSParam` (poisson.jl:221), `SpikeTime` (timed.jl:398), `record_plast!` (record.jl:1017).
- `src/stimuli/poisson_layer.jl:70` with `:29-32`: struct default `param = PoissonLayer(-1)` calls `PoissonLayer(rate; kwargs...)`, which reads `kwargs[:N]` and fails (verified).
- `src/stimuli/poisson_layer.jl:151-186`: both `stimulate!` methods ignore `param.active`; `set_active!(layer, false)` has no effect (verified: 66 layer spikes in 100 steps while inactive).
- `src/stimuli/timed.jl:268-275`: `next_neuron` returns `[]` when the last spike is still pending (`next_index == length`), and throws `BoundsError` once all spikes are delivered (`next_index == -1`, verified).
- `src/stimuli/timed.jl:331-335`, `:357-365`: `shift_spikes!(::SpikeTimeStimulus, ...)` and `update_spikes!` index `spiketimes[1]` and fail on an empty spike list; `update_spikes!` does not sort the new spikes.
- `src/stimuli/stimulus_group.jl:291`: `neurons(::StimulusGroup)` is `vcat(map(...))` of a single vector, returning a vector of vectors instead of concatenated indices (verified `[[1, 2], [3]]`).
- `src/stimuli/stimulus_group.jl:214` and `:245`: `StimulusGroup.param` is restricted to `PoissonStimulusParameter`, and `MultiCompartmentStimulusGroup` passes `comp` as a keyword, which the layer/spike-time `Stimulus` methods do not accept (they take `comp` positionally); groups therefore only work with Poisson parameters.
- `src/stimuli/stimuli.jl:28-30`: `set_variable!` on a scalar parameter field (e.g. `PoissonFixed.rate`) tries `getfield(...) .= value` on a `Float32` and throws.
- `src/utils/record.jl:517-538`: `monitor!(obj, [(:fire, idx)])` ignores `idx` (the `:fire` branch never sets `records[:indices][:fire]`), so all neurons are recorded (verified), although `record_fire_dense!` supports indices.
- `src/utils/record.jl:648-652` (`get_measure_interval`, used by `interpolated_record`/`record`): the time axis is `range(start_time, end_time, nsamples)`, where `start_time` is the first `record!` call and `end_time` the last step, while samples are taken at global steps that are multiples of the period. Misaligned when monitoring starts at `t > 0` or the duration is not a multiple of the period. Verified: `:W` monitored at 10 Hz from 2 s to 4 s has true sample times 2100:100:4000 ms but `r = 2000.125:105.26:4000`; `:v` at 1 kHz for 10.5 ms gives `r = 0:1.05:10.5` for samples at 0:1:10.
- `src/utils/record.jl:629-631`: `interpolated_record(p, :fire, τ)` ignores `τ` (always `20ms`).
- `src/utils/record.jl:780`: assertion message interpolates the undefined `r_v`, so an out-of-bounds `interval` raises `UndefVarError` instead of the intended assertion message.
- Not a bug but worth knowing: `CurrentNoise` noise amplitude is per step and not scaled with `dt`, so results depend on `dt`.

### Utilities, IO, analysis, API reference, umbrella module

- utils/macros.jl:403: single-assignment `@update base a.b = v` uses the undefined `current_config` -> `UndefVarError` at macro expansion. Only the `begin ... end` form works.
- utils/macros.jl:515: single-assignment `@update! base a.b = v` interpolates `base` and `rhs` unescaped -> `UndefVarError` (resolved in SNNModels).
- utils/macros.jl:134,175: `@snn_kw` emits `KwStrSentinel` unqualified; any struct with type parameters defined with `@snn_kw` outside SNNModels fails at construction with `UndefVarError: KwStrSentinel` unless it is imported.
- utils/structs.jl:70: `Time(time)` does `Int32(time / 0.125f0)`, `InexactError` for times that are not multiples of 0.125 ms.
- utils/util.jl:177: `name = haskey(kwargs, :name) ? args.name : name` is dead code (`name` is a keyword, never in `kwargs`; `args.name` would error).
- utils/graph.jl:150: `filter_edge_props` returns `[]` when nothing matches; callers destructure `_edges, _ids = ...` (print_model, util.jl:244,269), which throws for an empty result.
- utils/graph.jl:164-165: `find_key_graph` uses undefined `e` and the typo `insothing`.
- utils/io.jl:113: `SNNload(path, name, info, kwargs...)` collects positional varargs and splats them as keywords.
- utils/io.jl:148: `load_data(path, name, info)` refers to undefined `kwargs` (UndefVarError when `info` is not a NamedTuple).
- utils/io.jl:168-171: `load_or_run` replaces `name` by `savename(name, info)` before `save_model`, so the model is saved in `savename(savename(name, info), info)` and never found by the next `load_model(path, name, info)`.
- utils/io.jl:303,309: `data2model` checks `joinpath(path, savename(name, info, "data.jld2"))`, which is not the `SNNsave` layout (`SNNfolder(path,name,info)/data-.jld2`).
- utils/io.jl:486,493: `String(key) == "study" || String(key)=="models" && continue` only skips `models` (precedence).
- utils/io.jl:546: `read_folder` default filter matches `"<type>.jld2"`, but `SNNsave` writes `"<type>-.jld2"` with the default empty suffix; `name` keyword unused.
- utils/io.jl:589: exports undefined `get_path`.
- utils/spatial.jl:49-55: `periodic_distance(::Vector, ::Vector, ::Vector)` computes `sqrt(sum(d)^2) = sum(d)` (L1) instead of the Euclidean distance (checked: distance of (0.1,0.1) from origin = 0.2).
- utils/spatial.jl:211: the `:gaussian` rule of `compute_connections` excludes `i == j` also when `pre != post`.
- utils/spatial.jl:322: time windows `(1+(t-1)T):(tT-1)` skip the last column of every window.
- analysis/spikes.jl:457,462,509,510: `bin_spiketimes(...; max_lag, bin_width)` without the required `interval` keyword -> `compute_cross_correlogram` and `compute_covariance_density` always throw; spikes.jl:479 uses undefined `auto_corr`.
- analysis/spikes.jl:427: `average_firing_rate(populations; interval)` assigns a local `spiketimes` that shadows the function it calls -> `UndefVarError`.
- analysis/spikes.jl:850: `st_order(spiketimes, pop, intervals)` calls undefined `spike_statistics`.
- analysis/spikes.jl:822,835: `FanoFactor` default `interval = nothing` and the tuple built by the population method are rejected by `bin_spiketimes` (`interval::AbstractRange`).
- analysis/spikes.jl:1086: `isnothing(findall(...))` is never true (dead check).
- analysis/spikes.jl:1047-1050: `sample_inputs(N, ::Matrix{<:Integer}, ...)` ignores `N` and `rate_factor`.
- analysis/targets.jl:82: `KDE` uses undefined `data` (unexported, unused).
- Undefined exports in my files: `autocorrelogram`, `gcamp6_kernel`, `isi_cv` (analysis/spikes.jl:1099-1114), `filter_populations` (analysis/populations.jl:103), `get_path` (utils/io.jl:589). Full list of undefined SNNModels exports (whole package): BSParam, HUMAN, MOUSE, MultiRecetorSynapse, NMDA_CANAHP, PSParam, SpikeTime, SpikingSynapseDelay, Synapse_CANAHP, autocorrelogram, filter_populations, gcamp6_kernel, get_path, get_synapse_symbols, isi_cv, no_PlasticityVariables, no_STDPParameter, record_plast!, synapsearray.
- Umbrella SpikingNeuralNetworks.jl exports undefined `LTPParam`, `STPParam`, `SNNModel`, `make_copy`, `raster!` (verified with `isdefined`).

### SNNUtils and SNNPlots

SNNPlots
- `src/raster.jl:3-20` `raster(spiketimes::Spiketimes, ...)`: spike times are plotted in ms but
  the x label says "Time (s)"; `markersize` is positional, not a keyword.
- `src/raster.jl:121` (and `:18`): `ylims!(0, ...)` without the axis argument sets the limits of
  the current axis, not of `ax`, in `raster!`.
- `src/raster.jl:82`: in `raster!(ax, P, ...)` without `populations`, the `names` keyword is
  overwritten and ignored.
- `src/raster.jl:113` and the Makie branch: extra `kwargs` are silently ignored.
- `src/vecplot.jl:24`: `vecplot(p, syms::Vector{Symbol})` passes `labels = [string(s)]`, but
  `vecplot!` has a `label` keyword; the legend label is always `"nothing"`.
- `src/vecplot.jl:33` and `:167` (`vecplot(P::Array, sym)`, `vecplot(P, syms::Array)`): call the
  Plots.jl `plot(...; size, layout)` API, which fails under Makie.
- `src/vecplot.jl:80`: `SNN.interpolated_record`, but `SNN` is not defined in SNNPlots
  (`factor::Symbol` path throws UndefVarError).
- `src/vecplot.jl:83`: `factor(neurons, :)` calls a Matrix (the `factor::Matrix` path throws).
- `src/stdp_plots.jl:1-13` `stdp_test`: the postsynaptic `Identity` neuron is driven by the
  plastic synapse and spikes ~0.1 ms after the presynaptic spike, so every measured change
  includes an extra causal pair. With `STDPGerstner()`: `+1.6e-4` at ΔT = +10 ms and `+3.9e-5`
  (instead of a depression) at ΔT = -10 ms. The plotted STDP kernels are shifted accordingly.
- `src/stdp_plots.jl:130-136` and `src/backend/makie.jl:46`, `src/SNNPlots.jl:45-55`: exported but
  undefined names: `stp_plot`, `plot_weights`, `plot_activity`, `dendrite_gplot`, `soma_gplot`,
  `stdp_weight_decorrelated`, `default_colors`, `nature_figure`, `plot_model`, `plot_stimulus`,
  `plot_connections`.
- `cm` exported by SNNPlots is the SNNModels length unit (1.0f0) created by `@load_units`; it
  shadows `Measures.cm`, while `inch` and `pt` are Measures lengths: mixing them is misleading.
- `raster!` is not exported by SNNPlots, but SpikingNeuralNetworks exports `raster!`
  (src/SpikingNeuralNetworks.jl:16), which is therefore undefined in `SNN`.
- SNNPlots depends on Makie only: no backend is loaded, `Makie.current_backend()` is `missing`
  after `using SpikingNeuralNetworks`; the user must load CairoMakie/GLMakie (CairoMakie is a
  [deps] entry of SNNPlots but is never `using`-ed).

SNNUtils
- `src/stimuli/sequence/sequence.jl:205`: `merge_intervals` drops the last interval when it does
  not touch the previous one (`[[0,1],[3,4],[5,6]]` -> `[[0,1],[3,4]]`).
- `src/stimuli/sequence/stimuli.jl:44`: `step_input` default `targets = [nothing]` is not accepted
  by `MultiCompartmentStimulusGroup` (MethodError); point-neuron targets unsupported.
- `src/stimuli/balance_EI/compute_kei.jl:45,59`: `PostSpike(A = 10.0, τA = 30.0)`: keyword `A`
  does not exist (`At`); `DendNeuronParameter(C = ..., gl = ..., ...)` keywords do not exist;
  `:49,76` `EyalEquivalentNAR` (defined only in the unloaded models/quaresima_2024_updown.jl) and
  `synapsearray` (exported by SNNModels but undefined); fallback uses undefined `AdExSoma`,
  `Multipod`. Hence `get_model`, `residual_current`, `compute_kei`, `optimal_kei` always fail.
- `src/stimuli/balance_EI/compute_kei.jl:146-154`: `residual_current` keyword defaults refer to
  themselves (`λ = λ`), so omitting any of them raises UndefVarError.
- `src/analysis/classifiers.jl:211` and `src/stimuli/bioseq/import_bioseq.jl:218`: `SNN.record`,
  `SNN` undefined in SNNUtils -> `sym_features` and `store_activity_data` throw.
- `src/analysis/classifiers.jl:361`: `make_set_index` undefined -> `MultinomialLogisticRegression`
  always throws; it also z-scores `X` in place (`transform!`) and prints with `@show`.
- `src/analysis/classifiers.jl:278`: in `score_spikes`, `activity_matrix` accumulates across all
  tested delays (never reset); the SpinLock is useless (serial loop).
- `src/analysis/classifiers.jl:137`: `trial_average` averages along `ave_dim` (last dimension of
  the output) instead of the trial dimension of the slice when `dim` is not the last dimension.
- `src/analysis/classifiers.jl:460`: `pca` exported but undefined.
- `src/stimuli/bioseq/import_bioseq.jl:6-22`: `import_bioseq_tasks` lists files in
  `generator_path` and opens them from `task_path` (and vice versa).
- `src/stimuli/bioseq/import_bioseq.jl:136`: neuron ranges built in the order E, SST(`I2.N`),
  PV(`I1.N`), while spikes are concatenated as E, I1, I2: labels are wrong if `I1.N != I2.N`.
- `src/stimuli/sequence/sequences/word_phonemes.jl:40,85`: `@show` debug output in a library
  function; `:fixed` mode errors (`pop!` on empty list) when rounding leaves fewer words than
  `presentations`.
- `src/models/*`: only `stp_het.jl` is included. `duarte2019.jl`, `lkd2014.jl`,
  `dendrite_STM.jl`, `mongillo_WM2008.jl` (and hence `models.jl`) use types/keywords removed from
  SNNModels (`IFParameterGsyn`, `AdExParameterGsyn`, `AdExSoma`, `IFCurrent`,
  `IFCurrentDeltaParameter`, `AdExSinExpParameter`, `IFSinExpParameter`, keywords `τabs`, `τri`,
  `E_i`, `At` of `AdExParameter`). `quaresima_2024_updown.jl:18` exports undefined
  `quaresima2022_nonmda`; `quaresima2023.jl:35` exports undefined `ballstick_network`.
- `src/models/connections.jl`, `quaresima2023.jl`: connection rules store `dist = Normal`
  (a type) but SNNModels 1.8.4 `sparse_matrix` does `getfield(Distributions, dist)` and needs a
  Symbol (`:Normal`): TypeError when passed as `conn`.

### Tutorial, index, models extension, contributing pages

- SNNModels connections/fl_synapse.jl:1, pinning_synapse.jl:1 (and fl_sparse_synapse.jl, pinning_sparse_synapse.jl
  parameter structs): `FLSynapseParameter`, `PINningSynapseParameter` are plain `struct ... end`, not
  `<: AbstractConnectionParameter`. The loop calls `forward!(c, c.param, dt, T)` and `update_traces!(c, c.param, dt, T)`
  whose generic methods (connections/connections.jl:43-50) require `AbstractConnectionParameter`, so
  `sim!` fails with `MethodError: forward!(::FLSynapse, ::FLSynapseParameter, ::Float32, ::Time)` and `train!` with
  `MethodError: update_traces!(...)`. Verified for FLSynapse and PINningSynapse. The FORCE tutorial cannot run.
- SpikingNeuralNetworks src/SpikingNeuralNetworks.jl: exports `raster!`, which SNNPlots defines but does not export,
  so `SNN.raster!` is `UndefVarError`; must use `SNNPlots.raster!`. (Also exports undefined LTPParam, STPParam,
  SNNModel, make_copy, already in the inventory.)
- SNNPlots raster.jl (Makie branch of `raster!`): `ylims!(0, maximum(Y) + 1)` acts on the current axis, not `ax`.
- SNNPlots vecplot.jl: `vecplot(P::Array, ...)` and `vecplot(P, syms::Array)` call Plots-style `plot(...; layout)`,
  which is not available with the Makie backend.
- SNNModels utils/macros.jl: `@snn_kw` rejects field types written as parametric types (`x::Vector{Bool} = ...` fails
  with `Cannot convert Expr to Symbol`) in structs without matching type parameters; and a docstring placed
  directly before `@snn_kw struct` errors ("cannot document the following expression"). Limitations, not
  necessarily bugs; documented in models_ext.md.
- SNNModels utils/util.jl `compose`: `name = haskey(kwargs, :name) ? args.name : name` — `args` is a Tuple, so
  this branch would error; unreachable in practice because `name` is a named keyword and never in `kwargs`.


## Uncertainties

### Point neurons, rate models, spike sources

- HH kinetics attributed (in the docstring, explicitly as not cited in the code) to the Traub-Miles form of the COBAHH benchmark (Brette et al. 2007). The default conductance densities and Vt = -63 mV match that benchmark; El = -65 mV does not match my recollection of the benchmark (-60 mV), so I did not claim the defaults are identical.
- Izhikevich (2003) regular-spiking parameters quoted as a=0.02, b=0.2, c=-65, d=8 (standard values); the default d = 2 / a = 0.01 is not a named class.
- Page numbers given: Brette & Gerstner 2005 J. Neurophysiol. 94:3637-3642; Hodgkin & Huxley 1952 J. Physiol. 117:500-544; Morris & Lecar 1981 Biophys. J. 35:193-213; Wilson & Cowan 1972 Biophys. J. 12:1-24. These are canonical and I am confident, but they were not in the code (except the Morris-Lecar URL, consistent with 35:193).
- The description of HetRec as "heterogeneous timescale, non-recurrent layer" comes from the original docstring; the name suggests "recurrent". Not resolvable from the code.

### Multicompartment neurons, synapse and receptor models

- Whether the missing `gl` in the dendritic-model exponential term is intentional (written as the
  code does, flagged in docs).
- Units of `b` in NMDAVoltageDependency stated as mM (it divides `mg`, in mM); the code gives no unit.
- `Physiology` comment says `Cd` in pF/cm^2 and `Ri` in Ω*cm; in storage these are library units
  (GΩ cm, GΩ cm², pF/cm²). Documented the storage units and the user-facing input form
  (`200Ω*cm`).
- Confavreux et al. (2025) reference not identified; left as "not given in the code".

### Connections, connectivity, metaplasticity

- RateSynapse learning rule: no reference in the code; I describe the code literally (it resembles a reconstruction/Oja-type rule but I did not name it).
- Whether `RateSynapse` is meant to accumulate `g` (the line `fill!(g, 0)` is commented out); documented as accumulating, and that `g` reset is up to the population.
- AggregateScaling and SynapseNormalization have no literature reference in the code; written "Reference not given in the code".

### Plasticity rules (LTP, STP)

- `STDPTriplet` defaults are documented in the code as "Table 4 of Pfister & Gerstner (2006), hippocampal culture data set, all-to-all minimal model". I could not verify these values against the paper; I kept the statement attributed to the code.
- vSTDP default amplitudes (A_LTD = 8e-4, A_LTP = 1.4e-3) and thresholds (-70, -49 mV): no reference in the code. The thresholds look like Litwin-Kumar & Doiron (2014), whose amplitudes I recall the other way round (A_LTD = 14e-4, A_LTP = 8e-4), but I did not verify, so the docstring says "default parameter values are not referenced in the code".
- "Confavreux et al. 2025": only the name is in the code; full reference not given.
- Festa, Cusseddu, Gjorgjieva (2024): title as cited in the code; venue not given.
- `PlasticityParameter` and `PlasticityVariables` (assigned to me) are defined with stub docstrings in `connections/connections.jl`, a C1 file; I did not edit it. Suggested text: "Abstract supertype of all plasticity rule parameters (`LTPParameter`, `STPParameter`, metaplasticity rules)" / "Abstract supertype of the state objects created by `plasticityvariables`".

### Stimuli, recording, simulation loop

- `BalancedParameter` equations are transcribed literally from the code (including the hard-coded 400 ms relaxation of the offset `r`); given the defects above, the intended model is unclear. No reference is given in the code.
- `record(pops, sym; interval)` (collection method) was documented from the code but not executed.
- The `Stimulus` methods of different files all attach to the same binding; Documenter shows them on the catalogue page under each file's `@autodocs` (they are method docstrings, so no duplicate warning is expected, but not verified with a build).

### Utilities, IO, analysis, API reference, umbrella module

- `linear_network`: the baseline formula for `w_0` has no reference in the code; documented as "Reference not given in the code".
- `asynchronous_state`: the "synchrony index" is the mean of the full neuron covariance matrix (including variances); documented as computed, without claiming a literature definition.
- STTC reference (Cutts & Eglen 2014, J Neurosci 34(43):14288-14303) is not in the code; added as the canonical source of the coefficient. ISI_CV2 (Holt et al. 1996) and Fano factor (Softky & Koch 1993) references were already in the code.
- `convolve` (exported) is `Distributions.convolve`; `load`/`save`/`savename` are DrWatson/FileIO re-exports. Their docstrings come from those packages and are not included by the SNNModels `@autodocs`.

### SNNUtils and SNNPlots

- References inferred from file/function names (not cited in the code), written as such on the
  page: Duarte & Morrison 2019 (duarte2019.jl), Mongillo, Barak & Tsodyks 2008
  (mongillo_WM2008.jl), Campagnola et al. 2022 (sample_stp_campagnola; the CSV columns match the
  Allen Institute synaptic-physiology dataset), Tripod neuron of Quaresima et al.
  (quaresima2022.jl), Silverman 1981 for the bimodality test (title and JSTOR link are in the
  comments). quaresima2023 / quaresima_2024_updown / dendrite_STM: "reference not given in the
  code".
- `do_pca`/`standardize`: the orientation implied by `ZScoreTransform(dims = 1)` was not
  re-derived; the docstrings only state the arguments passed.
- Building the full site: SNNPlots/SNNUtils are not in `makedocs(modules = ...)` of make.jl; add
  `SNNPlots, SNNUtils` if their docstrings should be checked by `checkdocs`. A mini build of my
  three pages with `modules = [SNNPlots, SNNUtils], checkdocs = :all` gave no missing-docs and no
  cross-reference warnings (only links to pages absent from the mini build).

### Tutorial, index, models extension, contributing pages

- The tutorial figures in docs/src/assets/examples were produced by earlier (Plots) versions; they are kept as
  illustrations and are not regenerated. Parameter values of the Recurrent EI network were mapped from the old
  `IFSinExpParameter(τm = 200pF/10nS, R = 1/10nS, τabs = 2ms, τe = τi = 5ms, E_e = 0, E_i = -80mV)` onto
  `IFParameter(C = 200pF, gl = 10nS)` + `SingleExpSynapse` + `PostSpike(τabs = 2ms)`; "Zerlaut et al. 2019"
  is from the original variable name, not a verified citation (phrased as "in the spirit of").
- Ball-and-stick example uses `ds = [(160um, 160um)]` (fixed length) instead of the old `[160um]`;
  `create_dendrite` with a tuple draws the length in the interval.

