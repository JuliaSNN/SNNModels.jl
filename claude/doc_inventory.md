# SNNModels / SpikingNeuralNetworks documentation inventory

Working file of the documentation sweep (branch `docs/sweep`). One row per exported symbol of
SNNModels, SNNUtils, SNNPlots and SpikingNeuralNetworks (`SNN`), plus public non-exported
types and functions that were documented. Columns:

- Exp: exported by its package (y/n).
- Doc before / after: a docstring is attached to the binding (measured with the Julia doc
  system on `dev` before the sweep and on `docs/sweep` after it, for exported names).
- Correct before: the old docstring matched the code (y/n; na when there was none). The
  'What was wrong' column says what did not match; it was fixed on `docs/sweep`.
- Site before / after: a docstring of that name is rendered on the Documenter site (baseline
  build of `dev` / final build of `docs/sweep`).
- kind `undef`: the name is exported but not defined (stale export, reported, not fixed).

## Summary

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

Undocumented after the sweep (defined names): `SNNModels.Multipod / MultipodNeurons`, `SNNPlots.cm`, `SNNPlots.inch`, `SNNPlots.plot!`, `SNNPlots.plot`, `SNNPlots.pt`.

Exported but undefined: `SNN.LTPParam`, `SNN.SNNModel`, `SNN.STPParam`, `SNN.make_copy`, `SNN.raster!`, `SNNModels.BSParam`, `SNNModels.HUMAN`, `SNNModels.MOUSE`, `SNNModels.MultiRecetorSynapse`, `SNNModels.NMDA_CANAHP`, `SNNModels.PSParam`, `SNNModels.SpikeTime`, `SNNModels.SpikingSynapseDelay`, `SNNModels.Synapse_CANAHP`, `SNNModels.autocorrelogram`, `SNNModels.filter_populations`, `SNNModels.gcamp6_kernel`, `SNNModels.get_path`, `SNNModels.get_synapse_symbols`, `SNNModels.isi_cv`, `SNNModels.no_PlasticityVariables`, `SNNModels.no_STDPParameter`, `SNNModels.record_plast!`, `SNNModels.synapsearray`, `SNNPlots.default_colors`, `SNNPlots.dendrite_gplot`, `SNNPlots.nature_figure`, `SNNPlots.plot_activity`, `SNNPlots.plot_connections`, `SNNPlots.plot_model`, `SNNPlots.plot_stimulus`, `SNNPlots.plot_weights`, `SNNPlots.soma_gplot`, `SNNPlots.stdp_weight_decorrelated`, `SNNPlots.stp_plot`, `SNNUtils.pca`.

## Neuron models

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `AdEx` | SNNModels | type | populations/generalized_if/adex.jl | y | y | n | type parameters in the signature do not exist; no equations; spike detection at 0 mV, 20 mV peak and adaptive threshold not described | y | y | y |
| `AdExParameter` | SNNModels | type | populations/generalized_if/adex.jl | y | y | n | Vt described as spike threshold (spikes are detected at v >= 0 mV; Vt is the exponential/threshold-rest value); otherwise defaults correct | y | y | y |
| `ExtendedIF` | SNNModels | type | populations/generalized_if/if_extended.jl | y | n | na |  | y | n | y |
| `ExtendedIFParameter` | SNNModels | type | populations/generalized_if/if_extended.jl | y | n | na |  | y | n | y |
| `HetRec` | SNNModels | type | populations/hetrec.jl | y | n | na |  | y | n | y |
| `HetRecParameter` | SNNModels | type | populations/hetrec.jl | y | y | n | listed a field N that does not exist (N is a Population keyword); units of rate not given | y | y | y |
| `HH` | SNNModels | type | populations/hh.jl | y | y | y | Wikipedia link only; no equations, fields or level-test spike semantics | y | y | y |
| `HHParameter` | SNNModels | type | populations/hh.jl | y | n | na |  | y | n | y |
| `IF` | SNNModels | type | populations/generalized_if/if.jl | y | y | n | listed non-existent fields glu, gaba and type parameters VBT/GIFT; tabs documented as VFT zeros (code: Vector{Int} ones); no equations or integration scheme | y | y | y |
| `IFParameter` | SNNModels | type | populations/generalized_if/if.jl | y | y | n | defaults wrong (C 281 pF, gl 40 nS, τm 20 ms; code: C=-1pF, gl=-1nS sentinels, τm=C/gl or 15 ms, R=1nS/gl or 0.06); 'based on the standard Izhikevich model' wrong; ΔT undocumented | y | y | y |
| `IZ` | SNNModels | type | populations/iz.jl | y | y | n | spike condition given as v >= 30 (code v > 30); synaptic conductance term missing from the equations | y | y | y |
| `IZParameter` | SNNModels | type | populations/iz.jl | y | y | y |  | y | y | y |
| `MorrisLecar` | SNNModels | type | populations/morrislecar.jl | y | y | y | link to the paper only | y | y | y |
| `MorrisLecarParameter` | SNNModels | type (not exported) | populations/morrislecar.jl | n | n | na |  | y | n | y |
| `PostSpike` | SNNModels | type | populations/spike/postspike.jl | y | y | n | listed fields A and τA only; A does not exist; At, AP_membrane, τabs, up undocumented | y | y | y |
| `Rate` | SNNModels | type | populations/rate.jl | y | y | y | did not mention that g is never reset | y | y | y |
| `RateParameter` | SNNModels | type (not exported) | populations/rate.jl | n | y | y |  | y | y | y |
| `WCParameter` | SNNModels | type (not exported) | populations/wilsoncowan.jl | n | n | na |  | y | n | y |
| `WilsonCowan` | SNNModels | type | populations/wilsoncowan.jl | y | n | na |  | y | n | y |

## Multicompartment neurons

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `all_lengths` | SNNModels | Vector{Tuple} | populations/multicompartment/dendrite.jl | y | n | na |  | y | n | y |
| `BallAndStick` | SNNModels | type | populations/multicompartment/ballandstick.jl | y | y | n | listed non-existent fields gaba_d/glu_d/gaba_s/glu_s (actual: receptors_s/receptors_d); no equations; did not say Is/Id are unused; type parameter RECT missing | y | y | y |
| `BallAndStickParameter` | SNNModels | function | populations/multicompartment/dendneuron_parameter.jl | y | n | na |  | y | n | y |
| `C_mem` | SNNModels | function | populations/multicompartment/dendrite.jl | y | y | y | units of inputs not stated (minor) | y | y | y |
| `create_dendrite` | SNNModels | function | populations/multicompartment/dendrite.jl | y | n | na |  | y | n | y |
| `DendNeuronParameter` | SNNModels | type | populations/multicompartment/dendneuron_parameter.jl | y | y | y | incomplete: no constructor signature, did not say that length(ds) selects Tripod/BallAndStick and other lengths error, nor that geometry is unused | y | y | y |
| `Dendrite` | SNNModels | type | populations/multicompartment/dendrite.jl | y | y | n | fields documented as scalars with defaults El=-70.6mV, C=10pF, gax=10nS, gm=1nS, l=150um, d=4um; actual: Vector fields of length N with default zeros (El=-70.6 set only by create_dendrite); N field missing | y | y | y |
| `G_axial` | SNNModels | function | populations/multicompartment/dendrite.jl | y | y | y | units of inputs not stated (minor) | y | y | y |
| `G_mem` | SNNModels | function | populations/multicompartment/dendrite.jl | y | y | y | units of inputs not stated (minor) | y | y | y |
| `HUMAN` | SNNModels | undef | populations/multicompartment/dendrite.jl | y | n | na | exported but not defined | n | n | n |
| `human_dend` | SNNModels | Physiology | populations/multicompartment/dendrite.jl | y | n | na |  | y | n | y |
| `integrate!(::BallAndStick)` | SNNModels | method | populations/multicompartment/ballandstick.jl | n | n | na |  | y | n | n |
| `integrate!(::Tripod)` | SNNModels | method | populations/multicompartment/tripod.jl | n | n | na |  | y | n | n |
| `MOUSE` | SNNModels | undef | populations/multicompartment/dendrite.jl | y | n | na | exported but not defined | n | n | n |
| `mouse_dend` | SNNModels | Physiology | populations/multicompartment/dendrite.jl | y | n | na |  | y | n | y |
| `Multipod / MultipodNeurons` | SNNModels | not loaded | populations/multicompartment/multipod.jl | n | n | na | file not included by SNNModels 1.8.4; does not compile (AdExSoma, synapsearray undefined) | n | n | n |
| `Physiology` | SNNModels | type | populations/multicompartment/dendrite.jl | y | n | na |  | y | n | y |
| `proximal` | SNNModels | Vector{Tuple} | populations/multicompartment/dendrite.jl | y | n | na |  | y | n | y |
| `proximal_distal` | SNNModels | Vector{Tuple} | populations/multicompartment/dendrite.jl | y | n | na |  | y | n | y |
| `proximal_proximal` | SNNModels | Vector{Tuple} | populations/multicompartment/dendrite.jl | y | n | na |  | y | n | y |
| `Tripod` | SNNModels | type | populations/multicompartment/tripod.jl | y | y | n | no equations; said soma has dynamic threshold for spike generation (spike threshold is a hard-coded -10 mV, θ only enters the exponential term); type parameters RECTS/RECTD missing; I_d not said to be shared by both dendrites; otherwise fields correct | y | y | y |
| `TripodParameter` | SNNModels | function | populations/multicompartment/dendneuron_parameter.jl | y | n | na |  | y | n | y |

## Spike sources (populations)

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `Identity` | SNNModels | type | populations/identity.jl | y | y | n | listed the type parameters VFT, IT as fields; state fields g, h, spikecount, fire undocumented | y | y | y |
| `IdentityParam` | SNNModels | type | populations/identity.jl | y | n | na |  | y | n | y |
| `InhomogeneousPoisson` | SNNModels | type | populations/inhomogeneous_poisson.jl | y | n | na |  | y | n | y |
| `InhomogeneousPoissonParam` | SNNModels | type | populations/inhomogeneous_poisson.jl | y | n | na |  | y | n | y |
| `Poisson` | SNNModels | type | populations/poisson.jl | y | y | y | link only | y | y | y |
| `PoissonHetParameter` | SNNModels | type | populations/poisson.jl | y | n | na |  | y | n | y |
| `PoissonHomoParameter` | SNNModels | type | populations/poisson.jl | y | n | na |  | y | n | y |
| `PoissonParameter` | SNNModels | abstract type | populations/poisson.jl | y | n | na |  | y | n | y |
| `VariablePoisson` | SNNModels | type | populations/poisson.jl | y | n | na |  | y | n | y |
| `VariablePoissonParameter` | SNNModels | type | populations/poisson.jl | y | n | na |  | y | n | y |

## Population infrastructure

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `integrate!` | SNNModels | function | populations/populations.jl (+ one method per model) | y | y | y | only method docstrings for Rate and IZ; no generic docstring | y | y | y |
| `make_heterogeneous` | SNNModels | function | populations/populations.jl | y | n | na |  | y | n | y |
| `Population` | SNNModels | function | populations/populations.jl (+ methods in model files) | y | n | na |  | y | n | y |
| `Population(::DendNeuronParameter)` | SNNModels | method | populations/multicompartment/dendneuron_parameter.jl | n | n | na |  | y | n | n |
| `update_neuron!` | SNNModels | function | populations/generalized_if/if.jl (+ adex.jl, if_extended.jl) | y | n | na |  | y | n | y |

## Synapse and receptor models

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `Confavreux2025Synapse` | SNNModels | type | populations/synapse/synapses/Confraveux2025.jl | y | y | n | header said DoubleExpSynapse{FT} <: AbstractDoubleExpParameter; τAMPA/τGABA described as rise times (they are decay times); α described as NMDA voltage dependence (it is the AMPA fraction of excitation, no voltage dependence); 'separate rise and decay' wrong | y | y | y |
| `Confavreux2025SynapseVars` | SNNModels | type (not exported) | populations/synapse/synapses/Confraveux2025.jl | n | y | n | header said DoubleExpSynapseVars | y | y | y |
| `CurrentSynapse` | SNNModels | type | populations/synapse/synapses/CurrentSynapse.jl | y | y | n | listed fields E_i and E_e that do not exist; no equations | y | y | y |
| `CurrentSynapseVars` | SNNModels | type (not exported) | populations/synapse/synapses/CurrentSynapse.jl | n | y | n | ge/gi described as conductances (they are currents, pA) | y | y | y |
| `DeltaSynapse` | SNNModels | type | populations/synapse/synapses/DeltaSynapse.jl | y | y | n | signature DeltaSynapse{FT} (the struct has no type parameter); did not say it is incompatible with Tripod/BallAndStick | y | y | y |
| `DeltaSynapseVars` | SNNModels | type (not exported) | populations/synapse/synapses/DeltaSynapse.jl | n | y | y | indented docstring body (rendered as code block) | y | y | y |
| `DoubleExpCurrentSynapse` | SNNModels | type | populations/synapse/synapses/DoubleExpCurrentSynapse.jl | y | y | y | no equations; did not say ge/gi are currents | y | y | y |
| `DoubleExpCurrentSynapseVars` | SNNModels | type (not exported) | populations/synapse/synapses/DoubleExpCurrentSynapse.jl | n | y | n | ge/gi described as conductances (they are currents) | y | y | y |
| `DoubleExpSynapse` | SNNModels | type | populations/synapse/synapses/DoubleExpSynapse.jl | y | y | y | no equations or integration scheme | y | y | y |
| `DoubleExpSynapseVars` | SNNModels | type (not exported) | populations/synapse/synapses/DoubleExpSynapse.jl | n | y | y |  | y | y | y |
| `EyalNMDA` | SNNModels | NMDAVoltageDependency | populations/synapse/receptor_types.jl | y | n | na |  | y | n | y |
| `GABAergic` | SNNModels | type | populations/synapse/receptors.jl | y | y | y | fields typed ::T (they are ::Receptor) | y | y | y |
| `get_synapse_symbol` | SNNModels | function (not exported) | populations/synapse/synapses.jl | n | n | na |  | y | n | y |
| `get_synapse_symbols` | SNNModels | undef | populations/synapse/synapses.jl | y | n | na | exported but not defined (defined name: get_synapse_symbol, not exported) | n | n | n |
| `Glutamatergic` | SNNModels | type | populations/synapse/receptors.jl | y | y | y | fields typed ::T (they are ::Receptor) | y | y | y |
| `infer_receptors` | SNNModels | function (not exported) | populations/synapse/receptors.jl | n | y | n | claimed it throws for target :none (it only logs @error); example not runnable | y | y | y |
| `MultiReceptorSynapse` | SNNModels | type | populations/synapse/synapses/ReceptorSynapse.jl | y | y | n | verbatim copy of the ReceptorSynapse docstring: listed fields glu_receptors/gaba_receptors that do not exist, missing receptors field, did not describe per-target inputs; positional constructor undocumented | y | y | y |
| `MultiRecetorSynapse` | SNNModels | undef | populations/synapse/synapses.jl | y | n | na | exported but not defined (typo of MultiReceptorSynapse) | n | n | n |
| `NMDA_CANAHP` | SNNModels | undef | populations/synapse/receptor_types.jl | y | n | na | exported but not defined (definition commented out) | n | n | n |
| `nmda_gating` | SNNModels | function | populations/synapse/receptors.jl | y | n | na |  | y | n | y |
| `NMDAVoltageDependency` | SNNModels | type | populations/synapse/receptors.jl | y | y | y | no formula, no units; b/k both called 'voltage dependence factor' | y | y | y |
| `norm_synapse` | SNNModels | function | populations/synapse/receptors.jl | y | y | y | no formula | y | y | y |
| `Receptor` | SNNModels | type | populations/synapse/receptors.jl | y | y | n | field target missing, name missing; no signature; units missing; gsyn description vague (actual g0*norm_synapse if g0>0 else 0) | y | y | y |
| `ReceptorArray` | SNNModels | alias | populations/synapse/receptors.jl | y | y | na |  | y | n | y |
| `Receptors` | SNNModels | function | populations/synapse/receptors.jl | y | y | n | described as a struct with fields AMPA/NMDA/GABAa/GABAb; it is a function returning Vector{Receptor{Float32}} | y | y | y |
| `ReceptorSynapse` | SNNModels | type | populations/synapse/synapses/ReceptorSynapse.jl | y | y | n | signature listed non-existent type parameters FT and VFT; no equations | y | y | y |
| `ReceptorSynapseVars` | SNNModels | type (not exported) | populations/synapse/synapses/ReceptorSynapse.jl | n | y | y |  | y | y | y |
| `ReceptorVoltage` | SNNModels | alias | populations/synapse/receptors.jl | y | n | na |  | y | n | y |
| `SingleExpSynapse` | SNNModels | type | populations/synapse/synapses/SingleExpSynapse.jl | y | y | n | τi described as rise time constant (it is the inhibitory decay time constant); no equations | y | y | y |
| `SingleExpSynapseVars` | SNNModels | type (not exported) | populations/synapse/synapses/SingleExpSynapse.jl | n | y | y |  | y | y | y |
| `SomaNMDA` | SNNModels | NMDAVoltageDependency | populations/synapse/receptor_types.jl | y | n | na |  | y | n | y |
| `SomaReceptors` | SNNModels | ReceptorArray | populations/synapse/receptor_types.jl | y | n | na |  | y | n | y |
| `SomaSynapse` | SNNModels | ReceptorSynapse | populations/synapse/receptor_types.jl | y | n | na |  | y | n | y |
| `Synapse_CANAHP` | SNNModels | undef | populations/synapse/receptor_types.jl | y | n | na | exported but not defined (definition commented out) | n | n | n |
| `synapsearray` | SNNModels | undef | populations/synapse/receptors.jl | y | n | na | exported but not defined | n | n | n |
| `synaptic_current!` | SNNModels | function | populations/synapse/synapses.jl | y | n | na |  | y | n | y |
| `synaptic_receptors` | SNNModels | function (not exported) | populations/synapse/synapses.jl | n | n | na |  | y | n | y |
| `synaptic_target` | SNNModels | function | populations/synapse/synaptic_targets.jl;populations/multicompartment/dendneuron_parameter.jl | y | n | na |  | y | n | y |
| `synaptic_variables` | SNNModels | function | populations/synapse/synapses.jl | y | n | na |  | y | n | y |
| `TripodDendSynapse` | SNNModels | ReceptorSynapse | populations/synapse/receptor_types.jl | y | n | na |  | y | n | y |
| `TripodSomaSynapse` | SNNModels | ReceptorSynapse | populations/synapse/receptor_types.jl | y | n | na |  | y | n | y |
| `update_synapses!` | SNNModels | function | populations/synapse/synapses.jl | y | n | na |  | y | n | y |
| `α_synapse` | SNNModels | function (not exported) | populations/synapse/receptors.jl | n | y | y |  | y | y | y |

## Connections and connectivity

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `connect!` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `dsparse` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `EmptySynapse` | SNNModels | type | connections/empty.jl | y | n | na |  | y | n | y |
| `FLSparseSynapse` | SNNModels | type | connections/fl_sparse_synapse.jl | n | y | n | link only; constructor and forward! are broken (not stated) | y | y | y |
| `FLSynapse` | SNNModels | type | connections/fl_synapse.jl | y | y | n | docstring was only a link; no equations/defaults; did not say that sim!/train! cannot dispatch on FLSynapseParameter | y | y | y |
| `FLSynapseParameter` | SNNModels | type | connections/fl_synapse.jl | n | n | na |  | y | n | y |
| `forward!` | SNNModels | function | connections/connections.jl | n | n | na | generic method undocumented | y | n | y |
| `has_plasticity` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `indices` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `matrix` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `PINningSparseSynapse` | SNNModels | type | connections/pinning_sparse_synapse.jl | n | y | n | link only | y | y | y |
| `PINningSynapse` | SNNModels | type | connections/pinning_synapse.jl | y | y | n | link titled 'PINing Sparse Receptors' on the dense type; no equations/defaults | y | y | y |
| `PINningSynapseParameter` | SNNModels | type | connections/pinning_synapse.jl | n | n | na |  | y | n | y |
| `postsynaptic` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `postsynaptic_idxs` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `presynaptic` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `presynaptic_idxs` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `RateSynapse` | SNNModels | type | connections/rate_synapse.jl | y | y | n | docstring was only a link to a Brian2 synapse tutorial unrelated to the implemented rule; no signature, equations or defaults | y | y | y |
| `RateSynapseParameter` | SNNModels | type | connections/rate_synapse.jl | n | n | na |  | y | n | y |
| `remove_autapses!` | SNNModels | function | utils/sparse_matrix.jl | n | y | y |  | y | y | y |
| `set_plasticity!` | SNNModels | function | utils/sparse_matrix.jl;connections/sparse_plasticity.jl | y | n | na | two-argument method in sparse_matrix.jl documented here; methods in sparse_plasticity.jl owned by C2 | y | n | y |
| `sparse_matrix` | SNNModels | function | utils/sparse_matrix.jl | y | y | y | minor: 'dense generator used up to SNNModels 1.8' (direct CSC since 1.8.2); did not mention that non-positive draws lower the degree and that unknown keys are ignored | y | y | y |
| `SpikeRateSynapse` | SNNModels | type | connections/spike_rate_synapse.jl | n | y | n | link to Brian2 tutorial only; no description | y | y | y |
| `SpikingSynapse` | SNNModels | type | connections/spiking_synapse.jl | y | y | y | minor: example not self-contained (no using/@load_units); dt keyword and :ge/:he mapping, delay semantics undocumented | y | y | y |
| `SpikingSynapseDelay` | SNNModels | undef | connections/spiking_synapse.jl | y | n | na | exported but not defined (the delay parameter type is SpikingSynapseDelayParameter, not exported) | n | n | n |
| `SpikingSynapseDelayParameter` | SNNModels | type | connections/spiking_synapse.jl | n | n | na |  | y | n | y |
| `SpikingSynapseParameter` | SNNModels | type | connections/spiking_synapse.jl | y | y | n | header only; stated supertype AbstractConnectionParameter (actual AbstractSpikingSynapseParameter) | y | y | y |
| `update_plasticity!` | SNNModels | function | connections/spiking_synapse.jl | y | n | na |  | y | n | y |
| `update_sparse_matrix!` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `update_weights!` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |

## Long-term plasticity (LTP/STDP/iSTDP/vSTDP)

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `change_plasticity!` | SNNModels | function | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `iSTDPParameter (not exported)` | SNNModels | abstract type | connections/sparse_plasticity/iSTDP.jl | n | y | y |  | y | n | n |
| `iSTDPPotential` | SNNModels | type | connections/sparse_plasticity/iSTDP.jl | y | y | y | correct; added equations and the 0 mV initial trace | y | y | y |
| `iSTDPRate` | SNNModels | type | connections/sparse_plasticity/iSTDP.jl | y | y | y | correct; example not self-contained (undefined SNN in a fresh module), no equations | y | y | y |
| `iSTDPTime` | SNNModels | type | connections/sparse_plasticity/iSTDP.jl | y | y | y |  | y | y | y |
| `iSTDPVariables` | SNNModels | type | connections/sparse_plasticity/iSTDP.jl | y | y | y |  | y | y | y |
| `LTP` | SNNModels | type | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `no_PlasticityVariables` | SNNModels | undef | connections/sparse_plasticity.jl | y | n | na | exported but not defined | n | n | n |
| `no_STDPParameter` | SNNModels | undef | connections/sparse_plasticity.jl | y | n | na | exported but not defined | n | n | n |
| `NoLTP` | SNNModels | type | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `NoSTDP` | SNNModels | NoLTP instance | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `NoVariables` | SNNModels | type | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `plasticity!` | SNNModels | function | connections/sparse_plasticity.jl;connections/sparse_plasticity/*.jl | y | y | n | vSTDP method docstring described a normalisation step (c.normalize, operator, τ) that does not exist and gave a 3-argument signature; iSTDPPotential method said traces decay only when the neuron does not fire and that the update always increases the weight; generic sparse-synapse method undocumented | y | y | y |
| `plasticityvariables` | SNNModels | function | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `set_LTP!` | SNNModels | function | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `STDPAntiSymmetric` | SNNModels | type | connections/sparse_plasticity/STDP_structured.jl | y | n | na |  | y | n | y |
| `STDPConfavreux2025` | SNNModels | type | connections/sparse_plasticity/STDP_traces.jl | y | y | y | correct; did not say which pairing κ and γ weight; no example | y | y | y |
| `STDPGerstner` | SNNModels | type | connections/sparse_plasticity/STDP_traces.jl | y | y | y | correct; example used undefined pre/post | y | y | y |
| `STDPMexicanHat` | SNNModels | type | connections/sparse_plasticity/STDP_traces.jl | y | y | n | claimed the kernel integral is zero; for z=(Δt/τ)^2 the integral of (1-z)exp(-z/sqrt2) is sqrt(pi*sqrt2)(1-1/sqrt2) != 0; ambiguous log(x)^2 notation | y | y | y |
| `STDPStructuredVariables (not exported)` | SNNModels | type | connections/sparse_plasticity/STDP_structured.jl | n | n | na | field comments mislabelled pre/post traces | y | n | n |
| `STDPSymmetric` | SNNModels | type | connections/sparse_plasticity/STDP_structured.jl | y | y | n | non-raw docstring with \frac and \tau (rendered as form-feed and tab), unbalanced formula with 1/τ in the denominator instead of 2τ, no fields/defaults | y | y | y |
| `STDPTriplet` | SNNModels | type | connections/sparse_plasticity/STDP_triplet.jl | y | y | y | correct; example used undefined pre/post; Table 4 provenance of the defaults not verified | y | y | y |
| `STDPTripletVariables` | SNNModels | type | connections/sparse_plasticity/STDP_triplet.jl | y | y | y |  | y | y | y |
| `STDPVariables` | SNNModels | type | connections/sparse_plasticity/STDP_traces.jl | y | y | y |  | y | y | y |
| `STDPWeightDependent` | SNNModels | type | connections/sparse_plasticity/STDP_weight_dependent.jl | y | y | y | correct; example used undefined pre/post | y | y | y |
| `vSTDPParameter` | SNNModels | type | connections/sparse_plasticity/vSTDP.jl | y | y | n | declared as <: SpikingSynapseParameter (it is <: LTPParameter); no defaults, no equations, did not say that LTP is applied every step without dt | y | y | y |
| `vSTDPVariables` | SNNModels | type | connections/sparse_plasticity/vSTDP.jl | y | n | na |  | y | n | y |

## Short-term plasticity

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `AbstractMarkramSTPParameter (not exported)` | SNNModels | abstract type | connections/sparse_plasticity/STP.jl | n | y | n | held the docstring of the concrete MarkramSTPParameter (misattached); MarkramSTPParameter itself had no docstring | y | n | n |
| `MarkramSTPParameter` | SNNModels | alias (non-const global) | connections/sparse_plasticity/STP.jl | y | n | na | its docstring was attached to AbstractMarkramSTPParameter; described U as 'maximum utilization' and the Markram 1998 model while the code is the Mongillo 2008 form with efficacy u^- x^- and depletion by u^+ | y | n | y |
| `MarkramSTPParameterEvent` | SNNModels | type | connections/sparse_plasticity/STP.jl | y | n | na |  | y | n | y |
| `MarkramSTPParameterHet` | SNNModels | type | connections/sparse_plasticity/STP.jl | y | n | na |  | y | n | y |
| `MarkramSTPParameterTimestep` | SNNModels | type | connections/sparse_plasticity/STP.jl | y | n | na |  | y | n | y |
| `MarkramSTPVariables` | SNNModels | type | connections/sparse_plasticity/STP.jl | y | n | na | a docstring existed in the source but was not attached (separated from the @snn_kw struct by a blank line; a docstring cannot attach to @snn_kw output anyway) | y | n | y |
| `NoSTP` | SNNModels | type | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `set_STP!` | SNNModels | function | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `STP` | SNNModels | type | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |
| `update_traces!` | SNNModels | function | connections/sparse_plasticity.jl | y | n | na |  | y | n | y |

## Metaplasticity

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `ActivityDependentTurnover` | SNNModels | type | connections/metaplasticity/turnover.jl | y | y | n | header only with a wrong type parameter (VFT <: Vector{Float32}; actual scalar FT) | y | y | y |
| `AdditiveNorm` | SNNModels | type | connections/metaplasticity/normalization.jl | y | y | n | said tau defaults to 0.0 (required); did not say the additive offset does not restore the initial sum | y | y | y |
| `AggregateScaling` | SNNModels | type | connections/metaplasticity/aggregate_scaling.jl | y | y | n | only docstring was copy-pasted from SynapseNormalization (title 'SynapseNormalization(N; param...)', 'N: number of synapses' - N is the number of postsynaptic neurons); plasticity! docstring also a SynapseNormalization copy | y | y | y |
| `AggregateScalingParameter` | SNNModels | type | connections/metaplasticity/aggregate_scaling.jl | y | n | na |  | y | n | y |
| `MetaPlasticity` | SNNModels | function | connections/metaplasticity/normalization.jl;connections/metaplasticity/turnover.jl | y | n | na |  | y | n | y |
| `MultiplicativeNorm` | SNNModels | type | connections/metaplasticity/normalization.jl | y | y | n | type parameter given as FT = Int32 (is Float32); said tau defaults to 0.0 (tau is required, no default) | y | y | y |
| `NormParam` | SNNModels | abstract type | connections/metaplasticity/normalization.jl | y | y | y | one-line only | y | y | y |
| `RandomTurnover` | SNNModels | type | connections/metaplasticity/turnover.jl | y | y | n | header only, no fields/defaults; did not say train! fails for it | y | y | y |
| `SynapseNormalization` | SNNModels | type | connections/metaplasticity/normalization.jl | y | y | n | type parameter MFT does not exist (VST); field t described as time points (unused); plasticity! docstring had wrong signature (param::AdditiveNorm, 3 args) and described mu as a 'rate of change' | y | y | y |
| `synaptic_turnover!` | SNNModels | function | connections/metaplasticity/turnover.jl | y | y | n | documented a non-existent keyword p_pre; p_new documented as one-argument (it is called with (post, pre), so the default fails); missing p_values keyword; argument type is any AbstractConnection | y | y | y |
| `Turnover` | SNNModels | type | connections/metaplasticity/turnover.jl | y | n | na |  | y | n | y |
| `TurnoverParam` | SNNModels | abstract type | connections/metaplasticity/turnover.jl | y | y | y | one-line only | y | y | y |

## Stimuli

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `BalancedParameter` | SNNModels | type | stimuli/balanced.jl | y | y | n | titled BalancedStimulusParameter{VFT} <: AbstractParameter; field w omitted; no equations | y | y | y |
| `BalancedStimulus` | SNNModels | type | stimuli/balanced.jl | y | y | n | listed non-existent fields (neurons, colptr, rowptr, I, J, index, randcache); constructor doc gave a non-existent signature (post, sym, r, neurons; N_pre, p_post, μ); stimulus is broken in 1.8.4 (not stated) | y | y | y |
| `BSParam` | SNNModels | undef | stimuli/balanced.jl | y | n | na | exported but not defined (also referenced in BalancedStimulus constructor) | n | n | n |
| `CurrentNoise` | SNNModels | type | stimuli/current.jl | y | y | y | incomplete: no update equation, α described only as 'decay factor' | y | y | y |
| `CurrentStimulus` | SNNModels | type | stimuli/current.jl | y | y | n | field param typed as CurrentStimulus; the Stimulus method was documented as CurrentStimulus(param::CurrentStimulus, ...) | y | y | y |
| `EmptyStimulus` | SNNModels | type | stimuli/empty.jl | y | n | na |  | y | n | y |
| `get_poisson_rate` | SNNModels | function | stimuli/poisson.jl | y | n | na |  | y | n | y |
| `max_neuron` | SNNModels | function | stimuli/timed.jl | y | y | y |  | y | y | y |
| `MultiCompartmentStimulusGroup` | SNNModels | function | stimuli/stimulus_group.jl | y | y | y | did not state that only Poisson parameters work | y | y | y |
| `neurons` | SNNModels | function | stimuli/stimuli.jl;stimulus_group.jl | y | y | n | StimulusGroup method claimed a concatenated vector; it returns a vector of vectors | y | y | y |
| `next_neuron` | SNNModels | function | stimuli/timed.jl | y | y | n | claimed [] only when no spikes remain; returns [] already when the last spike is pending and throws BoundsError after exhaustion | y | y | y |
| `PoissonFixed` | SNNModels | type | stimuli/poisson.jl | y | y | n | rate documented as Vector{R} (it is a scalar); field μ omitted | y | y | y |
| `PoissonInterval` | SNNModels | type | stimuli/poisson.jl | y | y | n | described as per-cell rates of an N-neuron layer (copied from PoissonLayer); rate is a scalar; μ omitted; interval semantics (strict inequalities) not stated | y | y | y |
| `PoissonLayer` | SNNModels | type | stimuli/poisson_layer.jl | y | y | n | documented a non-existent field ϵ and rate 'rate*N*ϵ'; rate documented as a Vector; active flag is not read by stimulate! | y | y | y |
| `PoissonLayerHet` | SNNModels | type | stimuli/poisson_layer.jl | y | n | na |  | y | n | y |
| `PoissonStimulus` | SNNModels | type | stimuli/poisson.jl | y | n | na |  | y | n | y |
| `PoissonStimulusLayer` | SNNModels | type | stimuli/poisson_layer.jl | y | y | y | param type given as PoissonLayer only; stimulate! method docstring had the wrong signature (p::PoissonStimulus) and was detached from the method by a blank line | y | y | y |
| `PoissonVariable` | SNNModels | type | stimuli/poisson.jl | y | y | n | field μ omitted; signature of the rate function (t, variables) not given | y | y | y |
| `PSParam` | SNNModels | undef | stimuli/poisson.jl | y | n | na | exported but not defined | n | n | n |
| `set_active!` | SNNModels | function | stimuli/stimuli.jl;stimulus_group.jl | y | y | y | only the StimulusGroup method was documented | y | y | y |
| `set_intervals!` | SNNModels | function | stimuli/stimuli.jl;stimulus_group.jl | y | y | y | only the StimulusGroup method was documented | y | y | y |
| `set_variable!` | SNNModels | function | stimuli/stimuli.jl;stimulus_group.jl | y | y | y | only the StimulusGroup method was documented | y | y | y |
| `shift_spikes!` | SNNModels | function | stimuli/timed.jl | y | y | y | (the method in analysis/spikes.jl is owned by another worker) | y | y | y |
| `SpikeTime` | SNNModels | undef | stimuli/timed.jl | y | n | na | exported but not defined | n | n | n |
| `SpikeTimeParameter` | SNNModels | function | stimuli/timed.jl | y | n | na | (described only inside the SpikeTimeStimulusParameter docstring) | y | n | y |
| `SpikeTimeStimulus` | SNNModels | type | stimuli/timed.jl | y | y | n | listed keyword arguments p, μ, σ, w, dist, rule that do not exist; the API takes conn (NamedTuple or matrix) | y | y | y |
| `SpikeTimeStimulusIdentity` | SNNModels | function | stimuli/timed.jl | y | y | n | third argument named target and typed AbstractCompartment (it is an optional comp) | y | y | y |
| `SpikeTimeStimulusParameter` | SNNModels | type | stimuli/timed.jl | y | y | n | claimed the keyword constructor SpikeTimeParameter(; spiketimes, neurons) sorts the spikes (it does not) | y | y | y |
| `stimulate!` | SNNModels | function | stimuli/stimuli.jl (generic) | y | y | y | no generic docstring; per-method docstrings only; layer method docstring detached and with wrong signature | y | y | y |
| `Stimulus` | SNNModels | function | stimuli/poisson.jl;poisson_layer.jl;current.jl;timed.jl;balanced.jl | y | y | n | only the current-stimulus method had a docstring and it carried a wrong signature | y | y | y |
| `StimulusGroup` | SNNModels | type | stimuli/stimulus_group.jl | y | y | y |  | y | y | y |
| `update_spikes!` | SNNModels | function | stimuli/timed.jl | y | y | y | did not state that spikes are not sorted and N/W are unchanged | y | y | y |

## Recording API

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `add_endtime!` | SNNModels | function | utils/record.jl | y | n | na |  | y | n | y |
| `add_starttime!` | SNNModels | function | utils/record.jl | y | n | na |  | y | n | y |
| `clear_monitor!` | SNNModels | function | utils/record.jl | y | y | y |  | y | y | y |
| `clear_records!` | SNNModels | function | utils/record.jl | y | y | y |  | y | y | y |
| `get_measure_interval` | SNNModels | function | utils/record.jl | y | n | na |  | y | n | y |
| `getrecord` | SNNModels | function | utils/record.jl | y | y | y |  | y | y | y |
| `getvariable` | SNNModels | function | utils/record.jl | y | y | y | incomplete (no dense/view behaviour) | y | y | y |
| `interpolated_record` | SNNModels | function | utils/record.jl | y | y | n | signature omitted τ; suggested bracket indexing at arbitrary time points (StackOverflowError; call syntax is needed); time-axis assumptions not stated | y | y | y |
| `matrix_record` | SNNModels | function | utils/sparse_matrix.jl | y | n | na |  | y | n | y |
| `monitor!` | SNNModels | function | utils/record.jl | y | n | na | a docstring existed but was attached to const _RECORD_META_KEYS, not to monitor!; it claimed indexed recording is legacy-only (Vector fields with indices are dense) and did not give the 200Hz default of the collection methods | y | n | y |
| `record` | SNNModels | function | utils/record.jl | y | y | n | example used interval = (0.0, 1.0) (MethodError: an AbstractRange is required); keywords interpolate and variables undocumented | y | y | y |
| `record!` | SNNModels | function | utils/record.jl | y | y | y |  | y | y | y |
| `record_fire!` | SNNModels | function | utils/record.jl | y | y | y |  | y | y | y |
| `record_plast!` | SNNModels | undef | utils/record.jl | y | n | na | exported but not defined | n | n | n |
| `record_sym!` | SNNModels | function | utils/record.jl | y | y | y |  | y | y | y |

## Simulation and model composition

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `compose` | SNNModels | function | utils/util.jl | y | y | n | signature `compose(kwargs...; syn, pop)` wrong (real: compose(args...; name, silent, time, kwargs...)); claimed to return a tuple (pop, syn) (returns NamedTuple with pop, syn, stim, name, time); empty Example section; key prefixing not described | y | y | y |
| `exp256` | SNNModels | function | utils/util.jl | y | y | n | said '256 iterations' and 'clamps' (8 squarings; x<-10 mapped to 0) | y | y | y |
| `exp64` | SNNModels | function | utils/util.jl | y | y | n | said '64 iterations' and 'clamps' (6 squarings of 1+x/64; x<-10 mapped to exactly 0) | y | y | y |
| `extract_items` | SNNModels | function | utils/util.jl | y | y | y |  | y | y | y |
| `get_dt` | SNNModels | function | utils/record.jl | y | y | y |  | y | y | y |
| `get_interval` | SNNModels | function | utils/record.jl | y | y | y | return type given as StepRange (it is a Float32 StepRangeLen) | y | y | y |
| `get_step` | SNNModels | function | utils/record.jl | y | y | y |  | y | y | y |
| `get_time` | SNNModels | function | utils/record.jl | y | y | y | NamedTuple method undocumented | y | y | y |
| `graph` | SNNModels | function | utils/graph.jl | y | y | n | edge property list incomplete/wrong (:type values, :pop, :meta, :target, :count, :multi); dangling 'Returns a MetaGraphs.MetaDiGraph where:' | y | y | y |
| `merge_models` | SNNModels | function | utils/util.jl | y | n | na | no docstring | y | n | y |
| `modelcopy` | SNNModels | function | utils/copying.jl | y | y | n | copied text of Base.deepcopy; did not say that recorded data, start/end times and perturbation records are emptied | y | y | y |
| `name` | SNNModels | function | utils/util.jl | y | y | y |  | y | y | y |
| `NetworkModel` | SNNModels | type | utils/structs.jl | y | n | na | no docstring | y | n | y |
| `print_model` | SNNModels | function | utils/util.jl | y | y | n | signature missing get_keys argument | y | y | y |
| `remove_element` | SNNModels | function | utils/util.jl | y | y | n | claimed ArgumentError on missing key (only logs @info) | y | y | y |
| `reset_time!` | SNNModels | function | utils/record.jl | y | n | na |  | y | n | y |
| `sim!` | SNNModels | function | utils/main.jl | y | y | n | default dt given as 0.1f0 (code: 0.125f0); stimuli argument, model/keyword forms, perturbation! and return value undocumented | y | y | y |
| `Spiketimes` | SNNModels | type | utils/structs.jl | y | y | n | no units/example (minor) | y | y | y |
| `str_name` | SNNModels | function | utils/util.jl | y | y | y |  | y | y | y |
| `Time` | SNNModels | type | utils/structs.jl | y | y | n | listed tt as Vector{Int} (is Vector{Int32}); no defaults or constructor signature; duplicated lines | y | y | y |
| `train!` | SNNModels | function | utils/main.jl | y | y | n | default dt given as 0.1ms (code: 0.125ms); stimuli argument, model forms, perturbation!, pbar and return value undocumented | y | y | y |
| `update_time!` | SNNModels | function | utils/record.jl | y | y | y | Time-copy method undocumented | y | y | y |

## Analysis functions

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `alpha_function` | SNNModels | function | analysis/spikes.jl | y | y | n | signature given as alpha_function(t; t0, τ) with peak time t0 (real: alpha_function(t, τ), peak at τ); indented | y | y | y |
| `alpha_kernel` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `asynchronous_state` | SNNModels | function | analysis/targets.jl | y | y | n | docstring was a stale copy of inter_spike_interval (wrong function) | y | y | y |
| `autocorrelogram` | SNNModels | undef | analysis/spikes.jl | y | n | na | exported but not defined | n | n | n |
| `average_conn_strength` | SNNModels | function | analysis/populations.jl | y | n | na | no docstring | y | n | y |
| `average_firing_rate` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `bin_spiketimes` | SNNModels | function | analysis/spikes.jl | y | y | n | signature bin_spiketimes(spiketimes, interval, sr) and sampling-rate argument do not exist; claimed bin centres returned (returns the interval itself); return types wrong | y | y | y |
| `clear_perturbation_monitor!` | SNNModels | function | utils/perturbation.jl | y | y | y |  | y | y | y |
| `clear_perturbation_records!` | SNNModels | function | utils/perturbation.jl | y | y | n | did not say n is ignored without condition | y | y | y |
| `compute_covariance_density` | SNNModels | function | analysis/spikes.jl | y | y | n | signature (t_post, t_pre, T; τ, sr) does not exist; did not say the function is non-functional | y | y | y |
| `convolve` | SNNModels | function | analysis/spikes.jl (re-export of Distributions.convolve) | y | y | na | re-export of an external package; docstring from that package | y | n | n |
| `FanoFactor` | SNNModels | function | analysis/spikes.jl | y | y | n | signature with window=100ms does not exist (keyword interval::AbstractRange required); indented | y | y | y |
| `filter_items` | SNNModels | function | analysis/populations.jl | y | n | na | no docstring; documented regex argument (real: keyword condition::Function); empty Examples; docstring was attached to no_noise instead of filter_items | y | n | y |
| `filter_populations` | SNNModels | undef | analysis/populations.jl | y | n | na | exported but not defined | n | n | n |
| `find_interval_indices` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `firing_rate` | SNNModels | function | analysis/spikes.jl | y | y | n | stated that omitting interval throws (an interval tt0:20ms:last spike is built); population and collection methods undocumented; example used undefined model | y | y | y |
| `gaussian_kernel` | SNNModels | function | analysis/targets.jl | y | y | n | argument named length (is ll); sampling grid not stated | y | y | y |
| `gaussian_kernel_estimate` | SNNModels | function | analysis/targets.jl | y | y | n | signature had a length argument (does not exist); described only closed boundaries (default is periodic :continuous) | y | y | y |
| `gaussian_smooth` | SNNModels | function | analysis/spikes.jl | y | y | y |  | y | y | y |
| `gcamp6_kernel` | SNNModels | undef | analysis/spikes.jl | y | n | na | exported but not defined | n | n | n |
| `get_maxima` | SNNModels | function | analysis/targets.jl | y | n | na | no docstring | y | n | y |
| `infer_spikes` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `infer_spiketimes` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `interval_standard_spikes` | SNNModels | function | analysis/spikes.jl | y | y | n | signature without margin | y | y | y |
| `interval_standard_spikes!` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `is_attractor_state` | SNNModels | function | analysis/targets.jl | y | y | n | signature (spiketimes, interval, N) does not exist (real: pop, interval; ratio, σ, false_value); returns (width or false_value, kde), not a Bool | y | y | y |
| `is_unimodal` | SNNModels | function | analysis/targets.jl | y | n | na | no docstring | y | n | y |
| `isi` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `ISI_CV` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `isi_cv` | SNNModels | undef | analysis/spikes.jl | y | n | na | exported but not defined | n | n | n |
| `ISI_CV2` | SNNModels | function | analysis/spikes.jl | y | y | n | indented text; methods and interval handling not documented | y | y | y |
| `merge_spiketimes` | SNNModels | function | analysis/spikes.jl | y | y | n | numpy-style indented text; in-place shifting of the input not mentioned; Spiketimes method undocumented | y | y | y |
| `perturbation_record` | SNNModels | function | utils/perturbation.jl | y | y | y |  | y | y | y |
| `perturbation_test` | SNNModels | function | utils/perturbation.jl | y | y | n | documented a positional `condition!` argument that does not exist (the hook is the keyword `trigger!`); example used the positional form (MethodError); perturbation! keyword undocumented; claimed the original model is never mutated (it is when add_records is set) | y | y | y |
| `population_indices` | SNNModels | function | analysis/populations.jl | y | y | n | documented a non-existent `type` argument and a Dict input | y | y | y |
| `relative_time!` | SNNModels | function | analysis/spikes.jl | y | y | n | docstring named relative_time (non-mutating) | y | y | y |
| `resample_spikes` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `sample_inputs` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `sample_spikes` | SNNModels | function | analysis/spikes.jl | y | y | n | rate units (Hz) and rate_factor keyword not documented | y | y | y |
| `spikecount` | SNNModels | function | analysis/spikes.jl | y | y | n | signature named the first argument model (is pop::AbstractPopulation); indented text | y | y | y |
| `spikes_in_interval` | SNNModels | function | analysis/spikes.jl | y | y | n | argument list unformatted; margin/collapse not documented | y | y | y |
| `spikes_in_intervals` | SNNModels | function | analysis/spikes.jl | y | n | na | no docstring | y | n | y |
| `spiketimes` | SNNModels | function | analysis/spikes.jl | y | y | y |  | y | y | y |
| `st_order` | SNNModels | function | analysis/spikes.jl | y | y | n | empty docstring; spike_statistics-based methods are broken | y | y | y |
| `STTC` | SNNModels | function | analysis/targets.jl | y | y | n | pairwise method signature missing the interval argument; no formula/reference | y | y | y |
| `subpopulations` | SNNModels | function | analysis/populations.jl | y | y | n | subset argument undocumented; returns a NamedTuple, not (names, pops) | y | y | y |
| `target_neurons` | SNNModels | function | analysis/populations.jl | y | n | na | no docstring | y | n | y |
| `tile_interval` | SNNModels | function | analysis/targets.jl | y | n | na | no docstring | y | n | y |

## IO and utilities

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `compute_connections` | SNNModels | function | utils/spatial.jl | y | y | n | docstring had the name compute_long_short_connections with non-existent arguments (dc, pl, ϵ, grid_size); return values wrong (L, W, P with P zero for :critical_distance); :gaussian rule undocumented | y | y | y |
| `compute_cross_correlogram` | SNNModels | function | analysis/spikes.jl | n | y | n | docstring titled autocor(spiketimes; interval); did not say the function is non-functional | y | y | y |
| `data2model` | SNNModels | function | utils/io.jl | y | y | n | did not mention that its paths do not match the SNNsave folder layout | y | y | y |
| `EmptyParam` | SNNModels | type | utils/structs.jl | n | y | y |  | y | y | y |
| `gaussian_weight` | SNNModels | function | utils/spatial.jl | n | y | n | no formula; exponent lacks factor 1/2, not stated | y | y | y |
| `get_git_commit_hash` | SNNModels | function | utils/io.jl | n | y | y |  | y | y | y |
| `get_path` | SNNModels | undef | utils/io.jl | y | n | na | exported but not defined | n | n | n |
| `isa_model` | SNNModels | function | utils/structs.jl | n | y | y |  | y | y | y |
| `linear_network` | SNNModels | function | utils/spatial.jl | y | y | n | signature showed positional σ_w, w_max (keywords); return type Matrix{Float32} (is Float64); equation missing | y | y | y |
| `load` | SNNModels | function | utils/io.jl (re-export of DrWatson/FileIO load) | y | y | na | re-export of an external package; docstring from that package | y | n | n |
| `load_data` | SNNModels | function | utils/io.jl | y | y | y |  | y | y | y |
| `load_model` | SNNModels | function | utils/io.jl | y | y | y |  | y | y | y |
| `load_or_run` | SNNModels | function | utils/io.jl | y | y | n | did not mention that saving uses savename(name, info) as name, so the saved folder differs from the one loaded | y | y | y |
| `neurons_outside_area` | SNNModels | function | utils/spatial.jl | y | y | y |  | y | y | y |
| `neurons_within_circle` | SNNModels | function | utils/spatial.jl | y | y | n | documented as returning indices; returns a Bool mask | y | y | y |
| `periodic_distance` | SNNModels | function | utils/spatial.jl | y | y | n | documented as Euclidean for all methods; the vector-grid method returns the L1 (sum) distance; types Float64 (code uses Float32) | y | y | y |
| `place_populations` | SNNModels | function | utils/spatial.jl | y | y | n | signature given as place_populations(config) (real: place_populations(Npop, grid_size)); Int64 filter not stated | y | y | y |
| `print_summary` | SNNModels | function | utils/io.jl | y | y | n | indented text rendered as code | y | y | y |
| `read_folder` | SNNModels | function | utils/io.jl | y | y | n | default filter documented as matching .jld2 files (matches names ending in '<type>.jld2', not SNNsave's 'model-.jld2'); name argument unused | y | y | y |
| `read_folder!` | SNNModels | function | utils/io.jl | y | y | y |  | y | y | y |
| `save` | SNNModels | function | utils/io.jl (re-export of DrWatson save) | y | y | na | re-export of an external package; docstring from that package | y | n | n |
| `save_config` | SNNModels | function | utils/io.jl | y | y | y |  | y | y | y |
| `save_model` | SNNModels | function | utils/io.jl | y | y | y |  | y | y | y |
| `savename` | SNNModels | function | utils/io.jl (re-export of DrWatson.savename) | y | y | na | re-export of an external package; docstring from that package | y | n | n |
| `SNNfile` | SNNModels | type | utils/io.jl | n | y | n | signature missing suffix; file-name pattern not given | y | y | y |
| `SNNfolder` | SNNModels | function | utils/io.jl | y | y | y |  | y | y | y |
| `SNNload` | SNNModels | function | utils/io.jl | y | y | n | default count documented as 1 (is 0); suffix keyword missing; return nothing on missing file not documented | y | y | y |
| `SNNModels (module)` | SNNModels | type | SNNModels.jl | n | n | na | no docstring | y | n | n |
| `SNNpath` | SNNModels | function | utils/io.jl | y | y | y |  | y | y | y |
| `SNNsave` | SNNModels | function | utils/io.jl | y | y | n | default count documented as 1 (is 0); suffix missing; type=:data documented but not supported | y | y | y |
| `spatial_activity` | SNNModels | function | utils/spatial.jl | y | y | n | signature/arguments wrong: required keyword T (time windows) missing, N described as time steps (is number of cells), L/N exclusive not stated, grid_size default wrong; example called it without T and with both N and L (fails) | y | y | y |
| `spiketimes_split` | SNNModels | function | analysis/spikes.jl | n | y | n | indented text, return of names not documented | y | y | y |
| `write_config` | SNNModels | function | utils/io.jl | y | y | n | claimed to skip 'study' fields (only 'models' skipped due to operator precedence); kwargs said to be saved (ignored) | y | y | y |

## Macros

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `@load_units` | SNNModels | macro | utils/unit.jl | y | y | n | whole docstring indented (rendered as code block); kHz described as base unit without values; constant list missing | y | y | y |
| `@snn_kw` | SNNModels | macro | utils/macros.jl | y | y | n | did not document inference of type parameters without defaults; example fails outside SNNModels unless KwStrSentinel is imported | y | y | y |
| `@symdict` | SNNModels | macro | utils/macros.jl | y | y | y |  | y | y | y |
| `@update` | SNNModels | macro | utils/macros.jl | y | y | n | advertised single-assignment form `@update config b.c = 5`, which throws UndefVarError at expansion | y | y | y |
| `@update!` | SNNModels | macro | utils/macros.jl | y | n | na | no docstring | y | n | y |
| `pretty_nt_print` | SNNModels | function | utils/macros.jl | y | n | na | no docstring | y | n | y |
| `update_with_merge` | SNNModels | function | utils/macros.jl | y | y | y |  | y | y | y |

## Abstract types

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `AbstractComponent` | SNNModels | abstract type | utils/structs.jl | y | n | na | no docstring | y | n | y |
| `AbstractConnection` | SNNModels | abstract type | utils/structs.jl | y | y | n | interface given as forward!(c, param) (code calls forward!(c, param, dt, T)); placeholder type name Receptors | y | y | y |
| `AbstractConnectionParameter` | SNNModels | abstract type | connections/connections.jl | y | y | y | header only | y | y | y |
| `AbstractConnectivity` | SNNModels | abstract type | connections/connections.jl | n | y | y | header only (no concrete subtype exists) | y | y | y |
| `AbstractDendriteIF` | SNNModels | abstract type | populations/populations.jl | y | n | na |  | y | n | y |
| `AbstractGeneralizedIF` | SNNModels | abstract type | populations/populations.jl | y | n | na |  | y | n | y |
| `AbstractGeneralizedIFParameter` | SNNModels | abstract type | populations/populations.jl | y | n | na |  | y | n | y |
| `AbstractGroup` | SNNModels | type | utils/structs.jl | n | n | na | no docstring | y | n | y |
| `AbstractMetaPlasticity` | SNNModels | abstract type | connections/connections.jl | n | n | na |  | y | n | y |
| `AbstractNormalization` | SNNModels | abstract type | connections/connections.jl | n | y | n | stated supertype AbstractConnection (actual AbstractMetaPlasticity) | y | y | y |
| `AbstractParameter` | SNNModels | abstract type | utils/structs.jl | y | y | y |  | y | y | y |
| `AbstractPopulation` | SNNModels | abstract type | utils/structs.jl | y | y | n | interface listed only integrate!/plasticity! with placeholder type names; update_traces! missing; train!-only plasticity not stated | y | y | y |
| `AbstractPopulationParameter` | SNNModels | abstract type | populations/populations.jl | y | y | y | signature only, no description | y | y | y |
| `AbstractSparseSynapse` | SNNModels | abstract type | connections/connections.jl | n | y | y | header only | y | y | y |
| `AbstractSpikeParameter` | SNNModels | abstract type (not exported) | populations/populations.jl | n | n | na |  | y | n | y |
| `AbstractSpikingSynapse` | SNNModels | abstract type | connections/connections.jl | n | y | y | header only | y | y | y |
| `AbstractSpikingSynapseParameter` | SNNModels | abstract type | connections/connections.jl | n | y | y | header only | y | y | y |
| `AbstractStimulus` | SNNModels | abstract type | utils/structs.jl | y | y | n | placeholder type names; required fields not listed | y | y | y |
| `AbstractStimulusGroup` | SNNModels | type | utils/structs.jl | n | y | n | referred to stimulate!(group) (groups are expanded into elements) | y | y | y |
| `AbstractStimulusParameter` | SNNModels | abstract type | stimuli/stimuli.jl | y | y | y | docstring was only the signature line | y | y | y |
| `AbstractSynapseParameter` | SNNModels | abstract type | populations/synapse/synapses.jl | y | y | n | method list referenced update_synapses!(p, synapse, glu, gaba, synvars, dt), a legacy error-only fallback; the real interface is (p, synapse, receptors::NamedTuple, synvars, dt); text truncated | y | y | y |
| `AbstractSynapseVariable` | SNNModels | abstract type | populations/synapse/synapses.jl | y | y | n | subtype list missing DoubleExpCurrentSynapseVars and Confavreux2025SynapseVars; named DoubleExpSynapseVars etc. without field info | y | y | y |
| `CurrentParameter` | SNNModels | abstract type | stimuli/current.jl | y | y | y |  | y | y | y |
| `LTPParameter (not exported)` | SNNModels | abstract type | connections/sparse_plasticity.jl | n | n | na |  | y | n | n |
| `MetaPlasticityParameter` | SNNModels | abstract type | connections/connections.jl | n | n | na |  | y | n | y |
| `PlasticityParameter` | SNNModels | abstract type | connections/connections.jl | y | y | y | header only | y | y | y |
| `PlasticityVariables` | SNNModels | abstract type | connections/connections.jl | n | y | y | header only | y | y | y |
| `PoissonLayerParameter` | SNNModels | abstract type (not exported) | stimuli/poisson_layer.jl | n | n | na |  | y | n | y |
| `PoissonStimulusParameter` | SNNModels | abstract type | stimuli/poisson.jl | y | n | na |  | y | n | y |
| `STDPParameter (not exported)` | SNNModels | abstract type | connections/sparse_plasticity.jl | n | n | na |  | y | n | n |
| `STDPStructuredParameter (not exported)` | SNNModels | abstract type | connections/sparse_plasticity/STDP_structured.jl | n | n | na |  | y | n | n |
| `STPParameter (not exported)` | SNNModels | abstract type | connections/sparse_plasticity.jl | n | n | na |  | y | n | n |

## Plotting (SNNPlots)

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `@makie_default` | SNNPlots | macro | backend/makie.jl | y | n | na |  | y | n | y |
| `cm` | SNNPlots | Float32 |  | y | n | na | SNNModels length unit (1.0f0) defined by @load_units, shadows Measures.cm; documented in visualization.md prose only | n | n | n |
| `default_colors` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `dendrite_gplot` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `inch` | SNNPlots | Measures.AbsoluteLen |  | y | n | na | re-export of Measures length constant | n | n | n |
| `load_model` | SNNPlots | function | utils/io.jl | y | y | na | re-export of SNNModels function (documented there) | y | y | y |
| `makie_default!` | SNNPlots | function | backend/makie.jl | y | y | y |  | y | y | y |
| `nature_figure` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `okabe_ito_10` | SNNPlots | Vector{ColorTypes.RG |  | y | n | na |  | y | n | y |
| `plot` | SNNPlots | function |  | y | n | na | re-export of Makie.plot/plot! (documented by Makie) | n | n | n |
| `plot!` | SNNPlots | function |  | y | n | na | re-export of Makie.plot/plot! (documented by Makie) | n | n | n |
| `plot_activity` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `plot_connection_distances` | SNNPlots | function | spatial.jl | y | y | n | `probability` kwarg undocumented | y | y | y |
| `plot_connections` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `plot_model` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `plot_spatial_connectivity` | SNNPlots | function | spatial.jl | y | y | n | `do_legend` kwarg and return values undocumented | y | y | y |
| `plot_stimulus` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `plot_weights` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `pt` | SNNPlots | Measures.AbsoluteLen |  | y | n | na | re-export of Measures length constant | n | n | n |
| `raster` | SNNPlots | function | raster.jl | y | n | na |  | y | n | y |
| `raster!` | SNNPlots | function | raster.jl | n | n | na | not exported by SNNPlots; exported by SpikingNeuralNetworks but undefined there (SNN.raster! fails) | y | n | y |
| `save_model` | SNNPlots | function | utils/io.jl | y | y | na | re-export of SNNModels function (documented there) | y | y | y |
| `SNNPlots` | SNNPlots | module | SNNPlots.jl | n | n | na | module docstring added | y | n | y |
| `soma_gplot` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `stdp_kernel` | SNNPlots | function | stdp_plots.jl | y | n | na |  | y | n | y |
| `stdp_kernel!` | SNNPlots | function | stdp_plots.jl | y | y | n | documented as `stdp_kernel(stdp_param; ΔT = -97.5:5:100ms)` returning a Plots.Plot; actual `stdp_kernel!(ax, stdp_param; ΔTs = ...)` returning a Makie plot | y | y | y |
| `stdp_test` | SNNPlots | function | stdp_plots.jl | y | n | na |  | y | n | y |
| `stdp_weight_decorrelated` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `stp_plot` | SNNPlots | undef |  | y | n | na | exported but not defined | n | n | n |
| `vecplot` | SNNPlots | function | vecplot.jl | y | n | na |  | y | n | y |
| `vecplot!` | SNNPlots | function | vecplot.jl | y | n | na |  | y | n | y |

## SNNUtils models

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `sample_stp_campagnola` | SNNUtils | function | models/stp_het.jl | y | n | na |  | y | n | y |
| `sample_stp_params` | SNNUtils | function | models/stp_het.jl | y | n | na |  | y | n | y |

## SNNUtils tools

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `all_intervals` | SNNUtils | function | stimuli/sequence/sequence.jl | y | n | na |  | y | n | y |
| `all_windows` | SNNUtils | function | stimuli/balance_EI/bimodal_kernel.jl | n | n | na | not exported | y | n | y |
| `average_weight` | SNNUtils | function | analysis/weights.jl | n | n | na | not exported | y | n | y |
| `average_weight_dynamics` | SNNUtils | function | analysis/weights.jl | y | y | n | synapse described as 'Receptors object'; record row order (CSC storage order) not stated | y | n | y |
| `bioseq_epochs` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `bioseq_lexicon` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `compute_kei` | SNNUtils | function | stimuli/balance_EI/compute_kei.jl | y | y | n | default Vs documented as -52mV (actual -55mV); did not state that it fails with SNNModels 1.8.4 | y | n | y |
| `critical_window` | SNNUtils | function | stimuli/balance_EI/bimodal_kernel.jl | n | n | na | not exported | y | n | y |
| `do_pca` | SNNUtils | function | analysis/classifiers.jl | y | y | n | claimed to return a PCA object; returns the projected data matrix | y | n | y |
| `generate_balanced_sequence` | SNNUtils | function | stimuli/sequence/sequence.jl | y | n | na |  | y | n | y |
| `generate_lexicon` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | n | field documented as `silence_symbol` (actual `silence`); ph_duration described as scalar | y | n | y |
| `generate_sequence` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | n | signature wrong (documented `generate_sequence(lexicon, config, seed=nothing)`; actual `generate_sequence(seq_function; lexicon, seed=-1, kwargs...)`); return layout undocumented | y | n | y |
| `get_lexicon` | SNNUtils | function | stimuli/sequence/sequence.jl | y | n | na |  | y | n | y |
| `get_model` | SNNUtils | function | stimuli/balance_EI/compute_kei.jl | y | y | n | described as working; it fails with SNNModels 1.8.4 (PostSpike(A=...), EyalEquivalentNAR, synapsearray undefined) | y | n | y |
| `getdictionary` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | n | `insert` argument undocumented | y | n | y |
| `getduration` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | y |  | y | n | y |
| `getneurons` | SNNUtils | function | stimuli/sequence/sequence.jl | y | n | na |  | y | n | y |
| `getphonemes` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | n | did not mention that the silence symbol :_ is appended | y | n | y |
| `getstim` | SNNUtils | function | stimuli/sequence/sequence.jl | y | n | na |  | y | n | y |
| `getstimsym` | SNNUtils | function | stimuli/sequence/sequence.jl | y | n | na |  | y | n | y |
| `import_bioseq_tasks` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `LogRegtrain` | SNNUtils | function | analysis/classifiers.jl | n | n | na | not exported | y | n | y |
| `merge_intervals` | SNNUtils | function | stimuli/sequence/sequence.jl | y | n | na |  | y | n | y |
| `MultinomialLogisticRegression` | SNNUtils | function | analysis/classifiers.jl | y | y | n | described as working; always throws UndefVarError (make_set_index undefined); X modified in place not stated | y | n | y |
| `nmda_curr` | SNNUtils | function | stimuli/balance_EI/compute_kei.jl | n | y | n | described as 'NMDA current'; it is the dimensionless Mg-block factor; not exported | y | n | y |
| `optimal_kei` | SNNUtils | function | stimuli/balance_EI/compute_kei.jl | y | n | na |  | y | n | y |
| `pca` | SNNUtils | undef |  | y | n | na | exported but not defined | n | n | n |
| `residual_current` | SNNUtils | function | stimuli/balance_EI/compute_kei.jl | y | y | n | kwargs shown as optional (they are required); did not state that it fails with SNNModels 1.8.4 | y | n | y |
| `root_path` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `score_spikes` | SNNUtils | function | analysis/classifiers.jl | y | y | n | pop default documented :E (actual :Exc), delay shown positional (keyword), claimed parallel processing (serial) | y | n | y |
| `seq_bioseq` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `sequence_end` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | y |  | y | n | y |
| `set_stimuli!` | SNNUtils | function | stimuli/sequence/stimuli.jl | y | y | n | documented `targets` argument that does not exist; claimed to return the model (returns nothing); did not state the w_/p_ prefixes | y | n | y |
| `sign_intervals` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | y |  | y | n | y |
| `SNNUtils` | SNNUtils | module | SNNUtils.jl | n | n | na | module docstring added | y | n | y |
| `spikecount_features` | SNNUtils | function | analysis/classifiers.jl | y | y | n | return type documented Float32 (actual Float64) | y | n | y |
| `standardize` | SNNUtils | function | analysis/classifiers.jl | y | y | y |  | y | n | y |
| `start_interval` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | y |  | y | n | y |
| `step_input` | SNNUtils | function | stimuli/sequence/stimuli.jl | y | y | n | documented kwarg `lexicon` (actual `inputs`), nonexistent `start_rate`, pop default :E (actual :Exc), return type PoissonStimulus (actual MultiCompartmentStimulusGroup); default targets=[nothing] fails (not stated) | y | n | y |
| `stimuli_names` | SNNUtils | function | stimuli/sequence/sequence.jl | y | n | na |  | y | n | y |
| `store_activity_data` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `store_experiment_data` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `store_labels` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `store_target_pops` | SNNUtils | function | stimuli/bioseq/import_bioseq.jl | y | n | na |  | y | n | y |
| `SVCtrain` | SNNUtils | function | analysis/classifiers.jl | y | y | n | `labels` kwarg undocumented; z-scoring and fallback when sets are too small not stated (otherwise correct) | y | n | y |
| `sym_features` | SNNUtils | function | analysis/classifiers.jl | y | y | n | claimed working and thread-safe; it throws UndefVarError (SNN not defined in SNNUtils) | y | n | y |
| `symbol_names` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | n | docstring titled `symbolnames` and claimed words are prefixed with w_ (they are not; that is stimuli_names) | y | n | y |
| `symbols_to_int` | SNNUtils | function | analysis/classifiers.jl | y | y | y |  | y | n | y |
| `time_in_interval` | SNNUtils | function | stimuli/sequence/sequence.jl | y | y | y |  | y | n | y |
| `trial_average` | SNNUtils | function | analysis/classifiers.jl | y | y | n | claimed arbitrary `dim` works; averaging is along the last dimension of the slices | y | n | y |
| `trial_sort` | SNNUtils | function | analysis/classifiers.jl | y | n | na |  | y | n | y |
| `update_stimuli!` | SNNUtils | function | stimuli/sequence/stimuli.jl | y | y | n | documented `targets` argument that does not exist | y | n | y |
| `word_phonemes_sequence` | SNNUtils | function | stimuli/sequence/sequences/word_phonemes.jl | y | y | n | docstring described a different function `generate_random_word_sequence(sequence_length, dictionary, silence_symbol; ...)`; kwargs mode/presentations/seed/lexicon undocumented | y | n | y |

## Names exported by SpikingNeuralNetworks (`SNN`)

Re-exports keep the docstring of the defining package; 'What was wrong' refers to that docstring.

| Symbol | Pkg | Kind | File | Exp | Doc before | Correct before | What was wrong | Doc after | Site before | Site after |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `@load_units` | SNN | macro | re-export from SNNModels (utils/unit.jl) | y | y | na |  | y | y | y |
| `@makie_default` | SNN | macro | re-export from SNNPlots (backend/makie.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `@update` | SNN | macro | re-export from SNNModels (utils/macros.jl) | y | y | na |  | y | y | y |
| `@update!` | SNN | macro | re-export from SNNModels (utils/macros.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `AdExParameter` | SNN | type (re-export) | populations/generalized_if/adex.jl | y | y | n | see SNNModels row | y | y | y |
| `asynchronous_state` | SNN | function | re-export from SNNModels (analysis/targets.jl) | y | y | na |  | y | y | y |
| `bin_spiketimes` | SNN | function | re-export from SNNModels (analysis/spikes.jl) | y | y | na |  | y | y | y |
| `change_plasticity!` | SNN | function | re-export from SNNModels (connections/sparse_plasticity.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `clear_monitor!` | SNN | function | re-export from SNNModels (utils/record.jl) | y | y | na |  | y | y | y |
| `clear_perturbation_monitor!` | SNN | function | re-export from SNNModels (utils/perturbation.jl) | y | y | na |  | y | y | y |
| `clear_perturbation_records!` | SNN | function | re-export from SNNModels (utils/perturbation.jl) | y | y | na |  | y | y | y |
| `clear_records!` | SNN | function | re-export from SNNModels (utils/record.jl) | y | y | na |  | y | y | y |
| `compose` | SNN | function | re-export from SNNModels (utils/util.jl) | y | y | na |  | y | y | y |
| `compute_connections` | SNN | function | re-export from SNNModels (utils/spatial.jl) | y | y | na |  | y | y | y |
| `DendNeuronParameter` | SNN | type | re-export from SNNModels (populations/multicompartment/dendneuron_parameter.jl) | y | y | na |  | y | y | y |
| `DOCS_ASSETS_PATH` | SNN | String | SpikingNeuralNetworks.jl | y | n | na | no docstring | y | n | y |
| `DoubleExpSynapse` | SNN | type | re-export from SNNModels (populations/synapse/synapses/DoubleExpSynapse.jl) | y | y | na |  | y | y | y |
| `ExtendedIFParameter` | SNN | type (re-export) | populations/generalized_if/if_extended.jl | y | n | na | no docstring (see owner row) | y | n | y |
| `firing_rate` | SNN | function | re-export from SNNModels (analysis/spikes.jl) | y | y | na |  | y | y | y |
| `GABAergic` | SNN | type | re-export from SNNModels (populations/synapse/receptors.jl) | y | y | na |  | y | y | y |
| `get_time` | SNN | function | re-export from SNNModels (utils/record.jl) | y | y | na |  | y | y | y |
| `Glutamatergic` | SNN | type | re-export from SNNModels (populations/synapse/receptors.jl) | y | y | na |  | y | y | y |
| `Identity` | SNN | type (re-export) | populations/identity.jl | y | y | n | see SNNModels row | y | y | y |
| `IFParameter` | SNN | type (re-export) | populations/generalized_if/if.jl | y | y | n | see SNNModels row | y | y | y |
| `iSTDPPotential` | SNN | type | re-export from SNNModels (connections/sparse_plasticity/iSTDP.jl) | y | y | na |  | y | y | y |
| `iSTDPRate` | SNN | type | re-export from SNNModels (connections/sparse_plasticity/iSTDP.jl) | y | y | na |  | y | y | y |
| `load_model` | SNN | function | re-export from SNNModels (utils/io.jl) | y | y | na |  | y | y | y |
| `LTPParam` | SNN | undef | SpikingNeuralNetworks.jl/src/SpikingNeuralNetworks.jl | y | n | na | exported but not defined (it is the SpikingSynapse keyword/field name; the type is LTPParameter); exported but not defined | n | n | n |
| `make_copy` | SNN | undef | SpikingNeuralNetworks.jl export list | y | n | na | exported but not defined | n | n | n |
| `MarkramSTPParameter` | SNN | type | re-export from SNNModels (connections/sparse_plasticity/STP.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `MarkramSTPParameterHet` | SNN | type | re-export from SNNModels (connections/sparse_plasticity/STP.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `matrix` | SNN | function | utils/sparse_matrix.jl | y | n | na | re-export; no docstring (see owner row) | y | n | y |
| `matrix_record` | SNN | function | utils/sparse_matrix.jl | y | n | na | re-export; no docstring (see owner row) | y | n | y |
| `monitor!` | SNN | function | re-export from SNNModels (utils/record.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `MultiCompartmentStimulusGroup` | SNN | function | re-export from SNNModels (stimuli/stimulus_group.jl) | y | y | na |  | y | y | y |
| `MultiplicativeNorm` | SNN | type | connections/metaplasticity/normalization.jl | y | y | n | re-export (see SNNModels row) | y | y | y |
| `name` | SNN | function | re-export from SNNModels (utils/util.jl) | y | y | na |  | y | y | y |
| `NMDAVoltageDependency` | SNN | type | re-export from SNNModels (populations/synapse/receptors.jl) | y | y | na |  | y | y | y |
| `NoLTP` | SNN | type | re-export from SNNModels (connections/sparse_plasticity.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `NoSTP` | SNN | type | re-export from SNNModels (connections/sparse_plasticity.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `okabe_ito_10` | SNN | Vector{ColorTypes.RG | SpikingNeuralNetworks.jl | y | n | na | no docstring (see owner row) | y | n | y |
| `perturbation_record` | SNN | function | re-export from SNNModels (utils/perturbation.jl) | y | y | na |  | y | y | y |
| `perturbation_test` | SNN | function | re-export from SNNModels (utils/perturbation.jl) | y | y | na |  | y | y | y |
| `place_populations` | SNN | function | re-export from SNNModels (utils/spatial.jl) | y | y | na |  | y | y | y |
| `Poisson` | SNN | type (re-export) | populations/poisson.jl | y | y | y | link only | y | y | y |
| `PoissonLayer` | SNN | type | re-export from SNNModels (stimuli/poisson_layer.jl) | y | y | na |  | y | y | y |
| `PoissonParameter` | SNN | abstract type (re-export) | populations/poisson.jl | y | n | na | no docstring (see owner row) | y | n | y |
| `Population` | SNN | function (re-export) | populations/populations.jl | y | n | na | no docstring (see owner row) | y | n | y |
| `PostSpike` | SNN | type (re-export) | populations/spike/postspike.jl | y | y | n | see SNNModels row | y | y | y |
| `raster` | SNN | function | re-export from SNNPlots (raster.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `raster!` | SNN | undef | SpikingNeuralNetworks.jl export list | y | n | na | exported but not defined | n | n | y |
| `Receptor` | SNN | type | re-export from SNNModels (populations/synapse/receptors.jl) | y | y | na |  | y | y | y |
| `Receptors` | SNN | function | re-export from SNNModels (populations/synapse/receptors.jl) | y | y | na |  | y | y | y |
| `ReceptorSynapse` | SNN | type | re-export from SNNModels (populations/synapse/synapses/ReceptorSynapse.jl) | y | y | na |  | y | y | y |
| `ReceptorVoltage` | SNN | type | re-export from SNNModels (populations/synapse/receptors.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `record` | SNN | function | re-export from SNNModels (utils/record.jl;stimuli/stimulus_group.jl) | y | y | na |  | y | y | y |
| `record!` | SNN | function | re-export from SNNModels (utils/record.jl) | y | y | na |  | y | y | y |
| `reset_time!` | SNN | function | re-export from SNNModels (utils/record.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `sample_inputs` | SNN | function | re-export from SNNModels (analysis/spikes.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `save_model` | SNN | function | re-export from SNNModels (utils/io.jl) | y | y | na |  | y | y | y |
| `set_LTP!` | SNN | function | re-export from SNNModels (connections/sparse_plasticity.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `set_plasticity!` | SNN | function | utils/sparse_matrix.jl;connections/sparse_plasticity.jl | y | n | na | re-export; no docstring (see owner row) | y | n | y |
| `set_STP!` | SNN | function | re-export from SNNModels (connections/sparse_plasticity.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `sim!` | SNN | function | re-export from SNNModels (utils/main.jl) | y | y | na |  | y | y | y |
| `SingleExpSynapse` | SNN | type | re-export from SNNModels (populations/synapse/synapses/SingleExpSynapse.jl) | y | y | na |  | y | y | y |
| `SNN` | SNN | module | alias of SpikingNeuralNetworks (module) | y | n | na | no docstring (module docstring added to SpikingNeuralNetworks) | y | n | n |
| `SNNload` | SNN | function | re-export from SNNModels (utils/io.jl) | y | y | na |  | y | y | y |
| `SNNModel` | SNN | undef | SpikingNeuralNetworks.jl export list | y | n | na | exported but not defined | n | n | n |
| `SNNModels` | SNN | module | module SNNModels | y | n | na | no docstring (see owner row) | y | n | y |
| `SNNPlots` | SNN | module | module SNNPlots | y | n | na | no docstring (see owner row) | y | n | y |
| `SNNsave` | SNN | function | re-export from SNNModels (utils/io.jl) | y | y | na |  | y | y | y |
| `SNNUtils` | SNN | module | module SNNUtils | y | n | na | no docstring (see owner row) | y | n | y |
| `SpikeTimeParameter` | SNN | function | re-export from SNNModels (stimuli/timed.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `SpikeTimeStimulus` | SNN | type | re-export from SNNModels (stimuli/timed.jl) | y | y | na |  | y | y | y |
| `SpikeTimeStimulusParameter` | SNN | type | re-export from SNNModels (stimuli/timed.jl) | y | y | na |  | y | y | y |
| `SpikingSynapse` | SNN | type | connections/spiking_synapse.jl | y | y | y | re-export (see SNNModels row) | y | y | y |
| `SpikingSynapseParameter` | SNN | type | connections/spiking_synapse.jl | y | y | n | re-export (see SNNModels row) | y | y | y |
| `Stimulus` | SNN | function | re-export from SNNModels (stimuli/poisson_layer.jl;stimuli/timed.jl;stimuli/current.jl) | y | y | na |  | y | y | y |
| `StimulusGroup` | SNN | type | re-export from SNNModels (stimuli/stimulus_group.jl) | y | y | na |  | y | y | y |
| `STPParam` | SNN | undef | SpikingNeuralNetworks.jl/src/SpikingNeuralNetworks.jl | y | n | na | exported but not defined (keyword/field name; the type is STPParameter); exported but not defined | n | n | n |
| `str_name` | SNN | function | re-export from SNNModels (utils/util.jl) | y | y | na |  | y | y | y |
| `SVCtrain` | SNN | function | re-export from SNNUtils (analysis/classifiers.jl) | y | y | na |  | y | n | y |
| `train!` | SNN | function | re-export from SNNModels (utils/main.jl) | y | y | na |  | y | y | y |
| `TripodParameter` | SNN | function | re-export from SNNModels (populations/multicompartment/dendneuron_parameter.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `update_spikes!` | SNN | function | re-export from SNNModels (stimuli/timed.jl) | y | y | na |  | y | y | y |
| `update_traces!` | SNN | function | re-export from SNNModels (populations/populations.jl;connections/sparse_plasticity.jl;connections/connections.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `vecplot` | SNN | function | re-export from SNNPlots (vecplot.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `vecplot!` | SNN | function | re-export from SNNPlots (vecplot.jl) | y | n | na | no docstring (see owner row) | y | n | y |
| `vSTDPParameter` | SNN | type | re-export from SNNModels (connections/sparse_plasticity/vSTDP.jl) | y | y | na |  | y | y | y |

