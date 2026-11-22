# Validation of the sweep bug fixes (SNNModels/SNNPlots/SNNUtils branch fix/sweep-bugs)

Fixed vs unfixed comparison. "unfixed" is the `docs/sweep` commit of SNNModels (779f5a8), whose
code is identical to the released SNNModels 1.8.4. Variants for the decomposition of the Tripod
fixes are scratch commits (not branches) listed in `run_all.sh`. BallAndStick ignored its
external currents in 1.8.4, so its current-step protocols use scratch variants in which only
`Is`/`Id` are applied (`unfixedIs`, and the same patch in `dtonly`/`glonly`).

- `measure_multicompartment.jl <variant>`: Tripod/BallAndStick single neuron (f-I, rheobase,
  waveform, adaptation, ISIs, EPSP attenuation, NMDA plateau, bAP, dt study) -> `data/mc_*.jld2`
- `network_model.jl`, `measure_network.jl <variant> [seed]`: Tripod E/I network -> `data/net_*.jld2`
- `measure_other.jl <variant>`: HH/MorrisLecar, vSTDP, AggregateScaling, HetRec, Confavreux,
  recording axis, stdp_kernel, Rate, AdditiveNorm, BalancedStimulus -> `data/other_*.jld2`
- `summarize_*.jl`: tables `data/*_summary.txt`
- `plot_*.jl`: figures in `figures/` (SNNPlots style), regenerated from the JLD2 files
- `run_all.sh`: runs everything
