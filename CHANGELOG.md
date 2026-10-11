# SNNModels changelog

The full, itemised development notes (every change with its measurement and migration) are in
`claude/CHANGELOG.md`. This file is the release summary.

## 2.0.0

Major release: types of the synapses, populations and stimuli changed (concretely typed plasticity rules,
spike lists, lazy synapse arrays), and saved models of 1.9.1 and earlier are refused at load time. Dependent
packages must use SNNModels 2: SNNUtils 2, SNNPlots 2, SpikingNeuralNetworks 2, CalciumSpike 2.

### Performance (single core, same machine, Float32)
- E/I balanced LIF network (K = 100): 8 to 12 times faster than 1.9.1; about 1.1 to 1.7 ns per neuron-step
  at N = 1e4, level with Auryn and Brian2 (cpp_standalone) on one core up to the sizes measured (to 5e4).
- Synapse memory: model arrays 29.2 to 9.2 bytes per synapse (`J`, `index`, `rho` are built only when a
  plasticity rule or a reader needs them; delays stored once); peak RSS slope 49.6 to 34.7 bytes per synapse.
- Event-driven voltage-based STDP (`vSTDPParameter`; dense rule is `vSTDPDenseParameter`) and lazy trace decay.
- Delays use a ring buffer; the delayed `forward!` of a common delay is 1.3 to 1.5 times cheaper (Poisson at
  500 Hz: from 1.46 times Auryn's time to 0.86 times, 8 threads).
- One spike list per population (IF, AdEx, ExtendedIF, HetRec, Tripod, BallAndStick, HH, IZ, MorrisLecar),
  shared by the recorder and the connections; recording overhead for IF at N = 1e4 from 1.19 to 1.06.
- Tripod and BallAndStick Heun loops vectorised (2.0 to 2.3 times faster per neuron-step).
- Flush-to-zero of subnormal numbers during `sim!`/`train!` (`flush_subnormals = true`): small networks
  (N = 500) no longer slow down in quiet phases (0.051 to 0.009 s per simulated second).
- No allocation per `sim!`/`train!` step; `using SNNModels` 2.58 to 2.03 s and 190 to 138 packages (unused
  dependencies, among them Documenter, removed).
- Opt-in thread parallelism of the population updates (`neurons_per_thread`, Polyester `@batch`), off by default.

### New
- Float64 selected by the population type (`VFT = Vector{Float64}`) for every model except those listed in the
  changelog; cost reported there.
- `with_plasticity`, `without_plasticity`, `change_plasticity(model, name; LTP, STP)` (new model with the
  synapse replaced): plasticity can be switched on and off; switching off returns the never-plastic memory
  and speed; delayed spikes in flight survive the switch.
- `ensure_transpose!`, `plasticity_value(syn, :W)` (read weights of lazy and event-driven rules).
- Recorder fed from the spike lists.

### Breaking
- `SpikingSynapse`, `PoissonStimulus`, `PoissonStimulusLayer`, `SpikeTimeStimulus` and the IF-family
  populations have new fields and type parameters; `LTPParam`/`STPParam` are type parameters (use
  `with_plasticity`/`change_plasticity`); `syn.J`, `syn.index`, `syn.rho` are empty until needed.
- Saved models and parameters of 1.9.1 and earlier are refused by `load_model`.
- Random streams changed: Poisson and sparse stimuli draw differently, results of stochastic runs differ.
- `vSTDPParameter` is event-driven; `iSTDPTime` removed (it could not run).
- Delays are rounded up to whole steps (`ceil(delay / dt)`).
- Removed (not loaded since 1.x): Multipod, IF_CANAHP, AdExMultiTimescale, `VariableInputParameter`
  (`variable_inputs.jl`), CaRule drafts.
- Bug fix with a behaviour change: Tripod and BallAndStick now use `param.physiology` (default
  `human_dend`, unchanged).

See `claude/CHANGELOG.md` for the migration of every item.
