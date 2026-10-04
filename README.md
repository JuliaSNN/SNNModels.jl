<div align="center">
    <img src="https://github.com/JuliaSNN/SpikingNeuralNetworks.jl/blob/main/docs/src/assets/SNNLogo.svg" alt="SpikingNeuralNetworks.jl" width="200">
</div>

<h2 align="center"> Models, types, and functions for Julia SpikingNeuralNetworks.jl 
<p align="center">
    <a href="https://github.com/JuliaSNN/SNNModels.jl/actions">
    <img src="https://github.com/JuliaSNN/SNNModels.jl/workflows/CI/badge.svg"
         alt="Build Status">
  </a>
  <a href="https://juliasnn.github.io/SpikingNeuralNetworks.jl/dev/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="stable documentation">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yelllow"
       alt="bibtex">
  </a>

</p>
</h2>


# SNNModels

The package is the Base package of [SpikingNeuralNetworks.jl](https://github.com/JuliaSNN/SpikingNeuralNetworks.jl). It contains model types and core functionalities for the SpikingNeuralNetworks.jl ecosystem.

## Documentation

The package defines models and parameters for `Population`, `Connection`, and `Stimulus`:

- `Population <: AbstractPopulation`
- `Connection <: AbstractConnection`
- `Stimulus <: AbstractStimulus`
- `PopulationParameter <: AbstractPopulationParameter`
- `ConnectionParameter <: AbstractConnectionParameter`
- `StimulusParameter <: AbstractStimulusParameter`
- `SpikeTimes = Vector{Vector{Float32}}`.

Populations, connections, and stimuli are defined under the respective folders in `src`

Under `src/utils`, the package defines macros and functions that support the functionalities of the SpikingNeuralNetwork.jl ecosystem:

- `struct.jl` defines the abstract model types.
- `main.jl` defines the `sim!` and `train!` functions that run the network simulations. 
- `io.jl` defines functions to save and load models using `.jld2` format.
- `record.jl` implements the recording ofthe  model's variables during simulation time.
- `macros.jl` implements useful macros to define model types and update parameter structs.
- `spatial.jl` defines functions to create spatial network arrangements.
- `unit.jl` defines convenient shortcut for _cgm_ unit system.
- `util.jl` add functions to manipulate sparse matrix representations.

## Functioning

The library leverages Julia multidispatching to run models of types ` <: AbstractPopulation`,
`<: AbstractConnection`, and `AbstractStimulus`. 

```julia
function sim!(p::Vector{AbstractPopulation}, c::Vector{AbstractConnection}, duration<:Real) end
function train!(p::Vector{AbstractConnection}, c:Vector{AbstractConnection}, duration<:Real) end
```

The functions support simulation with and without neural plasticity: `sim!` propagates spikes with frozen weights, `train!` additionally calls `update_traces!` and `plasticity!` (STDP, iSTDP, vSTDP, STP). A synapse with an `LTPParam` does not learn under `sim!`. The model is defined within the arguments passed to the functions. 
Models are composed of 'AbstractPopulation' and 'AbstractConnection' arrays. 

Any elements of `AbstractPopulation` must implement the methods: 
```julia
function integrate!(p, p.param, dt) end
function plasticity!(p, p.param, dt, T) end

```

`AbstractConnection` must implement the methods: 

```julia
function forward!(p, p.param) end
function plasticity!(c, c.param, dt) end
```


`AbstractStimulus` must implement the methods: 

```julia
function stimulate!(p, p.param) end
```

## Plasticity rules

Long-term plasticity rules are passed to `SpikingSynapse` with the `LTPParam` keyword and run under `train!`.

| Rule | Description |
|---|---|
| `STDPGerstner` | additive pair STDP, all-to-all, signed amplitudes `A_pre`, `A_post` (default `A_post < 0`) |
| `STDPTriplet` | minimal triplet rule of Pfister & Gerstner (2006), all-to-all (Auryn `MinimalTriplet`) |
| `STDPWeightDependent` | soft-bound pair STDP of Gütig et al. (2003) (Auryn `STDPwd`) |
| `STDPConfavreux2025` | pair STDP with rate terms |
| `STDPMexicanHat`, `STDPSymmetric`, `STDPAntiSymmetric` | kernels with zero integral / structured inhibition (Euler traces) |
| `iSTDPRate`, `iSTDPPotential` | inhibitory STDP of Vogels et al. (2011) |
| `vSTDPParameter` | voltage-based STDP of Clopath et al. (2010) |

The trace-based pair and triplet rules are event-driven with Auryn's ordering (pre spike: LTD
from the postsynaptic trace over outgoing synapses; post spike: LTP from the presynaptic trace
over incoming synapses; traces read before the current step's spikes). They agree with Brian2
(2e-6) and Auryn (2e-7). Details: `src/connections/sparse_plasticity/STDP_kernels.jl` and
`docs/stdp_rules_memo.md`.

## Release notes: 1.9

- **Bug fix, iSTDP.** In `iSTDPRate` (and the former `iSTDPTime`) the potentiation at a
  postsynaptic spike was applied to the wrong synapses (a `@turbo` loop with a reassigned loop
  variable). Affected: SNNModels 1.5.0 - 1.8.1 and SpikingNeuralNetworks.jl 1.0.0 onwards.
  Results obtained with these versions change; rerun simulations that used them.
- **Behaviour change, `STDPGerstner`.** The amplitudes `A_pre`/`A_post` were applied twice
  (effective `A^2`, sign lost). They are now applied once, and the default `A_post` is `-1e-4`.
- **New rules:** `STDPTriplet`, `STDPWeightDependent`.
- **Behaviour change, `sparse_matrix`.** Built directly in CSC form (no dense `Npost x Npre`
  matrix); seeded networks no longer reproduce earlier realisations (identical statistics);
  autapses are removed structurally. The old generator is `SNNModels.sparse_matrix_dense_legacy`.
- **Float32** synaptic data (weights, delays, STP `ρ`) at all constructor boundaries.
- Event-driven STDP is 18-83x faster than the clock-driven implementation (see `claude/performance.md`).
