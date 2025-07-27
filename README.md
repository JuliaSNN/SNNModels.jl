# SNNModels

The package contains model types and core functionalities for the SpikingNeuralNetworks.jl ecosystem.

## Documentation

The package defines models and parameters for `Population`, `Connection`, and `Stimulus`:

- `Population <: AbstractPopulation`
- `Connection <: AbstractConnection`
- `Stimulus <: AbstractStimulus`
- `PopulationParameter <: AbstractPopulationParameter`
- `ConnectionParameter <: AbstractConnectionParameter`
- `StimulusParameter <: AbstractStimulusParameter`
- `SpikeTimes = Vector{Vector{Float32}}`.

Populations, connections and stimuli are defined under the respective folders in `src`

Under `src/utils` the package defines macros and functions that support the functionalities of the SpikingNeuralNetwork.jl ecosystem:

- `struct.jl` defines the abstract model types.
- `main.jl` defines the `sim!` and `train!` functions that run the network simulations. 
- `io.jl` defines functions to save and load models using `.jld2` format.
- `record.jl` implements the recording of model's variables during simulation time.
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

The functions support simulation with and without neural plasticity; the model is defined within the arguments passed to the functions. 
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
