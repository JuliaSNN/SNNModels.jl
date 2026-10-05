"""
    AbstractParameter

Root of all parameter types (population, connection and stimulus parameters).

Every model object stores its parameter struct in the field `param`; the simulation loop
dispatches `integrate!`, `forward!` and `stimulate!` on `typeof(obj.param)`. Direct subtypes are
`AbstractPopulationParameter`, `AbstractConnectionParameter` and `AbstractStimulusParameter`.
"""
abstract type AbstractParameter end

"""
    AbstractComponent

Root of all network components: populations (`AbstractPopulation`), connections
(`AbstractConnection`), stimuli (`AbstractStimulus`), groups of stimuli (`AbstractGroup`) and
synapse models (`AbstractSynapseParameter`, `AbstractSynapseVariable`).
"""
abstract type AbstractComponent end

"""
    AbstractGroup <: AbstractComponent

Abstract type for containers of components that act as one unit (e.g. `StimulusGroup`).
"""
abstract type AbstractGroup <: AbstractComponent end
"""
    Spiketimes

Type alias for `Vector{Vector{Float32}}`: one vector of spike times (ms) per neuron.

# Example
```julia
using SpikingNeuralNetworks
st = SNN.Spiketimes([[1.0f0, 5.0f0], Float32[]])   # neuron 1 fires at 1 and 5 ms, neuron 2 never
```
"""
Spiketimes = Vector{Vector{Float32}}

"""
    EmptyParam

A struct representing an empty parameter.

# Fields
- `type::Symbol`: The type of the parameter, default is `:empty`.
"""
EmptyParam

@snn_kw struct EmptyParam
    type::Symbol = :empty
end

"""
    Time(; t = [0.0f0], tt = Int32[0], dt = 0.125f0)

Mutable simulation clock shared by all components of a model.

The values are stored in one-element vectors so that the clock can be shared and updated in place
by `update_time!`. Use `get_time`, `get_step`, `get_dt` and `reset_time!` to read or reset it.

# Fields
- `t::Vector{Float32} = [0.0f0]`: current time (ms), `t[1]`.
- `tt::Vector{Int32} = Int32[0]`: number of integration steps performed, `tt[1]`.
- `dt::Float32 = 0.125f0`: time step (ms) of the last update.

# Example
```julia
using SpikingNeuralNetworks
T = SNN.Time()
SNN.get_time(T)   # 0.0f0
```
"""
Time

@kwdef mutable struct Time
    t::Vector{Float32} = [0.0f0]
    tt::Vector{Int32} = Int32[0]
    dt::Float32 = 0.125f0
end

"""
    Time(time::Number)

Create a clock set at `time` (ms), with `dt = 0.125f0` and step counter `tt = time / 0.125`.

`time` must be an integer multiple of 0.125 ms, otherwise the conversion `Int32(time / 0.125)`
throws an `InexactError`.

# Example
```julia
using SpikingNeuralNetworks
T = SNN.Time(100.0)   # t = 100 ms, tt = 800
```
"""
function Time(time::Number)
    tts = time / 0.125f0
    return Time([Float32(time)], Int32[Int32(tts)], 0.125f0)
end

export Spiketimes, Time, NetworkModel
export AbstractParameter,
    AbstractComponent,
    AbstractConnectionParameter,
    AbstractPopulationParameter,
    AbstractStimulusParameter


"""
    AbstractStimulus <: AbstractComponent

Abstract type of external inputs. A concrete stimulus has at least the fields `param`
(an `AbstractStimulusParameter`), `id`, `name` and `records`, and implements

- `stimulate!(s, param, T::Time, dt::Float32)`: called once per time step, before the populations
  are integrated, to write the input into the target population variable.
"""
abstract type AbstractStimulus <: AbstractComponent end

"""
    AbstractStimulusGroup <: AbstractGroup

Abstract type of groups of stimuli. When a model is simulated, the `elements` of a group are
expanded and each element is stimulated as an individual `AbstractStimulus`
(see `sim!`/`train!`).
"""
abstract type AbstractStimulusGroup <: AbstractGroup end

"""
    AbstractPopulation <: AbstractComponent

Abstract type of neuron populations. A concrete population has at least the fields `N`, `param`
(an `AbstractPopulationParameter`), `id`, `name` and `records` (checked by
`validate_population_model`), and implements

- `integrate!(p, param, dt::Float32)`: advance the state by one step `dt` (called by `sim!` and
  `train!`).
- optionally `update_traces!(p, param, dt, T)` and `plasticity!(p, param, dt, T)`, called only by
  `train!` (the defaults do nothing).
"""
abstract type AbstractPopulation <: AbstractComponent end

"""
    AbstractConnection <: AbstractComponent

Abstract type of connections between populations (synapses, metaplasticity operators). A concrete
connection has at least the fields `param` (an `AbstractConnectionParameter`), `id`, `name` and
`records` (checked by `validate_synapse_model`), and implements

- `forward!(c, param, dt::Float32, T::Time)`: propagate presynaptic activity to the postsynaptic
  target variable; called every step by `sim!` and `train!`, after all populations are integrated.
- optionally `update_traces!(c, param, dt, T)` and `plasticity!(c, param, dt, T)`, called only by
  `train!`.
"""
abstract type AbstractConnection <: AbstractComponent end


export AbstractConnection, AbstractPopulation, AbstractStimulus
Component = Union{AbstractPopulation, AbstractConnection, AbstractStimulus}



"""
    NetworkModel

Alias of `NamedTuple`. A network model, as returned by `compose`, is a `NamedTuple` with fields
`pop`, `syn`, `stim` (each a `NamedTuple` of components), `time::Time` and `name::String`.
"""
NetworkModel = NamedTuple

VBT = Vector{Bool}
VIT = Vector{Int}
# VDT =Dict{Symbol,Any}


"""
    isa_model(model)

Validate that a model has the required structure for a network model.

# Arguments
- `model`: The model to validate

# Returns
- `true` if valid

# Throws
- AssertionError if required fields (pop, syn, stim, time, name) are missing
- AssertionError if any component fails validation

# Details
- Checks for presence of all required fields
- Validates each population, synapse, and stimulus
- Ensures time field is a Time struct
"""
function isa_model(model)
    # assert it has all the fields of a network model
    @assert hasproperty(model, :pop)
    @assert hasproperty(model, :syn)
    @assert hasproperty(model, :stim)
    @assert hasproperty(model, :time)
    @assert hasproperty(model, :name)
    for p in values(model.pop)
        validate_population_model(p)
    end
    for s in values(model.syn)
        validate_synapse_model(s)
    end
    for st in values(model.stim)
        validate_stimulus_model(st)
    end
    @assert typeof(model.time) <: Time
    return true
end

"""
    validate_population_model(model)

Validate a population model structure.

# Arguments
- `model`: The population model to validate

# Throws
- AssertionError if model doesn't inherit from AbstractPopulation
- AssertionError if param doesn't inherit from AbstractPopulationParameter  
- AssertionError if required fields (N, param, id, name, records) are missing
"""
function validate_population_model(model)
    # Validate the population model structure and types
    @assert typeof(model) <: AbstractPopulation "Population $(model.name) must inherit from AbstractPopulation"
    @assert typeof(model.param) <: AbstractPopulationParameter "Population $(model.name) must inherit from AbstractPopulationParameter"

    # Validate required fields
    required_fields = [:N, :param, :id, :name, :records]
    for field in required_fields
        @assert hasproperty(model, field) "Population $(model.name) must have a field $(field)"
    end
end

"""
    validate_synapse_model(model)

Validate a synapse/connection model structure.

# Arguments
- `model`: The synapse model to validate

# Throws
- AssertionError if model doesn't inherit from AbstractConnection
- AssertionError if param doesn't inherit from AbstractConnectionParameter
- AssertionError if required fields (param, id, name, records) are missing
"""
function validate_synapse_model(model)
    # Validate the synapse model structure and types
    @assert typeof(model) <: AbstractConnection "Receptors $(model.name) must inherit from AbstractConnection"
    @assert typeof(model.param) <: AbstractConnectionParameter "Receptors $(model.name) must inherit from AbstractConnectionParameter"

    # Validate required fields
    required_fields = [:param, :id, :name, :records]
    for field in required_fields
        @assert hasproperty(model, field) "Receptors  $(model.name) must have a field $(field)"
    end
end

"""
    validate_stimulus_model(model)

Validate a stimulus model structure.

# Arguments
- `model`: The stimulus model to validate

# Throws
- AssertionError if model doesn't inherit from AbstractStimulus
- AssertionError if param doesn't inherit from AbstractStimulusParameter
- AssertionError if required fields (param, id, name, records) are missing
"""
function validate_stimulus_model(model)
    # Validate the stimulus model structure and types
    @assert typeof(model) <: AbstractStimulus "Stimulus $(model.name) must inherit from AbstractStimulus"
    @assert typeof(model.param) <: AbstractStimulusParameter "Stimulus $(model.name) parameter must inherit from AbstractStimulusParameter"

    # Validate required fields
    required_fields = [:param, :id, :name, :records]
    for field in required_fields
        @assert hasproperty(model, field) "Stimulus $(model.name) must have a field $(field)"
    end
end
