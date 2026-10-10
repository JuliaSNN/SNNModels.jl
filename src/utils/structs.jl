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
    Time{FT}(; t = [zero(FT)], tt = Int32[0], dt = FT(0.125))

Mutable simulation clock shared by all components of a model. `Time()` is a `Time{Float32}`; a model
whose populations are `Float64` (see [`float_type`](@ref)) gets a `Time{Float64}` from `compose`.

The values are stored in one-element vectors so that the clock can be shared and updated in place
by `update_time!`. Use `get_time`, `get_step`, `get_dt` and `reset_time!` to read or reset it.

# Fields
- `t::Vector{FT} = [0.0f0]`: current time (ms), `t[1]`.
- `tt::Vector{Int32} = Int32[0]`: number of integration steps performed, `tt[1]`.
- `dt::FT = 0.125f0`: time step (ms) of the last update.

# Example
```julia
using SpikingNeuralNetworks
T = SNN.Time()
SNN.get_time(T)   # 0.0f0
```
"""
Time

mutable struct Time{FT<:AbstractFloat}
    t::Vector{FT}
    tt::Vector{Int32}
    dt::FT
end

Time{FT}(; t = [zero(FT)], tt = Int32[0], dt = FT(0.125)) where {FT<:AbstractFloat} =
    Time{FT}(Vector{FT}(t), Vector{Int32}(tt), FT(dt))
Time(; kwargs...) = Time{Float32}(; kwargs...)

"""
    Time(time::Number; dt = 0.125f0)
    Time{FT}(time::Number; dt = 0.125)

Create a clock set at `time` (ms), with time step `dt` and step counter
`tt = round(time / dt)` (`Float32` unless `FT` is given). (Up to SNNModels 1.8.4 `dt` was fixed to 0.125 ms and a `time` that
was not a multiple of it threw an `InexactError`.)

# Example
```julia
using SpikingNeuralNetworks
T = SNN.Time(100.0)   # t = 100 ms, tt = 800
```
"""
function Time{FT}(time::Number; dt::Real = 0.125) where {FT<:AbstractFloat}
    tts = round(Int32, FT(time) / FT(dt))
    return Time{FT}([FT(time)], Int32[tts], FT(dt))
end
Time(time::Number; dt::Real = 0.125f0) = Time{Float32}(time; dt)

"""
    float_type(x)

Floating-point element type of a model object: `Float32` or `Float64`. The precision of a model is chosen once, on
its populations (a population built with `Vector{Float64}` state, for example `Identity(N = 10, VFT = Vector{Float64})`), and every
connection and stimulus built on a population takes the type of the arrays it writes into (the target `g` of
the population): `float_type(c) = eltype(c.W)` for a sparse synapse, `eltype(g)` for a stimulus. The state of the
plasticity rules, the delay buffers, the caches and the clock follow. For a type it is the first type parameter that
is a floating-point type or a vector of one (`Identity{Vector{Float64},Int32}` gives `Float64`), `Float32` if there is none.
For a network (a `NamedTuple` as returned by `compose`) it is the promotion of the types of the populations.
"""
# generated: the answer is a constant of the type, so a call with a known type costs nothing in the step loop
@generated float_type(::Type{T}) where {T} = (R = something(_float_type(T), Float32); :($R))
float_type(x) = float_type(typeof(x))
_float_type(::Type{T}) where {T<:AbstractFloat} = T
_float_type(::Type{<:AbstractArray{T}}) where {T<:AbstractFloat} = T
function _float_type(::Type{P}) where {P}
    ps = P isa DataType ? P.parameters : ()
    for p in ps   # a float, or a vector of floats, among the type parameters
        p isa Type && (p <: AbstractFloat || p <: AbstractArray{<:AbstractFloat}) && return _float_type(p)
    end
    for p in ps   # parameters that are themselves parametric (PostSpike{Float32}, ...)
        if p isa DataType && !isempty(p.parameters)
            T = _float_type(p)
            isnothing(T) || return T
        end
    end
    return nothing
end

# Components whose model is only implemented in Float32 refuse a target of another precision at construction (their
# `VFT = Vector{Float32}` would otherwise copy the target variable and silently stop writing into it).
_require_float32(what, arrays...) = all(a -> eltype(a) === Float32, arrays) ? nothing : throw(ArgumentError(
    "$what is only implemented for Float32 (the variable it acts on has element type $(eltype(first(arrays)))). " *
    "Float64 is implemented for: populations Identity, IF; connections SpikingSynapse (all plasticity rules), " *
    "AggregateScaling, SynapseNormalization; stimuli SpikeTimeStimulus, PoissonStimulus, PoissonStimulusLayer. See `float_type`."))

export Spiketimes, Time, NetworkModel, float_type, convert_precision
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
