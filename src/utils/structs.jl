"""
    AbstractParameter

An abstract type representing a parameter.
"""
abstract type AbstractParameter end

abstract type AbstractComponent end

"""
    AbstractConnectionParameter <: AbstractParameter

An abstract type representing a connection parameter.
"""
abstract type AbstractConnectionParameter <: AbstractParameter end

"""
    AbstractPopulationParameter <: AbstractParameter

An abstract type representing a population parameter.
"""
abstract type AbstractPopulationParameter <: AbstractParameter end

"""
    AbstractStimulusParameter <: AbstractParameter

An abstract type representing a stimulus parameter.
"""
abstract type AbstractStimulusParameter <: AbstractParameter end

"""
    AbstractConnection

An abstract type representing a connection. Any struct inheriting from this type must implement:

# Methods
- `forward!(c::Receptors, param::SynapseParameter)`: Propagates the signal through the synapse.
- `plasticity!(c::Receptors, param::SynapseParameter, dt::Float32, T::Time)`: Updates the synapse parameters based on plasticity rules.
"""
abstract type AbstractConnection <: AbstractComponent end

"""
    AbstractPopulation

An abstract type representing a population. Any struct inheriting from this type must implement:

# Methods
- `integrate!(p::NeuronModel, param::NeuronModelParam, dt::Float32)`: Integrates the neuron model over a time step `dt` using the given parameters.
"""
abstract type AbstractPopulation <: AbstractComponent end

"""
    AbstractStimulus

An abstract type representing a stimulus. Any struct inheriting from this type must implement:

# Methods
- `stimulate!(p::Stimulus, param::StimulusParameter, time::Time, dt::Float32)`: Applies the stimulus to the population.
"""
abstract type AbstractStimulus <: AbstractComponent end

"""
    AbstractSparseSynapse <: AbstractConnection

An abstract type representing a sparse synapse connection.
"""
abstract type AbstractSparseSynapse <: AbstractConnection end

"""
    AbstractNormalization <: AbstractConnection

An abstract type representing a normalization connection.
"""
abstract type AbstractNormalization <: AbstractConnection end

"""
    PlasticityVariables

An abstract type representing plasticity variables.
"""
abstract type PlasticityVariables end

"""
    Spiketimes

A type alias for a vector of vectors of Float32, representing spike times.
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
    struct Time
    Time

A mutable struct representing time. 
A mutable struct representing time.

# Fields
- `t::Vector{Float32}`: A vector containing the current time.
- `tt::Vector{Int}`: A vector containing the current time step.
- `dt::Float32`: The time step size.

"""
Time

@kwdef mutable struct Time
    t::Vector{Float32} = [0.0f0]
    tt::Vector{Int32} = Int32[0]
    dt::Float32 = 0.125f0
end

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
export AbstractConnection, AbstractPopulation, AbstractStimulus


NetworkModel = NamedTuple

VBT = Vector{Bool}
VIT = Vector{Int}
# VDT =Dict{Symbol,Any}


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
