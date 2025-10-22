"""
    AbstractSynapseParameter <: AbstractComponent

Abstract type representing parameters for synaptic models.

# Methods

- `get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractSynapseParameter}`: Returns the appropriate synapse symbol based on input.

- `synaptic_variables(synapse::AbstractSynapseParameter, N::Int)`: Creates synaptic variables for a given synapse type and number of neurons.

- `update_synapses!(p::P, synapse::T, glu::Vector{Float32}, gaba::Vector{Float32}, synvars::AbstractSynapseVariable, dt::Float32) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}`: Updates synaptic variables.

- `synaptic_current!(p::P, synapse::T, synvars::AbstractSynapseVariable, v::VT1, syncurr::VT2) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter, VT1 <:AbstractVector, VT2 <:AbstractVector}`: Computes synaptic current.

"""
abstract type AbstractSynapseParameter <: AbstractComponent end

"""
    AbstractSynapseVariable <: AbstractComponent
Abstract type representing synaptic variables for synaptic models.

This type serves as a base for various synaptic variable implementations corresponding to different synapse parameter types. The synaptic variables store state information necessary for simulating synaptic dynamics.

# Available subtypes
- `CurrentSynapseVars`
- `DeltaSynapseVars`
- `DoubleExpSynapseVars`
- `SingleExpSynapseVars`
- `ReceptorSynapseVars`
"""
abstract type AbstractSynapseVariable <: AbstractComponent end

include("CurrentSynapse.jl")
include("DeltaSynapse.jl")
include("DoubleExpSynapse.jl")
include("SingleExpSynapse.jl")
include("ReceptorSynapse.jl")

function get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractSynapseParameter}
    sym == :glu && return :glu
    sym == :gaba && return :gaba
    sym == :he && return :glu
    sym == :hi && return :gaba
    sym == :ge && return :glu
    sym == :gi && return :gaba
    error("Synapse symbol $sym not found in DoubleExpSynapse")
end

function synaptic_variables(synapse::AbstractSynapseParameter, N::Int)
    error("synaptic_variables not implemented for synapse type $(typeof(synapse))")
end

function update_synapses!(
    p::P,
    synapse::T,
    glu::Vector{Float32},
    gaba::Vector{Float32},
    synvars::AbstractSynapseVariable,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    error("update_synapses! not implemented for synapse type $(typeof(synapse))")
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::AbstractSynapseVariable,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {
    P<:AbstractGeneralizedIF,
    T<:AbstractSinExpParameter,
    VT1<:AbstractVector,
    VT2<:AbstractVector,
}
    error("synaptic_current! not implemented for synapse type $(typeof(synapse))")
end

export synaptic_current!,
    update_synapses!,
    synaptic_variables,
    synaptic_target,
    get_synapse_symbols,
    CurrentSynapse,
    DeltaSynapse,
    DoubleExpSynapse,
    SingleExpSynapse,
    ReceptorSynapse,
    AbstractSynapseParameter,
    AbstractSynapseVariable
