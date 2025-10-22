abstract type AbstractSinExpParameter <: AbstractSynapseParameter end

"""
    SingleExpSynapse{FT} <: AbstractSinExpParameter

A synaptic parameter type that models single exponential synaptic dynamics.

# Fields
- `τe::FT`: Decay time constant for excitatory synapses (default: 6ms)
- `τi::FT`: Rise time constant for inhibitory synapses (default: 0.5ms)
- `E_i::FT`: Reversal potential for inhibitory synapses (default: -75mV)
- `E_e::FT`: Reversal potential for excitatory synapses (default: 0mV)
- `gsyn_e::FT`: Synaptic conductance for excitatory synapses (default: 1.0f0)
- `gsyn_i::FT`: Synaptic conductance for inhibitory synapses (default: 1.0f0)

# Type Parameters
- `FT`: Floating point type (default: `Float32`)

This type implements single exponential synaptic dynamics, where synaptic currents are calculated using separate time constants for both excitatory and inhibitory synapses.
"""
SingleExpSynapse

@snn_kw struct SingleExpSynapse{FT = Float32} <: AbstractSinExpParameter
    ## Synapses
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 0.5ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses
    E_e::FT = 0mV # Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
end

"""
    SingleExpSynapseVars{VFT} <: AbstractSynapseVariable

A synaptic variable type that stores the state variables for single exponential synaptic dynamics.

# Fields
- `N::Int`: Number of synapses
- `ge::VFT`: Vector of excitatory conductances
- `gi::VFT`: Vector of inhibitory conductances
"""
SingleExpSynapseVars

@snn_kw struct SingleExpSynapseVars{VFT = Vector{Float32}}  <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
end

function synaptic_variables(
    synapse::SingleExpSynapse,
    N::Int,
) 
    return SingleExpSynapseVars(;
        N = N,
        ge = zeros(Float32, N),
        gi = zeros(Float32, N),
    )
end

function update_synapses!(
    p::P,
    synapse::T,
    glu::Vector{Float32},
    gaba::Vector{Float32},
    synvars::SingleExpSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    @unpack N, ge, gi = synvars
    @unpack τe, τi = synapse
    @fastmath @inbounds for i ∈ 1:N
        ge[i] += glu[i]
        gi[i] += gaba[i]
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
    fill!(glu, 0f0)
    fill!(gaba, 0f0)
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::SingleExpSynapseVars,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter, VT1 <:AbstractVector, VT2 <:AbstractVector}
    @unpack gsyn_e, gsyn_i, E_e, E_i = synapse
    @unpack N, = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:N
        syncurr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

export SingleExpSynapse