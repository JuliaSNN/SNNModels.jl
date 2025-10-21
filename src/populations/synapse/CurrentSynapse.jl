abstract type AbstractCurrentParameter <: AbstractSynapseParameter end

"""
    CurrentSynapse{FT} <: AbstractCurrentParameter

A synaptic parameter type that models current-based synaptic dynamics.

# Fields
- `τe::FT`: Decay time constant for excitatory synapses (default: 6ms)
- `τi::FT`: Decay time constant for inhibitory synapses (default: 2ms)
- `E_i::FT`: Reversal potential for inhibitory synapses (default: -75mV)
- `E_e::FT`: Reversal potential for excitatory synapses (default: 0mV)

# Type Parameters
- `FT`: Floating point type (default: `Float32`)

This type implements current-based synaptic dynamics, where synaptic currents are calculated using separate time constants for both excitatory and inhibitory synapses.
"""
CurrentSynapse

@snn_kw struct CurrentSynapse{FT = Float32} <: AbstractCurrentParameter
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential for inhibitory synapses
    E_e::FT = 0mV # Reversal potential for excitatory synapses
end

function synaptic_variables(
    synapse::CurrentSynapse,
    N::Int,
) 
    return CurrentSynapseVars(;
        N = N,
        ge = zeros(Float32, N),
        gi = zeros(Float32, N),
    )
end

@snn_kw struct CurrentSynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
end

@inline function update_synapses!(
    p::P,
    param::T,
    synvars::CurrentSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractCurrentParameter}
    @unpack glu, gaba = p
    @unpack N, ge, gi = synvars
    @unpack τe, τi = param
    @fastmath @inbounds @simd for i ∈ 1:N
        ge[i] += glu[i]
        gi[i] += gaba[i]
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
    fill!(glu, 0.0f0)
    fill!(gaba, 0.0f0)
end

@inline function synaptic_current!(
    p::P,
    param::T,
    synvars::CurrentSynapseVars,
) where {P<:AbstractGeneralizedIF,T<:AbstractCurrentParameter}
    @unpack E_e, E_i = param
    @unpack N, v, syn_curr = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = ge[i] * (v[i] - E_e) - gi[i] * (v[i] - E_i)
    end
end

export CurrentSynapse