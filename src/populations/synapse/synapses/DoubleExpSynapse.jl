abstract type AbstractDoubleExpParameter <: AbstractSynapseParameter end

"""
    DoubleExpSynapse{FT} <: AbstractDoubleExpParameter

A synaptic parameter type that models double exponential synaptic dynamics.

# Fields
- `τre::FT`: Rise time constant for excitatory synapses (default: 1ms)
- `τde::FT`: Decay time constant for excitatory synapses (default: 6ms)
- `τri::FT`: Rise time constant for inhibitory synapses (default: 0.5ms)
- `τdi::FT`: Decay time constant for inhibitory synapses (default: 2ms)
- `E_i::FT`: Reversal potential for inhibitory synapses (default: -75mV)
- `E_e::FT`: Reversal potential for excitatory synapses (default: 0mV)
- `gsyn_e::FT`: Synaptic conductance for excitatory synapses (default: 1.0f0)
- `gsyn_i::FT`: Synaptic conductance for inhibitory synapses (default: 1.0f0)

# Type Parameters
- `FT`: Floating point type (default: `Float32`)

This type implements double exponential synaptic dynamics, where synaptic currents are calculated using separate rise and decay time constants for both excitatory and inhibitory synapses.
"""
DoubleExpSynapse

@snn_kw struct DoubleExpSynapse{FT = Float32} <: AbstractDoubleExpParameter
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
end

"""
    DoubleExpSynapseVars{VFT} <: AbstractSynapseVariable
A synaptic variable type that stores the state variables for double exponential synaptic dynamics.
# Fields
- `N::Int`: Number of synapses
- `ge::VFT`: Vector of excitatory conductances
- `gi::VFT`: Vector of inhibitory conductances
- `he::VFT`: Vector of auxiliary variables for excitatory synapses
- `hi::VFT`: Vector of auxiliary variables for inhibitory synapses
"""
DoubleExpSynapseVars
@snn_kw struct DoubleExpSynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
    he::VFT = zeros(Float32, N)
    hi::VFT = zeros(Float32, N)
end

function synaptic_variables(synapse::DoubleExpSynapse, N::Int)
    return DoubleExpSynapseVars(;
        N = N,
        ge = zeros(Float32, N),
        gi = zeros(Float32, N),
        he = zeros(Float32, N),
        hi = zeros(Float32, N),
    )
end

function update_synapses!(
    p::P,
    synapse::T,
    receptors::RECT,
    synvars::DoubleExpSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpParameter,RECT<:NamedTuple}
    @unpack N, ge, gi, he, hi = synvars
    @unpack τde, τre, τdi, τri = synapse
    @unpack gaba, glu = receptors
    @inbounds @simd for i ∈ 1:N
        he[i] += glu[i]
        hi[i] += gaba[i]
        ge[i] += dt * (-ge[i] / τde + he[i])
        he[i] += dt * (-he[i] / τre)
        gi[i] += dt * (-gi[i] / τdi + hi[i])
        hi[i] += dt * (-hi[i] / τri)
    end
    fill!(glu, 0.0f0)
    fill!(gaba, 0.0f0)
end


@inline function synaptic_current!(
    p::T,
    synapse::DoubleExpSynapse,
    synvars::DoubleExpSynapseVars,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {T<:AbstractPopulation,VT1<:AbstractVector,VT2<:AbstractVector}
    @unpack gsyn_e, gsyn_i, E_e, E_i = synapse
    @unpack N = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:N
        syncurr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

export DoubleExpSynapse
