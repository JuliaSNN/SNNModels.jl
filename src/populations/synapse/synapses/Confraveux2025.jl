abstract type AbstractConfavreux2025 <: AbstractSynapseParameter end

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
Confavreux2025Synapse

@snn_kw struct Confavreux2025Synapse{FT = Float32} <: AbstractConfavreux2025
    τAMPA::FT = 5ms # Rise time for excitatory synapses
    τNMDA::FT = 100ms # Decay time for excitatory synapses
    τGABA::FT = 10ms # Rise time for inhibitory synapses
    E_i::FT = -80mV # Reversal potential excitatory synapses
    E_e::FT = 0mV #Reversal potential excitatory synapses
    α::FT = 0.23f0 # NMDA voltage dependence parameter
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
Confavreux2025SynapseVars

@snn_kw struct Confavreux2025SynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    gAMPA::VFT = zeros(Float32, N)
    gNMDA::VFT = zeros(Float32, N)
    gGABA::VFT = zeros(Float32, N)
end

function synaptic_variables(synapse::Confavreux2025Synapse, N::Int)
    return Confavreux2025SynapseVars(;
        N = N,
    )
end

function update_synapses!(
    p::P,
    synapse::T,
    receptors::RECT,
    synvars::Confavreux2025SynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractConfavreux2025,RECT<:NamedTuple}
    @unpack N, gAMPA, gNMDA, gGABA = synvars
    @unpack τAMPA, τNMDA, τGABA = synapse
    @unpack gaba, glu = receptors
    @inbounds @simd for i ∈ 1:N
        gAMPA[i] += dt * (-gAMPA[i] / τAMPA + glu[i])
        gGABA[i] += dt * (-gGABA[i] / τGABA + gaba[i])
        gNMDA[i] += dt * (gAMPA[i] - gNMDA[i])/ τNMDA
    end
    fill!(glu, 0.0f0)
    fill!(gaba, 0.0f0)
end


@inline function synaptic_current!(
    p::T,
    synapse::Confavreux2025Synapse,
    synvars::Confavreux2025SynapseVars,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {T<:AbstractPopulation,VT1<:AbstractVector,VT2<:AbstractVector}
    @unpack gAMPA, gNMDA, gGABA = synvars
    @unpack E_e, E_i, α = synapse
    @unpack N = p
    @inbounds @simd for i ∈ 1:N
        syncurr[i] = (α * gAMPA[i] + (1 - α)*gNMDA[i]) * (v[i] - E_e) + gGABA[i] * (v[i] - E_i)
    end
end

export Confavreux2025Synapse