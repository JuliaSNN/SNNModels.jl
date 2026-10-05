abstract type AbstractSinExpParameter <: AbstractSynapseParameter end

@doc raw"""
    SingleExpSynapse(; τe = 6ms, τi = 0.5ms, E_i = -75mV, E_e = 0mV, gsyn_e = 1, gsyn_i = 1)

Conductance-based synapse with single-exponential kinetics, one excitatory and one inhibitory
conductance per neuron.

# Equations
```math
\frac{dg_e}{dt} = -\frac{g_e}{\tau_e} + \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_i}{dt} = -\frac{g_i}{\tau_i} + \sum_k w_k\,\delta(t - t_k)
```
```math
I_{syn} = g_{syn,e}\, g_e\,(V - E_e) + g_{syn,i}\, g_i\,(V - E_i)
```

# Integration
The accumulated input is added to `ge`/`gi`, then one forward Euler step of the decay is
applied (`ge += -dt ge / τe`, `gi += -dt gi / τi`). The receptor buffers are then zeroed.

# Fields
- `τe::FT = 6ms`: decay time constant of the excitatory conductance (ms).
- `τi::FT = 0.5ms`: decay time constant of the inhibitory conductance (ms).
- `E_i::FT = -75mV`: inhibitory reversal potential (mV).
- `E_e::FT = 0mV`: excitatory reversal potential (mV).
- `gsyn_e::FT = 1.0`: scaling of the excitatory conductance (dimensionless; weights are in nS).
- `gsyn_i::FT = 1.0`: scaling of the inhibitory conductance (dimensionless).

`FT` defaults to `Float32`. State variables: `SingleExpSynapseVars`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.SingleExpSynapse(τe = 5ms, τi = 10ms))
```
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

State variables of `SingleExpSynapse`, created by `synaptic_variables(::SingleExpSynapse, N)`.

# Fields
- `N::Int = 100`: number of neurons.
- `ge::VFT`: excitatory conductance (nS).
- `gi::VFT`: inhibitory conductance (nS).
"""
SingleExpSynapseVars

@snn_kw struct SingleExpSynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
end

function synaptic_variables(synapse::SingleExpSynapse, N::Int)
    return SingleExpSynapseVars(; N = N, ge = zeros(Float32, N), gi = zeros(Float32, N))
end

function update_synapses!(
    p::P,
    synapse::T,
    receptors::RECT,
    synvars::SingleExpSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter,RECT<:NamedTuple}
    @unpack N, ge, gi = synvars
    @unpack τe, τi = synapse
    @unpack glu, gaba = receptors
    @fastmath @inbounds for i ∈ 1:N
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
    synapse::T,
    synvars::SingleExpSynapseVars,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {
    P<:AbstractGeneralizedIF,
    T<:AbstractSinExpParameter,
    VT1<:AbstractVector,
    VT2<:AbstractVector,
}
    @unpack gsyn_e, gsyn_i, E_e, E_i = synapse
    @unpack N, = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:N
        syncurr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

export SingleExpSynapse
