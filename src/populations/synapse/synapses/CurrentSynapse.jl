abstract type AbstractCurrentParameter <: AbstractSynapseParameter end

@doc raw"""
    CurrentSynapse(; τe = 6ms, τi = 2ms)

Current-based synapse with single-exponential kinetics for the excitatory (`glu`) and the
inhibitory (`gaba`) inputs.

The weights of the presynaptic spikes are added instantaneously to the synaptic variables
``g_e`` and ``g_i``, which then decay exponentially. They are currents (pA), not conductances:
the synaptic current does not depend on the membrane potential.

# Equations
```math
\frac{dg_e}{dt} = -\frac{g_e}{\tau_e} + \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_i}{dt} = -\frac{g_i}{\tau_i} + \sum_k w_k\,\delta(t - t_k)
```
```math
I_{syn} = -(g_e - g_i)
```
so that with the neuron convention ``C\,dV/dt = \ldots - I_{syn}`` excitation depolarises and
inhibition hyperpolarises (inhibitory weights are positive).

# Integration
At each step the accumulated input is added (`ge += glu`, `gi += gaba`), then one forward
Euler step of the decay is applied (`ge += -dt ge / τe`). The receptor buffers are then zeroed.

# Fields
- `τe::FT = 6ms`: decay time constant of the excitatory current (ms).
- `τi::FT = 2ms`: decay time constant of the inhibitory current (ms).

`FT` defaults to `Float32`. State variables: `CurrentSynapseVars`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.CurrentSynapse(τe = 5ms, τi = 10ms))
```
"""
CurrentSynapse

@snn_kw struct CurrentSynapse{FT = Float32} <: AbstractCurrentParameter
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 2ms # Decay time for inhibitory synapses
end

"""
    CurrentSynapseVars{VFT} <: AbstractSynapseVariable

State variables of `CurrentSynapse`, created by `synaptic_variables(::CurrentSynapse, N)`.

# Fields
- `N::Int = 100`: number of neurons.
- `ge::VFT`: excitatory synaptic current (pA), one entry per neuron.
- `gi::VFT`: inhibitory synaptic current (pA), one entry per neuron.
"""
CurrentSynapseVars
@snn_kw struct CurrentSynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
end

function synaptic_variables(synapse::CurrentSynapse, N::Int)
    return CurrentSynapseVars(; N = N, ge = zeros(Float32, N), gi = zeros(Float32, N))
end

@inline function update_synapses!(
    p::P,
    param::T,
    receptors::RECT,
    synvars::CurrentSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractCurrentParameter,RECT<:NamedTuple}
    @unpack glu, gaba = receptors
    @unpack N, ge, gi = synvars
    @unpack τe, τi = param
    @fastmath @inbounds @simd for i ∈ 1:input_N(p)
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
    @unpack N, v, syn_curr = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:input_N(p)
        syn_curr[i] = -(ge[i] - gi[i])
    end
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::AbstractSynapseVariable,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {
    P<:AbstractGeneralizedIF,
    T<:AbstractCurrentParameter,
    VT1<:AbstractVector,
    VT2<:AbstractVector,
}
    @unpack ge, gi = synvars
    @unpack N = p
    @inbounds @simd for i ∈ 1:input_N(p)
        syncurr[i] = -(ge[i] - gi[i])
    end
end

export CurrentSynapse
