abstract type AbstractDoubleExpCurrentParameter <: AbstractSynapseParameter end

@doc raw"""
    DoubleExpCurrentSynapse(; τre = 1ms, τde = 6ms, τri = 0.5ms, τdi = 2ms)

Current-based synapse with double-exponential (rise and decay) kinetics. Same kinetics as
`DoubleExpSynapse`, but `ge` and `gi` are currents (pA) and do not depend on the membrane
potential.

# Equations
```math
\frac{dh_e}{dt} = -\frac{h_e}{\tau_{re}} + \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_e}{dt} = -\frac{g_e}{\tau_{de}} + h_e
```
(same for the inhibitory pair), and ``I_{syn} = -(g_e - g_i)``.

# Integration
Forward Euler, in the same order as `DoubleExpSynapse` (input into `he`/`hi`, update `ge`/`gi`
with the new `he`/`hi`, then decay `he`/`hi`). The receptor buffers are then zeroed.

# Fields
- `τre::FT = 1ms`: rise time constant, excitatory (ms).
- `τde::FT = 6ms`: decay time constant, excitatory (ms).
- `τri::FT = 0.5ms`: rise time constant, inhibitory (ms).
- `τdi::FT = 2ms`: decay time constant, inhibitory (ms).

`FT` defaults to `Float32`. State variables: `DoubleExpCurrentSynapseVars`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.DoubleExpCurrentSynapse())
```
"""
DoubleExpCurrentSynapse

@snn_kw struct DoubleExpCurrentSynapse{FT = Float32} <: AbstractDoubleExpCurrentParameter
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
end

"""
    DoubleExpCurrentSynapseVars{VFT} <: AbstractSynapseVariable

State variables of `DoubleExpCurrentSynapse`.

# Fields
- `N::Int = 100`: number of neurons.
- `ge::VFT`: excitatory synaptic current (pA).
- `gi::VFT`: inhibitory synaptic current (pA).
- `he::VFT`: excitatory rise (auxiliary) variable (pA/ms).
- `hi::VFT`: inhibitory rise (auxiliary) variable (pA/ms).
"""
DoubleExpCurrentSynapseVars
@snn_kw struct DoubleExpCurrentSynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
    he::VFT = zeros(Float32, N)
    hi::VFT = zeros(Float32, N)
end

function synaptic_variables(synapse::DoubleExpCurrentSynapse, N::Int)
    return DoubleExpCurrentSynapseVars(;
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
    synvars::DoubleExpCurrentSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpCurrentParameter,RECT<:NamedTuple}
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
    synapse::DoubleExpCurrentSynapse,
    synvars::DoubleExpCurrentSynapseVars,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {T<:AbstractPopulation,VT1<:AbstractVector,VT2<:AbstractVector}
    @unpack N = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:N
        syncurr[i] = -(ge[i] - gi[i] )
    end
end

export DoubleExpCurrentSynapse