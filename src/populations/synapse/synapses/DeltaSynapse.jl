abstract type AbstractDeltaParameter <: AbstractSynapseParameter end

@doc raw"""
    DeltaSynapse()

Instantaneous (delta) current synapse. The sum of the weights of the spikes received during
the previous step is applied as a current during one integration step and then discarded.

# Equations
```math
I_{syn}(t) = -\left(\sum_{k \in \text{exc}} w_k - \sum_{k \in \text{inh}} w_k\right)
```
where the sums run over the spikes delivered since the previous step. The struct has no fields.

# Integration
`update_synapses!` adds the receptor buffers to `ge`, `gi` and zeroes the buffers;
`synaptic_current!(p, synapse, synvars)` sets `syn_curr = -(ge - gi)` and resets `ge` and `gi`
to zero. With the forward Euler membrane update ``V \mathrel{+}= \frac{dt}{\tau_m} R (\ldots - I_{syn})``
of `IF`, a weight `w` therefore produces a voltage jump of ``R\,w\,dt/\tau_m``, which depends on `dt`.

Only the three-argument `synaptic_current!` method exists, so `DeltaSynapse` works with the
generalized IF point neurons (`IF`, `AdEx`, `ExtendedIF`) but not with `Tripod` or
`BallAndStick`, which call the five-argument form.

State variables: `DeltaSynapseVars`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS), synapse = SNN.DeltaSynapse())
```
"""
DeltaSynapse

struct DeltaSynapse <: AbstractDeltaParameter end

"""
    DeltaSynapseVars{VFT} <: AbstractSynapseVariable

State variables of `DeltaSynapse`, created by `synaptic_variables(::DeltaSynapse, N)`.

# Fields
- `N::Int = 100`: number of neurons.
- `ge::VFT`: excitatory input accumulated for the current step (pA).
- `gi::VFT`: inhibitory input accumulated for the current step (pA).
"""
DeltaSynapseVars
@snn_kw struct DeltaSynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
end


function synaptic_variables(synapse::DeltaSynapse, N::Int)
    return DeltaSynapseVars(; N = N, ge = zeros(Float32, N), gi = zeros(Float32, N))
end

@inline function update_synapses!(
    p::P,
    synapse::T,
    receptors::RECT,
    synvars::DeltaSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter,RECT<:NamedTuple}
    @unpack N, ge, gi = synvars
    @unpack glu, gaba = receptors
    @fastmath @inbounds for i ∈ 1:N
        ge[i] += glu[i]
        gi[i] += gaba[i]
    end
    fill!(glu, 0.0f0)
    fill!(gaba, 0.0f0)
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::DeltaSynapseVars,
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter}
    @unpack N, v, syn_curr = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = -(ge[i] - gi[i])
        ge[i] = 0.0f0
        gi[i] = 0.0f0
    end
end

export DeltaSynapse
