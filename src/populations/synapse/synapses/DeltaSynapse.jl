abstract type AbstractDeltaParameter <: AbstractSynapseParameter end

"""
    DeltaSynapse{FT} <: AbstractDeltaParameter

A synaptic parameter type that models delta (instantaneous) synaptic dynamics.

# Fields
None - this type implements instantaneous synaptic dynamics where synaptic inputs are applied directly without any time constants.

# Type Parameters
- `FT`: Floating point type (default: `Float32`)

This type implements delta synaptic dynamics, where synaptic inputs are applied instantaneously without any time delays or decay. The synaptic current is calculated as the difference between excitatory and inhibitory inputs.
"""
DeltaSynapse

struct DeltaSynapse <: AbstractDeltaParameter end

"""
    DeltaSynapseVars{VFT} <: AbstractSynapseVariable

    A synaptic variable type that stores the state variables for delta synaptic dynamics.
    # Fields
    - `N::Int`: Number of synapses
    - `ge::VFT`: Vector of excitatory conductances
    - `gi::VFT`: Vector of inhibitory conductances
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
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter, RECT<:NamedTuple }
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
