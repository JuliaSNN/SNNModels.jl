# abstract type AbstractIFParameter <: AbstractGeneralizedIFParameter end

"""
    integrate!(p::AbstractGeneralizedIF, param::AbstractGeneralizedIFParameter, dt::Float32)

One integration step of a generalized integrate-and-fire population (`IF`, `AdEx`, `Tripod`,
`BallAndStick`). Calls, in this order,
1. `update_synapses!(p, p.synapse, p.receptors, p.synvars, dt)`: the spikes written by the
   connections into `p.receptors` during the previous step are added to the synaptic variables,
   which are then integrated over `dt`; the receptor buffers are zeroed;
2. `synaptic_current!(p, p.synapse, p.synvars)`: computes `p.syn_curr` from the synaptic variables
   and the current membrane potential `p.v`;
3. `update_neuron!(p, param, dt)`: integrates the membrane (and adaptation) equations, detects
   spikes and applies reset and refractoriness.
"""
function integrate!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractGeneralizedIFParameter}
    update_synapses!(p, p.synapse, p.receptors, p.synvars, dt)
    synaptic_current!(p, p.synapse, p.synvars)
    update_neuron!(p, param, dt)
end

# Three-argument form used by `integrate!`: writes the synaptic current of every neuron into
# `p.syn_curr`, using the membrane potential `p.v`.
@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::SYN,
) where {P<:AbstractGeneralizedIF,T<:AbstractSynapseParameter,SYN<:AbstractSynapseVariable}
    @unpack N, v, syn_curr = p
    synaptic_current!(p, synapse, synvars, v, syn_curr)
end
