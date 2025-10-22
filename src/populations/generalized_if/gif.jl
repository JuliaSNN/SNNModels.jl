# abstract type AbstractIFParameter <: AbstractGeneralizedIFParameter end

function integrate!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractGeneralizedIFParameter}
    update_synapses!(p, p.synapse, p.synvars, dt)
    synaptic_current!(p, p.synapse, p.synvars)
    update_neuron!(p, param, dt)
end

function update_synapses!(
    p::P,
    synapse::T,
    synvars::SYN,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractReceptorParameter,SYN<:AbstractSynapseVariable}
    @unpack N, glu, gaba = p
    update_synapses!(p, synapse, glu, gaba, synvars, dt)
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::SYN,
) where {P<:AbstractGeneralizedIF,T<:AbstractSynapseParameter, SYN<:AbstractSynapseVariable}
    @unpack N, v, syn_curr = p
    synaptic_current!(
        p,
        synapse,
        synvars,
        v,
        syn_curr
    )
end

