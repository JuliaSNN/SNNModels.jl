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



"""
    PostSpike

A structure defining the parameters of a post-synaptic spike event.

# Fields
- `A::FT`: Amplitude of the Post-Synaptic Potential (PSP).
- `τA::FT`: Time constant of the PSP.

The type `FT` represents Float32.
"""
PostSpike

@snn_kw struct PostSpike{FT = Float32}
    At::FT = 0mV
    τA::FT = -1mV
    AP_membrane::FT = 10.0f0mV
    τabs::FT = 1ms # Absolute refractory period
    up::FT = 1ms
end

export PostSpike

