# abstract type AbstractIFParameter <: AbstractGeneralizedIFParameter end

function integrate!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractGeneralizedIFParameter}
    update_synapses!(p, param, dt)
    synaptic_current!(p, param)
    update_neuron!(p, param, dt)
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

@snn_kw struct PostSpike{FT<:Float32}
    A::FT
    τA::FT
end

export PostSpike

