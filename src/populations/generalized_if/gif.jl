# abstract type AbstractIFParameter <: AbstractGeneralizedIFParameter end

"""
    integrate!(p::AbstractGeneralizedIF, param::AbstractGeneralizedIFParameter, dt)

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
# populations whose model is implemented in Float64 (a Float32 step is always accepted)
_float64_supported(::AbstractGeneralizedIF) = false

function integrate!(
    p::P,
    param::T,
    dt::AbstractFloat,
) where {P<:AbstractGeneralizedIF,T<:AbstractGeneralizedIFParameter}
    dt isa Float32 || _float64_supported(p) || throw(ArgumentError(
        "population $(nameof(typeof(p))) is not implemented in $(typeof(dt)) (Float64 state: Identity, IF with DoubleExpSynapse)"))
    if _chunkable(p, p.synapse)
        # the three stages are per-neuron and independent across neurons: one fork/join per population and step
        synapse, receptors, synvars = p.synapse, p.receptors, p.synvars
        foreach_chunk(Int(p.N)) do rng
            update_synapses!(p, synapse, receptors, synvars, dt, rng)
            synaptic_current!(p, synapse, synvars, rng)
            update_neuron!(p, param, dt, rng)
        end
    else
        update_synapses!(p, p.synapse, p.receptors, p.synvars, dt)
        synaptic_current!(p, p.synapse, p.synvars)
        update_neuron!(p, param, dt)
    end
end

# Populations and synapse models whose three stages have chunk methods (`..., rng::UnitRange{Int}`), see `foreach_chunk`.
_chunk_population(::AbstractGeneralizedIF) = false
_chunk_synapse(::AbstractSynapseParameter) = false
@inline _chunkable(p, synapse) = _chunk_population(p) && _chunk_synapse(synapse)

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

# chunk form: the neurons `rng` only (synapse models with a six-argument method)
@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::SYN,
    rng::UnitRange{Int},
) where {P<:AbstractGeneralizedIF,T<:AbstractSynapseParameter,SYN<:AbstractSynapseVariable}
    synaptic_current!(p, synapse, synvars, p.v, p.syn_curr, rng)
end
