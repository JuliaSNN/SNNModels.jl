
abstract type AbstractSynapseParameter <: AbstractComponent end
abstract type AbstractSynapseVariable <: AbstractComponent end

include("CurrentSynapse.jl")
include("DeltaSynapse.jl")
include("DoubleExpSynapse.jl")
include("SingleExpSynapse.jl")
include("ReceptorSynapse.jl")


function get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractSynapseParameter}
    sym == :glu && return :glu
    sym == :gaba && return :gaba
    sym == :he && return :glu
    sym == :hi && return :gaba
    sym == :ge && return :glu
    sym == :gi && return :gaba
    error("Synapse symbol $sym not found in DoubleExpSynapse")
end

export synaptic_current!, update_synapses!, synaptic_variables, synaptic_target, get_synapse_symbols