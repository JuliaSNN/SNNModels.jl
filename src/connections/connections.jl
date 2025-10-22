
abstract type AbstractSpikingSynapse <: AbstractSparseSynapse end


include("empty.jl")
include("normalization.jl")
include("aggregate_scaling.jl")
include("rate_synapse.jl")
include("fl_synapse.jl")
include("fl_sparse_synapse.jl")
include("pinning_synapse.jl")
include("pinning_sparse_synapse.jl")
include("spike_rate_synapse.jl")

struct SpikingSynapseParameter <: AbstractConnectionParameter end
include("sparse_plasticity.jl")
include("spiking_synapse.jl")

# function synaptic_target(
#     targets::Dict,
#     post::T,
#     sym::Symbol,
#     target::Nothing,
# ) where {T<:AbstractPopulation}
#     @show "Synaptic target called"
#     v_post = zeros(Float32, post.N)
#     sym_synapse = get_synapse_symbol(post, sym)
#     g = getfield(post, sym_synapse)
#     hasfield(typeof(post), :v) && (v_post = getfield(post, :v))
#     push!(targets, :sym => sym)
#     return g, v_post
# end

# function synaptic_target(
#     targets::Dict,
#     post::T,
#     sym::Symbol,
#     target::Symbol,
# ) where {T<:AbstractPopulation}
#     @show "Synaptic target called"
#     v_post = zeros(Float32, post.N)
#     sym_synapse = get_synapse_symbol(post, sym)
#     _sym = Symbol("$(sym)_$target")
#     _v = Symbol("v_$target")

#     g = getfield(post, sym_synapse)
#     hasfield(typeof(post), _v) && (v_post = getfield(post, _v))
#     push!(targets, :sym => _sym)
#     return g, v_post
# end

# function synaptic_target(
#     targets::Dict,
#     post::T,
#     sym::Symbol,
#     target::Int,
# ) where {T<:AbstractPopulation}
#     @show "Synaptic target called"
#     v_post = zeros(Float32, post.N)
#     sym_synapse = get_synapse_symbol(post, sym)
#     _sym = Symbol("$(sym_synapse)_d")
#     _v = Symbol("v_d")
#     g = getfield(post, _sym)[target]
#     v_post = getfield(post, _v)[target]
#     push!(targets, :sym => Symbol(string(_sym, target)))
# end

# function synaptic_target(
#     targets::Dict,
#     post::T,
#     sym::Nothing,
#     target,
# ) where {T<:AbstractPopulation}
#     @warn "Synaptic target not instatiated, returning non-pointing arrays"
#     g = zeros(Float32, post.N)
#     v = zeros(Float32, post.N)
#     return g, v
# end

# synaptic_target(post::T, sym::Symbol, target) where {T<:AbstractPopulation} = synaptic_target(Dict(), post, sym, target)


# export synaptic_target
