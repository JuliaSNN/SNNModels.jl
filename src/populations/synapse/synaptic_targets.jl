"""
    synaptic_target(targets::Dict, post::AbstractPopulation, sym::Symbol, target = nothing) -> (g, v_post)

Resolve where a connection onto `post` delivers its spikes. Called by the connection
constructors (e.g. `SpikingSynapse(pre, post, sym, target; conn)`); each population type
implements a method.

Returns the input buffer `g` (a `Vector{Float32}` of length `post.N`, into which `forward!`
adds the weights of the presynaptic spikes) and the membrane potential vector `v_post` of the
target compartment (used by voltage-dependent plasticity rules). It also records in `targets`
the entry `:sym` (name of the target, e.g. `:glu` or `"glu_d1"`) used by `print_model`/`graph`.

- Generalized IF point neurons (`IF`, `AdEx`, ...): `sym` is mapped by `get_synapse_symbol`
  (`:ge`, `:he` -> `:glu`; `:gi`, `:hi` -> `:gaba`; other symbols unchanged, e.g. the receptor
  groups of a `MultiReceptorSynapse`) and `g = post.receptors.<sym>`; `target` is ignored.
- Dendritic neurons (`Tripod`, `BallAndStick`): `target` is the compartment symbol
  (`:s`, `:d1`, `:d2` for `Tripod`; `:s`, `:d` for `BallAndStick`), `g` is the buffer
  `post.receptors_<target>.<sym>` and `v_post = post.v_<target>`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 10, param = SNN.PoissonParameter(10Hz))
T = SNN.Tripod(N = 5)
syn = SNN.SpikingSynapse(E, T, :glu, :d1; conn = (p = 0.5, μ = 1.0))
syn.targets[:sym]   # "glu_d1"
```
"""
synaptic_target

# function synaptic_target(
#     targets::Dict,
#     post::T,
#     sym::Symbol,
#     target,
# ) where {T<:AbstractPopulation}
#     @warn "Synaptic target not defined for this type. Please implement a method for $(T)"
#     g = zeros(Float32, post.N)
#     v_post = zeros(Float32, post.N)
#     if isnothing(target)
#         g = getfield(post, sym)
#         _v = :v
#         hasfield(typeof(post), _v) && (v_post = getfield(post, _v))
#         push!(targets, :sym => sym)
#     elseif typeof(target) == Symbol
#         _sym = Symbol("$(sym)_$target")
#         _v = Symbol("v_$target")
#         g = getfield(post, _sym)
#         hasfield(typeof(post), _v) && (v_post = getfield(post, _v))
#         push!(targets, :sym => _sym)
#     elseif typeof(target) == Int
#         if typeof(post) <: AbstractDendriteIF
#             _sym = Symbol("$(sym)_d")
#             _v = Symbol("v_d")
#             g = getfield(post, _sym)[target]
#             v_post = getfield(post, _v)[target]
#             push!(targets, :sym => Symbol(string(_sym, target)))
#         elseif isa(post, AdExMultiTimescale)
#             g = getfield(post, sym)[target]
#             v_post = getfield(post, :v)
#             push!(targets, :sym => Symbol(string(sym, target)))
#         end
#     end
#     return g, v_post
#     # return zeros(Float32, post.N), zeros(Float32, post.N)
# end

# function synaptic_target(
#     targets::Dict,
#     post::Any,
# ) 
#     @error "Synaptic target not instatiated, returning non-pointing arrays"
#     g = zeros(Float32, post.N)
#     v = zeros(Float32, post.N)
#     return g, v
# end
