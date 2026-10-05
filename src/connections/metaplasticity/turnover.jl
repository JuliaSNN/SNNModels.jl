"""
    TurnoverParam <: MetaPlasticityParameter

Abstract type of the structural-turnover rules used by `Turnover`: `RandomTurnover` and
`ActivityDependentTurnover`.
"""
abstract type TurnoverParam <: MetaPlasticityParameter end

@doc raw"""
    RandomTurnover(; rate = -1.0f0, τ = 1 / rate, threshold = 0.1f0, μ = 3.0f0)

Parameters of random structural turnover (intended: every `τ`, a random subset of synapses
is moved to new, randomly chosen postsynaptic targets).

# Fields
- `rate::Float32 = -1`: turnover rate (1/ms); only used to compute the default `τ`.
- `τ::Float32 = 1 / rate`: interval between turnover events (ms).
- `threshold::Float32 = 0.1`: intended fraction of rewired synapses; unused by the code.
- `μ::Float32 = 3.0`: mean of the new weights, drawn from ``\mathcal{N}(\mu, \sqrt{\mu})``.

# Status in SNNModels 1.8.4
No `plasticity!(c::Turnover, ::RandomTurnover, dt, T)` method exists, so `train!` raises a
`MethodError` for a `Turnover` with this parameter. Calling
`plasticity!(c, c.param)` directly runs `synaptic_turnover!` with `p_rewire = c.p_rewire[1]`
(0 unless set by the user) and fresh uniform `p_values`.
"""
RandomTurnover

@snn_kw struct RandomTurnover{FT<:AbstractFloat} <: TurnoverParam
    rate::FT = -1.0f0
    τ::FT = 1/rate
    threshold::FT = 0.1f0
    μ::FT = 3.0f0
end

@doc raw"""
    ActivityDependentTurnover(; rate = -1, τ = 1 / rate, fraction = 0.1f0,
                              τpre = 250ms, τpost = 250ms, μ = 3.0f0)

Parameters of activity-dependent structural turnover: every `τ`, the `fraction` of
synapses with the lowest pre/post co-activity are moved to new postsynaptic targets.

# Fields
- `rate = -1`: turnover rate (1/ms); only used for the default `τ`. Pass `rate` or `τ`.
- `τ = 1 / rate`: interval between turnover events (ms).
- `fraction::Float32 = 0.1`: quantile of co-activity below which synapses are rewired.
- `τpre::Float32 = 250ms`, `τpost::Float32 = 250ms`: time constants of the pre- and
  postsynaptic activity traces (ms).
- `μ::Float32 = 3.0`: mean of the new weights, ``\mathcal{N}(\mu, \sqrt{\mu})``.

The element type `FT` is inferred from the arguments (no default), so give `rate`/`τ` as
`Float32` values for a `Float32` struct.
"""
ActivityDependentTurnover

@snn_kw struct ActivityDependentTurnover{FT} <: TurnoverParam
    rate::FT = -1
    τ::FT = 1/rate
    fraction::FT = 0.1f0
    τpre::FT = 250.0f0ms
    τpost::FT = 250.0f0ms
    μ::FT = 3.0f0
end


@doc raw"""
    Turnover{VFT, MFT, ST} <: AbstractMetaPlasticity

Structural turnover acting on one `SpikingSynapse` (`synapse`). Built with
`MetaPlasticity(param::TurnoverParam, synapse)` and added to the model as a connection; it
transmits nothing (`forward!` is a no-op), its `plasticity!` runs only under `train!`.

# Update (`ActivityDependentTurnover`)
At every `train!` step the presynaptic and postsynaptic traces are updated as
```math
a^{pre}_j \leftarrow a^{pre}_j + \frac{-a^{pre}_j\,dt + \delta_j}{\tau_{pre}}, \qquad
a^{post}_i \leftarrow a^{post}_i + \frac{-a^{post}_i\,dt + \delta_i}{\tau_{post}}
```
(``\delta = 1`` for a spike in this step). Every `round(Int, τ / dt)` steps:
1. ``p_s = a^{pre}_{j(s)}\, a^{post}_{i(s)}`` for every synapse ``s``;
2. `p_rewire` = the `fraction`-quantile of ``p``;
3. `synaptic_turnover!` moves every synapse with ``p_s \le`` `p_rewire` to a new
   postsynaptic neuron, chosen without replacement among the neurons not yet targeted by the
   same presynaptic neuron (uniform weights `p[post, pre] = 1`), and draws its weight from
   ``\mathcal{N}(\mu, \sqrt{\mu})``; the sparse structure is then rebuilt.

# Fields
- `param::TurnoverParam`; `synapse`: the rewired connection.
- `pre`, `post::Vector{Float32}`: activity traces.
- `p::Matrix{Float32}`: `N_post x N_pre` sampling weights of new targets (all ones).
- `p_rewire::Vector{Float32}`: current rewiring threshold (1 element).
- `p_values::Vector{Float32}`: co-activity of every synapse.
- `id`, `name = "Turnover"`, `targets`, `records`.

# References
Reference not given in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 2.0))
TO = SNN.MetaPlasticity(SNN.ActivityDependentTurnover(τ = 50.0f0ms), EE)
model = SNN.compose(; E, EE, TO)
SNN.train!(model; duration = 100ms)
```
"""
Turnover

@snn_kw struct Turnover{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    ST<:AbstractConnection,
} <: AbstractMetaPlasticity
    id::String = randstring(12)
    name::String = "Turnover"
    param::TurnoverParam = RandomTurnover(0)
    synapse::ST = SpikingSynapse()
    pre::VFT = zeros(Float32, length(synapse.fireJ))
    post::VFT = zeros(Float32, length(synapse.fireI))
    p::MFT = ones(Float32, length(synapse.fireI), length(synapse.fireJ))
    p_rewire::VFT = zeros(Float32, 1)
    p_values::VFT = zeros(Float32, length(synapse.W))
    targets::Dict = Dict()
    records::Dict = Dict()
end


"""
    MetaPlasticity(param::TurnoverParam, synapse; kwargs...)

Build a `Turnover` acting on `synapse`.
"""
function MetaPlasticity(param::T, synapse; kwargs...) where {T<:TurnoverParam}
    targets = Dict(:synapses => [synapse.id], :post => synapse.targets[:post])
    Turnover(; param, kwargs..., synapse, targets)
end

function forward!(c::Turnover, param::T) where {T<:TurnoverParam} end

"""
    plasticity!(c::Turnover, param::ActivityDependentTurnover, dt::Float32, T::Time)

Update the activity traces and, every `round(Int, τ / dt)` steps, rewire the least
co-active synapses (see `Turnover`). Called only by `train!`.
"""
function plasticity!(c::Turnover, param::ActivityDependentTurnover, dt::Float32, T::Time)
    @unpack synapse, p, pre, post, p_values = c
    @unpack fraction, μ, τpre, τpost = param
    @turbo for i in eachindex(synapse.fireJ)
        pre[i] += (-pre[i]*dt + synapse.fireJ[i])/τpre
    end
    @turbo for i in eachindex(synapse.fireI)
        post[i] += (-post[i]*dt + synapse.fireI[i])/τpost
    end

    ##
    @unpack τ = param
    tt = get_step(T)
    if ((tt) % round(Int, τ / dt)) < dt
        @simd for j in eachindex(synapse.fireJ)
            for s in postsynaptic_idxs(synapse, j)
                p_values[s] = pre[j] * post[synapse.I[s]]
            end
        end
        c.p_rewire[1] = quantile(p_values, fraction)
        # @show mean(p_values)
        # @show c.p_rewire[1]
        # @show quantile(p_values, fraction)
        # @show minimum(p_values)
        # @show maximum(p_values)
        plasticity!(c, param)
    end
end

"""
    plasticity!(c::Turnover, param::TurnoverParam)

Run `synaptic_turnover!` on `c.synapse` with the current `c.p_rewire[1]`, `c.p_values` and
`param.μ`.
"""
function plasticity!(c::Turnover, param::TT) where {TT<:TurnoverParam}
    @unpack synapse, p = c
    @unpack μ = param

    synaptic_turnover!(
        c.synapse,
        p_rewire = c.p_rewire[1],
        p_new = (post, pre)->p[post, pre],
        p_values = c.p_values,
        μ = μ,
    )
end

export TurnoverParam, RandomTurnover, ActivityDependentTurnover, Turnover, MetaPlasticity


@doc raw"""
    synaptic_turnover!(C; p_rewire = 0.05, p_new = x -> rand(), μ = 3.0, p_values = nothing)

Rewire, in place, the synapses of a sparse connection `C` (e.g. `SpikingSynapse`).

# Keyword arguments
- `p_rewire = 0.05`: threshold; synapse ``s`` is rewired if `p_values[s] <= p_rewire`.
- `p_values = nothing`: one value per synapse (CSC order of `C.W`); if `nothing`, drawn
  uniformly in [0, 1], so that a fraction `p_rewire` of synapses is rewired on average.
- `p_new`: function `(post, pre) -> weight` giving the sampling weight of each candidate new
  postsynaptic neuron. It is called with two arguments: the default `x -> rand()` takes one
  argument and raises a `MethodError`, so always pass a two-argument function.
- `μ = 3.0`: new weights are drawn from ``\mathcal{N}(\mu, \sqrt{\mu})``.

# Procedure
1. For every presynaptic neuron, the candidate targets are the postsynaptic neurons it does
   not contact yet, weighted by `p_new(post, pre)`.
2. The synapses to rewire are selected with `p_values[s] <= p_rewire`.
3. For each presynaptic neuron, as many new targets as rewired synapses are sampled without
   replacement; the postsynaptic index `C.I[s]` and the weight `C.W[s]` are replaced.
4. `update_sparse_matrix!(C)` rebuilds `colptr`, `rowptr`, `J` and `index`.

Per-synapse arrays other than `I`, `J`, `W`, `index` (short-term efficacy `ρ`, plasticity
variables) are not reordered.
"""
function synaptic_turnover!(
    C::S;
    p_rewire = 0.05,
    p_new = x->rand(),
    μ = 3.0,
    p_values = nothing,
) where {S<:AbstractConnection}
    # @info "Performing synaptic turnover on $(C.name)"
    p_values = isnothing(p_values) ? rand(Uniform(0, 1), length(C.W)) : p_values
    all_post = Set(1:length(C.fireI))
    all_posts = postsynaptic(C)
    new_connections = map(eachindex(all_posts)) do pre
        my_post = all_posts[pre]
        plausible_post = setdiff(all_post, my_post) |> collect
        # plausible_post[sortperm([p_new(post, pre) for post in plausible_post])]
        (plausible_post, Weights([p_new(post, pre) for post in plausible_post]))
    end

    rep_connections = Int[]
    rep_neurons = Int[]
    # @show typeof(p_values), size(p_values), p_values[1]
    @unpack rowptr, colptr, I, J, index, W, fireJ = C
    for j in eachindex(fireJ)
        post_n = 0
        for s in postsynaptic_idxs(C, j)
            p_values[s] > p_rewire && continue
            push!(rep_connections, s)
            post_n += 1
        end
        # @show "Changing $(post_n), p_rewire=$(p_rewire)"
        post_n == 0 && continue
        plausible_post, weights = new_connections[j]
        append!(rep_neurons, sample(plausible_post, weights, post_n; replace = false))
    end
    for (s, new_post) in zip(rep_connections, rep_neurons)
        C.I[s] = new_post
        C.W[s] = rand(Normal(μ, sqrt(μ)))
    end
    update_sparse_matrix!(C)
end
