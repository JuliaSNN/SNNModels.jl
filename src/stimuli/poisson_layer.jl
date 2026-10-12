"""
    PoissonLayerParameter <: AbstractStimulusParameter

Supertype of the parameters of a [`PoissonStimulusLayer`](@ref): `PoissonLayer`
(one rate for all layer neurons) and `PoissonLayerHet` (one rate per layer neuron).
"""
abstract type PoissonLayerParameter <: AbstractStimulusParameter end

"""
    PoissonLayer(; rate, N = 1, active = [true])
    PoissonLayer(rate::Real; N)

Parameter of a [`PoissonStimulusLayer`](@ref) in which the `N` neurons of the input layer
fire as independent Poisson processes with the same rate `rate`.

The connectivity between the layer and the target population is not part of the parameter:
it is given by the `conn` keyword of `Stimulus` (see `PoissonStimulusLayer`). A target neuron
with ``K`` afferent layer neurons of weight ``w`` receives in total ``K`` independent trains
of rate `rate`, i.e. a total input rate of ``K`` times `rate` with increments ``w``.

# Fields
- `rate::Float32` (required): firing rate of each layer neuron (use `Hz`).
- `N::Int32 = 1`: number of neurons in the layer.
- `active::Vector{Bool} = [true]`: when `false` (`set_active!`), the layer does not fire.

The positional form is `PoissonLayer(rate; N = 1)`. (Up to SNNModels 1.8.4 `active` was not
read and the positional form required `N`.)

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
stim = SNN.Stimulus(SNN.PoissonLayer(rate = 10Hz, N = 200), E, :ge; conn = (p = 0.1, μ = 1.0))
```
"""
PoissonLayer

@snn_kw mutable struct PoissonLayer{FT = Float32} <: PoissonLayerParameter
    rate::FT  # Default rate in Hz
    N::Int32 = 1
    active::VBT = [true]
end

"""
    PoissonLayerHet(; N = 1, rates, active = [true])
    PoissonLayerHet(rate::Real; N)

Parameter of a [`PoissonStimulusLayer`](@ref) in which layer neuron `j` fires as a Poisson
process with its own rate `rates[j]`.

# Fields
- `N::Int32 = 1`: number of neurons in the layer.
- `rates::Vector{Float32}` (required): rate of each layer neuron (length `N`, use `Hz`).
  It is a vector, so it can be changed at runtime with `set_variable!(stim, :rates, r)`.
- `active::Vector{Bool} = [true]`: when `false`, the layer does not fire.

`PoissonLayerHet(rate; N)` builds `rates = fill(rate, N)`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
param = SNN.PoissonLayerHet(N = 50, rates = collect(range(1Hz, 20Hz, length = 50)))
stim = SNN.Stimulus(param, E, :ge; conn = (p = 0.2, μ = 0.5))
```
"""
PoissonLayerHet

@snn_kw struct PoissonLayerHet{VFT = Vector{Float32}} <: PoissonLayerParameter
    N::Int32 = 1
    rates::VFT
    active::VBT = [true]
end

function PoissonLayer(rate::R; N = 1, kwargs...) where {R<:Real}
    return PoissonLayer(; N = N, rate = rate, kwargs...)
end

function PoissonLayerHet(rate::R; N = 1, kwargs...) where {R<:Real}
    return PoissonLayerHet(; N = N, rates = fill(Float32(rate), N), kwargs...)
end

@doc raw"""
    Stimulus(param::PoissonLayerParameter, post::AbstractPopulation, sym::Symbol, comp = nothing;
             conn::NamedTuple, name = "Poisson")
    PoissonStimulusLayer(post, sym, comp = nothing; conn, param::PoissonLayerParameter, name = "Poisson")

A layer of `param.N` Poisson neurons connected to `post` through a sparse weight matrix.

The weight matrix ``W`` (`post.N x param.N`) is generated with `sparse_matrix(param.N,
post.N, conn)`; `conn` is a connectivity NamedTuple such as `(p = 0.1, μ = 1.0)` (see
`sparse_matrix` for the rules `:Fixed`, `:Bernoulli`, ... and the weight distribution).
The `PoissonStimulusLayer(post, sym, comp; conn, ...)` form also accepts a matrix as `conn`.

# Equations
At each step every layer neuron ``j`` fires with probability ``\nu_j\,\Delta t``
(Bernoulli approximation of a Poisson process, at most one spike per step), with
``\nu_j`` = `param.rate` (`PoissonLayer`) or `param.rates[j]` (`PoissonLayerHet`). For
each spike, all its targets ``i`` are incremented:
```math
g_i \leftarrow g_i + W_{ij}
```

The spikes are sampled sparsely, at one random draw per spike (see [`stimulate!`](@ref)).

# Fields
- `N::Int`: number of layer neurons; `id`, `name = "Poisson"`
- `param::PoissonLayerParameter`
- `g::VFT`: target variable of `post` (shared); `VFT`, `W` and `param` follow the precision of `post`.
- `colptr`, `rowptr`, `I`, `J`, `index`, `W`: sparse connectivity, as returned by `dsparse`
  (`I` are postsynaptic indices, `W` the weights).
- `fire::Vector{Bool}`: spikes of the layer neurons in the last step (can be recorded with
  `monitor!(stim, :fire)`).
- `spikes::SpikeIndex`: ascending list of the layer neurons that fired in the last step (`spike_index(stim)`).
- `inbox`: log of the neurons of `post` written (see `attach_driver!`), or `nothing`.
- `records::Dict`, `targets::Dict`

`param` is required (the keyword struct constructor has no default for it).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
stim = SNN.Stimulus(SNN.PoissonLayer(rate = 10Hz, N = 1000), E, :ge; conn = (p = 0.05, μ = 1.0))
SNN.monitor!(stim, :fire)
model = SNN.compose(; E, stim)
SNN.sim!(; model, duration = 100ms)
```
"""
PoissonStimulusLayer
@snn_kw struct PoissonStimulusLayer{VFT = Vector{Float32},PT<:PoissonLayerParameter,TLT = Nothing} <:
               AbstractStimulus
    N::Int
    id::String = randstring(12)
    name::String = "Poisson"
    param::PT
    ##
    g::VFT # target conductance for soma
    colptr::VIT
    rowptr::VIT
    I::VIT
    J::VIT
    index::VIT
    W::VFT
    fire::VBT = zeros(Bool, N)
    spikes::SpikeIndex = fed_spike_index(N)  # ascending list of the layer neurons with fire[j]
    ##
    records::Dict = Dict()
    targets::Dict = Dict()
    inbox::TLT = nothing  # log of the neurons of `post` written (TouchList), if post has a sparse update
end


function PoissonStimulusLayer(
    post::T,
    sym::Symbol,
    comp = nothing;
    conn::Connectivity,
    param::PoissonLayerParameter,
    name::String = "Poisson",
) where {T<:AbstractPopulation}
    # @warn "PoissonStimulusLayer is deprecated. Please use Stimulus(param, post, sym, comp; conn) instead."

    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, comp)
    FT = eltype(g)    # the stimulus has the precision of its target (`float_type`)
    param = convert_precision(param, FT)
    w = sparse_matrix(param.N, post.N, conn, FT)
    rowptr, colptr, I, J, index, W = dsparse(w)
    inbox = attach_driver!(post)

    return PoissonStimulusLayer(;
        VFT = Vector{FT},
        param = param,
        PT = typeof(param),
        inbox = inbox,
        TLT = typeof(inbox),
        N = param.N,
        targets = targets,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
        name = name,
    )
end

function Stimulus(
    param::PoissonLayerParameter,
    post::T,
    sym::Symbol,
    comp = nothing;
    conn::NamedTuple,
    name::String = "Poisson",
) where {T<:AbstractPopulation}

    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, comp)
    FT = eltype(g)    # the stimulus has the precision of its target (`float_type`)
    param = convert_precision(param, FT)
    w = sparse_matrix(param.N, post.N; conn..., FT)
    rowptr, colptr, I, J, index, W = dsparse(w)
    inbox = attach_driver!(post)

    return PoissonStimulusLayer(;
        VFT = Vector{FT},
        param = param,
        PT = typeof(param),
        inbox = inbox,
        TLT = typeof(inbox),
        N = param.N,
        targets = targets,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
        name = name,
    )
end


# One layer neuron fires: flag it, list it, add its weights to the target, log the touched targets.
@inline function _layer_fire!(p::PoissonStimulusLayer, j::Int)
    @unpack colptr, W, I, g, fire, spikes, inbox = p
    @inbounds begin
        fire[j] = true
        n = spikes.n + 1
        spikes.n = n
        spikes.idx[n] = j
        @fastmath for s ∈ colptr[j]:(colptr[j+1]-1)
            i = I[s]
            g[i] += W[s]
            _touch!(inbox, i)
        end
    end
    return nothing
end

# Clear the flags set in the previous step (where they were set, not with a pass over `N`).
@inline function _layer_clear!(p::PoissonStimulusLayer)
    @unpack fire, spikes = p
    @inbounds for q in 1:spikes.n
        fire[spikes.idx[q]] = false
    end
    spikes.n = 0
    return nothing
end

# Above this per-step probability sampling every neuron is cheaper than skipping.
const _DENSE_SPIKE_P = 0.2

"""
    stimulate!(p::PoissonStimulusLayer, param::Union{PoissonLayer,PoissonLayerHet}, time::Time, dt)

One step of a Poisson layer: layer neuron `j` fires with probability `rate * dt` (`rates[j] * dt`
for `PoissonLayerHet`), independently, at most once per step; for each firing neuron the weights
of its outgoing connections are added to `p.g`. `p.fire` and `p.spikes` hold the layer spikes of this step.
The layer does not fire while `param.active[1]` is `false`. Targets written are logged in `p.inbox` (see `attach_driver!`).

Sampling (global RNG), with one random draw per spike:
- `PoissonLayer`: the gap to the next spike of the Bernoulli(`p`) sequence is geometric, `floor(log(1 - U) / log(1 - p))`.
- `PoissonLayerHet`: candidates drawn as above at the largest probability `pmax` (recomputed every step), each kept
  with probability `rates[j] * dt / pmax`; the cost grows with the largest rate.
- From a probability of $(_DENSE_SPIKE_P) on, every neuron is tested with one uniform draw.
"""
function stimulate!(p::PoissonStimulusLayer, param::PoissonLayer, time::Time, dt::AbstractFloat)
    _layer_clear!(p)
    param.active[1] || return nothing
    pr = Float64(param.rate * dt)
    _bernoulli_layer!(p, pr)
    return nothing
end

function stimulate!(
    p::PoissonStimulusLayer,
    param::PoissonLayerHet,
    time::Time,
    dt::AbstractFloat,
)
    _layer_clear!(p)
    param.active[1] || return nothing
    rates = param.rates
    length(rates) == p.N || throw(DimensionMismatch("rates must have one entry per layer neuron"))
    rmax = eltype(rates)(-Inf)
    @inbounds @fastmath for j in eachindex(rates)
        rmax = max(rmax, rates[j])
    end
    pmax = Float64(rmax * dt)
    pmax > 0 || return nothing
    N = p.N
    if pmax >= _DENSE_SPIKE_P
        @inbounds for j in 1:N
            rand() < Float64(rates[j] * dt) && _layer_fire!(p, j)
        end
        return nothing
    end
    lq = log1p(-pmax)
    j = 0
    @inbounds while true
        x = log1p(-rand()) / lq
        j + 1.0 + x >= N + 1.0 && break
        j += 1 + floor(Int, x)
        pj = Float64(rates[j] * dt)
        (pj >= pmax || rand() * pmax < pj) && _layer_fire!(p, j)
    end
    return nothing
end

# Bernoulli(pr) trial for every layer neuron, by geometric skipping.
function _bernoulli_layer!(p::PoissonStimulusLayer, pr::Float64)
    pr > 0 || return nothing
    N = p.N
    if pr >= _DENSE_SPIKE_P
        @inbounds for j in 1:N
            rand() < pr && _layer_fire!(p, j)
        end
        return nothing
    end
    lq = log1p(-pr)
    j = 0
    @inbounds while true
        x = log1p(-rand()) / lq
        j + 1.0 + x >= N + 1.0 && break
        j += 1 + floor(Int, x)
        _layer_fire!(p, j)
    end
    return nothing
end

spike_index(p::PoissonStimulusLayer) = p.spikes

export PoissonLayer, stimulate!, PoissonStimulusLayer, Stimulus, PoissonLayerHet
