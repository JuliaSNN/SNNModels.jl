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
- `active::Vector{Bool} = [true]`: present for API uniformity but NOT read by
  `stimulate!` in SNNModels 1.8.4 (the layer always fires).

The positional form `PoissonLayer(rate; N)` requires the keyword `N`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
stim = SNN.Stimulus(SNN.PoissonLayer(rate = 10Hz, N = 200), E, :ge; conn = (p = 0.1, μ = 1.0))
```
"""
PoissonLayer

@snn_kw struct PoissonLayer{FT = Float32} <: PoissonLayerParameter
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
- `active::Vector{Bool} = [true]`: NOT read by `stimulate!` in SNNModels 1.8.4.

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

function PoissonLayer(rate::R; kwargs...) where {R<:Real}
    N = kwargs[:N]
    return PoissonLayer(; N = N, rate = rate)
end

function PoissonLayerHet(rate::R; kwargs...) where {R<:Real}
    N = kwargs[:N]
    return PoissonLayerHet(; N = N, rates = fill(rate, N))
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

# Fields
- `N::Int`: number of layer neurons; `id`, `name = "Poisson"`
- `param::PoissonLayerParameter`
- `g::Vector{Float32}`: target variable of `post` (shared).
- `colptr`, `rowptr`, `I`, `J`, `index`, `W`: sparse connectivity, as returned by `dsparse`
  (`I` are postsynaptic indices, `W` the weights).
- `fire::Vector{Bool}`: spikes of the layer neurons in the last step (can be recorded with
  `monitor!(stim, :fire)`).
- `randcache::Vector{Float32}`: buffer of uniform random numbers.
- `records::Dict`, `targets::Dict`

Do not rely on the struct default `param = PoissonLayer(-1)`: it calls the positional
constructor without `N` and errors; always pass `param`.

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
@snn_kw struct PoissonStimulusLayer{VFT = Vector{Float32},PT<:PoissonLayerParameter} <:
               AbstractStimulus
    N::Int
    id::String = randstring(12)
    name::String = "Poisson"
    param::PT = PoissonLayer(-1)
    ##
    g::VFT # target conductance for soma
    colptr::VIT
    rowptr::VIT
    I::VIT
    J::VIT
    index::VIT
    W::VFT
    fire::VBT = zeros(Bool, N)
    ##
    randcache::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
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

    w = sparse_matrix(param.N, post.N, conn)
    rowptr, colptr, I, J, index, W = dsparse(w)
    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, comp)

    # Construct the SpikingSynapse instance
    return PoissonStimulusLayer(;
        param = param,
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

    w = sparse_matrix(param.N, post.N; conn...)
    rowptr, colptr, I, J, index, W = dsparse(w)
    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, comp)

    # Construct the SpikingSynapse instance
    return PoissonStimulusLayer(;
        param = param,
        N = param.N,
        targets = targets,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
        name = name,
    )
end


"""
    stimulate!(p::PoissonStimulusLayer, param::Union{PoissonLayer,PoissonLayerHet}, time::Time, dt::Float32)

One step of a Poisson layer: each layer neuron `j` fires if a uniform draw is smaller than
`rate * dt` (`rates[j] * dt` for `PoissonLayerHet`); for each firing neuron the weights of its
outgoing connections are added to `p.g`. `p.fire` holds the layer spikes of this step. The
`active` flag is not checked.
"""
function stimulate!(p::PoissonStimulusLayer, param::PoissonLayer, time::Time, dt::Float32)
    @unpack N, randcache, fire, colptr, W, I, g = p
    @unpack rate = param
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < rate * dt
            fire[j] = true
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s]
            end
        else
            fire[j] = false
        end
    end
end

function stimulate!(
    p::PoissonStimulusLayer,
    param::PoissonLayerHet,
    time::Time,
    dt::Float32,
)
    @unpack N, randcache, fire, colptr, W, I, g = p
    @unpack rates = param
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < rates[j] * dt
            fire[j] = true
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s]
            end
        else
            fire[j] = false
        end
    end
end

export PoissonLayer, stimulate!, PoissonStimulusLayer, Stimulus, PoissonLayerHet
