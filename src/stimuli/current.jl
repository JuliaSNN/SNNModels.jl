"""
    CurrentParameter <: AbstractStimulusParameter

Supertype of the parameters of a [`CurrentStimulus`](@ref). The only subtype loaded in
SNNModels 1.8.4 is [`CurrentNoise`](@ref). (A time-dependent current parameter exists in
`src/stimuli/variable_inputs.jl`, which is not included by the package.)
"""
abstract type CurrentParameter <: AbstractStimulusParameter end


@doc raw"""
    CurrentNoise(N::Union{Number,AbstractPopulation}; I_base = 0, I_dist = Normal(0.0, 0.0), α = 0.0)
    CurrentNoise(; I_base = zeros(Float32, 0), I_dist = Normal(0.0, 0.0), α = ones(Float32, 0))

Parameter of a [`CurrentStimulus`](@ref): a constant current plus noise, optionally
low-pass filtered.

# Equations
At every step, for each stimulated neuron ``i`` (with one independent draw
``\xi_i \sim`` `I_dist` per stimulated neuron):
```math
I_i \leftarrow (1 - \alpha_i)\,(I^{base}_i + \xi_i) + \alpha_i\, I_i
```
With ``\alpha_i = 0`` the current is redrawn independently every step (white noise around
`I_base`); with ``0 < \alpha_i < 1`` it is an AR(1) (exponentially filtered) process with
correlation time ``\Delta t / (1 - \alpha_i)``. The noise amplitude is not scaled with
`dt`: the standard deviation of `I_dist` is the per-step standard deviation.

# Fields
- `I_base::Vector{Float32}`: baseline current per neuron (pA), indexed by neuron id, so its
  length must be the size of the target population.
- `I_dist::Distribution{Univariate,Continuous} = Normal(0.0, 0.0)`: noise distribution (pA).
- `α::Vector{Float32}`: filter coefficient per neuron (dimensionless, in ``[0, 1]``),
  indexed by neuron id.

The convenience constructor `CurrentNoise(N; I_base, I_dist, α)` takes the population (or
its size) and fills `I_base` and `α` with the given scalars (default `α = 0.0`). The keyword
struct defaults are empty vectors, to be filled by the user.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
param = SNN.CurrentNoise(E; I_base = 100pA, I_dist = SNN.SNNModels.Normal(0.0, 20.0), α = 0.9)
stim = SNN.Stimulus(param, E)
```
"""
CurrentNoise

@snn_kw struct CurrentNoise{
    VFT = Vector{Float32},
    DT = Distribution{Univariate,Continuous},
} <: CurrentParameter
    I_base::VFT = zeros(Float32, 0)
    I_dist::DT = Normal(0.0, 0.0)
    α::VFT = ones(Float32, 0)
end

function CurrentNoise(
    N::Union{Number,AbstractPopulation};
    I_base::Number = 0,
    I_dist::Distribution = Normal(0.0, 0.0),
    α::Number = 0.0,
)
    if isa(N, AbstractPopulation)
        N = N.N
    end
    return CurrentNoise(
        I_base = fill(Float32(I_base), N),
        I_dist = I_dist,
        α = fill(Float32(α), N),
    )
end

"""
    CurrentStimulus(post::AbstractPopulation, sym::Symbol = :I; neurons = :ALL, param::CurrentParameter, kwargs...)
    Stimulus(param::CurrentParameter, post::AbstractPopulation, sym::Symbol = :I; kwargs...)

A stimulus that writes an external current into the field `sym` (default `:I`) of the
population `post`, for the neurons in `neurons`. The update rule is defined by the
parameter type (see [`CurrentNoise`](@ref)).

The stimulus writes (does not add to) the target vector, so two current stimuli on the
same field and neurons overwrite each other. Neurons that are not in `neurons` are never
modified. The current keeps its last value after the stimulus stops being called.

Since SNNModels 1.8.4 a subset of neurons is handled correctly: the per-step random
numbers are indexed by position in `neurons`, while `I`, `I_base` and `α` are indexed by
neuron id.

# Constructor arguments
- `post`: target population; `sym = :I`: field of `post` that receives the current.
- `neurons = :ALL`: `1:post.N`, or a vector of neuron indices.
- `param` (required): a `CurrentParameter`.
- remaining keywords (`name = "Current"`, `id`) are forwarded to the struct.

# Fields
- `param::CurrentParameter`; `name::String = "Current"`; `id::String`
- `neurons::Vector{Int}`: stimulated neurons.
- `randcache::Vector{Float32}`: one random number per stimulated neuron.
- `I::Vector{Float32}`: the target field of `post` (shared, not copied).
- `records::Dict`, `targets::Dict`

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS))
stim = SNN.CurrentStimulus(E; neurons = [1, 2, 3], param = SNN.CurrentNoise(E; I_base = 500pA))
model = SNN.compose(; E, stim)
SNN.monitor!(E, [:v, :fire])
SNN.sim!(; model, duration = 100ms)
```
"""
CurrentStimulus
@snn_kw struct CurrentStimulus{VFT = Vector{Float32}} <: AbstractStimulus
    param::CurrentParameter
    name::String = "Current"
    id::String = randstring(12)
    neurons::VIT
    ##

    randcache::VFT = rand(length(neurons)) # random cache
    I::VFT # target input current
    records::Dict = Dict()
    targets::Dict = Dict()
end

#### Constructors

function CurrentStimulus(
    post::T,
    sym::Symbol = :I;
    neurons = :ALL,
    param,
    kwargs...,
) where {T<:AbstractPopulation}
    if neurons == :ALL
        neurons = 1:post.N
    end
    targets =
        Dict(:pre => :Current, :post => post.id, :sym => :soma, :type=>:CurrentStimulus)
    return CurrentStimulus(
        neurons = neurons,
        I = getfield(post, sym),
        targets = targets;
        param = param,
        kwargs...,
    )
end

"""
    Stimulus(param::CurrentParameter, post::AbstractPopulation, sym::Symbol = :I; kwargs...)

Build a [`CurrentStimulus`](@ref); equivalent to `CurrentStimulus(post, sym; param, kwargs...)`.
"""
function Stimulus(
    param::CurrentParameter,
    post::T,
    sym::Symbol = :I;
    kwargs...,
) where {T<:AbstractPopulation}
    return CurrentStimulus(post, sym; param, kwargs...)
end

#### Methods

"""
    stimulate!(p::CurrentStimulus, param::CurrentNoise, time::Time, dt::Float32)

Draw one noise sample per stimulated neuron from `param.I_dist` and set, for each
`i = p.neurons[k]`, `p.I[i] = (I_base[i] + ξ[k]) * (1 - α[i]) + p.I[i] * α[i]`.
"""
function stimulate!(p, param::CurrentNoise, time::Time, dt::Float32)
    @unpack I, neurons, randcache = p
    @unpack I_base, I_dist, α = param
    rand!(I_dist, randcache)
    # randcache has one entry per stimulated neuron: index it by position k, not by neuron id i
    @inbounds @simd for k in eachindex(neurons)
        i = neurons[k]
        I[i] = (I_base[i] + randcache[k])*(1-α[i]) + I[i] * (α[i])
    end
end

export CurrentStimulus, CurrentParameter, stimulate!, CurrentNoise
