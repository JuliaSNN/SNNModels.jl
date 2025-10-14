"""
    CurrentParameter

Abstract type representing parameters for current stimuli.
"""
abstract type CurrentParameter end


"""
    CurrentNoise{VFT = Vector{Float32}} <: CurrentParameter

A parameter type for current stimuli that generates noisy current inputs.

# Fields
- `I_base::VFT`: Base current values for each neuron.
- `I_dist::Distribution{Univariate,Continuous}`: Distribution for generating noise.
- `α::VFT`: Decay factors for the current.
"""
CurrentNoise

@snn_kw struct CurrentNoise{VFT = Vector{Float32}, DT = Distribution{Univariate,Continuous}} <: CurrentParameter
    I_base::VFT = zeros(Float32, 0)
    I_dist::DT = Normal(0.0, 0.0)
    α::VFT = ones(Float32, 0)
end

"""
    CurrentNoise(N::Union{Number,AbstractPopulation}; I_base::Number = 0, I_dist::Distribution = Normal(0.0, 0.0), α::Number = 0.0)

Construct a `CurrentNoise` with the given parameters.

# Arguments
- `N`: Number of neurons or a population.
- `I_base`: Base current value (default: 0).
- `I_dist`: Distribution for generating noise (default: Normal(0.0, 0.0)).
- `α`: Decay factor (default: 0.0).
"""
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
    CurrentStimulus{
        FT = Float32,
        VFT = Vector{Float32},
        DT = Distribution{Univariate,Continuous},
        VIT = Vector{Int},
    } <: AbstractStimulus

A stimulus that applies current to neurons.

# Fields
- `param::CurrentStimulus`:  for the stimulus.
- `name::String`: Name of the stimulus (default: "Current").
- `id::String`: Unique identifier for the stimulus.
- `neurons::VIT`: Indices of neurons to stimulate.
- `randcache::VFT`: Cache for random values.
- `I::VFT`: Target input current.
- `records::Dict`: Dictionary for recording data.
- `targets::Dict`: Dictionary describing the targets of the stimulus.
"""
CurrentStimulus
@snn_kw struct CurrentStimulus{
    VFT = Vector{Float32},
} <: AbstractStimulus
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

"""
    CurrentStimulus(post::T, sym::Symbol = :I; neurons = :ALL, param, kwargs...) where {T<:AbstractPopulation}
Construct a `CurrentStimulus` for a postsynaptic population.
    
# Arguments
- `post`: Postsynaptic population.
- `sym`: Symbol for the input current field (default: :I).
- `neurons`: Indices of neurons to stimulate (default: :ALL).
- `param`: Parameters for the stimulus.
- `kwargs`: Additional keyword arguments.
"""
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
    CurrentStimulus(param::CurrentStimulus, post::T, sym::Symbol = :I; kwargs...) where {T<:AbstractPopulation}

Construct a `CurrentStimulus` with the given parameters and postsynaptic population.

# Arguments
- `param`: Parameters for the stimulus.
- `post`: Postsynaptic population.
- `sym`: Symbol for the input current field (default: :I).
- `kwargs`: Additional keyword arguments.
"""
function Stimulus(param::CurrentParameter, post::T, sym::Symbol = :I; kwargs...) where {T<:AbstractPopulation}
    return CurrentStimulus(post, sym; param, kwargs...)
end

#### Methods

"""
    stimulate!(p, param::CurrentNoise, time::Time, dt::Float32)

Generate a noisy current stimulus for a postsynaptic population.

# Arguments
- `p`: Current stimulus.
- `param`: Parameters for the noise.
- `time`: Current time.
- `dt`: Time step.
"""
function stimulate!(p, param::CurrentNoise, time::Time, dt::Float32)
    @unpack I, neurons, randcache = p
    @unpack I_base, I_dist, α = param
    rand!(I_dist, randcache)
    @inbounds @simd for i in p.neurons
        I[i] = (I_base[i] + randcache[i])*(1-α[i]) + I[i] * (α[i])
    end
end

export CurrentStimulus,
    CurrentParameter,
    stimulate!,
    CurrentNoise

