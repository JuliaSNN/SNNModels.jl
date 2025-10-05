"""
    CurrentStimulusParameter

Abstract type representing parameters for current stimuli.
"""
abstract type CurrentStimulusParameter end

"""
    CurrentVariableParameter{VFT = Vector{Float32}} <: CurrentStimulusParameter

A parameter type for current stimuli that uses a function to compute the current based on variables.

# Fields
- `variables::Dict{Symbol,VFT}`: Dictionary of variables used in the function.
- `func::Function`: Function that computes the current based on variables, time, and neuron index.
"""
@snn_kw struct CurrentVariableParameter{VFT = Vector{Float32}} <: CurrentStimulusParameter
    variables::Dict{Symbol,VFT} = Dict{Symbol,VFT}()
    func::Function
end

"""
    CurrentNoiseParameter{VFT = Vector{Float32}} <: CurrentStimulusParameter

A parameter type for current stimuli that generates noisy current inputs.

# Fields
- `I_base::VFT`: Base current values for each neuron.
- `I_dist::Distribution{Univariate,Continuous}`: Distribution for generating noise.
- `α::VFT`: Decay factors for the current.
"""
@snn_kw struct CurrentNoiseParameter{VFT = Vector{Float32}} <: CurrentStimulusParameter
    I_base::VFT = zeros(Float32, 0)
    I_dist::Distribution{Univariate,Continuous} = Normal(0.0, 0.0)
    α::VFT = ones(Float32, 0)
end

"""
    CurrentNoiseParameter(N::Union{Number,AbstractPopulation}; I_base::Number = 0, I_dist::Distribution = Normal(0.0, 0.0), α::Number = 0.0)

Construct a `CurrentNoiseParameter` with the given parameters.

# Arguments
- `N`: Number of neurons or a population.
- `I_base`: Base current value (default: 0).
- `I_dist`: Distribution for generating noise (default: Normal(0.0, 0.0)).
- `α`: Decay factor (default: 0.0).
"""
function CurrentNoiseParameter(
    N::Union{Number,AbstractPopulation};
    I_base::Number = 0,
    I_dist::Distribution = Normal(0.0, 0.0),
    α::Number = 0.0,
)
    if isa(N, AbstractPopulation)
        N = N.N
    end
    return CurrentNoiseParameter(
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
- `param::CurrentStimulusParameter`: Parameters for the stimulus.
- `name::String`: Name of the stimulus (default: "Current").
- `id::String`: Unique identifier for the stimulus.
- `neurons::VIT`: Indices of neurons to stimulate.
- `randcache::VFT`: Cache for random values.
- `I::VFT`: Target input current.
- `records::Dict`: Dictionary for recording data.
- `targets::Dict`: Dictionary describing the targets of the stimulus.
"""
@snn_kw struct CurrentStimulus{
    FT = Float32,
    VFT = Vector{Float32},
    DT = Distribution{Univariate,Continuous},
    VIT = Vector{Int},
} <: AbstractStimulus
    param::CurrentStimulusParameter
    name::String = "Current"
    id::String = randstring(12)
    neurons::VIT
    ##

    randcache::VFT = rand(length(neurons)) # random cache
    I::VFT # target input current
    records::Dict = Dict()
    targets::Dict = Dict()
end

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
    CurrentStimulus(param::CurrentStimulusParameter, post::T, sym::Symbol = :I; kwargs...) where {T<:AbstractPopulation}

Construct a `CurrentStimulus` with the given parameters and postsynaptic population.

# Arguments
- `param`: Parameters for the stimulus.
- `post`: Postsynaptic population.
- `sym`: Symbol for the input current field (default: :I).
- `kwargs`: Additional keyword arguments.
"""
function CurrentStimulus(param::CurrentStimulusParameter, post::T, sym::Symbol = :I; kwargs...) where {T<:AbstractPopulation}
    return CurrentStimulus(post, sym; param, kwargs...)
end

"""
    stimulate!(p, param::CurrentNoiseParameter, time::Time, dt::Float32)

Generate a noisy current stimulus for a postsynaptic population.

# Arguments
- `p`: Current stimulus.
- `param`: Parameters for the noise.
- `time`: Current time.
- `dt`: Time step.
"""
function stimulate!(p, param::CurrentNoiseParameter, time::Time, dt::Float32)
    @unpack I, neurons, randcache = p
    @unpack I_base, I_dist, α = param
    rand!(I_dist, randcache)
    @inbounds @simd for i in p.neurons
        I[i] = (I_base[i] + randcache[i])*(1-α[i]) + I[i] * (α[i])
    end
end

"""
    stimulate!(p, param::CurrentVariableParameter, time::Time, dt::Float32)

Generate a current stimulus based on variables and a function.

# Arguments
- `p`: Current stimulus.
- `param`: Parameters for the variable current.
- `time`: Current time.
- `dt`: Time step.
"""
function stimulate!(p, param::CurrentVariableParameter, time::Time, dt::Float32)
    @unpack I, neurons, randcache = p
    @unpack variables, func = param
    @inbounds @simd for i in p.neurons
        I[i] = func(variables, get_time(time), i)
    end
end

"""
    ramping_current(variables::Dict, t::Float32, args...)

Compute a ramping current based on variables.

# Arguments
- `variables`: Dictionary of variables (must contain :peak, :start_time, :peak_time, :end_time).
- `t`: Current time.
- `args...`: Additional arguments (unused).
"""
function ramping_current(variables::Dict, t::Float32, args...)
    peak = variables[:peak]
    start_time = variables[:start_time]
    peak_time = variables[:peak_time]
    end_time = variables[:end_time]
    if t < start_time || t > end_time
        return 0pA
    end
    if t >= start_time && t <= peak_time
        return peak * (t - start_time) / (peak_time - start_time)
    end
end

export CurrentStimulus,
    CurrentStimulusParameter,
    stimulate!,
    CurrentNoiseParameter,
    CurrentVariableParameter,
    ramping_current

