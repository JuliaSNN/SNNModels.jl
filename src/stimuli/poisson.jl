abstract type PoissonStimulusParameter end

"""
    PoissonVariable

    Poisson stimulus with rate defined with a function.
    
    # Fields
    - `variables::Dict{Symbol,Any}`: A dictionary containing the variables for the function.
    - `rate::Function`: A function defining the rate of the Poisson stimulus.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonVariable

@snn_kw struct PoissonVariable{VFT} <: PoissonStimulusParameter
    variables::Dict{Symbol,Any}
    rate::Function
    μ::Float32 = 1.0f0
    active::Vector{Bool} = [true]
end

"""
    PoissonFixed

    Poisson stimulus with fixed rate. The rate arrives to all the neuronstargeted
    by the stimulus.

    # Fields
    - `rate::Vector{R}`: A vector containing the rate of the Poisson stimulus.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonFixed

@snn_kw struct PoissonFixed{R = Float32} <: PoissonStimulusParameter
    rate::R = 0
    μ::R = 1.0f0
    active::Vector{Bool} = [true]
end


"""
    PoissonInterval

    Poisson stimulus with rate defined for each cell in the layer. Each neuron of the 'N' Poisson population fires with 'rate' in the intervals defined by 'intervals'.
    
    # Fields
    - `rate::Vector{R}`: A vector containing the rate of the Poisson stimulus.
    - `intervals::Vector{Vector{R}}`: A vector of vectors containing the intervals in which the Poisson stimulus is active.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonInterval

@snn_kw struct PoissonInterval{R = Float32} <: PoissonStimulusParameter
    rate::R
    intervals::Vector{Vector{R}}
    μ::R = 1.0f0
    active::Vector{Bool} = [true]
end


@snn_kw struct PoissonStimulus{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    VIT = Vector{Int},
    IT = Int32,
} <: AbstractStimulus
    id::String = randstring(12)
    name::String = "Poisson"
    param::PoissonStimulusParameter
    ##
    neurons::VIT
    g::VFT # target conductance for soma
    targets::Dict = Dict()
    records ::Dict = Dict()
end


function Stimulus(
    param::PoissonStimulusParameter,
    post::T,
    sym::Symbol,
    target = nothing;
    neurons = :ALL,
    μ = 1.0f0,
    kwargs...,
) where {T<:AbstractPopulation}
    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, target)
    neurons = neurons == :ALL ? eachindex(1:post.N) : neurons

    # Construct the SpikingSynapse instance
    return PoissonStimulus(;
        param = param,
        targets = targets,
        neurons = neurons,
        g = g,
        kwargs...,
    )
end


function get_poisson_rate(param::PoissonVariable, time::Time)
    return param.rate(get_time(time), param.variables)
end

function get_poisson_rate(param::PoissonFixed, time::Time)
    return param.rate
end

function get_poisson_rate(param::PoissonInterval, time::Time)
    for int in param.intervals
        if get_time(time) > int[1] && get_time(time) < int[end]
            return param.rate
        end
    end
    return 0
end

function stimulate!(
    p::PoissonStimulus,
    param::PoissonStimulusParameter,
    time::Time,
    dt::Float32,
)
    @unpack active = param
    if !active[1]
        return
    end
    @unpack μ = param
    @unpack neurons, g = p
    my_rate = Distributions.Poisson{Float32}(get_poisson_rate(param, time) * dt)
    for _ in 1:rand(my_rate)
        @fastmath @simd for n in neurons
            g[n] += μ 
        end
    end
end



export PoissonStimulus,
    stimulate!,
    PSParam,
    PoissonStimulusParameter,
    PoissonVariable,
    PoissonFixed,
    PoissonInterval,
    get_poisson_rate
