# variables = Dict(
#     :amplitude => 50pA,
#     :frequency => 1Hz,
#     :shift_phase => π*3/4, # Phase shift for each neuron
# )

# current_param = SNN.CurrentVariableParameter(variables, sinusoidal_current )
struct VariableInputParameter{R<:Number} <: AbstractParameter
    variables::Dict{Symbol,Any}
    target::Symbol
    func::Function
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

"""
    stimulate!(p, param::CurrentVariableParameter, time::Time, dt::Float32)

Generate a current stimulus based on variables and a function.

# Arguments
- `p`: Current stimulus.
- `param`: Parameters for the variable current.
- `time`: Current time.
- `dt`: Time step.
"""
function stimulate!(p, param::VariablesParameter, time::Time, dt::Float32)
    @unpack I, neurons, randcache = p
    @unpack variables, func = param
    @inbounds @simd for i in p.neurons
        I[i] = func(variables, get_time(time), i)
    end
end



function OrnsteinUhlenbeckProcess(x::Float32, param::PSParam)
    X::Float32 = param.variables[:X]
    θ::Float32 = param.variables[:θ]
    μ::Float32 = param.variables[:μ]
    σ::Float32 = param.variables[:σ]
    dt::Float32 = param.variables[:dt]

    ξ = rand(Normal())
    X = X + θ * (μ - X) * dt + σ * ξ * dt
    X = X > 0.0f0 ? X : 0.0f0

    param.variables[:X] = X
    return X
end

function SinWaveNoise(x::Float32, param::PSParam)
    X::Float32 = param.variables[:X]
    θ::Float32 = param.variables[:θ]
    σ::Float32 = param.variables[:σ]
    dt::Float32 = param.variables[:dt]
    ν::Float32 = param.variables[:ν]
    μ::Float32 = param.variables[:μ]

    W = σ * rand(Normal()) * sqrt(dt)
    X = X + θ * (μ - X) * dt - W
    param.variables[:X] = X

    Y = sin(x * 2π * ν)
    return X * 0.1 + Y * μ
end
