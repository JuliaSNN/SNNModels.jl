"""
    PoissonStimulusParameter <: AbstractStimulusParameter

Supertype of the rate parameters of a [`PoissonStimulus`](@ref): `PoissonFixed`,
`PoissonInterval` and `PoissonVariable`. Each subtype defines `get_poisson_rate(param, time)`
and has the fields `μ` (increment per input spike) and `active` (on/off switch).
"""
abstract type PoissonStimulusParameter <: AbstractStimulusParameter end

"""
    PoissonVariable(; variables::Dict{Symbol,Any}, rate::Function, μ = 1.0f0, active = [true])

Rate parameter of a [`PoissonStimulus`](@ref) whose rate is a function of time.

At every step the rate is `rate(get_time(time), variables)`: `rate` must accept the current
time (ms) and the `variables` dictionary and return a rate in library units (use `Hz`).
`variables` can be changed at runtime with `set_variable!(stim, key, value)`.

# Fields
- `variables::Dict{Symbol,Any}` (required): arguments passed to `rate`.
- `rate::Function` (required): `(t, variables) -> rate`.
- `μ::Float32 = 1.0`: increment of the target variable per input spike (units of the target
  variable, e.g. nS for a conductance).
- `active::Vector{Bool} = [true]`: the stimulus is applied only if `active[1]`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
osc(t, v) = v[:r0] * (1 + sin(2π * t / v[:period]))
param = SNN.PoissonVariable(variables = Dict{Symbol,Any}(:r0 => 10Hz, :period => 100ms), rate = osc)
stim = SNN.Stimulus(param, E, :ge)
```
"""
PoissonVariable

@snn_kw struct PoissonVariable{FT = Float32,VDT = Dict{Symbol,Any}} <: PoissonStimulusParameter
    variables::VDT
    rate::Function
    μ::FT = 1.0f0
    active::VBT = [true]
end

"""
    PoissonFixed(; rate = 0, μ = 1.0f0, active = [true])

Rate parameter of a [`PoissonStimulus`](@ref) with a constant rate. Every targeted neuron
receives an independent Poisson spike train of rate `rate`.

# Fields
- `rate::Float32 = 0`: rate of the Poisson input per target neuron (library units, use
  `Hz`; e.g. `rate = 10Hz`).
- `μ::Float32 = 1.0`: increment of the target variable per input spike.
- `active::Vector{Bool} = [true]`: the stimulus is applied only if `active[1]`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
stim = SNN.Stimulus(SNN.PoissonFixed(rate = 100Hz, μ = 0.5), E, :ge)
```
"""
PoissonFixed

@snn_kw mutable struct PoissonFixed{R = Float32} <: PoissonStimulusParameter
    rate::R = 0
    μ::R = 1.0f0
    active::VBT = [true]
end


"""
    PoissonInterval(; rate, intervals = Vector{Vector{Float32}}([]), μ = 1.0f0, active = [true])

Rate parameter of a [`PoissonStimulus`](@ref) that is on only inside given time intervals.
The rate is `rate` when the current time `t` satisfies `int[1] < t < int[end]` for at least
one interval `int` in `intervals`, and `0` otherwise.

# Fields
- `rate::Float32` (required): rate per target neuron while active (use `Hz`).
- `intervals::Vector{Vector{Float32}} = []`: list of `[start, end]` intervals in ms; can be
  replaced at runtime with `set_intervals!`.
- `μ::Float32 = 1.0`: increment of the target variable per input spike.
- `active::Vector{Bool} = [true]`: the stimulus is applied only if `active[1]`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
param = SNN.PoissonInterval(rate = 50Hz, intervals = [[100ms, 200ms], [400ms, 500ms]])
stim = SNN.Stimulus(param, E, :ge)
```
"""
PoissonInterval

@snn_kw mutable struct PoissonInterval{R = Float32,VVFT = Vector{Vector{Float32}}} <: PoissonStimulusParameter
    rate::R
    intervals::VVFT= Vector{Vector{Float32}}([])
    μ::R = 1.0f0
    active::VBT = [true]
end


@doc raw"""
    PoissonStimulus(post::AbstractPopulation, sym::Symbol; param::PoissonStimulusParameter,
                    neurons = :ALL, comp = nothing, p_post = -1, name = "Poisson")
    Stimulus(param::PoissonStimulusParameter, post, sym; kwargs...)

Independent Poisson spike trains delivered directly to a target variable of `post`.

Each targeted neuron `n` receives its own Poisson process of rate ``\nu(t)`` given by
`get_poisson_rate(param, time)`. There is no presynaptic population and no weight matrix:
every input spike adds `param.μ` to the target variable `g` (the receptor or conductance
selected by `sym`, and `comp` for multicompartment neurons).

# Equations
At each step, for every targeted neuron ``n``:
```math
g_n \leftarrow g_n + \mu\, k_n, \qquad k_n \sim \mathrm{Poisson}(\nu(t)\,\Delta t)
```
Nothing is added when ``\nu(t)\,\Delta t \le 0`` (or approximately 0), or when
`param.active[1] == false`.

# Constructor arguments
- `post`: target population.
- `sym::Symbol`: target variable, resolved by `synaptic_target` of the population type
  (e.g. `:ge`/`:glu` and `:gi`/`:gaba` for generalized IF neurons).
- `param` (required): `PoissonFixed`, `PoissonInterval` or `PoissonVariable`.
- `neurons = :ALL`: all neurons of `post`; a `Vector{Int}` of indices; or `:p_post` to sample
  `round(Int, p_post * post.N)` neurons without replacement (if `p_post` is outside
  `[0, 1]` a warning is emitted and all neurons are used).
- `comp = nothing`: compartment (multicompartment models, e.g. `:d1`).
- `p_post = -1`: fraction of targeted neurons, used only with `neurons = :p_post`.
- remaining keywords (e.g. `name = "Poisson"`, `id`) are forwarded to the struct.

# Fields
- `id::String`, `name::String = "Poisson"`
- `param::PoissonStimulusParameter`
- `neurons::Vector{Int}`: targeted neuron indices.
- `g::Vector{Float32}`: the target variable of `post` (shared, not copied).
- `targets::Dict`, `records::Dict`

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
stim = SNN.PoissonStimulus(E, :ge; param = SNN.PoissonFixed(rate = 20Hz), neurons = :p_post, p_post = 0.5)
model = SNN.compose(; E, stim)
SNN.monitor!(E, :fire)
SNN.sim!(; model, duration = 100ms)
```
"""
PoissonStimulus

@snn_kw struct PoissonStimulus{VFT = Vector{Float32}} <: AbstractStimulus
    id::String = randstring(12)
    name::String = "Poisson"
    param::PoissonStimulusParameter
    ##
    neurons::VIT
    g::VFT # target conductance for soma
    targets::Dict = Dict()
    records::Dict = Dict()
end


function PoissonStimulus(
    post::T,
    sym::Symbol;
    param::PoissonStimulusParameter,
    neurons = :ALL,
    comp = nothing,
    p_post = -1,
    kwargs...,
) where {T<:AbstractPopulation}
    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, comp)
    if neurons == :ALL
        neurons = eachindex(1:post.N)
    elseif isa(neurons, Vector{Int})
        neurons = neurons
    elseif neurons == :p_post
        if p_post < 0 || p_post > 1
            @warn "p_post should be between 0 and 1. Setting neurons to :ALL."
            neurons = eachindex(1:post.N)
        else
            neurons = sample(1:post.N, round(Int, p_post * post.N); replace = false)
        end
    end

    # Construct the SpikingSynapse instance
    return PoissonStimulus(;
        param = param,
        targets = targets,
        neurons = neurons,
        g = g,
        kwargs...,
    )
end


"""
    Stimulus(param::PoissonStimulusParameter, post::AbstractPopulation, sym::Symbol; kwargs...)
    Stimulus(param::PoissonStimulusParameter, post::AbstractPopulation, sym::Symbol, comp; kwargs...)

Build a [`PoissonStimulus`](@ref) on `post`; equivalent to
`PoissonStimulus(post, sym; param, kwargs...)` (with `comp` the compartment, as for the other
stimulus types).
"""
function Stimulus(
    param::PoissonStimulusParameter,
    post::T,
    sym::Symbol;
    kwargs...,
) where {T<:AbstractPopulation}
    return PoissonStimulus(post, sym; param, kwargs...)
end

Stimulus(param::PoissonStimulusParameter, post::T, sym::Symbol, comp; kwargs...) where {T<:AbstractPopulation} =
    PoissonStimulus(post, sym; param, comp, kwargs...)


"""
    get_poisson_rate(param::PoissonStimulusParameter, time::Time)

Rate of a Poisson stimulus at the current time: `param.rate(get_time(time), param.variables)`
for `PoissonVariable`, `param.rate` for `PoissonFixed`, and for `PoissonInterval` `param.rate`
inside one of `param.intervals` (strict inequalities) and `0` outside.
"""
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

"""
    stimulate!(p::PoissonStimulus, param::PoissonStimulusParameter, time::Time, dt::Float32)

If `param.active[1]`, draw for every neuron in `p.neurons` a Poisson count with mean
`get_poisson_rate(param, time) * dt` and add `param.μ` times that count to `p.g`.
"""
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
    poisson_rate = get_poisson_rate(param, time) * dt
    poisson_rate ≈ 0 && return
    poisson_rate < 0 && return
    my_rate = Distributions.Poisson{Float32}(poisson_rate)
    @fastmath @simd for n in neurons
        g[n] += μ * rand(my_rate)
    end
end



export PoissonStimulus,
    stimulate!,
    PoissonStimulusParameter,
    PoissonVariable,
    PoissonFixed,
    PoissonInterval,
    get_poisson_rate
