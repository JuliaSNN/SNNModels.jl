
"""
    train!(P::Vector{<:AbstractPopulation}, C::Vector{<:AbstractConnection} = [EmptySynapse()],
           S::Vector{<:AbstractStimulus} = [EmptyStimulus()];
           dt = 0.125ms, duration = 10ms, time = Time(), perturbation! = nothing, pbar = false)
    train!(; model, duration = 10ms, dt = 0.125ms, pbar = false, perturbation! = nothing)
    train!(model::NamedTuple, duration = 1s; kwargs...)
    train!(args...; model, kwargs...)

Simulate a network with plasticity for `duration` (ms) with time step `dt` (ms).

`train!` is the only entry point that applies plasticity. Each step runs:
1. `update_time!` (the clock advances by `dt` first);
2. for every stimulus: `stimulate!`, `record!`;
3. for every population: `update_traces!`, `integrate!`, `plasticity!`, `record!`;
4. for every connection: `update_traces!`, `forward!`, `plasticity!`, `record!`.

`sim!` runs the same sequence without `update_traces!` and `plasticity!`, so weights and STP
variables only change under `train!` (long-term STDP/iSTDP/vSTDP rules, short-term
plasticity, metaplasticity components).

# Arguments
- `model`: a NamedTuple built with `compose`; its populations, connections and stimuli
  (stimulus groups are unpacked) are simulated and `model.time` is advanced.
- `args...`: extra components (or vectors of them) added to those of `model`.
- `dt = 0.125ms`: time step (converted to `Float32`).
- `duration = 10ms` (vector and keyword forms) or `1s` (positional model form): simulated time; the loop
  runs over `0:dt:(duration - dt)`, i.e. `round(duration / dt)` steps.
- `time = Time()`: clock of the vector form.
- `perturbation!`: optional function called before every step as
  `perturbation!(; P, C, S, t, dt, start_time)`.
- `pbar = false`: show a progress bar with the running firing rates of the populations that
  record `:fire`.

Recording buffers are sized before the loop from `duration` (see `monitor!`). If the model
time is 0, a first sample of every record is taken before the first step.

# Returns
The `Time` object (vector form) or the model time in ms after the run (model forms).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
stim = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 0.5), LTPParam = SNN.STDPGerstner())
model = SNN.compose(; E, EE, stim)
SNN.train!(; model, duration = 100ms)
SNN.get_time(model.time)   # 100.0
```
"""
function train!(
    P::Vector{TP},
    C::Vector{TC} = [EmptySynapse()],
    S::Vector{TS} = [EmptyStimulus()];
    dt = 0.125ms,
    duration = 10ms,
    time = Time(),
    perturbation! = nothing,
    pbar = false,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    dt = Float32(dt)
    _allocate_records!(P, C, S, dt, Float32(duration))
    dts = 0.0f0:dt:(duration-dt)
    iter = pbar ? ProgressBar(dts, printing_delay=0.1) : dts
    start_time = get_time(time)
    firing_rates = Dict{String, Float32}(p.name => 0.f0 for p in P if haskey(p.records, :fire))
    τ_rate = 100.0f0
    for t in iter
        if pbar
            map(P) do p
                if haskey(p.records, :fire)
                    rate = mean(p.fire)
                    firing_rates[p.name] += rate / p.N - firing_rates[p.name]/τ_rate
                end
            end
            set_multiline_postfix(iter, join(vcat(
                        ["$(name) rate =  $(round(mean(firing_rates[name])*s*dt *τ_rate, digits=2))Hz\n"
                        for name in keys(firing_rates)],
                        "Time = $(round(get_time(time)/s, digits=3))s")
                        ))
        end
        if !isnothing(perturbation!) 
            perturbation!(;P, C, S, t=t, dt=dt, start_time = start_time)
        end
        train!(P, C, S, dt, time)
    end
    return time
end

function _args_model(args, model)
    pop = Vector{AbstractPopulation}([])
    syn = Vector{AbstractConnection}([])
    stim = Vector{AbstractStimulus}([])
    haskey(model, :pop) && append!(pop, model.pop)
    haskey(model, :syn) && append!(syn, model.syn)
    if haskey(model, :stim) 
        for s in model.stim 
            isa(s, AbstractStimulus) && push!(stim, s)
            isa(s, AbstractStimulusGroup) && append!(stim, s.elements)
        end
    end

    for arg in args
        if typeof(arg) <: AbstractPopulation
            push!(pop, arg)
        elseif typeof(arg) <: AbstractConnection
            push!(syn, arg)
        elseif typeof(arg) <: AbstractStimulus
            push!(stim, arg)

        elseif typeof(arg) <: Vector{AbstractPopulation}
            append!(pop, arg)
        elseif typeof(arg) <: Vector{AbstractConnection}
            append!(syn, arg)
        elseif typeof(arg) <: Vector{AbstractStimulus}
            append!(stim, arg)
        else
            error("Invalid argument type: $(typeof(arg))")
        end
    end
    return pop, syn, stim
end

function train!(args...; model = (time = Time(), name = "Model"), kwargs...)
    pop, syn, stim = _args_model(args, model)
    mytime = train!(pop, syn, stim; time = model.time, kwargs...)
    update_time!(model.time, mytime)
    return get_time(model.time)
end

train!(model::NamedTuple, duration::R = 1s; kwargs...) where {R<:Real} =
    train!(; model, duration, kwargs...)


function train!(
    P::Vector{TP},
    C::Vector{TC},
    S::Vector{TS},
    dt::Float32,
    T::Time,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    record_zero!(P, C, S, T)
    update_time!(T, dt)
    for s in S
        stimulate!(s, getfield(s, :param), T, dt)
        record!(s, T)
    end
    for p in P
        update_traces!(p, p.param, dt, T)
        integrate!(p, p.param, dt)
        plasticity!(p, p.param, dt, T)
        record!(p, T)
    end
    for c in C
        update_traces!(c, c.param, dt, T)
        forward!(c, c.param, dt, T)
        plasticity!(c, c.param, dt, T)
        record!(c, T)
    end
    # flush(stdout)  # removed from hot path: flushed every dt, ~10k calls/s
end

function record_zero!(P, C, S, T)
    get_time(T) > 0.0f0 && return
    for p in P
        record!(p, T)
    end
    for c in C
        record!(c, T)
    end
    for s in S
        record!(s, T)
    end
end
##

"""
    sim!(P::Vector{<:AbstractPopulation}, C::Vector{<:AbstractConnection} = [EmptySynapse()],
         S::Vector{<:AbstractStimulus} = [EmptyStimulus()];
         dt = 0.125f0, duration = 10.0f0, pbar = false, time = Time(), perturbation! = nothing)
    sim!(; model, duration = 10ms, dt = 0.125ms, pbar = false, perturbation! = nothing)
    sim!(model::NamedTuple, duration = 1s; kwargs...)
    sim!(args...; model, kwargs...)

Simulate a network without plasticity for `duration` (ms) with time step `dt` (ms).

Each step runs:
1. `update_time!` (the clock advances by `dt` first);
2. for every stimulus: `stimulate!(s, s.param, T, dt)`, `record!`;
3. for every population: `integrate!(p, p.param, dt)`, `record!`;
4. for every connection: `forward!(c, c.param, dt, T)`, `record!`.

No plasticity is applied: `sim!` never calls `update_traces!` or `plasticity!`, so weights
and short-term plasticity variables stay constant even if a synapse carries an `LTPParam`
or `STPParam`. Use [`train!`](@ref) for plastic networks.

# Arguments
- `model`: a NamedTuple built with `compose`; its populations, connections and stimuli
  (stimulus groups are unpacked) are simulated and `model.time` is advanced.
- `args...`: extra components (or vectors of them) added to those of `model`.
- `dt = 0.125` ms: time step (converted to `Float32`).
- `duration = 10` ms (vector and keyword forms) or `1s` (positional model form): the loop runs over
  `0:dt:(duration - dt)`.
- `time = Time()`: clock of the vector form.
- `perturbation!`: optional function called before every step as
  `perturbation!(; P, C, S, t, dt, start_time)`.
- `pbar = false`: show a progress bar with the running firing rates of the populations that
  record `:fire`.

Recording buffers are sized before the loop (see `monitor!`). If the model time is 0, a first
sample of every record is taken before the first step. Simulations can be continued by
calling `sim!` again: time and recordings continue from where they stopped.

# Returns
The `Time` object (vector form) or the model time in ms after the run (model forms).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
stim = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
model = SNN.compose(; E, stim)
SNN.monitor!(E, :fire)
SNN.sim!(; model, duration = 1s)
SNN.sim!(model, 500ms)          # continue for 500 ms
SNN.get_time(model.time)        # 1500.0
```
"""
function sim!(
    P::Vector{TP},
    C::Vector{TC} = [EmptySynapse()],
    S::Vector{TS} = [EmptyStimulus()];
    dt = 0.125f0,
    duration = 10.0f0,
    pbar = false,
    time = Time(),
    perturbation! = nothing,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    dt = Float32(dt)
    duration = Float32(duration)
    _allocate_records!(P, C, S, dt, duration)
    dts = 0.0f0:dt:(duration-dt)
    iter = pbar ? ProgressBar(dts, printing_delay=0.1) : dts
    start_time = get_time(time)
    firing_rates = Dict{String, Float32}(p.name => 0.f0 for p in P if haskey(p.records, :fire))
    τ_rate = 100.0f0
    for t in iter
        if pbar
            map(P) do p
                if haskey(p.records, :fire)
                    rate = mean(p.fire)
                    firing_rates[p.name] += rate / p.N - firing_rates[p.name]/100ms
                end
            end
            set_multiline_postfix(iter, join(vcat(
                        ["$(name) rate =  $(round(mean(firing_rates[name])*s*dt *τ_rate, digits=2))Hz\n"
                        for name in keys(firing_rates)],
                        "Time = $(round(get_time(time)/s, digits=3))s")
                        ))
        end
        if !isnothing(perturbation!)
            perturbation!(; P, C, S, t=t, dt=dt, start_time=start_time)
        end
        sim!(P, C, S, dt, time)
    end
    return time
end




sim!(model::NamedTuple, duration::R = 1s; kwargs...) where {R<:Real} =
    sim!(; model, duration, kwargs...)

function sim!(args...; model = (time = Time(), name = "Model"), kwargs...)
    pop, syn, stim = _args_model(args, model)
    mytime = sim!(pop, syn, stim; time = model.time, kwargs...)
    update_time!(model.time, mytime)
    return get_time(model.time)
    # sim!(collect(model.pop), collect(model.syn), collect(model.stim); kwargs...)
end


function sim!(
    P::Vector{TP},
    C::Vector{TC},
    S::Vector{TS},
    dt::Float32,
    T::Time,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    record_zero!(P, C, S, T)
    update_time!(T, dt)
    for s in S
        stimulate!(s, getfield(s, :param), T, dt)
        record!(s, T)
    end
    for p in P
        integrate!(p, getfield(p, :param), dt)
        record!(p, T)
    end
    for c in C
        forward!(c, getfield(c, :param), dt, T)
        record!(c, T)
    end
    # flush(stdout)
end


function initialize!(; kwargs...)
    train!(; kwargs...)
end

#########

export sim!, train!
