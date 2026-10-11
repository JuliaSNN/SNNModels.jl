
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
- `neurons_per_thread = nothing`: chunk size of the threaded population updates for the duration of the run (`nothing`: keep
  the current setting, default 0: threading off; a positive value such as 512 turns it on, `0` or `Inf` off). See [`neurons_per_thread`](@ref) and [`foreach_chunk`](@ref).
- `flush_subnormals = true`: run with the flush-to-zero floating-point mode (`set_zero_subnormals(true)` on the stepping thread and,
  when threaded, on the worker threads; restored afterwards). Results that would be subnormal (below `floatmin`, 1.2e-38 in
  Float32) are 0, so decaying variables of silent synapses and quiet populations stop at a tiny normal value instead of
  staying subnormal, which avoids the slow subnormal arithmetic (up to 60x per step); spikes and states above `floatmin` are
  unchanged. `false` keeps IEEE gradual underflow.
  See `SNNModels.with_flush_subnormals`.

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
    time = Time{_model_float(P)}(),
    perturbation! = nothing,
    pbar = false,
    neurons_per_thread = nothing,
    flush_subnormals::Bool = true,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    FT = eltype(time.t)
    dt = FT(dt)
    _allocate_records!(P, C, S, Float32(dt), Float32(duration))
    dts = zero(FT):dt:(duration-dt)
    iter = pbar ? ProgressBar(dts, printing_delay=0.1) : dts
    start_time = get_time(time)
    firing_rates = Dict{String, Float32}(p.name => 0.f0 for p in P if haskey(p.records, :fire))
    τ_rate = 100.0f0
    with_neurons_per_thread(neurons_per_thread) do
    with_flush_subnormals(flush_subnormals) do
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
    end
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
    dt::AbstractFloat,
    T::Time,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    record_zero!(P, C, S, T)
    update_time!(T, dt)
    # every component gets `dt` in its own precision (float_type): a Float32 population in a Float64 model runs in Float32
    for s in S
        dts = _step_dt(s, dt)
        stimulate!(s, getfield(s, :param), T, dts)
        record!(s, T)
    end
    for p in P
        dtp = _step_dt(p, dt)
        update_traces!(p, p.param, dtp, T)
        integrate!(p, p.param, dtp)
        plasticity!(p, p.param, dtp, T)
        record!(p, T)
    end
    for c in C
        dtc = _step_dt(c, dt)
        prepare_spikes!(c)
        update_traces!(c, c.param, dtc, T)
        forward!(c, c.param, dtc, T)
        plasticity!(c, c.param, dtc, T)
        record!(c, T)
        release_spikes!(c)
    end
    # flush(stdout)  # removed from hot path: flushed every dt, ~10k calls/s
end

# `dt` in the precision of the component (a no-op when they agree)
@inline _step_dt(c, dt::AbstractFloat) = convert(float_type(typeof(c)), dt)

@inline function record_zero!(P, C, S, T)
    get_time(T) > 0 && return nothing
    return _record_zero!(P, C, S, T)
end

function _record_zero!(P, C, S, T)
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
- `neurons_per_thread = nothing`: chunk size of the threaded population updates for the duration of the run (`nothing`: keep
  the current setting, default 0: threading off; a positive value such as 512 turns it on, `0` or `Inf` off). See [`neurons_per_thread`](@ref) and [`foreach_chunk`](@ref).
- `flush_subnormals = true`: run with the flush-to-zero floating-point mode (`set_zero_subnormals(true)` on the stepping thread and,
  when threaded, on the worker threads; restored afterwards). Results that would be subnormal (below `floatmin`, 1.2e-38 in
  Float32) are 0, so decaying variables of silent synapses and quiet populations stop at a tiny normal value instead of
  staying subnormal, which avoids the slow subnormal arithmetic (up to 60x per step); spikes and states above `floatmin` are
  unchanged. `false` keeps IEEE gradual underflow.
  See `SNNModels.with_flush_subnormals`.

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
    time = Time{_model_float(P)}(),
    perturbation! = nothing,
    neurons_per_thread = nothing,
    flush_subnormals::Bool = true,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    FT = eltype(time.t)
    dt = FT(dt)
    duration = FT(duration)
    _allocate_records!(P, C, S, Float32(dt), Float32(duration))
    dts = zero(FT):dt:(duration-dt)
    iter = pbar ? ProgressBar(dts, printing_delay=0.1) : dts
    start_time = get_time(time)
    firing_rates = Dict{String, Float32}(p.name => 0.f0 for p in P if haskey(p.records, :fire))
    τ_rate = 100.0f0
    with_neurons_per_thread(neurons_per_thread) do
    with_flush_subnormals(flush_subnormals) do
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
    end
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
    dt::AbstractFloat,
    T::Time,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    record_zero!(P, C, S, T)
    update_time!(T, dt)
    for s in S
        stimulate!(s, getfield(s, :param), T, _step_dt(s, dt))
        record!(s, T)
    end
    for p in P
        integrate!(p, getfield(p, :param), _step_dt(p, dt))
        record!(p, T)
    end
    for c in C
        prepare_spikes!(c)
        forward!(c, getfield(c, :param), _step_dt(c, dt), T)
        record!(c, T)
        release_spikes!(c)
    end
    # flush(stdout)
end


function initialize!(; kwargs...)
    train!(; kwargs...)
end

#########

export sim!, train!
