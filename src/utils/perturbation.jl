"""
    perturbation_test(model, simtime, condition!; from_state, add_records, train, kwargs...)

Run a perturbed copy of `model` for `simtime` ms and return or archive results.

The copy is created with `modelcopy` (deep copy with empty record buffers), `condition!` is
applied in-place, then the simulation is run. The original `model` is never mutated.

# Arguments
- `model`: source model (archive). Never modified.
- `simtime`: duration in ms, e.g. `500ms`.
- `condition!`: `(pert_model) -> nothing` applied to the copy before sim. Examples:
  ```julia
  m -> set_active!(m.stim.noise, false)   # silence a stimulus
  m -> (m.pop.pv.I .= 200f0pA)           # inject current
  m -> set_LTP!(m.syn.exc_exc, false)     # freeze plasticity
  ```

# Keyword arguments
- `from_state` (`nothing`): pre-built model to use instead of `modelcopy(model)`.
  The caller is responsible for empty records and correct starting state.
- `add_records` (`nothing`): if `nothing`, returns the perturbed model.
  If a `String`/`Symbol`, flushes all monitored variables into
  `model.<component>.records[:perturbation][variable][condition][n]`
  (n auto-increments) and returns `nothing`.
- `train` (`true`): if `true` runs `train!` (plasticity active); if `false` runs `sim!`.
- Remaining `kwargs` forwarded to `train!`/`sim!`.

# Storage schema (when `add_records` is set)

```
obj.records[:perturbation][variable::Symbol][condition::String][n::Int]
    = Dict("interval" => Float32[tstart, tend],   # absolute sim time (ms)
           "data"     => Matrix{Float32}           # (N_neurons, T) for dense vars
                       | Spiketimes)               # for :fire
```

Retrieve with `perturbation_record(obj, variable, condition, interval)`.

# Example
```julia
monitor!(model.pop.exc, [:v_s, :fire]; sr = 1kHz)

ck = modelcopy(model)   # deep copy: same state, empty record buffers

perturbation_test(ck, 500ms,
    m -> set_active!(m.stim.noise, false); add_records = "no_noise")
perturbation_test(ck, 500ms,
    m -> set_active!(m.stim.noise, false); add_records = "no_noise") # n=2

sim!(ck, 500ms)   # baseline over same window

v_patched, r = perturbation_record(ck.pop.exc, :v_s, "no_noise", 0:1ms:500ms)
```
"""
function perturbation_test(
    model,
    simtime,
    condition!;
    from_state = nothing,
    add_records = nothing,
    train = true,
    kwargs...,
)
    # If from_state is provided the caller has already prepared an empty/clean model
    # at the desired starting state — use it directly without copying.
    # Otherwise fall back to modelcopy (deep copy with empty record buffers).
    pert_model = isnothing(from_state) ? modelcopy(model) : from_state
    condition!(pert_model)
    if train
        train!(pert_model, simtime; kwargs...)
    else
        sim!(pert_model, simtime; kwargs...)
    end
    if isnothing(add_records)
        return pert_model
    else
        _flush_perturbation!(model, pert_model, String(add_records))
        return nothing
    end
end

# Walk model and pert_model in lockstep and flush each component pair.
function _flush_perturbation!(model, pert_model, condition::String)
    for field in (:pop, :syn, :stim)
        haskey(model, field) || continue
        haskey(pert_model, field) || continue
        orig_group = getfield(model, field)
        pert_group = getfield(pert_model, field)
        for key in keys(orig_group)
            haskey(pert_group, key) || continue
            _flush_component!(getfield(orig_group, key), getfield(pert_group, key), condition)
        end
    end
end

function _flush_component!(orig_obj, pert_obj, condition::String)
    haskey(pert_obj.records, :data) || return
    isempty(pert_obj.records[:data]) && return
    haskey(orig_obj.records, :perturbation) ||
        (orig_obj.records[:perturbation] = Dict{Symbol,Any}())
    pert_store = orig_obj.records[:perturbation]

    for key in pert_obj.records[:data]
        tstart = get(pert_obj.records[:start_time], key, NaN32)
        tend   = get(pert_obj.records[:end_time],   key, NaN32)
        (isnan(tstart) || isnan(tend)) && continue

        data = if key === :fire
            spiketimes(pert_obj)                  # Spiketimes
        else
            copy(Float32.(getvariable(pert_obj, key)))   # Matrix{Float32}
        end

        haskey(pert_store, key)            || (pert_store[key]           = Dict{String,Any}())
        haskey(pert_store[key], condition) || (pert_store[key][condition] = Dict{Int,Any}())
        n = length(pert_store[key][condition]) + 1
        pert_store[key][condition][n] = Dict{String,Any}(
            "interval" => Float32[tstart, tend],
            "data"     => data,
        )
    end
end

"""
    perturbation_record(obj, variable, condition, interval) -> (data, interval)

Reconstruct a variable trace over `interval` by splicing stored perturbation
windows into the baseline recording.

The baseline is retrieved from `obj`'s normal records, aligned to `interval` via
interpolation. Each stored entry for `condition` is then overlaid at its recorded
`[tstart, tend]` window (in chronological order). The result is a continuous
trace where perturbed segments replace the baseline.

# Arguments
- `obj`: any `AbstractPopulation`, `AbstractStimulus`, or `AbstractConnection`
  with a `:perturbation` record populated by `perturbation_test`.
- `variable`: `Symbol` of a monitored field (e.g. `:v_s`, `:fire`, `:ge`).
- `condition`: the `add_records` name string used in `perturbation_test`.
- `interval`: `AbstractRange` in ms, e.g. `0:1ms:2s`.

# Returns
- Continuous variables: `(Matrix{Float32} of shape (N_neurons, T), interval)`.
  `T = length(clipped interval)` where the interval is clipped to the recorded
  baseline range.
- `:fire`: `(Spiketimes, interval)` — baseline spikes in each perturbation window
  removed and replaced by the recorded perturbed spike times.

Returns unmodified baseline when no perturbation data exists for the condition.

# Interval alignment detail

`interpolated_record(obj, variable)` produces a `ScaledInterpolation` over the
full baseline `[start_time, end_time]`. Evaluating it on `eval_r` (the requested
`interval` clipped to the recorded range) yields an exactly time-aligned
`(N, T)` Float32 matrix regardless of the original sampling rate. Each
perturbation entry's `[tstart, tend]` is then mapped to column indices in this
grid via `searchsortedfirst`/`searchsortedlast` on `eval_r`.
"""
function perturbation_record(
    obj,
    variable::Symbol,
    condition::String,
    interval::AbstractRange,
)
    if variable === :fire
        return _perturbation_record_fire(obj, condition, interval)
    else
        return _perturbation_record_dense(obj, variable, condition, interval)
    end
end

function _perturbation_record_dense(obj, variable::Symbol, condition::String, interval::AbstractRange)
    # Retrieve baseline aligned to interval via interpolation.
    # record(..., interpolate=false) returns the raw buffer ignoring interval;
    # interpolated_record gives a ScaledInterpolation over [start_time, end_time]
    # which we evaluate at the (clipped) requested interval.
    v_interp, r_full = interpolated_record(obj, variable)
    t_lo   = max(Float32(first(interval)), Float32(first(r_full)))
    t_hi   = min(Float32(last(interval)),  Float32(last(r_full)))
    eval_r = range(t_lo, t_hi; step = Float32(step(interval)))
    N      = size(v_interp, 1)
    result = Float32.(v_interp(1:N, eval_r))   # (N, T) — owns its data

    pert       = get(obj.records, :perturbation, nothing)
    isnothing(pert) && return result, eval_r
    var_store  = get(pert, variable, nothing)
    isnothing(var_store) && return result, eval_r
    cond_store = get(var_store, condition, nothing)
    isnothing(cond_store) && return result, eval_r

    eval_r_vec = collect(eval_r)
    for (_, entry) in sort(collect(cond_store); by = x -> x[1])   # chronological
        tstart, tend = entry["interval"]
        pert_data    = entry["data"]::Matrix{Float32}
        t_lo_idx = searchsortedfirst(eval_r_vec, Float32(tstart))
        t_hi_idx = searchsortedlast(eval_r_vec,  Float32(tend))
        n_win    = t_hi_idx - t_lo_idx + 1
        n_win > 0 || continue
        T_pert = size(pert_data, 2)
        pert_t = range(Float32(tstart), Float32(tend); length = T_pert)
        ax     = map(1:(ndims(pert_data)-1)) do i; axes(pert_data, i); end
        itp    = scale(interpolate(pert_data, get_interpolator(pert_data)), ax..., pert_t)
        result[:, t_lo_idx:t_hi_idx] .= itp(ax..., eval_r[t_lo_idx:t_hi_idx])
    end
    return result, eval_r
end

function _perturbation_record_fire(obj, condition::String, interval::AbstractRange)
    result = deepcopy(spiketimes(obj; interval))

    pert       = get(obj.records, :perturbation, nothing)
    isnothing(pert) && return result, interval
    var_store  = get(pert, :fire, nothing)
    isnothing(var_store) && return result, interval
    cond_store = get(var_store, condition, nothing)
    isnothing(cond_store) && return result, interval

    for (_, entry) in sort(collect(cond_store); by = x -> x[1])
        tstart, tend = entry["interval"]
        pert_st = entry["data"]::Spiketimes
        @inbounds for n in eachindex(result)
            filter!(t -> !(t > tstart && t < tend), result[n])
            for t in pert_st[n]
                t > tstart && t < tend && push!(result[n], t)
            end
            sort!(result[n])
        end
    end
    return result, interval
end

"""
    clear_perturbation_records!(obj)
    clear_perturbation_records!(obj, condition)
    clear_perturbation_records!(obj, condition; variable, n)

Delete stored perturbation data from `obj`.

- No `condition`: removes **all** perturbation data (all variables, all conditions).
- With `condition`: removes all entries for that condition across all variables.
- With `variable` kwarg: restricts deletion to that variable only.
- With both `condition` and `n` kwarg: removes only the `n`-th recording for that
  condition (and variable if specified).

Accepts a single `AbstractComponent`, a `NamedTuple` of components (e.g.
`model.pop`), or a full model `NamedTuple`.
"""
function clear_perturbation_records!(obj; variable = nothing, n = nothing)
    if obj isa AbstractPopulation || obj isa AbstractStimulus || obj isa AbstractConnection
        _clear_pert_component!(obj, nothing; variable, n)
    elseif obj isa NamedTuple && haskey(obj, :pop)
        for field in (:pop, :syn, :stim)
            for v in values(getfield(obj, field))
                _clear_pert_component!(v, nothing; variable, n)
            end
        end
    else
        for v in values(obj)
            v isa AbstractPopulation || v isa AbstractStimulus || v isa AbstractConnection || continue
            _clear_pert_component!(v, nothing; variable, n)
        end
    end
end

function clear_perturbation_records!(obj, condition::String; variable = nothing, n = nothing)
    if obj isa AbstractPopulation || obj isa AbstractStimulus || obj isa AbstractConnection
        _clear_pert_component!(obj, condition; variable, n)
    elseif obj isa NamedTuple && haskey(obj, :pop)
        for field in (:pop, :syn, :stim)
            for v in values(getfield(obj, field))
                _clear_pert_component!(v, condition; variable, n)
            end
        end
    else
        for v in values(obj)
            v isa AbstractPopulation || v isa AbstractStimulus || v isa AbstractConnection || continue
            _clear_pert_component!(v, condition; variable, n)
        end
    end
end

function _clear_pert_component!(obj, condition; variable, n)
    pert = get(obj.records, :perturbation, nothing)
    isnothing(pert) && return

    target_vars = isnothing(variable) ? collect(keys(pert)) : [Symbol(variable)]

    for var in target_vars
        haskey(pert, var) || continue
        if isnothing(condition)
            delete!(pert, var)
        else
            cond_store = get(pert[var], condition, nothing)
            isnothing(cond_store) && continue
            if isnothing(n)
                delete!(pert[var], condition)
            else
                delete!(cond_store, n)
                isempty(cond_store) && delete!(pert[var], condition)
            end
            isempty(pert[var]) && delete!(pert, var)
        end
    end
    isempty(pert) && delete!(obj.records, :perturbation)
end

"""
    clear_perturbation_monitor!(obj, variable)
    clear_perturbation_monitor!(obj, variable, condition)
    clear_perturbation_monitor!(obj, variable, condition; n)

Remove perturbation records for a specific `variable` (required), optionally
filtered by `condition` and `n`.

This is the targeted complement to `clear_perturbation_records!`:

| Call | Effect |
|:---|:---|
| `clear_perturbation_monitor!(obj, :v_s)` | remove all conditions for `:v_s` |
| `clear_perturbation_monitor!(obj, :v_s, "silence")` | remove all `n` under `"silence"` for `:v_s` |
| `clear_perturbation_monitor!(obj, :v_s, "silence"; n=2)` | remove only the 2nd recording |
"""
function clear_perturbation_monitor!(obj, variable::Symbol)
    clear_perturbation_records!(obj; variable)
end

function clear_perturbation_monitor!(obj, variable::Symbol, condition::String; n = nothing)
    clear_perturbation_records!(obj, condition; variable, n)
end

export perturbation_test,
    perturbation_record,
    clear_perturbation_records!,
    clear_perturbation_monitor!
