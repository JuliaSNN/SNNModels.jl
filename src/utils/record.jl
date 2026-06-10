import Interpolations: scale, interpolate, BSpline, Linear, NoInterp
"""
    get_time(T::Time)

Get the current time.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The current time.

"""
get_time(T::Time)::Float32 = T.t[1]

get_time(model::NamedTuple)::Float32 = model.time.t[1]

"""
    get_step(T::Time)

Get the current simulation step counter (integer number of dt steps elapsed).

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The step counter `T.tt[1]`, cast to Float32.

"""
get_step(T::Time)::Float32 = T.tt[1]

"""
    get_dt(T::Time)

Get the time step size.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The time step size.

"""
get_dt(T::Time)::Float32 = T.dt

"""
    get_interval(T::Time)

Get the time interval from 0 to the current time.

# Arguments
- `T::Time`: The Time object.

# Returns
- `StepRange{Float32}`: The time interval.

"""
get_interval(T::Time) = Float32(T.dt):Float32(T.dt):get_time(T)

"""
    update_time!(T::Time, dt::Float32)

Update the current time and time step.

# Arguments
- `T::Time`: The Time object.
- `dt::Float32`: The time step size.

"""
function update_time!(T::Time, dt::Float32)
    T.t[1] += dt
    T.tt[1] += 1
end

function update_time!(T::Time, myT::Time)
    T.t[1] = myT.t[1]
    T.tt[1] = myT.tt[1]
end

function reset_time!(T::Time)
    T.t[1] = 0.0f0
    T.tt[1] = 0
end

function reset_time!(model::NamedTuple)
    model.time.t[1] = 0.0f0
    model.time.tt[1] = 0
end

"""
    record_fire!(fire, fire_rec, T, indices)

Legacy fire recorder. Appends the current timestep's spike set to the jagged
`:time` / `:neurons` vectors in `fire_rec`. Used when `meta[:mode][:fire] == :legacy`.
The dense path uses `record_fire_dense!` instead, which writes into pre-allocated
COO buffers without allocating.
"""
function record_fire!(
    fire::Vector{Bool},
    record::Dict{Symbol,AbstractVector},
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
)
    # @unpack fire = obj
    # @unpack records = obj
    sum(fire) == 0 && return
    ind::Vector{Int} = haskey(indices, :fire) ? indices[:fire] : collect(eachindex(fire))
    t::Float32 = get_time(T)
    push!(record[:time], t)
    push!(record[:neurons], findall(fire[ind]))
end

"""
    record_sym!(my_record, obj, key, T, indices, sr)

Legacy variable recorder. Appends a snapshot of `my_record` to `obj.records[key]`
at the rate set by `sr`. Gated by `record_step`. Used when `meta[:mode][key] == :legacy`.
The dense path uses `record_sym_dense!` instead.
"""
function record_sym!(
    my_record,
    obj,
    key::Symbol,
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
    sr::Float32,
)
    !record_step(T, sr) && return
    ind::Vector{Int} = haskey(indices, key) ? indices[key] : axes(my_record, 1)
    @inbounds _record_sym(my_record, obj.records[key], ind)
end

@inline function _record_sym(
    my_record::Vector{T},
    records::Vector{Vector{T}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind])
end

@inline function _record_sym(
    my_record::T,
    records::Vector{T},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record)
end

@inline function _record_sym(
    my_record::Array{T,3},
    records::Vector{Array{T,3}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind, :, :])
end

@inline function _record_sym(
    my_record::Vector{Vector{T}},
    records::Vector{Vector{Vector{T}}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, deepcopy(my_record[ind]))
end

@inline function _record_sym(
    my_record::Matrix{T},
    records::Vector{Matrix{T}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind, :])
end

@inline function record_step(T, sr)
    period = max(1, floor(Int, 1.0f0 / sr / get_dt(T)))
    (get_step(T) % period) == 0
end

# ── Dense write path (manifesto §4) ─────────────────────────────────────────
# `snapshot` is the live field value (scalar / Vector / Matrix / Array{3}). The
# write column is derived from the `allocated` countdown. Phase is driven off the
# global step `get_step(T)` to be bit-identical to the legacy `record_step` gate.
function record_sym_dense!(snapshot, records, meta, key::Symbol, T::Time)
    period = meta[:period][key]
    (get_step(T) % period) == 0 || return
    if get(meta[:allocated], key, 0) == 0
        _grow_dense!(records, meta, key)
    end
    buffer = records[key]
    nd = ndims(buffer)
    wp = size(buffer, nd) - meta[:allocated][key] + 1
    @inbounds selectdim(buffer, nd, wp) .= snapshot
    meta[:allocated][key] -= 1
    return
end

# Overflow: double the time capacity, copy written slices, keep remaining count.
function _grow_dense!(records, meta, key::Symbol)
    buffer = records[key]
    nd = ndims(buffer)
    old_cap = size(buffer, nd)
    write_ptr = old_cap  # allocated == 0 ⇒ all slots written
    new_cap = max(1, 2 * old_cap)
    snap_size = size(buffer)[1:(nd-1)]
    new_buffer = Array{Float32}(undef, snap_size..., new_cap)
    selectdim(new_buffer, nd, 1:write_ptr) .= selectdim(buffer, nd, 1:write_ptr)
    records[key] = new_buffer
    meta[:allocated][key] = new_cap - write_ptr
    if !get(meta[:grew], key, false)
        @warn "Dense recording buffer for $key overflowed; doubled to $new_cap slots. Pass monitor_time= to avoid reallocation."
        meta[:grew][key] = true
    end
    return
end

# ── Dense fire write (manifesto §5) ─────────────────────────────────────────
function record_fire_dense!(
    fire::Vector{Bool},
    records,
    meta,
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
)
    rec = records[:fire]
    times_buf = rec[:times_buf]::Vector{Float32}
    neurons_buf = rec[:neurons_buf]::Vector{Int}
    t = get_time(T)
    has_ind = haskey(indices, :fire)
    ind = has_ind ? indices[:fire] : Int[]
    rng = has_ind ? eachindex(ind) : eachindex(fire)
    @inbounds for j in rng
        i = has_ind ? ind[j] : j
        fire[i] || continue
        if meta[:allocated][:fire] == 0
            _grow_fire!(rec, meta)
            times_buf = rec[:times_buf]::Vector{Float32}
            neurons_buf = rec[:neurons_buf]::Vector{Int}
        end
        wp = length(times_buf) - meta[:allocated][:fire] + 1
        times_buf[wp] = t
        neurons_buf[wp] = has_ind ? j : i
        meta[:allocated][:fire] -= 1
    end
    return
end

function _grow_fire!(rec, meta)
    old_cap = length(rec[:times_buf])
    new_cap = max(1, 2 * old_cap)
    new_times = Vector{Float32}(undef, new_cap)
    new_neurons = Vector{Int}(undef, new_cap)
    copyto!(new_times, 1, rec[:times_buf], 1, old_cap)
    copyto!(new_neurons, 1, rec[:neurons_buf], 1, old_cap)
    rec[:times_buf] = new_times
    rec[:neurons_buf] = new_neurons
    meta[:allocated][:fire] = new_cap - old_cap
    if !get(meta[:grew], :fire, false)
        @warn "Fire COO buffer overflowed; doubled to $new_cap spike slots. Pass monitor_rate= to avoid reallocation."
        meta[:grew][:fire] = true
    end
    return
end

@inline function get_model_field(obj, key::Symbol, var_map::Dict{Symbol,Tuple{Symbol,Symbol}})
    if haskey(var_map, key)
        parent, sub = var_map[key]
        return getfield(getfield(obj, parent), sub)
    else
        return getfield(obj, key)
    end
end

"""
    _allocate_records!(P, C, S, dt::Float32, duration::Float32)

Pre-allocate dense recording buffers and the flat `:fire` COO event list for every
monitored component. Idempotent across chunked `sim!` calls: a key already allocated
at the same `dt` is skipped; on `dt` mismatch the buffer is reallocated.
"""
function _allocate_records!(P, C, S, dt::Float32, duration::Float32)
    for obj in Iterators.flatten((P, C, S))
        (obj isa AbstractPopulation || obj isa AbstractConnection || obj isa AbstractStimulus) || continue
        _allocate_component!(obj, dt, duration)
    end
end

function _allocate_component!(obj, dt::Float32, duration::Float32)
    haskey(obj.records, :data) || return
    haskey(obj.records, :meta) || return
    meta = obj.records[:meta]
    haskey(meta, :mode) || return
    N = hasfield(typeof(obj), :N) ? Int(getfield(obj, :N)) : 0
    for key in obj.records[:data]
        get(meta[:mode], key, :legacy) === :dense || continue
        if key === :fire
            _allocate_fire!(obj, meta, N, duration)
            continue
        end
        # Reconcile period against the real dt now that it is known.
        sr = obj.records[:sr][key]
        period = max(1, floor(Int, 1.0f0 / Float32(sr) / dt))
        meta[:period][key] = period

        snap_size = meta[:snapshot_size][key]
        buf = obj.records[key]
        allocated_buffer = buf isa AbstractArray && ndims(buf) == length(snap_size) + 1
        dt_changed = allocated_buffer && get(meta[:dt], key, dt) != dt

        # Samples a chunk of `duration` produces. The +1 (record_zero! sample at
        # T.tt = 0) only applies to the very first chunk on a fresh buffer.
        hint_ms = if haskey(meta[:monitor_time], key)
            meta[:monitor_time][key]
        elseif duration >= 1000f0
            duration
        else
            1000f0
        end
        zero_sample = (!allocated_buffer || dt_changed) ? 1 : 0
        chunk_capacity = floor(Int, floor(Int, hint_ms / dt) / period) + zero_sample

        if !allocated_buffer || dt_changed
            # First allocation (or dt mismatch → fresh buffer).
            dt_changed && @warn "dt changed for record $key in $(obj.name); reallocating buffer."
            obj.records[key] = Array{Float32}(undef, snap_size..., chunk_capacity)
            meta[:allocated][key] = chunk_capacity
            meta[:dt][key] = dt
        else
            # Existing buffer persists across chunks. Ensure room for this chunk;
            # extend in place if the remaining countdown is insufficient.
            remaining = get(meta[:allocated], key, 0)
            if remaining < chunk_capacity
                _extend_dense!(obj.records, meta, key, chunk_capacity - remaining)
            end
        end
    end
end

# Append `extra` time-slots to an existing dense buffer, preserving written data.
function _extend_dense!(records, meta, key::Symbol, extra::Int)
    buffer = records[key]
    nd = ndims(buffer)
    old_cap = size(buffer, nd)
    write_ptr = old_cap - get(meta[:allocated], key, 0)
    new_cap = old_cap + extra
    snap_size = size(buffer)[1:(nd-1)]
    new_buffer = Array{Float32}(undef, snap_size..., new_cap)
    write_ptr > 0 && (selectdim(new_buffer, nd, 1:write_ptr) .= selectdim(buffer, nd, 1:write_ptr))
    records[key] = new_buffer
    meta[:allocated][key] = new_cap - write_ptr
    return
end

function _allocate_fire!(obj, meta, N::Int, duration::Float32)
    fire = obj.records[:fire]
    rate = get(meta[:monitor_rate], :fire, 20f0)  # Hz; conservative default
    dur_s = duration / 1000f0
    chunk_cap = max(1, ceil(Int, rate * max(N, 1) * dur_s))
    allocated_buffer = haskey(fire, :times_buf) && !isempty(fire[:times_buf])
    if !allocated_buffer
        # First allocation.
        fire[:times_buf] = Vector{Float32}(undef, chunk_cap)
        fire[:neurons_buf] = Vector{Int}(undef, chunk_cap)
        meta[:allocated][:fire] = chunk_cap
    else
        # Extend across chunks if remaining room is insufficient for this chunk.
        remaining = get(meta[:allocated], :fire, 0)
        if remaining < chunk_cap
            _extend_fire!(fire, meta, chunk_cap - remaining)
        end
    end
end

function _extend_fire!(fire, meta, extra::Int)
    old_cap = length(fire[:times_buf])
    write_ptr = old_cap - get(meta[:allocated], :fire, 0)
    new_cap = old_cap + extra
    new_times = Vector{Float32}(undef, new_cap)
    new_neurons = Vector{Int}(undef, new_cap)
    write_ptr > 0 && copyto!(new_times, 1, fire[:times_buf], 1, write_ptr)
    write_ptr > 0 && copyto!(new_neurons, 1, fire[:neurons_buf], 1, write_ptr)
    fire[:times_buf] = new_times
    fire[:neurons_buf] = new_neurons
    meta[:allocated][:fire] = new_cap - write_ptr
    return
end

"""
    record!(obj, T::Time)

Record all monitored variables of `obj` at the current time `T`. Dispatches to
the dense or legacy path per `meta[:mode][key]`. Called every dt inside the sim loop.
"""
function record!(obj, T::Time)
    @unpack records = obj
    !haskey(records, :data) && return
    meta = records[:meta]
    time = get_time(T)
    mode = meta[:mode]
    for key::Symbol in records[:data]
        isnan(get(records[:start_time], key, NaN32)) && (records[:start_time][key] = time)
        records[:end_time][key] = time
        if key === :fire
            if get(mode, :fire, :legacy) === :dense
                record_fire_dense!(obj.fire, records, meta, T, records[:indices])
            else
                record_fire!(obj.fire, records[:fire], T, records[:indices])
            end
        elseif get(mode, key, :legacy) === :dense
            record_sym_dense!(
                get_model_field(obj, key, meta[:var_map]),
                records, meta, key, T,
            )
        else
            record_sym!(
                get_model_field(obj, key, meta[:var_map]),
                obj, key, T, records[:indices], records[:sr][key],
            )
        end
    end
end

"""
    monitor!(obj, keys; sr=1000Hz, variables=:none, monitor_time=nothing, monitor_rate=nothing)

Register variables on `obj` for recording during the next `sim!`/`train!` call.

Buffers are allocated lazily by `_allocate_records!` just before the time loop, not here.

# Arguments
- `keys`: Vector of `Symbol` or `(Symbol, indices)` tuples. A tuple restricts recording
  to the listed neuron indices (legacy path only).
- `sr`: Sampling rate (default 1000 Hz). Ignored for `:fire`.
- `variables`: Nested field group (e.g. `:STPVars`). If set, `keys` are looked up inside
  `obj.<variables>` and stored under compound keys like `STPVars_x`.
- `monitor_time`: Explicit capacity hint in ms. Overrides the `sim!` duration for
  buffer sizing. Useful when the total simulation time is unknown at monitor time.
- `monitor_rate`: Expected maximum firing rate in Hz, used to size the `:fire` COO
  buffers. Defaults to a conservative 20 Hz if not set.

# Recording modes
- **Dense** (`:v`, `:ge`, `:gi`, scalar/Vector/Matrix Float32): a contiguous
  `(snapshot_size..., capacity)` array written in-place via a countdown index.
  `getvariable` returns a zero-copy `selectdim` view over written columns.
- **COO event list** (`:fire`): two flat pre-allocated vectors `times_buf` and
  `neurons_buf`, one entry per spike. Read via `spiketimes(obj)`.
- **Legacy** (`push!`-based): ragged `Vector{Vector{Float32}}` for types that
  cannot be pre-allocated (e.g. indexed subsets, `Vector{Vector{Float32}}`).

# Notes
- Already-monitored keys are silently skipped (warning with `verbose=true`).
- Missing fields emit a warning with `verbose=true` and are skipped.
"""
# Keys created by _init_records!. _clear skips these (they are schema, not data).
# Add here when adding a new metadata key to _init_records!.
const _RECORD_META_KEYS = (:indices, :sr, :variables, :data, :meta)

function _init_records!(records::Dict)
    haskey(records, :indices)    || (records[:indices]    = Dict{Symbol,Vector{Int}}())
    haskey(records, :sr)         || (records[:sr]         = Dict{Symbol,Float32}())
    haskey(records, :variables)  || (records[:variables]  = Vector{Symbol}())
    haskey(records, :start_time) || (records[:start_time] = Dict{Symbol,Float32}())
    haskey(records, :end_time)   || (records[:end_time]   = Dict{Symbol,Float32}())
    haskey(records, :data)       || (records[:data]       = Symbol[])
    haskey(records, :meta)       || (records[:meta]       = Dict{Symbol,Any}(
        :var_map => Dict{Symbol,Tuple{Symbol,Symbol}}(),
    ))
    meta = records[:meta]
    # Dense-recording metadata sub-dicts (manifesto §2). Additive only.
    haskey(meta, :var_map)       || (meta[:var_map]       = Dict{Symbol,Tuple{Symbol,Symbol}}())
    haskey(meta, :mode)          || (meta[:mode]          = Dict{Symbol,Symbol}())
    haskey(meta, :snapshot_size) || (meta[:snapshot_size] = Dict{Symbol,Tuple}())
    haskey(meta, :allocated)     || (meta[:allocated]     = Dict{Symbol,Int}())
    haskey(meta, :monitor_time)  || (meta[:monitor_time]  = Dict{Symbol,Float32}())
    haskey(meta, :period)        || (meta[:period]        = Dict{Symbol,Int}())
    haskey(meta, :step_count)    || (meta[:step_count]    = Dict{Symbol,Int}())
    haskey(meta, :grew)          || (meta[:grew]          = Dict{Symbol,Bool}())
    haskey(meta, :dt)            || (meta[:dt]            = Dict{Symbol,Float32}())
    haskey(meta, :monitor_rate)  || (meta[:monitor_rate]  = Dict{Symbol,Float32}())
    return records
end

# Snapshot eligibility (manifesto §8). `sample` is a live value pulled from the
# object field at monitor! time. Returns (:dense, snapshot_size) or (:legacy, ()).
# Forced :legacy when: explicit :indices, Vector{Vector} (multipod, deferred per
# scope decision Q2), or any non-Float32-array snapshot type.
function _classify_record(sample, has_indices::Bool)
    has_indices && return (:legacy, ())
    if sample isa Float32
        return (:dense, ())
    elseif sample isa Vector{Float32}
        return (:dense, (length(sample),))
    elseif sample isa Matrix{Float32}
        return (:dense, size(sample))
    elseif sample isa Array{Float32,3}
        return (:dense, size(sample))
    else
        # Vector{Vector{Float32}} (ragged multipod) and everything else → legacy.
        return (:legacy, ())
    end
end

function monitor!(
    obj::Item,
    keys::Vector;
    sr = 1000Hz,
    variables::Symbol = :none,
    monitor_time = nothing,
    monitor_rate = nothing,
    verbose = false
) where {Item<:Union{AbstractPopulation,AbstractStimulus,AbstractConnection}}
    _init_records!(obj.records)
    meta = obj.records[:meta]
    dt_hint = 0.125f0  # only used to precompute :period; reconciled in _allocate_records!
    ## If the key is a tuple, then the first element is the symbol and the second element is the list of neurons to record.
    for key in keys
        sym, ind = isa(key, Tuple) ? key : (key, [])
        if sym == :fire
            if haskey(obj.records, :fire)
                verbose && @warn "Field :fire already being monitored in $(obj.name)"
                :fire ∉ obj.records[:data] && push!(obj.records[:data], :fire)
                continue
            end
            # COO fire event list: flat parallel arrays, pre-allocated in _allocate_records!.
            obj.records[:fire] = Dict{Symbol,AbstractVector}(
                :time => Vector{Float32}(),
                :neurons => Vector{Vector{Int}}(),
                :times_buf => Vector{Float32}(),
                :neurons_buf => Vector{Int}(),
            )
            obj.records[:start_time][:fire] = NaN32
            meta[:mode][:fire] = :dense
            meta[:allocated][:fire] = 0
            meta[:step_count][:fire] = 0
            meta[:grew][:fire] = false
            monitor_rate !== nothing && (meta[:monitor_rate][:fire] = Float32(monitor_rate))
            @debug "Monitoring :fire in $(obj.name)"
            :fire ∉ obj.records[:data] && push!(obj.records[:data], :fire)
            continue
        end
        if variables == :none
            if hasfield(typeof(obj), sym)
                typ = typeof(getfield(obj, sym))
                sample = getfield(obj, sym)
                key = sym
            else
                verbose && @warn "Field $sym not found in $(nameof(typeof(obj)))"
                continue
            end
        else
            if hasproperty(obj, variables)
                if hasproperty(getfield(obj, variables), sym)
                    typ = typeof(getfield(getfield(obj, variables), sym))
                    sample = getfield(getfield(obj, variables), sym)
                    key = Symbol(variables, "_", sym)
                    if !(variables ∈ obj.records[:variables])
                        @debug "Monitoring $(variables)"
                        push!(obj.records[:variables], variables)
                    end
                else
                    verbose && @warn "Field $sym not found in $(nameof(typeof(getfield(obj, variables))))"
                    continue
                end
            else
                verbose && @warn "Field $variables not found in $(nameof(typeof(obj)))"
                continue
            end
        end
        @debug "Monitoring :$(key) in $(obj.name)"

        if haskey(obj.records, key)
            verbose && @warn "Key $key already being monitored in $(obj.name)"
            continue
        end
        !isempty(ind) && (obj.records[:indices][key] = ind)
        obj.records[:sr][key] = sr
        push!(obj.records[:data], key)
        obj.records[:start_time][key] = NaN32
        if variables != :none
            meta[:var_map][key] = (variables, sym)
        end

        # Classify dense vs legacy (manifesto §8) and init meta.
        mode, snap_size = _classify_record(sample, !isempty(ind))
        meta[:mode][key] = mode
        meta[:period][key] = max(1, floor(Int, 1.0f0 / Float32(sr) / dt_hint))
        meta[:step_count][key] = 0
        meta[:grew][key] = false
        meta[:allocated][key] = 0
        monitor_time !== nothing && (meta[:monitor_time][key] = Float32(monitor_time))
        if mode === :dense
            meta[:snapshot_size][key] = snap_size
            # Buffer allocated lazily in _allocate_records!; placeholder for now.
            obj.records[key] = Vector{typ}()
        else
            obj.records[key] = Vector{typ}()
        end
    end
end


function monitor!(objs::Array, keys::Vector; sr = 200Hz, kwargs...)
    for obj in objs
        monitor!(obj, keys, sr = sr; kwargs...)
    end
end

function monitor!(objs::NamedTuple, keys::Vector; sr = 200Hz, kwargs...)
    for obj in values(objs)
        monitor!(obj, keys, sr = sr; kwargs...)
    end
end

monitor!(obj, keys::Symbol; kwargs...) = monitor!(obj, [keys]; kwargs...)

monitor!(obj, keys::Tuple; kwargs...) = monitor!(obj, [keys]; kwargs...)

monitor!(objs, keys, variables::Symbol; kwargs...) =
    monitor!(objs, keys; variables = variables, kwargs...)

"""
    interpolated_record(p, sym)

    Returns the recording with interpolated time values and the extrema of the recorded time points.

    N.B. 
    ----
    The element can be accessed at whichever time point by using the index of the array. The time point must be within the range of the recorded time points, in r_v.
"""
function interpolated_record(p, sym, τ = 20ms)
    if sym == :fire
        return firing_rate(p, τ = 20ms)
    end
    v_dt = getvariable(p, sym)

    # Set NoInterp in the singleton dimensions:
    interp = get_interpolator(v_dt)
    v = interpolate(v_dt, interp)
    # Last dimension is time
    ax = map(1:(length(size(v_dt))-1)) do i
        axes(v_dt, i)
    end

    r_v = get_measure_interval(p, sym, size(v_dt, ndims(v_dt)))
    y = scale(v, ax..., r_v)
    return y, r_v
end

function get_measure_interval(p::AbstractComponent, sym::Symbol, steps::Int)
    _start = p.records[:start_time][sym]
    _end = p.records[:end_time][sym]
    return range(_start, _end, steps)
end

function get_measure_interval(
    p::AbstractComponent,
    sym::Symbol,
    step::R,
) where {R<:Union{Float32,Float64}}
    _start = p.records[:start_time][sym]
    _end = p.records[:end_time][sym]
    return range(_start, _end, step = step)
end



function add_endtime!(model::NamedTuple)
    @assert isa_model(model) "Model is not a valid NetworkModel"
    time = model.time
    for obj in values(model)
        obj isa String && continue
        obj isa Time && continue
        for v in obj
            if v isa AbstractPopulation ||
               v isa AbstractStimulus ||
               v isa AbstractConnection
                # @info "Adding end time for $(v.name)"
                !haskey(v.records, :end_time) &&
                    (v.records[:end_time] = Dict{Symbol,Float32}())
                for key in get(v.records, :data, Symbol[])
                    if !haskey(v.records[:end_time], key)
                        v.records[:end_time][key] = get_time(time)
                    end
                end
            end
        end
    end
end

function add_starttime!(model::NamedTuple)
    @assert isa_model(model) "Model is not a valid NetworkModel"
    for obj in values(model)
        obj isa String && continue
        obj isa Time && continue
        for v in obj
            if v isa AbstractPopulation ||
               v isa AbstractStimulus ||
               v isa AbstractConnection
                # @info "Adding start time for $(v.name)"
                !haskey(v.records, :start_time) &&
                    (v.records[:start_time] = Dict{Symbol,Float32}())
                for key in get(v.records, :data, Symbol[])
                    if !haskey(v.records[:start_time], key)
                        v.records[:start_time][key] = 0
                    end
                end
            end
        end
    end
end


function squeeze(A::AbstractArray)
    singleton_dims = tuple((d for d = 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims = singleton_dims)
end

function get_interpolator(A::AbstractArray)
    singleton_dims = tuple((d for d = 1:ndims(A) if size(A, d) == 1)...)
    interp = repeat(Vector{Any}([BSpline(Linear())]), ndims(A))
    for d in singleton_dims
        interp[d] = NoInterp()
    end
    return Tuple(interp)
end

"""
    record(p, sym::Symbol; range = false, interval = nothing, kwargs...)

Record data from a population `p` based on the specified symbol `sym`.

# Arguments
- `p`: The population from which to record data.
- `sym::Symbol`: The type of data to record. Valid options are `:fire` for firing rate, `:spiketimes` or `:spikes` for spike times.
- `range::Bool=false`: If `true`, return both the recorded data and the range. Default is `false`.
- `interval`: The time interval for recording. Required for firing rate recording (`sym = :fire`).
- `kwargs...`: Additional keyword arguments to pass to the recording function.

# Returns
- If `sym = :fire` and `range = true`, returns a tuple `(v, r)` where `v` is the firing rate and `r` is the range.
- If `sym = :fire` and `range = false`, returns the firing rate `v`.
- If `sym = :spiketimes` or `sym = :spikes`, returns the spike times.
- For other symbols, returns a tuple `(v, r)` if `range = true`, or `v` if `range = false`.

# Examples
```julia
# Record firing rate for a population p over a specific interval
v = record(p, :fire; interval = (0.0, 1.0))

# Record firing rate and range for a population p over a specific interval
v, r = record(p, :fire; range = true, interval = (0.0, 1.0))

# Record spike times for a population p
spikes = record(p, :spiketimes)
```
"""
function record(
    p::C,
    sym::Symbol;
    range = false,
    interval = nothing,
    interpolate = true,
    variables=  nothing,
    kwargs...,
) where {C<:Component}
    if sym == :fire
        @assert !isnothing(interval) "Range must be provided for firing rate recording"
        v, r = firing_rate(p, interval; interpolate, kwargs...)
    elseif sym == :spiketimes || sym == :spikes
        return spiketimes(p)
    else
        sym = isnothing(variables) ? sym : Symbol(variables, "_", sym)
        # not interpolate
        if !interpolate
            v = getvariable(p, sym)
            r = interval
            # interpolate
        else
            v, r = interpolated_record(p, sym)
            if !isnothing(interval)
                @assert interval[1] .>= r[1] "Interval start $(interval[1]) is out of bounds $(r_v[1])"
                @assert interval[end] .<= r[end] "Interval end $(interval[end]) is out of bounds $(r[end])"
                v_dt = v(axes(v, 1), interval)
                r = interval
                ax = map(i -> axes(v_dt, i), 1:(length(size(v))-1))
                v = scale(
                    Interpolations.interpolate(v_dt, get_interpolator(v_dt)),
                    ax...,
                    r,
                )
            end
        end
    end
    if range
        return v, r
    else
        return v
    end
end

function record(pops, sym::Symbol; interval=nothing, interpolate=true, kwargs...)
    @assert !isnothing(interval) "Interval must be provided for recording of multiple populations"
    v_dt = map(pops) do pop
        !haskey(pop.records, sym) && return fill(0.0f0, (0, length(interval)))
        rr = record(pop, sym; interval, range = false, interpolate, kwargs...)
        rr(axes(rr, 1), interval)
    end |> x-> vcat(x...)
    r = interval
    ax = map(i -> axes(v_dt, i), 1:(length(size(v_dt))-1))
    scale(
        Interpolations.interpolate(v_dt, get_interpolator(v_dt)),
        ax...,
        r,
    )
end


record(p::T, sym::Symbol, interval::R; kwargs...) where {R<:AbstractRange, T<:Union{Component, Vector{<:Component}}} =
    record(p, sym; interval, kwargs...)



"""
getvariable(obj, key, id=nothing)

Returns the recorded values for a given object and key. If an id is provided, returns the recorded values for that specific id.
"""
function getvariable(obj, key, id = nothing)
    # Dense path (manifesto §6): zero-copy view over written time-slots.
    meta = get(obj.records, :meta, nothing)
    if meta !== nothing && get(get(meta, :mode, Dict()), key, :legacy) === :dense
        buffer = obj.records[key]
        nd = ndims(buffer)
        wp = size(buffer, nd) - get(meta[:allocated], key, 0)
        v = selectdim(buffer, nd, 1:wp)
        if isnothing(id)
            return v
        else
            return selectdim(v, 1, id)
        end
    end
    rec = getrecord(obj, key)
    if isa(rec[1], Matrix)
        @debug "Matrix recording"
        array = zeros(size(rec[1])..., length(rec))
        for i in eachindex(rec)
            array[:, :, i] = rec[i]
        end
        return array
    elseif typeof(rec[1]) <: Vector{Vector{typeof(rec[1][1][1])}} # it is a multipod
        @debug "Multipod recording"
        i = length(rec)
        n = length(rec[1])
        d = length(rec[1][1])
        array = zeros(d, n, i)
        for i in eachindex(rec)
            for n in eachindex(rec[i])
                array[:, n, i] = rec[i][n]
            end
        end
        return array
    else
        @debug "Vector recording"
        isnothing(id) && return hcat(rec...)
        return hcat(rec...)[id, :]
    end
end

"""
    getrecord(p, sym)

Return `p.records[sym]`, or throw `ArgumentError` if not found.
"""
function getrecord(p, sym)
    haskey(p.records, sym) && return p.records[sym]
    throw(ArgumentError("The record $sym is not found"))
end

"""
clear_records!(obj)

Clears all the records of a given object.
"""
function clear_records!(obj)
    if obj isa AbstractPopulation || obj isa AbstractStimulus || obj isa AbstractConnection
        _clear(obj.records)
    else
        for v in obj
            if v isa AbstractPopulation ||
               v isa AbstractStimulus ||
               v isa AbstractConnection 
                @debug "Removing records from $(v.name)"
                _clear(v.records)
            elseif v isa AbstractGroup
                for g in v.elements
                    clear_records!(g)
                end
            elseif v isa String
                continue
            elseif v isa Time
                continue
            else
                clear_records!(v)
            end
        end
    end

end

function _clear(z)
    meta = get(z, :meta, nothing)
    mode = meta isa Dict ? get(meta, :mode, Dict{Symbol,Symbol}()) : Dict{Symbol,Symbol}()
    for (key, val) in z
        key ∈ _RECORD_META_KEYS && continue
        if key == :start_time || key == :end_time
            empty!(val)  # reset time bounds so next sim uses fresh start/end
            continue
        end
        # Dense reset (manifesto §7): drop the buffer and zero the countdown so
        # _allocate_records! reallocates on the next sim!. Keeps the legacy
        # contract length(records[key]) == 0 after clear.
        if get(mode, key, :legacy) === :dense
            if key === :fire && isa(val, Dict)
                haskey(val, :times_buf) && (val[:times_buf] = Vector{Float32}())
                haskey(val, :neurons_buf) && (val[:neurons_buf] = Vector{Int}())
                haskey(val, :time) && empty!(val[:time])
                haskey(val, :neurons) && empty!(val[:neurons])
            else
                z[key] = Vector{Float32}()
            end
            meta[:allocated][key] = 0
            haskey(meta[:step_count], key) && (meta[:step_count][key] = 0)
            continue
        end
        if isa(val, Dict)
            _clear(val)
        else
            try
                empty!(val)
            catch e
                # @warn "Could not clear records for $key"
            end
        end
    end
end

"""
clear_records!(obj, sym::Symbol)

Clears the records of a given object for a specific symbol.
"""
function clear_records!(obj, sym::Symbol)
    meta = get(obj.records, :meta, nothing)
    mode = meta isa Dict ? get(meta, :mode, Dict{Symbol,Symbol}()) : Dict{Symbol,Symbol}()
    for (key, val) in obj.records
        key == sym || continue
        if get(mode, key, :legacy) === :dense
            if key === :fire && isa(val, Dict)
                haskey(val, :times_buf) && (val[:times_buf] = Vector{Float32}())
                haskey(val, :neurons_buf) && (val[:neurons_buf] = Vector{Int}())
                haskey(val, :time) && empty!(val[:time])
                haskey(val, :neurons) && empty!(val[:neurons])
            else
                obj.records[key] = Vector{Float32}()
            end
            meta[:allocated][key] = 0
            haskey(meta[:step_count], key) && (meta[:step_count][key] = 0)
        else
            empty!(val)
        end
    end
end

"""
clear_records!(objs::AbstractArray)

Clears the records of multiple objects.
"""
function clear_records!(objs::AbstractArray)
    for obj in objs
        clear_records!(obj)
    end
end


"""
    clear_monitor!(obj)

Remove all monitoring from `obj` by deleting every entry in `obj.records`. Unlike
`clear_records!`, which empties data but preserves the monitoring schema, this
function removes the schema too — `monitor!` must be called again before the next sim.
"""
function clear_monitor!(obj)
    for (k, val) in obj.records
        delete!(obj.records, k)
    end
end

function clear_monitor!(objs::NamedTuple)
    for obj in values(objs)
        try
            clear_monitor!(obj)
        catch
            typeof(obj) == String && continue
            typeof(obj) == Time && continue
            @warn "Could not clear monitor for $(nameof(typeof(obj))): $obj"
        end
    end
end


export Time,
    get_time,
    get_step,
    get_dt,
    get_interval,
    update_time!,
    record_plast!,
    record_fire!,
    record_sym!,
    record!,
    monitor!,
    getvariable,
    getrecord,
    clear_records!,
    clear_monitor!,
    record,
    reset_time!,
    interpolated_record,
    get_measure_interval,
    add_endtime!,
    add_starttime!
