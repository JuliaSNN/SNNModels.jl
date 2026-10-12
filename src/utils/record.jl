import Interpolations: scale, interpolate, BSpline, Linear, NoInterp
"""
    get_time(T::Time)
    get_time(model::NamedTuple)

Current simulation time in ms (`T.t[1]`, or `model.time.t[1]` for a model built with
`compose`), in the precision of the clock.
"""
get_time(T::Time) = T.t[1]

get_time(model::NamedTuple) = model.time.t[1]

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
get_dt(T::Time) = T.dt

"""
    get_interval(T::Time)

Time points `dt:dt:get_time(T)` (ms) of the steps simulated so far, in the precision of the clock.
"""
get_interval(T::Time{FT}) where {FT} = FT(T.dt):FT(T.dt):get_time(T)

"""
    update_time!(T::Time, dt)
    update_time!(T::Time, myT::Time)

Advance the clock by one step: `T.t[1] += dt` and the step counter `T.tt[1] += 1`. Called at
the beginning of every `sim!`/`train!` step, before the stimuli. The second method copies
time and step counter from `myT` into `T`.
"""
function update_time!(T::Time{FT}, dt::AbstractFloat) where {FT}
    T.t[1] += FT(dt)
    T.tt[1] += 1
end

function update_time!(T::Time{FT}, myT::Time) where {FT}
    T.t[1] = FT(myT.t[1])
    T.tt[1] = myT.tt[1]
end

"""
    reset_time!(T::Time)
    reset_time!(model::NamedTuple)

Set the time to `0` and the step counter to `0` (for a model, its `model.time`). Recordings
are not cleared; use `clear_records!` for that.
"""
function reset_time!(T::Time{FT}) where {FT}
    T.t[1] = zero(FT)
    T.tt[1] = 0
end

function reset_time!(model::NamedTuple)
    model.time.t[1] = zero(eltype(model.time.t))
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
    push!(record[:neurons], ind[findall(fire[ind])])
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
    period = max(1, round(Int, 1.0f0 / sr / get_dt(T)))
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

# ── Dense fire write: from the list of the neurons that fired ───────────────
# `SpikeIndex` the `:fire` recorder reads instead of scanning `obj.fire` (`nothing`: scan). Contract: whenever the list is
# `fed` (or `owned` and `active`) when `record!` runs, it is the complete ascending list of the `true` entries of `fire`
# (never return a list that can be unsorted or repeated, e.g. the `spiking` log of `SpikeTimeStimulus`).
_recording_spike_index(::Any) = nothing
_recording_spike_index(
    p::Union{Poisson,VariablePoisson,InhomogeneousPoisson,Identity,PoissonStimulusLayer},
) = getfield(p, :spikes)
_recording_spike_index(p::_OWNED_LIST_POPULATIONS) = getfield(p, :spikes)

# whether the recorder can read the list of the step from `src` (fed, or owned with the window open)
_has_step_list(::Nothing) = false
_has_step_list(s::SpikeIndex) = s.fed || (s.owned && s.active)

# Ascending indices of the neurons that fired this step: the fed list, the owned list while its window is open, else a
# scan of `fire` into the recorder's scratch list `meta[:fire_scratch]`.
@inline _fire_step_list(::Nothing, fire::AbstractVector{Bool}, meta) = _scan_fire!(fire, meta)

@inline function _fire_step_list(s::SpikeIndex, fire::AbstractVector{Bool}, meta)
    s.fed && return @inbounds view(s.idx, 1:s.n)
    (s.owned && s.active) && return spike_list!(s, fire)
    return _scan_fire!(fire, meta)
end

@inline function _scan_fire!(fire::AbstractVector{Bool}, meta)
    scratch = _fire_scratch!(meta, length(fire))
    n = _flags_to_list!(scratch, fire)
    return @inbounds view(scratch, 1:n)
end

# Scratch list of the recorder, created on first use (also for copied/loaded models and manual loops).
function _fire_scratch!(meta, N::Int)
    s = get(meta, :fire_scratch, nothing)
    s isa Vector{Int32} && length(s) == N && return s
    s = Vector{Int32}(undef, N)
    meta[:fire_scratch] = s
    return s
end

"""
    _fire_mask!(meta, ind, N, owner = nothing)

Membership mask (length `N`) of the recorded subset `ind`, and whether `ind` is strictly increasing. Cached in `meta`
until `ind` is another vector or `N` changes; rebuilt at every `sim!`/`train!` call (in-place edits of `ind` within a
manual loop are not seen). Throws `ArgumentError` if an index is outside `1:N`.
"""
function _fire_mask!(meta, ind::Vector{Int}, N::Int, owner = nothing)
    mask = get(meta, :fire_mask, nothing)
    if mask isa Vector{Bool} && length(mask) == N && get(meta, :fire_mask_src, nothing) === ind
        return mask, meta[:fire_ind_sorted]::Bool
    end
    for i in ind
        1 <= i <= N || throw(ArgumentError(
            "monitor!(…, (:fire, indices)) of $(_owner_name(owner)): index $i outside 1:$N"))
    end
    m = (mask isa Vector{Bool} && length(mask) == N) ? fill!(mask, false) : zeros(Bool, N)
    @inbounds for i in ind
        m[i] = true
    end
    sorted = issorted(ind, lt = <=)   # strictly increasing: sorted and no duplicates
    meta[:fire_mask] = m
    meta[:fire_mask_src] = ind
    meta[:fire_ind_sorted] = sorted
    return m, sorted
end

_owner_name(::Nothing) = "the component"
_owner_name(obj) = hasproperty(obj, :name) ? "$(getproperty(obj, :name)) ($(nameof(typeof(obj))))" : string(nameof(typeof(obj)))

"""
    record_fire_dense!(obj, fire, records, meta, T, indices)

Append the spikes of this step of `obj` (time `Float32(get_time(T))`, neuron id) to the COO buffers
`records[:fire][:times_buf]`, `[:neurons_buf]`, in ascending neuron order (from `_fire_step_list`). With
`indices[:fire]` only the listed neurons are recorded; an unsorted or repeated index list is visited in its own order.
"""
function record_fire_dense!(
    obj,
    fire::AbstractVector{Bool},
    records,
    meta,
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
)
    rec = records[:fire]::Dict{Symbol,AbstractVector}
    alloc = meta[:allocated]::Dict{Symbol,Int}
    t = Float32(get_time(T))
    src = _recording_spike_index(obj)
    if haskey(indices, :fire)
        ind = indices[:fire]
        isempty(ind) && return nothing
        mask, sorted = _fire_mask!(meta, ind, length(fire), obj)
        if !sorted || (!_has_step_list(src) && 8 * length(ind) < length(fire))
            return _write_fire_ordered!(rec, meta, alloc, fire, ind, t)
        end
        return _write_fire_masked!(rec, meta, alloc, _fire_step_list(src, fire, meta), mask, t)
    end
    return _write_fire_list!(rec, meta, alloc, _fire_step_list(src, fire, meta), t)
end

# every neuron of the (ascending) list
function _write_fire_list!(rec, meta, alloc::Dict{Symbol,Int}, L::AbstractVector{Int32}, t::Float32)
    k = length(L)
    k == 0 && return nothing
    times_buf, neurons_buf, wp = _fire_slots!(rec, meta, alloc, k)
    @inbounds for q in 1:k
        times_buf[wp+q] = t
        neurons_buf[wp+q] = Int(L[q])
    end
    alloc[:fire] -= k
    return nothing
end

# the neurons of the (ascending) list that are in the subset
function _write_fire_masked!(rec, meta, alloc::Dict{Symbol,Int}, L::AbstractVector{Int32}, mask::Vector{Bool}, t::Float32)
    m = 0
    @inbounds for q in eachindex(L)
        m += mask[L[q]]
    end
    m == 0 && return nothing
    times_buf, neurons_buf, wp = _fire_slots!(rec, meta, alloc, m)
    j = wp
    @inbounds for q in eachindex(L)
        i = Int(L[q])
        if mask[i]
            j += 1
            times_buf[j] = t
            neurons_buf[j] = i
        end
    end
    alloc[:fire] -= m
    return nothing
end

# the subset in its own order (unsorted, repeated, or short subsets of a population without a fed list)
function _write_fire_ordered!(rec, meta, alloc::Dict{Symbol,Int}, fire::AbstractVector{Bool}, ind::Vector{Int}, t::Float32)
    m = 0
    for i in ind
        m += fire[i]
    end
    m == 0 && return nothing
    times_buf, neurons_buf, wp = _fire_slots!(rec, meta, alloc, m)
    j = wp
    for i in ind
        if fire[i]
            j += 1
            @inbounds times_buf[j] = t
            @inbounds neurons_buf[j] = i
        end
    end
    alloc[:fire] -= m
    return nothing
end

# Buffers with room for `need` more spikes, and the index of the last written slot.
@inline function _fire_slots!(rec, meta, alloc::Dict{Symbol,Int}, need::Int)
    _ensure_fire_capacity!(rec, meta, need)
    times_buf = rec[:times_buf]::Vector{Float32}
    neurons_buf = rec[:neurons_buf]::Vector{Int}
    return times_buf, neurons_buf, length(times_buf) - alloc[:fire]
end

"""
    _ensure_fire_capacity!(rec, meta, need)

Make room for `need` more spikes in the `:fire` COO buffers (called once per step, before the
writes). If the countdown `meta[:allocated][:fire]` is short, the buffers grow to
`max(2 * capacity, written + need, 1)` slots, keeping the written prefix, with one warning per recording.
"""
function _ensure_fire_capacity!(rec, meta, need::Int)
    alloc = meta[:allocated]::Dict{Symbol,Int}
    remaining = get(alloc, :fire, 0)
    remaining >= need && remaining <= length(rec[:times_buf]::Vector{Float32}) && return nothing
    _grow_fire!(rec, meta, alloc, remaining, need)
    return nothing
end

@noinline function _grow_fire!(rec, meta, alloc::Dict{Symbol,Int}, remaining::Int, need::Int)
    times_buf = rec[:times_buf]::Vector{Float32}
    neurons_buf = rec[:neurons_buf]::Vector{Int}
    cap = length(times_buf)
    written = clamp(cap - remaining, 0, cap)   # a countdown larger than the buffer (copied model): nothing written
    new_cap = max(2 * cap, written + need, 1)
    new_times = Vector{Float32}(undef, new_cap)
    new_neurons = Vector{Int}(undef, new_cap)
    copyto!(new_times, 1, times_buf, 1, written)
    copyto!(new_neurons, 1, neurons_buf, 1, written)
    rec[:times_buf] = new_times
    rec[:neurons_buf] = new_neurons
    alloc[:fire] = new_cap - written
    grew = meta[:grew]
    if !get(grew, :fire, false)
        @warn "Fire COO buffer overflowed; doubled to $new_cap spike slots. Pass monitor_rate= to avoid reallocation."
        grew[:fire] = true
    end
    return nothing
end

@inline function get_model_field(obj, key::Symbol, var_map::Dict{Symbol,Tuple{Symbol,Symbol}}, T::Time)
    if haskey(var_map, key)
        parent, sub = var_map[key]
        # variables of a plasticity rule are recorded as values at time T (see `monitor!`)
        return plasticity_value(obj, parent, sub, T)
    elseif key === :W && obj isa AbstractSparseSynapse
        # weights of rules that defer their update (vSTDPParameter) are read through the accessor
        return plasticity_value(obj, :W)
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
    # one holder for (dt, duration): the per-component dynamic call does not box two Float32s
    ctx = Ref((dt, duration))
    _allocate_each!(P, ctx)
    _allocate_each!(C, ctx)
    _allocate_each!(S, ctx)
    return nothing
end

function _allocate_each!(objs, ctx::Base.RefValue{Tuple{Float32,Float32}})
    for obj in objs
        (obj isa AbstractPopulation || obj isa AbstractConnection || obj isa AbstractStimulus) || continue
        isempty(getfield(obj, :records)) && continue   # unmonitored: nothing to allocate
        _allocate_component!(obj, ctx)
    end
    return nothing
end

_allocate_component!(obj, dt::Real, duration::Real) =
    _allocate_component!(obj, Ref((Float32(dt), Float32(duration))))

function _allocate_component!(obj, ctx::Base.RefValue{Tuple{Float32,Float32}})
    records = obj.records
    haskey(records, :data) || return
    haskey(records, :meta) || return
    meta = records[:meta]
    haskey(meta, :mode) || return
    # function barrier on the usual dictionary types: static calls, `obj` and the Float32s are not boxed
    if records isa Dict{Any,Any} && meta isa Dict{Symbol,Any}
        return _allocate_keys!(obj, records, meta, ctx)
    end
    return _allocate_keys!(obj, records, meta, ctx)
end

_buffer_matches(buf::AbstractArray{<:Any,N}, snap_size::Tuple) where {N} = N == length(snap_size) + 1
_buffer_matches(buf, snap_size) = false

function _allocate_keys!(obj, records, meta, ctx::Base.RefValue{Tuple{Float32,Float32}})
    dt, duration = ctx[]
    mode = meta[:mode]::Dict{Symbol,Symbol}
    alloc = meta[:allocated]::Dict{Symbol,Int}
    monitor_time = meta[:monitor_time]::Dict{Symbol,Float32}
    periods = meta[:period]::Dict{Symbol,Int}
    dts = meta[:dt]::Dict{Symbol,Float32}
    srs = records[:sr]::Dict{Symbol,Float32}
    N = hasfield(typeof(obj), :N) ? Int(getfield(obj, :N)) : 0
    for key in records[:data]::Vector{Symbol}
        get(mode, key, :legacy) === :dense || continue
        if key === :fire
            _allocate_fire!(obj, meta, N, duration)
            continue
        end
        # Reconcile period against the real dt now that it is known.
        sr = srs[key]
        period = max(1, round(Int, 1.0f0 / sr / dt))
        periods[key] = period

        snap_size = meta[:snapshot_size][key]::Tuple
        buf = records[key]
        allocated_buffer = _buffer_matches(buf, snap_size)
        dt_changed = allocated_buffer && get(dts, key, dt) != dt

        # Samples a chunk of `duration` produces. The +1 (record_zero! sample at
        # T.tt = 0) only applies to the very first chunk on a fresh buffer.
        hint_ms = if haskey(monitor_time, key)
            monitor_time[key]
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
            records[key] = Array{Float32}(undef, snap_size..., chunk_capacity)
            alloc[key] = chunk_capacity
            dts[key] = dt
        else
            # Existing buffer persists across chunks. Ensure room for this chunk;
            # extend in place if the remaining countdown is insufficient.
            remaining = get(alloc, key, 0)
            if remaining < chunk_capacity
                _extend_dense!(records, meta, key, chunk_capacity - remaining)
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
    # Geometric growth: chunked runs reallocate O(log n) times instead of once per chunk.
    new_cap = max(old_cap + extra, 2 * old_cap)
    snap_size = size(buffer)[1:(nd-1)]
    new_buffer = Array{Float32}(undef, snap_size..., new_cap)
    write_ptr > 0 && (selectdim(new_buffer, nd, 1:write_ptr) .= selectdim(buffer, nd, 1:write_ptr))
    records[key] = new_buffer
    meta[:allocated][key] = new_cap - write_ptr
    if !get(meta[:grew], key, false)
        @warn "Dense recording buffer for $key extended across sim!/train! chunks to $new_cap slots (grows geometrically). Pass monitor_time= with the total duration to allocate it once."
        meta[:grew][key] = true
    end
    return
end

function _allocate_fire!(obj, meta, N::Int, duration::Float32)
    fire = obj.records[:fire]::Dict{Symbol,AbstractVector}
    alloc = meta[:allocated]::Dict{Symbol,Int}
    rate = get(meta[:monitor_rate]::Dict{Symbol,Float32}, :fire, 20f0)  # Hz; conservative default
    dur_s = duration / 1000f0
    chunk_cap = max(1, ceil(Int, rate * max(N, 1) * dur_s))
    # the subset mask is rebuilt at the first step of every sim!/train! (covers in-place edits of the indices)
    delete!(meta, :fire_mask_src)
    allocated_buffer = haskey(fire, :times_buf) && !isempty(fire[:times_buf])
    if !allocated_buffer
        # First allocation.
        fire[:times_buf] = Vector{Float32}(undef, chunk_cap)
        fire[:neurons_buf] = Vector{Int}(undef, chunk_cap)
        alloc[:fire] = chunk_cap
    else
        # Extend across chunks if remaining room is insufficient for this chunk.
        remaining = get(alloc, :fire, 0)
        if remaining < chunk_cap
            _extend_fire!(fire, meta, chunk_cap - remaining)
        end
    end
end

function _extend_fire!(fire, meta, extra::Int)
    old_cap = length(fire[:times_buf])
    write_ptr = old_cap - get(meta[:allocated], :fire, 0)
    # Geometric growth: chunked runs reallocate O(log n) times instead of once per chunk.
    new_cap = max(old_cap + extra, 2 * old_cap)
    new_times = Vector{Float32}(undef, new_cap)
    new_neurons = Vector{Int}(undef, new_cap)
    write_ptr > 0 && copyto!(new_times, 1, fire[:times_buf], 1, write_ptr)
    write_ptr > 0 && copyto!(new_neurons, 1, fire[:neurons_buf], 1, write_ptr)
    fire[:times_buf] = new_times
    fire[:neurons_buf] = new_neurons
    meta[:allocated][:fire] = new_cap - write_ptr
    # Same one-time latch as _grow_fire!: at most one warning per recording, whichever path grows first.
    if !get(meta[:grew], :fire, false)
        @warn "Fire COO buffer extended across sim!/train! chunks to $new_cap spike slots (grows geometrically). monitor_rate= sizes each chunk; spikes accumulate across chunks."
        meta[:grew][:fire] = true
    end
    return
end

"""
    record!(obj, T::Time)

Record all monitored variables of `obj` at the current time `T`. Dispatches to
the dense or legacy path per `meta[:mode][key]`. Called every dt inside the sim loop.
"""
@inline function record!(obj, T::Time)
    # inlined fast exit for unmonitored objects (hot path, every dt)
    isempty(getfield(obj, :records)) && return nothing
    return _record!(obj, T)
end

function _record!(obj, T::Time)
    records = getfield(obj, :records)
    !haskey(records, :data) && return
    meta = records[:meta]
    # function barrier: typed records/meta for the usual dictionaries (hot path, every dt)
    if records isa Dict{Any,Any} && meta isa Dict{Symbol,Any}
        return _record_keys!(obj, records, meta, T)
    end
    return _record_keys!(obj, records, meta, T)
end

function _record_keys!(obj, records, meta, T::Time)
    time = get_time(T)
    mode = meta[:mode]::Dict{Symbol,Symbol}
    start_time = records[:start_time]
    end_time = records[:end_time]
    for key in records[:data]::Vector{Symbol}
        # start/end time = times of the first and last sample actually taken (every step for
        # :fire, every `period` steps of the global step counter otherwise)
        sampled = if key === :fire
            true
        elseif get(mode, key, :legacy) === :dense
            (get_step(T) % (meta[:period]::Dict{Symbol,Int})[key]) == 0
        else
            record_step(T, records[:sr][key])
        end
        if sampled
            # same code twice: the `isa` branch is the typed one (no boxed `time`), the other covers old files
            if start_time isa Dict{Symbol,Float32} && end_time isa Dict{Symbol,Float32}
                isnan(get(start_time, key, NaN32)) && (start_time[key] = time)
                end_time[key] = time
            else
                isnan(get(start_time, key, NaN32)) && (start_time[key] = time)
                end_time[key] = time
            end
        end
        if key === :fire
            if get(mode, :fire, :legacy) === :dense
                record_fire_dense!(obj, obj.fire, records, meta, T, records[:indices]::Dict{Symbol,Vector{Int}})
            else
                record_fire!(obj.fire, records[:fire], T, records[:indices])
            end
        elseif !sampled
            # not a sampling step: nothing is read (a lazily stored variable would be materialised for nothing)
            continue
        elseif get(mode, key, :legacy) === :dense
            snapshot = get_model_field(obj, key, meta[:var_map], T)
            ind = get(records[:indices], key, Int[])
            if !isempty(ind)
                snapshot isa AbstractArray || throw(ArgumentError("Indexed recording for $key requires an array-like field, got $(typeof(snapshot))"))
                snapshot = view(snapshot, ind)
            end
            record_sym_dense!(snapshot, records, meta, key, T)
        else
            record_sym!(
                get_model_field(obj, key, meta[:var_map], T),
                obj, key, T, records[:indices], records[:sr][key],
            )
        end
    end
end

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
# When indices are provided the snapshot_size reflects len(ind), not the full field.
function _classify_record(sample, has_indices::Bool, n_indices::Int = 0)
    if sample isa Float32
        return (:dense, ())
    elseif sample isa Vector{Float32}
        snap = has_indices ? n_indices : length(sample)
        return (:dense, (snap,))
    elseif sample isa Matrix{Float32}
        return has_indices ? (:legacy, ()) : (:dense, size(sample))
    elseif sample isa Array{Float32,3}
        return has_indices ? (:legacy, ()) : (:dense, size(sample))
    else
        return (:legacy, ())
    end
end

# monitor_rate in Hz. Rates carry units (1Hz = 0.001 per ms): a value above 1kHz is a plain number of Hz.
function _rate_Hz(monitor_rate)
    monitor_rate > kHz && throw(ArgumentError(
        "monitor_rate = $monitor_rate is $(monitor_rate / Hz) Hz: rates carry units, write e.g. monitor_rate = $(monitor_rate)Hz"))
    return Float32(monitor_rate / Hz)
end

"""
    monitor!(obj, keys; sr = 1000Hz, variables = :none, monitor_time = nothing, monitor_rate = nothing, verbose = false)
    monitor!(objs::Union{Array,NamedTuple}, keys; sr = 200Hz, kwargs...)
    monitor!(obj, keys, variables::Symbol; kwargs...)

Register variables of a population, connection or stimulus for recording during the next
`sim!`/`train!` calls. `keys` is a `Symbol`, a `(Symbol, indices)` tuple, or a vector of
them. Buffers are allocated just before the time loop starts (`_allocate_records!`), not
here.

# Arguments
- `obj`: an `AbstractPopulation`, `AbstractConnection` or `AbstractStimulus`. For a vector
  or a NamedTuple (e.g. `model.pop`) the call is repeated on each element, and the default
  sampling rate is then `200Hz` instead of `1000Hz`.
- `keys`: field names of `obj` (e.g. `:v`, `:w`, `:W`, `:ρ`, `:fire`). A tuple
  `(sym, indices)` records only the listed elements of a vector field.
- `sr = 1000Hz`: sampling rate. The sampling period is `max(1, round(1 / (sr * dt)))` steps,
  so `sr` larger than `1/dt` records every step. Ignored for `:fire`.
- `variables = :none`: name of a nested field holding the variables, e.g. `:STPVars` or
  `:LTPVars` of a `SpikingSynapse`, or `:synvars`. The key `sym` is then read from
  `obj.<variables>.<sym>` and stored as `Symbol(variables, "_", sym)`, e.g. `:STPVars_u`.
  The positional form `monitor!(obj, keys, :STPVars)` is equivalent. Variables of a plasticity
  rule are recorded at their value at the sampling time, read through `plasticity_value` (lazily decayed
  traces are not stored at that value; reading the field directly returns the stored value). Recording never
  changes the state. A lazily stored variable without a `plasticity_value` method is an `ArgumentError`.
- `monitor_time`: capacity hint in ms used to size the buffers (default: the duration of the
  `sim!` call, at least 1 s). Recording beyond the capacity is still correct: the buffer is
  doubled with a one-time warning.
- `monitor_rate`: expected maximum firing rate, with units (e.g. `50Hz`), used to size the
  `:fire` buffers (default `20Hz`). A plain number (any value above `1kHz`) is an `ArgumentError`.
- `verbose = false`: warn about keys already monitored or not found.

# Storage
- `:fire` is stored as a list of `(time, neuron)` events; read it with `spiketimes(obj)`,
  `firing_rate`, or `record(obj, :fire; interval)`.
- `Float32` scalars, `Vector{Float32}` (also with indices), and `Matrix{Float32}` /
  `Array{Float32,3}` without indices are stored in a dense array with time as the last
  dimension; read with `getvariable` (raw samples) or `record` (interpolated in time).
- Other field types (e.g. matrices with indices, vectors of vectors) are stored as a vector
  of snapshots.
- A first sample is taken at `t = 0` when the model time is 0.

Already-monitored keys and missing fields are skipped silently (with a warning if
`verbose = true`).

With `(:fire, indices)` only the spikes of the listed neurons are recorded (with their
original neuron indices, so `spiketimes` still returns one entry per neuron of the
population). An index outside `1:N` throws an
`ArgumentError` at the first step of the next `sim!`/`train!`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
stim = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
SNN.monitor!(E, [:v, :fire]; sr = 2kHz)
SNN.monitor!(E, [(:w, [1, 2, 3])])
model = SNN.compose(; E, stim)
SNN.sim!(; model, duration = 200ms)
v = SNN.getvariable(E, :v)          # 100 x 401 samples
st = SNN.spiketimes(E)
```
"""
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
            !isempty(ind) && (obj.records[:indices][:fire] = collect(Int, ind))
            meta[:mode][:fire] = :dense
            meta[:allocated][:fire] = 0
            meta[:step_count][:fire] = 0
            meta[:grew][:fire] = false
            monitor_rate !== nothing && (meta[:monitor_rate][:fire] = _rate_Hz(monitor_rate))
            @debug "Monitoring :fire in $(obj.name)"
            :fire ∉ obj.records[:data] && push!(obj.records[:data], :fire)
            continue
        end
        if variables == :none
            if hasfield(typeof(obj), sym)
                # the efficacy of a SpikingSynapse without short-term rule is not stored: store it (ones) so that it can be recorded
                (sym === :ρ && obj isa SpikingSynapse) && materialize_efficacy!(obj)
                # read through the accessor, as the recorder will: throws now rather than mid-run
                sample = (sym === :W && obj isa AbstractSparseSynapse) ? plasticity_value(obj, :W) : getfield(obj, sym)
                typ = typeof(sample)
                key = sym
            else
                verbose && @warn "Field $sym not found in $(nameof(typeof(obj)))"
                continue
            end
        else
            if hasproperty(obj, variables)
                if hasproperty(getfield(obj, variables), sym)
                    # read through the accessor, as the recorder will: throws now rather than mid-run
                    sample = plasticity_value(obj, variables, sym, 0.0f0)
                    typ = typeof(sample)
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
        mode, snap_size = _classify_record(sample, !isempty(ind), length(ind))
        meta[:mode][key] = mode
        meta[:period][key] = max(1, round(Int, 1.0f0 / Float32(sr) / dt_hint))
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
    interpolated_record(p, sym, τ = 20ms)

Return `(y, r)`: the recording `sym` of `p` as an `Interpolations` object `y` with time
as the last axis, and the range `r` of time points (ms) assigned to the samples.
It is evaluated with call syntax, `y(i, t)` (indices or ranges), at any `t` within `r` (linear
interpolation; singleton dimensions are not interpolated). Indexing with square brackets and
a non-integer time does not work.

Samples are taken at the steps whose global step counter is a multiple of the sampling
period (including `t = 0` for a fresh model). `r` spans from the time of the first sample to
the time of the last one (`records[:start_time][sym]` to `records[:end_time][sym]`) with as many
points as samples, so it is the exact sample-time axis, also when monitoring starts after
`t = 0` or the simulated time is not a multiple of the sampling period.

For `sym == :fire` it returns `firing_rate(p, τ = τ)`.

!!! note "Changed after SNNModels 1.8.4"
    Up to 1.8.4 the start and end times were those of the first and last `record!` call, not
    of the first and last sample, so the axis was shifted or stretched by up to one sampling
    period (e.g. `:W` sampled at 10 Hz from 2 s to 4 s: true times `2100:100:4000` ms, axis
    `2000.125:105.26:4000`); the `τ` argument was ignored for `:fire`.
"""
function interpolated_record(p, sym, τ = 20ms)
    if sym == :fire
        return firing_rate(p, τ = τ)
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

"""
    get_measure_interval(p, sym::Symbol, steps::Int)
    get_measure_interval(p, sym::Symbol, step::AbstractFloat)

Time range from `p.records[:start_time][sym]` to `p.records[:end_time][sym]`, with `steps`
points or with step `step` (ms).
"""
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



"""
    add_endtime!(model::NamedTuple)

For every monitored key of every component of `model` that has no end time, set
`records[:end_time][key]` to the current model time. Used for records loaded from older
files.
"""
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

"""
    add_starttime!(model::NamedTuple)

For every monitored key of every component of `model` that has no start time, set
`records[:start_time][key] = 0`. Used for records loaded from older files.
"""
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
    record(p, sym::Symbol; range = false, interval = nothing, interpolate = true, variables = nothing, kwargs...)
    record(p, sym::Symbol, interval::AbstractRange; kwargs...)
    record(pops, sym::Symbol; interval, interpolate = true, kwargs...)

Read a recording of a population, connection or stimulus `p`.

- `sym == :fire`: firing rates computed by `firing_rate(p, interval; interpolate, kwargs...)`;
  `interval` (an `AbstractRange`, ms) is required.
- `sym == :spiketimes` or `:spikes`: returns `spiketimes(p)` (ignores `range`).
- any other `sym`: with `interpolate = true` (default) an interpolation object over time,
  as returned by [`interpolated_record`](@ref), resampled on `interval` if given; with
  `interpolate = false` the raw samples `getvariable(p, sym)` (and `r = interval`).
- `variables`: name of the variable group used in `monitor!` (e.g. `:STPVars`); the key read is
  `Symbol(variables, "_", sym)`. Equivalent to passing `:STPVars_u` directly.
- `range = true` returns `(v, r)` instead of `v`.

The method on a collection `pops` stacks the interpolated recordings of all elements sampled
on `interval` (required); elements that do not record `sym` contribute no rows.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 50)
stim = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
SNN.monitor!(E, [:v, :fire])
model = SNN.compose(; E, stim)
SNN.sim!(; model, duration = 500ms)
v = SNN.record(E, :v)                   # interpolated: v(1, 100.5ms)
v, r = SNN.record(E, :v, range = true)  # r: time axis in ms
v_raw = SNN.record(E, :v, interpolate = false)
fr, r = SNN.record(E, :fire; interval = 0:10ms:500ms, range = true)
st = SNN.record(E, :spikes)
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
                @assert interval[1] .>= r[1] "Interval start $(interval[1]) is out of bounds $(r[1])"
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
    getvariable(obj, key::Symbol, id = nothing)

Raw recorded samples of `key`, without interpolation, with time as the last dimension
(neurons x samples for a vector field). For dense records this is a view on the buffer
restricted to the samples written so far; `id` selects an index (or indices) along the
first dimension. Legacy records are concatenated into a new array.
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
    elseif typeof(rec[1]) <: Vector{Vector{typeof(rec[1][1][1])}} # nested vectors (one vector per neuron, e.g. multicompartment variables)
        @debug "Nested vector recording"
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

Return the raw storage `p.records[sym]` (for dense records the full pre-allocated buffer,
including unwritten slots; prefer `getvariable`). Throws `ArgumentError` if `sym` is not
recorded.
"""
function getrecord(p, sym)
    haskey(p.records, sym) && return p.records[sym]
    throw(ArgumentError("The record $sym is not found"))
end

"""
    clear_records!(obj)

Delete the recorded data of `obj` (a component, or a model/NamedTuple/collection of
components, recursing into groups) but keep the monitoring set-up: the next `sim!` records
the same variables again, with new start and end times.
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

Delete the recorded data of the key `sym` of `obj` only, keeping it monitored.
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

Call `clear_records!` on each element of `objs`.
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
