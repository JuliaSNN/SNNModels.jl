"""
    perturbation_test(model, simtime; trigger! = (m -> nothing), perturbation! = nothing,
                      from_state = nothing, add_records = nothing, train = true, kwargs...)

Run a perturbed copy of `model` for `simtime` ms and return it, or archive its recordings in
`model`.

The copy is created with `modelcopy` (deep copy with empty record buffers), `trigger!` is
applied to it once, then the copy is simulated with `train!` or `sim!`. The original `model` is
modified only when `add_records` is given (its `records[:perturbation]` entries are filled).

# Arguments
- `model`: source model.
- `simtime`: duration in ms, e.g. `500ms`.

# Keyword arguments
- `trigger!` (no-op): `(pert_model) -> nothing`, applied once to the copy before the run, e.g.
  ```julia
  m -> SNN.set_active!(m.stim.noise, false)   # silence a stimulus
  m -> (m.pop.pv.I .= 200pA)                  # inject current
  m -> SNN.set_LTP!(m.syn.exc_exc, false)     # freeze plasticity
  ```
- `perturbation!` (`nothing`): forwarded to `train!`/`sim!`, which call
  `perturbation!(; P, C, S, t, dt, start_time)` at every time step.
- `from_state` (`nothing`): pre-built model to use instead of `modelcopy(model)`.
  The caller is responsible for empty records and correct starting state.
- `add_records` (`nothing`): if `nothing`, returns the perturbed model.
  If a `String`/`Symbol`, appends the recording window of every monitored variable of the perturbed model to
  `model.<component>.records[:perturbation][variable][condition]` (see "Storage schema")
  and returns `nothing`.
- `train` (`true`): if `true` runs `train!` (plasticity active); if `false` runs `sim!`.
- Remaining `kwargs` forwarded to `train!`/`sim!`.

# Storage schema (when `add_records` is set)

From SNNModels 1.9.2 / 2.0.2 the windows of one `(variable, condition)` are stored in one ragged
store (append, delete by id):

```
obj.records[:perturbation][variable::Symbol][condition::String]
    = PerturbationStore         # dense variables: N x (sum of window lengths) Float32
    | PerturbationSpikeStore    # :fire: flat spike times + neuron indices
```

See [`PerturbationStore`](@ref) and [`PerturbationSpikeStore`](@ref) for the layout, and
[`perturbation_windows`](@ref) to read the windows one by one. Records written by earlier versions
(one `Dict("interval" => ..., "data" => ...)` per window, under
`records[:perturbation][variable][condition][n]`) are NOT read, appended to or edited window by
window: the old and the new layout never coexist in a working model, and every flush or reader that
meets a legacy store throws an `ArgumentError` naming object, variable and condition. Any other value (e.g. a `NamedTuple`) in place of a store is refused the same way, with its type. Convert such files
once with [`compact_perturbation_records!`](@ref) and save them again. The old layout stored 17,160
small dictionaries per variable for 1,320 presentations x 13 conditions; the new one stores a handful of
arrays. Only 2-D (N x T) dense windows can be stored; a flush of an N-dimensional window throws an
`ArgumentError`. Files with compact records need SNNModels >= 1.9.2 / >= 2.0.2 in every process that
reads them.

Retrieve with `perturbation_record(obj, variable, condition, interval)`.

# Example
```julia
using SpikingNeuralNetworks
@load_units
E = SNN.IF(N = 20, name = "E")
noise = SNN.CurrentStimulus(E, :I, param = SNN.CurrentNoise(E; I_base = 300pA))
model = SNN.compose(; E, noise, silent = true)
SNN.monitor!(E, [:v, :fire]; sr = 1kHz)
sim!(model, 100ms)                               # baseline
SNN.perturbation_test(model, 50ms; trigger! = m -> (m.pop.E.I .+= 100pA),
                      add_records = "kick", train = false)
v, r = SNN.perturbation_record(E, :v, "kick", 0:1ms:150ms)
```
"""
function perturbation_test(
    model,
    simtime;
    trigger! = ((x) -> nothing),
    perturbation! = nothing,
    from_state = nothing,
    add_records = nothing,
    train = true,
    kwargs...,
)
    # If from_state is provided the caller has already prepared an empty/clean model
    # at the desired starting state — use it directly without copying.
    # Otherwise fall back to modelcopy (deep copy with empty record buffers).
    pert_model = isnothing(from_state) ? modelcopy(model) : from_state
    trigger!(pert_model)
    if train
        train!(pert_model, simtime; kwargs..., perturbation! )
    else
        sim!(pert_model, simtime; kwargs..., perturbation!)
    end
    if isnothing(add_records)
        return pert_model
    else
        _flush_perturbation!(model, pert_model, String(add_records))
        return nothing
    end
end

# Walk model and pert_model in lockstep and flush each component pair. All pairs are checked
# first, so a refused flush leaves every record untouched.
function _flush_perturbation!(model, pert_model, condition::String)
    pairs = Tuple{Any,Any}[]
    for field in (:pop, :syn, :stim)
        haskey(model, field) || continue
        haskey(pert_model, field) || continue
        orig_group = getfield(model, field)
        pert_group = getfield(pert_model, field)
        for key in keys(orig_group)
            haskey(pert_group, key) || continue
            push!(pairs, (getfield(orig_group, key), getfield(pert_group, key)))
        end
    end
    for (o, p) in pairs
        _check_flush(o, p, condition)
    end
    for (o, p) in pairs
        _flush_component!(o, p, condition)
    end
end

_legacy_layout_error(obj, variable, condition) = throw(ArgumentError(
    "perturbation records of $(_cname(obj)), variable :$variable, condition \"$condition\" are in the " *
    "pre-1.9.2 layout (one Dict per window); convert them once with compact_perturbation_records! " *
    "(then save the model again) before reading, appending to or deleting single windows of them"))

# Throw if cs (the stored object of a (variable, condition)) is anything but a store (or nothing /
# an empty Dict): a legacy Dict-per-window store, or a foreign layout such as a NamedTuple.
function _check_not_legacy(obj, variable, condition, cs)
    (isnothing(cs) || cs isa AbstractPerturbationStore) && return cs
    cs isa AbstractDict && isempty(cs) && return cs
    cs isa AbstractDict && _legacy_layout_error(obj, variable, condition)
    throw(ArgumentError(
        "perturbation records of $(_cname(obj)), variable :$variable, condition \"$condition\" hold a value of type " *
        "$(typeof(cs)), not a PerturbationStore / PerturbationSpikeStore; this SNNModels reads and writes only those " *
        "(the interim layouts of other projects must be rebuilt as stores)"))
end

function _stored_condition(obj, variable, condition)
    pert = get(obj.records, :perturbation, nothing)
    isnothing(pert) && return nothing
    vs = get(pert, variable, nothing)
    isnothing(vs) && return nothing
    return get(vs, condition, nothing)
end

# Refusals of a flush, raised before anything is written.
function _check_flush(orig_obj, pert_obj, condition::String)
    haskey(pert_obj.records, :data) || return
    isempty(pert_obj.records[:data]) && return
    for key in pert_obj.records[:data]
        tstart = get(pert_obj.records[:start_time], key, NaN32)
        tend   = get(pert_obj.records[:end_time],   key, NaN32)
        (isnan(tstart) || isnan(tend)) && continue
        _check_not_legacy(orig_obj, key, condition, _stored_condition(orig_obj, key, condition))
        if key !== :fire
            nd = ndims(getvariable(pert_obj, key))
            nd == 2 || throw(ArgumentError(
                "perturbation window of $(_cname(orig_obj)), variable :$key, condition \"$condition\" has $nd dimensions; the perturbation store holds matrices (N, T) only"))
        end
    end
end

function _flush_component!(orig_obj, pert_obj, condition::String)
    haskey(pert_obj.records, :data) || return
    isempty(pert_obj.records[:data]) && return
    _check_flush(orig_obj, pert_obj, condition)
    haskey(orig_obj.records, :perturbation) ||
        (orig_obj.records[:perturbation] = Dict{Symbol,Any}())
    pert_store = orig_obj.records[:perturbation]

    for key in pert_obj.records[:data]
        tstart = get(pert_obj.records[:start_time], key, NaN32)
        tend   = get(pert_obj.records[:end_time],   key, NaN32)
        (isnan(tstart) || isnan(tend)) && continue

        haskey(pert_store, key) || (pert_store[key] = Dict{String,Any}())
        var_store = pert_store[key]
        cond_store = get(var_store, condition, nothing)
        # an empty Dict (nothing stored) is replaced by a store
        cond_store isa AbstractDict && (cond_store = nothing)

        if key === :fire
            st = spiketimes(pert_obj)
            isnothing(cond_store) && (var_store[condition] = cond_store = PerturbationSpikeStore(length(st)))
            _push_window!(cond_store, st, tstart, tend)
        else
            raw = getvariable(pert_obj, key)
            isnothing(cond_store) && (var_store[condition] = cond_store = PerturbationStore(size(raw, 1)))
            _push_window!(cond_store, raw, tstart, tend)
        end
    end
end

# ── Compact layout ────────────────────────────────────────────────────────────

"""
    AbstractPerturbationStore

Supertype of [`PerturbationStore`](@ref) and [`PerturbationSpikeStore`](@ref): the ragged
containers of the perturbation windows of one `(variable, condition)`.
They support `length` (number of windows), `isempty`, `keys` (the window numbers `n`),
`haskey(store, n)` and `store[n]` (a legacy-style `Dict("interval" => ..., "data" => ...)`,
a copy).
"""
abstract type AbstractPerturbationStore end

"""
    PerturbationStore(N)

Dense perturbation windows of one `(variable, condition)` of an object with `N` rows
(neurons or synapses). Built by `perturbation_test(...; add_records = condition)`; fields:

- `version::Int`: layout version (1); readers and writers throw an `ArgumentError` ("upgrade SNNModels") for any other number.
- `N::Int`: rows.
- `data::Vector{Float32}`: the windows concatenated along time, column-major. Viewed as an
  `N x (sum of window lengths)` matrix, window `j` is the column block
  `offsets[j] : offsets[j+1]-1`, i.e. `data[(offsets[j]-1)*N+1 : (offsets[j+1]-1)*N]`. Nothing divides by `N`,
  so objects with 0 rows work.
- `offsets::Vector{Int}`: `length(store) + 1` entries, `offsets[1] = 1`; window `j` is the columns
  `offsets[j] : offsets[j+1]-1` (zero columns allowed; an empty store has `offsets == [1]`).
- `tstart::Vector{Float32}`, `tend::Vector{Float32}`: absolute simulation interval (ms) of each window.
- `ids::Vector{Int}`: window number `n` (the argument `n` of `clear_perturbation_records!`);
  increasing.
- `next_id::Int`: the number the next window gets (then incremented). Window numbers are never
  reused, not even after every window was deleted; the store is mutable for this counter.
  `compact_perturbation_records!` sets it to `maximum(legacy keys) + 1`.

Windows may differ in length. Appending is amortised O(window size); deleting a window is O(store
size) (its columns are removed from `data` and the later `offsets` shift down). The fields are
plain arrays of bits types and integers: JLD2 stores them as they are.
"""
mutable struct PerturbationStore <: AbstractPerturbationStore
    version::Int
    N::Int
    data::Vector{Float32}
    offsets::Vector{Int}
    tstart::Vector{Float32}
    tend::Vector{Float32}
    ids::Vector{Int}
    next_id::Int
end
PerturbationStore(N::Integer) =
    PerturbationStore(1, Int(N), Float32[], [1], Float32[], Float32[], Int[], 1)

"""
    PerturbationSpikeStore(N)

Spike windows (variable `:fire`) of one condition of an object with `N` neurons. Fields:

- `version::Int`: layout version (1).
- `N::Int`: neurons.
- `times::Vector{Float32}`, `neuron::Vector{Int32}`: all spikes of all windows, parallel arrays.
  Inside a window the spikes are sorted by neuron, then by time (the order of
  `spiketimes(pert_obj)` flattened neuron by neuron).
- `offsets::Vector{Int}`: `length(store) + 1` entries, `offsets[1] = 1`; window `j` is the spikes
  `offsets[j] : offsets[j+1]-1` of `times`/`neuron` (an empty store has `offsets == [1]`).
- `tstart`, `tend`, `ids`, `next_id`: as in [`PerturbationStore`](@ref).
"""
mutable struct PerturbationSpikeStore <: AbstractPerturbationStore
    version::Int
    N::Int
    times::Vector{Float32}
    neuron::Vector{Int32}
    offsets::Vector{Int}
    tstart::Vector{Float32}
    tend::Vector{Float32}
    ids::Vector{Int}
    next_id::Int
end
PerturbationSpikeStore(N::Integer) =
    PerturbationSpikeStore(1, Int(N), Float32[], Int32[], [1], Float32[], Float32[], Int[], 1)

Base.length(s::AbstractPerturbationStore) = length(s.offsets) - 1
Base.isempty(s::AbstractPerturbationStore) = length(s) == 0

# Readers and writers refuse a layout newer than this code knows.
function _check_version(s::AbstractPerturbationStore)
    s.version == 1 || throw(ArgumentError(
        "perturbation store of layout version $(s.version), but this SNNModels reads version 1 only: upgrade SNNModels"))
    return s
end
Base.keys(s::AbstractPerturbationStore) = copy(s.ids)
Base.haskey(s::AbstractPerturbationStore, n::Integer) = n in s.ids
function Base.getindex(s::AbstractPerturbationStore, n::Integer)
    j = findfirst(==(n), s.ids)
    isnothing(j) && throw(KeyError(n))
    w = _window_at(s, j)
    data = w.data isa NamedTuple ? _unflatten_spikes(w.data, s.N) : copy(w.data)
    return Dict{String,Any}("interval" => Float32[w.tstart, w.tend], "data" => data)
end
function Base.show(io::IO, s::PerturbationStore)
    print(io, "PerturbationStore(N = $(s.N), $(length(s)) windows, $(s.offsets[end] - 1) columns)")
end
function Base.show(io::IO, s::PerturbationSpikeStore)
    print(io, "PerturbationSpikeStore(N = $(s.N), $(length(s)) windows, $(length(s.times)) spikes)")
end

function _take_id!(s::AbstractPerturbationStore)
    id = s.next_id
    s.next_id = id + 1
    return id
end

function _push_window!(s::PerturbationStore, M::AbstractMatrix, tstart, tend)
    _check_version(s)
    size(M, 1) == s.N ||
        throw(ArgumentError("perturbation window has $(size(M, 1)) rows, the store has $(s.N)"))
    append!(s.data, M)                    # column-major, converted to Float32
    push!(s.offsets, s.offsets[end] + size(M, 2))
    push!(s.tstart, tstart); push!(s.tend, tend)
    push!(s.ids, _take_id!(s))
    return s
end

function _push_window!(s::PerturbationSpikeStore, st::AbstractVector, tstart, tend)
    _check_version(s)
    length(st) == s.N ||
        throw(ArgumentError("perturbation window has $(length(st)) neurons, the store has $(s.N)"))
    n0 = length(s.times)
    for n in eachindex(st)
        for t in st[n]
            push!(s.times, t)
            push!(s.neuron, n)
        end
    end
    push!(s.offsets, s.offsets[end] + length(s.times) - n0)
    push!(s.tstart, tstart); push!(s.tend, tend)
    push!(s.ids, _take_id!(s))
    return s
end

_window_range(s::PerturbationStore, j) = ((s.offsets[j] - 1) * s.N + 1):((s.offsets[j+1] - 1) * s.N)
_window_range(s::PerturbationSpikeStore, j) = s.offsets[j]:(s.offsets[j+1] - 1)

# Window j as (tstart, tend, data); data is a view (no copy).
function _window_at(s::PerturbationStore, j)
    _check_version(s)
    r = _window_range(s, j)
    return (tstart = s.tstart[j], tend = s.tend[j],
            data = reshape(view(s.data, r), s.N, s.offsets[j+1] - s.offsets[j]))
end
function _window_at(s::PerturbationSpikeStore, j)
    _check_version(s)
    r = _window_range(s, j)
    return (tstart = s.tstart[j], tend = s.tend[j],
            data = (neuron = view(s.neuron, r), times = view(s.times, r)))
end

_window_keys(cs::AbstractPerturbationStore) = (_check_version(cs); 1:length(cs))

# (neuron, times) <-> Spiketimes
function _flatten_spikes(st::AbstractVector)
    nspk = sum(length, st; init = 0)
    times = Vector{Float32}(undef, nspk); neuron = Vector{Int32}(undef, nspk)
    k = 0
    for n in eachindex(st), t in st[n]
        k += 1; times[k] = t; neuron[k] = n
    end
    return (neuron = neuron, times = times)
end
function _unflatten_spikes(nt, N)
    st = Spiketimes([Float32[] for _ in 1:N])
    for k in eachindex(nt.times)
        push!(st[nt.neuron[k]], nt.times[k])
    end
    return st
end

# Delete window with number n; returns true if it existed.
function _delete_window!(s::PerturbationStore, n)
    _check_version(s)
    j = findfirst(==(n), s.ids)
    isnothing(j) && return false
    r = _window_range(s, j)
    ncol = s.offsets[j+1] - s.offsets[j]
    deleteat!(s.data, r)
    deleteat!(s.offsets, j)
    for i in j:length(s.offsets)
        s.offsets[i] -= ncol
    end
    deleteat!(s.tstart, j); deleteat!(s.tend, j); deleteat!(s.ids, j)
    return true
end
function _delete_window!(s::PerturbationSpikeStore, n)
    _check_version(s)
    j = findfirst(==(n), s.ids)
    isnothing(j) && return false
    r = _window_range(s, j)
    deleteat!(s.times, r); deleteat!(s.neuron, r)
    deleteat!(s.offsets, j)
    for i in j:length(s.offsets)
        s.offsets[i] -= length(r)
    end
    deleteat!(s.tstart, j); deleteat!(s.tend, j); deleteat!(s.ids, j)
    return true
end

"""
    perturbation_windows(obj, variable, condition)

The perturbation windows stored in `obj.records[:perturbation][variable][condition]`, in
insertion order (the order of the window numbers `n`; not sorted by `tstart`), as a vector of `(tstart, tend, data)` named tuples. `tstart` and
`tend` are `Float32` absolute simulation times in ms. Returns an empty vector when nothing is stored.
Throws an `ArgumentError` if the records are still in the pre-1.9.2 layout (one `Dict` per window):
convert them once with [`compact_perturbation_records!`](@ref).

`data` is
- dense variables: the `N x T` window (a `reshape(view(...))` of the store, no copy; `T`
  can differ between windows, and the samples span `[tstart, tend]` evenly);
- `:fire`: `(neuron, times)`, two equally long views (`Int32` neuron indices, `Float32` spike times in
  ms): the spikes of the window sorted by neuron, then by time. `times[neuron .== i]` are the spike
  times of neuron `i`.

The views alias the store and are read-only; a later flush or `clear_*` invalidates them (their contents may change or be stale), so copy what you keep.

```julia
for (tstart, tend, v) in perturbation_windows(E, :v, "kick")
    @show tstart, tend, size(v)
end
```
"""
function perturbation_windows(obj, variable::Symbol, condition::String)
    cs = _stored_condition(obj, variable, condition)
    (isnothing(cs) || cs isa AbstractDict && isempty(cs)) && return NamedTuple[]
    _check_not_legacy(obj, variable, condition, cs)
    return [_window_at(cs, k) for k in _window_keys(cs)]
end

# ── One-time conversion of the pre-1.9.2 layout ───────────────────────────────

"""
    compact_perturbation_records!(model_or_obj) -> Dict{Symbol,NamedTuple{(:windows,:bytes)}}

The one-time migration of files written before SNNModels 1.9.2 / 2.0.2. Convert, in place, perturbation records written in the legacy layout (one
`Dict("interval", "data")` per window, `records[:perturbation][variable][condition][n]`) into
[`PerturbationStore`](@ref) / [`PerturbationSpikeStore`](@ref). The legacy keys `n` (which may have
gaps) become the window `ids`, `next_id` becomes `maximum(n) + 1`; the insertion order is the order of increasing `n`.
Until a file is converted, every reader, flush and single-window deletion on its legacy records throws an
`ArgumentError` naming the object, variable and condition (legacy and compact records never coexist in
a working model); clearing a whole condition, variable or all records still works on legacy records.
Save the model again after converting.

Accepts a single component, a `NamedTuple` of components, a model (`pop`, `syn`, `stim`) or the
`NamedTuple` returned by `load_data` (its `model`). The conversion goes one object and one
variable at a time: the new stores of a variable are built, checked, replace the dictionaries,
and `GC.gc()` runs before the next variable, so two full copies of a variable are never held.

Safety: before the dictionaries of a (variable, condition) are dropped, the new store is compared
with them bit by bit (interval, data, spike times, numbers `n`); on any difference an error is thrown
and the legacy data of that condition stays as it was. A condition that cannot be stored (windows that
are not `Float32` matrices with the same number of rows, or spike windows with different neuron
counts) throws an `ArgumentError` naming the object, variable, condition and `n`, without
touching it. Conditions already compact are skipped, so running it twice is a no-op.

Returns, per variable, the number of windows converted and the legacy size in bytes
(`Base.summarysize`), summed over the objects; empty when nothing needed conversion.

Run this once on each file written by an older version: it shrinks the file (save again afterwards)
and cuts load time and memory. Files with compact records need SNNModels >= 1.9.2 / >= 2.0.2 in
every process that reads them.
"""
function compact_perturbation_records!(obj)
    summary = Dict{Symbol,NamedTuple{(:windows, :bytes),Tuple{Int,Int}}}()
    for c in _pert_components(obj)
        pert = get(c.records, :perturbation, nothing)
        isnothing(pert) && continue
        for var in collect(keys(pert))
            var_store = pert[var]
            converted = false
            for cond in collect(keys(var_store))
                cs = var_store[cond]
                cs isa AbstractPerturbationStore && continue
                cs isa AbstractDict || throw(ArgumentError(
                    "compact_perturbation_records!: $(_cname(c)), variable $var, condition \"$cond\" holds a value of type $(typeof(cs)); only the legacy Dict-per-window layout is converted"))
                isempty(cs) && continue
                why = _not_compactable(cs, var)
                isnothing(why) ||
                    throw(ArgumentError("compact_perturbation_records!: $(_cname(c)), variable $var, condition \"$cond\": $why"))
                nbytes = Base.summarysize(cs)
                new = _build_compact(cs, var)
                _verify_compact(cs, new, var) ||
                    error("compact_perturbation_records!: $(_cname(c)), variable $var, condition \"$cond\": compact store differs from the legacy windows; legacy data left untouched")
                var_store[cond] = new
                old = get(summary, var, (windows = 0, bytes = 0))
                summary[var] = (windows = old.windows + length(new), bytes = old.bytes + nbytes)
                converted = true
                cs = nothing
            end
            converted && GC.gc()
        end
    end
    return summary
end

_cname(c) = hasproperty(c, :name) ? "$(typeof(c).name.name) \"$(c.name)\"" : string(typeof(c).name.name)

# nothing if the legacy condition can be stored compactly, else the reason (with the offending n).
function _not_compactable(cs::AbstractDict, var::Symbol)
    ks = sort!(collect(keys(cs)))
    k1 = ks[1]
    for k in ks
        e = cs[k]
        (e isa AbstractDict && haskey(e, "interval") && haskey(e, "data") && length(e["interval"]) == 2) ||
            return "window n = $k has no \"interval\"/\"data\" entries"
        d = e["data"]
        if var === :fire
            (d isa AbstractVector && all(x -> x isa AbstractVector{Float32}, d)) ||
                return "window n = $k is not a Spiketimes (Vector of Float32 vectors)"
            length(d) == length(cs[k1]["data"]) ||
                return "window n = $k has $(length(d)) neurons, window n = $k1 has $(length(cs[k1]["data"]))"
        else
            d isa AbstractMatrix{Float32} ||
                return "window n = $k holds $(typeof(d)), expected a Float32 matrix (N, T) ($(ndims(d))-dimensional data is not supported by the compact layout)"
            size(d, 1) == size(cs[k1]["data"], 1) ||
                return "window n = $k has $(size(d, 1)) rows, window n = $k1 has $(size(cs[k1]["data"], 1))"
        end
    end
    return nothing
end

function _build_compact(cs::AbstractDict, var::Symbol)
    ks = sort!(collect(keys(cs)))
    first_data = cs[ks[1]]["data"]
    if var === :fire
        s = PerturbationSpikeStore(length(first_data))
        nspk = sum(k -> sum(length, cs[k]["data"]; init = 0), ks)
        sizehint!(s.times, nspk); sizehint!(s.neuron, nspk)
    else
        s = PerturbationStore(size(first_data, 1))
        sizehint!(s.data, sum(k -> length(cs[k]["data"]), ks))
    end
    for k in ks
        ts, te = cs[k]["interval"]
        _push_window!(s, cs[k]["data"], Float32(ts), Float32(te))
        s.ids[end] = k
    end
    s.next_id = maximum(ks) + 1
    return s
end

_bits(x::AbstractArray{Float32}) = reinterpret(UInt32, vec(collect(x)))
_bits(x::Real) = reinterpret(UInt32, Float32(x))

# Bitwise comparison of a compact store with the legacy dictionaries it replaces.
function _verify_compact(cs::AbstractDict, new::AbstractPerturbationStore, var::Symbol)
    ks = sort!(collect(keys(cs)))
    length(new) == length(ks) && new.ids == ks && new.next_id == maximum(ks) + 1 || return false
    for (j, k) in enumerate(ks)
        e = cs[k]
        w = _window_at(new, j)
        _bits(Float32(e["interval"][1])) == _bits(w.tstart) || return false
        _bits(Float32(e["interval"][2])) == _bits(w.tend) || return false
        if var === :fire
            f = _flatten_spikes(e["data"])
            f.neuron == w.data.neuron || return false
            _bits(f.times) == _bits(w.data.times) || return false
        else
            size(e["data"]) == size(w.data) || return false
            _bits(e["data"]) == _bits(w.data) || return false
        end
    end
    return true
end

_is_component(x) = x isa AbstractPopulation || x isa AbstractStimulus || x isa AbstractConnection
function _pert_components(obj)
    _is_component(obj) && return Any[obj]
    out = Any[]
    if obj isa NamedTuple && haskey(obj, :pop)
        for field in (:pop, :syn, :stim)
            haskey(obj, field) || continue
            append!(out, filter(_is_component, collect(values(getfield(obj, field)))))
        end
    elseif obj isa NamedTuple && haskey(obj, :model)
        return _pert_components(obj.model)
    else
        append!(out, filter(_is_component, collect(values(obj))))
    end
    return out
end

"""
    perturbation_record(obj, variable, condition, interval) -> (data, interval)

Reconstruct a variable trace over `interval` by splicing stored perturbation
windows into the baseline recording.

The baseline is retrieved from `obj`'s normal records, aligned to `interval` via
interpolation. Each stored entry for `condition` is then overlaid at its recorded
`[tstart, tend]` window (in insertion order `n`, not sorted by time: overlapping windows overwrite earlier ones). The result is a continuous
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
    !haskey(obj.records, variable) && return (fill(0.0f0, (0, length(interval))), interval)
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
    _check_not_legacy(obj, variable, condition, cond_store)

    eval_r_vec = collect(eval_r)
    for k in _window_keys(cond_store)   # insertion order (position j)
        w = _window_at(cond_store, k)
        tstart, tend = w.tstart, w.tend
        pert_data    = Matrix{Float32}(w.data)
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
    _check_not_legacy(obj, :fire, condition, cond_store)

    for k in _window_keys(cond_store)
        w = _window_at(cond_store, k)
        tstart, tend = w.tstart, w.tend
        neuron, times = w.data.neuron, w.data.times
        i = 1
        nspk = length(times)
        @inbounds for n in eachindex(result)
            filter!(t -> !(t > tstart && t < tend), result[n])
            while i <= nspk && neuron[i] == n
                t = times[i]
                t > tstart && t < tend && push!(result[n], t)
                i += 1
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

- No `condition`: removes **all** perturbation data (all variables, all conditions); `n` is
  ignored.
- With `condition`: removes all entries for that condition across all variables.
- With `variable` kwarg: restricts deletion to that variable only.
- With both `condition` and `n` kwarg: removes only the window numbered `n` for that
  condition (and variable if specified): its columns (dense) or spikes (`:fire`) are removed from
  the store and the later windows keep their numbers. Window numbers are the `ids` of the store (1, 2, ...
  in order of recording); they are never reused, also after the last window was deleted (the empty
  store stays and keeps counting). Deleting a window of a condition still in the pre-1.9.2 layout throws
  an `ArgumentError` (convert with [`compact_perturbation_records!`](@ref) first); deleting whole
  conditions, variables or everything works on any layout.

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
                _check_not_legacy(obj, var, condition, cond_store)
                cond_store isa AbstractDict || _delete_window!(cond_store, n)
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
| `clear_perturbation_monitor!(obj, :v_s, "silence"; n=2)` | remove only the window with id 2 (ids are never reused) |
"""
function clear_perturbation_monitor!(obj, variable::Symbol)
    clear_perturbation_records!(obj; variable)
end

function clear_perturbation_monitor!(obj, variable::Symbol, condition::String; n = nothing)
    clear_perturbation_records!(obj, condition; variable, n)
end

export perturbation_test,
    perturbation_record,
    perturbation_windows,
    compact_perturbation_records!,
    PerturbationStore,
    PerturbationSpikeStore,
    clear_perturbation_records!,
    clear_perturbation_monitor!
