"""
    SpikeIndex(N)

Ascending `Int32` indices of the neurons that fired in the current step, derived from a
population's `fire` flags so that `forward!`, the event-driven plasticity kernels and the `:fire`
recorder scan the flags at most once per step. While `active` the list is scanned once (`valid`)
and reused; otherwise every `spike_list!` call rescans `fire`.

Three kinds:
- scratch (`SpikeIndex(N)`): one per connection, opened and closed around the connection's step
  by `prepare_spikes!`/`release_spikes!`.
- `owned` (`owned_spike_index`): the list of an integrate-and-fire population, shared by its
  connections and recorder; its window is opened and closed by the simulation loop (see `spike_index`).
- `fed` (`fed_spike_index`): written by its owner every step; `fire` is never scanned.
"""
mutable struct SpikeIndex
    idx::Vector{Int32}
    n::Int
    valid::Bool
    active::Bool
    fed::Bool
    owned::Bool
end
SpikeIndex(N::Integer) = SpikeIndex(Vector{Int32}(undef, N), 0, false, false, false, false)

# Index list of the set flags; skips 8 flags at a time with one UInt64 load.
function _flags_to_list!(idx::Vector{Int32}, fire::Vector{Bool})
    n = 0
    N = length(fire)
    GC.@preserve fire begin
        p = pointer(fire)
        nw = N >>> 3
        @inbounds for w in 0:nw-1
            if unsafe_load(Ptr{UInt64}(p), w + 1) != 0
                base = w << 3
                for b in 1:8
                    if fire[base+b]
                        n += 1
                        idx[n] = base + b
                    end
                end
            end
        end
        @inbounds for k in (nw<<3)+1:N
            if fire[k]
                n += 1
                idx[n] = k
            end
        end
    end
    return n
end

function _flags_to_list!(idx::Vector{Int32}, fire::AbstractVector{Bool})
    n = 0
    @inbounds for k in eachindex(fire)
        if fire[k]
            n += 1
            idx[n] = k
        end
    end
    return n
end

"""
    spike_list!(s::SpikeIndex, fire)

Indices (ascending, `Int32`) of the `true` entries of `fire`, as a view of the buffer of `s`.
Not thread-safe on a shared (owned) list: call on the stepping thread, never inside a `foreach_chunk` body.
"""
@inline function spike_list!(s::SpikeIndex, fire::AbstractVector{Bool})
    s.fed && return @inbounds view(s.idx, 1:s.n)
    if !(s.active && s.valid)
        length(s.idx) == length(fire) || resize!(s.idx, length(fire))
        s.n = _flags_to_list!(s.idx, fire)
        s.valid = s.active
    end
    return @inbounds view(s.idx, 1:s.n)
end

"""
    fed_spike_index(N)
    owned_spike_index(N)

`SpikeIndex` of capacity `N` that is fed by its owner, or owned by a population (see `SpikeIndex`).
"""
fed_spike_index(N::Integer) = SpikeIndex(Vector{Int32}(undef, N), 0, false, false, true, false)

owned_spike_index(N::Integer) = SpikeIndex(Vector{Int32}(undef, N), 0, false, false, false, true)

# per-connection window of a scratch list; fed and owned lists are left to their owner
_arm!(s::SpikeIndex) = ((s.fed || s.owned) || (s.active = true; s.valid = false); nothing)
_disarm!(s::SpikeIndex) = ((s.fed || s.owned) || (s.active = false; s.valid = false); nothing)

# Window of an owned list (see `spike_index`): `_publish_spikes!` opens it after `integrate!` (`active`, not `valid`),
# `_retire_spikes!` closes it; no-ops for populations without an owned list.
_publish_spikes!(::AbstractPopulation) = nothing
_retire_spikes!(::AbstractPopulation) = nothing
function _retire_all_spikes!(P)
    for p in P
        _retire_spikes!(p)
    end
    return nothing
end

"""
    TouchList(N)

Indices of the neurons whose input was written by list-aware stimuli since the last `integrate!`, in write order,
possibly repeated. `n` counts the writes; `n > length(idx)` means overflow and the population falls back to its dense update.
"""
mutable struct TouchList
    idx::Vector{Int32}
    n::Int
end
TouchList(N::Integer) = TouchList(Vector{Int32}(undef, N), 0)

@inline function _touch!(b::TouchList, i::Integer)
    n = b.n + 1
    b.n = n
    n <= length(b.idx) && (@inbounds b.idx[n] = i)
    return nothing
end
@inline _touch!(::Nothing, i::Integer) = nothing

# append to a list that grows when needed (n never exceeds length(idx))
@inline function _push!(b::TouchList, i::Integer)
    n = b.n + 1
    n > length(b.idx) && resize!(b.idx, max(2n, 8))
    @inbounds b.idx[n] = i
    b.n = n
    return nothing
end

"""
    attach_driver!(post::AbstractPopulation)

Called by the constructor of a stimulus that logs the neurons it writes. Returns the `TouchList` in which the
stimulus logs (`_touch!`) every neuron of `post` whose input it increments, or `nothing` if `post` has no sparse
update (the default). `post` (`Identity`) uses its sparse update only while every writer of its input is such a driver.
"""
attach_driver!(::AbstractPopulation) = nothing

"""
    spike_index(pop::AbstractPopulation)

`SpikeIndex` used by a connection that reads the spikes of `pop`: a new scratch list by default, else the list `pop` keeps:
- the integrate-and-fire family (`IF`, `AdEx`, `ExtendedIF`, `HetRec`, `Tripod`, `BallAndStick`, `HH`, `IZ`, `MorrisLecar`)
  owns one list (`pop.spikes`). `sim!`/`train!` open it after `integrate!` and close it at the end of the step; inside
  this window the first consumer scans `fire` and later consumers reuse the list. Contract: nothing may write `pop.fire`
  between `integrate!` of `pop` and the end of the step. Outside the window every call rescans `fire`. A failed step is
  a fatal error and the run is discarded: nothing closes the window after an exception (call
  `SNNModels._retire_all_spikes!(P)` before calling `forward!`, `plasticity!` or `record!` by hand); the next step
  publishes afresh.
- `Identity` driven only by list-aware stimuli, and the Poisson populations in their sparse regime, return their fed list.
"""
spike_index(p::AbstractPopulation) = SpikeIndex(length(p.fire))
