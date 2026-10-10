"""
    SpikeIndex(N)

Scratch list of the indices of the neurons that fired in the current step, built from the
`Bool` firing vector of a population by one scan (`spike_list!`). Populations and stimuli keep
writing `fire`; the list is a derived view of it used by `forward!` and the event-driven
plasticity kernels, so that the spike flags are scanned once per step instead of once per
consumer. `active` is set by `prepare_spikes!` at the start of the step of a connection and
cleared by `release_spikes!` at its end: only while active is the list reused (`valid`);
otherwise every call scans `fire` again, so direct calls of `forward!` or `plasticity!` with a
modified `fire` always see it.

A list can also be `fed`: its owner (a population that learns its spikes from a stimulus, see
`Identity`) writes `idx[1:n]` itself every step, in ascending order, and `spike_list!` returns it
without scanning `fire` at all. The connections that read the population share the owner's
`SpikeIndex`; `prepare_spikes!`/`release_spikes!` leave a fed list alone.
"""
mutable struct SpikeIndex
    idx::Vector{Int32}
    n::Int
    valid::Bool
    active::Bool
    fed::Bool
end
SpikeIndex(N::Integer) = SpikeIndex(Vector{Int32}(undef, N), 0, false, false, false)

# Index list of the set flags, scanning 8 flags per UInt64 load (most steps have no spike in most words).
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
Scans `fire` unless `s` is fed by its owner or is active and already holds the list of this step.
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

`SpikeIndex` of capacity `N` whose owner writes the ascending list `idx[1:n]` of the firing neurons itself
every step (`fed`), as the `Identity` population and the Poisson stimuli do.
"""
fed_spike_index(N::Integer) = SpikeIndex(Vector{Int32}(undef, N), 0, false, false, true)

_arm!(s::SpikeIndex) = (s.fed || (s.active = true; s.valid = false); nothing)
_disarm!(s::SpikeIndex) = (s.fed || (s.active = false; s.valid = false); nothing)

"""
    TouchList(N)

Inbox of a population whose input is written by list-aware stimuli: the indices of the neurons
whose input variable was written since the last `integrate!`, in write order and possibly with
repeats. `n` counts the writes; when `n > length(idx)` the list overflowed (more writes than `N`)
and the population falls back to its dense update. See `attach_driver!`.
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

Called by the constructor of a stimulus that records the neurons it writes (`SpikeTimeStimulus`).
Returns the `TouchList` in which the stimulus must log the index of every neuron of `post` whose
input it increments (`_touch!`), or `nothing` if `post` has no sparse update (the default), in
which case the stimulus just writes the input variable and `post` integrates densely.

A population with a sparse update (`Identity`) keeps count of the components that write its
input variable (`synaptic_target` calls) and of the drivers that log their writes; it updates only
the logged neurons while the two numbers agree. Any other writer (a connection, a stimulus without
a log) switches it back to the dense update.
"""
attach_driver!(::AbstractPopulation) = nothing

"""
    spike_index(pop::AbstractPopulation)

`SpikeIndex` that a connection reading the spikes of `pop` uses: a new scratch list, except for
populations that keep their own list up to date (`Identity` with list-aware stimuli), which
return it so that the connections share it and never scan `pop.fire`.
"""
spike_index(p::AbstractPopulation) = SpikeIndex(length(p.fire))
