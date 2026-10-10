#=
Event-driven STDP kernels (Auryn style)
=======================================

Shared loops used by the trace-based STDP rules (STDPGerstner, STDPConfavreux2025,
STDPWeightDependent, STDPTriplet, STDPMexicanHat). The algorithm follows Auryn's
`STDPConnection` / `TripletConnection` (Zenke & Gerstner 2014):

Data layout of `AbstractSparseSynapse` (see `dsparse`):
  - by presynaptic neuron j (CSC):  synapses s in colptr[j]:colptr[j+1]-1, post = I[s]
  - by postsynaptic neuron i:       st in rowptr[i]:rowptr[i+1]-1, s = index[st], pre = J[s]
`fireJ` (pre) and `fireI` (post) hold the spikes emitted in the current step.

Ordering convention, for one call of `plasticity!` at step n (time t_n):
  1. pre-spike pass  : for every j with fireJ[j], for every outgoing synapse s:
                       W[s] += f_pre(W[s], i, j); clamp W[s] to [Wmin, Wmax]
  2. post-spike pass : for every i with fireI[i], for every incoming synapse s:
                       W[s] += f_post(W[s], i, j); clamp W[s] to [Wmin, Wmax]
  3. trace increment : x[k] += 1 for every neuron k that fired at t_n
  4. trace decay     : x[k] *= exp(-dt/τ) for every neuron (exact, factor precomputed)

Step 4 is lazy: no pass over the neurons. A trace is stored as `x` together with one
scalar `scale` per trace type (all pre or all post neurons share it), the value being
`x[k] * scale`. Decaying by one step multiplies `scale` by the one-step factor; an increment
of 1 adds `1/scale` to `x[k]`; reads multiply by `scale`. Only when `scale` has fallen below
`_SCALE_MIN` the stored values are multiplied by it and `scale` is reset to 1 (`_rescale!`,
one pass every `log(_SCALE_MIN)/log(factor)` steps, about 2000 steps for τ = 20 ms,
dt = 0.125 ms). `sync_traces!` does that pass on demand, `trace_values` returns the
current values without changing the state. Direct reads of the stored vectors (`syn.LTPVars.tpre`)
are NOT the trace values; the recorder reads through `plasticity_value`.

So in 1-2 every trace is read BEFORE the spikes of step n are added: the value read
at t_n is  sum_{m < n} exp(-(t_n - t_m)/τ)  over the earlier spikes t_m of that
neuron. A pre and a post spike in the same step do not interact with each other
(their earlier history does). This is the ordering of Auryn's System::run
(evolve neurons -> propagate = forward + plasticity -> evolve_traces) and of Brian2
`on_pre`/`on_post` code that updates `w` before incrementing the trace.

Cost per step: O(spikes x fan-out) for the weight updates and O(spikes) for the trace
increments, plus the decay: O(1) per step plus an O(Npre + Npost) rescale every few thousand steps. Only touched
weights are clamped. Single-threaded (parallelisation is deferred: splitting
pre-spikes over threads would race on shared postsynaptic rows).
=#

# Stored traces are multiplied by `scale` to give the value; `scale` is kept in [_SCALE_MIN, 1].
# Larger stored values (up to 1/_SCALE_MIN times the trace) are far from the Float32 limit.
const _SCALE_MIN = 1.0e-6

"""
    _rescale!(x, s)

Multiply every stored trace by `s` (`x .*= s`). Values below `floatmin(T)` (`T = eltype(x)`) are set to
zero: subnormal products are about 100 times slower, and a trace of that size changes no weight
(it is below 1e-38 for `Float32`, 1e-308 for `Float64`).
"""
@inline function _rescale!(x::AbstractVector{T}, s::T) where {T<:AbstractFloat}
    fmin = floatmin(T)
    @inbounds @simd for k in eachindex(x)
        y = x[k] * s
        x[k] = ifelse(abs(y) < fmin, zero(T), y)
    end
end

"""
    _advance_scale!(scale, k, x, factor)

Decay trace type `k` by one step: `scale[k] *= factor` (in `Float64`, so that the chain of the
products of the `Float32` factor is not rounded at every step). When `scale[k]` falls below
`_SCALE_MIN` the stored values `x` absorb it (`_rescale!`) and `scale[k]` is reset to 1.
"""
@inline function _advance_scale!(scale::Vector{Float64}, k::Int, x::AbstractVector{T}, factor::Real) where {T<:AbstractFloat}
    s = scale[k] * Float64(factor)
    if s < _SCALE_MIN
        _rescale!(x, T(s))
        s = 1.0
    end
    scale[k] = s
    return nothing
end

"""
    _read_scale(scale, k, x) -> eltype(x)

Factor that turns a stored trace `x` of type `k` into its value at the start of this step (the value
after the decay of the previous step), the one the weight updates read, in the precision of `x`.
"""
@inline _read_scale(scale::Vector{Float64}, k::Int, x::AbstractVector{T}) where {T<:AbstractFloat} = T(scale[k])

"""
    _write_inc(scale, k, x) -> eltype(x)

Stored amount that adds 1 to the value of a trace `x` of type `k`: `1/scale[k]`.
"""
@inline _write_inc(scale::Vector{Float64}, k::Int, x::AbstractVector{T}) where {T<:AbstractFloat} = T(1.0 / scale[k])

"""
    _decay_stale(cache, dt, τ...) -> Bool

`true` if the factors in `cache` were computed for other `dt` or `τ...` (always at the first call).
A rule that finds its cache stale calls `sync_traces!` before the new factors are used, because
the pending `scale` of its traces was accumulated with the old ones.
"""
@inline function _decay_stale(cache::Vector{Float64}, dt, τ::Vararg{Real,N}) where {N}
    hit = cache[1] == dt
    for k in 1:N
        hit &= cache[1+k] == τ[k]
    end
    return !hit
end

"""
    _decay_factors!(cache, dt, τ...)

The one-step decay factors `exp(-dt/τ)` of the time constants `τ...`, in the precision of `dt` (`Float32(exp(-dt/τ))` for
a `Float32` step). The factors are
stored in `cache` (`NaN` initially: `1 + 2N` entries for `N` time constants) together with the
`dt` and `τ` they were computed for, and `exp` is evaluated only when one of them changes.
"""
@inline function _decay_factors!(cache::Vector{Float64}, dt::T, τ::Vararg{Real,N}) where {N,T<:AbstractFloat}
    if _decay_stale(cache, dt, τ...)
        cache[1] = dt
        for k in 1:N
            cache[1+k] = τ[k]
            cache[1+N+k] = T(exp(-dt / τ[k]))
        end
    end
    return ntuple(k -> T(cache[1+N+k]), Val(N))
end

"""
    _euler_factors!(cache, dt, τ...)

Like `_decay_factors!` for the rules whose traces are integrated by forward Euler
(`x += -dt x / τ`): the one-step factor is `1 - dt/τ` (kept in `Float64`, not rounded to `Float32`: the
scale chain is `Float64`) instead of `exp(-dt/τ)`. A trace decayed lazily with this factor follows the
Euler recursion, so these rules keep their numerics when their traces become lazy. The stale test and
the cache layout (`1 + 2N` entries) are those of `_decay_factors!`.
"""
@inline function _euler_factors!(cache::Vector{Float64}, dt, τ::Vararg{Real,N}) where {N}
    if _decay_stale(cache, dt, τ...)
        cache[1] = dt
        for k in 1:N
            cache[1+k] = τ[k]
            cache[1+N+k] = 1 - Float64(dt) / Float64(τ[k])
        end
    end
    return ntuple(k -> cache[1+N+k], Val(N))
end

"""
    _spike_increment!(x, last, fire, t, a = 1)

Add `a` to the trace of every neuron that fired this step and store its spike time.
"""
@inline function _spike_increment!(x, last, fire, t::Real, a::Real = one(eltype(x)))
    @inbounds for k in eachindex(fire)
        if fire[k]
            x[k] += a
            last[k] = t
        end
    end
end

"""
    _spike_increment!(x, last, spikes::AbstractVector{Int32}, t, a = 1)

Same as above for the index list of the neurons that fired (`presynaptic_spikes`,
`postsynaptic_spikes`): no scan of the whole spike vector.
"""
@inline function _spike_increment!(x, last, spikes::AbstractVector{Int32}, t::Real, a::Real = one(eltype(x)))
    @inbounds for k in spikes
        x[k] += a
        last[k] = t
    end
end

"""
    _initial_clamp!(W, initialized, Wmin, Wmax)

Clamp all weights once, on the first plasticity step after the variables were
created. The clock-driven implementation clamped every weight at every step; the
event-driven one clamps only touched weights, so the one-off clamp keeps the two
identical for initial weights outside [Wmin, Wmax].
"""
@inline function _initial_clamp!(W, initialized, Wmin, Wmax)
    initialized[1] && return nothing
    @inbounds @simd for s in eachindex(W)
        W[s] = clamp(W[s], Wmin, Wmax)
    end
    initialized[1] = true
    return nothing
end

"""
    _pre_spike_pass!(f, W, colptr, I, fireJ, Wmin, Wmax)

For each presynaptic spike j and each outgoing synapse s (post i = I[s]):
`W[s] = clamp(W[s] + f(W[s], i, j), Wmin, Wmax)`.
"""
@inline function _pre_spike_pass!(f::F, W, colptr, I, fireJ, Wmin, Wmax) where {F}
    @inbounds for j in eachindex(fireJ)
        fireJ[j] || continue
        for s = colptr[j]:(colptr[j+1]-1)
            W[s] = clamp(W[s] + f(W[s], I[s], j), Wmin, Wmax)
        end
    end
    return nothing
end

"""
    _pre_spike_pass!(f, W, colptr, I, spikes::AbstractVector{Int32}, Wmin, Wmax)

Same as above, for the index list `spikes` of the presynaptic neurons that fired.
"""
@inline function _pre_spike_pass!(f::F, W, colptr, I, spikes::AbstractVector{Int32}, Wmin, Wmax) where {F}
    @inbounds for j in spikes
        for s = colptr[j]:(colptr[j+1]-1)
            W[s] = clamp(W[s] + f(W[s], I[s], j), Wmin, Wmax)
        end
    end
    return nothing
end

"""
    _post_spike_pass!(f, W, rowptr, index, J, fireI, Wmin, Wmax)

For each postsynaptic spike i and each incoming synapse s = index[st] (pre j = J[s]):
`W[s] = clamp(W[s] + f(W[s], i, j), Wmin, Wmax)`.
"""
@inline function _post_spike_pass!(f::F, W, rowptr, index, J, fireI, Wmin, Wmax) where {F}
    @inbounds for i in eachindex(fireI)
        fireI[i] || continue
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] = clamp(W[s] + f(W[s], i, J[s]), Wmin, Wmax)
        end
    end
    return nothing
end

"""
    _post_spike_pass!(f, W, rowptr, index, J, spikes::AbstractVector{Int32}, Wmin, Wmax)

Same as above, for the index list `spikes` of the postsynaptic neurons that fired.
"""
@inline function _post_spike_pass!(f::F, W, rowptr, index, J, spikes::AbstractVector{Int32}, Wmin, Wmax) where {F}
    @inbounds for i in spikes
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] = clamp(W[s] + f(W[s], i, J[s]), Wmin, Wmax)
        end
    end
    return nothing
end

"""
    _pre_spike_add!(f, W, colptr, I, spikes::AbstractVector{Int32})

Presynaptic pass without clamping: `W[s] += f(W[s], I[s], j)` for each spike `j` and each outgoing
synapse `s`. For the rules that clamp after both passes (`_clamp_touched!`).
"""
@inline function _pre_spike_add!(f::F, W, colptr, I, spikes::AbstractVector{Int32}) where {F}
    @inbounds for j in spikes
        for s = colptr[j]:(colptr[j+1]-1)
            W[s] += f(W[s], I[s], j)
        end
    end
    return nothing
end

"""
    _post_spike_add!(f, W, rowptr, index, J, spikes::AbstractVector{Int32})

Postsynaptic pass without clamping: `W[s] += f(W[s], i, J[s])` for each spike `i` and each incoming
synapse `s = index[st]`.
"""
@inline function _post_spike_add!(f::F, W, rowptr, index, J, spikes::AbstractVector{Int32}) where {F}
    @inbounds for i in spikes
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] += f(W[s], i, J[s])
        end
    end
    return nothing
end

"""
    _clamp_touched!(W, colptr, rowptr, index, spikes_pre::AbstractVector{Int32}, spikes_post::AbstractVector{Int32}, Wmin, Wmax)

Same as the method with flags, for the index lists of the neurons that fired.
"""
@inline function _clamp_touched!(W, colptr, rowptr, index, spikes_pre::AbstractVector{Int32},
                                 spikes_post::AbstractVector{Int32}, Wmin, Wmax)
    @inbounds for j in spikes_pre
        for s = colptr[j]:(colptr[j+1]-1)
            W[s] = clamp(W[s], Wmin, Wmax)
        end
    end
    @inbounds for i in spikes_post
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] = clamp(W[s], Wmin, Wmax)
        end
    end
    return nothing
end

"""
    _clamp_touched!(W, colptr, rowptr, index, fireJ, fireI, Wmin, Wmax)

Clamp only the synapses of neurons that fired this step (used by rules that apply
both passes before clamping, e.g. STDPMexicanHat and the structured rules).
"""
@inline function _clamp_touched!(W, colptr, rowptr, index, fireJ, fireI, Wmin, Wmax)
    @inbounds for j in eachindex(fireJ)
        fireJ[j] || continue
        for s = colptr[j]:(colptr[j+1]-1)
            W[s] = clamp(W[s], Wmin, Wmax)
        end
    end
    @inbounds for i in eachindex(fireI)
        fireI[i] || continue
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] = clamp(W[s], Wmin, Wmax)
        end
    end
    return nothing
end
