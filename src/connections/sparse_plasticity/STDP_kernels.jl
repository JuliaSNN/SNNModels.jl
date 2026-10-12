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

Lazy traces (step 4 makes no pass over the neurons): a trace is stored as `x` with one scalar
`scale` per trace type (all pre, or all post neurons), its value being `x[k] * scale`. A step of
decay multiplies `scale` by the one-step factor; an increment of 1 adds `1/scale` to `x[k]`. When
`scale` falls below `_SCALE_MIN` the stored values absorb it and `scale` is reset to 1 (`_rescale!`).
`sync_traces!` does this on demand; `trace_values` and `plasticity_value` read values without
changing the state. The stored vectors (`syn.LTPVars.tpre`) are not the trace values.

So in 1-2 every trace is read BEFORE the spikes of step n are added: the value read
at t_n is  sum_{m < n} exp(-(t_n - t_m)/τ)  over the earlier spikes t_m of that
neuron. A pre and a post spike in the same step do not interact with each other
(their earlier history does). This is the ordering of Auryn's System::run
(evolve neurons -> propagate = forward + plasticity -> evolve_traces) and of Brian2
`on_pre`/`on_post` code that updates `w` before incrementing the trace.

Cost per step: O(spikes x fan-out) for the weight updates and O(spikes) for the trace
increments, plus an O(Npre + Npost) rescale every few thousand steps. Only touched weights are
clamped. The weight passes are threaded by postsynaptic range (see below); trace updates are serial.
=#

# `scale` is kept in [_SCALE_MIN, 1]: stored values stay within 1/_SCALE_MIN of the trace, far from the Float32 limit.
const _SCALE_MIN = 1.0e-6

# `x .*= s`, flushing results below `floatmin` to zero (subnormal arithmetic is slow; such a trace changes no weight)
@inline function _rescale!(x::AbstractVector{T}, s::T) where {T<:AbstractFloat}
    fmin = floatmin(T)
    @inbounds @simd for k in eachindex(x)
        y = x[k] * s
        x[k] = ifelse(abs(y) < fmin, zero(T), y)
    end
end

# Decay trace type `k` by one step: `scale[k] *= factor`, in Float64 so the product chain is not rounded to Float32
# every step; below `_SCALE_MIN` the stored values absorb it.
@inline function _advance_scale!(scale::Vector{Float64}, k::Int, x::AbstractVector{T}, factor::Real) where {T<:AbstractFloat}
    s = scale[k] * Float64(factor)
    if s < _SCALE_MIN
        _rescale!(x, T(s))
        s = 1.0
    end
    scale[k] = s
    return nothing
end

# factor from stored trace `x` (type `k`) to its value at the start of this step, in the precision of `x`
@inline _read_scale(scale::Vector{Float64}, k::Int, x::AbstractVector{T}) where {T<:AbstractFloat} = T(scale[k])

# stored amount that adds 1 to the value of a trace `x` of type `k`
@inline _write_inc(scale::Vector{Float64}, k::Int, x::AbstractVector{T}) where {T<:AbstractFloat} = T(1.0 / scale[k])

# Whether `cache` was computed for another `dt` or `τ...` (always at the first call). The caller then runs
# `sync_traces!` first: the pending `scale` was accumulated with the old factors.
@inline function _decay_stale(cache::Vector{Float64}, dt, τ::Vararg{Real,N}) where {N}
    hit = cache[1] == dt
    for k in 1:N
        hit &= cache[1+k] == τ[k]
    end
    return !hit
end

# One-step factors `exp(-dt/τ)` in the precision of `dt`, cached in `cache` (`1 + 2N` entries: dt, the τs, the
# factors) and recomputed only when `dt` or a τ changes.
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

# As `_decay_factors!` with the forward-Euler factor `1 - dt/τ` (in Float64), for the rules whose traces are Euler-integrated.
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

# same, for the index list of the neurons that fired
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

Same, for the index list of the neurons that fired.
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

# same, for the index list of the neurons that fired
@inline function _post_spike_pass!(f::F, W, rowptr, index, J, spikes::AbstractVector{Int32}, Wmin, Wmax) where {F}
    @inbounds for i in spikes
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] = clamp(W[s] + f(W[s], i, J[s]), Wmin, Wmax)
        end
    end
    return nothing
end

# pre- and postsynaptic passes without clamping, for the rules that clamp after both passes (`_clamp_touched!`)
@inline function _pre_spike_add!(f::F, W, colptr, I, spikes::AbstractVector{Int32}) where {F}
    @inbounds for j in spikes
        for s = colptr[j]:(colptr[j+1]-1)
            W[s] += f(W[s], I[s], j)
        end
    end
    return nothing
end

@inline function _post_spike_add!(f::F, W, rowptr, index, J, spikes::AbstractVector{Int32}) where {F}
    @inbounds for i in spikes
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] += f(W[s], i, J[s])
        end
    end
    return nothing
end

# clamp only the synapses of the neurons that fired this step (index lists or flags)
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


#=
Threaded weight passes (partition by postsynaptic range)
--------------------------------------------------------
Chunk `c` of the postsynaptic population (see `ChunkPtr`) owns the entries whose post lies in its range and runs both passes on
them: pre pass over the sub-rows `ptr[c, j]:ptr[c+1, j]-1`, then post pass over the postsynaptic spikes in its range. Valid
because the updates read only traces that the passes do not modify; each weight sees the same operations in the same order as
serially, so results are bit-identical for any chunking. Trace updates stay serial.
=#
_chunkptr(::AbstractSparseSynapse) = nothing
@inline _syn_chunk_ptr(::Nothing, args...) = nothing

@inline function _pre_spike_pass_c!(f::F, W, I, ptr, ch, spikes, Wmin, Wmax) where {F}
    @inbounds for j in spikes
        for s = ptr[ch, j]:(ptr[ch+1, j]-1)
            W[s] = clamp(W[s] + f(W[s], I[s], j), Wmin, Wmax)
        end
    end
    return nothing
end
@inline function _post_spike_pass_c!(f::F, W, rowptr, index, J, spikes, lo, hi, Wmin, Wmax) where {F}
    @inbounds for i in spikes
        (lo <= i <= hi) || continue
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] = clamp(W[s] + f(W[s], i, J[s]), Wmin, Wmax)
        end
    end
    return nothing
end
@inline function _pre_spike_add_c!(f::F, W, I, ptr, ch, spikes) where {F}
    @inbounds for j in spikes
        for s = ptr[ch, j]:(ptr[ch+1, j]-1)
            W[s] += f(W[s], I[s], j)
        end
    end
    return nothing
end
@inline function _post_spike_add_c!(f::F, W, rowptr, index, J, spikes, lo, hi) where {F}
    @inbounds for i in spikes
        (lo <= i <= hi) || continue
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] += f(W[s], i, J[s])
        end
    end
    return nothing
end
@inline function _clamp_touched_c!(W, I, ptr, ch, rowptr, index, spikes_pre, spikes_post, lo, hi, Wmin, Wmax)
    @inbounds for j in spikes_pre
        for s = ptr[ch, j]:(ptr[ch+1, j]-1)
            W[s] = clamp(W[s], Wmin, Wmax)
        end
    end
    @inbounds for i in spikes_post
        (lo <= i <= hi) || continue
        for st = rowptr[i]:(rowptr[i+1]-1)
            s = index[st]
            W[s] = clamp(W[s], Wmin, Wmax)
        end
    end
    return nothing
end

"""
    _stdp_passes!(c, f_pre, f_post, spikes_pre, spikes_post, Wmin, Wmax)

Clamped pre-spike pass, then clamped post-spike pass, threaded by postsynaptic range when there is enough work.
"""
@inline function _stdp_passes!(c, f_pre::F1, f_post::F2, spikes_pre, spikes_post, Wmin, Wmax) where {F1,F2}
    @unpack rowptr, colptr, I, J, index, W = c
    Npost = length(c.fireI)
    ptr = _syn_chunk_ptr(_chunkptr(c), Npost, colptr, I, spikes_pre, _spike_ops(spikes_post, rowptr))
    if ptr === nothing
        _pre_spike_pass!(f_pre, W, colptr, I, spikes_pre, Wmin, Wmax)
        _post_spike_pass!(f_post, W, rowptr, index, J, spikes_post, Wmin, Wmax)
    else
        foreach_chunk_c(Npost) do ch, rng
            _pre_spike_pass_c!(f_pre, W, I, ptr, ch, spikes_pre, Wmin, Wmax)
            _post_spike_pass_c!(f_post, W, rowptr, index, J, spikes_post, first(rng), last(rng), Wmin, Wmax)
        end
    end
    return nothing
end

"""
    _add_passes!(c, f_pre, f_post, spikes_pre, spikes_post, Wmin, Wmax)

`_pre_spike_add!`, `_post_spike_add!` and `_clamp_touched!` in this order (the rules that clamp after both passes), threaded like `_stdp_passes!`.
"""
@inline function _add_passes!(c, f_pre::F1, f_post::F2, spikes_pre, spikes_post, Wmin, Wmax) where {F1,F2}
    @unpack rowptr, colptr, I, J, index, W = c
    Npost = length(c.fireI)
    ptr = _syn_chunk_ptr(_chunkptr(c), Npost, colptr, I, spikes_pre, _spike_ops(spikes_post, rowptr))
    if ptr === nothing
        _pre_spike_add!(f_pre, W, colptr, I, spikes_pre)
        _post_spike_add!(f_post, W, rowptr, index, J, spikes_post)
        _clamp_touched!(W, colptr, rowptr, index, spikes_pre, spikes_post, Wmin, Wmax)
    else
        foreach_chunk_c(Npost) do ch, rng
            _pre_spike_add_c!(f_pre, W, I, ptr, ch, spikes_pre)
            _post_spike_add_c!(f_post, W, rowptr, index, J, spikes_post, first(rng), last(rng))
            _clamp_touched_c!(W, I, ptr, ch, rowptr, index, spikes_pre, spikes_post, first(rng), last(rng), Wmin, Wmax)
        end
    end
    return nothing
end

"""
    _pre_pass!(c, f, spikes_pre, Wmin, Wmax)
    _post_pass!(c, f, spikes_post, Wmin, Wmax)

One clamped pass alone (rules with trace updates between the two passes, e.g. `iSTDPRate`), threaded like `_stdp_passes!`.
"""
@inline function _pre_pass!(c, f::F, spikes_pre, Wmin, Wmax) where {F}
    @unpack colptr, I, W = c
    Npost = length(c.fireI)
    ptr = _syn_chunk_ptr(_chunkptr(c), Npost, colptr, I, spikes_pre)
    if ptr === nothing
        _pre_spike_pass!(f, W, colptr, I, spikes_pre, Wmin, Wmax)
    else
        foreach_chunk_c(Npost) do ch, rng
            _pre_spike_pass_c!(f, W, I, ptr, ch, spikes_pre, Wmin, Wmax)
        end
    end
    return nothing
end
@inline function _post_pass!(c, f::F, spikes_post, Wmin, Wmax) where {F}
    @unpack rowptr, colptr, I, J, index, W = c
    Npost = length(c.fireI)
    ptr = _syn_chunk_ptr(_chunkptr(c), Npost, colptr, I, (), _spike_ops(spikes_post, rowptr))
    if ptr === nothing
        _post_spike_pass!(f, W, rowptr, index, J, spikes_post, Wmin, Wmax)
    else
        foreach_chunk_c(Npost) do ch, rng
            _post_spike_pass_c!(f, W, rowptr, index, J, spikes_post, first(rng), last(rng), Wmin, Wmax)
        end
    end
    return nothing
end
