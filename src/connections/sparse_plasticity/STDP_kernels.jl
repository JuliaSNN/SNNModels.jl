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

So in 1-2 every trace is read BEFORE the spikes of step n are added: the value read
at t_n is  sum_{m < n} exp(-(t_n - t_m)/τ)  over the earlier spikes t_m of that
neuron. A pre and a post spike in the same step do not interact with each other
(their earlier history does). This is the ordering of Auryn's System::run
(evolve neurons -> propagate = forward + plasticity -> evolve_traces) and of Brian2
`on_pre`/`on_post` code that updates `w` before incrementing the trace.

Cost per step: O(Npre + Npost) for the spike scan and trace decay (one multiply per
neuron, no exp), plus O(spikes x fan-out) for the weight updates. Only touched
weights are clamped. Single-threaded (parallelisation is deferred: splitting
pre-spikes over threads would race on shared postsynaptic rows).
=#

"""
    _decay!(x, factor)

Multiply every trace by the exact one-step decay factor `exp(-dt/τ)`.
"""
@inline function _decay!(x::AbstractVector{Float32}, factor::Float32)
    @inbounds @simd for k in eachindex(x)
        x[k] *= factor
    end
end

"""
    _spike_increment!(x, last, fire, t, a = 1f0)

Add `a` to the trace of every neuron that fired this step and store its spike time.
"""
@inline function _spike_increment!(x, last, fire, t::Float32, a::Float32 = 1.0f0)
    @inbounds for k in eachindex(fire)
        if fire[k]
            x[k] += a
            last[k] = t
        end
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
