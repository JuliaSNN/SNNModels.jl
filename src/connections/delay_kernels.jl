# Constant-delay kernels of the delayed `forward!`: the delay is stored once for all synapses (`delaysteps` of length 1, see
# `SpikingSynapse`), so `forward!` passes it as an integer and `_forward_delay!` dispatches here. The arrival column `col` (0-based
# ring slot) is the same for every event of the step, so the scatter needs neither the per-synapse delay nor the slot wrap.
# Invariants (bit-identical to the general loop): the increment is `_wρ(W, ρ, s)` (`W[s] * ρ[s]`, or `W[s]` without stored
# efficacy) at emission, summed in the column in emission order (spike list, then CSC order); no `@fastmath` (no FMA contraction of
# `buffer + W * ρ`); the threaded kernel touches only the rows `rng` of the buffer and of `g`. `ivdep` is valid because the rows
# `I[s]` of one CSC column are distinct (the vectorised scatter has no conflicting lanes); measured 16 % faster than `@simd` alone.

# common delay in steps (`-1`: per-synapse delays, empty connection, or not yet computed by `_delay_steps!`)
_dconst(param::SpikingSynapseDelayParameter) = length(param.delaysteps) == 1 ? Int(param.delaysteps[1]) : -1

function _forward_delay!(g, W, ρ, I, colptr, ptr, spikes, Npost, d::Integer, buffer, pos, nslots)
    col = pos + d >= nslots ? pos + d - nslots : pos + d
    if ptr === nothing
        _delay_scatter_const!(buffer, W, ρ, I, colptr, spikes, col)
        @inbounds @simd for i ∈ axes(buffer, 1)   # arrivals of this step
            g[i] += buffer[i, pos+1]
            buffer[i, pos+1] = 0
        end
    else
        foreach_chunk_c(Npost) do ch, rng
            _delay_chunk_const!(g, W, ρ, I, ptr, ch, rng, spikes, buffer, pos, col)
        end
    end
    return nothing
end

# serial scatter of the spikes of this step into ring column `col`
@inline function _delay_scatter_const!(buffer, W, ρ, I, colptr, spikes, col)
    off = col * size(buffer, 1)
    @inbounds for j ∈ spikes
        @simd ivdep for s ∈ colptr[j]:(colptr[j+1]-1)
            buffer[off+I[s]] += _wρ(W, ρ, s)
        end
    end
    return nothing
end

# threaded chunk `ch`: scatter of its sub-rows into column `col`, then the arrivals (column `pos`) of its targets `rng`
@inline function _delay_chunk_const!(g, W, ρ, I, ptr, ch, rng, spikes, buffer, pos, col)
    off = col * size(buffer, 1)
    @inbounds for j ∈ spikes
        @simd ivdep for s ∈ ptr[ch, j]:(ptr[ch+1, j]-1)
            buffer[off+I[s]] += _wρ(W, ρ, s)
        end
    end
    @inbounds @simd for i ∈ rng
        g[i] += buffer[i, pos+1]
        buffer[i, pos+1] = 0
    end
    return nothing
end
