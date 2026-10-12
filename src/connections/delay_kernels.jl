# Kernels for a delay common to all synapses (`delaysteps` of length 1): every event of a step lands in the same ring column.
# Bit-identical to the per-synapse loop: increments summed in emission order, no `@fastmath` (it would contract
# `buffer + W * ρ` into an FMA); a threaded chunk touches only its rows `rng`. `ivdep` is valid because the rows `I[s]`
# of one CSC column are distinct.

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

# Spike-list path (common delay `d`, NoLTP, NoSTP): `param.spike_ring` keeps the spike lists of the last `d + 1` steps (slot
# `pos` = this step) and the list emitted `d` steps ago is scattered into `g` directly, with `W` read at arrival. Not
# bit-identical to the buffer path (increments added one by one; a weight changed in flight is delivered with its new value).
@inline _use_spike_ring(c::SpikingSynapse) = c.LTPParam isa NoLTP && c.STPParam isa NoSTP && _dconst(c.param) >= 0

function _forward_spike_ring!(c::SpikingSynapse, param::SpikingSynapseDelayParameter, Npost::Int)
    @unpack colptr, I, W, g = c
    ring = param.spike_ring
    nslots = _dconst(param) + 1
    if length(ring) != nslots          # first call or new delay in steps: empty ring (spikes in flight dropped, as the buffer)
        resize!(ring, nslots)
        for k in eachindex(ring)
            ring[k] = Int32[]
        end
    end
    # increments still in the buffer after a switch from the buffer path: delivered on their arrival step, then the buffer is released
    buf = param.buffer
    if size(buf, 1) > 0 && param.drain[1] == 0     # fresh from the buffer path: the d columns ahead hold increments
        if nslots > 1
            param.drain[1] = nslots - 1
        else
            param.buffer = zeros(eltype(buf), 0, nslots)
        end
    end
    pos = param.pos[1]
    spikes = presynaptic_spikes(c)
    cur = ring[pos+1]
    resize!(cur, length(spikes))
    copyto!(cur, spikes)
    nxt = pos + 1 == nslots ? 0 : pos + 1
    arr = ring[nxt+1]                  # emitted d steps ago ((pos - d) mod nslots); `cur` itself when d == 0
    ptr = _syn_chunk_ptr(c.chunkptr, Npost, colptr, I, arr)
    _ring_deliver!(g, W, _efficacy_or_nothing(c), I, colptr, ptr, arr, Npost)   # function barrier on the efficacy
    if param.drain[1] > 0
        @inbounds @simd for i ∈ 1:size(buf, 1)
            g[i] += buf[i, pos+1]
            buf[i, pos+1] = 0
        end
        param.drain[1] -= 1
        param.drain[1] == 0 && (param.buffer = zeros(eltype(buf), 0, nslots))
    end
    return param.pos[1] = nxt
end

function _ring_deliver!(g, W, ρ, I, colptr, ptr, arr, Npost)
    if ptr === nothing
        @inbounds for j ∈ arr
            @simd ivdep for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += _wρ(W, ρ, s)
            end
        end
    else
        foreach_chunk_c(Npost) do ch, rng
            _ring_chunk!(g, W, ρ, I, ptr, ch, arr)
        end
    end
    return nothing
end

# Buffer path: allocate the `Npost` rows of the buffer and move the spikes in flight of the spike-list path into the buffer
# column of their arrival, with the current `W`; then drop the spike lists. `pos` is common to buffer columns and ring slots.
@inline function _delay_buffer!(c::SpikingSynapse, param::SpikingSynapseDelayParameter, Npost::Int)
    size(param.buffer, 1) == Npost || (param.buffer = zeros(eltype(param.buffer), Npost, size(param.buffer, 2)))
    param.drain[1] = 0
    ring = param.spike_ring
    if !isempty(ring)
        ns = size(param.buffer, 2)
        if length(ring) == ns && Npost > 0
            @unpack colptr, I, W = c
            buffer, pos = param.buffer, param.pos[1]
            for a ∈ 1:(ns-1)
                list = ring[mod(pos - a, ns)+1]
                off = mod(pos + ns - 1 - a, ns) * Npost       # column pos + d - a, d = ns - 1
                @inbounds for j ∈ list
                    for s ∈ colptr[j]:(colptr[j+1]-1)
                        buffer[off+I[s]] += W[s]
                    end
                end
            end
        end
        empty!(ring)
    end
    return nothing
end

@inline function _ring_chunk!(g, W, ρ, I, ptr, ch, spikes)
    @inbounds for j ∈ spikes
        @simd ivdep for s ∈ ptr[ch, j]:(ptr[ch+1, j]-1)
            g[I[s]] += _wρ(W, ρ, s)
        end
    end
    return nothing
end
