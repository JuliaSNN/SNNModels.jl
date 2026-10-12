# Chunked thread parallelism of the population updates.
#
# A population of N neurons is split in batches of BATCH = 128 neurons (512 bytes of Float32: no false sharing at chunk
# boundaries); a chunk is `bpc` batches (`neurons_per_thread / 128`, 0 = threading off), with at most MAX_CHUNKS chunks.
# The chunking depends on N and the setting only, never on the number of threads, and per-neuron updates have no
# reduction, so results are bit-identical for every setting and thread count. With one thread, threading off or a
# single chunk, the plain serial loop runs and `@batch` is never called.
using Polyester: @batch

const MAX_CHUNKS = 8
const BATCH = 128                           # neurons per batch
const _BPC = Ref{Int}(0)                    # batches per chunk; 0 = threading off (default)

# neurons_per_thread -> batches per chunk (0 = off)
_bpc_value(x::Integer) = x < 0 ? throw(ArgumentError("neurons_per_thread must be >= 0 (0 or Inf = off)")) :
                         (x == 0 ? 0 : max(1, round(Int, x / BATCH)))
_bpc_value(x::AbstractFloat) = isnan(x) ? throw(ArgumentError("neurons_per_thread is NaN")) :
                               ((isinf(x) || x <= 0) ? 0 : max(1, round(Int, x / BATCH)))
_bpc_value(::Nothing) = _BPC[]

"""
    neurons_per_thread() -> Int
    set_neurons_per_thread!(n)

Neurons per chunk of the threaded updates, rounded to whole 128-neuron batches (default 0: threading off;
`set_neurons_per_thread!(500)` sets 512; `0` or `Inf` switches it off). At most 8 chunks per population; see `foreach_chunk`.
`sim!` and `train!` accept it as a keyword for the duration of the run.
"""
neurons_per_thread() = _BPC[] * BATCH
function set_neurons_per_thread!(n)
    _BPC[] = _bpc_value(n)
    return neurons_per_thread()
end

"""
    with_neurons_per_thread(f, n)

Run `f()` with `neurons_per_thread = n` and restore the previous value afterwards (`n = nothing`: unchanged).
"""
function with_neurons_per_thread(f::F, n) where {F}
    n === nothing && return f()
    old = _BPC[]
    _BPC[] = _bpc_value(n)
    try
        return f()
    finally
        _BPC[] = old
    end
end

# batches per chunk actually used for N neurons: the setting, raised so that at most MAX_CHUNKS chunks cover the population
@inline _bpc_eff(N::Integer, bpc::Integer) = max(bpc, cld(cld(Int(N), BATCH), MAX_CHUNKS))

"""
    nchunks(N, bpc) -> Int

Number of chunks of a population of `N` neurons for `bpc` batches (of 128 neurons) per chunk (`bpc <= 0`: 1).
"""
@inline nchunks(N::Integer, bpc::Integer) = bpc <= 0 ? 1 : clamp(cld(cld(Int(N), BATCH), _bpc_eff(N, bpc)), 1, MAX_CHUNKS)

"""
    chunk_range(N, bpc, c) -> UnitRange{Int}

Neurons of chunk `c`: whole batches, boundaries multiples of 128; the chunks `1:nchunks(N, bpc)` cover `1:N` exactly once.
"""
@inline function chunk_range(N::Integer, bpc::Integer, c::Integer)
    w = _bpc_eff(N, bpc) * BATCH
    return ((c - 1) * w + 1):min(Int(N), c * w)
end

"""
    foreach_chunk(f, N, bpc = batches per chunk setting)

Call `f(range)` for every chunk range of `1:N` (see `nchunks`, `chunk_range`), in parallel on the Polyester
workers when there is more than one chunk and more than one Julia thread; otherwise `f(1:N)` once on the calling thread.
`f` must write only the entries of its own range and must not allocate. Typical use:

    foreach_chunk(p.N) do rng
        @inbounds @simd for i in rng
            ...
        end
    end
"""
@inline function foreach_chunk(f::F, N::Integer, bpc::Integer = _BPC[]) where {F}
    nc = nchunks(N, bpc)
    if nc == 1 || Threads.nthreads() == 1
        f(1:Int(N))
    else
        n = Int(N)
        @batch for c in 1:nc
            f(chunk_range(n, bpc, c))
        end
    end
    return nothing
end

"""
    foreach_chunk_c(f, N, bpc = batches per chunk setting)

Like `foreach_chunk`, but `f(c, range)` also gets the chunk index `c` (1-based), for per-chunk scratch memory or tables.
On the serial path `f(1, 1:N)` is called once.
"""
@inline function foreach_chunk_c(f::F, N::Integer, bpc::Integer = _BPC[]) where {F}
    nc = nchunks(N, bpc)
    if nc == 1 || Threads.nthreads() == 1
        f(1, 1:Int(N))
    else
        n = Int(N)
        @batch for c in 1:nc
            f(c, chunk_range(n, bpc, c))
        end
    end
    return nothing
end

# floats between the scratch slots of two chunks: 128 bytes, so no cache line is shared whatever the alignment
const SCRATCH_STRIDE = 32

"""
    @chunks N [bpc] rng body

Macro form of `foreach_chunk`: `@chunks p.N r begin ... end` runs the block once per chunk with `r` bound to
the range of the chunk (`@chunks N bpc r body` with an explicit number of batches per chunk).
"""
macro chunks(N, rng, body)
    return :(foreach_chunk($(esc(N))) do $(esc(rng)); $(esc(body)); end)
end
macro chunks(N, bpc, rng, body)
    return :(foreach_chunk($(esc(N)), $(esc(bpc))) do $(esc(rng)); $(esc(body)); end)
end

# Synaptic-current stage of the multicompartment populations: the chunk method where the synapse model has one (see `_chunk_synapse`).
# The other models only get here with the whole range (the populations call the serial stages for them).
@inline function _synaptic_current!(p, synapse, synvars, v, syncurr, rng::UnitRange{Int})
    if _chunk_synapse(synapse)
        synaptic_current!(p, synapse, synvars, v, syncurr, rng)
    else
        synaptic_current!(p, synapse, synvars, v, syncurr)
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Spiking synapses: partition by postsynaptic range
#
# The chunks of a synapse are those of its postsynaptic population: chunk `c` adds to `g[chunk_range(Npost, bpc, c)]` (and the
# delay-buffer rows and weights of the synapses ending there). The CSC entries of a column are sorted by postsynaptic index, so
# those of chunk `c` are the contiguous range `ptr[c, j] : ptr[c+1, j] - 1` (`ChunkPtr`). The spike list is built before the
# fork; every chunk visits the spikes in the same order, so each target gets its additions in serial order (bit-identical).
"""
    ChunkPtr()

Cache of the sub-row pointers of a sparse synapse (`SpikingSynapse.chunkptr`): `ptr[c, j] : ptr[c+1, j] - 1` are the entries
of presynaptic column `j` whose postsynaptic neuron lies in chunk `c`. Built on first use (`chunk_ptr!`), rebuilt when
`colptr`, `I`, `Npost` or the chunking change; `invalidate_chunk_ptr!` forces a rebuild (`update_sparse_matrix!` calls it).
`builds` counts the (re)builds, `buildtime` is the duration (s) of the last one.
"""
mutable struct ChunkPtr
    ptr::Matrix{Int32}
    nc::Int
    w::Int
    Npost::Int
    colptr::Any
    I::Any
    nnz::Int
    sorted::Bool
    builds::Int
    buildtime::Float64
end
ChunkPtr() = ChunkPtr(zeros(Int32, 0, 0), 0, 0, 0, nothing, nothing, -1, true, 0, 0.0)
invalidate_chunk_ptr!(cp::ChunkPtr) = (cp.nnz = -1; nothing)

function _build_chunk_ptr!(cp::ChunkPtr, colptr, I, Npost::Int, nc::Int, w::Int)
    Npre = length(colptr) - 1
    ptr = size(cp.ptr) == (nc + 1, Npre) ? cp.ptr : Matrix{Int32}(undef, nc + 1, Npre)
    sorted = true
    @inbounds for j in 1:Npre
        lo = Int(colptr[j])
        hi = Int(colptr[j+1]) - 1
        for s in (lo+1):hi
            sorted &= I[s-1] < I[s]
        end
        ptr[1, j] = lo
        s = lo
        for c in 2:nc
            bound = (c - 1) * w                 # post <= bound belongs to a chunk before c: first entry with I > bound
            a = s
            b = hi + 1
            while a < b
                m = (a + b) >>> 1
                if I[m] > bound
                    b = m
                else
                    a = m + 1
                end
            end
            s = a
            ptr[c, j] = s
        end
        ptr[nc+1, j] = hi + 1
    end
    cp.ptr = ptr; cp.nc = nc; cp.w = w; cp.Npost = Npost; cp.colptr = colptr; cp.I = I; cp.nnz = length(I); cp.sorted = sorted
    return cp
end

"""
    chunk_ptr!(cp::ChunkPtr, colptr, I, Npost, bpc) -> Matrix{Int32}

The sub-row pointer table for `Npost` postsynaptic neurons and `bpc` batches per chunk, (re)built if needed.
Not thread-safe: call before the fork. If `cp.sorted` is false (unsorted columns) the connection must run serially.
"""
function chunk_ptr!(cp::ChunkPtr, colptr, I, Npost::Int, bpc::Int)
    nc = nchunks(Npost, bpc)
    w = _bpc_eff(Npost, bpc) * BATCH
    if !(cp.nnz == length(I) && cp.colptr === colptr && cp.I === I && cp.nc == nc && cp.w == w && cp.Npost == Npost)
        t0 = time_ns()
        _build_chunk_ptr!(cp, colptr, I, Npost, nc, w)
        cp.buildtime = (time_ns() - t0) * 1e-9
        cp.builds += 1
    end
    return cp.ptr
end

# Synaptic operations per step (spikes x fan-out, plus `ops0` from the caller) below which a synapse runs on the calling
# thread, where the fork costs more than it saves; same result either way. `0` forces the fork, `typemax(Int)` keeps all serial.
const SYN_MIN_OPS = Ref(8000)

@inline function _spike_ops(spikes, ptr_)
    n = 0
    @inbounds for j in spikes
        n += Int(ptr_[j+1]) - Int(ptr_[j])
    end
    return n
end

# `nothing` (run serially) or the sub-row pointer table of the synapse `cp` for this step: threading on, more than one chunk of the
# postsynaptic population, and enough synaptic operations (spikes x fan-out, `ops0` more from the caller)
@inline function _syn_chunk_ptr(cp::ChunkPtr, Npost::Int, colptr, I, spikes, ops0::Int = 0)
    (Threads.nthreads() == 1 || nchunks(Npost, _BPC[]) == 1) && return nothing
    (ops0 + _spike_ops(spikes, colptr) < SYN_MIN_OPS[]) && return nothing
    ptr = chunk_ptr!(cp, colptr, I, Npost, _BPC[])
    return cp.sorted ? ptr : nothing
end

export ChunkPtr, chunk_ptr!, invalidate_chunk_ptr!

export neurons_per_thread, set_neurons_per_thread!, with_neurons_per_thread, foreach_chunk, foreach_chunk_c, @chunks
