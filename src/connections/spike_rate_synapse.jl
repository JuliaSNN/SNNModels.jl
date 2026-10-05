
@snn_kw mutable struct SpikeRateSynapse{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = VBT,
} <: AbstractConnection
    name::String = "SpikeRateSynapse"
    id::String = randstring(12)
    param::RateSynapseParameter = RateSynapseParameter()
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    fireJ::VBT # presynaptic firing
    g::VFT  # postsynaptic input
    targets::Dict = Dict()
    records::Dict = Dict()
end

@doc raw"""
    SpikeRateSynapse(pre, post; μ = 0.0, p = 0.0, kwargs...)

Sparse connection from a spiking population to a rate population (not exported).
``W = \mu X/\sqrt{p N_{pre}}`` with ``X`` = `sprandn(N_post, N_pre, p)`; `p` must be in
`(0, 1]`. The target is `post.g` (see `synaptic_target` of `Rate`).

`forward!(c, param, dt, T)`: for every presynaptic neuron ``j`` that fired in this step,
``g_i \leftarrow g_i + W_{ij}/dt``, i.e. each spike is a delta input that makes the state ``x_i``
of the `Rate` unit jump by ``W_{ij}`` (independent of `dt`; `Rate` resets `g` after each step).

The connection has no learning rule: `plasticity!` is a no-op (the `RateSynapse` rule needs a
presynaptic rate, which a spiking population does not have).

!!! note "Changed after SNNModels 1.8.4"
    Up to 1.8.4 a spike added ``W_{ij}`` to `g` (never reset, so `g` was the running sum of all
    past inputs), `plasticity!` failed (`rJ` undefined), the struct had no `name` field and
    `targets` was empty. The unused fields `Apre` and `tpre` were removed.
"""
SpikeRateSynapse

function SpikeRateSynapse(pre, post; μ = 0.0, p = 0.0, kwargs...)
    0 < p <= 1 || throw(ArgumentError("SpikeRateSynapse needs a connection probability 0 < p <= 1, got p = $p"))
    w = SparseMatrixCSC{Float32,Int}(μ / √(p * pre.N) * sprandn(post.N, pre.N, p))
    rowptr, colptr, I, J, index, W = dsparse(w)
    targets = Dict{Symbol,Any}(:fire => pre.id, :post => post.id, :pre => pre.id, :type => :SpikeRateSynapse)
    g, v_post = synaptic_target(targets, post)
    rI, fireJ = post.r, pre.fire
    SpikeRateSynapse(; @symdict(colptr, I, W, rI, fireJ, g)..., kwargs..., targets = targets)
end

"""
    forward!(c::SpikeRateSynapse, param::RateSynapseParameter, dt::Float32, T::Time)

For each presynaptic spike add ``W_{ij}/dt`` to `c.g[i]` (a delta input of integral ``W_{ij}``).
"""
function forward!(c::SpikeRateSynapse, param::RateSynapseParameter, dt::Float32, T::Time)
    @unpack colptr, I, W, fireJ, g = c
    @inbounds for j = 1:(length(colptr)-1)
        if fireJ[j]
            for s = colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s] / dt
            end
        end
    end
end

plasticity!(c::SpikeRateSynapse, param::RateSynapseParameter, dt::Float32, T::Time) = nothing
