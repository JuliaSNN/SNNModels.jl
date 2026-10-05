"""
    PINningSparseSynapseParameter()

Parameter of the non-exported `PINningSparseSynapse` (no fields; a subtype of
`AbstractConnectionParameter`).
"""
struct PINningSparseSynapseParameter <: AbstractConnectionParameter end

@snn_kw mutable struct PINningSparseSynapse{VIT = Vector{Int32},VFT = Vector{Float32}} <:
                       AbstractConnection
    name::String = "PINningSparseSynapse"
    id::String = randstring(12)
    param::PINningSparseSynapseParameter = PINningSparseSynapseParameter()
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    P::VFT  # <rᵢrⱼ>⁻¹
    q::VFT  # P * r
    f::VFT  # postsynaptic traget
    targets::Dict = Dict()
    records::Dict = Dict()
end

@doc raw"""
    PINningSparseSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)

Sparse variant of `PINningSynapse` (not exported). ``W = \mu X/\sqrt{p N_{pre}}`` with
``X`` = `sprandn(N_post, N_pre, p)`; ``P`` lives on the sparsity pattern of ``W``, initialised
to ``\alpha`` on stored diagonal entries.

`forward!(c, param)` resets `g` and `q` and accumulates, over stored synapses,
``g_i = \sum_j W_{ij} r_j`` and ``q_i = \sum_j P_{ij} r_j``.
`plasticity!(c, param, dt, T)` with ``C = 1/(1 + q^\top r^{post})`` updates every stored
synapse: ``P_{ij} \leftarrow P_{ij} - C q_i q_j`` and
``W_{ij} \leftarrow W_{ij} + C (f_i - g_i) q_j``.

`p` must be in `(0, 1]` (an `ArgumentError` is raised otherwise). Up to SNNModels 1.8.4 the
parameter type had no supertype, so `sim!`/`train!` did not dispatch on it.

# References
Rajan K, Harvey CD, Tank DW (2016). Neuron 90:128-142 (PubMed 26971945, linked in the code).
"""
PINningSparseSynapse

function PINningSparseSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)
    0 < p <= 1 || throw(ArgumentError("PINningSparseSynapse needs a connection probability 0 < p <= 1, got p = $p"))
    w = μ / √(p * pre.N) * sprandn(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, rJ = post.r, pre.r
    P = α .* (I .== J)
    f, q = zeros(post.N), zeros(post.N)

    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:PinningSparseSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, :g, nothing)

    PINningSparseSynapse(;
        @symdict(colptr, I, W, rI, rJ, g, P, q, f)...,
        kwargs...,
        targets = targets,
    )
end

function forward!(c::PINningSparseSynapse, param::PINningSparseSynapseParameter)
    @unpack colptr, I, W, rJ, g, P, q = c
    fill!(q, zero(Float32))
    fill!(g, zero(Float32))
    @inbounds for j = 1:(length(colptr)-1)
        rJj = rJ[j]
        for s = colptr[j]:(colptr[j+1]-1)
            i = I[s]
            g[i] += W[s] * rJj
            q[i] += P[s] * rJj
        end
    end
end

function plasticity!(
    c::PINningSparseSynapse,
    param::PINningSparseSynapseParameter,
    dt::Float32,
    T::Time,
)
    @unpack colptr, I, W, rI, g, P, q, f = c
    C = 1 / (1 + dot(q, rI))
    @inbounds for j = 1:(length(colptr)-1)
        for s = colptr[j]:(colptr[j+1]-1)
            i = I[s]
            P[s] += -C * q[i] * q[j]
            W[s] += C * (f[i] - g[i]) * q[j]
        end
    end
end
