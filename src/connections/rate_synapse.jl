"""
    RateSynapseParameter(; lr = 1e-3)

Parameter of `RateSynapse` (and of the non-exported `SpikeRateSynapse`).

# Fields
- `lr::Float32 = 1e-3`: learning rate of the weight update applied by `plasticity!`
  (dimensionless; the update is applied once per `train!` step, independent of `dt`).
"""
RateSynapseParameter

@snn_kw struct RateSynapseParameter{FT = Float32} <: AbstractConnectionParameter
    lr::FT = 1e-3
end

@snn_kw mutable struct RateSynapse{VIT = Vector{Int32},VFT = Vector{Float32}} <:
                       AbstractConnection
    name::String="RateSynapse"
    id::String = randstring(12)
    param::RateSynapseParameter = RateSynapseParameter()
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    targets::Dict = Dict()
    records::Dict = Dict()
end

@doc raw"""
    RateSynapse(pre, post; μ = 0.0, p = 0.0, kwargs...)

Sparse connection between rate populations (`pre` and `post` need a rate field `r`; the
target is the field `g` of `post`, e.g. `Rate`).

The weight matrix is ``W = \frac{\mu}{\sqrt{p N_{pre}}} X`` with ``X`` a sparse
`N_post x N_pre` matrix with density `p` and standard normal non-zeros (`sprandn`).

# Transmission (`forward!`, every step of `sim!` and `train!`)
```math
g_i \leftarrow g_i + \sum_j W_{ij}\, r_j
```
`g` is not reset by the synapse.

# Plasticity (`plasticity!`, only under `train!`)
For every presynaptic neuron ``j``, with learning rate ``\eta`` = `param.lr`:
```math
\Delta_j = \eta \left(r_j - \sum_i r_i W_{ij}\right), \qquad
W_{ij} \leftarrow W_{ij} + r_i\, \Delta_j
```
(``r_i`` postsynaptic, ``r_j`` presynaptic rate). Reference not given in the code (the
docstring previously linked a Brian2 tutorial on synapses, which does not describe this rule).

# Keyword arguments
- `μ = 0.0`: weight scale; `p = 0.0`: connection density. Always pass `p > 0`: with
  `p = 0` the scale ``\mu/\sqrt{p N_{pre}}`` is not finite and every entry of the
  `N_post x N_pre` matrix becomes `NaN` (the matrix is stored densely).

`RateSynapse` stores only `colptr`, `I`, `W` (no `rowptr`, `J`, `index`), so the
connectivity helpers that need them (`matrix`, `presynaptic`, ...) do not apply.
- `kwargs...`: forwarded to the struct (e.g. `param = RateSynapseParameter(lr = 1e-4)`,
  `name`).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
RR = SNN.RateSynapse(R, R; μ = 1.0, p = 0.2)
SNN.sim!([R], [RR]; duration = 10ms)
```
"""
RateSynapse

function RateSynapse(pre, post; μ = 0.0, p = 0.0, kwargs...)
    w = SparseMatrixCSC{Float32,Int}(μ / √(p * pre.N) * sprandn(post.N, pre.N, p))
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, rJ = post.r, pre.r
    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:RateSynapse,
    )
    @views g, v_post = synaptic_target(targets, post)

    RateSynapse(; @symdict(colptr, I, W, rI, rJ, g)..., kwargs..., targets = targets)
end

@doc raw"""
    forward!(c::RateSynapse, param::RateSynapseParameter)

Add ``\sum_j W_{ij} r_j`` to the postsynaptic input `c.g[i]`.
"""
function forward!(c::RateSynapse, param::RateSynapseParameter)
    @unpack colptr, I, W, rI, rJ, g = c
    @unpack lr = param
    # fill!(g, zero(eltype(g)))
    @inbounds for j = 1:(length(colptr)-1)
        rJj = rJ[j]
        for s = colptr[j]:(colptr[j+1]-1)
            g[I[s]] += W[s] * rJj
        end
    end
end

"""
    plasticity!(c::RateSynapse, param::RateSynapseParameter, dt, T)

Weight update of `RateSynapse` (see its docstring); called only by `train!`.
"""
function plasticity!(c::RateSynapse, param::RateSynapseParameter, dt::Float32, T::Time)
    @unpack colptr, I, W, rI, rJ, g = c
    @unpack lr = param
    @inbounds for j = 1:(length(colptr)-1)
        s_row = colptr[j]:(colptr[j+1]-1)
        rIW = zero(Float32)
        for s in s_row
            rIW += rI[I[s]] * W[s]
        end
        Δ = lr * (rJ[j] - rIW)
        for s in s_row
            W[s] += rI[I[s]] * Δ
        end
    end
end

export RateSynapse
