"""
    FLSynapseParameter()

Parameter of `FLSynapse` (no fields), a subtype of `AbstractConnectionParameter`: `sim!` calls
`forward!(c, param)` through the generic four-argument method, `train!` also calls
`plasticity!`.
"""
struct FLSynapseParameter <: AbstractConnectionParameter end

@snn_kw mutable struct FLSynapse{
    MFT = Matrix{Float32},
    VFT = Vector{Float32},
    FT = Float32,
} <: AbstractConnection
    name::String = "FLSynapse"
    id::String = randstring(12)
    param::FLSynapseParameter = FLSynapseParameter()
    W::MFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    P::MFT  # <rᵢrⱼ>⁻¹
    q::VFT  # P * r
    u::VFT # force weight
    w::VFT # output weight
    f::FT = 0 # postsynaptic traget
    z::FT = 0.5randn()  # output z ≈ f
    targets::Dict = Dict()
    records::Dict = Dict()
end

@doc raw"""
    FLSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)

Dense recurrent connection between rate populations trained with FORCE learning
(recursive least squares on a linear readout fed back into the network).

# Model
State: recurrent weights ``W`` (`N_post x N_pre`), readout weights ``w``, feedback weights
``u``, running inverse correlation matrix ``P``, readout ``z`` and target ``f`` (scalar field
`c.f`, to be set by the user at every step).

`forward!(c, param)`:
```math
z = w^\top r^{post}, \qquad q = P\, r^{pre}, \qquad g = W\, r^{pre} + z\, u
```
(`g` is overwritten). `plasticity!(c, param, dt, T)`:
```math
C = \frac{1}{1 + q^\top r^{post}}, \qquad
w \leftarrow w + C\,(f - z)\, q, \qquad
P \leftarrow P - C\, q\, q^\top
```

# Initialisation
- ``W_{ij} = \mu\, \xi_{ij} / \sqrt{N_{pre}}``, ``\xi_{ij} \sim \mathcal{N}(0, 1)`` (dense).
- ``w_i \sim \mathcal{U}(-1, 1) / \sqrt{N_{post}}``, ``u_i \sim \mathcal{U}(-1, 1)``.
- ``P = \alpha\, \mathbb{1}`` (`N_post x N_post`), ``z \sim 0.5\,\mathcal{N}(0, 1)``, ``f = 0``.

# Keyword arguments
- `μ = 1.5`: gain of the recurrent weights; `p`: unused (the matrix is dense);
  `α = 1`: initial value of the diagonal of ``P``; `kwargs...`: forwarded to the struct.

# Notes
- `pre` and `post` must have a rate field `r`, and `post` a target `:g` (e.g. `Rate`).
  ``P`` is `N_post x N_post` and multiplies ``r^{pre}``, so the connection only works for
  `pre.N == post.N` (recurrent use).
- `sim!` applies `forward!` every step; `train!` also applies the RLS `plasticity!`. Set `c.f`
  before each step (e.g. with a stimulus or a loop of one-step `train!` calls). Up to SNNModels
  1.8.4 `FLSynapseParameter` was not a subtype of `AbstractConnectionParameter` and
  `sim!`/`train!` raised a `MethodError`.

# References
Sussillo D, Abbott LF (2009). Generating coherent patterns of activity from chaotic neural
networks. Neuron 63:544-557 (linked in the code).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
F = SNN.FLSynapse(R, R; μ = 1.5, α = 1)
SNN.SNNModels.forward!(F, F.param)                 # one transmission step
F.f = 1.0f0                                        # target of the readout
SNN.SNNModels.plasticity!(F, F.param, 0.125f0, SNN.SNNModels.Time())
```
"""
FLSynapse

function FLSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)
    rI, rJ, = post.r, pre.r
    W = μ * 1 / √pre.N * randn(post.N, pre.N) # normalized recurrent weight
    w = 1 / √post.N * (2rand(post.N) .- 1) # initial output weight
    u = 2rand(post.N) .- 1 # initial force weight
    P = α * I(post.N) # initial inverse of   = <rr'>
    q = zeros(post.N)

    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:FLSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, :g, nothing)

    FLSynapse(; @symdict(W, rI, rJ, g, P, q, u, w)..., kwargs..., targets = targets)
end

"""
    forward!(c::FLSynapse, param::FLSynapseParameter)

Compute the readout `c.z`, the vector `c.q = P r_pre` and overwrite the target with
`g = W r_pre + z u`.
"""
function forward!(c::FLSynapse, param::FLSynapseParameter)
    @unpack W, rI, rJ, g, P, q, u, w, z = c
    c.z = dot(w, rI)
    # @show z
    mul!(q, P, rJ)
    mul!(g, W, rJ)
    axpy!(c.z, u, g)
end

"""
    plasticity!(c::FLSynapse, param::FLSynapseParameter, dt, T)

Recursive-least-squares update of the readout `w` and of `P` (see `FLSynapse`).
"""
function plasticity!(c::FLSynapse, param::FLSynapseParameter, dt::Float32, T::Time)
    @unpack rI, P, q, w, f, z = c
    C = 1 / (1 + dot(q, rI))
    axpy!(C * (f - z), q, w)
    BLAS.ger!(-C, q, q, P)
end

export FLSynapse
