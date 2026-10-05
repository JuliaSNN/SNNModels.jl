"""
    PINningSynapseParameter()

Parameter of `PINningSynapse` (no fields). Not a subtype of `AbstractConnectionParameter`
(see the `PINningSynapse` docstring).
"""
struct PINningSynapseParameter end

@snn_kw mutable struct PINningSynapse{MFT = Matrix{Float32},VFT = Vector{Float32}} <:
                       AbstractConnection
    name::String = "PINningSynapse"
    id::String = randstring(12)
    param::PINningSynapseParameter = PINningSynapseParameter()
    W::MFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    P::MFT  # <rᵢrⱼ>⁻¹
    q::VFT  # P * r
    f::VFT  # postsynaptic traget
    targets::Dict = Dict()
    records::Dict = Dict()
end

@doc raw"""
    PINningSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)

Dense recurrent connection between rate populations trained with partial in-network
training (PINning): the recurrent weights themselves are adjusted by recursive least
squares so that the input of every postsynaptic unit follows its target ``f_i``.

# Model
`forward!(c, param)`:
```math
q = P\, r^{pre}, \qquad g = W\, r^{pre}
```
(`g` is overwritten). `plasticity!(c, param, dt, T)`:
```math
C = \frac{1}{1 + q^\top r^{post}}, \qquad
W \leftarrow W + C\,(f - g)\, q^\top, \qquad
P \leftarrow P - C\, q\, q^\top
```
`c.f` (vector of length `N_post`, initialised to zeros) holds the targets and is set by the user.

# Initialisation
``W_{ij} = \mu\, \xi_{ij}/\sqrt{N_{pre}}``, ``\xi_{ij}\sim\mathcal{N}(0,1)``;
``P = \alpha\,\mathbb{1}`` (`N_post x N_post`); `q = f = 0`.

# Keyword arguments
- `μ = 1.5`: gain; `p`: unused (dense); `α = 1`: initial diagonal of ``P``;
  `kwargs...`: forwarded to the struct.

# Notes
- `pre` and `post` must have a rate field `r` and `post` a target `:g` (e.g. `Rate`);
  only `pre.N == post.N` works.
- `PINningSynapseParameter` is not a subtype of `AbstractConnectionParameter`; in SNNModels
  1.8.4 `sim!`/`train!` raise a `MethodError` for this connection. `forward!(c, c.param)` and
  `plasticity!(c, c.param, dt, T)` can be called directly.

# References
Rajan K, Harvey CD, Tank DW (2016). Recurrent network models of sequence generation and
memory. Neuron 90:128-142 (PubMed 26971945, linked in the code).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
P = SNN.PINningSynapse(R, R; μ = 1.5)
SNN.SNNModels.forward!(P, P.param)
P.f .= 0.1f0
SNN.SNNModels.plasticity!(P, P.param, 0.125f0, SNN.SNNModels.Time())
```
"""
PINningSynapse

function PINningSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)
    rI, rJ = post.r, pre.r
    W = μ * 1 / √pre.N * randn(post.N, pre.N) # normalized recurrent weight
    P = α * I(post.N) # initial inverse of C = <rr'>
    f, q = zeros(post.N), zeros(post.N)
    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:PinningSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, :g, nothing)

    PINningSynapse(; @symdict(W, rI, rJ, g, P, q, f)..., kwargs..., targets = targets)
end

"""
    forward!(c::PINningSynapse, param::PINningSynapseParameter)

Compute `c.q = P r_pre` and overwrite the target with `g = W r_pre`.
"""
function forward!(c::PINningSynapse, param::PINningSynapseParameter)
    @unpack W, rI, rJ, g, P, q = c
    mul!(q, P, rJ)
    mul!(g, W, rJ)
end

"""
    plasticity!(c::PINningSynapse, param::PINningSynapseParameter, dt, T)

Recursive-least-squares update of `W` and `P` (see `PINningSynapse`).
"""
function plasticity!(
    c::PINningSynapse,
    param::PINningSynapseParameter,
    dt::Float32,
    T::Time,
)
    @unpack W, rI, g, P, q, f = c
    C = 1 / (1 + dot(q, rI))
    BLAS.ger!(C, f - g, q, W)
    BLAS.ger!(-C, q, q, P)
end

export PINningSynapse
