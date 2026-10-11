"""
    WCParameter <: AbstractPopulationParameter

Parameter type of `WilsonCowan`; it has no fields. Not exported (`SNNModels.WCParameter`).
"""
struct WCParameter <: AbstractPopulationParameter end

@doc raw"""
    WilsonCowan(; N = 100, name = "WilsonCowan", kwargs...)

Population of rate units named after the Wilson-Cowan model. In SNNModels 1.8.4 its dynamics
are identical to `Rate`: a single variable per unit with unit time constant and `tanh`
transfer function. It does not implement the coupled excitatory/inhibitory Wilson-Cowan
equations (no separate E and I variables, no refractory term, no sigmoid parameters).

# Equations
```math
\frac{dx}{dt} = -x + g + I, \qquad r = \tanh(x)
```
(``t`` in ms), integrated with forward Euler; `g` is reset to zero after each step, as in `Rate`.

# Fields
- `id::String = randstring(12)`, `name::String = "WilsonCowan"`, `param::WCParameter`.
- `N::Int32 = 100`; `x = 0.5randn(N)`; `r = tanh.(x)`; `g`, `I`: inputs, zeros; `records::Dict`.
- `VFT = Vector{Float32}`: type of the state vectors; `VFT = Vector{Float64}` gives a `Float64` population (see `float_type`).

`RateSynapse` (and the FORCE connections) target `g`, as for `Rate` (any target symbol is mapped
to `:g`). Up to SNNModels 1.8.4 there was no `synaptic_target` method for `WilsonCowan`.

# References
The canonical model is Wilson H. R., Cowan J. D. (1972). Excitatory and inhibitory interactions
in localized populations of model neurons. Biophys. J. 12:1-24. The code does not cite it and
does not implement it in full (see above).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
W = SNN.WilsonCowan(N = 10)
SNN.monitor!(W, [:r])
SNN.sim!([W]; duration = 20ms)
```
"""
WilsonCowan

@snn_kw mutable struct WilsonCowan{VFT = Vector{Float32}} <: AbstractPopulation
    id::String = randstring(12)
    name::String = "WilsonCowan"
    param::WCParameter = WCParameter()
    N::Int32 = 100
    x::VFT = 0.5randn(N)
    r::VFT = tanh.(x)
    g::VFT = zeros(N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

function integrate!(p::WilsonCowan, param::WCParameter, dt::AbstractFloat)
    @unpack N, x, r, g, I = p
    _check_state_lengths(N, x, r, g, I)
    # `_tanh_inl`: the bits of `tanh`, with the exponential inlined so that the loop vectorises
    @inbounds @simd for i = 1:N
        xi = x[i] + dt * (-x[i] + g[i] + I[i])
        x[i] = xi
        r[i] = _tanh_inl(xi)
        g[i] = zero(eltype(g)) # g is the synaptic input of this step (connections add to it)
    end
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym = nothing,
    target = nothing,
) where {T<:WilsonCowan}
    g = getfield(post, :g)
    v_post = getfield(post, :r)
    push!(targets, :sym => :g)
    return g, v_post
end

export WilsonCowan
