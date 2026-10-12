"""
    RateParameter <: AbstractPopulationParameter

Parameter type of the `Rate` population. It has no fields: the model has a fixed unit time
constant and a `tanh` transfer function. Not exported (`SNNModels.RateParameter`).
"""
RateParameter

struct RateParameter <: AbstractPopulationParameter end

@doc raw"""
    Rate(; N = 100, name = "Rate", kwargs...)

Population of rate units with a `tanh` transfer function.

# Equations
```math
\frac{dx}{dt} = -x + g + I, \qquad r = \tanh(x)
```
with ``t`` in ms, i.e. the time constant is fixed to 1 ms. ``g`` is the input written by
`RateSynapse` (``g_i \mathrel{+}= \sum_j W_{ij} r_j`` each step), ``I`` an external input.

# Integration
Forward Euler, `x += dt * (-x + g + I)`, then `r = tanh(x)`, then `g` is set to zero: ``g`` is
the synaptic input of one step, written by the connections after the population update
(`RateSynapse` adds ``W r``, the FORCE connections overwrite it). Use `I` for a constant input.
The population has no `fire` field, so only `:x`, `:r`, `:g` can be recorded; populations are
recorded right after their update, so a recorded `:g` is zero.

# Fields
- `id::String = randstring(12)`, `name::String = "Rate"`, `param::RateParameter = RateParameter()`.
- `N::Int32 = 100`.
- `VFT = Vector{Float32}`: type of the state vectors; `Vector{Float64}` gives a `Float64` population (see `float_type`).
- `x::VFT = 0.5randn(N)`: internal state.
- `r::VFT = tanh.(x)`: output rate, in ``(-1, 1)``.
- `g::VFT`: synaptic input, zeros; the target of `RateSynapse` (any target symbol
  is mapped to `:g`).
- `I::VFT`: external input, zeros.
- `records::Dict`.

# References
- [Neuronal Dynamics - Rate Models](https://neuronaldynamics.epfl.ch/online/Ch15.S3.html)

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 10)
R.I .= 0.5
SNN.monitor!(R, [:r])
SNN.sim!([R]; duration = 20ms)
```
"""
Rate 

@snn_kw mutable struct Rate{VFT = Vector{Float32}} <: AbstractPopulation
    id::String = randstring(12)
    name::String = "Rate"
    param::RateParameter = RateParameter()
    N::Int32 = 100
    x::VFT = 0.5randn(N)
    r::VFT = tanh.(x)
    g::VFT = zeros(N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym = nothing,
    target = nothing,
) where {T<:Rate}
    sym = :g
    g = getfield(post, sym)
    v_post = getfield(post, :r)
    push!(targets, :sym => sym)
    return g, v_post
end

"""
    integrate!(p::Rate, param::RateParameter, dt)

One forward-Euler step of the rate units: `x += dt * (-x + g + I)`, `r = tanh(x)`, then
`g = 0` (see `Rate`).
"""
function integrate!(p::Rate, param::RateParameter, dt_::AbstractFloat)
    dt = _step_dt(p, dt_)   # in the precision of the population
    @unpack N, x, r, g, I = p
    _check_state_lengths(N, x, r, g, I)
    # `_tanh_inl`: the bits of `tanh`, inlined so that the loop vectorises
    @inbounds @simd for i = 1:N
        xi = x[i] + dt * (-x[i] + g[i] + I[i])
        x[i] = xi
        r[i] = _tanh_inl(xi)
        g[i] = zero(eltype(g)) # g is the synaptic input of this step (connections add to it)
    end
end


export Rate
