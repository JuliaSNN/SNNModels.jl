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
Up to SNNModels 1.8.4 `g` was never reset, so with a `RateSynapse` it was the running sum of all
past inputs.

The population has no `fire` field, so only `:x`, `:r`, `:g` can be recorded.

# Fields
- `id::String = randstring(12)`, `name::String = "Rate"`, `param::RateParameter = RateParameter()`.
- `N::Int32 = 100`.
- `x::Vector{Float32} = 0.5randn(N)`: internal state.
- `r::Vector{Float32} = tanh.(x)`: output rate, in ``(-1, 1)``.
- `g::Vector{Float32}`: synaptic input, zeros; the target of `RateSynapse` (any target symbol
  is mapped to `:g`).
- `I::Vector{Float32}`: external input, zeros.
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
    integrate!(p::Rate, param::RateParameter, dt::Float32)

One forward-Euler step of the rate units: `x += dt * (-x + g + I)`, `r = tanh(x)`, then
`g = 0` (see `Rate`).
"""
function integrate!(p::Rate, param::RateParameter, dt::Float32)
    @unpack N, x, r, g, I = p
    @inbounds for i = 1:N
        x[i] += dt * (-x[i] + g[i] + I[i])
        r[i] = tanh(x[i]) #max(0, x[i])
        g[i] = 0.0f0 # g is the synaptic input of this step (connections add to it)
    end
end


export Rate
