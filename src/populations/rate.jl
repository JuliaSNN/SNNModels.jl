"""
    RateParameter <: AbstractPopulationParameter

Parameters for rate-based neuron model. This model has no adjustable parameters - it uses a simple tanh transfer function.
"""
RateParameter

struct RateParameter <: AbstractPopulationParameter end

"""
    Rate{VFT} <: AbstractPopulation

Rate-based neuron population model with tanh activation function.

# Fields
## Population Info
- `id::String`: Unique identifier (default: random 12-character string)
- `name::String`: Population name (default: "Rate")
- `N::Int32`: Number of neurons (default: 100)

## Parameters
- `param::RateParameter`: Model parameters (no adjustable parameters)

## State Variables
- `x::VFT`: Internal state/activation variable (initialized with random values)
- `r::VFT`: Output firing rate, r = tanh(x) (initialized from x)
- `g::VFT`: Synaptic input (default: zeros)
- `I::VFT`: External input current (default: zeros)

## Recordings
- `records::Dict`: Dictionary for storing simulation data

# Model Equations
- dx/dt = -x + g + I
- r = tanh(x)

# References
- [Neuronal Dynamics - Rate Models](https://neuronaldynamics.epfl.ch/online/Ch15.S3.html)
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

Update rate-based neuron population for one timestep.

# Arguments
- `p::Rate`: The neuron population
- `param::RateParameter`: Model parameters (unused)
- `dt::Float32`: Time step size

# Details
- Updates internal state x: dx/dt = -x + g + I
- Computes output rate: r = tanh(x)
- Rate r is bounded between -1 and 1
"""
function integrate!(p::Rate, param::RateParameter, dt::Float32)
    @unpack N, x, r, g, I = p
    @inbounds for i = 1:N
        x[i] += dt * (-x[i] + g[i] + I[i])
        r[i] = tanh(x[i]) #max(0, x[i])
    end
end


export Rate
