"""
    IZParameter{FT<:AbstractFloat}

Parameters for the Izhikevich neuron model.

# Fields
- `a::FT`: Time scale of recovery variable u (default: 0.01)
- `b::FT`: Sensitivity of u to v (default: 0.2)
- `c::FT`: After-spike reset value of v in mV (default: -65)
- `d::FT`: After-spike increment of u (default: 2)
- `τe::FT`: Excitatory synaptic time constant (default: 5ms)
- `τi::FT`: Inhibitory synaptic time constant (default: 10ms)
- `Ee::FT`: Excitatory reversal potential (default: 0mV)
- `Ei::FT`: Inhibitory reversal potential (default: -80mV)

# References
- Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on neural networks, 14(6), 1569-1572.
"""
IZParameter
@snn_kw struct IZParameter{FT = Float32}
    a::FT = 0.01
    b::FT = 0.2
    c::FT = -65
    d::FT = 2
    τe::FT = 5ms
    τi::FT = 10ms
    Ee::FT = 0mV
    Ei::FT = -80mV
end

"""
    IZ{VFT, VBT} <: AbstractPopulation

Izhikevich neuron population model. Simple yet biologically plausible model that can reproduce various spiking patterns.

# Fields
## Population Info
- `id::String`: Unique identifier (default: random 12-character string)
- `name::String`: Population name (default: "IZ")
- `N::Int32`: Number of neurons (default: 100)

## Parameters
- `param::IZParameter`: Model parameters (default: `IZParameter()`)

## State Variables
- `v::VFT`: Membrane potential (initialized to -65mV)
- `u::VFT`: Recovery variable (initialized to b*v)
- `fire::VBT`: Spike flags (default: false)
- `I::VFT`: External input current (default: zeros)
- `ge::VFT`: Excitatory conductance (initialized with random values)
- `gi::VFT`: Inhibitory conductance (initialized with random values)

## Recordings
- `records::Dict`: Dictionary for storing simulation data

# Model Equations
- dv/dt = 0.04v² + 5v + 140 - u + I
- du/dt = a(bv - u)
- if v ≥ 30mV: v ← c, u ← u + d

# References
- [Izhikevich, 2003](https://www.izhikevich.org/publications/spikes.htm)
"""
IZ

@snn_kw mutable struct IZ{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractPopulation
    id::String = randstring(12)
    name::String = "IZ"
    param::IZParameter = IZParameter()
    N::Int32 = 100
    v::VFT = fill(-65.0, N)
    u::VFT = param.b * v
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
    ge::VFT = (1.5randn(N) .+ 4) .* 10nS
    gi::VFT = (12randn(N) .+ 20) .* 10nS
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym = nothing,
    target = nothing,
) where {T<:IZ}
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end

"""
    integrate!(p::IZ, param::IZParameter, dt::Float32)

Update Izhikevich neuron population for one timestep.

# Arguments
- `p::IZ`: The neuron population
- `param::IZParameter`: Model parameters
- `dt::Float32`: Time step size

# Details
- Updates synaptic conductances exponentially
- Integrates membrane potential using Euler method (two half-steps for stability)
- Updates recovery variable u
- Applies synaptic currents
- Detects spikes (v > 30mV) and applies reset
"""
function integrate!(p::IZ, param::IZParameter, dt::Float32)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d = param
    @unpack ge, gi = p
    @inbounds for i = 1:N
        ge[i] += dt * -ge[i] / param.τe
        gi[i] += dt * -gi[i] / param.τi
    end
    @inbounds for i = 1:N
        v[i] += 0.5f0 * dt * (0.04f0 * v[i]^2 + 5.0f0 * v[i] + 140.0f0 - u[i] + I[i])
        v[i] += 0.5f0 * dt * (0.04f0 * v[i]^2 + 5.0f0 * v[i] + 140.0f0 - u[i] + I[i])
        u[i] += dt * (a * (b * v[i] - u[i]))
        v[i] += dt * (ge[i] * (param.Ee - v[i]) + gi[i] * (param.Ei - v[i]))
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > 30.0f0
        v[i] = ifelse(fire[i], c, v[i])
        u[i] += ifelse(fire[i], d, 0.0f0)
    end
end


export IZ, IZParameter
