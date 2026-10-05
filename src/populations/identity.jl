"""
    IdentityParam <: AbstractPopulationParameter

Parameter type of `Identity` (no fields). `Population(IdentityParam(); kwargs...)` returns an
`Identity`.
"""
struct IdentityParam <: AbstractPopulationParameter end

"""
    Identity(; N = 100, name = "identity", kwargs...)

Relay population: each neuron emits a spike in every step in which it received positive input,
so that spikes arriving through a connection are relayed to downstream connections. Because
populations are integrated before connections are forwarded, input written in step ``t`` is
turned into a spike in step ``t + 1`` (one-step delay). Any target symbol used by a connection is mapped to the input field `g`.

# Integration
For each neuron, per step: `h += g`; `spikecount = g` if `g > 0` (else 0); `fire = g > 0`;
then `g = 0`.

# Fields
- `name::String = "identity"`, `id::String = randstring(12)`, `param::IdentityParam`.
- `N::Int32 = 100`.
- `g::Vector{Float32}`: input received in the current step (summed synaptic weights), reset
  each step.
- `spikecount::Vector{Float32}`: input of the last step for neurons that fired (0 otherwise).
- `h::Vector{Float32}`: cumulative input since construction.
- `fire::Vector{Bool}`; `records::Dict`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
P = SNN.Poisson(N = 10, param = SNN.PoissonParameter(50Hz))
Id = SNN.Identity(N = 10)
s = SNN.SpikingSynapse(P, Id, :g; conn = (μ = 1, p = 1.0))
model = SNN.compose(; P, Id, s)
SNN.monitor!(Id, [:fire])
SNN.sim!(model, 100ms)
```
"""
Identity
@snn_kw mutable struct Identity{VFT = Vector{Float32},IT = Int32} <: AbstractPopulation
    name::String = "identity"
    id::String = randstring(12)
    param::IdentityParam = IdentityParam()
    N::IT = 100
    g::VFT = zeros(N)
    spikecount::VFT = zeros(N)
    h::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    records::Dict = Dict()
end

function integrate!(p::Identity, param::IdentityParam, dt::Float32)
    @unpack g, h, fire, spikecount = p
    for i in eachindex(g)
        h[i] += g[i]
        spikecount[i] = 0.0f0
        if g[i] > 0
            fire[i] = true
            spikecount[i] += Float32(g[i])
        else
            fire[i] = false
        end
        g[i] = 0
    end
end

function Population(::IdentityParam; kwargs...)
    return Identity(; kwargs...)
end

function synaptic_target(targets::Dict, post::Identity, sym::Symbol, target)
    g = getfield(post, :g)
    v_post = zeros(Float32, length(g))
    push!(targets, :sym => :g)
    return g, v_post
end



export Identity, IdentityParam
