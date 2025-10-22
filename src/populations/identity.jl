struct IdentityParam <: AbstractPopulationParameter end

"""
    Identity{VFT, IT} <: AbstractPopulation

A simple population type that acts as an identity function, passing input directly to output.

# Fields
- `VFT`: Vector type for storing floating-point values (default: `Vector{Float32}`)
- `IT`: Integer type for storing population size (default: `Int32`)

This population type is useful for testing and as a building block in more complex networks.
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

function synaptic_target(targets::Dict, post::Identity, sym::Symbol, target::Nothing)
    v = :spikecount
    g = getfield(post, sym)
    v_post = getfield(post, v)
    push!(targets, :sym => sym)
    return g, v_post
end



export Identity, IdentityParam
