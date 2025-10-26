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

function synaptic_target(targets::Dict, post::T, sym=nothing, target=nothing) where {T<:IZ}
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end

"""
[Izhikevich Neuron](https://www.izhikevich.org/publications/spikes.htm)
"""
IZ

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
