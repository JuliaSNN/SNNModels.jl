
@snn_kw struct InhomogeneousPoissonParam{FT = Float32} <: AbstractPopulationParameter
    β::FT = 0.0
    τ::FT = 50.0ms
    r0::FT = 1kHz
    rate_timescale::FT = 400ms
end

@snn_kw struct InhomogeneousPoisson{VFT = Vector{Float32},IT = Int32} <: AbstractPopulation
    id::String = randstring(12)
    param::InhomogeneousPoissonParam
    name::String = "InhomogeneousPoisson"
    ##
    N::IT=100
    fire::VBT = falses(N)
    r::VFT= ones(Float32, N) * param.r0
    noise::VFT = zeros(Float32, N)
    # sparse connectivity
    randcache_β::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
end

function Population(p::InhomogeneousPoissonParam; kwargs...)
    return InhomogeneousPoisson(param = p; kwargs...)
end

function integrate!(p::InhomogeneousPoisson, param::InhomogeneousPoissonParam, dt::Float32)
    @unpack N, randcache_β, fire = p
    ## Inhomogeneous Poisson process
    @unpack r0, β, τ, rate_timescale = param
    @unpack noise, r = p
    R(x::Float32, v0::Float32 = 0.0f0) = x > 0.0f0 ? x : v0

    re::Float32 = 0.0f0
    cc::Float32 = 0.0f0
    Erate::Float32 = 0.0f0
    rand!(randcache_β)
    fire .= false
    @inbounds @fastmath for i = 1:N
        re = randcache_β[i] - 0.5f0
        cc = 1.0f0 - dt / τ
        noise[i] = (noise[i] - re) * cc + re
        Erate = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
        r[i] += (r0 - Erate) / rate_timescale * dt
        @assert Erate >= 0
        if rand(Distributions.Poisson{Float32}(Erate * dt)) > 0
            fire[i] = true
        end
    end
end

export InhomogeneousPoisson, InhomogeneousPoissonParam, integrate!
