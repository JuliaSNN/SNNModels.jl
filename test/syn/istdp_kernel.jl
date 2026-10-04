using SNNModels
using Test
using Random
@load_units

# iSTDP kernels (iSTDPRate, iSTDPPotential) against a plain reference loop that walks the
# CSC arrays only (no rowptr/index indirection). Regression test for the former @turbo
# loop that reassigned its loop variable (`st = index[st]`).

function _ref_istdp!(W, tpre, tpost, syn, param, dt, vpost)
    colptr, I, J, fireI, fireJ = syn.colptr, syn.I, syn.J, syn.fireI, syn.fireJ
    pot = param isa iSTDPPotential
    for j in eachindex(fireJ)
        tpre[j] += dt * (-tpre[j]) / param.τy
        if fireJ[j]
            tpre[j] += 1
            target = pot ? param.v0 : 2 * param.r * param.τy
            for s = colptr[j]:(colptr[j+1]-1)
                W[s] = clamp(W[s] + param.η * (tpost[I[s]] - target), param.Wmin, param.Wmax)
            end
        end
    end
    for i in eachindex(fireI)
        if pot
            tpost[i] += dt * -(tpost[i] - vpost[i]) / param.τy
        else
            tpost[i] += dt * (-tpost[i]) / param.τy
            fireI[i] && (tpost[i] += 1)
        end
        if fireI[i]
            for s in eachindex(I)          # all synapses onto i, found by scanning CSC
                I[s] == i && (W[s] = clamp(W[s] + param.η * tpre[J[s]], param.Wmin, param.Wmax))
            end
        end
    end
end

@testset "iSTDP kernels vs plain reference" begin
    params = (
        iSTDPRate(η = 0.2f0, r = 5Hz, τy = 20ms, Wmin = 0.0f0, Wmax = 50.0f0),
        iSTDPPotential(η = 0.05f0, v0 = -55mV, τy = 20ms, Wmin = 0.0f0, Wmax = 50.0f0),
    )
    @testset "$(nameof(typeof(param)))" for param in params
        Random.seed!(11)
        pre, post = IF(N = 120), IF(N = 90)
        syn = SpikingSynapse(pre, post, :gi; conn = (p = 0.15f0, μ = 10.0f0, σ = 3.0f0),
                             LTPParam = param)
        vars = syn.LTPVars
        W = copy(syn.W)
        W0 = copy(syn.W)
        tpre, tpost = copy(vars.tpre), copy(vars.tpost)
        T = Time()
        dt = 0.1f0
        maxdiff = 0.0f0
        for step = 1:3000
            pre.fire .= rand(pre.N) .< 0.05
            post.fire .= rand(post.N) .< 0.05
            post.v .= -70.0f0 .+ 20.0f0 .* rand(Float32, post.N)
            update_time!(T, dt)
            SNNModels.plasticity!(syn, param, vars, dt, T)
            _ref_istdp!(W, tpre, tpost, syn, param, dt, post.v)
            maxdiff = max(maxdiff, maximum(abs.(W .- syn.W)))
        end
        @test syn.W ≈ W rtol = 1e-5
        @test vars.tpre ≈ tpre rtol = 1e-5
        @test vars.tpost ≈ tpost rtol = 1e-5
        @test maxdiff < 1e-3
        @test syn.W != W0                     # the rule did act
        @test extrema(syn.W)[1] >= param.Wmin && extrema(syn.W)[2] <= param.Wmax
    end
end
