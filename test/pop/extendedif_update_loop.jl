# Bit-identity of the branchless `update_neuron!` and the vectorised `update_synapses!` of `ExtendedIF` against the original
# loops of SNNModels <= 1.9, kept here as the reference.
using Random
include(joinpath(@__DIR__, "_loop_equiv.jl"))

function _eif_ref_neuron!(p::ExtendedIF, param, dt)
    (; N, v, g_Exc, g_PV, g_SST, w, I, tabs, fire) = p
    (; El, E_i, E_e, τabs, gl, α) = param
    (; Vt, Vr) = param
    @inbounds for i = 1:N
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end
        dv =
            (
                gl * (El - v[i]) +
                g_Exc[i] * (E_e - v[i]) +
                g_PV[i] * (E_i - v[i]) +
                g_SST[i] * (E_i - v[i]) +
                -α * g_Exc[i] * g_SST[i] * (E_e - v[i]) +
                + I[i]
            ) / param.Cm
        v[i] += dt * dv
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
end

function _eif_ref_synapses!(p::ExtendedIF, param, dt)
    (; N, g_Exc, g_PV, g_SST) = p
    (; τe, τi) = param
    @inbounds for i = 1:N
        g_Exc[i] += dt * (-g_Exc[i] / τe)
        g_PV[i] += dt * (-g_PV[i] / τi)
        g_SST[i] += dt * (-g_SST[i] / τi)
    end
end

const _EIF_FIELDS = (:v, :g_Exc, :g_PV, :g_SST, :fire, :tabs, :I)

function _eif_pair(N; α = 0.0f0, τabs = 5ms, nsteps = 300, seed = 1)
    Random.seed!(seed)
    p = ExtendedIF(; N, param = ExtendedIFParameter(α = α, τabs = τabs))
    p.v .= -70f0 .+ 35f0 .* rand(Float32, N)
    p.tabs .= Float32.(rand(0:4, N))
    p.g_Exc .= 5f0 .* rand(Float32, N); p.g_PV .= 5f0 .* rand(Float32, N); p.g_SST .= 5f0 .* rand(Float32, N)
    q = deepcopy(p)
    dt = 0.1f0
    nspk = 0; nref = 0
    for n in 1:nsteps
        Iv = 3000f0 .* rand(Float32, N)
        p.I .= Iv; q.I .= Iv
        inp = 2f0 .* rand(Float32, N) .* (rand(Float32, N) .< 0.2)
        p.g_Exc .+= inp; q.g_Exc .+= inp                  # what a connection would add
        SNNModels.integrate!(p, p.param, dt)
        _eif_ref_synapses!(q, q.param, dt); _eif_ref_neuron!(q, q.param, dt)
        loop_state_equal(p, q, _EIF_FIELDS) || return (false, n, loop_first_diff(p, q, _EIF_FIELDS), nspk, nref)
        nspk += count(p.fire); nref += count(>(0), p.tabs)
    end
    return (true, nsteps, nothing, nspk, nref)
end

@testset "ExtendedIF branchless loops are bit-identical to the reference" begin
    for N in (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1001), α in (0.0f0, 0.01f0)
        ok, n, f, nspk, nref = _eif_pair(N; α)
        @test ok
        if N >= 100
            @test nspk > 0
            @test nref > 0
        end
    end
    @test _eif_pair(64; τabs = 0ms)[1]
    let p = ExtendedIF(N = 3, param = ExtendedIFParameter(τabs = 1ms))
        p.v .= [-50, -50, -30]; p.tabs .= [3, 0, 0]; p.I .= 0; p.g_Exc .= 0; p.g_PV .= 0; p.g_SST .= 0
        v0 = copy(p.v)
        update_neuron!(p, p.param, 0.1f0)
        @test p.v[1] == v0[1] && p.tabs[1] == 2 && !p.fire[1]
        @test p.fire[3] && p.v[3] == p.param.Vr && p.tabs[3] == 10
    end
    let p = ExtendedIF(N = 8)
        resize!(p.g_PV, 5)
        @test_throws DimensionMismatch update_neuron!(p, p.param, 0.1f0)
        @test_throws DimensionMismatch update_synapses!(p, p.param, 0.1f0)
    end
    let p = ExtendedIF(N = 10), q = ExtendedIF(N = 10000)
        update_neuron!(p, p.param, 0.1f0); update_neuron!(q, q.param, 0.1f0)
        @test (@allocated update_neuron!(q, q.param, 0.1f0)) == (@allocated update_neuron!(p, p.param, 0.1f0))
    end
end

true
