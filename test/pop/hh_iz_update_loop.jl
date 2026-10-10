# Bit-identity of the vectorised `integrate!` of `HH` and `IZ` against the original loops of SNNModels <= 1.9 (kept here as the reference).
using Random
include(joinpath(@__DIR__, "_loop_equiv.jl"))

function _hh_ref!(p::HH, param, dt)
    (; N, v, m, n, h, ge, gi, fire, I) = p
    (; Cm, gl, El, Ek, En, gn, gk, Vt, τe, τi, Ee, Ei) = param
    @inbounds for i = 1:N
        v_old = v[i]
        m[i] +=
            dt * (
                0.32f0 * (13.0f0 - v[i] + Vt) /
                (exp((13.0f0 - v[i] + Vt) / 4.0f0) - 1.0f0) * (1.0f0 - m[i]) -
                0.28f0 * (v[i] - Vt - 40.0f0) /
                (exp((v[i] - Vt - 40.0f0) / 5.0f0) - 1.0f0) * m[i]
            )
        n[i] +=
            dt * (
                0.032f0 * (15.0f0 - v[i] + Vt) /
                (exp((15.0f0 - v[i] + Vt) / 5.0f0) - 1.0f0) * (1.0f0 - n[i]) -
                0.5f0 * exp((10.0f0 - v[i] + Vt) / 40.0f0) * n[i]
            )
        h[i] +=
            dt * (
                0.128f0 * exp((17.0f0 - v[i] + Vt) / 18.0f0) * (1.0f0 - h[i]) -
                4.0f0 / (1.0f0 + exp((40.0f0 - v[i] + Vt) / 5.0f0)) * h[i]
            )
        v[i] +=
            dt / Cm * (
                I[i] +
                gl * (El - v[i]) +
                ge[i] * (Ee - v[i]) +
                gi[i] * (Ei - v[i]) +
                gn * m[i]^3 * h[i] * (En - v[i]) +
                gk * n[i]^4 * (Ek - v[i])
            )
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        fire[i] = (v_old <= -20.0f0) & (v[i] > -20.0f0)
    end
end

function _iz_ref!(p::IZ, param, dt)
    (; N, v, u, fire, I) = p
    (; a, b, c, d) = param
    (; ge, gi) = p
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

function _run_hh_pair(N; nsteps = 200, seed = 1, dt = 0.01f0)
    Random.seed!(seed)
    p = HH(; N)
    p.v .= -80f0 .+ 70f0 .* rand(Float32, N)
    p.m .= rand(Float32, N); p.n .= rand(Float32, N); p.h .= rand(Float32, N)
    q = deepcopy(p)
    flds = (:v, :m, :n, :h, :ge, :gi, :fire, :I)
    nspk = 0
    for s in 1:nsteps
        Iv = 0.05f0 .* rand(Float32, N); p.I .= Iv; q.I .= Iv
        SNNModels.integrate!(p, p.param, dt); _hh_ref!(q, q.param, dt)
        loop_state_equal(p, q, flds) || return (false, s, loop_first_diff(p, q, flds), nspk)
        nspk += count(p.fire)
    end
    return (true, nsteps, nothing, nspk)
end

function _run_iz_pair(N; nsteps = 300, seed = 1, dt = 0.5f0, param = IZParameter(a = 0.02, b = 0.2, c = -65, d = 8))
    Random.seed!(seed)
    p = IZ(; N, param)
    p.v .= -80f0 .+ 100f0 .* rand(Float32, N)
    p.u .= 4f0 .* randn(Float32, N)
    q = deepcopy(p)
    flds = (:v, :u, :ge, :gi, :fire, :I)
    nspk = 0
    for s in 1:nsteps
        Iv = 20f0 .* rand(Float32, N); p.I .= Iv; q.I .= Iv
        inp = 0.5f0 .* rand(Float32, N) .* (rand(Float32, N) .< 0.2); p.ge .+= inp; q.ge .+= inp
        SNNModels.integrate!(p, p.param, dt); _iz_ref!(q, q.param, dt)
        loop_state_equal(p, q, flds) || return (false, s, loop_first_diff(p, q, flds), nspk)
        nspk += count(p.fire)
    end
    return (true, nsteps, nothing, nspk)
end

@testset "HH and IZ vectorised loops are bit-identical to the reference" begin
    for N in (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1001)
        ok, s, f, nspk = _run_hh_pair(N)
        @test ok
        ok, s, f, nspk = _run_iz_pair(N)
        @test ok
        N >= 100 && @test nspk > 0
    end
    @test _run_hh_pair(100; dt = 0.05f0)[1]
    @test _run_iz_pair(100; param = IZParameter(a = 0.1, b = 0.26, c = -55, d = 2))[1]
    let p = HH(N = 8)
        resize!(p.m, 3)
        @test_throws DimensionMismatch SNNModels.integrate!(p, p.param, 0.01f0)
    end
    let p = IZ(N = 8)
        resize!(p.u, 3)
        @test_throws DimensionMismatch SNNModels.integrate!(p, p.param, 0.5f0)
    end
    # no allocation per step that grows with N
    let p = HH(N = 10), q = HH(N = 10000)
        SNNModels.integrate!(p, p.param, 0.01f0); SNNModels.integrate!(q, q.param, 0.01f0)
        @test (@allocated SNNModels.integrate!(q, q.param, 0.01f0)) == (@allocated SNNModels.integrate!(p, p.param, 0.01f0))
    end
    let p = IZ(N = 10), q = IZ(N = 10000)
        SNNModels.integrate!(p, p.param, 0.5f0); SNNModels.integrate!(q, q.param, 0.5f0)
        @test (@allocated SNNModels.integrate!(q, q.param, 0.5f0)) == (@allocated SNNModels.integrate!(p, p.param, 0.5f0))
    end
end

true
