# Bit-identity of the vectorised `integrate!` of `MorrisLecar`, `Rate` and `WilsonCowan` against the original loops of SNNModels <= 1.9
# (Base `tanh` / `cosh`, kept here as the reference).
using Random
include(joinpath(@__DIR__, "_loop_equiv.jl"))

function _ml_ref_dv(v::Float32, w::Float32, I::Float32, param)
    (; El, EK, ECa, gl, gK, gCa, V1, V2) = param
    m_ss = 0.5 * (1 + tanh((v - V1) / V2))
    return I + gl * (El - v) + gCa * (ECa - v) * m_ss + gK * (EK - v) * w
end
function _ml_ref_dw(v::Float32, w::Float32, param)
    (; V3, V4, ϕ) = param
    n_ss = 0.5 * (1 + tanh((v - V3) / V4))
    τ = 1 / (ϕ * cosh((v - V3) / (2V4)))
    return (n_ss - w) / τ
end
function _ml_ref!(p::MorrisLecar, param, dt)
    (; N, v, w, ge, gi, fire, I) = p
    (; Cm, τe, τi, Ee, Ei) = param
    @inbounds for i = 1:N
        v_old = v[i]
        v[i] += dt / Cm * _ml_ref_dv(v[i], w[i], I[i], param)
        w[i] += dt * _ml_ref_dw(v[i], w[i], param)
        v[i] += dt / Cm * (ge[i] * (Ee - v[i]) + gi[i] * (Ei - v[i]))
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        fire[i] = (v_old <= 20.0f0) & (v[i] > 20.0f0)
    end
end
function _rate_ref!(p, dt)
    (; N, x, r, g, I) = p
    @inbounds for i = 1:N
        x[i] += dt * (-x[i] + g[i] + I[i])
        r[i] = tanh(x[i])
        g[i] = 0.0f0
    end
end

function _run_ml_pair(N; nsteps = 300, seed = 1, dt = 0.1f0)
    Random.seed!(seed)
    p = MorrisLecar(; N)
    p.v .= -75f0 .+ 120f0 .* rand(Float32, N)
    p.w .= rand(Float32, N)
    q = deepcopy(p)
    flds = (:v, :w, :ge, :gi, :fire, :I)
    nspk = 0
    for s in 1:nsteps
        Iv = 300f0 .* rand(Float32, N); p.I .= Iv; q.I .= Iv
        inp = 3f0 .* rand(Float32, N) .* (rand(Float32, N) .< 0.2); p.ge .+= inp; q.ge .+= inp; p.gi .+= inp; q.gi .+= inp
        SNNModels.integrate!(p, p.param, dt); _ml_ref!(q, q.param, dt)
        loop_state_equal(p, q, flds) || return (false, s, loop_first_diff(p, q, flds), nspk)
        nspk += count(p.fire)
    end
    return (true, nsteps, nothing, nspk)
end

function _run_rate_pair(T, N; nsteps = 200, seed = 1, dt = 0.1f0)
    Random.seed!(seed)
    p = T(; N)
    p.x .= 4f0 .* randn(Float32, N)
    q = deepcopy(p)
    flds = (:x, :r, :g, :I)
    for s in 1:nsteps
        gv = 3f0 .* randn(Float32, N); p.g .= gv; q.g .= gv
        p.I .= 0.5f0 .* randn(Float32, N); q.I .= p.I
        SNNModels.integrate!(p, p.param, dt); _rate_ref!(q, dt)
        loop_state_equal(p, q, flds) || return (false, s, loop_first_diff(p, q, flds))
    end
    return (true, nsteps, nothing)
end

@testset "MorrisLecar, Rate, WilsonCowan vectorised loops are bit-identical to the reference" begin
    for N in (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1001)
        @test _run_ml_pair(N)[1]
        @test _run_rate_pair(Rate, N)[1]
        @test _run_rate_pair(WilsonCowan, N)[1]
        N >= 100 && @test _run_ml_pair(N)[4] > 0
    end
    @test _run_ml_pair(64; dt = 0.01f0)[1]
    for (T, dt) in ((MorrisLecar, 0.1f0), (Rate, 0.1f0), (WilsonCowan, 0.1f0))
        let p = T(N = 8)
            fld = T === MorrisLecar ? :w : :r
            resize!(getfield(p, fld), 3)
            @test_throws DimensionMismatch SNNModels.integrate!(p, p.param, dt)
        end
        let p = T(N = 10), q = T(N = 10000)
            SNNModels.integrate!(p, p.param, dt); SNNModels.integrate!(q, q.param, dt)
            @test (@allocated SNNModels.integrate!(q, q.param, dt)) == (@allocated SNNModels.integrate!(p, p.param, dt))
        end
    end
end

true
