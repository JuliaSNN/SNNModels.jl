# HH (Traub-Miles / Brette et al. 2007 form): removable singularities of the rate functions and the f-I curve.
using SNNModels
using Test

function _hh_count(I, dt, T; FT = Float32)
    E = HH(N = 1, VFT = Vector{FT})
    E.ge .= 0; E.gi .= 0; E.v .= -65; E.I .= I
    n = 0
    for _ = 1:round(Int, T / dt)
        SNNModels.integrate!(E, E.param, FT(dt))
        n += E.fire[1]
    end
    return n, E
end

@testset "HH" begin
    @testset "rate functions at their 0/0 point" begin
        # v = -50, -48, -23 mV give u = v - Vt = 13, 15, 40 mV (Vt = -63): alpha_m, alpha_n, beta_m are 0/0
        for v0 in (-50.0f0, -48.0f0, -23.0f0)
            E = HH(N = 1); E.ge .= 0; E.gi .= 0; E.v .= v0
            SNNModels.integrate!(E, E.param, 0.01f0)
            @test all(isfinite, (E.v[1], E.m[1], E.n[1], E.h[1]))
        end
        # the limit is continuous: gating step at the singular point lies between its neighbours
        step(v0) = (E = HH(N = 1; VFT = Vector{Float64}, param = HHParameter(FT = Float64)); E.ge .= 0; E.gi .= 0; E.v .= v0;
                    SNNModels.integrate!(E, E.param, 0.01); E.m[1])
        @test step(-50.0 - 1e-6) ≈ step(-50.0) rtol = 1e-4
        @test step(-50.0 + 1e-6) ≈ step(-50.0) rtol = 1e-4
    end
    @testset "f-I curve (dt = 0.01 ms) is monotonic and converged" begin
        Is = (0, 50, 100, 300, 500, 1500)
        n10 = [_hh_count(I, 0.01, 500)[1] for I in Is]
        n05 = [_hh_count(I, 0.005, 500)[1] for I in Is]
        @test n10[1] == 0
        @test all(diff(n10) .> 0)             # no plateau
        @test n10[end] > 4 * n10[2]
        @test all(abs.(n10 .- n05) .<= 2)     # converged in dt
        @test all(isfinite, _hh_count(500, 0.01, 100)[2].v)
    end
    @testset "Float64 population" begin
        @test _hh_count(500, 0.01, 500; FT = Float64)[1] > 20
    end
end
true
