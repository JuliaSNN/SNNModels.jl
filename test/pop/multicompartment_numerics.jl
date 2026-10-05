# Numerical regression tests for the Tripod and BallAndStick integrators.
using SNNModels
using Test
@load_units

# Right-hand side of the Tripod / BallAndStick equations (see the docstrings), without synapses.
function _rhs(x, p, dends, I, Id)
    a = p.adex
    vs, w = x[1], x[end]
    dv = similar(x)
    axial = 0.0
    for (k, d) in enumerate(dends)
        vd = x[1+k]
        ic = (vs - vd) * d.gax[1]
        axial += ic
        dv[1+k] = ((d.El[1] - vd) * d.gm[1] + ic + Id) / d.C[1]
    end
    dv[1] = (a.gl * (a.El - vs) + 1.0 * a.ΔT * exp((vs - p.θ[1]) / a.ΔT) - w - axial + I) / a.C
    dv[end] = (a.a * (vs - a.El) - w) / a.τw
    return dv
end

function _heun(x, p, dends, I, Id, dt)
    k1 = _rhs(x, p, dends, I, Id)
    k2 = _rhs(x .+ dt .* k1, p, dends, I, Id)
    return x .+ dt / 2 .* (k1 .+ k2)
end

function _tripod_spikes(dt; I = 1500.0f0, T = 120.0f0)
    p = Tripod(N = 1, param = TripodParameter(ds = [200um, 300um]))
    p.v_s .= -70.6f0; p.v_d1 .= -70.6f0; p.v_d2 .= -70.6f0; p.w_s .= 0.0f0
    p.I .= I
    st = Float32[]
    for k = 1:round(Int, T / dt)
        SNNModels.integrate!(p, p.param, Float32(dt))
        p.fire[1] && push!(st, k * dt)
    end
    return st, p.w_s[1]
end

@testset "Tripod one Heun step" begin
    p = Tripod(N = 1, param = TripodParameter(ds = [200um, 300um]))
    p.v_s .= -55.0f0; p.v_d1 .= -60.0f0; p.v_d2 .= -65.0f0; p.w_s .= 50.0f0
    p.θ .= p.adex.Vt; p.I .= 500.0f0; p.I_d .= 20.0f0; p.tabs .= 0
    x0 = Float64[-55.0, -60.0, -65.0, 50.0]
    dt = 0.1f0
    ref = _heun(x0, p, (p.d1, p.d2), 500.0, 20.0, dt)
    SNNModels.integrate!(p, p.param, dt)
    @test isapprox([p.v_s[1], p.v_d1[1], p.v_d2[1], p.w_s[1]], ref; rtol = 1e-5)
end

@testset "BallAndStick one Heun step" begin
    p = BallAndStick(N = 1, param = BallAndStickParameter(ds = [300um]))
    p.v_s .= -55.0f0; p.v_d .= -60.0f0; p.w_s .= 50.0f0
    p.θ .= p.adex.Vt; p.tabs .= 0
    x0 = Float64[-55.0, -60.0, 50.0]
    dt = 0.1f0
    ref = _heun(x0, p, (p.d,), 0.0, 0.0, dt)
    SNNModels.integrate!(p, p.param, dt)
    @test isapprox([p.v_s[1], p.v_d[1], p.w_s[1]], ref; rtol = 1e-5)
end

@testset "Tripod spike times converge with dt" begin
    st1, w1 = _tripod_spikes(0.025f0)
    st2, w2 = _tripod_spikes(0.0125f0)
    @test length(st1) == length(st2) > 2
    @test maximum(abs.(st1 .- st2)) <= 0.05
    @test abs(w1 - w2) < 0.5
end
