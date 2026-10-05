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
    dv[1] = (a.gl * (a.El - vs) + a.gl * a.ΔT * exp((vs - p.θ[1]) / a.ΔT) - w - axial + I) / a.C
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
    # Spike detection on the time grid makes spike times first-order in dt.
    st1, w1 = _tripod_spikes(0.025f0)
    st2, w2 = _tripod_spikes(0.00625f0)
    @test length(st1) == length(st2) > 2
    @test maximum(abs.(st1 .- st2)) <= 0.15
    @test abs(w1 - w2) < 0.5
end

@testset "Tripod/BallAndStick spike threshold, dendritic El, external currents" begin
    # Default Vspike reproduces the previous hard-coded -10 mV threshold.
    @test Tripod(N = 1).Vspike == -10.0f0
    @test BallAndStick(N = 1).Vspike == -10.0f0
    # A threshold above AP_membrane can never be reached on the upstroke prediction:
    # a lower threshold detects spikes earlier.
    function nspikes(Vspike)
        p = Tripod(N = 1, param = TripodParameter(ds = [200um, 300um]), Vspike = Vspike)
        p.v_s .= -70.6f0; p.v_d1 .= -70.6f0; p.v_d2 .= -70.6f0; p.I .= 1500.0f0
        n = 0
        for _ = 1:2000
            SNNModels.integrate!(p, p.param, 0.05f0)
            n += p.fire[1]
        end
        n
    end
    @test nspikes(-10.0f0) > 0
    @test nspikes(-45.0f0) >= nspikes(-10.0f0)
    # Dendrites relax to their own El (default: adex.El).
    p = Tripod(N = 1, adex = AdExParameter(El = -65mV))
    @test all(p.d1.El .== -65.0f0) && all(p.d2.El .== -65.0f0)
    p.d1.El .= -60.0f0; p.d2.El .= -60.0f0
    p.v_s .= -60.0f0; p.v_d1 .= -60.0f0; p.v_d2 .= -60.0f0
    leak_old = 0.1f0 * (-65.0f0 + 60.0f0) * p.d1.gm[1] / p.d1.C[1] # step with the somatic El
    SNNModels.integrate!(p, p.param, 0.1f0)
    @test abs(p.v_d1[1] + 60.0f0) < abs(leak_old) / 2
    # BallAndStick external currents are applied.
    b = BallAndStick(N = 1, param = BallAndStickParameter(ds = [300um]))
    b.v_s .= -70.6f0; b.v_d .= -70.6f0
    b.Is .= 200.0f0
    for _ = 1:200
        SNNModels.integrate!(b, b.param, 0.1f0)
    end
    @test b.v_s[1] > -68.0f0
    b2 = BallAndStick(N = 1, param = BallAndStickParameter(ds = [300um]))
    b2.v_s .= -70.6f0; b2.v_d .= -70.6f0
    b2.Id .= 100.0f0
    for _ = 1:200
        SNNModels.integrate!(b2, b2.param, 0.1f0)
    end
    @test b2.v_d[1] > b2.v_s[1] > -70.5f0
end

@testset "Dendritic models: invalid targets and DeltaSynapse" begin
    E = SNNModels.Poisson(N = 5, param = PoissonParameter(10Hz))
    T = Tripod(N = 2)
    @test_throws ArgumentError SpikingSynapse(E, T, :glu, :d3; conn = (p = 1.0, μ = 1.0))
    @test SpikingSynapse(E, T, :glu, :d2; conn = (p = 1.0, μ = 1.0)) isa SpikingSynapse
    Td = Tripod(N = 2, soma_syn = DeltaSynapse())
    @test_throws ArgumentError SNNModels.integrate!(Td, Td.param, 0.1f0)
end

@testset "Refractory period survives rounding (up = τabs = 0.1 ms)" begin
    function rate(dt)
        p = Tripod(N = 1, param = TripodParameter(ds = [160um, 200um]),
                   spike = PostSpike(At = 10.0mV, τA = 30.0ms, τabs = 0.1ms, up = 0.1ms))
        p.v_s .= -70.6f0; p.v_d1 .= -70.6f0; p.v_d2 .= -70.6f0; p.I .= 1500.0f0
        n = 0
        for _ = 1:round(Int, 1000 / dt)
            SNNModels.integrate!(p, p.param, Float32(dt))
            n += p.fire[1]
        end
        n
    end
    r1, r2 = rate(0.1), rate(0.125)
    @test r1 < 100 && abs(r1 - r2) <= 3   # it was ~1100 Hz at dt = 0.125 ms
end
