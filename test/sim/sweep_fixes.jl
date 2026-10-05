# Regression tests for the bugs found by the documentation sweep (simulation loop level).
using SNNModels
using Test, Random, Statistics
@load_units

@testset "train! with IZ, HH, MorrisLecar and without connections" begin
    for P in (IZ(N = 3, param = IZParameter()), HH(N = 3), MorrisLecar(N = 3), IF(N = 3), AdEx(N = 3))
        @test train!([P]; duration = 5ms) isa SNNModels.Time
        @test sim!([P]; duration = 5ms) isa SNNModels.Time
    end
    E = IF(N = 5)
    @test train!([E], [EmptySynapse()]; duration = 5ms) isa SNNModels.Time
end

@testset "FORCE / PINning connections run under sim! and train!" begin
    Random.seed!(1)
    R = Rate(N = 100)
    F = FLSynapse(R, R; μ = 1.5, α = 1)
    dt = 0.125f0
    T = SNNModels.Time()
    err = Float32[]
    for k = 1:4000
        F.f = sin(2π * k * dt / 20)
        train!([R], [F], SNNModels.AbstractStimulus[], dt, T)
        push!(err, abs(F.z - F.f))
    end
    @test mean(err[end-400:end]) < mean(err[1:400])
    @test sim!([R], [F]; duration = 5ms) isa SNNModels.Time
    for C in (
        SNNModels.FLSparseSynapse(R, R; μ = 1.5, p = 0.2),
        PINningSynapse(R, R),
        SNNModels.PINningSparseSynapse(R, R; p = 0.2),
    )
        @test sim!([R], [C]; duration = 5ms) isa SNNModels.Time
        @test train!([R], [C]; duration = 5ms) isa SNNModels.Time
    end
    @test_throws ArgumentError SNNModels.FLSparseSynapse(R, R)
end

@testset "MorrisLecar, ExtendedIF, WilsonCowan receive connections" begin
    E = Poisson(N = 50, param = PoissonParameter(50Hz))
    M = MorrisLecar(N = 5)
    X = ExtendedIF(N = 5)
    sM = SpikingSynapse(E, M, :ge; conn = (p = 1.0, μ = 1.0))
    sX = SpikingSynapse(E, X, :glu; conn = (p = 1.0, μ = 1.0))
    sX2 = SpikingSynapse(E, X, :g_SST; conn = (p = 1.0, μ = 1.0))
    @test sM.g === M.ge
    @test sX.g === X.g_Exc && sX2.g === X.g_SST
    monitor!(M, [:ge]); monitor!(X, [:g_Exc])
    sim!([E, M, X], [sM, sX, sX2]; duration = 50ms)
    @test maximum(getvariable(M, :ge)) > 0
    @test maximum(getvariable(X, :g_Exc)) > 0
    @test_throws ArgumentError SpikingSynapse(E, M, :g_SST; conn = (p = 1.0, μ = 1.0))
    R = Rate(N = 10)
    W = WilsonCowan(N = 10)
    RW = RateSynapse(R, W; μ = 1.0, p = 0.5)
    @test RW.g === W.g
    @test sim!([R, W], [RW]; duration = 5ms) isa SNNModels.Time
end

@testset "HH and MorrisLecar: one spike flag per action potential" begin
    function flags_and_peaks(P, I, dt, T)
        P.I .= I
        nflag = 0; npeak = 0; vprev = P.v[1]; up = false
        for _ = 1:round(Int, T / dt)
            SNNModels.integrate!(P, P.param, Float32(dt))
            nflag += P.fire[1]
            # count action potentials as local maxima above the detection threshold
            if P.v[1] < vprev && up && vprev > (P isa HH ? -20.0f0 : 20.0f0)
                npeak += 1
            end
            up = P.v[1] > vprev
            vprev = P.v[1]
        end
        nflag, npeak
    end
    H = HH(N = 1); H.ge .= 0; H.gi .= 0
    nf, np = flags_and_peaks(H, 500.0f0, 0.01f0, 200.0f0)
    @test np > 2 && nf == np
    # Default MorrisLecar fires once and then stays depolarised under a constant current:
    # use three current pulses separated by pauses.
    M = MorrisLecar(N = 1)
    nf = 0; np = 0
    for _ = 1:3
        a, b = flags_and_peaks(M, 100.0f0, 0.05f0, 50.0f0); nf += a; np += b
        a, b = flags_and_peaks(M, 0.0f0, 0.05f0, 300.0f0); nf += a; np += b
    end
    @test nf == 3 # one action potential per pulse (peak > 50 mV)
    @test SNNModels.MorrisLecar_w_nullcline(0.0f0, M.param) ≈ 0.5f0
end
