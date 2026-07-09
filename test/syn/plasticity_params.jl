using SNNModels
using Test
@load_units

@testset "LTP parameters" begin

    @testset "STDPGerstner" begin
        p = STDPGerstner()
        @test p isa STDPGerstner
        @test hasproperty(p, :A_pre) && hasproperty(p, :A_post)
        @test hasproperty(p, :τpre) && hasproperty(p, :τpost)
        @test hasproperty(p, :Wmax) && hasproperty(p, :Wmin)

        p2 = STDPGerstner(τpre = 30ms, τpost = 25ms, Wmax = 50pF)
        @test p2.τpre  ≈ 30ms
        @test p2.τpost ≈ 25ms
        @test p2.Wmax  ≈ 50pF
    end

    @testset "STDPConfavreux2025" begin
        p = STDPConfavreux2025()
        @test p isa STDPConfavreux2025
        @test hasproperty(p, :η) && hasproperty(p, :κ) && hasproperty(p, :γ)
        @test hasproperty(p, :τpre) && hasproperty(p, :τpost)

        p2 = STDPConfavreux2025(η = 0.02, κ = 2.0f0, γ = 0.5f0)
        @test p2.η ≈ 0.02f0
        @test p2.κ ≈ 2.0f0
    end

    @testset "STDPMexicanHat" begin
        p = STDPMexicanHat()
        @test p isa STDPMexicanHat
        @test hasproperty(p, :A) && hasproperty(p, :τ)

        p2 = STDPMexicanHat(A = 0.05, τ = 30ms)
        @test p2.τ ≈ 30ms
    end

    @testset "iSTDPRate" begin
        p = iSTDPRate()
        @test p isa iSTDPRate
        @test hasproperty(p, :η) && hasproperty(p, :r) && hasproperty(p, :τy)

        p2 = iSTDPRate(η = 0.02pA, r = 5Hz, τy = 80ms)
        @test p2.r  ≈ 5Hz
        @test p2.τy ≈ 80ms
    end

    @testset "iSTDPTime" begin
        p = iSTDPTime()
        @test p isa iSTDPTime
        @test hasproperty(p, :η) && hasproperty(p, :τy)
    end

    @testset "iSTDPPotential" begin
        p = iSTDPPotential()
        @test p isa iSTDPPotential
        @test hasproperty(p, :η) && hasproperty(p, :v0) && hasproperty(p, :τy)

        p2 = iSTDPPotential(v0 = -55mV, τy = 150ms)
        @test p2.v0 ≈ -55mV
        @test p2.τy ≈ 150ms
    end

    @testset "vSTDPParameter" begin
        p = vSTDPParameter()
        @test p isa vSTDPParameter
        @test hasproperty(p, :A_LTD) && hasproperty(p, :A_LTP)
        @test hasproperty(p, :θ_LTD) && hasproperty(p, :θ_LTP)

        p2 = vSTDPParameter(A_LTD = 1e-3, A_LTP = 2e-3, Wmax = 50pF)
        @test p2.A_LTD < p2.A_LTP
        @test p2.Wmax  ≈ 50pF
    end

    @testset "STDPSymmetric / STDPAntiSymmetric" begin
        s = STDPSymmetric()
        @test s isa STDPSymmetric
        @test hasproperty(s, :A_x)

        a = STDPAntiSymmetric()
        @test a isa STDPAntiSymmetric
        @test hasproperty(a, :A_x)
    end

end

@testset "STP parameters" begin

    @testset "MarkramSTPParameter (Event)" begin
        p = MarkramSTPParameter()
        @test p isa MarkramSTPParameterEvent
        @test hasproperty(p, :τD) && hasproperty(p, :τF) && hasproperty(p, :U)

        p2 = MarkramSTPParameter(τD = 400ms, τF = 100ms, U = 0.3f0)
        @test p2.τD ≈ 400ms
        @test p2.U  ≈ 0.3f0
    end

    @testset "MarkramSTPParameterTimestep" begin
        p = MarkramSTPParameterTimestep()
        @test p isa MarkramSTPParameterTimestep
        @test hasproperty(p, :τD) && hasproperty(p, :U)
    end

    @testset "MarkramSTPParameterHet" begin
        N   = 10
        het = MarkramSTPParameterHet(
            τD = fill(200ms, N),
            τF = fill(1500ms, N),
            U  = fill(0.2f0, N),
        )
        @test het isa MarkramSTPParameterHet
        @test length(het.U)  == N
        @test length(het.τD) == N
    end

end

@testset "Plasticity variable allocation" begin

    @testset "STDPVariables" begin
        v = SNNModels.STDPVariables(Npre = 15, Npost = 10)
        @test length(v.tpre)    == 15
        @test length(v.tpost)   == 10
        @test length(v.last_pre)  == 15
        @test length(v.last_post) == 10
        @test v.active == [true]
    end

    @testset "iSTDPVariables" begin
        v = SNNModels.iSTDPVariables(Npre = 12, Npost = 8)
        @test length(v.tpre)  == 12
        @test length(v.tpost) == 8
        @test v.active == [true]
    end

    @testset "MarkramSTPVariables" begin
        v = SNNModels.MarkramSTPVariables(Npre = 10, Npost = 5)
        @test length(v.u) == 10
        @test length(v.x) == 10
        @test all(v.x .== 1.0f0)
        @test v.active == [true]
    end

    @testset "SpikingSynapse + STDPGerstner — variables allocated" begin
        E = IF(N = 20)
        s = SpikingSynapse(E, E, :ge;
            conn     = (p = 0.4f0, μ = 0.5f0),
            LTPParam = STDPGerstner())
        @test s.LTPVars isa SNNModels.STDPVariables
        @test length(s.LTPVars.tpre)  == E.N
        @test length(s.LTPVars.tpost) == E.N
    end

    @testset "SpikingSynapse + iSTDPRate — variables allocated" begin
        pre  = IF(N = 20)
        post = IF(N = 10)
        s    = SpikingSynapse(pre, post, :gi;
            conn     = (p = 0.5f0, μ = 1.0f0),
            LTPParam = iSTDPRate())
        @test s.LTPVars isa SNNModels.iSTDPVariables
        @test length(s.LTPVars.tpre)  == pre.N
        @test length(s.LTPVars.tpost) == post.N
    end

    @testset "SpikingSynapse + MarkramSTP — variables allocated" begin
        E = IF(N = 20)
        s = SpikingSynapse(E, E, :ge;
            conn     = (p = 0.5f0, μ = 0.5f0),
            STPParam = MarkramSTPParameter())
        @test s.STPVars isa SNNModels.MarkramSTPVariables
        @test length(s.STPVars.u) == E.N
    end

end

@testset "set_LTP! / set_STP!" begin
    E = IF(N = 15)
    s = SpikingSynapse(E, E, :ge;
        conn     = (p = 0.5f0, μ = 0.5f0),
        LTPParam = STDPGerstner(),
        STPParam = MarkramSTPParameter())

    set_LTP!(s, false);  @test !s.LTPVars.active[1]
    set_LTP!(s, true);   @test  s.LTPVars.active[1]
    set_STP!(s, false);  @test !s.STPVars.active[1]
    set_STP!(s, true);   @test  s.STPVars.active[1]
end
true
