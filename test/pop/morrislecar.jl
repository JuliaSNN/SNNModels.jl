using SNNModels
using Test
@load_units

@testset "MorrisLecar" begin

    @testset "Default construction" begin
        E = MorrisLecar()
        @test E isa MorrisLecar
        @test E.N == 100
        @test length(E.v)    == 100
        @test length(E.w)    == 100
        @test length(E.fire) == 100
        @test length(E.ge)   == 100
        @test length(E.gi)   == 100
        @test E.records isa Dict
        @test hasproperty(E, :id)
        @test hasproperty(E, :name)
    end

    @testset "Custom N" begin
        E = MorrisLecar(N = 20)
        @test E.N == 20
        @test length(E.v) == 20
    end

    @testset "Custom parameter" begin
        p = SNNModels.MorrisLecarParameter(gCa = 2nS, gK = 3nS, τe = 8ms)
        E = MorrisLecar(N = 5, param = p)
        @test E.param.gCa ≈ 2nS
        @test E.param.gK  ≈ 3nS
        @test E.param.τe  ≈ 8ms
    end

    @testset "integrate! — runs without error" begin
        E = MorrisLecar(N = 10)
        monitor!(E, [:v, :fire])
        sim!([E]; duration = 100ms)
        @test true
    end

    @testset "integrate! — with current injection" begin
        E = MorrisLecar(N = 5)
        E.I .= 5.0f0
        monitor!(E, [:v, :fire])
        sim!([E]; duration = 200ms)
        @test true
    end

end
true
