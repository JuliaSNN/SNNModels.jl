using SNNModels
using Test
@load_units

@testset "SNNModels.WilsonCowan" begin

    @testset "Default construction" begin
        wc = WilsonCowan()
        @test wc isa WilsonCowan
        @test wc.N == 100
        @test length(wc.r) == 100
        @test length(wc.x) == 100
        @test length(wc.g) == 100
        @test wc.records isa Dict
        @test hasproperty(wc, :id)
        @test hasproperty(wc, :name)
    end

    @testset "Custom N" begin
        wc = WilsonCowan(N = 30)
        @test wc.N == 30
        @test length(wc.r) == 30
    end

    @testset "integrate! — runs without error" begin
        wc = WilsonCowan(N = 20)
        monitor!(wc, [:r])
        sim!([wc]; duration = 100ms)
        @test true
    end

    @testset "r = tanh(x) invariant at construction" begin
        wc = WilsonCowan(N = 10)
        @test all(wc.r .≈ tanh.(wc.x))
    end

end
true
