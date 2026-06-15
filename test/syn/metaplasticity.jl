using SNNModels
using Test
@load_units

@testset "Metaplasticity" begin

    @testset "MultiplicativeNorm / AdditiveNorm — constructors" begin
        mn = MultiplicativeNorm(τ = 100ms)
        @test mn isa MultiplicativeNorm
        @test mn.τ ≈ 100ms

        an = AdditiveNorm(τ = 200ms)
        @test an isa AdditiveNorm
        @test an.τ ≈ 200ms
    end

    @testset "SynapseNormalization — MultiplicativeNorm" begin
        E  = IF(N = 20)
        s1 = SpikingSynapse(E, E, :ge; conn = (p = 0.5f0, μ = 1.0f0))
        s2 = SpikingSynapse(E, E, :ge; conn = (p = 0.3f0, μ = 0.5f0))
        norm = SynapseNormalization([s1, s2]; param = MultiplicativeNorm(τ = 100ms))
        @test norm isa SynapseNormalization
        @test length(norm.W0) == E.N
        @test length(norm.W1) == E.N
        @test norm.targets[:post] == E.id
        @test length(norm.targets[:synapses]) == 2
    end

    @testset "SynapseNormalization — AdditiveNorm" begin
        E  = IF(N = 15)
        s  = SpikingSynapse(E, E, :ge; conn = (p = 0.5f0, μ = 1.0f0))
        norm = SynapseNormalization([s]; param = AdditiveNorm(τ = 200ms))
        @test norm isa SynapseNormalization
        @test length(norm.W0) == E.N
    end

    @testset "SynapseNormalization — rejects different post populations" begin
        E = IF(N = 10)
        I = IF(N = 10)
        s1 = SpikingSynapse(E, E, :ge; conn = (p = 0.5f0, μ = 1.0f0))
        s2 = SpikingSynapse(E, I, :ge; conn = (p = 0.5f0, μ = 1.0f0))
        @test_throws AssertionError SynapseNormalization([s1, s2]; param = MultiplicativeNorm(τ = 100ms))
    end

    @testset "SynapseNormalization — W0 sums initial weights" begin
        E = IF(N = 10)
        I = IF(N = 10)
        # use distinct pre/post to avoid diagonal removal (pre==post zeros autapses)
        s = SpikingSynapse(E, I, :ge; conn = (p = 1.0f0, μ = 1.0f0))
        norm = SynapseNormalization([s]; param = MultiplicativeNorm(τ = 100ms))
        # full connectivity, μ=1 → each post neuron receives N pre connections each of weight 1
        @test all(norm.W0 .≈ E.N)
    end

    @testset "AggregateScalingParameter — constructor" begin
        p = AggregateScalingParameter(10, 5Hz; τ = 10ms, τa = 100ms, τe = 100ms)
        @test p isa AggregateScalingParameter
        @test length(p.Y) == 10
        @test p.τ  ≈ 10ms
        @test p.τa ≈ 100ms
    end

    @testset "RandomTurnover — constructor" begin
        rt = RandomTurnover(rate = 0.01f0, threshold = 0.1f0)
        @test rt isa RandomTurnover
        @test rt.rate      ≈ 0.01f0
        @test rt.threshold ≈ 0.1f0
    end

    @testset "ActivityDependentTurnover — constructor" begin
        at = ActivityDependentTurnover(rate = 0.01f0, fraction = 0.05f0)
        @test at isa ActivityDependentTurnover
        @test at.fraction ≈ 0.05f0
    end

end
true
