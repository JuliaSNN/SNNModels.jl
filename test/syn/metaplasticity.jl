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

    @testset "constructors read synced weights (event-driven vSTDPParameter)" begin
        # `W` of an event-driven vSTDP synapse is stale between syncs: the constructors must sync first
        function build_vs()
            dt = 0.125f0; nsteps = 4000
            clock = accumulate((a, _) -> a + dt, 1:nsteps; init = 0.0f0)
            rng = SNNModels.Random.MersenneTwister(4)
            idx = [(k, i) for k in 1:nsteps-1 for i in 1:40 if rand(rng) < 40 * dt * 1e-3]
            src  = Identity(N = 40, name = "src")
            post = IF(N = 20, param = IFParameter(El = -49mV), name = "post")
            stim = SpikeTimeStimulusIdentity(src, :g, param = SpikeTimeParameter(Float32.(clock[first.(idx)]), last.(idx)), name = "stim")
            B = SpikingSynapse(src, post, :ge; conn = (p = 0.5, μ = 2.0), LTPParam = vSTDPParameter(θ_LTD = -60.0f0, θ_LTP = -55.0f0, A_LTP = 5f-3, A_LTD = 1f-3), name = "B")
            return compose(; src, post, B, stim, silent = true), B
        end
        rowsums(B) = (Wc = Float64.(plasticity_value(B, :W)); w = zeros(20); for (s_, i_) in enumerate(B.I); w[i_] += Wc[s_]; end; w)
        m, B = build_vs(); train!(m, 400ms)
        @test maximum(abs.(Float64.(B.W) .- Float64.(plasticity_value(B, :W)))) > 1f-4   # the premise: W is stale here
        nrm = SynapseNormalization([B]; param = MultiplicativeNorm(τ = 100ms))
        @test nrm.W0 ≈ rowsums(B) atol = 1e-4
        m, B = build_vs(); train!(m, 400ms)
        agg = AggregateScaling(20, [B]; param = AggregateScalingParameter(20, 5Hz))
        @test agg.WT ≈ rowsums(B) atol = 1e-4
    end

end
true
