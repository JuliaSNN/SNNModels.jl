using SNNModels
using Test
using Distributions, Random
@load_units

@testset "HetRec" begin

    @testset "HetRecParameter — defaults" begin
        p = HetRecParameter()
        @test p.Nd        == 2
        @test p.τm        ≈ 20ms
        @test p.τrate     ≈ 100ms
        @test p.steepness ≈ 1.0f0
        @test p.τabs      ≈ 5ms
        @test p.overlap   ≈ 0.5f0
    end

    @testset "HetRecParameter — custom" begin
        p = HetRecParameter(
            Nd = 4, overlap = 0.0f0,
            τm = 30ms, τrate = 150ms,
            τd = Uniform(20.0f0, 80.0f0),
        )
        @test p.Nd      == 4
        @test p.overlap ≈ 0.0f0
        @test p.τm      ≈ 30ms
    end

    @testset "Population(HetRecParameter) — construction" begin
        p   = HetRecParameter(Nd = 2, overlap = 0.0f0)
        pop = Population(p; N = 20)
        @test pop isa HetRec
        @test pop.N == 20
        @test length(pop.v_s)  == 20
        @test length(pop.v_d)  == 40    # N * Nd
        @test length(pop.fire) == 20
        @test length(pop.τd)   == 40
        @test pop.records isa Dict
        @test hasproperty(pop, :id)
    end

    @testset "HetRec — τd sampled from distribution" begin
        p   = HetRecParameter(Nd = 3, τd = Uniform(10.0f0, 100.0f0))
        pop = Population(p; N = 10)
        @test all(10ms .<= pop.τd .<= 100ms)
    end

    @testset "HetRec — integrate! runs" begin
        p   = HetRecParameter(Nd = 2, overlap = 0.0f0)
        pop = Population(p; N = 15)
        stim = Stimulus(PoissonFixed(rate = 30Hz, μ = 1.0f0), pop, :glu)  # HetRec uses :glu/:gaba not :ge/:gi
        model = compose(pop = pop, stim = stim, silent = true)
        sim!(model, 100ms)
        @test true
    end

end
true

@testset "HetRec — adaptive baseline and soma do not depend on dt" begin
    # deterministic check without spikes (r = 0): integrate 200 ms at two dt
    function run(dt)
        p = HetRecParameter(Nd = 2, overlap = 0.5f0, rate = Uniform(0.0f0, 1.0f-9))
        Random.seed!(3)
        pop = Population(p; N = 5)
        pop.v_d .= range(1.0f0, 10.0f0, length = length(pop.v_d))
        pop.τd .= 1.0f6 # frozen dendrites
        for _ = 1:round(Int, 200 / dt)
            SNNModels.integrate!(pop, pop.param, Float32(dt))
        end
        copy(pop.v_s), copy(pop.trace)
    end
    vs1, tr1 = run(0.125)
    vs2, tr2 = run(0.0625)
    @test isapprox(vs1, vs2; rtol = 1e-2)
    @test isapprox(tr1, tr2; rtol = 1e-2)
end
