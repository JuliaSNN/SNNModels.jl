using SNNModels
using Test

@testset "PoissonLayer" begin
    # Create the post population
    E = IF(; N = 3200, param = IFParameter(; El = -49mV))

    # Test PoissonLayer constructor with default parameters
    @testset "PoissonLayer with default parameters" begin
        param = PoissonLayer()
        @test param.rate ≈ 1.0f0
        @test param.N == 1
        @test param.rates ≈ [1.0f0]
        @test param.p ≈ 0.1f0
        @test param.μ ≈ 1.0f0
        @test param.σ ≈ 0.0f0
        @test param.dist == :Normal
        @test param.rule == :Fixed
        @test param.active == [true]
    end

    # Test PoissonLayer constructor with custom parameters
    @testset "PoissonLayer with custom parameters" begin
        param = PoissonLayer(; rate = 2.0f0, N = 10, p = 0.2f0, μ = 2.0f0, σ = 1.0f0, dist = :Uniform, rule = :Random, active = [false])
        @test param.rate ≈ 2.0f0
        @test param.N == 10
        @test param.rates ≈ fill(2.0f0, 10)
        @test param.p ≈ 0.2f0
        @test param.μ ≈ 2.0f0
        @test param.σ ≈ 1.0f0
        @test param.dist == :Uniform
        @test param.rule == :Random
        @test param.active == [false]
    end

    # Test PoissonLayer constructor with rate and N
    @testset "PoissonLayer with rate and N" begin
        param = PoissonLayer(2.0f0; N = 10)
        @test param.rate ≈ 2.0f0
        @test param.N == 10
        @test param.rates ≈ fill(2.0f0, 10)
    end

    # Test Stimulus constructor with PoissonLayer
    @testset "Stimulus with PoissonLayer" begin
        param = PoissonLayer(; rate = 2.0f0, N = 10, p = 0.2f0, μ = 0.0f0, σ = 10.0f0, dist = :Uniform, rule = :Fixed, active = [false])
        stim = Stimulus(param, E, :ge)
        @test stim.param == param
        @test stim.N == param.N
        @test stim.targets[:pre] == :PoissonStim
        @test stim.targets[:post] == E.id
        @test stim.name == "Poisson"
    end

    # Test stimulate! method with PoissonLayer
    @testset "stimulate! with PoissonLayer" begin
        param = PoissonLayer(; rate = 2.0f0, N = 10, p = 0.2f0, μ = 2.0f0, σ = 1.0f0, dist = :Normal, rule = :Fixed, active = [false])
        stim = Stimulus(param, E, :ge)
        time = Time()
        dt = 0.001f0
        stimulate!(stim, param, time, dt)
        @test stim.fire isa Vector{Bool}
        @test stim.g isa Vector{Float32}
    end

    # Test simulate with composed model
    @testset "Simulate with composed model" begin
        param = PoissonLayer(; rate = 2.0f0, N = 10, p = 0.2f0, μ = 2.0f0, σ = 1.0f0, dist = :Normal, rule = :Fixed, active = [false])
        stim = Stimulus(param, E, :ge)
        model = compose(E = E, S = stim, silent = true)
        monitor!(E, [:fire, :ge])
        sim!(model; duration = 1s)
        @test true
    end
end
true