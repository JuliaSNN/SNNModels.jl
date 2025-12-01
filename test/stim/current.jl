using SNNModels
using Test
using Distributions

@testset "CurrentStimulus" begin
    # Create the post population
    E = IF(; N = 3200, param = IFParameter(; El = -49mV))

    # Test CurrentNoise constructor with default parameters
    @testset "CurrentNoise with default parameters" begin
        param = CurrentNoise(3200)
        @test param.I_base ≈ zeros(Float32, 3200)
        @test param.I_dist == Distributions.Normal(0.0, 0.0)
        @test param.α ≈ zeros(Float32, 3200)
    end

    # Test CurrentNoise constructor with custom parameters
    @testset "CurrentNoise with custom parameters" begin
        param = CurrentNoise(3200; I_base = 10pA, I_dist = Normal(0.0, 1.0), α = 0.1)
        @test param.I_base ≈ fill(10pA, 3200)
        @test param.I_dist == Normal(0.0, 1.0)
        @test param.α ≈ fill(0.1, 3200)
    end

    # Test CurrentStimulus constructor with default parameters
    @testset "CurrentStimulus with default parameters" begin
        param = CurrentNoise(3200)
        stim = CurrentStimulus(E, :I; param = param)
        @test stim.param == param
        @test stim.name == "Current"
        @test stim.neurons == 1:E.N
        @test stim.I ≈ zeros(Float32, E.N)
        @test stim.targets[:pre] == :Current
        @test stim.targets[:post] == E.id
        @test stim.targets[:sym] == :soma
        @test stim.targets[:type] == :CurrentStimulus
    end

    # Test CurrentStimulus constructor with custom parameters
    @testset "CurrentStimulus with custom parameters" begin
        param = CurrentNoise(3200; I_base = 10pA, I_dist = Normal(0.0, 1.0), α = 0.1)
        stim = CurrentStimulus(E, :I; param = param)
        @test stim.param == param
        @test stim.name == "Current"
        @test stim.neurons == 1:E.N
        @test stim.I ≈ zeros(Float32, E.N)
        @test stim.targets[:pre] == :Current
        @test stim.targets[:post] == E.id
        @test stim.targets[:sym] == :soma
        @test stim.targets[:type] == :CurrentStimulus
    end

    # Test Stimulus constructor with CurrentParameter
    @testset "Stimulus with CurrentParameter" begin
        param = CurrentNoise(3200; I_base = 10pA, I_dist = Normal(0.0, 1.0), α = 0.1)
        stim = Stimulus(param, E, :I)
        @test stim.param == param
        @test stim.name == "Current"
        @test stim.neurons == 1:E.N
        @test stim.I ≈ zeros(Float32, E.N)
        @test stim.targets[:pre] == :Current
        @test stim.targets[:post] == E.id
        @test stim.targets[:sym] == :soma
        @test stim.targets[:type] == :CurrentStimulus
    end

    # Test stimulate! method with CurrentStimulus
    @testset "stimulate! with CurrentStimulus" begin
        param = CurrentNoise(3200; I_base = 10pA, I_dist = Normal(0.0, 1.0), α = 0.1)
        stim = CurrentStimulus(E, :I; param = param)
        time = Time()
        dt = 0.001f0
        stimulate!(stim, param, time, dt)
        @test stim.I isa Vector{Float32}
    end

    # Test simulate with composed model
    @testset "Simulate with composed model" begin
        param = CurrentNoise(3200; I_base = 10pA, I_dist = Normal(0.0, 1.0), α = 0.1)
        stim = CurrentStimulus(E, :I; param = param)
        model = compose(E = E, S = stim, silent = true)
        monitor!(E, [:fire, :I])
        sim!(model; duration = 1s)
        @test true
    end
end
true
