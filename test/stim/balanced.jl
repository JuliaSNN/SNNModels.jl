using SNNModels
using Test

@testset "BalancedStimulus" begin
    # Create the post population
    E = IF(; N = 3200, param = IFParameter(; El = -49mV))

    # Test BalancedParameter constructor with default parameters
    @testset "BalancedParameter with default parameters" begin
        param = BalancedParameter()
        @test param.kIE ≈ 1.0
        @test param.β ≈ 0.0
        @test param.τ ≈ 50.0ms
        @test param.r0 ≈ 1kHz
        @test param.wIE ≈ 1.0
        @test param.same_input == false
    end

    # Test BalancedParameter constructor with custom parameters
    @testset "BalancedParameter with custom parameters" begin
        param = BalancedParameter(;
            kIE = 2.0,
            β = 0.1,
            τ = 100.0ms,
            r0 = 2kHz,
            wIE = 2.0,
            same_input = true,
        )
        @test param.kIE ≈ 2.0
        @test param.β ≈ 0.1
        @test param.τ ≈ 100.0ms
        @test param.r0 ≈ 2kHz
        @test param.wIE ≈ 2.0
        @test param.same_input == true
    end

    # Test BalancedStimulus constructor with default parameters
    @testset "BalancedStimulus with default parameters" begin
        param = BalancedParameter()
        stim = BalancedStimulus(E, :ge, :gi; param = param)
        @test stim.param == param
        @test stim.N == E.N
        @test stim.targets[:pre] == :BalancedStim
        @test stim.targets[:post] == E.id
        @test stim.r ≈ ones(Float32, E.N) * param.r0
        @test stim.noise ≈ zeros(Float32, E.N)
        @test stim.ge ≈ zeros(Float32, E.N)
        @test stim.gi ≈ zeros(Float32, E.N)
    end

    # Test BalancedStimulus constructor with custom parameters
    @testset "BalancedStimulus with custom parameters" begin
        param = BalancedParameter(;
            kIE = 2.0,
            β = 0.1,
            τ = 100.0ms,
            r0 = 2kHz,
            wIE = 2.0,
            same_input = true,
        )
        stim = BalancedStimulus(E, :ge, :gi; param = param)
        @test stim.param == param
        @test stim.N == E.N
        @test stim.targets[:pre] == :BalancedStim
        @test stim.targets[:post] == E.id
        @test stim.r ≈ ones(Float32, E.N) * param.r0
        @test stim.noise ≈ zeros(Float32, E.N)
        @test stim.ge ≈ zeros(Float32, E.N)
        @test stim.gi ≈ zeros(Float32, E.N)
    end

    # Test Stimulus constructor with BalancedParameter
    @testset "Stimulus with BalancedParameter" begin
        param = BalancedParameter(;
            kIE = 2.0,
            β = 0.1,
            τ = 100.0ms,
            r0 = 2kHz,
            wIE = 2.0,
            same_input = true,
        )
        stim = Stimulus(param, E, :ge)
        @test stim.param == param
        @test stim.N == E.N
        @test stim.targets[:pre] == :BalancedStim
        @test stim.targets[:post] == E.id
        @test stim.r ≈ ones(Float32, E.N) * param.r0
        @test stim.noise ≈ zeros(Float32, E.N)
        @test stim.ge ≈ zeros(Float32, E.N)
        @test stim.gi ≈ zeros(Float32, E.N)
    end

    # Test stimulate! method with BalancedStimulus
    @testset "stimulate! with BalancedStimulus" begin
        param = BalancedParameter(;
            kIE = 2.0,
            β = 0.1,
            τ = 100.0ms,
            r0 = 2kHz,
            wIE = 2.0,
            same_input = true,
        )
        stim = BalancedStimulus(E, :ge, :gi; param = param)
        time = Time()
        dt = 0.001f0
        stimulate!(stim, param, time, dt)
        @test stim.fire isa Vector{Bool}
        @test stim.ge isa Vector{Float32}
        @test stim.gi isa Vector{Float32}
    end

    # Test simulate with composed model
    @testset "Simulate with composed model" begin
        param = BalancedParameter(;
            kIE = 2.0,
            β = 0.1,
            τ = 100.0ms,
            r0 = 2kHz,
            wIE = 2.0,
            same_input = true,
        )
        stim = BalancedStimulus(E, :ge, :gi; param = param)
        model = compose(E = E, S = stim, silent = true)
        monitor!(E, [:fire, :ge, :gi])
        sim!(model; duration = 1s)
        @test true
    end
end
true

@testset "BalancedStimulus runs with defaults; rates per neuron" begin
    for same_input in (false, true)
        P = IF(N = 50)
        param = BalancedParameter(r0 = 1kHz, kIE = 0.5, same_input = same_input)
        stim = Stimulus(param, P, :ge)
        @test stim.ge !== stim.gi
        T = SNNModels.Time()
        dt = 0.125f0
        nsteps = 4000
        for _ = 1:nsteps
            SNNModels.stimulate!(stim, stim.param, T, dt)
        end
        rate_e = stim.ge ./ (nsteps * dt)   # w = 1: summed draws per ms
        rate_i = stim.gi ./ (nsteps * dt)
        @test all(rate_e .> 0.5)            # every neuron is driven
        @test 0.9 < sum(rate_e) / 50 < 1.6  # r0 = 1/ms, adaptive offset relaxes from 1.5 r0
        @test isapprox(sum(rate_i) / 50, 0.5; rtol = 0.05)
    end
    P = IF(N = 5)
    @test BalancedStimulus(P, :ge, :gi; param = 2kHz).param.r0 ≈ 2kHz
    @test_throws ArgumentError Stimulus(BalancedParameter(), P, :foo)
    @test sim!([P], [EmptySynapse()], [Stimulus(BalancedParameter(), P, :ge)]; duration = 10ms) isa SNNModels.Time
end
