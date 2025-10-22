using SNNModels
using Test

@testset "PoissonStimulus" begin
    # Create the post population
    E = IF(; N = 3200, param = IFParameter(; El = -49mV))

    # Test PoissonVariable constructor with default parameters
    @testset "PoissonVariable with default parameters" begin
        variables = Dict(:a => 1.0, :b => 2.0)
        rate = (t, vars) -> vars[:a] + vars[:b]
        param = PoissonVariable(; variables, rate)
        @test param.variables == variables
        @test param.rate == rate
        @test param.μ ≈ 1.0f0
        @test param.active == [true]
    end

    # Test PoissonVariable constructor with custom parameters
    @testset "PoissonVariable with custom parameters" begin
        variables = Dict(:a => 1.0, :b => 2.0)
        rate = (t, vars) -> vars[:a] + vars[:b]
        param = PoissonVariable(; variables, rate, μ = 2.0f0, active = [false])
        @test param.variables == variables
        @test param.rate == rate
        @test param.μ ≈ 2.0f0
        @test param.active == [false]
    end

    # Test PoissonFixed constructor with default parameters
    @testset "PoissonFixed with default parameters" begin
        param = PoissonFixed()
        @test param.rate ≈ 0.0f0
        @test param.μ ≈ 1.0f0
        @test param.active == [true]
    end

    # Test PoissonFixed constructor with custom parameters
    @testset "PoissonFixed with custom parameters" begin
        param = PoissonFixed(; rate = 1.0f0, μ = 2.0f0, active = [false])
        @test param.rate ≈ 1.0f0
        @test param.μ ≈ 2.0f0
        @test param.active == [false]
    end

    # Test PoissonInterval constructor with default parameters
    @testset "PoissonInterval with default parameters" begin
        intervals = [[0.0, 1.0], [2.0, 3.0]]
        param = PoissonInterval(; rate = 1.0f0, intervals = intervals)
        @test param.rate ≈ 1.0f0
        @test param.intervals == intervals
        @test param.μ ≈ 1.0f0
        @test param.active == [true]
    end

    # Test PoissonInterval constructor with custom parameters
    @testset "PoissonInterval with custom parameters" begin
        intervals = [[0.0, 1.0], [2.0, 3.0]]
        param = PoissonInterval(;
            rate = 1.0f0,
            intervals = intervals,
            μ = 2.0f0,
            active = [false],
        )
        @test param.rate ≈ 1.0f0
        @test param.intervals == intervals
        @test param.μ ≈ 2.0f0
        @test param.active == [false]
    end

    # Test Stimulus constructor with PoissonVariable
    @testset "Stimulus with PoissonVariable" begin
        variables = Dict(:a => 1.0, :b => 2.0)
        rate = (t, vars) -> vars[:a] + vars[:b]
        param = PoissonVariable(; variables, rate)
        stim = Stimulus(param, E, :ge)
        @test stim.param == param
        @test stim.name == "Poisson"
        @test stim.neurons == 1:E.N
        @test stim.g ≈ zeros(Float32, E.N)
        @test stim.targets[:pre] == :PoissonStim
        @test stim.targets[:post] == E.id
    end

    # Test Stimulus constructor with PoissonFixed
    @testset "Stimulus with PoissonFixed" begin
        param = PoissonFixed(; rate = 1.0f0, μ = 2.0f0, active = [false])
        stim = Stimulus(param, E, :ge)
        @test stim.param == param
        @test stim.name == "Poisson"
        @test stim.neurons == 1:E.N
        @test stim.g ≈ zeros(Float32, E.N)
        @test stim.targets[:pre] == :PoissonStim
        @test stim.targets[:post] == E.id
    end

    # Test Stimulus constructor with PoissonInterval
    @testset "Stimulus with PoissonInterval" begin
        intervals = [[0.0, 1.0], [2.0, 3.0]]
        param = PoissonInterval(;
            rate = 1.0f0,
            intervals = intervals,
            μ = 2.0f0,
            active = [false],
        )
        stim = Stimulus(param, E, :ge)
        @test stim.param == param
        @test stim.name == "Poisson"
        @test stim.neurons == 1:E.N
        @test stim.g ≈ zeros(Float32, E.N)
        @test stim.targets[:pre] == :PoissonStim
        @test stim.targets[:post] == E.id
    end

    # Test get_poisson_rate method with PoissonVariable
    @testset "get_poisson_rate with PoissonVariable" begin
        variables = Dict(:a => 1.0, :b => 2.0)
        rate = (t, vars) -> vars[:a] + vars[:b]
        param = PoissonVariable(; variables, rate)
        time = Time()
        @test get_poisson_rate(param, time) ≈ 3.0
    end

    # Test get_poisson_rate method with PoissonFixed
    @testset "get_poisson_rate with PoissonFixed" begin
        param = PoissonFixed(; rate = 1.0f0, μ = 2.0f0, active = [false])
        time = Time()
        @test get_poisson_rate(param, time) ≈ 1.0
    end

    # Test get_poisson_rate method with PoissonInterval
    @testset "get_poisson_rate with PoissonInterval" begin
        intervals = [[0.0, 1.0], [2.0, 3.0]]
        param = PoissonInterval(;
            rate = 1.0f0,
            intervals = intervals,
            μ = 2.0f0,
            active = [false],
        )
        time = Time(0.5)
        @test get_poisson_rate(param, time) ≈ 1.0
        time = Time(1.5)
        @test get_poisson_rate(param, time) ≈ 0.0
    end

    # Test stimulate! method with PoissonVariable
    @testset "stimulate! with PoissonVariable" begin
        variables = Dict(:a => 1.0, :b => 2.0)
        rate = (t, vars) -> vars[:a] + vars[:b]
        param = PoissonVariable(; variables, rate)
        stim = Stimulus(param, E, :ge)
        time = Time(0.0)
        dt = 0.001f0
        stimulate!(stim, param, time, dt)
        @test stim.g isa Vector{Float32}
    end

    # Test stimulate! method with PoissonFixed
    @testset "stimulate! with PoissonFixed" begin
        param = PoissonFixed(; rate = 1.0f0, μ = 2.0f0, active = [false])
        stim = Stimulus(param, E, :ge)
        time = Time(0.0)
        dt = 0.001f0
        stimulate!(stim, param, time, dt)
        # @test 
        # stim.g == zeros(Float32, E.N)
    end

    # Test stimulate! method with PoissonInterval
    @testset "stimulate! with PoissonInterval" begin
        intervals = [[0.0, 1.0], [2.0, 3.0]]
        param = PoissonInterval(;
            rate = 1.0f0,
            intervals = intervals,
            μ = 2.0f0,
            active = [false],
        )
        stim = Stimulus(param, E, :ge)
        time = Time(0.5)
        dt = 0.001f0
        stimulate!(stim, param, time, dt)
        @test stim.g isa Vector{Float32}
    end

    # Test simulate with composed model
    @testset "Simulate with composed model" begin
        variables = Dict(:a => 1.0, :b => 2.0)
        rate = (t, vars) -> vars[:a] + vars[:b]
        param = PoissonVariable(; variables, rate)
        stim = Stimulus(param, E, :ge)
        model = compose(E = E, S = stim, silent = true)
        monitor!(E, [:fire, :ge])
        sim!(model; duration = 1s)
        @test true
    end
end
true
