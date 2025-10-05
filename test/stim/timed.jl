using SNNModels
using Test

@testset "SpikeTimeStimulus" begin
    # Create the post population
    E = IF(; N = 3200, param = IFParameter(; El = -49mV))

    # Test SpikeTimeParameter constructor with default parameters
    @testset "SpikeTimeParameter with default parameters" begin
        param = SpikeTimeParameter()
        @test param.spiketimes == []
        @test param.neurons == []
    end

    # Test SpikeTimeParameter constructor with custom parameters
    @testset "SpikeTimeParameter with custom parameters" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        @test param.spiketimes ≈ spiketimes
        @test param.neurons == neurons
    end

    # Test SpikeTimeParameter constructor with spiketimes and neurons
    @testset "SpikeTimeParameter with spiketimes and neurons" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        @test param.spiketimes ≈ spiketimes
        @test param.neurons == neurons
    end

    # Test SpikeTimeParameter constructor with Spiketimes
    @testset "SpikeTimeParameter with Spiketimes" begin
        spiketimes = Spiketimes([[0.1, 0.2, 0.3], [1, 2, 3]])
        param = SpikeTimeParameter(spiketimes)
        @test param.spiketimes ≈ [0.1, 0.2, 0.3, 1, 2, 3]
        @test param.neurons == [1, 1, 1, 2, 2, 2]
    end

    # Test SpikeTimeStimulus constructor with default parameters
    @testset "SpikeTimeStimulus with default parameters" begin
        param = SpikeTimeParameter()
        stim = SpikeTimeStimulus(E, :ge; param = param)
        @test stim.N == 0
        @test stim.name == "SpikeTime"
        @test stim.param == param
        @test stim.next_spike == [Inf]
        @test stim.next_index == [-1]
        @test stim.fire == falses(0)
        @test stim.targets[:pre] == :SpikeTimeStim
        @test stim.targets[:post] == E.id
    end

    # Test SpikeTimeStimulus constructor with custom parameters
    @testset "SpikeTimeStimulus with custom parameters" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        stim = SpikeTimeStimulus(E, :ge; param = param)
        @test stim.N == 3
        @test stim.name == "SpikeTime"
        @test stim.param == param
        @test stim.next_spike ≈ [0.1]
        @test stim.next_index == [1]
        @test stim.fire == falses(3)
        @test stim.targets[:pre] == :SpikeTimeStim
        @test stim.targets[:post] == E.id
    end

    # Test SpikeTimeStimulusIdentity constructor
    @testset "SpikeTimeStimulusIdentity" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        stim = SpikeTimeStimulusIdentity(E, :ge; param = param)
        @test stim.N == E.N
        @test stim.name == "SpikeTime"
        @test stim.param == param
        @test stim.next_spike ≈ [0.1]
        @test stim.next_index == [1]
        @test stim.fire == falses(E.N)
        @test stim.targets[:pre] == :SpikeTimeStim
        @test stim.targets[:post] == E.id
    end

    # Test stimulate! method with SpikeTimeStimulus
    @testset "stimulate! with SpikeTimeStimulus" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        stim = SpikeTimeStimulus(E, :ge; param = param)
        time = Time()
        dt = 0.001f0
        stimulate!(stim, param, time, dt)
        @test stim.fire == falses(3)
        @test stim.g ≈ zeros(Float32, E.N)
    end

    # Test next_neuron method with SpikeTimeStimulus
    @testset "next_neuron with SpikeTimeStimulus" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        stim = SpikeTimeStimulus(E, :ge; param = param)
        @test next_neuron(stim) == 1
    end

    # Test shift_spikes! method with SpikeTimeParameter
    @testset "shift_spikes! with SpikeTimeParameter" begin
        spiketimes = Vector{Float32}([0.1, 0.2, 0.3] )
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        shift_spikes!(param, 0.1f0)
        @test param.spiketimes ≈ [0.2, 0.3, 0.4]
    end

    # Test shift_spikes! method with SpikeTimeStimulus
    @testset "shift_spikes! with SpikeTimeStimulus" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        stim = SpikeTimeStimulus(E, :ge; param = param)
        shift_spikes!(stim, 0.1)
        @test stim.param.spiketimes ≈ [0.2, 0.3, 0.4]
        @test stim.next_index == [1]
        @test stim.next_spike ≈ [0.2]
    end

    # Test update_spikes! method with SpikeTimeStimulus
    @testset "update_spikes! with SpikeTimeStimulus" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        stim = SpikeTimeStimulus(E, :ge; param = param)
        new_spiketimes = [0.4, 0.5, 0.6]
        new_neurons = [4, 5, 6]
        new_spikes = SpikeTimeParameter(new_spiketimes, new_neurons)
        update_spikes!(stim, new_spikes)
        @test stim.param.spiketimes ≈ new_spiketimes
        @test stim.param.neurons == new_neurons
        @test stim.next_index == [1]
        @test stim.next_spike ≈ [0.4]
    end

    # Test max_neuron method with SpikeTimeParameter
    @testset "max_neuron with SpikeTimeParameter" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        @test max_neuron(param) == 3
    end

    # Test simulate with composed model
    @testset "Simulate with composed model" begin
        spiketimes = [0.1, 0.2, 0.3]
        neurons = [1, 2, 3]
        param = SpikeTimeParameter(spiketimes, neurons)
        stim = SpikeTimeStimulus(E, :ge; param = param)
        model = compose(E = E, S = stim, silent = true)
        monitor!(E, [:fire, :ge])
        sim!(model; duration = 1s)
        @test true
    end
end
true