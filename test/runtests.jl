using SNNModels
using Test
using Logging
using Random
@load_units

if VERSION > v"1.1"
    include("ctors.jl")
    include("records.jl")
    include("macros.jl")
end

errorlogger = ConsoleLogger(stderr, Logging.Error)
with_logger(errorlogger) do
    @testset "Neurons and stimuli" begin
        include("pop/hh_neuron.jl")
        include("pop/if_neuron.jl")
        include("pop/adex_neuron.jl")
        include("pop/iz_neuron.jl")
        include("pop/spiketime.jl")
        include("pop/ballandstick.jl")
        include("pop/tripod.jl")
        include("pop/dendrite.jl")
    end

    @testset "Stimuli" begin
        include("stim/poisson.jl")
        include("stim/poisson_layer.jl")
        include("stim/current.jl")
        include("stim/timed.jl")
        include("stim/balanced.jl")
    end

    # Set the default logger to output only errors:
    @testset "Networks and synapses" begin
        @test include("network/if_net.jl")
        @test include("network/chain.jl")
        @test include("network/iz_net.jl")
        @test include("network/hh_net.jl")
        @test include("network/oja.jl")
        @test include("network/rate_net.jl")
        @test include("network/stdp_demo.jl")
    end
end

# include("dendrite.jl")
# include("tripod_network.jl")
#include("tripod_soma.jl")
#include("tripod.jl")
#include("tripod_network.jl")
#include("spiketime.jl")
#include("ballandstick.jl")
