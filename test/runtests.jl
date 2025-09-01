using SNNModels
using Test
using Logging
using Random
@load_units

if VERSION > v"1.1"
    include("ctors.jl")
    include("records.jl")
end
##

@testset "Neurons and stimuli" begin
    @test include("hh_neuron.jl")
    @test include("if_neuron.jl")
    @test include("if_inputs.jl")
    @test include("adex_neuron.jl")
    @test include("adex_inputs.jl")
    @test include("iz_neuron.jl")
    @test include("spiketime.jl")
    @test include("ballandstick.jl")
    @test include("poisson_stim.jl")
end

## Set the default logger to output only errors:
errorlogger = ConsoleLogger(stderr, Logging.Error)
with_logger(errorlogger) do
    @testset "Networks and synapses" begin
        @test include("if_net.jl")
        @test include("chain.jl")
        @test include("iz_net.jl")
        @test include("hh_net.jl")
        @test include("oja.jl")
        @test include("rate_net.jl")
        @test include("stdp_demo.jl")
    end
end

# include("dendrite.jl")
# include("tripod_network.jl")
#include("tripod_soma.jl")
#include("tripod.jl")
#include("tripod_network.jl")
#include("spiketime.jl")
#include("ballandstick.jl")
