using SNNModels
using Test
using Random
@load_units

# Regression tests for two bugs reported against 1.8.1/1.8.2:
# - CurrentStimulus with a neuron subset indexed its random cache (length = number of
#   stimulated neurons) by neuron id, reading past its end for ids > length(neurons).
# - The positional MultiReceptorSynapse(syn) constructor referenced undefined names.

@testset "CurrentStimulus on a neuron subset" begin
    Random.seed!(7)
    pop = IF(N = 100)
    stim = CurrentStimulus(pop, :I; neurons = 51:100,
                           param = CurrentNoise(100; I_base = 10pA, I_dist = SNNModels.Normal(0, 1)))
    @test length(stim.randcache) == 50
    model = compose(; pop, stim, silent = true)
    @test (sim!(model = model, duration = 2ms, dt = 0.1f0, pbar = false); true)
    # only the stimulated neurons receive current, and each gets base + its own noise draw
    @test all(iszero, pop.I[1:50])
    @test all(!iszero, pop.I[51:100])
    @test length(unique(pop.I[51:100])) == 50
end

@testset "MultiReceptorSynapse positional constructor" begin
    kw = SNNModels.MultiReceptorSynapse(syn = SNNModels.SomaReceptors)
    pos = SNNModels.MultiReceptorSynapse(SNNModels.SomaReceptors)
    @test pos isa SNNModels.MultiReceptorSynapse
    @test pos.syn == kw.syn
    @test pos.NMDA == kw.NMDA
    @test keys(pos.receptors) == keys(kw.receptors)
end
