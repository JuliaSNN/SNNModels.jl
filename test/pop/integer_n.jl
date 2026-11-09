using SNNModels
using Test
@load_units

# Any Integer population size is accepted; the synaptic state is built with an Int size.
@testset "Integer N" begin
    for IT in (Int32, Int16, UInt8, Int64)
        pop = IF(N = IT(10))
        @test pop.N == 10
        @test length(pop.v) == 10
        @test length(pop.synvars.ge) == 10
        @test length(pop.receptors.glu) == 10
    end
    syn = DoubleExpSynapse()
    @test SNNModels.synaptic_variables(syn, Int32(7)) isa SNNModels.DoubleExpSynapseVars
    @test length(SNNModels.synaptic_receptors(syn, Int32(7), Float32).glu) == 7
    pop = IF(N = Int32(10))
    sim!([pop]; duration = 5ms)
    @test all(isfinite, pop.v)
end
