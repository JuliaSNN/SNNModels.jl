# Regression tests for the bugs found by the documentation sweep (simulation loop level).
using SNNModels
using Test
@load_units

@testset "train! with IZ, HH, MorrisLecar and without connections" begin
    for P in (IZ(N = 3, param = IZParameter()), HH(N = 3), MorrisLecar(N = 3), IF(N = 3), AdEx(N = 3))
        @test train!([P]; duration = 5ms) isa SNNModels.Time
        @test sim!([P]; duration = 5ms) isa SNNModels.Time
    end
    E = IF(N = 5)
    @test train!([E], [EmptySynapse()]; duration = 5ms) isa SNNModels.Time
end
