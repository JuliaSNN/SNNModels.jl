# Regression tests for the bugs found by the documentation sweep (simulation loop level).
using SNNModels
using Test, Random, Statistics
@load_units

@testset "train! with IZ, HH, MorrisLecar and without connections" begin
    for P in (IZ(N = 3, param = IZParameter()), HH(N = 3), MorrisLecar(N = 3), IF(N = 3), AdEx(N = 3))
        @test train!([P]; duration = 5ms) isa SNNModels.Time
        @test sim!([P]; duration = 5ms) isa SNNModels.Time
    end
    E = IF(N = 5)
    @test train!([E], [EmptySynapse()]; duration = 5ms) isa SNNModels.Time
end

@testset "FORCE / PINning connections run under sim! and train!" begin
    Random.seed!(1)
    R = Rate(N = 100)
    F = FLSynapse(R, R; μ = 1.5, α = 1)
    dt = 0.125f0
    T = SNNModels.Time()
    err = Float32[]
    for k = 1:4000
        F.f = sin(2π * k * dt / 20)
        train!([R], [F], SNNModels.AbstractStimulus[], dt, T)
        push!(err, abs(F.z - F.f))
    end
    @test mean(err[end-400:end]) < mean(err[1:400])
    @test sim!([R], [F]; duration = 5ms) isa SNNModels.Time
    for C in (
        SNNModels.FLSparseSynapse(R, R; μ = 1.5, p = 0.2),
        PINningSynapse(R, R),
        SNNModels.PINningSparseSynapse(R, R; p = 0.2),
    )
        @test sim!([R], [C]; duration = 5ms) isa SNNModels.Time
        @test train!([R], [C]; duration = 5ms) isa SNNModels.Time
    end
    @test_throws ArgumentError SNNModels.FLSparseSynapse(R, R)
end
