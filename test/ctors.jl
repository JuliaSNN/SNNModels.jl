# Tests for @snn_kw constructors

coretype(T::UnionAll) = coretype(T.body)
coretype(T::DataType) = T
paramnames(T) = coretype(T).parameters
function test_typeparams(Model; args = ())
    Model = coretype(Model)
    n = Model(args...)
    for idx = 1:length(fieldnames(Model))
        fieldtypes(Model)[idx] isa TypeVar || continue
        field, Tf = fieldnames(Model)[idx], fieldtypes(Model)[idx].name

        # TODO: Test where vector is not <: DenseArray
        if Tf == :FT
            @test getfield(n, field) isa Float32
            _n = Model(args...; FT = Float64)
            @test getfield(_n, field) isa Float64
        elseif Tf == :VFT
            @test getfield(n, field) isa Vector{Float32}
            _n = Model(args...; VFT = Vector{Float64})
            @test getfield(_n, field) isa Vector{Float64}
        elseif Tf == :MFT
            @test getfield(n, field) isa Matrix{Float32}
            _n = Model(args...; MFT = Matrix{Float64})
            @test getfield(_n, field) isa Matrix{Float64}
        elseif Tf == :VBT
            @test getfield(n, field) isa Vector{Bool}
            _n = Model(args...; VBT = Vector{Any})
            @test getfield(_n, field) isa Vector{Any}
            @test getfield(_n, field)[1] isa Bool
        end
    end
end


@testset "Constructors" begin
    @testset "Type parameters" begin
        for Model in (HH, IF, IZ, Rate, AdEx, Tripod, Poisson, BallAndStick, ExtendedIF)
            test_typeparams(Model)
        end
        test_typeparams(RateSynapse; args = (Rate(), Rate()))
        test_typeparams(PINningSynapse; args = (Rate(), Rate()))
        test_typeparams(FLSynapse; args = (Rate(), Rate()))
    end # Type parameters
end # Constructors


