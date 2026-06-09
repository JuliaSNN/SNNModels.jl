using SNNModels
using Test
@load_units

@testset "Utils - structs.jl" begin
    @testset "Time constructor" begin
        # Test default constructor
        t = Time()
        @test t.t[1] == 0.0f0
        @test t.tt[1] == 0
        @test t.dt == 0.125f0
        
        # Test numeric constructor
        t2 = Time(100.0)
        @test t2.t[1] == 100.0f0
        @test t2.tt[1] == Int32(800)  # 100/0.125 = 800
        @test t2.dt == 0.125f0
    end

    @testset "EmptyParam" begin
        ep = SNNModels.EmptyParam()
        @test ep.type == :empty
        ep2 = SNNModels.EmptyParam(type=:custom)
        @test ep2.type == :custom
    end

    @testset "Model validation" begin
        E = IF(N=100)
        model = compose(E=E, silent=true)
        @test SNNModels.isa_model(model)
        @test hasproperty(model, :pop)
        @test hasproperty(model, :syn)
        @test hasproperty(model, :stim)
        @test hasproperty(model, :time)
        @test hasproperty(model, :name)
    end

    @testset "validate_population_model" begin
        E = IF(N=100)
        @test SNNModels.validate_population_model(E) === nothing
        @test hasproperty(E, :N)
        @test hasproperty(E, :param)
        @test hasproperty(E, :id)
        @test hasproperty(E, :name)
        @test hasproperty(E, :records)
    end
end
