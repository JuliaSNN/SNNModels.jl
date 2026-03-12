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
        ep = EmptyParam()
        @test ep.type == :empty
        
        ep2 = EmptyParam(type=:custom)
        @test ep2.type == :custom
    end

    @testset "Model validation" begin
        # Create a valid simple model
        E = IF(N=100, name=:E)
        model = compose(E=E, name="test", silent=true)
        
        # Test isa_model
        @test isa_model(model)
        
        # Test that model has required fields
        @test hasproperty(model, :pop)
        @test hasproperty(model, :syn)
        @test hasproperty(model, :stim)
        @test hasproperty(model, :time)
        @test hasproperty(model, :name)
    end

    @testset "validate_population_model" begin
        E = IF(N=100, name=:E)
        
        # Should not throw for valid population
        @test validate_population_model(E) === nothing
        
        # Check required fields exist
        @test hasproperty(E, :N)
        @test hasproperty(E, :param)
        @test hasproperty(E, :id)
        @test hasproperty(E, :name)
        @test hasproperty(E, :records)
    end
end
