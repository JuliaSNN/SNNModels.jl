using SNNModels
using Test
@load_units

@testset "Utils - util.jl" begin
    @testset "rand_value" begin
        # Test basic functionality
        values = rand_value(100, 1.0, 5.0)
        @test length(values) == 100
        @test all(x -> 1.0 <= x <= 5.0, values)
        
        # Test reverse order bounds
        values2 = rand_value(50, 5.0, 1.0)
        @test length(values2) == 50
        @test all(x -> 1.0 <= x <= 5.0, values2)
        
        # Test equal bounds
        values3 = rand_value(10, 3.0, 3.0)
        @test all(x -> x ≈ 3.0, values3)
    end

    @testset "Fast exponential approximations" begin
        # Test exp32
        @test exp32(0.0f0) ≈ 1.0f0 rtol=0.01
        @test exp32(1.0f0) ≈ exp(1.0f0) rtol=0.05
        @test exp32(-1.0f0) ≈ exp(-1.0f0) rtol=0.05
        @test exp32(-20.0f0) > 0  # Should clamp, not underflow
        
        # Test exp64
        @test exp64(0.0f0) ≈ 1.0f0 rtol=0.01
        @test exp64(1.0f0) ≈ exp(1.0f0) rtol=0.02
        @test exp64(-1.0f0) ≈ exp(-1.0f0) rtol=0.02
        
        # Test exp256
        @test exp256(0.0f0) ≈ 1.0f0 rtol=0.001
        @test exp256(1.0f0) ≈ exp(1.0f0) rtol=0.01
        @test exp256(-1.0f0) ≈ exp(-1.0f0) rtol=0.01
    end

    @testset "Name generation" begin
        # Test name() Symbol generation
        @test name(:E, :I) == :E_to_I
        @test name(:E, :I, :AMPA) == :E_to_I_AMPA
        @test name("E", "I") == :E_to_I
        
        # Test str_name() String generation
        @test str_name(:E, :I) == "E_to_I"
        @test str_name(:E, :I, :AMPA) == "E_to_I_AMPA"
        @test str_name("pre", nothing) == "pre"
        @test str_name("pre", "suffix") == "pre_suffix"
    end

    @testset "f2l formatting" begin
        # Test padding
        @test f2l("test") == "test      "
        @test f2l("test", 5) == "test "
        
        # Test truncation
        @test f2l("verylongstring", 5) == "veryl"
        
        # Test numbers
        @test f2l(123, 5) == "123  "
    end

    @testset "compose" begin
        # Create simple populations
        E = IF(N=100, name=:E)
        I = IF(N=25, name=:I)
        
        # Compose model
        model = compose(E=E, I=I, name="test_model", silent=true)
        
        @test haskey(model.pop, :E)
        @test haskey(model.pop, :I)
        @test model.pop.E.N == 100
        @test model.pop.I.N == 25
        @test model.name == "test_model"
        @test typeof(model.time) <: Time
    end

    @testset "remove_element" begin
        # Create a simple model
        E = IF(N=100, name=:E)
        I = IF(N=25, name=:I)
        model = compose(E=E, I=I, name="test", silent=true)
        
        # Remove a population
        model2 = remove_element(model, :I)
        @test haskey(model2.pop, :E)
        @test !haskey(model2.pop, :I)
        
        # Test error on non-existent key
        @test_throws ArgumentError remove_element(model, :NonExistent)
    end
end
