using SNNModels
using Test
@load_units

@testset "Utils - util.jl" begin
    @testset "rand_value" begin
        values = SNNModels.rand_value(100, 1.0, 5.0)
        @test length(values) == 100
        @test all(x -> 1.0 <= x <= 5.0, values)
        values2 = SNNModels.rand_value(50, 5.0, 1.0)
        @test length(values2) == 50
        @test all(x -> 1.0 <= x <= 5.0, values2)
        values3 = SNNModels.rand_value(10, 3.0, 3.0)
        @test all(x -> x ≈ 3.0, values3)
    end

    @testset "Fast exponential approximations" begin
        @test SNNModels.exp32(0.0f0) ≈ 1.0f0 rtol=0.01
        @test SNNModels.exp32(1.0f0) ≈ exp(1.0f0) rtol=0.05
        @test SNNModels.exp32(-1.0f0) ≈ exp(-1.0f0) rtol=0.05
        @test SNNModels.exp32(-20.0f0) > 0  # clamped to -10 before squaring → positive
        @test SNNModels.exp64(0.0f0) ≈ 1.0f0 rtol=0.01
        @test SNNModels.exp64(1.0f0) ≈ exp(1.0f0) rtol=0.02
        @test SNNModels.exp256(0.0f0) ≈ 1.0f0 rtol=0.001
        @test SNNModels.exp256(1.0f0) ≈ exp(1.0f0) rtol=0.01
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
        @test SNNModels.f2l("test") == "test      "
        @test SNNModels.f2l("test", 5) == "test "
        @test SNNModels.f2l("verylongstring", 5) == "veryl"
        @test SNNModels.f2l(123, 5) == "123  "
    end

    @testset "compose" begin
        E = IF(N=100)
        I = IF(N=25)
        model = compose(E=E, I=I, silent=true)
        @test haskey(model.pop, :E)
        @test haskey(model.pop, :I)
        @test model.pop.E.N == 100
        @test model.pop.I.N == 25
        @test typeof(model.time) <: Time
    end

    @testset "remove_element" begin
        E = IF(N=100)
        I = IF(N=25)
        model = compose(E=E, I=I, silent=true)
        model2 = remove_element(model, :I)
        @test haskey(model2.pop, :E)
        @test !haskey(model2.pop, :I)
    end
end
