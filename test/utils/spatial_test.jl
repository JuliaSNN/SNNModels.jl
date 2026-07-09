using SNNModels
using Test
using LinearAlgebra
@load_units

@testset "Spatial utilities" begin

    @testset "place_populations — random placement" begin
        Npop = (E = 40, I = 10)
        grid = [1.0f0, 1.0f0]
        pops = place_populations(Npop, grid)
        @test hasproperty(pops, :E)
        @test hasproperty(pops, :I)
        @test length(pops.E) == 40
        @test length(pops.I) == 10
        for pt in pops.E
            @test all(0 .<= pt .<= 1)
        end
    end

    @testset "periodic_distance — scalar" begin
        @test periodic_distance(0.1f0, 0.9f0, 1.0f0) ≈ 0.2f0
        @test periodic_distance(0.0f0, 0.5f0, 1.0f0) ≈ 0.5f0
        @test periodic_distance(0.0f0, 0.0f0, 1.0f0) ≈ 0.0f0
        # symmetry
        @test periodic_distance(0.3f0, 0.8f0, 1.0f0) ≈ periodic_distance(0.8f0, 0.3f0, 1.0f0)
        # never exceeds grid_size/2
        @test periodic_distance(0.1f0, 0.9f0, 1.0f0) <= 0.5f0
    end

    @testset "linear_network — ring weight matrix" begin
        N = 20
        W = linear_network(N)
        @test size(W) == (N, N)
        # self-connections should be zero
        @test all(diag(W) .== 0)
        # weights are non-negative
        @test all(W .>= 0)
        # max weight ≈ w_max default (2.0)
        @test maximum(W) <= 2.1
    end

    @testset "linear_network — custom σ_w and w_max" begin
        N   = 16
        W   = linear_network(N; σ_w = 0.5, w_max = 3.0)
        @test size(W) == (N, N)
        @test maximum(W) <= 3.1
    end

end
true
