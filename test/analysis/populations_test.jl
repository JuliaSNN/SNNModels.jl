using SNNModels
using Test
using SparseArrays
@load_units

@testset "Population analysis" begin

    @testset "population_indices" begin
        E = IF(N = 10)
        I = IF(N = 5)
        pops = (E = E, I = I)
        idx  = population_indices(pops)
        @test hasproperty(idx, :E)
        @test hasproperty(idx, :I)
        @test length(idx.E) == 10
        @test length(idx.I) == 5
        # E and I ranges must be non-overlapping
        @test isempty(intersect(idx.E, idx.I))
        # union covers 1..15
        @test sort(collect(union(idx.E, idx.I))) == collect(1:15)
    end

    @testset "filter_items — no-noise default condition" begin
        E     = IF(N = 10)
        noise = IF(N = 5, name = "noise_pop")  # IF is immutable; set name at construction
        pops  = (E = E, noise = noise)
        filtered = filter_items(pops)
        # noise should be removed
        @test !hasproperty(filtered, :noise)
        @test  hasproperty(filtered, :E)
    end

    @testset "filter_items — custom condition" begin
        E = IF(N = 10)
        I = IF(N = 5)
        pops = (E = E, I = I)
        # keep only pops with N > 7
        filtered = filter_items(pops; condition = p -> p.N > 7)
        @test  hasproperty(filtered, :E)
        @test !hasproperty(filtered, :I)
    end

    @testset "average_conn_strength" begin
        # 3-neuron pops: pop A = [1,2], pop B = [3]
        M = Float32[1 2 0; 3 4 0; 0 0 5]
        pops = [[1, 2], [3]]
        result = average_conn_strength(M, pops, 1.0)
        @test size(result) == (2, 2)
        # A→A: mean([1,2,3,4]) / 1 = 2.5
        @test result[1, 1] ≈ mean(M[[1,2], [1,2]]) rtol = 1e-4
        # B→B: M[3,3] / 1 = 5
        @test result[2, 2] ≈ 5.0f0 rtol = 1e-4
    end

end
true
