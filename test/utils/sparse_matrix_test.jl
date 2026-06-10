using SNNModels
using Test
using SparseArrays
using LinearAlgebra
@load_units

@testset "Sparse matrix utilities" begin

    function _synapse(; Npre = 20, Npost = 15)
        pre  = IF(N = Npre)
        post = IF(N = Npost)
        SpikingSynapse(pre, post, :ge; conn = (p = 1.0f0, μ = 1.0f0))
    end

    @testset "matrix(c) — dense reconstruction" begin
        s = _synapse()
        M = matrix(s)
        @test M isa SparseMatrixCSC
        @test size(M) == (15, 20)
        @test nnz(M) > 0
        @test all(nonzeros(M) .> 0)
    end

    @testset "matrix(c, :W) — with field symbol" begin
        s = _synapse()
        M = matrix(s, :W)
        @test M isa SparseMatrixCSC
        @test size(M) == (15, 20)
    end

    @testset "presynaptic(c) — all post neurons" begin
        s = _synapse()
        pre = presynaptic(s)
        @test length(pre) == 15          # one entry per post neuron
        @test all(x -> x isa AbstractVector, pre)
        @test all(x -> all(1 .<= x .<= 20), pre)
    end

    @testset "presynaptic(c, i) — single post neuron" begin
        s = _synapse()
        p = presynaptic(s, 1)
        @test p isa AbstractVector
        @test all(1 .<= p .<= 20)
    end

    @testset "presynaptic(c, is) — vector of post neurons" begin
        s = _synapse()
        ps = presynaptic(s, [1, 2, 3])
        @test length(ps) == 3
    end

    @testset "postsynaptic(c) — all pre neurons" begin
        s = _synapse()
        post = postsynaptic(s)
        @test length(post) == 20         # one entry per pre neuron
        @test all(x -> x isa AbstractVector, post)
        @test all(x -> all(1 .<= x .<= 15), post)
    end

    @testset "postsynaptic(c, j) — single pre neuron" begin
        s = _synapse()
        p = postsynaptic(s, 1)
        @test p isa AbstractVector
        @test all(1 .<= p .<= 15)
    end

    @testset "update_weights!(c, j, i, w) — scalar" begin
        s = _synapse()
        # find a connected (j,i) pair
        M  = matrix(s)
        j0, i0 = findnz(M)[1:2] |> x -> (x[2][1], x[1][1])  # first nnz: row=i, col=j
        update_weights!(s, j0, i0, 9.0f0)
        M2 = matrix(s)
        @test M2[i0, j0] ≈ 9.0f0
    end

    @testset "update_weights!(c, js, is, w) — vector" begin
        s = _synapse()
        M = matrix(s)
        rows, cols, _ = findnz(M)
        j0 = cols[1:2]; i0 = rows[1:2]
        update_weights!(s, j0, i0, 0.0f0)
        M2 = matrix(s)
        for (ii, jj) in zip(i0, j0)
            @test M2[ii, jj] ≈ 0.0f0
        end
    end

    @testset "connect!(c, j, i, μ) — sets a weight" begin
        s = _synapse()
        M_before = matrix(s)
        # pick an existing connection and change its weight
        rows, cols, _ = findnz(M_before)
        i0, j0 = rows[1], cols[1]
        connect!(s, j0, i0, 42.0f0)
        M_after = matrix(s)
        @test M_after[i0, j0] ≈ 42.0f0
    end

end
true
