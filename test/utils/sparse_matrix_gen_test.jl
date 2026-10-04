using SNNModels
using Test
using Random
using SparseArrays
using Statistics
import SNNModels.Distributions: Binomial, cdf
@load_units

# `sparse_matrix` builds connectivity directly in CSC form. Its realisations differ from
# the old dense generator (`sparse_matrix_dense_legacy`), so the comparison is statistical.

const _legacy = SNNModels.sparse_matrix_dense_legacy

indeg(w) = vec(sum(!iszero, w; dims = 2))        # per postsynaptic row
outdeg(w) = diff(w.colptr)                        # per presynaptic column

# One-sample KS distance between integer samples and a discrete CDF.
function ks_discrete(x, F)
    xs = sort(x)
    n = length(xs)
    D = 0.0
    for (k, v) in enumerate(xs)
        (k < n && xs[k+1] == v) && continue
        D = max(D, abs(k / n - F(v)), abs((searchsortedfirst(xs, v) - 1) / n - F(v - 1)))
    end
    return D
end

# Two-sample KS distance.
function ks2(x, y)
    vals = sort(unique(vcat(x, y)))
    xs, ys = sort(x), sort(y)
    maximum(abs(searchsortedlast(xs, v) / length(xs) - searchsortedlast(ys, v) / length(ys)) for v in vals)
end
ks2_crit(n, m) = 1.63 * sqrt((n + m) / (n * m))   # α = 0.01

# Structural validity: rows in range, strictly increasing within each column, no stored zero.
function valid_csc(w)
    all(1 .<= rowvals(w) .<= size(w, 1)) || return false
    for j = 1:size(w, 2)
        r = rowvals(w)[nzrange(w, j)]
        all(diff(r) .> 0) || return false
    end
    return all(!iszero, nonzeros(w))
end

@testset "sparse_matrix — direct CSC generator" verbose = true begin
    Npre, Npost = 900, 700

    @testset "type and structure ($rule)" for rule in (:Fixed, :FixedIn, :FixedOut, :Bernoulli, :PowerLaw)
        conn = (p = 0.05, μ = 1.0, σ = 0.3, rule, γ = 2.5, kmin = 10)
        Random.seed!(1)
        w = sparse_matrix(Npre, Npost, conn)
        wl = _legacy(Npre, Npost; conn...)
        @test w isa SparseMatrixCSC{Float32,Int}
        @test wl isa SparseMatrixCSC{Float32,Int}
        @test size(w) == size(wl) == (Npost, Npre)
        @test valid_csc(w)
        @test all(nonzeros(w) .> 0)
        # downstream: dsparse and Int32/Int index structs still convert
        rowptr, colptr, I, J, index, W = dsparse(w)
        @test W isa Vector{Float32} && length(I) == nnz(w)
        @test Vector{Int32}(I) == I
    end

    @testset "fixed degrees are exact" begin
        p = 0.1
        w = sparse_matrix(Npre, Npost, (p = p, μ = 1.0, σ = 0.0, rule = :Fixed))
        @test all(indeg(w) .== Npre - round(Int, (1 - p) * Npre))
        @test all(indeg(w) .== indeg(_legacy(Npre, Npost; p, μ = 1.0, σ = 0.0, rule = :Fixed)))
        w = sparse_matrix(Npre, Npost, (p = p, μ = 1.0, σ = 0.0, rule = :FixedOut))
        @test all(outdeg(w) .== Npost - round(Int, (1 - p) * Npost))
        @test all(outdeg(w) .== outdeg(_legacy(Npre, Npost; p, μ = 1.0, σ = 0.0, rule = :FixedOut)))
        # empty / full limits
        @test nnz(sparse_matrix(50, 40, (p = 0.0, rule = :Fixed))) == 0
        @test nnz(sparse_matrix(50, 40, (p = 1.0, rule = :Fixed))) == 2000
        @test nnz(sparse_matrix(50, 40, (p = 0.0, rule = :Bernoulli))) == 0
        @test nnz(sparse_matrix(50, 40, (p = 1.0, rule = :Bernoulli))) == 2000
        @test nnz(sparse_matrix(50, 40, (p = 0.0, rule = :FixedOut))) == 0
        @test size(sparse_matrix(0, 40, (p = 0.3, rule = :Bernoulli))) == (40, 0)
    end

    @testset "Bernoulli degrees follow Binomial (KS), like the legacy generator" begin
        p = 0.02
        N1, N2 = 3000, 2500
        Random.seed!(2)
        w = sparse_matrix(N1, N2, (p = p, μ = 1.0, σ = 0.0, rule = :Bernoulli))
        wl = _legacy(N1, N2; p, μ = 1.0, σ = 0.0, rule = :Bernoulli)
        Fin = k -> cdf(Binomial(N1, p), k)
        Fout = k -> cdf(Binomial(N2, p), k)
        crit = 1.63 / sqrt(N2)
        @test ks_discrete(indeg(w), Fin) < crit
        @test ks_discrete(indeg(wl), Fin) < crit
        @test ks_discrete(outdeg(w), Fout) < 1.63 / sqrt(N1)
        @test ks2(indeg(w), indeg(wl)) < ks2_crit(N2, N2)
        μ_nnz, σ_nnz = N1 * N2 * p, sqrt(N1 * N2 * p * (1 - p))
        @test abs(nnz(w) - μ_nnz) < 5σ_nnz
        @test abs(mean(indeg(w)) - N1 * p) < 5 * sqrt(N1 * p / N2)
        @test abs(mean(outdeg(w)) - N2 * p) < 5 * sqrt(N2 * p / N1)
        # uniform over positions: column-major position mean ≈ centre
        r, c, _ = findnz(w)
        @test abs(mean(r) - (N2 + 1) / 2) < 5 * N2 / sqrt(12 * nnz(w))
        @test abs(mean(c) - (N1 + 1) / 2) < 5 * N1 / sqrt(12 * nnz(w))
    end

    @testset "PowerLaw out-degree matches the legacy generator (KS)" begin
        conn = (p = 0.1, μ = 1.0, σ = 0.0, rule = :PowerLaw, γ = 2.0, kmin = 5)
        Random.seed!(3)
        a = outdeg(sparse_matrix(4000, 600, conn))
        b = outdeg(_legacy(4000, 600; conn...))
        @test ks2(a, b) < ks2_crit(4000, 4000)
        @test maximum(a) <= 599 && minimum(a) >= 5
    end

    @testset "weights: moments, Float32, clamping of non-positive draws, sign" begin
        Random.seed!(4)
        conn = (p = 0.2, μ = 2.0, σ = 0.5, rule = :Bernoulli)
        w = sparse_matrix(1000, 1000, conn)
        wl = _legacy(1000, 1000; conn...)
        @test eltype(nonzeros(w)) == Float32
        @test abs(mean(nonzeros(w)) - 2.0) < 0.01
        @test abs(std(nonzeros(w)) - 0.5) < 0.01
        @test abs(mean(nonzeros(w)) - mean(nonzeros(wl))) < 0.01
        @test abs(std(nonzeros(w)) - std(nonzeros(wl))) < 0.01
        # draws <= 0 are dropped (truncated distribution), same fraction as legacy
        conn = (p = 0.5, μ = 0.5, σ = 1.0, rule = :Fixed)
        w = sparse_matrix(800, 800, conn)
        wl = _legacy(800, 800; conn...)
        frac(w) = nnz(w) / (800 * 400)
        expected = 1 - cdf(SNNModels.Distributions.Normal(0.5, 1.0), 0.0)
        @test abs(frac(w) - expected) < 0.005
        @test abs(frac(w) - frac(wl)) < 0.007
        @test all(nonzeros(w) .> 0)
        @test abs(mean(nonzeros(w)) - mean(nonzeros(wl))) < 0.01
        # negative μ: all weights negative, magnitude from |μ|
        w = sparse_matrix(300, 300, (p = 0.1, μ = -1.5, σ = 0.0, rule = :Fixed))
        @test all(nonzeros(w) .== -1.5f0)
        @test nnz(w) == 300 * 30
        # μ = 0, σ = 0: no positive draw, no synapse (as before)
        @test nnz(sparse_matrix(100, 100, (p = 0.5, μ = 0.0, σ = 0.0, rule = :Fixed))) == 0
        # other distributions keep working (Float32 parameters)
        w = sparse_matrix(200, 200, (p = 0.2, μ = 0.0, σ = 0.5, dist = :LogNormal, rule = :Bernoulli))
        @test eltype(w) == Float32 && all(nonzeros(w) .> 0)
    end

    @testset "autapses removed structurally" begin
        E = IF(N = 300)
        s = SpikingSynapse(E, E, :ge; conn = (p = 0.5f0, μ = 1.0f0))
        M = matrix(s)
        @test all(i != j for (i, j) in zip(findnz(M)[1:2]...))
        @test all(!iszero, s.W)                   # no stored zero synapses
        w = sparse_matrix(50, 50, (p = 1.0, rule = :Bernoulli))
        SNNModels.remove_autapses!(w)
        @test nnz(w) == 50 * 49 && valid_csc(w)
        # a user-supplied sparse matrix is copied, not mutated
        u = sparse(1.0f0 * SNNModels.LinearAlgebra.I, 20, 20) + sprand(Float32, 20, 20, 0.2)
        n0 = nnz(u)
        E2 = IF(N = 20)
        SpikingSynapse(E2, E2, :ge; conn = u)
        @test nnz(u) == n0
    end

    @testset "SpikeTimeStimulusIdentity has no dense N x N" begin
        E = IF(N = 30_000)
        bytes = @allocated SpikeTimeStimulusIdentity(E, :ge; param = SpikeTimeParameter([1.0], [1]))
        @test bytes < 30_000^2 ÷ 8                # far below even a BitMatrix
    end

    @testset "scaling: 1e5 x 1e5, p = 1e-3 (1e7 synapses)" begin
        sparse_matrix(1000, 1000, (p = 1e-3, rule = :Bernoulli))   # compile
        N = 100_000
        GC.gc()
        stats = @timed sparse_matrix(N, N, (p = 1e-3, μ = 1.0, σ = 0.2, rule = :Bernoulli))
        w = stats.value
        @test abs(nnz(w) - 1e7) < 5 * sqrt(1e7)
        @test stats.time < 10                         # a few seconds
        @test stats.bytes < 2^30                      # < 1 GiB allocated in total
        @test Base.summarysize(w) < 2^30
        @test stats.bytes < N * N ÷ 8                 # never an Npost x Npre array
    end

    @testset "no Npost x Npre allocation for any rule" begin
        N = 20_000
        for rule in (:Fixed, :FixedOut, :Bernoulli, :PowerLaw)
            conn = (p = 1e-3, μ = 1.0, σ = 0.2, rule, γ = 2.0, kmin = 5)
            sparse_matrix(100, 100, conn)
            bytes = @allocated sparse_matrix(N, N, conn)
            @test bytes < N * N ÷ 8
        end
    end
end
