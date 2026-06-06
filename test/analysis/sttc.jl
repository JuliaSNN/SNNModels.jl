using SNNModels
using Test

@testset "STTC internals" begin
    Δt     = 0.1f0
    istart = 0f0
    iend   = 1f0

    @testset "_tile_fraction" begin
        # empty train
        @test SNNModels._tile_fraction(Float32[], Δt, istart, iend) == 0f0

        # single spike: left Δt + right Δt, no loop
        tf1 = SNNModels._tile_fraction(Float32[0.5f0], Δt, istart, iend)
        @test tf1 ≈ 0.2f0 / (1f0 + 2f0 * Δt)

        # two spikes far apart (gap = 0.6 > 2Δt = 0.2) → each contributes full 2Δt
        tf2 = SNNModels._tile_fraction(Float32[0.2f0, 0.8f0], Δt, istart, iend)
        @test tf2 ≈ 0.4f0 / (1f0 + 2f0 * Δt)

        # two spikes close (gap = 0.05 < 2Δt) → tiles overlap
        gap = 0.05f0
        tf3 = SNNModels._tile_fraction(Float32[0.5f0, 0.5f0 + gap], Δt, istart, iend)
        @test tf3 ≈ (Δt + gap + Δt) / (1f0 + 2f0 * Δt)

        # tile fraction is in (0, 1]
        tf4 = SNNModels._tile_fraction(collect(Float32, 0.1f0:0.05f0:0.9f0), Δt, istart, iend)
        @test 0f0 < tf4 ≤ 1f0
    end

    @testset "_coincident_fraction" begin
        A = Float32[0.5f0]

        @test SNNModels._coincident_fraction(Float32[], Float32[0.5f0], Δt) == 0f0

        # exact match
        @test SNNModels._coincident_fraction(A, Float32[0.5f0], Δt) == 1f0

        # within window
        @test SNNModels._coincident_fraction(A, Float32[0.5f0 + Δt * 0.9f0], Δt) == 1f0

        # at boundary (inclusive)
        @test SNNModels._coincident_fraction(A, Float32[0.5f0 + Δt], Δt) == 1f0

        # outside window
        @test SNNModels._coincident_fraction(A, Float32[0.5f0 + Δt + 0.01f0], Δt) == 0f0

        # multiple spikes: only some coincident
        B = Float32[0.1f0, 0.5f0, 0.9f0]
        A2 = Float32[0.5f0, 0.91f0]
        # 0.5 hits B[2], 0.91 hits B[3] (0.91-0.9=0.01 < 0.1)
        @test SNNModels._coincident_fraction(A2, B, Δt) == 1f0
    end
end

@testset "STTC single pair" begin
    Δt       = 0.05f0
    interval = Float32[0f0, 1f0]

    A = Float32[0.1, 0.3, 0.5, 0.7, 0.9]
    B = Float32[0.15, 0.35, 0.55]

    # identical trains → 1.0
    @test STTC(A, A, Δt, interval) ≈ 1f0

    # symmetry
    @test STTC(A, B, Δt, interval) ≈ STTC(B, A, Δt, interval)

    # bounds
    @test -1f0 ≤ STTC(A, B, Δt, interval) ≤ 1f0

    # non-overlapping trains (gap >> Δt) → negative (PA=PB=0, so STTC = -(TA+TB)/2)
    C = Float32[0.05, 0.10, 0.15]
    D = Float32[0.80, 0.85, 0.90]
    @test STTC(C, D, Δt, interval) < 0f0
end

@testset "STTC matrix" begin
    Δt       = 0.05f0
    interval = Float32[0f0, 1f0]

    A = Float32[0.1, 0.2, 0.3, 0.4]
    B = Float32[0.1, 0.2, 0.3, 0.4]   # identical to A
    C = Float32[0.7, 0.8, 0.9]         # no overlap with A/B
    E = Float32[]                       # empty

    M = STTC([A, B, C, E], Δt, interval)

    @test size(M) == (4, 4)

    # diagonal = 1
    @test all(M[i, i] == 1f0 for i in 1:4)

    # symmetric
    @test M ≈ M'

    # identical trains → 1
    @test M[1, 2] ≈ 1f0

    # empty train → 0 for all off-diagonal entries
    @test all(M[4, 1:3] .== 0f0)
    @test all(M[1:3, 4] .== 0f0)

    # all values in [-1, 1]
    @test all(-1f0 .≤ M .≤ 1f0)
end
