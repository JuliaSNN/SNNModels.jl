using SNNModels
using Test
using SNNModels.Distributions
@load_units

# Synaptic data (weights, short-term variables, delays, plasticity traces) must be
# Float32 whatever numeric type the user passes for μ, σ, the connectivity matrix
# or the delay distribution.

@testset "Float32 synaptic data" begin
    E = IF(N = 30)

    @testset "sparse_matrix μ=$(typeof(μ)) σ=$(typeof(σ))" for (μ, σ) in (
        (1, 0), (0.5, 0.1), (0.5f0, 0.1f0), (-0.5, 0.1),
    )
        w = sparse_matrix(30, 20; p = 0.3, μ = μ, σ = σ)
        @test eltype(w) == Float32
        @test size(w) == (20, 30)
    end

    @testset "sparse_matrix from a Float64 / Bool matrix" begin
        @test eltype(sparse_matrix(4, 3, rand(3, 4))) == Float32
        @test eltype(sparse_matrix(3, 3, Matrix(SNNModels.LinearAlgebra.I(3)))) == Float32
    end

    @testset "SpikingSynapse conn=$(conn)" for conn in (
        (p = 0.4, μ = 0.5, σ = 0.1),
        (p = 0.4f0, μ = 0.5f0),
        (p = 0.4, μ = 1),
    )
        s = SpikingSynapse(E, E, :ge; conn = conn, LTPParam = STDPGerstner())
        @test eltype(s.W) == Float32
        @test eltype(s.ρ) == Float32
        @test all(eltype(getfield(s.LTPVars, f)) == Float32 for f in (:tpre, :tpost, :last_pre, :last_post))
    end

    @testset "SpikingSynapse with Float64 matrix and delays" begin
        s = SpikingSynapse(E, E, :ge; conn = rand(30, 30), delay_dist = Uniform(1.0, 2.0))
        @test eltype(s.W) == Float32
        @test eltype(s.param.delaytime) == Float32
        @test eltype(s.param.spike_time) == Vector{Float32}
        @test eltype(s.param.spike_w) == Vector{Float32}
    end

    @testset "triplet and weight-dependent variables" begin
        s = SpikingSynapse(E, E, :ge; conn = (p = 0.4, μ = 0.5), LTPParam = STDPTriplet())
        @test all(eltype(getfield(s.LTPVars, f)) == Float32 for f in (:r1, :r2, :o1, :o2))
        s = SpikingSynapse(E, E, :ge; conn = (p = 0.4, μ = 0.5), LTPParam = STDPWeightDependent())
        @test eltype(s.LTPVars.tpre) == Float32
    end
end
true
