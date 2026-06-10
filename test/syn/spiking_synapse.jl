using SNNModels
using Test
using LinearAlgebra
@load_units

@testset "SpikingSynapse constructors" begin

    @testset "(p, μ)" begin
        pre  = IF(N = 30)
        post = IF(N = 15)
        s = SpikingSynapse(pre, post, :ge; conn = (p = 0.3f0, μ = 1.0f0))
        @test s isa SpikingSynapse
        @test length(s.W) > 0
        @test s.targets[:fire] == pre.id
        @test s.targets[:post] == post.id
        @test s.targets[:sym]  == :glu  # :ge maps to :glu via get_synapse_symbol
    end

    @testset "(p, μ, σ)" begin
        pre  = IF(N = 20)
        post = IF(N = 10)
        s = SpikingSynapse(pre, post, :ge; conn = (p = 0.5f0, μ = 1.0f0, σ = 0.2f0))
        @test s isa SpikingSynapse
        @test length(s.W) > 0
    end

    @testset "explicit dense matrix" begin
        pre  = IF(N = 8)
        post = IF(N = 6)
        W    = rand(Float32, post.N, pre.N) .* 0.5f0
        s    = SpikingSynapse(pre, post, :ge; conn = W)  # pass matrix directly, not as NamedTuple
        @test s isa SpikingSynapse
        @test length(s.W) == count(!iszero, W)
    end

    @testset "identity (1-to-1)" begin
        N    = 8
        pre  = IF(N = N)
        post = IF(N = N)
        # qualify LinearAlgebra.I explicitly — other test files define I=IF(...) at module scope
        s    = SpikingSynapse(pre, post, :ge; conn = Matrix{Float32}(LinearAlgebra.I, N, N))
        @test s isa SpikingSynapse
        @test length(s.W) == N
    end

    @testset "inhibitory :gi" begin
        pre  = IF(N = 10)
        post = IF(N = 20)
        s    = SpikingSynapse(pre, post, :gi; conn = (p = 0.4f0, μ = 2.0f0))
        @test s isa SpikingSynapse
        @test s.targets[:sym] == :gaba  # :gi maps to :gaba via get_synapse_symbol
    end

    @testset "NoLTP / NoSTP defaults" begin
        pre  = IF(N = 10)
        post = IF(N = 10)
        s    = SpikingSynapse(pre, post, :ge; conn = (p = 0.5f0, μ = 0.5f0))
        @test s.LTPVars isa SNNModels.NoVariables
        @test s.STPVars isa SNNModels.NoVariables
    end

    @testset "custom name" begin
        pre  = IF(N = 5)
        post = IF(N = 5)
        s    = SpikingSynapse(pre, post, :ge; conn = (p = 1.0f0, μ = 0.5f0), name = "my_syn")
        @test s.name == "my_syn"
    end

    @testset "EmptySynapse" begin
        @test EmptySynapse() isa EmptySynapse
    end

end
true
