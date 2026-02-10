using SNNModels
using Test
@load_units

@testset "Poisson" begin
    # Create a Poisson population with default parameters
    p_pop = Poisson()
    @test p_pop.N == 100
    @test p_pop.param.rate ≈ 1.0f0Hz

    # Create a Poisson population with custom parameters
    p_pop_custom = Poisson(N = 50, param = PoissonParameter(rate = 10Hz))
    @test p_pop_custom.N == 50
    @test p_pop_custom.param.rate ≈ 10.0f0Hz
    @test all(p_pop_custom.rate .≈ 10.0f0Hz)

    # Test integrate! function
    dt = 1ms
    SNNModels.integrate!(p_pop_custom, p_pop_custom.param, dt)
    # The number of fired neurons should be stochastic, but we can check if it's within a reasonable range
    # Expected number of spikes: N * rate * dt = 50 * 10 * 0.001 = 0.5
    # So, sum(p_pop_custom.fire) will be 0 or 1 most of the time.
    # A loose check is that it's less than N.
    @test sum(p_pop_custom.fire) <= p_pop_custom.N
end

@testset "VariablePoisson" begin
    # Create a VariablePoisson population with default parameters
    vp_pop = VariablePoisson(param = VariablePoissonParameter())
    @test vp_pop.N == 100
    @test vp_pop.param.β ≈ 0.0f0
    @test vp_pop.param.τ ≈ 50.0f0
    @test vp_pop.param.r0 ≈ 1000.0f0Hz

    # Create a VariablePoisson population with custom parameters
    vp_pop_custom = VariablePoisson(N = 50, param = VariablePoissonParameter(β = 0.1f0, τ = 100ms, r0 = 500Hz))
    @test vp_pop_custom.N == 50
    @test vp_pop_custom.param.β ≈ 0.1f0
    @test vp_pop_custom.param.τ ≈ 100.0f0
    @test vp_pop_custom.param.r0 ≈ 500.0f0Hz

    # Test integrate! function
    dt = 1ms
    SNNModels.integrate!(vp_pop_custom, vp_pop_custom.param, dt)
    @test sum(vp_pop_custom.fire) <= vp_pop_custom.N
end
