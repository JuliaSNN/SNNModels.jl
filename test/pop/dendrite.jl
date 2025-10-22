using Test

# Test the G_axial function
@testset "G_axial function" begin
    @test G_axial(Ri = 200 * Ω * cm, d = 4um, l = 100um) > 0
    @test G_axial(Ri = 200 * Ω * cm, d = 4um, l = 200um) > 0
    @test G_axial(Ri = 200 * Ω * cm, d = 6um, l = 150um) > 0
end

# Test the G_mem function
@testset "G_mem function" begin
    @test G_mem(Rd = 38907 * Ω * cm^2, d = 4um, l = 100um) > 0
    @test G_mem(Rd = 38907 * Ω * cm^2, d = 4um, l = 200um) > 0
    @test G_mem(Rd = 1700Ω * cm^2, d = 6um, l = 150um) > 0
end

# Test the C_mem function
@testset "C_mem function" begin
    @test C_mem(Cd = 0.5μF / cm^2, d = 4um, l = 100um) > 0
    @test C_mem(Cd = 0.5μF / cm^2, d = 4um, l = 200um) > 0
    @test C_mem(Cd = 1μF / cm^2, d = 6um, l = 150um) > 0
end

# # Test the create_dendrite function
@testset "create_dendrite function" begin
    @test begin
        dd = create_dendrite(d = 4um, l = 100um, physiology = human_dend)
        isapprox(dd.gm, 0.32, atol = 1e-2) &&
            isapprox(dd.gax, 62.83, atol = 1e-2) &&
            isapprox(dd.C, 6.2831, atol = 1e-2)
    end
end

true
