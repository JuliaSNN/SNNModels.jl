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

# `param.physiology` reaches the dendrites of Tripod and BallAndStick (before 1.10 they always used `human_dend`)
@testset "Tripod/BallAndStick use param.physiology" begin
    for FT in (Float32, Float64)
        mk(::Val{:tripod}, phys) = Tripod(N = 10, VFT = Vector{FT}, param = TripodParameter(ds = [160um, 200um], physiology = phys))
        mk(::Val{:bas}, phys) = BallAndStick(N = 10, VFT = Vector{FT}, param = BallAndStickParameter(ds = [200um], physiology = phys))
        cd(p) = p isa Tripod ? (p.d1, p.d2) : (p.d,)
        ls(p) = p isa Tripod ? (160um, 200um) : (200um,)
        for kind in (:tripod, :bas)
            ph, pm = mk(Val(kind), human_dend), mk(Val(kind), mouse_dend)
            for (dh, dm, l) in zip(cd(ph), cd(pm), ls(ph))
                @test eltype(dm.C) === FT
                # the dendrite arrays are those of `create_dendrite` with the physiology of the parameter
                ref(phys) = FT === Float32 ? create_dendrite(l; physiology = phys) : SNNModels._create_dendrite_decimal(l; physiology = phys)
                for (d, phys) in ((dh, human_dend), (dm, mouse_dend))
                    r = ref(phys)
                    @test all(==(FT(r.gm)), d.gm) && all(==(FT(r.gax)), d.gax) && all(==(FT(r.C)), d.C)
                end
                # mouse: Cd = 1 vs 0.5 uF/cm2 (C twice), Rd = 1700 vs 38907 Ohm cm2 (gm 38907/1700 times); Ri equal (gax unchanged)
                @test dm.C != dh.C && dm.gm != dh.gm
                @test dm.C[1] / dh.C[1] ≈ 2 rtol = 1e-5
                @test dm.gm[1] / dh.gm[1] ≈ 38907 / 1700 rtol = 1e-5
                @test dm.gax == dh.gax
            end
            # the default is the human physiology
            pd = kind == :tripod ? Tripod(N = 10, VFT = Vector{FT}, param = TripodParameter(ds = [160um, 200um])) :
                                   BallAndStick(N = 10, VFT = Vector{FT}, param = BallAndStickParameter(ds = [200um]))
            @test all(((a, b),) -> a.C == b.C && a.gm == b.gm && a.gax == b.gax, zip(cd(pd), cd(ph)))
            # it still runs and spikes
            I = kind == :tripod ? pm.I : pm.Is
            I .= FT(2000)
            nsp = 0
            for _ in 1:2000
                integrate!(pm, pm.param, FT(0.1))
                nsp += count(pm.fire)
            end
            @test nsp > 0
            @test all(isfinite, pm.v_s)
        end
    end
end

true
