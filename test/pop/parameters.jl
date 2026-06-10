using SNNModels
using Test
@load_units

@testset "Population parameters" begin

    @testset "IFParameter" begin
        p = IFParameter()
        @test hasproperty(p, :Vt)
        @test hasproperty(p, :Vr)
        @test hasproperty(p, :El)
        @test hasproperty(p, :τm)

        p2 = IFParameter(Vt = -45mV, El = -60mV, Vr = -65mV)
        @test p2.Vt ≈ -45mV
        @test p2.El ≈ -60mV
        @test p2.Vr ≈ -65mV
    end

    @testset "AdExParameter" begin
        p = AdExParameter()
        @test p.C  ≈ 281pF
        @test p.gl ≈ 40nS
        @test p.Vt ≈ -50mV
        @test p.b  ≈ 80.5pA

        p2 = AdExParameter(C = 200pF, b = 100pA, τw = 200ms, a = 2nS)
        @test p2.C  ≈ 200pF
        @test p2.b  ≈ 100pA
        @test p2.τw ≈ 200ms
        @test p2.a  ≈ 2nS
    end

    @testset "IZParameter" begin
        p = IZParameter()
        @test hasproperty(p, :a)
        @test hasproperty(p, :b)
        @test hasproperty(p, :c)
        @test hasproperty(p, :d)

        p2 = IZParameter(a = 0.1, b = 0.2, c = -65, d = 2)
        @test p2.a ≈ 0.1f0
        @test p2.d ≈ 2.0f0
    end

    @testset "MorrisLecarParameter" begin
        p = SNNModels.MorrisLecarParameter()
        @test hasproperty(p, :Cm)
        @test hasproperty(p, :El)
        @test hasproperty(p, :gCa)
        @test hasproperty(p, :gK)
        p2 = SNNModels.MorrisLecarParameter(gCa = 2nS, gK = 3nS)
        @test p2.gCa ≈ 2nS
        @test p2.gK  ≈ 3nS
    end

    @testset "HetRecParameter" begin
        p = HetRecParameter()
        @test p.Nd        == 2
        @test p.τm        ≈ 20ms
        @test p.τrate     ≈ 100ms
        @test p.steepness ≈ 1.0f0
        @test p.τabs      ≈ 5ms

        p2 = HetRecParameter(Nd = 4, τm = 30ms, overlap = 0.2f0)
        @test p2.Nd      == 4
        @test p2.τm      ≈ 30ms
        @test p2.overlap ≈ 0.2f0
    end

    @testset "PostSpike" begin
        sp = PostSpike()
        @test hasproperty(sp, :At)
        @test hasproperty(sp, :τA)
        @test hasproperty(sp, :τabs)

        sp2 = PostSpike(At = 5.0, τA = 20ms)
        @test sp2.At ≈ 5.0f0
        @test sp2.τA ≈ 20ms
    end

    @testset "Synaptic parameter types" begin
        @test DoubleExpSynapse()           isa DoubleExpSynapse
        @test SingleExpSynapse()           isa SingleExpSynapse
        @test DeltaSynapse()               isa DeltaSynapse
        @test CurrentSynapse()             isa CurrentSynapse

        r = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73)
        @test r isa Receptor
        @test r.E_rev ≈ 0.0f0

        rs = Receptors(
            AMPA  = Receptor(E_rev = 0.0,  τr = 0.26, τd = 2.0,  g0 = 0.73),
            GABAa = Receptor(E_rev = -70.0, τr = 0.1,  τd = 15.0, g0 = 0.38),
        )
        @test length(rs) == 4  # Receptors always returns AMPA, NMDA, GABAa, GABAb
        @test all(r -> r isa Receptor, rs)

        nmda = NMDAVoltageDependency(mg = 1.0, b = 3.36f0, k = -0.077f0)
        @test nmda isa NMDAVoltageDependency

        rsyn = ReceptorSynapse(
            glu_receptors  = [1],
            gaba_receptors = [2],
            syn = Receptors(
                AMPA  = Receptor(E_rev = 0.0,  τr = 0.26, τd = 2.0,  g0 = 0.73),
                GABAa = Receptor(E_rev = -70.0, τr = 0.1,  τd = 15.0, g0 = 0.38),
            ),
        )
        @test rsyn isa ReceptorSynapse
    end

end
true
