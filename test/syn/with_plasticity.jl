using SNNModels
using Test
@load_units

# Each testset runs a short train! with a different LTP/STP rule and checks
# that it completes without error and leaves weights in valid range.

const _active_p = IFParameter(El = -49mV)

function _driven_model(; Npre = 15, Npost = 15, ltp = NoLTP(), stp = NoSTP())
    pre  = IF(N = Npre,  param = _active_p)
    post = Npre == Npost ? pre : IF(N = Npost, param = _active_p)
    sym  = ltp isa SNNModels.iSTDPParameter ? :gi : :ge
    syn = SpikingSynapse(pre, post, sym;
        conn     = (p = 0.4f0, μ = 0.5f0),
        LTPParam = ltp,
        STPParam = stp)
    if pre === post
        model = compose(pop = pre, syn = syn, silent = true)
    else
        model = compose(pre = pre, post = post, syn = syn, silent = true)
    end
    return model, syn
end

@testset "train! with LTP rules" begin

    @testset "STDPGerstner" begin
        model, syn = _driven_model(ltp = STDPGerstner())
        W0 = copy(syn.W)
        train!(model, 200ms)
        @test all(isfinite, syn.W)
        @test all(syn.W .>= STDPGerstner().Wmin)
        @test all(syn.W .<= STDPGerstner().Wmax)
    end

    @testset "STDPConfavreux2025" begin
        model, syn = _driven_model(ltp = STDPConfavreux2025())
        train!(model, 200ms)
        @test all(isfinite, syn.W)
    end

    @testset "STDPTriplet" begin
        model, syn = _driven_model(ltp = STDPTriplet())
        train!(model, 200ms)
        @test all(isfinite, syn.W)
        @test all(STDPTriplet().Wmin .<= syn.W .<= STDPTriplet().Wmax)
    end

    @testset "STDPWeightDependent" begin
        model, syn = _driven_model(ltp = STDPWeightDependent())
        train!(model, 200ms)
        @test all(isfinite, syn.W)
        @test all(STDPWeightDependent().Wmin .<= syn.W .<= STDPWeightDependent().Wmax)
    end

    @testset "STDPMexicanHat" begin
        model, syn = _driven_model(ltp = STDPMexicanHat())
        train!(model, 200ms)
        @test all(isfinite, syn.W)
    end

    @testset "iSTDPRate" begin
        pre  = IF(N = 15, param = _active_p)
        post = IF(N = 10, param = _active_p)
        syn  = SpikingSynapse(pre, post, :gi;
            conn     = (p = 0.5f0, μ = 1.0f0),
            LTPParam = iSTDPRate())
        model = compose(pre = pre, post = post, syn = syn, silent = true)
        train!(model, 200ms)
        @test all(isfinite, syn.W)
    end

    @testset "iSTDPPotential" begin
        pre  = IF(N = 10, param = _active_p)
        post = IF(N = 8,  param = _active_p)
        syn  = SpikingSynapse(pre, post, :gi;
            conn     = (p = 0.5f0, μ = 1.0f0),
            LTPParam = iSTDPPotential())
        model = compose(pre = pre, post = post, syn = syn, silent = true)
        train!(model, 200ms)
        @test all(isfinite, syn.W)
    end

    for Rule in (vSTDPParameter, vSTDPDenseParameter)
        @testset "$Rule (Npre == Npost)" begin
            model, syn = _driven_model(ltp = Rule())
            train!(model, 200ms)
            @test all(isfinite, syn.W)
            @test all(isfinite, plasticity_value(syn, :W))
        end

        # Regression: the trace x was sized Npost; @turbo silenced the OOB.
        # x is indexed by pre-neuron j → must be length Npre.
        @testset "$Rule (Npre > Npost, x size regression)" begin
            model, syn = _driven_model(Npre = 20, Npost = 5, ltp = Rule())
            @test length(syn.LTPVars.x) == 20  # Npre, not Npost
            train!(model, 200ms)
            @test all(isfinite, syn.W)
        end
    end

end

@testset "train! with STP rules" begin

    @testset "MarkramSTPParameter (event)" begin
        model, syn = _driven_model(stp = MarkramSTPParameter())
        train!(model, 200ms)
        @test all(isfinite, syn.W)
        @test all(0 .<= syn.STPVars.x .<= 1.1)  # resources ∈ [0,1] approx
        @test all(syn.STPVars.u .>= 0)
    end

    @testset "MarkramSTPParameterTimestep" begin
        model, syn = _driven_model(stp = MarkramSTPParameterTimestep())
        train!(model, 200ms)
        @test all(isfinite, syn.W)
    end

end

@testset "change_plasticity!" begin
    # Fix regression: param was undefined; LTP/STP kwargs must be forwarded.
    # The rule types are type parameters of SpikingSynapse: same type in place, other type via with_plasticity.
    @testset "same-type LTPParam in place" begin
        model, syn = _driven_model(ltp = STDPGerstner())
        change_plasticity!(syn; LTP = STDPGerstner(A_pre = 2f-4))
        @test syn.LTPParam.A_pre == 2f-4
        @test syn.LTPVars isa STDPVariables
        @test length(syn.LTPVars.tpre) == length(syn.fireJ)  # Npre
        @test length(syn.LTPVars.tpost) == length(syn.fireI) # Npost
        @test_throws ArgumentError change_plasticity!(syn; LTP = STDPConfavreux2025())
    end

    @testset "with_plasticity swaps the rule type" begin
        model, syn = _driven_model(ltp = STDPGerstner())
        syn2 = with_plasticity(syn; LTP = STDPConfavreux2025())
        @test syn2.LTPParam isa STDPConfavreux2025
        @test syn2.LTPVars isa STDPVariables
        @test syn2.W === syn.W && syn2.g === syn.g
        @test syn.LTPParam isa STDPGerstner  # original untouched
        @test length(syn2.LTPVars.tpre) == length(syn2.fireJ)
    end

    @testset "with_plasticity STP" begin
        model, syn = _driven_model(stp = MarkramSTPParameter())
        @test syn.STPParam isa MarkramSTPParameter
        syn2 = with_plasticity(syn; STP = MarkramSTPParameterTimestep())
        @test syn2.STPParam isa MarkramSTPParameterTimestep
    end
end

true
