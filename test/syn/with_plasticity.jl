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

    @testset "vSTDPParameter (Npre == Npost)" begin
        model, syn = _driven_model(ltp = vSTDPParameter())
        train!(model, 200ms)
        @test all(isfinite, syn.W)
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
true
