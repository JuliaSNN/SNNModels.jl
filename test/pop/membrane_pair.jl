# Membrane pair rule of AdExParameter and IFParameter (SNNModels 1.9.0): C, gl, R, τm are
# always consistent (τm = C / gl, R = 1nS / gl), constructed from a pair or the default pair,
# and updated by a fixed rule.
using SNNModels, Test, JLD2, Random
@load_units

consistent(p) = isapprox.(p.τm, p.C ./ p.gl; rtol = 1e-5) |> all && isapprox.(p.R, 1nS ./ p.gl; rtol = 1e-5) |> all
# the stored quadruple is exactly consistent when at most two values were given
quad(p) = (p.C, p.gl, p.R, p.τm)

@testset "defaults" begin
    p = AdExParameter()
    @test quad(p) == (281pF, 40nS, Float32(nS / 40nS), Float32(281pF / 40nS))
    q = IFParameter()
    @test q.τm ≈ 15ms && q.R ≈ 0.06 && q.gl ≈ 1nS / 0.06 && q.C ≈ 250pF
    @test consistent(p) && consistent(q)
end

@testset "every pair, $T" for T in (AdExParameter, IFParameter)
    C, gl, τm, R = 200pF, 10nS, 20ms, 0.1f0
    for kw in ((; C, gl), (; C, R), (; C, τm), (; gl, τm), (; R, τm))
        p = T(; kw...)
        @test consistent(p)
        @test all(isapprox.(quad(p), (C, gl, R, τm); rtol = 1e-5))
    end
    @test consistent(T(; C, gl, R, τm))           # all four, consistent
    @test consistent(T(; C, gl, τm, Vt = -45mV))  # three, consistent
    # published values rounded to four digits (Duarte 2019 excitatory cell) are accepted
    @test T(C = 104.54pF, τm = 10.72ms, R = 102.5MΩ, gl = 9.75nS).C ≈ 104.54pF
end

@testset "a value is never mixed with the defaults" begin
    for T in (AdExParameter, IFParameter), kw in ((C = 200pF,), (gl = 10nS,), (R = 0.1,), (τm = 20ms,), (gl = 10nS, R = 0.1))
        @test_throws ArgumentError T(; kw...)
    end
    @test_throws ArgumentError AdExParameter(C = 200pF, gl = 10nS, τm = 5ms)  # inconsistent
    @test_throws ArgumentError AdExParameter(gl = 10nS, R = 0.2)              # gl, R disagree
    @test_throws ArgumentError AdExParameter(C = 200pF, gl = 10nS, R = 0.2)
end

@testset "update rule" begin
    p = AdExParameter()                     # C 281, gl 40
    a = with_membrane(p; τm = 20ms)         # keeps gl
    @test all(isapprox.(quad(a), (800pF, 40nS, 0.025, 20ms); rtol = 1e-5))
    b = with_membrane(p; C = 400pF)         # keeps gl
    @test all(isapprox.(quad(b), (400pF, 40nS, 0.025, 10ms); rtol = 1e-5))
    c = with_membrane(p; gl = 20nS)         # keeps C
    @test all(isapprox.(quad(c), (281pF, 20nS, 0.05, 14.05ms); rtol = 1e-5))
    d = with_membrane(p; R = 0.05)          # same as gl = 20nS
    @test all(isapprox.(quad(d), quad(c); rtol = 1e-5))
    e = with_membrane(p; τm = 20ms, C = 281pF)  # two values: new pair
    @test all(isapprox.(quad(e), (281pF, 14.05nS, 1 / 14.05, 20ms); rtol = 1e-5))
    @test_throws ArgumentError with_membrane(p; Vt = 1)
    # other fields are kept
    f = with_membrane(AdExParameter(a = 2nS, Vt = -45mV); τm = 20ms)
    @test f.a == 2nS && f.Vt == -45mV

    # @update! on the struct and nested in a config
    q = AdExParameter()
    @update! q τm = 20ms
    @test quad(q) == quad(a)
    cfg = (exc = (adex = AdExParameter(), N = 1),)
    @update! cfg begin
        exc.adex.gl = 20nS
        exc.adex.Vt = -45mV
    end
    @test all(isapprox.(quad(cfg.exc.adex), quad(c); rtol = 1e-5)) && cfg.exc.adex.Vt == -45mV
    i = IFParameter(C = 281pF, gl = 40nS)
    @update! i τm = 20ms
    @test i isa IFParameter && i.C ≈ 800pF && i.R ≈ 0.025

    # property assignment on the mutable AdExParameter
    m = AdExParameter()
    m.τm = 20ms
    @test quad(m) == quad(a)
    m.R = 0.05
    @test all(isapprox.(quad(m), (800pF, 20nS, 0.05, 40ms); rtol = 1e-5))
    m.b = 10pA
    @test m.b == 10pA
end

@testset "heterogeneous" begin
    Random.seed!(1)
    h = make_heterogeneous(AdExParameter(), 50; C = SNNModels.Uniform(200pF, 300pF))
    @test consistent(h) && all(h.gl .== 40nS) && length(unique(h.τm)) == 50
    h2 = make_heterogeneous(IFParameter(C = 281pF, gl = 40nS), 20;
                            τm = SNNModels.Uniform(10ms, 20ms), gl = SNNModels.Uniform(10nS, 20nS))
    @test consistent(h2) && all(10nS .<= h2.gl .<= 20nS) && all(10ms .<= h2.τm .<= 20ms)
    @update! h τm = 10ms                    # scalar applies to every neuron, gl kept
    @test consistent(h) && all(h.τm .== 10ms) && all(h.C .≈ 400pF)
end

@testset "JLD2 round trip" begin
    p = AdExParameter(τm = 20ms, C = 281pF)
    f = tempname() * ".jld2"
    jldsave(f; p)
    p2 = load(f, "p")
    @test quad(p2) == quad(p) && consistent(p2)
    rm(f)
end

# Tripod integrates with C and gl: updating τm must change the dynamics exactly as giving the
# corresponding (C, gl) pair (up to 1.8.x it had no effect).
@testset "Tripod reads the updated τm" begin
    function tripod_vs(adex)
        Random.seed!(7)
        E = Tripod(; N = 20, adex = adex)
        P = PoissonLayer(10Hz, N = 200)
        S = Stimulus(P, E, :glu, :d1, conn = (μ = 0.5, ρ = 0.5), name = "noise")
        monitor!(E, [:v_s], sr = 1kHz)
        model = compose(; E, S, silent = true)
        sim!(model, 300ms)
        return record(E, :v_s)
    end
    base = AdExParameter()
    upd = AdExParameter(); @update! upd τm = 20ms
    vs_base = tripod_vs(base)
    vs_upd = tripod_vs(upd)
    vs_pair = tripod_vs(AdExParameter(C = 800pF, gl = 40nS))
    @test vs_upd ≈ vs_pair
    @test !(vs_upd ≈ vs_base)
end
