using SNNModels
using Test
using Random
@load_units

# `change_plasticity(model, syn_name; LTP, STP)`: a new model with one synapse replaced by `with_plasticity` (SNNModels 1.10).
# (Helpers are prefixed `_cp`: the test files are included into one module by runtests.jl.)

const _CPS = SNNModels

# E/I network built with `compose`, never plastic; EE, EI, IE, II
function _cp_model(; FT = Float32, seed = 11, NE = 400, NI = 100, ltpEE = NoLTP())
    Random.seed!(seed)
    vft = FT === Float64 ? (VFT = Vector{Float64},) : (;)
    E = IF(; N = NE, name = "E", vft...); I = IF(; N = NI, name = "I", vft...)
    E.v .= FT(-70) .+ FT(20) .* rand(FT, NE); I.v .= FT(-70) .+ FT(20) .* rand(FT, NI)
    E.I .= FT(250) .+ FT(150) .* rand(FT, NE); I.I .= FT(250) .+ FT(150) .* rand(FT, NI)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.05, μ = 0.25), name = "EE", LTPParam = ltpEE)
    EI = SpikingSynapse(E, I, :ge; conn = (p = 0.05, μ = 0.25), name = "EI")
    IE = SpikingSynapse(I, E, :gi; conn = (p = 0.05, μ = 2.0), name = "IE")
    II = SpikingSynapse(I, I, :gi; conn = (p = 0.05, μ = 2.0), name = "II")
    monitor!([E, I], [:fire]; monitor_rate = 400Hz)
    return compose(; E, I, EE, EI, IE, II, silent = true)
end

_cp_spikes(m) = [spiketimes(p) for p in values(m.pop)]
_cp_nevermem(c) = length(c.J) == 0 && length(c.index) == 0 && length(c.ρ) == 0

# model with EE rebuilt by the constructor from the weights of `m.syn.EE` (never plastic), on a deep copy of `m`
function _cp_rebuilt(m)
    X = deepcopy(m)
    old = X.syn.EE
    fresh = SpikingSynapse(X.pop.E, X.pop.E, :ge; conn = matrix(old), name = old.name)
    return merge(X, (; syn = merge(X.syn, (; EE = fresh))))
end

@testset "change_plasticity" begin
    for FT in (Float32, Float64)
        @testset "$FT" begin
            m0 = _cp_model(; FT)
            sim!(; model = m0, duration = 20ms)
            others = (m0.syn.EI, m0.syn.IE, m0.syn.II)
            types = map(typeof, others)
            old = m0.syn.EE
            W0 = copy(old.W)
            @test _cp_nevermem(old)

            # switch on one named synapse
            m1 = change_plasticity(m0, :EE; LTP = STDPGerstner())
            @test m1 isa NamedTuple && keys(m1) == keys(m0)
            @test keys(m1.syn) == keys(m0.syn)
            @test m1.syn.EE.LTPParam isa STDPGerstner
            @test _CPS.float_type(m1.syn.EE.LTPParam) == FT
            @test eltype(m1.syn.EE.W) == FT
            # the rest is shared, not copied
            @test m1.pop === m0.pop && m1.stim === m0.stim && m1.time === m0.time && m1.name == m0.name
            @test m1.syn.EI === m0.syn.EI && m1.syn.IE === m0.syn.IE && m1.syn.II === m0.syn.II
            # the new synapse points to the same populations
            @test m1.syn.EE.fireJ === m0.pop.E.fire && m1.syn.EE.fireI === m0.pop.E.fire
            @test m1.syn.EE.W === old.W && m1.syn.EE.colptr === old.colptr && m1.syn.EE.I === old.I
            # the old model and synapse are not modified
            @test old.LTPParam isa NoLTP && _cp_nevermem(old) && old.W == W0
            @test m0.syn.EE === old

            # run: the other synapses keep weights and type, EE learns
            before = map(c -> copy(c.W), others)
            train!(; model = m1, duration = 100ms)
            @test map(typeof, (m1.syn.EI, m1.syn.IE, m1.syn.II)) == types
            @test all(map((c, w) -> c.W == w, (m1.syn.EI, m1.syn.IE, m1.syn.II), before))
            @test all(c -> c.LTPParam isa NoLTP && c.STPParam isa NoSTP, (m1.syn.EI, m1.syn.IE, m1.syn.II))
            @test plasticity_value(m1.syn.EE, :W) != W0

            # switch off: never-plastic footprint, same run as a rebuilt synapse
            Wfinal = copy(plasticity_value(m1.syn.EE, :W))
            Y = _cp_rebuilt(m1)
            m2 = change_plasticity(m1, :EE; LTP = NoLTP(), STP = NoSTP())
            @test typeof(m2.syn.EE) === typeof(m0.syn.EE)
            @test m2.syn.EE.LTPParam isa NoLTP && m2.syn.EE.STPParam isa NoSTP
            @test _cp_nevermem(m2.syn.EE)
            @test m2.syn.EE.W == Wfinal
            m2b = change_plasticity(m1, :EE; LTP = NoLTP(), STP = NoSTP())
            @test typeof(m2b.syn.EE) === typeof(m2.syn.EE)
            @test m2.syn.EI === m1.syn.EI && m2.pop === m1.pop
            sim!(; model = m2, duration = 50ms)
            sim!(; model = Y, duration = 50ms)
            @test _cp_spikes(m2) == _cp_spikes(Y)
            @test all(p -> getproperty(m2.pop, p).v == getproperty(Y.pop, p).v, keys(m2.pop))
            @test m2.syn.EE.W == Wfinal

            # keep: a rule left to nothing is kept
            m3 = change_plasticity(m1, :EE; STP = MarkramSTPParameter())
            @test m3.syn.EE.LTPParam isa STDPGerstner && m3.syn.EE.STPParam isa MarkramSTPParameter
            m4 = change_plasticity(m1, :EE)
            @test typeof(m4.syn.EE) === typeof(m1.syn.EE)
        end
    end

    @testset "replaces the entry, input not doubled" begin
        m0 = _cp_model()
        m1 = change_plasticity(m0, :EE; LTP = STDPGerstner())
        @test keys(m1.syn) == keys(m0.syn) && length(m1.syn) == length(m0.syn)
        @test !any(c -> c === m0.syn.EE, values(m1.syn))
        @test count(c -> c.name == "EE", values(m1.syn)) == 1
        ref = _cp_model(; ltpEE = STDPGerstner())
        train!(; model = m1, duration = 100ms)
        train!(; model = ref, duration = 100ms)
        @test _cp_spikes(m1) == _cp_spikes(ref)
        @test sum(length, Iterators.flatten(_cp_spikes(m1))) > 0
    end

    @testset "other synapse" begin
        m0 = _cp_model()
        m = change_plasticity(m0, :IE; LTP = iSTDPRate())
        @test m.syn.IE.LTPParam isa iSTDPRate
        @test m.syn.EE === m0.syn.EE
        train!(; model = m, duration = 50ms)
        @test m.syn.EE.W == m0.syn.EE.W
    end

    @testset "unknown name" begin
        m0 = _cp_model()
        err = try change_plasticity(m0, :XX; LTP = STDPGerstner()); nothing catch e e end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin(":XX", msg)
        @test all(k -> occursin(":" * String(k), msg), keys(m0.syn))
        # a population name is not a synapse
        @test_throws ArgumentError change_plasticity(m0, :E; LTP = STDPGerstner())
    end

    @testset "compose collision message" begin
        m0 = _cp_model()
        new = with_plasticity(m0.syn.EE; LTP = STDPGerstner())
        err = try compose(; m0..., EE = new, silent = true); nothing catch e e end
        @test err isa AssertionError
        msg = sprint(showerror, err)
        @test occursin("EE", msg) && occursin("change_plasticity", msg)
        # change_plasticity needs no compose
        @test change_plasticity(m0, :EE; LTP = STDPGerstner()).syn.EE.LTPParam isa STDPGerstner
    end
end
