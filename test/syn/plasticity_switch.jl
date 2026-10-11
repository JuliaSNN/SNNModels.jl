using SNNModels
using Test
using Random
using SparseArrays: nnz
using Distributions: Dirac, Uniform
@load_units

# Switching plasticity on and off with `with_plasticity` / `without_plasticity` (SNNModels 1.10).
# Switching on never modifies the source synapse; switching off returns a synapse that is, in type, memory and per-step
# work, a never-plastic synapse with the frozen weights; records of a dropped rule stop cleanly.
# (Helpers are prefixed `_ps`: the test files are included into one module by runtests.jl.)

const _PS = SNNModels
_ps_bps(c) = Base.summarysize(c) / length(c.W)
_ps_built(c) = length(c.J) == length(c.W) && length(c.index) == length(c.W)
_ps_lazy(c) = isempty(c.J) && isempty(c.index)
_ps_arrays(c) = (T = _ps_built(c), lazyT = _ps_lazy(c), ρ = length(c.ρ) == length(c.W), ρempty = isempty(c.ρ))
# the arrays a synapse holds are exactly those its rules need
_ps_expect(c) = (_PS.needs_transpose(c) ? _ps_built(c) : _ps_lazy(c)) && length(c.ρ) == (c.STPParam isa NoSTP ? 0 : length(c.W))
# footprint bounds of a never-plastic synapse (test/syn/synapse_footprint.jl: 10 / 12 / 22 B/syn at E 1600 / I 400) plus
# 1.5 B/syn for the per-neuron arrays at the half in-degree of this smaller network (K = 10-40: never-plastic IE 10.1 B/syn)
# (a Float64 synapse stores 4 more bytes per weight, and per delay with heterogeneous delays)
_ps_bound(delay, FT = Float32) = 1.5 + (isnothing(delay) ? 10.0 : delay isa Dirac ? 12.0 : 22.0) +
                                 (sizeof(FT) - 4) * (delay isa Uniform ? 2 : 1)

# deterministic E/I network driven by heterogeneous constant currents; slots 1 EE, 2 EI, 3 IE, 4 II; one clock `T`
function _ps_net(; seed = 5, NE = 800, NI = 200, FT = Float32, delay = nothing,
                 ltpEE = NoLTP(), stpEE = NoSTP(), stpEI = NoSTP(), ltpIE = NoLTP())
    Random.seed!(seed)
    vft = FT === Float64 ? (VFT = Vector{Float64},) : (;)
    E = IF(; N = NE, name = "E", vft...); I = IF(; N = NI, name = "I", vft...)
    E.v .= FT(-70) .+ FT(20) .* rand(FT, NE); I.v .= FT(-70) .+ FT(20) .* rand(FT, NI)
    E.I .= FT(250) .+ FT(150) .* rand(FT, NE); I.I .= FT(250) .+ FT(150) .* rand(FT, NI)
    d = isnothing(delay) ? (;) : (delay_dist = delay,)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.05, μ = 0.25), name = "EE", LTPParam = ltpEE, STPParam = stpEE, d...)
    EI = SpikingSynapse(E, I, :ge; conn = (p = 0.05, μ = 0.25), name = "EI", STPParam = stpEI, d...)
    IE = SpikingSynapse(I, E, :gi; conn = (p = 0.05, μ = 2.0), name = "IE", LTPParam = ltpIE, d...)
    II = SpikingSynapse(I, I, :gi; conn = (p = 0.05, μ = 2.0), name = "II", d...)
    monitor!([E, I], [:fire]; monitor_rate = 400Hz)
    return (P = [E, I], C = _PS.AbstractConnection[EE, EI, IE, II], E = E, I = I, T = _PS.Time{FT}())
end
_ps_sim!(net, d; kw...) = sim!(net.P, net.C; duration = d, time = net.T, kw...)
_ps_train!(net, d; kw...) = train!(net.P, net.C; duration = d, time = net.T, kw...)
_ps_state(net) = (spikes = [spiketimes(p) for p in net.P], v = [copy(p.v) for p in net.P],
                  W = [copy(plasticity_value(c, :W)) for c in net.C], eff = [copy(_PS.efficacy(c)) for c in net.C])

# constructor kwargs of a slot for the rules `kw` (LTP/STP)
_ps_ctor(kw) = (LTPParam = get(kw, :LTP, NoLTP()), STPParam = get(kw, :STP, NoSTP()))
function _ps_netkw(i, kw)
    r = _ps_ctor(kw)
    i == 1 && return (ltpEE = r.LTPParam, stpEE = r.STPParam)
    i == 2 && (@assert r.LTPParam isa NoLTP; return (stpEI = r.STPParam,))
    i == 3 && (@assert r.STPParam isa NoSTP; return (ltpIE = r.LTPParam,))
    error("slot $i")
end
const _PS_SLOTS = ((:E, :E, :ge), (:E, :I, :ge), (:I, :E, :gi), (:I, :I, :gi))

# a deep copy of `net` whose synapse `i` is rebuilt by the constructor from its current weights (stored zeros kept) with
# the given rules; a delayed synapse takes over the delay parameter of the old one (in-flight increments kept)
function _ps_rebuilt(net, i; LTPParam = NoLTP(), STPParam = NoSTP())
    X = deepcopy(net)
    old = X.C[i]
    pre, post, sym = getfield(X, _PS_SLOTS[i][1]), getfield(X, _PS_SLOTS[i][2]), _PS_SLOTS[i][3]
    delayed = old.param isa _PS.SpikingSynapseDelayParameter
    fresh = SpikingSynapse(pre, post, sym; conn = matrix(old), name = old.name, LTPParam, STPParam,
                           delay_dist = delayed ? Dirac(0.8f0) : nothing)
    @assert fresh.W == plasticity_value(old, :W)
    delayed && (fresh.param = old.param)
    X.C[i] = fresh
    return X
end

_ps_lazykeys(vars) = Tuple(k for k in _PS.lazy_variables(vars) if k !== :W)
_ps_traces(c, T) = Dict((g, k) => copy(plasticity_value(c, g, k, T)) for g in (:LTPVars, :STPVars) for k in _ps_lazykeys(getfield(c, g)))
_ps_has_ltp(kw) = !(get(kw, :LTP, NoLTP()) isa NoLTP)
_ps_has_stp(kw) = !(get(kw, :STP, NoSTP()) isa NoSTP)

# switch on (rule1) -> off -> on again (rule2) in slot i, against twins built by the constructor; `alt` selects which of
# the two switch-off calls is inserted (both are computed and compared)
function _ps_roundtrip(i, r1, r2; FT = Float32, delay = nothing, alt = false)
    # 1. switch on: same run as a synapse built plastic, the source is not modified
    a = _ps_net(; FT, delay); b = _ps_net(; FT, delay, _ps_netkw(i, r1)...)
    _ps_sim!(a, 20ms); _ps_sim!(b, 20ms)
    src = a.C[i]
    @test _ps_lazy(src) && isempty(src.ρ)
    n0 = _ps_bps(src); src0W = copy(src.W)
    a.C[i] = with_plasticity(src; r1...)
    on = a.C[i]
    @test _ps_expect(on)
    @test _ps_lazy(src) && isempty(src.ρ) && _ps_bps(src) == n0
    @test on.W === src.W && on.param === src.param && on.g === src.g && on.records === src.records
    @test eltype(on.W) == FT && eltype(on.ρ) == FT && eltype(on.J) == Int32
    _ps_has_ltp(r1) && @test _PS.float_type(on.LTPParam) == FT
    _ps_has_stp(r1) && @test _PS.float_type(on.STPParam) == FT
    _ps_train!(a, 50ms); _ps_train!(b, 50ms)
    @test _ps_state(a) == _ps_state(b)
    _ps_has_ltp(r1) && @test plasticity_value(on, :W) != src0W
    _ps_has_stp(r1) && @test !all(==(1), _PS.efficacy(on))
    @test _ps_lazy(src) && isempty(src.ρ)          # still: simulating `on` writes only shared arrays

    # 2. switch off: the frozen weights, a never-plastic synapse
    t = _PS.get_time(a.T)
    Wfinal = copy(plasticity_value(on, :W))
    tv = _ps_traces(on, t)
    lens_on = (length(on.J), length(on.index), length(on.ρ))
    off1 = without_plasticity(on)
    off2 = with_plasticity(on; LTP = NoLTP(), STP = NoSTP())
    @test typeof(off1) === typeof(off2)
    @test map(f -> length(getfield(off1, f)), (:J, :index, :ρ)) == map(f -> length(getfield(off2, f)), (:J, :index, :ρ))
    off = alt ? off2 : off1
    @test off.W == Wfinal && off.W === on.W && on.W == Wfinal
    @test _ps_traces(on, t) == tv                 # the final values of the rule stay readable on the source
    @test off.LTPVars isa _PS.NoVariables && off.STPVars isa _PS.NoVariables
    @test off.LTPParam isa NoLTP && off.STPParam isa NoSTP
    @test _ps_lazy(off) && isempty(off.ρ) && eltype(off.J) == Int32 && eltype(off.ρ) == FT
    @test _ps_expect(on) && (length(on.J), length(on.index), length(on.ρ)) == lens_on   # nothing resized
    never = _ps_net(; FT, delay)
    _ps_sim!(never, 20ms)
    @test typeof(off) === typeof(never.C[i])
    @test _ps_bps(off) <= _ps_bps(never.C[i]) + 0.05
    @test _ps_bps(off) < _ps_bound(delay, FT)
    @test _ps_bps(off) < _ps_bps(on)
    a.C[i] = off
    X = _ps_rebuilt(a, i)
    _ps_train!(a, 50ms); _ps_train!(X, 50ms)
    @test off.W == Wfinal
    @test _ps_state(a) == _ps_state(X)
    _ps_sim!(a, 20ms); _ps_sim!(X, 20ms)
    @test off.W == Wfinal
    @test _ps_state(a) == _ps_state(X)

    # 3. switch on again (another rule): fresh arrays, the switched-off source stays lazy
    X2 = _ps_rebuilt(a, i; _ps_ctor(r2)...)
    a.C[i] = with_plasticity(off; r2...)
    @test _ps_expect(a.C[i])
    @test _ps_lazy(off) && isempty(off.ρ)
    _ps_train!(a, 50ms); _ps_train!(X2, 50ms)
    @test _ps_state(a) == _ps_state(X2)
    _ps_has_ltp(r2) && @test plasticity_value(a.C[i], :W) != Wfinal
    _ps_has_stp(r2) && @test !all(==(1), _PS.efficacy(a.C[i]))
    @test _ps_lazy(off) && isempty(off.ρ) && off.W === a.C[i].W
    return nothing
end

const _PS_CASES = (
    ("EE Gerstner -> Triplet",              1, (LTP = STDPGerstner(),),          (LTP = STDPTriplet(),)),
    ("EE Triplet -> WeightDependent",       1, (LTP = STDPTriplet(),),           (LTP = STDPWeightDependent(),)),
    ("EE WeightDependent -> Gerstner",      1, (LTP = STDPWeightDependent(),),   (LTP = STDPGerstner(),)),
    ("EE vSTDP event -> Gerstner",          1, (LTP = vSTDPParameter(),),        (LTP = STDPGerstner(),)),
    ("EE vSTDP dense -> vSTDP event",       1, (LTP = _PS.vSTDPDenseParameter(),),   (LTP = vSTDPParameter(),)),
    ("IE iSTDPRate -> iSTDPPotential",      3, (LTP = iSTDPRate(),),             (LTP = iSTDPPotential(),)),
    ("EI Markram event -> Markram timestep", 2, (STP = MarkramSTPParameter(),),  (STP = MarkramSTPParameterTimestep(),)),
    ("EE Gerstner+Markram -> Triplet+Markram timestep", 1, (LTP = STDPGerstner(), STP = MarkramSTPParameter()),
                                                            (LTP = STDPTriplet(), STP = MarkramSTPParameterTimestep())),
)

@testset "T1 round trip per rule (switch on, off, on again)" begin
    for (k, (label, i, r1, r2)) in enumerate(_PS_CASES)
        @testset "$label" begin
            _ps_roundtrip(i, r1, r2; alt = isodd(k))
        end
    end
    _, i, r1, r2 = _PS_CASES[end]
    @testset "EE Gerstner+Markram, Dirac delay" begin
        _ps_roundtrip(i, r1, r2; delay = Dirac(0.8f0))
    end
    @testset "EE Gerstner+Markram, heterogeneous delays" begin
        _ps_roundtrip(i, r1, r2; delay = Uniform(0.5f0, 1.5f0), alt = true)
    end
end

@testset "T2 the source of with_plasticity still simulates identically" begin
    # (i) switch-on result not inserted: the never-plastic source is unchanged and runs as an untouched twin
    a = _ps_net(); b = _ps_net()
    n0 = _ps_bps(a.C[1])
    on = with_plasticity(a.C[1]; LTP = STDPGerstner(), STP = MarkramSTPParameter())
    @test _ps_built(on) && length(on.ρ) == length(on.W)
    @test _ps_lazy(a.C[1]) && isempty(a.C[1].ρ) && _ps_bps(a.C[1]) == n0
    _ps_sim!(a, 30ms); _ps_sim!(b, 30ms)
    _ps_train!(a, 30ms); _ps_train!(b, 30ms)
    @test _ps_state(a) == _ps_state(b)
    @test _ps_lazy(a.C[1]) && isempty(a.C[1].ρ)
    # (ii) source of a switch-off keeps learning with its J/index
    a = _ps_net(ltpEE = STDPGerstner()); b = _ps_net(ltpEE = STDPGerstner())
    _ps_train!(a, 30ms); _ps_train!(b, 30ms)
    off = without_plasticity(a.C[1])
    @test _ps_lazy(off) && _ps_built(a.C[1])
    _ps_train!(a, 30ms); _ps_train!(b, 30ms)
    @test _ps_state(a) == _ps_state(b)
    @test _ps_built(a.C[1])
    # (iii) deferred weights: the only source mutation is sync_weights! (trajectory-neutral)
    a = _ps_net(ltpEE = vSTDPParameter()); b = _ps_net(ltpEE = vSTDPParameter())
    _ps_train!(a, 30ms); _ps_train!(b, 30ms)
    off = without_plasticity(a.C[1])
    _PS.sync_weights!(b.C[1])
    @test off.W == b.C[1].W
    _ps_train!(a, 30ms); _ps_train!(b, 30ms)
    @test _ps_state(a) == _ps_state(b)
end

@testset "T3 records across the switch" begin
    a = _ps_net(ltpEE = STDPGerstner(), stpEE = MarkramSTPParameter())
    c = a.C[1]
    monitor!(c, [:W, :ρ]; sr = 1kHz)
    monitor!(c, [:tpre], :LTPVars; sr = 1kHz)
    monitor!(c, [:u], :STPVars; sr = 1kHz)
    _ps_train!(a, 20ms)
    tpre20 = copy(getvariable(c, :LTPVars_tpre))
    a.C[1] = off = without_plasticity(c)
    @test !(:LTPVars_tpre in off.records[:data]) && !(:STPVars_u in off.records[:data])
    @test :W in off.records[:data] && :ρ in off.records[:data]
    @test length(off.ρ) == length(off.W) && all(==(1), off.ρ)     # recorded efficacy kept, value 1
    @test _ps_lazy(off)
    Wfrozen = copy(off.W)
    _ps_train!(a, 20ms)                                          # no FieldError (D4)
    for key in (:LTPVars_tpre, :STPVars_u)
        @test size(getvariable(off, key), 2) == 21
        @test off.records[:end_time][key] == 20
        @test off.records[:meta][:var_map][key] isa Tuple
    end
    @test getvariable(off, :LTPVars_tpre) == tpre20
    _, r = record(off, :LTPVars_tpre; range = true)
    @test first(r) == 0 && last(r) == 20
    Wr = getvariable(off, :W); ρr = getvariable(off, :ρ)
    @test size(Wr, 2) == 41 && size(ρr, 2) == 41
    @test all(s -> Wr[:, s] == Wfrozen, 22:41)
    @test all(==(1), ρr[:, 22:41]) && !all(==(1), ρr[:, 1:21])
    _, rW = record(off, :W; range = true)
    @test matrix_record(off, :W, rW[end]) == matrix(off)
    @test _ps_lazy(off) && isempty(off.J)
    # population records continue
    @test any(st -> any(>(20), st), spiketimes(a.E))
    # a record defined on the never-plastic source continues through on -> off, one sample per period
    n = _ps_net()
    monitor!(n.C[1], [:W]; sr = 1kHz)
    _ps_sim!(n, 10ms)
    n.C[1] = with_plasticity(n.C[1]; LTP = STDPGerstner())
    _ps_train!(n, 10ms)
    n.C[1] = without_plasticity(n.C[1])
    _ps_train!(n, 10ms)
    @test size(getvariable(n.C[1], :W), 2) == 31
    @test :W in n.C[1].records[:data]
    # a rule change keeps a record whose field the new rule has, and closes it otherwise
    for (rule, kept) in ((STDPWeightDependent(), true), (STDPTriplet(), false))
        m = _ps_net(ltpEE = STDPGerstner())
        monitor!(m.C[1], [:tpre], :LTPVars; sr = 1kHz)
        _ps_train!(m, 10ms)
        m.C[1] = with_plasticity(m.C[1]; LTP = rule)
        @test (:LTPVars_tpre in m.C[1].records[:data]) == kept
        _ps_train!(m, 10ms)
        @test size(getvariable(m.C[1], :LTPVars_tpre), 2) == (kept ? 21 : 11)
    end
    # an STP-only change leaves the LTP records alone, and an unmonitored synapse has nothing to close
    m = _ps_net(ltpEE = STDPGerstner(), stpEE = MarkramSTPParameter())
    monitor!(m.C[1], [:tpre], :LTPVars; sr = 1kHz)
    monitor!(m.C[1], [:u], :STPVars; sr = 1kHz)
    m.C[1] = with_plasticity(m.C[1]; STP = NoSTP())
    @test :LTPVars_tpre in m.C[1].records[:data] && !(:STPVars_u in m.C[1].records[:data])
    _ps_train!(m, 5ms)
    @test isempty(_ps_net().C[2].records) && without_plasticity(_ps_net().C[2]).records isa Dict
end

@testset "T4 Float64" begin
    _ps_roundtrip(1, (LTP = STDPGerstner(), STP = MarkramSTPParameter()), (LTP = STDPTriplet(), STP = MarkramSTPParameterTimestep()); FT = Float64)
    _ps_roundtrip(3, (LTP = iSTDPRate(),), (LTP = iSTDPPotential(),); FT = Float64, alt = true)
    a = _ps_net(FT = Float64)
    on = with_plasticity(a.C[1]; LTP = STDPGerstner(), STP = MarkramSTPParameter())
    @test typeof(on.LTPParam) <: STDPGerstner{Float64} && eltype(on.ρ) == Float64 && eltype(on.W) == Float64
    off = without_plasticity(on)
    @test eltype(off.J) == Int32 && isempty(off.J) && eltype(off.ρ) == Float64 && isempty(off.ρ)
end

@testset "T5 threads: switches give the serial result" begin
    old = _PS.SYN_MIN_OPS[]
    try
        _PS.SYN_MIN_OPS[] = 0
        a = _ps_net(delay = Dirac(0.8f0)); b = _ps_net(delay = Dirac(0.8f0))
        nt = (neurons_per_thread = 64,)
        _ps_sim!(a, 20ms); _ps_sim!(b, 20ms; nt...)
        @test _ps_state(a) == _ps_state(b)
        for n in (a, b)
            n.C[1] = with_plasticity(n.C[1]; LTP = STDPGerstner(), STP = MarkramSTPParameter())
        end
        _ps_train!(a, 30ms); _ps_train!(b, 30ms; nt...)
        @test _ps_state(a) == _ps_state(b)
        for n in (a, b)
            n.C[1] = without_plasticity(n.C[1])
        end
        _ps_train!(a, 30ms); _ps_train!(b, 30ms; nt...)
        @test _ps_state(a) == _ps_state(b)
        for n in (a, b)
            n.C[1] = with_plasticity(n.C[1]; LTP = STDPTriplet())
        end
        _ps_train!(a, 30ms); _ps_train!(b, 30ms; nt...)
        @test _ps_state(a) == _ps_state(b)
    finally
        _PS.SYN_MIN_OPS[] = old; set_neurons_per_thread!(0)
    end
end

@testset "T6 other readers of J/index build them at use" begin
    # (i) a normalization built on the source keeps using the source's index, on the shared weights
    a = _ps_net(ltpEE = STDPGerstner())
    norm = SynapseNormalization([a.C[1]]; param = MultiplicativeNorm(τ = 10ms))
    src = a.C[1]
    a.C[1] = without_plasticity(src)
    @test _ps_lazy(a.C[1]) && _ps_built(src)
    a.C[1].W .*= 1.1f0
    train!(a.P, [a.C; norm]; duration = 20ms, time = a.T)
    c = a.C[1]
    sums = zeros(Float64, length(c.fireI))
    for s in eachindex(c.W)
        sums[c.I[s]] += c.W[s]
    end
    has = [any(==(i), c.I) for i in eachindex(sums)]
    @test isapprox(sums[has], Float64.(norm.W0[has]); rtol = 1e-5)
    @test _ps_lazy(a.C[1])
    # (ii) a normalization built on the switched-off synapse builds its transpose
    SynapseNormalization([a.C[1]]; param = MultiplicativeNorm(τ = 10ms))
    @test _ps_built(a.C[1])
    # (iii) presynaptic builds it on demand, same result as on the source
    b = _ps_net(ltpEE = STDPGerstner())
    off = without_plasticity(b.C[1])
    @test _ps_lazy(off)
    @test presynaptic(off, 1) == presynaptic(b.C[1], 1)
    @test _ps_built(off)
    # (iv) a never-plastic synapse whose transpose was built by a reader: the switched-off copy is lean, the input keeps its own
    n = _ps_net()
    presynaptic(n.C[1], 1)
    @test _ps_built(n.C[1])
    off = without_plasticity(n.C[1])
    @test _ps_lazy(off) && _ps_built(n.C[1])
end

@testset "T7 misc" begin
    n = _ps_net()
    c = n.C[1]
    off = without_plasticity(c)
    @test off !== c && typeof(off) === typeof(c) && _ps_lazy(off) && isempty(off.ρ)
    @test off.W === c.W && off.id == c.id
    # no keyword: rules and state are kept; J/index are shared only if a rule of the source needs them
    m = _ps_net(ltpEE = STDPGerstner(), stpEE = MarkramSTPParameter())
    c = m.C[1]
    same = with_plasticity(c)
    @test same.J === c.J && same.index === c.index && same.ρ === c.ρ && same.LTPVars === c.LTPVars && same.STPVars === c.STPVars
    # a never-plastic source whose transpose a reader built: no rule needs it, so the copy gets fresh empty J/index
    r = _ps_net().C[1]
    presynaptic(r, 1)
    @test _ps_built(r)
    r2 = with_plasticity(r)
    @test _ps_lazy(r2) && _ps_built(r)
    # nothing keeps a rule and its state; only the given part is replaced
    k = with_plasticity(c; STP = NoSTP())
    @test k.LTPVars === c.LTPVars && k.J === c.J && isempty(k.ρ) && length(c.ρ) == length(c.W)
    k = with_plasticity(c; LTP = NoLTP())
    @test k.STPVars === c.STPVars && k.ρ === c.ρ && _ps_lazy(k) && _ps_built(c)
    # a new short-term rule on a synapse whose ρ is stored shares it; a new long-term rule shares a built transpose
    k = with_plasticity(c; STP = MarkramSTPParameterTimestep(), LTP = STDPTriplet())
    @test k.ρ === c.ρ && k.J === c.J && k.index === c.index && k.STPVars !== c.STPVars
    # the step guard of a rule-less synapse returns at once (no work, no allocation)
    T = _PS.Time()
    _PS.plasticity!(off, off.param, 0.125f0, T); _PS.update_traces!(off, off.param, 0.125f0, T)
    @test (@allocated _PS.plasticity!(off, off.param, 0.125f0, T)) == 0
    @test (@allocated _PS.update_traces!(off, off.param, 0.125f0, T)) == 0
    @test !_PS._has_rules(off) && _PS._has_rules(c) && _PS._has_rules(k)
    @test _ps_arrays(off) == (T = false, lazyT = true, ρ = false, ρempty = true)
end
