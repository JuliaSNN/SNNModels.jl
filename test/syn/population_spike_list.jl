using SNNModels
using Test
using Random
using Distributions: Dirac, Uniform
using SNNModels.Logging
@load_units

# One spike list per population (SNNModels 1.10): the integrate-and-fire family (IF, AdEx, ExtendedIF, HetRec, Tripod,
# BallAndStick, HH, IZ, MorrisLecar) owns one `SpikeIndex` (`p.spikes`), shared by every `SpikingSynapse` that reads the
# population and by the `:fire` recorder. The list is published (window opened, lazily) right after `integrate!` and
# retired at the end of the step, so `fire` is scanned at most once per step; outside the window every call rescans
# `fire` as before. Results must be bit-identical to the per-connection lists of c5aac91, reproduced here by building
# the same network with non-owned lists (`spikes = SpikeIndex(N)` on the populations and a fresh `SpikeIndex` per
# synapse list). The threading block is effective only with `julia -t N` (N > 1).
# (Helpers are prefixed `_pl`: the test files are included into one module by runtests.jl.)

const _PL = SNNModels
const _PL_DT = 0.125f0

# ── in-window probe: a connection placed last in C that logs the population's list ───────────────────────────────────
struct _PLProbeParam <: _PL.AbstractConnectionParameter end
mutable struct _PLProbe{P} <: _PL.AbstractConnection
    pop::P
    param::_PLProbeParam
    log::Vector{Any}
    records::Dict{Any,Any}
end
_PLProbe(pop) = _PLProbe(pop, _PLProbeParam(), Any[], Dict{Any,Any}())
function _PL.forward!(c::_PLProbe, ::_PLProbeParam, dt::AbstractFloat, T::_PL.Time)
    pop = c.pop
    s = pop.spikes
    active, valid = s.active, s.valid          # state left by the consumers that ran before the probe
    ref = Int32.(findall(pop.fire))
    # one scan per step: invert `fire` before reading the list. A rescan would return findall(.!fire) != ref; the
    # list built by the first consumer of the step is returned unchanged. `fire` is restored before returning.
    saved = copy(pop.fire)
    pop.fire .= .!saved
    list = collect(_PL.spike_list!(s, pop.fire))
    copyto!(pop.fire, saved)
    push!(c.log, (list, ref, active, valid))
    return nothing
end
_PL.plasticity!(c::_PLProbe, ::_PLProbeParam, dt::AbstractFloat, T::_PL.Time) = nothing
_PL.float_type(::Type{<:_PLProbe}) = Float32

# a connection that throws on its n-th forward! (exception inside the step, after the synapses consumed the lists)
mutable struct _PLThrow <: _PL.AbstractConnection
    param::_PLProbeParam
    n::Int
    at::Int
    records::Dict{Any,Any}
end
_PLThrow(at) = _PLThrow(_PLProbeParam(), 0, at, Dict{Any,Any}())
function _PL.forward!(c::_PLThrow, ::_PLProbeParam, dt::AbstractFloat, T::_PL.Time)
    c.n += 1
    c.n == c.at && error("_PLThrow: step $(c.at)")
    return nothing
end
_PL.plasticity!(c::_PLThrow, ::_PLProbeParam, dt::AbstractFloat, T::_PL.Time) = nothing
_PL.float_type(::Type{_PLThrow}) = Float32

# a population that is not in the integrate-and-fire family (generic `spike_index`: a new list per call)
mutable struct _PLPlain <: _PL.AbstractPopulation
    id::String
    name::String
    N::Int32
    fire::Vector{Bool}
    records::Dict{Any,Any}
end
_PLPlain(N) = _PLPlain("plain", "plain", Int32(N), zeros(Bool, N), Dict{Any,Any}())

const _PL_OWNERS = (IF, AdEx, ExtendedIF, HetRec, Tripod, BallAndStick, HH, IZ, MorrisLecar)

# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────────
_pl_strict(l) = all(k -> l[k] < l[k+1], 1:length(l)-1)

# STDP on the recurrent synapse under train!; none on HetRec (its synapses target the N * Nd dendrites, which the STDP
# kernels index as postsynaptic neurons: BoundsError at c5aac91 too, unrelated to the spike lists)
_pl_ltp(kind, mode) = (mode === :train && kind !== HetRec) ? STDPGerstner() : NoLTP()

# replace every spike list of the synapses by a fresh non-owned list (the per-connection lists of c5aac91)
function _pl_fresh_lists!(C)
    for c in C
        c isa SpikingSynapse || continue
        c.pre_spikes = _PL.SpikeIndex(length(c.fireJ))
        c.post_spikes = _PL.SpikeIndex(length(c.fireI))
    end
    return C
end

# every real array reachable from `x` (copied), skipping spike lists, record dictionaries and scratch memory
function _pl_arrays!(out, x, seen::IdDict, depth::Int)
    depth > 5 && return out
    (x isa _PL.SpikeIndex || x isa AbstractDict || x isa AbstractString || x isa Module || x isa Function) && return out
    if ismutable(x)
        haskey(seen, x) && return out
        seen[x] = true
    end
    if x isa AbstractArray{<:Real}
        push!(out, copy(x))
    elseif x isa AbstractArray
        for y in x
            _pl_arrays!(out, y, seen, depth + 1)
        end
    elseif isstructtype(typeof(x)) && !(x isa _PL.AbstractPopulation && depth > 0)
        for f in fieldnames(typeof(x))
            f in (:Δv_temp, :ic, :randcache, :chunkptr, :Wcur) && continue   # scratch (written before read)
            isdefined(x, f) || continue
            _pl_arrays!(out, getfield(x, f), seen, depth + 1)
        end
    end
    return out
end
_pl_state(objs) = (out = Any[]; seen = IdDict(); foreach(o -> _pl_arrays!(out, o, seen, 0), objs); out)

function _pl_recorded(obj)
    rec = obj.records[:fire]
    wp = length(rec[:times_buf]) - obj.records[:meta][:allocated][:fire]
    return rec[:times_buf][1:wp], rec[:neurons_buf][1:wp]
end

_pl_quiet(f) = with_logger(f, NullLogger())

function _pl_steps(f::F, P, C, S, r, T, n) where {F}
    for _ in 1:n
        f(P, C, S, r, T)
    end
    return nothing
end
_pl_steps_alloc(f::F, P, C, S, r, T, n) where {F} = @allocated _pl_steps(f, P, C, S, r, T, n)

# ── single population of each owner type, recurrent synapse, driven to fire ──────────────────────────────────────────
# `ref = true`: the c5aac91 behaviour (non-owned population list, fresh list per synapse list)
function _pl_single(kind; N = 50, seed = 1, FT = Float32, ref = false, ltp = NoLTP())
    Random.seed!(seed)
    sp = ref ? (spikes = _PL.SpikeIndex(N),) : (;)
    vft = FT === Float64 ? (VFT = Vector{FT},) : (;)
    S = _PL.AbstractStimulus[]
    if kind === IF
        p = IF(; N, sp..., vft...); p.I .= FT(400); sym = (:ge,)
    elseif kind === AdEx
        p = AdEx(; N, sp..., vft...); p.I .= FT(1500); sym = (:glu,)
    elseif kind === ExtendedIF
        p = ExtendedIF(; N, sp..., vft...); p.I .= FT(400); sym = (:glu,)
    elseif kind === HetRec
        p = _PL.Population(_PL.HetRecParameter(); N); sym = (:glu,)   # (does not forward `spikes`)
        push!(S, PoissonStimulus(p, :glu; param = PoissonFixed(rate = 2kHz)))
    elseif kind === Tripod
        p = Tripod(; N, sp..., vft...); sym = (:glu, :d1)
        push!(S, PoissonStimulus(p, :he; comp = :d1, param = PoissonFixed(rate = 10kHz)))
    elseif kind === BallAndStick
        p = BallAndStick(; N, sp..., vft...); sym = (:glu, :d)
        push!(S, PoissonStimulus(p, :he; comp = :d, param = PoissonFixed(rate = 10kHz)))
    elseif kind === HH
        p = HH(; N, sp..., vft...); p.I .= FT(600) .+ FT(600) .* rand(FT, N); sym = (:ge,)
    elseif kind === IZ
        p = IZ(; N, sp..., vft...); p.I .= FT(10); sym = (:ge,)
    elseif kind === MorrisLecar
        p = MorrisLecar(; N, sp..., vft...); p.I .= FT(100); sym = (:ge,)
    end
    c = SpikingSynapse(p, p, sym...; conn = (p = 0.1, μ = 0.5), LTPParam = ltp)
    C = _PL.AbstractConnection[c]
    ref && _pl_fresh_lists!(C)
    return (p = p, c = c, P = _PL.AbstractPopulation[p], C = C, S = S, T = _PL.Time{FT}())
end

function _pl_run!(net, mode, nsteps; probe = nothing, seed = 11, kw...)
    C = probe === nothing ? net.C : _PL.AbstractConnection[net.C..., probe]
    run = mode === :sim ? sim! : train!
    Random.seed!(seed)                          # same random stream for the stochastic models and stimuli
    _pl_quiet() do
        run(net.P, C, net.S; dt = _PL_DT, duration = nsteps * _PL_DT, time = net.T, kw...)
    end
    return net
end

function _pl_check_probe(probe)
    ok_list = all(e -> e[1] == e[2], probe.log)
    ok_sorted = all(e -> _pl_strict(e[1]), probe.log)
    ok_window = all(e -> e[3] && e[4], probe.log)
    nspk = sum(e -> length(e[1]), probe.log)
    return ok_list, ok_sorted, ok_window, nspk
end

# ── E/I network ───────────────────────────────────────────────────────────────────────────────────────────────────────
# IF (or Tripod E) network driven by heterogeneous constant currents; STDP on EE and iSTDP on IE; `:fire` recorded on E
# (all) and on I (`Isub`, `nothing`: all). `ref = true`: non-owned lists (c5aac91 behaviour).
function _pl_ei(; seed = 7, NE = 320, NI = 80, delay = nothing, kind = :if, FT = Float32, ref = false, Isub = 1:3:NI,
                  monitor = true)
    Random.seed!(seed)
    spE = ref ? (spikes = _PL.SpikeIndex(NE),) : (;)
    spI = ref ? (spikes = _PL.SpikeIndex(NI),) : (;)
    vft = FT === Float64 ? (VFT = Vector{FT},) : (;)
    E = kind === :tripod ? Tripod(; N = NE, name = "E", spE..., vft...) : IF(; N = NE, name = "E", spE..., vft...)
    I = IF(; N = NI, name = "I", spI..., vft...)
    if kind === :tripod
        E.v_s .= FT(-70) .+ FT(15) .* rand(FT, NE); E.I .= FT(1000) .+ FT(1000) .* rand(FT, NE)
    else
        E.v .= FT(-70) .+ FT(20) .* rand(FT, NE); E.I .= FT(250) .+ FT(150) .* rand(FT, NE)
    end
    I.v .= FT(-70) .+ FT(20) .* rand(FT, NI); I.I .= FT(250) .+ FT(150) .* rand(FT, NI)
    d = isnothing(delay) ? (;) : (delay_dist = delay,)
    tE = kind === :tripod ? (:glu, :d1) : (:ge,)
    tEi = kind === :tripod ? (:gaba, :s) : (:gi,)
    EE = SpikingSynapse(E, E, tE...; conn = (p = 0.05, μ = 0.5), name = "EE", LTPParam = STDPGerstner(), d...)
    EI = SpikingSynapse(E, I, :ge; conn = (p = 0.05, μ = 0.5), name = "EI", d...)
    IE = SpikingSynapse(I, E, tEi...; conn = (p = 0.05, μ = 2.0), name = "IE", LTPParam = iSTDPRate(), d...)
    II = SpikingSynapse(I, I, :gi; conn = (p = 0.05, μ = 2.0), name = "II", d...)
    C = _PL.AbstractConnection[EE, EI, IE, II]
    ref && _pl_fresh_lists!(C)
    if monitor
        monitor!(E, [:fire])
        Isub === nothing ? monitor!(I, [:fire]) : monitor!(I, [(:fire, collect(Isub))])
    end
    return (P = _PL.AbstractPopulation[E, I], C = C, S = _PL.AbstractStimulus[], E = E, I = I, T = _PL.Time{FT}())
end

function _pl_same(a, b)
    sa, sb = _pl_state([a.P..., a.C...]), _pl_state([b.P..., b.C...])
    length(sa) == length(sb) || return false
    return all(k -> isequal(sa[k], sb[k]), eachindex(sa))   # isequal: unused decay factors are NaN
end
_pl_same_records(a, b) = all(k -> _pl_recorded(a.P[k]) == _pl_recorded(b.P[k]), eachindex(a.P))

# sparse-order reference of `forward!` of an undelayed synapse without STP, from `fire` alone
function _pl_ref_forward(c, g0, fire)
    g = copy(g0)
    for j in findall(fire), s in c.colptr[j]:(c.colptr[j+1]-1)
        g[c.I[s]] += c.W[s]
    end
    return g
end

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
@testset "One spike list per population" begin

@testset "SpikeIndex: owned lists, arm/disarm, publish/retire" begin
    s = _PL.owned_spike_index(7)
    @test s.owned && !s.fed && !s.active && !s.valid && s.n == 0 && length(s.idx) == 7
    @test !_PL.SpikeIndex(7).owned && !_PL.fed_spike_index(7).owned && _PL.fed_spike_index(7).fed
    s.active = true; s.valid = true
    _PL.spike_list!(s, zeros(Bool, 7))
    _PL._arm!(s);    @test s.active && s.valid      # a connection can neither force a rescan ...
    _PL._disarm!(s); @test s.active && s.valid      # ... nor close the window of a shared list
    s.active = false
    _PL._arm!(s);    @test !s.active
    fire = Bool[0, 1, 0, 0, 1, 1, 0]
    @test _PL.spike_list!(s, fire) == [2, 5, 6]     # closed window: every call rescans
    fire[1] = true
    @test _PL.spike_list!(s, fire) == [1, 2, 5, 6]
    E = IF(N = 7)
    @test E.spikes isa _PL.SpikeIndex && E.spikes.owned
    @test _PL._publish_spikes!(E) === nothing
    @test E.spikes.active && !E.spikes.valid
    E.fire .= fire
    @test _PL.spike_list!(E.spikes, E.fire) == [1, 2, 5, 6] && E.spikes.valid
    E.fire[3] = true                                 # inside the window the list of the step is reused
    @test _PL.spike_list!(E.spikes, E.fire) == [1, 2, 5, 6]
    @test _PL._retire_spikes!(E) === nothing
    @test !E.spikes.active && !E.spikes.valid
    @test _PL.spike_list!(E.spikes, E.fire) == [1, 2, 3, 5, 6]
    Pn = Poisson(N = 3)                              # generic no-ops
    @test _PL._publish_spikes!(Pn) === nothing && _PL._retire_spikes!(Pn) === nothing
    _PL._publish_spikes!(E)
    @test _PL._retire_all_spikes!(_PL.AbstractPopulation[E, Pn]) === nothing
    @test !E.spikes.active
    _alloc_pub(p) = @allocated (_PL._publish_spikes!(p); _PL._retire_spikes!(p))
    _alloc_pub(E); _alloc_pub(Pn)
    @test _alloc_pub(E) == 0 && _alloc_pub(Pn) == 0
end

@testset "L1: list == findall(fire) at every step, scanned once (each owner type)" begin
    for kind in _PL_OWNERS, mode in (:sim, :train)
        net = _pl_single(kind; ltp = _pl_ltp(kind, mode))
        probe = _PLProbe(net.p)
        _pl_run!(net, mode, 400; probe)
        ok_list, ok_sorted, ok_window, nspk = _pl_check_probe(probe)
        @testset "$(nameof(kind)) $mode" begin
            @test length(probe.log) == 400
            @test ok_list                            # read with `fire` inverted: no rescan by the second consumer
            @test ok_sorted
            @test ok_window                          # the synapse already scanned: the probe reuses the list
            @test nspk >= 50
        end
    end
end

@testset "L1/L6: owned lists == per-connection lists (each owner type)" begin
    for kind in _PL_OWNERS, mode in (:sim, :train)
        ltp = _pl_ltp(kind, mode)
        a = _pl_single(kind; ltp)
        b = _pl_single(kind; ltp, ref = true)
        @test b.c.pre_spikes !== b.p.spikes && b.c.post_spikes !== b.p.spikes && b.c.pre_spikes !== b.c.post_spikes
        kind === HetRec || @test !b.p.spikes.owned
        _pl_run!(a, mode, 400); _pl_run!(b, mode, 400)
        @testset "$(nameof(kind)) $mode" begin
            @test _pl_same(a, b)
            @test a.p.fire == b.p.fire
        end
    end
end

@testset "L2: sharing" begin
    E = IF(N = 30); I = IF(N = 10)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 1.0))
    EI = SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 1.0))
    IE = SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 1.0))
    @test EE.pre_spikes === E.spikes && EE.post_spikes === E.spikes
    @test EI.pre_spikes === E.spikes && EI.post_spikes === I.spikes
    @test IE.pre_spikes === I.spikes && IE.post_spikes === E.spikes
    @test _PL.spike_index(E) === _PL.spike_index(E) === E.spikes
    for kind in _PL_OWNERS
        net = _pl_single(kind; N = 12)
        p = net.p
        @test _PL.spike_index(p) === p.spikes
        @test p.spikes.owned && !p.spikes.fed && length(p.spikes.idx) == 12
        @test net.c.pre_spikes === p.spikes && net.c.post_spikes === p.spikes
    end
    # Identity and Poisson unchanged: their own (fed) list
    Pn = Poisson(N = 8); Id = Identity(N = 8)
    sP = SpikingSynapse(Pn, E, :ge; conn = (p = 0.5, μ = 1.0))
    sI = SpikingSynapse(Id, E, :ge; conn = (p = 0.5, μ = 1.0))
    @test sP.pre_spikes === Pn.spikes && sI.pre_spikes === Id.spikes
    @test !Pn.spikes.owned && !Id.spikes.owned
    # a population outside the family: a new list per call
    pl = _PLPlain(8)
    @test _PL.spike_index(pl) !== _PL.spike_index(pl)
    # user-supplied non-owned list
    Eu = IF(N = 5, spikes = _PL.SpikeIndex(5))
    @test !Eu.spikes.owned && _PL.spike_index(Eu) === Eu.spikes
end

@testset "L3: with_plasticity / without_plasticity keep the population lists" begin
    E = IF(N = 30); I = IF(N = 10)
    c = SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 1.0))
    c2 = with_plasticity(c; LTP = STDPGerstner())
    @test c2.pre_spikes === E.spikes && c2.post_spikes === I.spikes
    c3 = without_plasticity(c2)
    @test c3.pre_spikes === E.spikes && c3.post_spikes === I.spikes
end

@testset "L4/L5: window closed after sim!/train!; direct calls see hand-set flags" begin
    for mode in (:sim, :train)
        net = _pl_ei(; monitor = false)
        _pl_run!(net, mode, 200)
        for p in net.P
            @test !p.spikes.active && !p.spikes.valid
        end
        # the step itself closes the window (internal `_sim_step!`/`_train_step!`, and before every perturbation!)
        step! = mode === :sim ? _PL._sim_step! : _PL._train_step!
        step!(net.P, net.C, _PL.AbstractStimulus[], Ref(_PL_DT), net.T)
        @test all(p -> !p.spikes.active && !p.spikes.valid, net.P)
        seen_open = Ref(false)
        pert = (; P, kw...) -> (seen_open[] |= any(p -> p.spikes.active, P))
        _pl_run!(net, mode, 20; perturbation! = pert)
        @test !seen_open[]
        EE, EI = net.C[1], net.C[2]
        _PL._arm!(EE.pre_spikes); @test !net.E.spikes.active
        # forward! with hand-set flags
        Random.seed!(3)
        for _ in 1:5
            net.E.fire .= rand(Bool, length(net.E.fire)) .& (rand(length(net.E.fire)) .< 0.1)
            fill!(EI.g, 0)
            ref = _pl_ref_forward(EI, EI.g, net.E.fire)
            _PL.forward!(EI, EI.param, _PL_DT, net.T)
            @test EI.g == ref
            @test any(!iszero, ref)
        end
        # plasticity! with STDP: same as a twin with per-connection lists
        twin = deepcopy(EE)
        _pl_fresh_lists!([twin])
        W0 = copy(EE.W)
        for _ in 1:5
            fl = rand(Bool, length(net.E.fire)) .& (rand(length(net.E.fire)) .< 0.2)
            net.E.fire .= fl; twin.fireJ .= fl; twin.fireI .= fl
            update_time!(net.T, _PL_DT)
            for c in (EE, twin)
                _PL.update_traces!(c, c.param, _PL_DT, net.T)
                _PL.plasticity!(c, c.param, _PL_DT, net.T)
            end
            @test EE.W == twin.W
        end
        @test EE.W != W0
    end
end

@testset "L6: bit identity of whole runs (owned vs per-connection lists)" begin
    for delay in (nothing, Dirac(1.0f0), Uniform(0.5f0, 3.0f0)), mode in (:sim, :train)
        a = _pl_ei(; delay)
        b = _pl_ei(; delay, ref = true)
        _pl_run!(a, mode, 1000); _pl_run!(b, mode, 1000)
        @testset "$(delay === nothing ? "no delay" : nameof(typeof(delay))) $mode" begin
            @test _pl_same(a, b)
            @test _pl_same_records(a, b)
            @test length(_pl_recorded(a.E)[2]) >= 100
            @test a.C[1].W == b.C[1].W && a.C[3].W == b.C[3].W
        end
    end
end

@testset "L7: threading (neurons_per_thread = 16) == serial" begin
    old = _PL.SYN_MIN_OPS[]
    try
        for kind in (:if, :tripod), mode in (:sim, :train)
            a = _pl_ei(; kind, delay = Dirac(1.0f0))
            b = _pl_ei(; kind, delay = Dirac(1.0f0))
            r = _pl_ei(; kind, delay = Dirac(1.0f0), ref = true)
            _pl_run!(a, mode, 600)
            _PL.SYN_MIN_OPS[] = 0
            _pl_run!(b, mode, 600; neurons_per_thread = 16)
            _PL.SYN_MIN_OPS[] = old
            _pl_run!(r, mode, 600)
            @testset "$kind $mode (threads = $(Threads.nthreads()))" begin
                @test _pl_same(a, b)
                @test _pl_same_records(a, b)
                @test _pl_same(a, r)
                @test length(_pl_recorded(a.E)[2]) >= 50
            end
        end
    finally
        _PL.SYN_MIN_OPS[] = old
    end
end

@testset "L8: recorder (all, sorted subsets, unsorted subset, record_zero!)" begin
    NI = 80
    for Isub in (nothing, [2, 5, 9, 30, 61], collect(1:2:NI), [50, 3, 17, 70, 9])
        a = _pl_ei(; Isub)
        b = _pl_ei(; Isub, ref = true)
        for x in (a, b)                       # flags at t = 0: sampled by record_zero! (window closed)
            x.E.fire[[3, 7]] .= true
            x.I.fire[[5, 9, 50]] .= true
        end
        _pl_run!(a, :sim, 800); _pl_run!(b, :sim, 800)
        @testset "I subset $(Isub === nothing ? "all" : Isub[1:min(end, 5)])" begin
            @test _pl_same_records(a, b)
            tA, iA = _pl_recorded(a.E)
            @test tA[1:2] == Float32[0, 0] && iA[1:2] == [3, 7]
            tI, iI = _pl_recorded(a.I)
            z = [i for i in (Isub === nothing ? (1:NI) : Isub) if i in (5, 9, 50)]
            @test iI[1:length(z)] == z && all(iszero, tI[1:length(z)])
            @test length(iI) > length(z)
        end
    end
end

@testset "L9: Float64 populations" begin
    for kind in (IF, AdEx, HH), mode in (:sim, :train)
        ltp = mode === :train ? STDPGerstner() : NoLTP()
        net = _pl_single(kind; FT = Float64, ltp)
        @test eltype(net.p.v) == Float64
        probe = _PLProbe(net.p)
        _pl_run!(net, mode, 400; probe)
        ok_list, ok_sorted, ok_window, nspk = _pl_check_probe(probe)
        b = _pl_single(kind; FT = Float64, ltp, ref = true)
        a = _pl_single(kind; FT = Float64, ltp)
        _pl_run!(a, mode, 400); _pl_run!(b, mode, 400)
        @testset "$(nameof(kind)) $mode" begin
            @test ok_list && ok_sorted && ok_window
            @test nspk >= 50
            @test _pl_same(a, b)
        end
    end
    a = _pl_ei(; FT = Float64); b = _pl_ei(; FT = Float64, ref = true)
    _pl_run!(a, :train, 600); _pl_run!(b, :train, 600)
    @test _pl_same(a, b) && _pl_same_records(a, b)
end

@testset "L10: exception inside sim! closes every list" begin
    # thrown by perturbation! (before step 5)
    net = _pl_ei(; monitor = false)
    pert = (; t, kw...) -> (t >= 4 * _PL_DT && error("perturbation at step 5"))
    @test_throws ErrorException sim!(net.P, net.C; dt = _PL_DT, duration = 20 * _PL_DT, time = net.T, perturbation! = pert)
    @test all(p -> !p.spikes.active, net.P)
    # thrown inside the step, after the synapses consumed the lists
    net = _pl_ei(; monitor = false)
    C = _PL.AbstractConnection[net.C..., _PLThrow(5)]
    @test_throws ErrorException sim!(net.P, C; dt = _PL_DT, duration = 20 * _PL_DT, time = net.T)
    @test all(p -> !p.spikes.active && !p.spikes.valid, net.P)
    C = _PL.AbstractConnection[net.C..., _PLThrow(5)]
    @test_throws ErrorException train!(net.P, C; dt = _PL_DT, duration = 20 * _PL_DT, time = net.T)
    @test all(p -> !p.spikes.active, net.P)
    # five-argument step
    th = _PLThrow(1)
    @test_throws ErrorException sim!(net.P, _PL.AbstractConnection[net.C..., th], _PL.AbstractStimulus[], _PL_DT, net.T)
    @test all(p -> !p.spikes.active, net.P)
    th = _PLThrow(1)
    @test_throws ErrorException train!(net.P, _PL.AbstractConnection[net.C..., th], _PL.AbstractStimulus[], _PL_DT, net.T)
    @test all(p -> !p.spikes.active, net.P)
    # then direct forward! sees hand-set flags
    EI = net.C[2]
    net.E.fire .= false; net.E.fire[[1, 4, 9, 100]] .= true
    fill!(EI.g, 0)
    ref = _pl_ref_forward(EI, EI.g, net.E.fire)
    _PL.forward!(EI, EI.param, _PL_DT, net.T)
    @test EI.g == ref
end

@testset "L11: modelcopy and save_model/load_model keep the sharing" begin
    net = _pl_ei(; monitor = false)
    model = compose(; E = net.E, I = net.I, EE = net.C[1], EI = net.C[2], IE = net.C[3], II = net.C[4], silent = true)
    _pl_quiet(() -> sim!(model, 50ms))
    cp = _PL.modelcopy(model)
    @test cp.syn.EE.pre_spikes === cp.pop.E.spikes && cp.syn.EE.post_spikes === cp.pop.E.spikes
    @test cp.syn.EI.post_spikes === cp.pop.I.spikes && cp.syn.IE.pre_spikes === cp.pop.I.spikes
    @test cp.pop.E.spikes !== model.pop.E.spikes
    dir = mktempdir()
    try
        info = (case = "poplist",)
        save_model(model = model, path = dir, name = "pl", info = info)
        ld = _pl_quiet(() -> load_model(dir, "pl", info)).model
        @test ld.syn.EE.pre_spikes === ld.pop.E.spikes && ld.syn.EI.pre_spikes === ld.pop.E.spikes
        @test ld.syn.IE.pre_spikes === ld.pop.I.spikes && ld.syn.EI.post_spikes === ld.pop.I.spikes
        @test ld.pop.E.spikes.owned && !ld.pop.E.spikes.active
        for m in (model, cp, ld)
            _pl_quiet(() -> train!(m, 50ms))
        end
        objs(m) = [m.pop.E, m.pop.I, m.syn.EE, m.syn.EI, m.syn.IE, m.syn.II]
        sm = _pl_state(objs(model))
        sc, sl = _pl_state(objs(cp)), _pl_state(objs(ld))
        @test length(sc) == length(sl) == length(sm)
        @test all(k -> isequal(sm[k], sc[k]), eachindex(sm))
        @test all(k -> isequal(sm[k], sl[k]), eachindex(sm))
    finally
        rm(dir; recursive = true, force = true)
    end
end

@testset "L12: memory" begin
    net = _pl_ei(; NE = 800, NI = 200, monitor = false)
    lists = IdDict{Any,Bool}()
    for o in (net.P..., net.C...), f in fieldnames(typeof(o))
        x = getfield(o, f)
        x isa _PL.SpikeIndex && (lists[x] = true)
    end
    @test length(lists) == 2
    N = 2000
    Random.seed!(1)
    Wd = Float32.(rand(N, N) .< 0.01)
    E = IF(N = N); pl = _PLPlain(N)
    # lists a synapse allocates: those not owned by its populations (owner pre: none; plain pre: one list of N Int32)
    _new_lists(c, pre, post) = filter(x -> !any(p -> hasproperty(p, :spikes) && x === p.spikes, (pre, post)),
                                      unique(objectid, [c.pre_spikes, c.post_spikes]))
    so = SpikingSynapse(E, E, :ge; conn = Wd)
    sp = SpikingSynapse(pl, E, :ge; conn = Wd)
    @test so.pre_spikes === E.spikes && so.post_spikes === E.spikes
    @test isempty(_new_lists(so, E, E))
    lp = _new_lists(sp, pl, E)
    @test length(lp) == 1 && lp[1] === sp.pre_spikes && sp.post_spikes === E.spikes
    @test sum(l -> length(l.idx), lp) == N && eltype(lp[1].idx) == Int32
    # allocation: 4N bytes saved per non-owner side (8N per synapse at c5aac91, where both lists were per synapse);
    # loose bound, the rest of the constructor varies by tens of bytes between calls; a full collection before each
    # call keeps pending finalizers (earlier tests) from running, and being counted, inside the measured call
    _ctor(pre, post) = (GC.gc(); @allocated SpikingSynapse(pre, post, :ge; conn = Wd))
    _ctor(E, E); _ctor(pl, E)
    a_owner = minimum(_ctor(E, E) for _ in 1:5)
    a_plain = minimum(_ctor(pl, E) for _ in 1:5)
    @test a_plain - a_owner >= 3N
end

@testset "L13: no allocation per step" begin
    net = _pl_ei(; monitor = false)
    P, C, S = net.P, net.C, _PL.AbstractStimulus[]
    r = Ref(_PL_DT)
    for f in (_PL._sim_step!, _PL._train_step!)
        _pl_steps_alloc(f, P, C, S, r, net.T, 10); _pl_steps_alloc(f, P, C, S, r, net.T, 10)
        @test _pl_steps_alloc(f, P, C, S, r, net.T, 1000) == 0
    end
    @test sum(net.E.fire) + sum(net.I.fire) >= 0
end

@testset "L14: population not in P feeding a synapse in C" begin
    Random.seed!(4)
    pre = IF(N = 60); post = IF(N = 40)        # both frozen (not in P): g changes only by forward!
    D = IF(N = 10); D.I .= 400
    c = SpikingSynapse(pre, post, :ge; conn = (p = 0.3, μ = 1.0))
    probe = _PLProbe(pre)
    P = _PL.AbstractPopulation[D]
    C = _PL.AbstractConnection[c, probe]
    T = _PL.Time()
    nsp = 0
    for _ in 1:50
        pre.fire .= rand(60) .< 0.15
        g0 = copy(c.g)
        ref = _pl_ref_forward(c, g0, pre.fire)
        sim!(P, C, _PL.AbstractStimulus[], _PL_DT, T)
        @test c.g == ref
        nsp += sum(pre.fire)
    end
    @test nsp > 50
    # never published: every call rescans, so the probe (which inverts `fire`) reads the complement of the spikes
    @test all(e -> e[1] == Int32.(setdiff(1:60, e[2])) && !e[3], probe.log)
end

end
