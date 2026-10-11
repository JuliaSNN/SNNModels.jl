using SNNModels
using Test
using Random
using JLD2
using SparseArrays: nnz, nonzeros
using Distributions: Dirac, Uniform
@load_units

# Memory footprint of SpikingSynapse (SNNModels 1.10): the efficacy `ρ` is empty without STP, the transposed index arrays
# `J`/`index` are built only when something reads them (plasticity rules that walk the inputs of a neuron, `presynaptic*`,
# normalization/scaling, `ensure_transpose!`), and a delay common to all synapses is stored once.
# Reference (6b7addc, before the change): 20.3-21.0 B/syn without delays and 28.8-31.1 B/syn with delays (T4 network).

const _FP = SNNModels
_bps(c) = Base.summarysize(c) / length(c.W)
_net_bps(cs) = sum(Base.summarysize, cs) / sum(c -> length(c.W), cs)
_built(c) = length(c.J) == length(c.W) && length(c.index) == length(c.W)
_lazy(c) = isempty(c.J) && isempty(c.index)
function _ref_T(c)
    _, _, _, J, index, _ = dsparse(matrix(c))
    return Int32.(J), Int32.(index)
end
_T_ok(c) = _built(c) && (c.J, c.index) == _ref_T(c)

# deterministic E/I network driven by heterogeneous constant currents (about 200 Hz), spikes recorded
function _fp_net(; seed = 3, NE = 1600, NI = 400, delay = nothing, ltpEE = NoLTP(), stpEI = NoSTP(), ltpIE = NoLTP())
    Random.seed!(seed)
    E = IF(N = NE, name = "E"); I = IF(N = NI, name = "I")
    E.v .= -70.0f0 .+ 20.0f0 .* rand(Float32, NE); I.v .= -70.0f0 .+ 20.0f0 .* rand(Float32, NI)
    E.I .= 250.0f0 .+ 150.0f0 .* rand(Float32, NE); I.I .= 250.0f0 .+ 150.0f0 .* rand(Float32, NI)
    d = isnothing(delay) ? (;) : (delay_dist = delay,)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.05, μ = 0.25), name = "EE", LTPParam = ltpEE, d...)
    EI = SpikingSynapse(E, I, :ge; conn = (p = 0.05, μ = 0.25), name = "EI", STPParam = stpEI, d...)
    IE = SpikingSynapse(I, E, :gi; conn = (p = 0.05, μ = 2.0), name = "IE", LTPParam = ltpIE, d...)
    II = SpikingSynapse(I, I, :gi; conn = (p = 0.05, μ = 2.0), name = "II", d...)
    monitor!([E, I], [:fire])
    return (P = [E, I], C = SNNModels.AbstractConnection[EE, EI, IE, II], E = E, I = I)
end
_fp_state(net) = (spikes = [spiketimes(p) for p in net.P], v = [copy(p.v) for p in net.P],
                  g = [copy(getfield(p.synvars, f)) for p in net.P for f in fieldnames(typeof(p.synvars)) if getfield(p.synvars, f) isa AbstractArray],
                  W = [copy(plasticity_value(c, :W)) for c in net.C], eff = [copy(_FP.efficacy(c)) for c in net.C])

@testset "memory bound of non-plastic synapses" begin
    net = _fp_net()
    for c in net.C
        @test _lazy(c) && isempty(c.ρ)
        @test _bps(c) < 10.0                   # I + W = 8 B, per-neuron arrays 0.3-1.5 B (K = 20-80)
    end
    @test _net_bps(net.C) < 9.0
    # a delay common to all synapses is stored once
    net = _fp_net(delay = Dirac(0.8f0))
    for c in net.C
        @test length(c.param.delaytime) == 1 && c.param.delaytime[1] == 0.8f0
    end
    sim!(net.P, net.C; duration = 10ms)
    for c in net.C
        @test length(c.param.delaysteps) == 1
        @test _bps(c) < 12.0                   # + ring buffer Npost x 8 slots
    end
    @test _net_bps(net.C) < 10.5
    # heterogeneous delays keep one value per synapse
    net = _fp_net(delay = Uniform(0.5f0, 1.5f0))
    sim!(net.P, net.C; duration = 10ms)
    for c in net.C
        @test length(c.param.delaytime) == length(c.W) == length(c.param.delaysteps)
        @test _bps(c) < 22.0                   # + 8 B of delays, ring buffer Npost x 13 slots
    end
    @test _net_bps(net.C) < 18.5
end

@testset "construction temporaries" begin
    Random.seed!(1)
    E = IF(N = 2000)
    A = _FP.sparse_matrix(2000, 2000, (p = 0.05, μ = 1.0), Float32)
    f() = SpikingSynapse(E, E, :ge; conn = A)
    c = f(); a = @allocated f()
    @test a / length(c.W) < 32          # 6b7addc: 64.5 B/syn (sparse(A'), Int64 J/index, Int64 -> Int32 copies)
    @test (c.rowptr, c.colptr, c.I, c.W) == let (r, cp, I, _, _, W) = dsparse(matrix(c)); (Int32.(r), Int32.(cp), Int32.(I), W) end
end

@testset "lazy transpose: stays empty when not read" begin
    net = _fp_net()
    EE = net.C[1]
    sim!(net.P, net.C; duration = 5ms)
    @test all(_lazy, net.C)
    monitor!(EE, [:W]; sr = 1kHz)
    sim!(net.P, net.C; duration = 5ms)
    Wr, r = record(EE, :W, range = true)
    @test _lazy(EE)
    M = matrix(EE); Mρ = matrix(EE, :ρ)
    @test size(M) == (1600, 1600) && nnz(M) == length(EE.W) && all(nonzeros(Mρ) .== 1)
    @test matrix_record(EE, :W, r[end]) == M
    update_weights!(EE, 1, EE.I[EE.colptr[1]], 2.0f0)
    @test !isempty(indices(EE, [1, 2], collect(1:1600)))
    @test postsynaptic(EE, 1) == EE.I[EE.colptr[1]:(EE.colptr[2]-1)]
    @test _lazy(EE)
    i = first(i for i in 1:1600 if i != 1 && M[i, 1] == 0)
    connect!(EE, 1, i, 1.5f0)
    @test matrix(EE)[i, 1] == 1.5f0 && _lazy(EE)
    update_sparse_matrix!(EE)
    @test _lazy(EE) && matrix(EE)[i, 1] == 1.5f0
    net = _fp_net()             # (the :W record of EE has the old number of synapses)
    train!(net.P, net.C; duration = 5ms)
    @test all(_lazy, net.C)
end

@testset "lazy transpose: built when read" begin
    net = _fp_net()
    EE, EI, IE, II = net.C
    pre = presynaptic(EE, 1)
    @test _T_ok(EE) && pre == findall(!iszero, matrix(EE)[1, :])
    rng = presynaptic_idxs(EI, 2)
    @test _T_ok(EI) && sort(EI.index[rng]) == findall(==(2), EI.I)
    SynapseNormalization([IE]; param = MultiplicativeNorm(τ = 10ms))
    @test _T_ok(IE)
    AggregateScaling(net.I, [II]; param = AggregateScalingParameter(net.I.N, 10Hz))
    @test _T_ok(II)
    net = _fp_net()
    EE = net.C[1]
    @test _FP.ensure_transpose!(EE) === EE && _T_ok(EE) && _FP.has_transpose(EE)
    # rebuilt correctly after the storage changes
    M = matrix(EE)
    i = first(i for i in 1:1600 if i != 3 && M[i, 3] == 0)
    connect!(EE, 3, i, 1.5f0)
    @test _T_ok(EE)
    synaptic_turnover!(EE; p_rewire = 0.2)
    @test _T_ok(EE)
    # rules that read the inputs of a neuron build it at construction, the others do not
    E = IF(N = 50)
    for rule in (STDPGerstner(), STDPMexicanHat(), STDPWeightDependent(), STDPTriplet(), STDPSymmetric(), iSTDPRate(), iSTDPPotential())
        @test _T_ok(SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 1.0), LTPParam = rule))
    end
    for rule in (vSTDPParameter(), _FP.vSTDPDenseParameter())
        @test _lazy(SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 1.0), LTPParam = rule))
    end
    for rule in (MarkramSTPParameter(), MarkramSTPParameterTimestep(), MarkramSTPParameterHet(τD = fill(200.0f0, 50), τF = fill(1500.0f0, 50), U = fill(0.2f0, 50)))
        @test _lazy(SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 1.0), STPParam = rule))
    end
    @test _FP.needs_transpose(STDPGerstner()) && !_FP.needs_transpose(NoLTP()) && !_FP.needs_transpose(vSTDPParameter())
    c = SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 1.0))
    @test _lazy(c)
    c2 = with_plasticity(c; LTP = STDPGerstner())
    @test _T_ok(c2)
    # the step guard builds the arrays of a plastic synapse whose transpose was dropped (e.g. by user code)
    a = _fp_net(ltpEE = STDPGerstner()); b = _fp_net(ltpEE = STDPGerstner())
    resize!(b.C[1].J, 0); resize!(b.C[1].index, 0)
    train!(a.P, a.C; duration = 1ms); train!(b.P, b.C; duration = 1ms)
    @test _T_ok(b.C[1])
    train!(a.P, a.C; duration = 50ms); train!(b.P, b.C; duration = 50ms)
    @test _fp_state(a) == _fp_state(b)
end

@testset "efficacy ρ empty without STP" begin
    net = _fp_net(stpEI = MarkramSTPParameter())
    EE, EI = net.C
    n = length(EE.W)
    @test isempty(EE.ρ) && eltype(EE.ρ) == Float32
    @test _FP.efficacy(EE) == ones(Float32, n) && !(_FP.efficacy(EE) === EE.ρ)
    @test length(EI.ρ) == length(EI.W) && _FP.efficacy(EI) === EI.ρ
    c = with_plasticity(EE; STP = MarkramSTPParameter())
    @test length(c.ρ) == n
    monitor!(net.C[3], [:ρ])
    @test length(net.C[3].ρ) == length(net.C[3].W)
    sim!(net.P, net.C; duration = 2ms)
    @test all(x -> all(==(1.0f0), x), record(net.C[3], :ρ))
    # writes to ρ take effect after materialize_efficacy!
    E = IF(N = 1); P = Identity(N = 1)
    s = SpikingSynapse(P, E, :ge; conn = fill(2.0f0, 1, 1))
    @test _FP.materialize_efficacy!(s) === s.ρ && s.ρ == [1.0f0]
    s.ρ .= 0.5f0
    P.fire[1] = true
    SNNModels.forward!(s, s.param, 0.1f0, SNNModels.Time())
    @test s.g[1] == 1.0f0
    # the step guard stores ρ of a short-term synapse whose ρ was dropped (hand-built, loaded, user code) before the
    # Markram kernels write it; the run matches the materialised twin
    for stp in (MarkramSTPParameter(), MarkramSTPParameterTimestep())
        a = _fp_net(stpEI = stp); b = _fp_net(stpEI = stp)
        resize!(b.C[2].ρ, 0)
        train!(a.P, a.C; duration = 1ms); train!(b.P, b.C; duration = 1ms)
        @test length(b.C[2].ρ) == length(b.C[2].W)
        train!(a.P, a.C; duration = 50ms); train!(b.P, b.C; duration = 50ms)
        @test _fp_state(a) == _fp_state(b)
    end
    # ...and stays a no-op without STP
    net = _fp_net()
    train!(net.P, net.C; duration = 2ms)
    @test all(c -> isempty(c.ρ), net.C)
end

@testset "with_plasticity: the source synapse still simulates" begin
    # `c2 = with_plasticity(c)` shares the arrays of `c`; `c` keeps its own (no) rules and gives the same run as an
    # untouched twin
    a = _fp_net(); b = _fp_net()
    c2 = with_plasticity(b.C[1]; LTP = STDPGerstner())
    c3 = with_plasticity(b.C[2]; STP = MarkramSTPParameter())
    @test c2.W === b.C[1].W && length(c3.ρ) == length(c3.W) && isempty(b.C[2].ρ)   # ρ is new: the source is not modified
    @test b.C[1].LTPParam isa NoLTP && b.C[2].STPParam isa NoSTP
    sim!(a.P, a.C; duration = 50ms); sim!(b.P, b.C; duration = 50ms)
    @test _fp_state(a) == _fp_state(b)
    train!(a.P, a.C; duration = 20ms); train!(b.P, b.C; duration = 20ms)
    @test _fp_state(a) == _fp_state(b)
end

@testset "bit-identity of the compact storage" begin
    # (a) empty ρ vs materialised ones, with and without delays
    for delay in (nothing, Dirac(0.8f0), Uniform(0.5f0, 1.5f0))
        a = _fp_net(; delay); b = _fp_net(; delay)
        foreach(_FP.materialize_efficacy!, b.C)
        sim!(a.P, a.C; duration = 50ms); sim!(b.P, b.C; duration = 50ms)
        @test sum(length, a.P[1] |> spiketimes) > 0
        @test _fp_state(a) == _fp_state(b)
    end
    # (b) a uniform delay stored once vs stored per synapse, on the buffer path: without plasticity a delay stored once delivers
    # the delayed spike list instead (not bit-identical, see test/syn/delay_uniform.jl); a rule that `sim!` never applies keeps the buffer
    _buffer_path!(net) = (net.C .= [with_plasticity(c; LTP = STDPGerstner(A_pre = 0.0, A_post = 0.0)) for c in net.C]; net)
    a = _buffer_path!(_fp_net(delay = Dirac(0.8f0))); b = _buffer_path!(_fp_net(delay = Dirac(0.8f0)))
    for c in b.C
        c.param.delaytime = fill(0.8f0, length(c.W))
    end
    sim!(a.P, a.C; duration = 50ms); sim!(b.P, b.C; duration = 50ms)
    @test length(b.C[1].param.delaysteps) == length(b.C[1].W) && length(a.C[1].param.delaysteps) == 1
    @test _fp_state(a) == _fp_state(b)
    # rewiring keeps the delays of the stored synapses and gives a new synapse the mean delay, as with per-synapse storage
    _alldelays(c) = length(c.param.delaytime) == length(c.W) ? copy(c.param.delaytime) : fill(c.param.delaytime[1], length(c.W))
    M = matrix(a.C[1]); i = first(i for i in 2:1600 if M[i, 1] == 0)
    connect!(a.C[1], 1, i, 0.3f0); connect!(b.C[1], 1, i, 0.3f0)
    @test _alldelays(a.C[1]) == _alldelays(b.C[1])
    update_sparse_matrix!(a.C[2]); update_sparse_matrix!(b.C[2])
    @test length(a.C[2].param.delaytime) == 1 && _alldelays(a.C[2]) == _alldelays(b.C[2])
    @test length(b.C[2].param.delaytime) == 1          # equal delays are stored once after a rebuild
    sim!(a.P, a.C; duration = 20ms); sim!(b.P, b.C; duration = 20ms)
    @test _fp_state(a) == _fp_state(b)
    # (c) plastic net: rules at construction (eager) vs with_plasticity on a lazy synapse
    a = _fp_net(ltpEE = STDPGerstner(), stpEI = MarkramSTPParameter())
    b = _fp_net()
    b.C[1] = with_plasticity(b.C[1]; LTP = STDPGerstner()); b.C[2] = with_plasticity(b.C[2]; STP = MarkramSTPParameter())
    train!(a.P, a.C; duration = 100ms); train!(b.P, b.C; duration = 100ms)
    @test _fp_state(a) == _fp_state(b)
    # (d) threaded vs serial (needs more than one thread)
    if Threads.nthreads() > 1
        old = SNNModels.SYN_MIN_OPS[]
        try
            SNNModels.SYN_MIN_OPS[] = 0
            for (delay, ltp, stp) in ((nothing, NoLTP(), NoSTP()), (Dirac(0.8f0), NoLTP(), NoSTP()), (nothing, STDPGerstner(), MarkramSTPParameter()))
                a = _fp_net(; delay, ltpEE = ltp, stpEI = stp); b = _fp_net(; delay, ltpEE = ltp, stpEI = stp)
                        train!(a.P, a.C; duration = 50ms)
                train!(b.P, b.C; duration = 50ms, neurons_per_thread = 64)
                @test _fp_state(a) == _fp_state(b)
            end
        finally
            SNNModels.SYN_MIN_OPS[] = old; set_neurons_per_thread!(0)
        end
    end
end

@testset "lazy vs eager transpose: same results (weight records, vSTDP, pair STDP)" begin
    # the transposed arrays never change what is computed: twins with J/index built by hand give identical runs
    for (ltp, rec) in ((NoLTP(), true), (vSTDPParameter(), false), (STDPGerstner(), true))
        a = _fp_net(ltpEE = ltp); b = _fp_net(ltpEE = ltp)
        foreach(_FP.ensure_transpose!, b.C)
        if rec
            monitor!(a.C[1], [:W]; sr = 1kHz); monitor!(b.C[1], [:W]; sr = 1kHz)
        end
        train!(a.P, a.C; duration = 50ms); train!(b.P, b.C; duration = 50ms)
        @test _fp_state(a) == _fp_state(b)
        @test all(_built, b.C) && (ltp isa STDPGerstner ? _built(a.C[1]) : _lazy(a.C[1])) && all(_lazy, a.C[2:end])
        rec && @test record(a.C[1], :W) == record(b.C[1], :W)
    end
end

@testset "JLD2 round trip keeps the compact storage" begin
    a = _fp_net(ltpEE = STDPGerstner(), delay = Dirac(0.8f0))
    f = tempname() * ".jld2"
    JLD2.jldsave(f; net = (P = a.P, C = a.C))
    b = JLD2.load(f, "net")
    rm(f)
    @test _T_ok(b.C[1]) && _lazy(b.C[2]) && isempty(b.C[2].ρ) && length(b.C[2].param.delaytime) == 1
    train!(a.P, a.C; duration = 20ms); train!(b.P, b.C; duration = 20ms)
    @test _fp_state(a) == _fp_state(b)
end
