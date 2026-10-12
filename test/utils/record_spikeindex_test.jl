# Spike recording from the spike lists (SNNModels 1.10): the `:fire` recorder reads the ascending list of
# the neurons that fired (the population's own fed `SpikeIndex`, or a word scan of `fire` into a scratch
# list) instead of visiting every neuron. The recorded events must be bit-identical (times, neuron ids,
# order within a step) to the dense per-neuron scan of SNNModels <= 1.9, reproduced here by
# `_ref_capture!`. The threading block is effective only with `julia -t N` (N > 1).
using SNNModels, Test, Random
using Distributions: Normal
using SNNModels.Logging
@load_units

const _RS = SNNModels
const _DT_RS = 0.125f0

# ── reference recorder ───────────────────────────────────────────────────────
# Reproduces the dense recorder of 1.9: scan `eachindex(fire)` (or `ind` in its order), one event per
# set flag with the clock converted to Float32. Driven by the `perturbation!` hook (called before every
# step, when `fire` and the clock still hold the previous step) plus one capture after each `sim!`,
# deduplicated by the step counter (the capture at tt = 0 reproduces `record_zero!`).
mutable struct _RefRec
    obj::Any
    ind::Union{Nothing,Vector{Int}}
    times::Vector{Float32}
    ids::Vector{Int}
    last_tt::Int
end
_RefRec(obj, ind = nothing) = _RefRec(obj, ind, Float32[], Int[], -1)

function _ref_capture!(ref::_RefRec, T::Time)
    tt = Int(T.tt[1])
    tt == ref.last_tt && return
    ref.last_tt = tt
    fire = ref.obj.fire
    t = Float32(get_time(T))
    for i in (ref.ind === nothing ? eachindex(fire) : ref.ind)
        if fire[i]
            push!(ref.times, t)
            push!(ref.ids, i)
        end
    end
end
_capture!(refs, T) = foreach(r -> _ref_capture!(r, T), refs)

function _simref!(refs, net, T; duration, quiet = true, kwargs...)
    hook = (; kw...) -> _capture!(refs, T)
    run() = sim!(net.P, net.C, net.S; dt = _DT_RS, duration = Float32(duration), time = T, perturbation! = hook, kwargs...)
    quiet ? with_logger(run, NullLogger()) : run()
    _capture!(refs, T)
    return refs
end

function _recorded(obj)
    rec = obj.records[:fire]
    wp = length(rec[:times_buf]) - obj.records[:meta][:allocated][:fire]
    return rec[:times_buf][1:wp], rec[:neurons_buf][1:wp]
end

_bits(x::Vector{Float32}) = reinterpret(UInt32, x)

function _ascending_per_step(times, ids)
    for k in 2:length(ids)
        times[k] == times[k-1] && ids[k] <= ids[k-1] && return false
    end
    return true
end

# recorded == reference, bit by bit; spiketimes agree with the reference events
function _same_as_ref(ref::_RefRec)
    times, ids = _recorded(ref.obj)
    _bits(times) == _bits(ref.times) && ids == ref.ids || return false
    st = spiketimes(ref.obj)
    st_ref = [Float32[] for _ in eachindex(ref.obj.fire)]
    for (t, i) in zip(ref.times, ref.ids)
        push!(st_ref[i], t)
    end
    return all(i -> collect(st[i]) == st_ref[i], eachindex(st_ref))
end

# ── networks ─────────────────────────────────────────────────────────────────
_Pv(x...) = AbstractPopulation[x...]
_Cv(x...) = isempty(x) ? AbstractConnection[EmptySynapse()] : AbstractConnection[x...]
_Sv(x...) = isempty(x) ? AbstractStimulus[EmptyStimulus()] : AbstractStimulus[x...]

_noise(E, I0, σ) = CurrentStimulus(E; param = CurrentNoise(E; I_base = I0, I_dist = Normal(0, σ)))

function _spiketime_param(Nin, nspikes, tmax; seed = 5, pair = true)
    rng = MersenneTwister(seed)
    times = sort(Float32.(rand(rng, nspikes) .* tmax .+ 1))
    neurons = rand(rng, 1:Nin, nspikes)
    if pair    # two spikes of different neurons at the same time, given in descending neuron order
        k = nspikes ÷ 2
        times[k+1] = times[k]
        neurons[k], neurons[k+1] = 9, 3
    end
    return SpikeTimeParameter(times, neurons)
end

const _BUILDERS = [
    :IF => (N) -> (E = IF(N = N); (P = _Pv(E), C = _Cv(), S = _Sv(Stimulus(PoissonFixed(rate = 1.5kHz), E, :ge)), target = E)),
    :AdEx => (N) -> (E = AdEx(N = N); (P = _Pv(E), C = _Cv(), S = _Sv(Stimulus(PoissonFixed(rate = 6kHz), E, :ge)), target = E)),
    :Tripod => (N) -> (E = Tripod(N = N); (P = _Pv(E), C = _Cv(), S = _Sv(_noise(E, 2000pA, 500)), target = E)),
    :HH => (N) -> (E = HH(N = N); (P = _Pv(E), C = _Cv(), S = _Sv(Stimulus(PoissonFixed(rate = 3kHz), E, :ge)), target = E)),
    :IZ => (N) -> (E = IZ(N = N); (P = _Pv(E), C = _Cv(), S = _Sv(_noise(E, 10, 5)), target = E)),
    :Poisson_sparse => (N) -> (E = Poisson(N = 5N, param = PoissonParameter(10Hz)); (P = _Pv(E), C = _Cv(), S = _Sv(), target = E)),
    :Poisson_dense => (N) -> (E = Poisson(N = N, param = PoissonParameter(300Hz)); (P = _Pv(E), C = _Cv(), S = _Sv(), target = E)),
    :Poisson_het => (N) -> (E = Poisson(N = 2N, param = PoissonParameter(collect(range(1Hz, 100Hz, length = 2N)))); (P = _Pv(E), C = _Cv(), S = _Sv(), target = E)),
    :VariablePoisson => (N) -> (E = VariablePoisson(N = 2N, param = VariablePoissonParameter(r0 = 100Hz)); (P = _Pv(E), C = _Cv(), S = _Sv(), target = E)),
    :InhomogeneousPoisson => (N) -> (E = InhomogeneousPoisson(N = 2N, param = InhomogeneousPoissonParam(r0 = 50Hz, β = 8.0f0)); (P = _Pv(E), C = _Cv(), S = _Sv(), target = E)),
    :Identity_fed => (N) -> begin
        E = Identity(N = N)
        s = SpikeTimeStimulus(E, :g; param = _spiketime_param(40, 3000, 480), conn = (p = 0.2, μ = 1.0))
        (P = _Pv(E), C = _Cv(), S = _Sv(s), target = E)
    end,
    :Identity_syn => (N) -> begin
        pre = Poisson(N = 50, param = PoissonParameter(50Hz))
        E = Identity(N = N)
        (P = _Pv(pre, E), C = _Cv(SpikingSynapse(pre, E, :g; conn = (p = 0.2, μ = 1.0))), S = _Sv(), target = E)
    end,
    :PoissonStimulusLayer => (N) -> begin
        E = IF(N = 50)
        s = PoissonStimulusLayer(E, :ge; param = PoissonLayer(10Hz, N = 5N), conn = (p = 0.1, μ = 1.0))
        (P = _Pv(E), C = _Cv(), S = _Sv(s), target = s)
    end,
    :SpikeTimeStimulus => (N) -> begin
        E = IF(N = 20)
        s = SpikeTimeStimulus(E, :ge; param = _spiketime_param(N, 400, 480), conn = (p = 0.2, μ = 0.1))
        (P = _Pv(E), C = _Cv(), S = _Sv(s), target = s)
    end,
]

function _run_case(mk, N, ind; seed = 1, duration = 500ms, monitor_kw = (;), kwargs...)
    Random.seed!(seed)
    net = mk(N)
    key = ind === nothing ? :fire : (:fire, ind)
    monitor!(net.target, [key]; monitor_kw...)
    stored = ind === nothing ? nothing : net.target.records[:indices][:fire]
    ref = _RefRec(net.target, ind === nothing ? nothing : stored)
    _simref!([ref], net, Time(); duration, kwargs...)
    return ref, net
end

@testset "T1: recorded spikes equal the dense reference scan" begin
    for (name, mk) in _BUILDERS
        @testset "$name" begin
            ref, net = _run_case(mk, 200, nothing)
            times, ids = _recorded(net.target)
            @test _same_as_ref(ref)
            @test _ascending_per_step(times, ids)
            @test length(ids) >= 50
            name === :Identity_fed && @test net.target.spikes.fed
            name === :Identity_syn && @test !net.target.spikes.fed
            name === :Poisson_sparse && @test net.target.spikes.fed
            name === :Poisson_dense && @test !net.target.spikes.fed
            name === :PoissonStimulusLayer && @test net.target.spikes.fed
        end
    end
end

@testset "T2: subsets (sorted, unsorted, duplicates, small, large)" begin
    for (name, mk) in _BUILDERS
        @testset "$name" begin
            Random.seed!(99)
            N = length(mk(200).target.fire)
            rng = MersenneTwister(7)
            subsets = [
                :large_sorted => sort(randperm(rng, N)[1:N÷2]),
                :small_sorted => sort(randperm(rng, N)[1:max(2, N ÷ 16)]),
                :unsorted => randperm(rng, N)[1:N÷2],
                :duplicates => rand(rng, 1:N, N ÷ 3),
                :sorted_duplicates => sort(rand(rng, 1:N, N ÷ 3)),
            ]
            for (sname, ind) in subsets
                ref, net = _run_case(mk, 200, ind)
                times, ids = _recorded(net.target)
                @test _same_as_ref(ref)
                @test all(in(Set(ind)), ids)
                sname === :large_sorted && @test length(ids) >= 10
                issorted(ind, lt = <=) && @test _ascending_per_step(times, ids)
            end
        end
    end

    @testset "index vector replaced or mutated between sim! calls" begin
        Random.seed!(3)
        net = _BUILDERS[1][2](200)
        ind = collect(1:2:200)
        monitor!(net.target, [(:fire, ind)])
        stored = net.target.records[:indices][:fire]
        ref = _RefRec(net.target, stored)
        T = Time()
        _simref!([ref], net, T; duration = 100ms)
        stored .= collect(2:2:200)                  # in place: the mask is rebuilt at the next sim!
        _simref!([ref], net, T; duration = 100ms)
        newind = collect(150:-1:101)                # replaced (unsorted): order-preserving path
        net.target.records[:indices][:fire] = newind
        ref.ind = newind
        _simref!([ref], net, T; duration = 100ms)
        @test _same_as_ref(ref)
        @test length(ref.ids) >= 50
    end

    @testset "empty index vector set by hand records nothing" begin
        Random.seed!(3)
        net = _BUILDERS[1][2](100)
        monitor!(net.target, [:fire])
        net.target.records[:indices][:fire] = Int[]
        with_logger(NullLogger()) do
            sim!(net.P, net.C, net.S; duration = 100ms)
        end
        @test isempty(_recorded(net.target)[2])
    end
end

@testset "T3: chunked runs, buffer growth, one warning" begin
    mk = _BUILDERS[1][2]
    single, _ = _run_case(mk, 100, nothing; duration = 1000ms, monitor_kw = (monitor_rate = 1Hz,))
    Random.seed!(1)
    net = mk(100)
    monitor!(net.target, [:fire]; monitor_rate = 1Hz)
    ref = _RefRec(net.target)
    T = Time()
    logger = Test.TestLogger(min_level = Logging.Warn)
    with_logger(logger) do
        for _ in 1:20
            _simref!([ref], net, T; duration = 50ms, quiet = false)
        end
    end
    times, ids = _recorded(net.target)
    times1, ids1 = _recorded(single.obj)
    @test _same_as_ref(ref)
    @test _bits(times) == _bits(times1) && ids == ids1
    @test length(ids) > 1000
    @test length(net.target.records[:fire][:times_buf]) < 2 * length(ids) + 1000
    msgs = [string(l.message) for l in logger.logs]
    @test count(m -> occursin("Fire COO buffer", m), msgs) == 1
end

@testset "T4: manual 5-argument loop without _allocate_records!, then clear_records!" begin
    for (name, mk) in _BUILDERS[[1, 6, 11, 13]]
        @testset "$name" begin
            Random.seed!(4)
            net = mk(200)
            monitor!(net.target, [:fire])
            T = Time()
            ref = _RefRec(net.target)
            _ref_capture!(ref, T)                   # record_zero! at t = 0
            with_logger(NullLogger()) do
                for _ in 1:2000
                    SNNModels.sim!(net.P, net.C, net.S, _DT_RS, T)
                    _ref_capture!(ref, T)
                end
            end
            @test _same_as_ref(ref)
            @test length(ref.ids) >= 50
            clear_records!(net.target)
            @test isempty(_recorded(net.target)[2])
            ref2 = _RefRec(net.target)
            with_logger(NullLogger()) do
                for _ in 1:2000
                    SNNModels.sim!(net.P, net.C, net.S, _DT_RS, T)
                    _ref_capture!(ref2, T)
                end
            end
            @test _same_as_ref(ref2)
            @test length(ref2.ids) >= 50
        end
    end
end

function _ei_network(; NE = 800, NI = 200)
    E = IF(N = NE, name = "E")
    I = IF(N = NI, name = "I")
    syns = _Cv(
        SpikingSynapse(E, E, :ge; conn = (p = 0.05, μ = 0.5)),
        SpikingSynapse(E, I, :ge; conn = (p = 0.05, μ = 0.5)),
        SpikingSynapse(I, E, :gi; conn = (p = 0.05, μ = 2.0)),
        SpikingSynapse(I, I, :gi; conn = (p = 0.05, μ = 2.0)),
    )
    S = _Sv(Stimulus(PoissonFixed(rate = 1.5kHz), E, :ge), Stimulus(PoissonFixed(rate = 1.5kHz), I, :ge))
    return (P = _Pv(E, I), C = syns, S = S, E = E, I = I)
end

@testset "T5: threading (neurons_per_thread = 64), nthreads = $(Threads.nthreads())" begin
    for (name, mk) in _BUILDERS[[1, 3, 12, 6, 7]]
        @testset "$name" begin
            serial, _ = _run_case(mk, 1000, nothing; duration = 200ms)
            threaded, net = _run_case(mk, 1000, nothing; duration = 200ms, neurons_per_thread = 64)
            ts, is = _recorded(serial.obj)
            tt, it = _recorded(threaded.obj)
            @test _same_as_ref(threaded)
            @test _bits(tt) == _bits(ts) && it == is
            @test _ascending_per_step(tt, it)
            @test length(it) >= 50
        end
    end
    @testset "E/I network with SpikingSynapses" begin
        function run(npt)
            Random.seed!(8)
            net = _ei_network()
            monitor!([net.E, net.I], [:fire])
            refs = [_RefRec(net.E), _RefRec(net.I)]
            _simref!(refs, net, Time(); duration = 300ms, neurons_per_thread = npt)
            return refs
        end
        rs, rt = run(nothing), run(64)
        for k in 1:2
            @test _same_as_ref(rt[k])
            @test _bits(rt[k].times) == _bits(rs[k].times) && rt[k].ids == rs[k].ids
            @test length(rt[k].ids) >= 50
        end
    end
end

function _steps!(P, C, S, T, n)
    for _ in 1:n
        SNNModels.sim!(P, C, S, _DT_RS, T)
    end
end

@testset "T6: no allocation per step from :fire recording" begin
    function measure(monitored, ind)
        Random.seed!(6)
        E = IF(N = 400)
        Q = Poisson(N = 1000, param = PoissonParameter(10Hz))
        net = (P = _Pv(E, Q), C = _Cv(), S = _Sv(Stimulus(PoissonFixed(rate = 1.5kHz), E, :ge)))
        if monitored   # buffers pre-sized for 1000 Hz, no growth
            monitor!(E, [ind === nothing ? :fire : (:fire, ind)]; monitor_rate = 1000Hz)
            monitor!(Q, [:fire]; monitor_rate = 1000Hz)
        end
        SNNModels._allocate_records!(net.P, net.C, net.S, _DT_RS, 3000f0)
        T = Time()
        _steps!(net.P, net.C, net.S, T, 20)          # warm-up: compilation, scratch list, mask
        return @allocated _steps!(net.P, net.C, net.S, T, 1000)
    end
    for ind in (nothing, collect(1:3:400), collect(1:7), collect(400:-2:1))
        measure(false, ind); measure(true, ind)
        a0 = measure(false, ind)
        a1 = measure(true, ind)
        @test (a1 - a0) / 1000 <= 0
    end
end

function _alloc_records(nunmon; monitored = false)
    P = AbstractPopulation[IF(N = 10) for _ in 1:nunmon]
    C = _Cv()
    S = _Sv()
    if monitored
        E = IF(N = 100)
        monitor!(E, [:fire, :v])
        P = _Pv(E)
    end
    SNNModels._allocate_records!(P, C, S, _DT_RS, 1000f0)   # first allocation of the buffers
    SNNModels._allocate_records!(P, C, S, _DT_RS, 1000f0)
    return @allocated SNNModels._allocate_records!(P, C, S, _DT_RS, 1000f0)
end

@testset "T7: _allocate_records! allocations" begin
    a7, a13 = _alloc_records(5), _alloc_records(11)       # + EmptySynapse + EmptyStimulus
    @test a7 == a13
    @test a13 <= 64
    am = _alloc_records(0; monitored = true)              # monitored IF (:fire, :v) with room, 3 components
    @info "_allocate_records! bytes" unmonitored_7 = a7 unmonitored_13 = a13 monitored_IF = am
    @test am <= 64 * 3
end

@testset "T8: Float64 populations and Time{Float64}" begin
    Random.seed!(1)
    E = IF(N = 200, param = IFParameter(FT = Float64), spike = PostSpike(FT = Float64),
           synapse = DoubleExpSynapse(FT = Float64), VFT = Vector{Float64})
    Id = Identity(N = 100, VFT = Vector{Float64})
    net = (P = _Pv(E, Id), C = _Cv(SpikingSynapse(E, Id, :g; conn = (p = 0.2, μ = 1.0))),
           S = _Sv(Stimulus(PoissonFixed(rate = 1.5kHz), E, :ge)))
    monitor!(E, [:fire])
    monitor!(Id, [(:fire, collect(1:2:100))])
    refs = [_RefRec(E), _RefRec(Id, Id.records[:indices][:fire])]
    T = Time{Float64}()
    _simref!(refs, net, T; duration = 500.0)
    for r in refs
        @test _same_as_ref(r)
        @test length(r.ids) >= 50
    end
    @test eltype(E.records[:fire][:times_buf]) == Float32
end

_owner_types(f, generic) = Set(
    Base.typename(T).wrapper
    for m in methods(f)
    for T in Base.uniontypes(Base.unwrap_unionall(m.sig).parameters[2])
    if T !== generic
)

@testset "T9: types owning a spike list == types the recorder reads the list of" begin
    owners = _owner_types(SNNModels.spike_index, AbstractPopulation)
    readers = _owner_types(SNNModels._recording_spike_index, Any)
    @test owners == readers
    @test length(owners) == 14                   # 5 fed-list types + the 9 owners of the integrate-and-fire family
    E = IF(N = 3)
    @test SNNModels._recording_spike_index(E) === E.spikes === SNNModels.spike_index(E)
    P = Poisson(N = 3)
    @test SNNModels._recording_spike_index(P) === P.spikes
end

@testset "T10: out-of-range subset index" begin
    for bad in ([1, 0], [5, 101], [101])
        Random.seed!(1)
        net = _BUILDERS[1][2](100)
        monitor!(net.target, [(:fire, bad)])
        err = try
            with_logger(NullLogger()) do
                sim!(net.P, net.C, net.S; duration = 10ms)
            end
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin(net.target.name, err.msg)
    end
end

@testset "T11: zero spikes and all-spike steps" begin
    E = IF(N = 100)
    net = (P = _Pv(E), C = _Cv(), S = _Sv())
    monitor!(E, [:fire])
    ref = _RefRec(E)
    _simref!([ref], net, Time(); duration = 100ms)
    @test isempty(ref.ids) && isempty(_recorded(E)[2])

    Id = Identity(N = 100)
    monitor!(Id, [:fire])
    T = Time()
    with_logger(NullLogger()) do
        for _ in 1:3
            Id.g .= 1
            SNNModels.sim!(_Pv(Id), _Cv(), _Sv(), _DT_RS, T)
        end
    end
    times, ids = _recorded(Id)
    @test ids == repeat(1:100, 3)
    @test _bits(times) == _bits(Float32.(repeat(Float32[1, 2, 3] .* _DT_RS, inner = 100)))
end

@testset "internal helpers" begin
    @testset "_ensure_fire_capacity!" begin
        rec = Dict{Symbol,AbstractVector}(:times_buf => Float32[1, 2, 3, 0], :neurons_buf => [1, 2, 3, 0])
        meta = Dict{Symbol,Any}(:allocated => Dict(:fire => 1), :grew => Dict(:fire => false))
        @test_logs SNNModels._ensure_fire_capacity!(rec, meta, 1)       # room: nothing happens
        @test length(rec[:times_buf]) == 4
        @test_logs (:warn, r"Fire COO buffer overflowed") SNNModels._ensure_fire_capacity!(rec, meta, 2)
        @test length(rec[:times_buf]) == 8                               # max(2 * 4, 3 + 2, 1)
        @test rec[:times_buf][1:3] == Float32[1, 2, 3] && rec[:neurons_buf][1:3] == [1, 2, 3]
        @test meta[:allocated][:fire] == 5
        @test_logs SNNModels._ensure_fire_capacity!(rec, meta, 20)      # latched: no second warning
        @test length(rec[:times_buf]) == 23                              # max(16, 3 + 20, 1)
        @test meta[:allocated][:fire] == 20
        empty = Dict{Symbol,AbstractVector}(:times_buf => Float32[], :neurons_buf => Int[])
        meta0 = Dict{Symbol,Any}(:allocated => Dict(:fire => 0), :grew => Dict(:fire => true))
        SNNModels._ensure_fire_capacity!(empty, meta0, 7)
        @test length(empty[:times_buf]) == 7 && meta0[:allocated][:fire] == 7
    end

    @testset "_fire_step_list" begin
        fire = zeros(Bool, 37)
        fire[[2, 9, 17, 36, 37]] .= true
        meta = Dict{Symbol,Any}()
        @test collect(SNNModels._fire_step_list(nothing, fire, meta)) == [2, 9, 17, 36, 37]
        s = SNNModels.fed_spike_index(37)
        s.idx[1:2] .= (4, 5); s.n = 2
        @test collect(SNNModels._fire_step_list(s, fire, meta)) == [4, 5]      # fed: the owner's list
        s.fed = false
        @test collect(SNNModels._fire_step_list(s, fire, meta)) == [2, 9, 17, 36, 37]
        @test eltype(SNNModels._fire_step_list(s, fire, meta)) == Int32
    end

    @testset "_fire_mask!" begin
        meta = Dict{Symbol,Any}()
        ind = [3, 1, 4]
        mask, sorted = SNNModels._fire_mask!(meta, ind, 5)
        @test mask == Bool[1, 0, 1, 1, 0] && !sorted
        ind2 = [1, 4]
        mask2, sorted2 = SNNModels._fire_mask!(meta, ind2, 5)
        @test mask2 == Bool[1, 0, 0, 1, 0] && sorted2
        @test !SNNModels._fire_mask!(meta, [2, 2], 5)[2]                     # duplicates: not strictly increasing
        @test_throws ArgumentError SNNModels._fire_mask!(meta, [6], 5)
    end
end
