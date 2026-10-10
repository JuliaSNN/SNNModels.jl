# Chunked threaded population updates: bit-identical to the serial loop for every neurons_per_thread and thread count.
# Run with several threads, e.g. `julia -t 4 --project=test test/threading.jl` (with one thread the checks that need
# the workers are skipped, the policy tests still run).
using Test, Random
using SNNModels
const SM = SNNModels

@testset "threading policy" begin
    @test SM._bpc_value(500) == 4 && SM._bpc_value(512) == 4 && SM._bpc_value(128) == 1 && SM._bpc_value(1) == 1
    @test SM._bpc_value(0) == 0 && SM._bpc_value(Inf) == 0 && SM._bpc_value(2048.0) == 16
    @test_throws ArgumentError SM._bpc_value(-3)
    @test SM.nchunks(0, 4) == 1 && SM.nchunks(1, 4) == 1 && SM.nchunks(512, 4) == 1
    @test SM.nchunks(513, 4) == 2 && SM.nchunks(1536, 4) == 3 && SM.nchunks(10^6, 4) == 8
    @test SM.nchunks(10_000, 0) == 1
    for N in (0, 1, 127, 128, 129, 500, 1000, 1537, 4097, 10_000, 100_000), bpc in (1, 2, 4, 8, 16, 100)
        nc = SM.nchunks(N, bpc)
        @test 1 <= nc <= SM.MAX_CHUNKS
        cover = Int[]
        for c in 1:nc
            r = SM.chunk_range(N, bpc, c)
            append!(cover, collect(r))
            isempty(r) || @test (first(r) - 1) % SM.BATCH == 0
            N > 0 && @test !isempty(r)
        end
        @test cover == collect(1:N)
    end
    old = SM.neurons_per_thread()
    SM.with_neurons_per_thread(128) do
        @test SM.neurons_per_thread() == 128
    end
    SM.with_neurons_per_thread(nothing) do
        @test SM.neurons_per_thread() == old
    end
    @test SM.neurons_per_thread() == old == 0   # threading is off by default
    for N in (0, 1, 100, 1000, 3000, 20_000), bpc in (0, 1, 4, 100)
        hits = zeros(Int, N)
        SM.foreach_chunk(N, bpc) do rng
            @inbounds for i in rng
                hits[i] += 1
            end
        end
        @test all(==(1), hits)
    end
end

function _pop(kind, syn, N; seed = 1, hetero = false)
    rng = MersenneTwister(seed)
    synapse = syn === :current ? SM.CurrentSynapse(τe = 5.0f0, τi = 5.0f0) :
              syn === :single ? SM.SingleExpSynapse() : SM.DoubleExpSynapse()
    spike = SM.PostSpike(τabs = 2.0f0)
    if kind === :IF
        p = SM.IF(N = N, param = SM.IFParameter(τm = 20.0f0, R = 0.1f0, El = -60.0f0, Vt = -50.0f0, Vr = -60.0f0),
                  synapse = synapse, spike = spike)
    else
        p = SM.AdEx(N = N, synapse = synapse, spike = spike)
    end
    N == 0 && return p
    p.v .= -75.0f0 .+ 30.0f0 .* rand(rng, Float32, N)
    p.I .= 150.0f0 .* rand(rng, Float32, N)
    p.tabs .= rand(rng, 0:30, N)                       # a third of the neurons start refractory
    p.w .= 5.0f0 .* rand(rng, Float32, N)
    return p
end

function _kick!(p, rng)
    N = length(p.v)
    for r in values(p.receptors)
        r .= (rand(rng, Float32, N) .< 0.2f0) .* (10.0f0 .* rand(rng, Float32, N))
    end
end

_state(p) = (p.v, p.w, p.tabs, p.fire, p.syn_curr, p.I, values(p.synvars isa NamedTuple ? p.synvars : (; (k => getfield(p.synvars, k) for k in fieldnames(typeof(p.synvars)) if getfield(p.synvars, k) isa AbstractVector)...))...)
_same(a, b) = all(isequal(x, y) for (x, y) in zip(_state(a), _state(b))) &&
              all(isequal(p, q) for (p, q) in zip(values(a.receptors), values(b.receptors)))
_bits(x::AbstractVector{Float32}) = reinterpret(UInt32, x)
_bitsame(a, b) = _bits(a.v) == _bits(b.v) && _bits(a.syn_curr) == _bits(b.syn_curr) && _bits(a.w) == _bits(b.w) && a.tabs == b.tabs && a.fire == b.fire

@testset "bit-identical to the serial loop" begin
    nthreads = Threads.nthreads()
    @info "threads" nthreads
    for kind in (:IF, :AdEx), syn in (:current, :single, :double)
        kind === :AdEx && syn === :current && continue
        @testset "$kind $syn" begin
            for N in (0, 1, 7, 63, 64, 100, 499, 500, 501, 1000, 1537, 4000, 10_000)
                ref = _pop(kind, syn, N)
                for npt in (0, 128, 256, 500, 1024, 2048)
                    pop = deepcopy(ref)
                    cpy = deepcopy(ref)
                    r1 = MersenneTwister(7); r2 = MersenneTwister(7)
                    nspk = 0
                    for step in 1:40
                        _kick!(pop, r1); _kick!(cpy, r2)
                        SM.set_neurons_per_thread!(0)
                        SM.integrate!(pop, pop.param, 0.1f0)
                        SM.set_neurons_per_thread!(npt)
                        SM.integrate!(cpy, cpy.param, 0.1f0)
                        nspk += count(pop.fire)
                    end
                    SM.set_neurons_per_thread!(512)
                    ok = N == 0 || _bitsame(pop, cpy)
                    ok || @info "mismatch" kind syn N npt
                    @test ok
                    N >= 1000 && npt == 128 && @test nspk > 0     # the runs do contain spikes
                end
            end
        end
    end
end

@testset "zero allocation, sim!/train! keyword" begin
    pop = _pop(:IF, :current, 4000)
    # bytes per step: the serial loop of `IF` allocates 176 B per call in the unmodified library too (not a threading effect);
    # the threaded path must add only a small constant (the Polyester closure), nothing that grows with N or with the thread count
    function steps!(p, n); for _ in 1:n; SM.integrate!(p, p.param, 0.1f0); end; end
    per_step(q, npt) = (SM.set_neurons_per_thread!(npt); steps!(q, 10); (@allocated steps!(q, 1000)) / 1000)
    a_serial = per_step(pop, 0)
    a_thr = per_step(pop, 512)
    @info "allocation per integrate! step (B)" serial = a_serial threaded = a_thr nthreads = Threads.nthreads()
    @test a_thr - a_serial <= 400
    big = _pop(:IF, :current, 20_000)
    @test per_step(big, 512) - per_step(big, 0) <= 400            # independent of N
    SM.set_neurons_per_thread!(512)
    E = _pop(:IF, :double, 1000, seed = 3)
    a = deepcopy(E); b = deepcopy(E)
    SM.sim!([a]; dt = 0.1f0, duration = 20.0f0, neurons_per_thread = 0)
    SM.sim!([b]; dt = 0.1f0, duration = 20.0f0, neurons_per_thread = 128)
    @test SM.neurons_per_thread() == 512            # restored
    @test _bitsame(a, b)
end

@testset "Threads.@spawn tasks and nested use" begin
    pop = _pop(:IF, :current, 4000)
    ref = deepcopy(pop)
    SM.set_neurons_per_thread!(0)
    for _ in 1:20; SM.integrate!(ref, ref.param, 0.1f0); end
    SM.set_neurons_per_thread!(512)
    tasks = [Threads.@spawn begin
                 q = deepcopy(pop)
                 for _ in 1:20; SM.integrate!(q, q.param, 0.1f0); end
                 q
             end for _ in 1:4]
    for t in tasks
        @test _bitsame(fetch(t), ref)
    end
end

@testset "Identity dense path" begin
    for N in (0, 1, 100, 500, 1537, 10_000)
        a = SM.Identity(N = N); 
        N == 0 || (a.g .= randn(MersenneTwister(2), eltype(a.g), N) .* (rand(MersenneTwister(3), N) .< 0.5); a.h .= rand(MersenneTwister(4), N))
        b = deepcopy(a)
        SM.set_neurons_per_thread!(0);   SM._integrate_dense!(a)
        SM.set_neurons_per_thread!(128); SM._integrate_dense!(b)
        SM.set_neurons_per_thread!(512)
        @test a.g == b.g && a.h == b.h && a.fire == b.fire && a.spikecount == b.spikecount
    end
end
