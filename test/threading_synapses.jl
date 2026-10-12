# Threaded Tripod / BallAndStick (own scratch per chunk) and threaded spiking synapses (forward!, delay ring, plasticity partitioned by
# postsynaptic range): bit-identical to the serial code for every neurons_per_thread and thread count.
# Run with several threads, e.g. `julia -t 4 --project=test test/threading_synapses.jl` (with one thread everything runs serially and
# the comparisons are trivial).
using Test, Random, SHA
using SNNModels
const SM = SNNModels
@load_units

I_in(i, r) = first(r) <= i <= last(r)

function _arrays_hash(objs)
    io = IOBuffer()
    for o in objs
        o isa AbstractArray{<:Real} && write(io, o)
    end
    return bytes2hex(sha1(take!(io)))
end
function _vars_arrays(c)
    xs = Any[]
    for v in (c.LTPVars, c.STPVars), f in fieldnames(typeof(v))
        f === :Wcur && continue          # scratch of the event-driven vSTDP
        x = getfield(v, f)
        x isa AbstractArray{<:Real} && push!(xs, x)
    end
    return xs
end

# population alone: state hash after 120 steps with random input in the receptor buffers
function _mc_run(kind, N)
    Random.seed!(11)
    p = kind === :tripod ? Tripod(N = N) : kind === :bas ? BallAndStick(N = N) :
        IF(N = N, param = IFParameter(C = 281pF, gl = 40nS), synapse = ReceptorSynapse())
    rng = MersenneTwister(5)
    nsp = 0
    for _ in 1:120
        for f in (:receptors_s, :receptors_d, :receptors_d1, :receptors_d2, :receptors)
            hasproperty(p, f) || continue
            for v in values(getproperty(p, f)), i in 1:N
                rand(rng) < 0.05 && (v[i] += 20f0 * rand(rng, Float32))
            end
        end
        for f in (:I, :Is)
            hasproperty(p, f) && (getproperty(p, f) .= 400f0 .* rand(rng, Float32, N) .+ 100f0)
        end
        SM.integrate!(p, p.param, 0.1f0)
        nsp += count(p.fire)
    end
    objs = Any[]
    for f in fieldnames(typeof(p))
        f === :ic && continue            # scratch, its length depends on the chunking
        x = getfield(p, f)
        x isa AbstractArray{<:Real} && push!(objs, x)
        x isa NamedTuple && append!(objs, [y for y in x if y isa AbstractArray{<:Real}])
        hasfield(typeof(x), :g) && hasfield(typeof(x), :h) && push!(objs, x.g, x.h)
    end
    return _arrays_hash(objs), nsp
end

# the threaded Heun stage keeps the axial currents in registers: the `ic` field is neither grown nor written. Run with
# neurons_per_thread = 128 on 1537 neurons: after the MAX_CHUNKS cap the split is 7 chunks, 6 x 256 + 1 (measured), so that with several
# threads the chunked path is the one checked.
function _mc_ic_untouched(kind)
    p = kind === :tripod ? Tripod(N = 1537) : BallAndStick(N = 1537)
    n = length(p.ic)
    hasproperty(p, :I) ? (p.I .= 600f0) : (p.Is .= 600f0)
    SM.with_neurons_per_thread(128) do
        for _ in 1:20
            SM.integrate!(p, p.param, 0.1f0)
        end
    end
    return length(p.ic) == n && all(iszero, p.ic)
end

@testset "threaded Tripod / BallAndStick / IF with receptors" begin
    old = SM.neurons_per_thread()
    try
        for kind in (:tripod, :bas, :if_rec), N in (0, 1, 7, 127, 129, 500, 1000, 1537, 2049, 4097)
            SM.set_neurons_per_thread!(0)
            ref = _mc_run(kind, N)
            for s in (128, 256, 512, 1024, 2048)
                SM.set_neurons_per_thread!(s)
                @test _mc_run(kind, N) == ref
            end
        end
        for kind in (:tripod, :bas)
            @test _mc_ic_untouched(kind)
        end
    finally
        SM.set_neurons_per_thread!(old)
    end
end

function _syn_rules(name)
    name == "fixed"      && return (SM.NoLTP(), SM.NoSTP(), false)
    name == "delay"      && return (SM.NoLTP(), SM.NoSTP(), true)
    name == "pair"       && return (SM.STDPGerstner(A_pre = 5e-3, A_post = -5e-3), SM.NoSTP(), false)
    name == "pair_delay" && return (SM.STDPGerstner(A_pre = 5e-3, A_post = -5e-3), SM.NoSTP(), true)
    name == "delay_const" && return (SM.NoLTP(), SM.NoSTP(), :const)          # constant delays: one ring column per step
    name == "delay_zero"  && return (SM.NoLTP(), SM.NoSTP(), :zero)
    name == "pair_delay_const" && return (SM.STDPGerstner(A_pre = 5e-3, A_post = -5e-3), SM.NoSTP(), :const)
    name == "stp_delay_const" && return (SM.NoLTP(), SM.MarkramSTPParameterEvent(), :const)   # column path with STP
    name == "triplet"    && return (SM.STDPTriplet(), SM.NoSTP(), false)
    name == "wdep"       && return (SM.STDPWeightDependent(η = 1e-2), SM.NoSTP(), false)
    name == "mexhat"     && return (SM.STDPMexicanHat(), SM.NoSTP(), false)
    name == "sym"        && return (SM.STDPSymmetric(), SM.NoSTP(), false)
    name == "antisym"    && return (SM.STDPAntiSymmetric(), SM.NoSTP(), false)
    name == "istdp"      && return (SM.iSTDPRate(), SM.NoSTP(), false)
    name == "vstdp"      && return (SM.vSTDPParameter(), SM.NoSTP(), false)
    name == "stp"        && return (SM.NoLTP(), SM.MarkramSTPParameterEvent(), false)
    name == "stp_pair_delay" && return (SM.STDPGerstner(A_pre = 5e-3, A_post = -5e-3), SM.MarkramSTPParameterEvent(), true)
    error(name)
end
_delay_kw(delay) = delay === :const ? (; delay_dist = SM.Dirac(0.8ms)) : delay === :zero ? (; delay_dist = SM.Dirac(0.0)) :
                   delay ? (; delay_dist = SM.Uniform(0.2ms, 3.0ms)) : (;)
function _syn_run(name, Npost; steps = 250, drive = 1.0)
    Random.seed!(3)
    ltp, stp, delay = _syn_rules(name)
    Npre = max(10, Npost ÷ 2 + 7)
    pre = SM.Poisson(N = Npre, param = PoissonParameter(40Hz))
    post = IF(N = Npost, param = IFParameter(τm = 20ms, R = 0.06, El = -60mV, Vt = -50mV, Vr = -60mV))
    kw = _delay_kw(delay)
    if name == "istdp"
        s1 = SpikingSynapse(pre, post, :ge; conn = (p = 0.15, μ = 6.0 * drive), kw...)
        s2 = SpikingSynapse(post, post, :gi; conn = (p = 0.1, μ = 3.0 * drive), LTPParam = ltp, kw...)
    else
        s1 = SpikingSynapse(pre, post, :ge; conn = (p = 0.15, μ = 6.0 * drive), LTPParam = ltp, STPParam = stp, kw...)
        s2 = SpikingSynapse(post, post, :ge; conn = (p = 0.1, μ = 2.0 * drive), LTPParam = ltp, STPParam = stp, kw...)
    end
    P = [pre, post]; C = [s1, s2]; T = Time(); nsp = 0
    for _ in 1:steps
        SM.train!(P, C, SM.AbstractStimulus[], 0.1f0, T)
        nsp += count(post.fire)
    end
    SM.sync_traces!(s1); SM.sync_traces!(s2)
    return _arrays_hash(Any[s1.W, s1.ρ, s2.W, s2.ρ, post.v, post.syn_curr, _vars_arrays(s1)..., _vars_arrays(s2)...]), nsp, s1.chunkptr.builds + s2.chunkptr.builds
end

@testset "threaded spiking synapses" begin
    old, oldops = SM.neurons_per_thread(), SM.SYN_MIN_OPS[]
    forked = 0
    try
        for name in ("fixed", "delay", "delay_const", "delay_zero", "pair", "pair_delay", "pair_delay_const", "stp_delay_const", "triplet", "wdep", "mexhat", "sym", "antisym", "istdp", "vstdp", "stp", "stp_pair_delay"),
            N in (1, 100, 700, 2100), drive in (1.0, 0.06)
            SM.set_neurons_per_thread!(0)
            ref = _syn_run(name, N; drive)[1:2]
            for s in (128, 512, 2048), mo in (0, 1000)
                SM.set_neurons_per_thread!(s); SM.SYN_MIN_OPS[] = mo
                h, nsp, nb = _syn_run(name, N; drive)
                @test (h, nsp) == ref
                forked += nb
            end
        end
    finally
        SM.set_neurons_per_thread!(old); SM.SYN_MIN_OPS[] = oldops
    end
    Threads.nthreads() > 1 && @test forked > 0      # the threaded paths were exercised
end

@testset "sub-row pointers" begin
    Random.seed!(4)
    pre = SM.Poisson(N = 300, param = PoissonParameter(10Hz)); post = IF(N = 1500)
    c = SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 1.0))
    for bpc in (1, 2, 4, 16)
        ptr = SM.chunk_ptr!(c.chunkptr, c.colptr, c.I, 1500, bpc)
        nc = SM.nchunks(1500, bpc)
        @test size(ptr) == (nc + 1, 300) && c.chunkptr.sorted
        for j in 1:300, ch in 1:nc
            r = SM.chunk_range(1500, bpc, ch)
            @test all(s -> I_in(c.I[s], r), ptr[ch, j]:(ptr[ch+1, j]-1))
        end
        for j in 1:300
            @test ptr[1, j] == c.colptr[j] && ptr[nc+1, j] == c.colptr[j+1]
        end
    end
    # rewiring invalidates the table
    SM.chunk_ptr!(c.chunkptr, c.colptr, c.I, 1500, 4)
    b = c.chunkptr.builds
    SM.chunk_ptr!(c.chunkptr, c.colptr, c.I, 1500, 4); @test c.chunkptr.builds == b      # unchanged: no rebuild
    SM.chunk_ptr!(c.chunkptr, c.colptr, c.I, 1500, 2); @test c.chunkptr.builds == b + 1  # other chunking: rebuilt
    SM.chunk_ptr!(c.chunkptr, c.colptr, c.I, 1500, 4); b = c.chunkptr.builds
    SM.update_sparse_matrix!(c, SM.sparse(rand(1:1500, 4000), rand(1:300, 4000), ones(Float32, 4000), 1500, 300))
    SM.chunk_ptr!(c.chunkptr, c.colptr, c.I, 1500, 4); @test c.chunkptr.builds == b + 1
end

# direct calls of forward! / plasticity! on hand-made spike patterns, without prepare_spikes! (the list is scanned from the flags on every call):
# empty, sparse, half of the neurons and all neurons spiking (the dense case), with and without delays
function _direct_run(density, delay, ltp, setting, minops)
    SM.set_neurons_per_thread!(setting); SM.SYN_MIN_OPS[] = minops
    Random.seed!(21)
    pre = IF(N = 700); post = IF(N = 1100)
    kw = _delay_kw(delay)
    c = SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 2.0), LTPParam = ltp, STPParam = SM.NoSTP(), kw...)
    rng = MersenneTwister(8)
    h = UInt8[]
    for step in 1:40
        pre.fire .= rand(rng, 700) .< density
        post.fire .= rand(rng, 1100) .< density
        SM.forward!(c, c.param, 0.1f0, SM.Time())
        any(c.LTPVars.active) && SM.plasticity!(c, c.param, 0.1f0, SM.Time())
        append!(h, reinterpret(UInt8, copy(post.receptors.glu)))
        post.receptors.glu .= 0
    end
    SM.sync_traces!(c)
    return bytes2hex(sha1(vcat(h, reinterpret(UInt8, c.W), reinterpret(UInt8, c.ρ))))
end
@testset "direct forward!/plasticity! with sparse and dense spike patterns" begin
    old, oldops = SM.neurons_per_thread(), SM.SYN_MIN_OPS[]
    try
        for density in (0.0, 0.01, 0.5, 1.0), delay in (false, true, :const, :zero), ltp in (SM.NoLTP(), SM.STDPGerstner(A_pre = 5e-3, A_post = -5e-3), SM.STDPTriplet())
            ref = _direct_run(density, delay, ltp, 0, 8000)
            for s in (128, 512), mo in (0, 8000)
                @test _direct_run(density, delay, ltp, s, mo) == ref
            end
        end
    finally
        SM.set_neurons_per_thread!(old); SM.SYN_MIN_OPS[] = oldops
    end
end
