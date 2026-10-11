# Constant delays (the delay is stored once for all synapses, `_dconst(param) >= 0`).
# - With plasticity (LTP or STP): `forward!` delivers through one ring-buffer column per step. It must be bit-identical to the general
#   per-synapse path (the same delay stored per synapse, `_dconst == -1`): same `g`, buffer, ring position, weights, STP state and post
#   membrane potential after every step.
# - Without plasticity (NoLTP, NoSTP): the spike list is delayed instead (ring of the last d + 1 lists, scattered into `g` on arrival,
#   no buffer). Same arrival step; the increments of one step are added to `g` one by one (rounding only) and `W` is read at arrival.
using Test, Random, SNNModels
using SNNModels: SparseArrays
@load_units
const _DU = SNNModels
const _DUD = SNNModels.Distributions

# one model with two identical branches driven by the same Poisson population: branch A keeps the constant-delay path, branch B is
# forced onto the general path (`_du_general!`: the common delay stored once per synapse)
function _du_branch(pre, FT, delay, ltp, stp, name)
    post = FT === Float64 ? IF(N = 40, VFT = Vector{Float64}, name = name) :
           IF(N = 40, param = IFParameter(τm = 20ms, R = 0.06, El = -60mV, Vt = -50mV, Vr = -60mV), name = name)
    kw = (; delay_dist = delay, LTPParam = ltp, STPParam = stp)
    sin = SpikingSynapse(pre, post, :ge; conn = (p = 0.3, μ = FT === Float64 ? 4.0 : 8.0), kw...)
    srec = SpikingSynapse(post, post, :ge; conn = (p = 0.2, μ = FT === Float64 ? 0.5 : 2.0), kw...)
    return post, sin, srec
end

_du_same(a, b) = isequal(a, b) && typeof(a) === typeof(b)
# general path: the delay stored once is expanded to one (equal) value per synapse; `_delay_steps!` recomputes the steps
function _du_general!(c)
    p = c.param
    length(p.delaytime) == 1 && length(c.W) > 1 && (p.delaytime = fill(p.delaytime[1], length(c.W)))
    return c
end
function _du_state(post, s)
    return (s.g, s.param.buffer, s.param.pos, s.W, s.ρ, post.v, post.fire)
end

function _du_compare(FT, delay, dt, ltp, stp; steps = 2000, dt2 = nothing)
    Random.seed!(17)
    pre = _DU.Poisson(N = 60, param = PoissonParameter(60Hz))
    Random.seed!(23); postA, inA, recA = _du_branch(pre, FT, delay, ltp, stp, "A")
    Random.seed!(23); postB, inB, recB = _du_branch(pre, FT, delay, ltp, stp, "B")
    _du_general!(inB); _du_general!(recB)
    @test inA.W == inB.W && recA.W == recB.W
    P = [pre, postA, postB]; C = [inA, recA, inB, recB]; T = Time{FT}()
    nsp = 0; ok = true
    for n in 1:steps
        h = (dt2 !== nothing && n > steps ÷ 2) ? FT(dt2) : FT(dt)
        _DU.train!(P, C, _DU.AbstractStimulus[], h, T)
        for (a, b) in ((inA, inB), (recA, recB))
            for (x, y) in zip(_du_state(postA, a), _du_state(postB, b))
                _du_same(x, y) || (ok = false)
            end
        end
        nsp += count(postA.fire)
        ok || (@info "first difference" FT delay dt n; break)
    end
    @test ok
    @test _DU._dconst(inA.param) >= 0 && _DU._dconst(recA.param) >= 0          # A: constant-delay path
    @test _DU._dconst(inB.param) == -1 && _DU._dconst(recB.param) == -1        # B: general path, also after a change of dt
    return nsp, inA, recA
end

@testset "constant delay: _dconst" begin
    pre, post = Identity(N = 3), Identity(N = 4)
    mk(d) = SpikingSynapse(pre, post, :g; conn = fill(1.0f0, 4, 3), delay_dist = d)
    for (d, dt, k) in ((_DUD.Dirac(0.0), 0.1f0, 0), (_DUD.Dirac(0.1), 0.1f0, 1), (_DUD.Dirac(0.8), 0.1f0, 8),
                       (_DUD.Dirac(0.8), 0.125f0, 7), (_DUD.Dirac(0.75), 0.125f0, 6), (_DUD.Dirac(0.05), 0.1f0, 1))
        c = mk(d)
        @test _DU._dconst(c.param) == -1                   # not computed before the first forward!
        _DU.forward!(c, c.param, dt, Time())
        @test _DU._dconst(c.param) == k
        @test all(==(k), c.param.delaysteps) && size(c.param.buffer, 2) == k + 1
    end
    c = mk(_DUD.Uniform(0.2, 3.0))
    _DU.forward!(c, c.param, 0.1f0, Time())
    @test _DU._dconst(c.param) == -1
    # empty connection: general path, nothing delivered
    c = SpikingSynapse(pre, post, :g; conn = SparseArrays.spzeros(Float32, 4, 3), delay_dist = _DUD.Dirac(0.8))
    pre.fire .= true
    _DU.forward!(c, c.param, 0.1f0, Time())
    @test _DU._dconst(c.param) == -1 && isempty(c.param.delaysteps) && size(c.param.buffer) == (4, 1)
    @test all(iszero, c.g)
    pre.fire .= false
    # a change of dt recomputes the common delay in steps (8 steps at 0.1 ms -> 7 steps at 0.125 ms)
    c = mk(_DUD.Dirac(0.8))
    _DU.forward!(c, c.param, 0.1f0, Time()); @test _DU._dconst(c.param) == 8
    _DU.forward!(c, c.param, 0.125f0, Time()); @test _DU._dconst(c.param) == 7
    # rewiring (sparse matrix rebuilt) marks the delays stale; a heterogeneous rewired delay vector falls back to the general path
    c = mk(_DUD.Dirac(0.8))
    _DU.forward!(c, c.param, 0.1f0, Time()); @test _DU._dconst(c.param) == 8
    _DU.update_sparse_matrix!(c, SparseArrays.sparse([1, 2, 3], [1, 2, 3], ones(Float32, 3), 4, 3))
    @test length(c.param.delaytime) == 1                 # stored once again
    c.param.delaytime = Float32[0.3, 0.8, 0.8]
    _DU.forward!(c, c.param, 0.1f0, Time()); @test _DU._dconst(c.param) == -1 && c.param.delaysteps[1] == 3
end

@testset "constant delay: arrival step" begin
    # 0 steps: the increment arrives in the same forward! call; 1 step: in the next one; 8 steps at 0.8 ms
    for (d, k) in ((0.0, 0), (0.1, 1), (0.8, 8))
        pre, post = Identity(N = 2), Identity(N = 3)
        c = SpikingSynapse(pre, post, :g; conn = Float32[1 0; 0 2; 3 0], delay_dist = _DUD.Dirac(d))
        T = Time(); arrivals = Int[]
        for n in 1:20
            pre.fire .= (n == 3, n == 3)
            update_time!(T, 0.1f0)
            _DU.forward!(c, c.param, 0.1f0, T)
            if any(!iszero, c.g)
                push!(arrivals, n)
                @test c.g == Float32[1, 2, 3]       # each target once, W of its synapse
                c.g .= 0
            end
        end
        @test _DU._dconst(c.param) == k
        @test arrivals == [3 + k]
    end
end

@testset "constant delay bit-identical to the general path: $FT, $dname, dt $dt, $pname" for FT in (Float32, Float64),
        (dname, delay) in (("0", _DUD.Dirac(0.0)), ("dt", nothing), ("0.8", _DUD.Dirac(0.8))),
        dt in (0.1, 0.125),
        (pname, ltp, stp) in (("STDP", _DU.STDPGerstner(A_pre = 5e-3, A_post = -5e-3), _DU.NoSTP()),
                              ("STP", _DU.NoLTP(), _DU.MarkramSTPParameterEvent()))
    d = delay === nothing ? _DUD.Dirac(dt) : delay
    nsp, sA, _ = _du_compare(FT, d, dt, ltp, stp)
    @test nsp > 0                                     # the post neurons fire: the recurrent delayed path is exercised
    @test eltype(sA.param.buffer) == FT
    pname == "STDP" && @test any(!=(sA.W[1]), sA.W)   # plasticity acted
end

@testset "constant delay bit-identical across a change of dt, $FT" for FT in (Float32, Float64)
    nsp, sA, _ = _du_compare(FT, _DUD.Dirac(0.8), 0.1, _DU.STDPGerstner(A_pre = 5e-3, A_post = -5e-3), _DU.NoSTP(); dt2 = 0.125)
    @test nsp > 0
    @test _DU._dconst(sA.param) == 7
end

# ---- constant delay without plasticity: delayed spike list ----
@testset "spike-list delay: selection" begin
    pre, post = Identity(N = 3), Identity(N = 4)
    mk(d; kw...) = SpikingSynapse(pre, post, :g; conn = fill(1.0f0, 4, 3), delay_dist = d, kw...)
    for (c, used) in ((mk(_DUD.Dirac(0.8)), true), (mk(_DUD.Dirac(0.0)), true), (mk(_DUD.Uniform(0.2, 3.0)), false),
                      (mk(_DUD.Dirac(0.8); LTPParam = _DU.STDPGerstner()), false),
                      (mk(_DUD.Dirac(0.8); STPParam = _DU.MarkramSTPParameterEvent()), false))
        _DU.forward!(c, c.param, 0.1f0, Time())
        @test _DU._use_spike_ring(c) == used
        @test length(c.param.spike_ring) == (used ? _DU._dconst(c.param) + 1 : 0)
    end
    @test mk(_DUD.Dirac(0.8)).param.spike_ring !== mk(_DUD.Dirac(0.8)).param.spike_ring   # not a shared default
end

# feed-forward replay: the same spike trains through the spike-list path and the general path; g after every step
function _du_ff(FT, d, dt; steps = 1500, rate = 0.08, wchange = false)
    Random.seed!(5)
    pre = Identity(N = 80)
    postA = Identity(N = 50, VFT = Vector{FT}); postB = Identity(N = 50, VFT = Vector{FT})
    conn = SparseArrays.sprand(FT, 50, 80, 0.3)
    a = SpikingSynapse(pre, postA, :g; conn = copy(conn), delay_dist = _DUD.Dirac(d))
    b = SpikingSynapse(pre, postB, :g; conn = copy(conn), delay_dist = _DUD.Dirac(d))
    _du_general!(b)                                                 # b: general path from the first step
    rng = MersenneTwister(9); T = Time{FT}()
    maxrel = 0.0; same_step = true; nmulti = 0; exact_single = true
    for n in 1:steps
        pre.fire .= rand(rng, 80) .< rate
        update_time!(T, FT(dt))
        _DU.forward!(a, a.param, FT(dt), T)
        _DU.forward!(b, b.param, FT(dt), T)
        same_step &= (a.g .!= 0) == (b.g .!= 0)
        for i in eachindex(a.g)
            Δ = abs(Float64(a.g[i]) - Float64(b.g[i]))
            maxrel = max(maxrel, Δ / max(abs(Float64(b.g[i])), eps(FT)))
        end
        a.g .= 0; b.g .= 0
        if wchange && n == steps ÷ 2
            a.W .*= 2; b.W .*= 2
        end
    end
    return a, b, maxrel, same_step
end

@testset "spike-list delay against the general path, $FT, delay $d, dt $dt" for FT in (Float32, Float64), d in (0.0, 0.1, 0.8), dt in (0.1, 0.125)
    a, b, maxrel, same_step = _du_ff(FT, d, dt)
    @test _DU._use_spike_ring(a) && !_DU._use_spike_ring(b)
    @test same_step                                   # every increment arrives in the same step as on the general path
    @test maxrel <= 8 * eps(FT)                       # only the rounding of the per-step sum differs
end

@testset "spike-list delay: one increment per target is exact; W read at arrival" begin
    pre, post = Identity(N = 2), Identity(N = 3)
    c = SpikingSynapse(pre, post, :g; conn = Float32[1 0; 0 2; 3 0], delay_dist = _DUD.Dirac(0.5))
    T = Time(); got = Vector{Float32}[]
    for n in 1:12
        pre.fire .= (n == 2, n == 2)
        n == 4 && (c.W .*= 10)                       # weight changed between emission (step 2) and arrival (step 7)
        update_time!(T, 0.1f0)
        _DU.forward!(c, c.param, 0.1f0, T)
        any(!iszero, c.g) && (push!(got, copy(c.g)); c.g .= 0)
    end
    @test got == [Float32[10, 20, 30]]
    @test _DU._dconst(c.param) == 5
end

_du_alloc(c, T) = @allocated _DU.forward!(c, c.param, 0.1f0, T)
@testset "spike-list delay: no allocation after warm-up; dt change resizes the ring" begin
    Random.seed!(3)
    pre, post = Identity(N = 200), Identity(N = 100)
    c = SpikingSynapse(pre, post, :g; conn = (p = 0.2, μ = 1.0), delay_dist = _DUD.Dirac(0.8))
    T = Time()
    step!() = (pre.fire .= rand(200) .< 0.5; _DU.forward!(c, c.param, 0.1f0, T))
    for _ in 1:30; step!(); end
    pre.fire .= rand(200) .< 0.5
    @test _du_alloc(c, T) == 0
    @test length(c.param.spike_ring) == 9
    _DU.forward!(c, c.param, 0.125f0, T)
    @test _DU._dconst(c.param) == 7 && length(c.param.spike_ring) == 8
    @test (@inferred _DU.forward!(c, c.param, 0.125f0, T)) isa Int
end

# ---- switching between the spike-list path and the buffer path; dt changes and rewiring with spikes in flight ----
# one spike of both presynaptic neurons at `fire_at`; returns the steps at which `g` was nonzero. `events[n]` is called before step n.
function _du_arrivals(c, dts, fire_at; events = Dict{Int,Function}(), steps = length(dts), vals = nothing)
    pre = c.fireJ; T = Time(); arr = Int[]
    for n in 1:steps
        haskey(events, n) && (c = events[n](c))
        pre .= (n in fire_at)
        update_time!(T, dts[n])
        _DU.forward!(c, c.param, dts[n], T)
        any(!iszero, c.g) && (push!(arr, n); isnothing(vals) || push!(vals, copy(c.g)); c.g .= 0)
    end
    return arr, c
end
_du_mk(d; kw...) = SpikingSynapse(Identity(N = 2), Identity(N = 3), :g; conn = Float32[1 0; 0 2; 3 0], delay_dist = _DUD.Dirac(d), kw...)
const _DU_STDP = _DU.STDPGerstner(A_pre = 0.0, A_post = 0.0)   # a rule that leaves W unchanged: only the path changes

@testset "spike-list delay: switching the rule carries the spikes in flight over, each arrives once with its weight" begin
    sw(rule) = c -> _DU.with_plasticity(c; LTP = rule)
    dts = fill(0.1f0, 20)                                     # Dirac(0.5): 5 steps
    Wsum = Float32[1, 2, 3]                                   # both presynaptic neurons fire: one W per target
    # NoLTP -> STDP -> NoLTP, spike in the ring at the first switch (step 2), in the buffer at the second (step 4)
    vals = Vector{Float32}[]
    arr, c = _du_arrivals(_du_mk(0.5), dts, (2, 4); events = Dict(3 => sw(_DU_STDP), 5 => sw(_DU.NoLTP())), vals)
    @test arr == [7, 9] && all(==(Wsum), vals)
    @test _DU._use_spike_ring(c)
    # the buffer path was drained d steps after the second switch and released
    @test size(c.param.buffer, 1) == 0
    # STDP -> NoLTP -> STDP, spike in the buffer at the first switch (step 2), in the ring at the second (step 4)
    vals = Vector{Float32}[]
    arr, c = _du_arrivals(_du_mk(0.5; LTPParam = _DU_STDP), dts, (2, 4); events = Dict(3 => sw(_DU.NoLTP()), 5 => sw(_DU_STDP)), vals)
    @test arr == [7, 9] && all(==(Wsum), vals)
    @test !_DU._use_spike_ring(c) && isempty(c.param.spike_ring) && size(c.param.buffer) == (3, 6)
    # a switch on the very step of the emission, on the arrival step, and with a train of spikes
    for (fire_at, sws) in (((2,), (2, 7)), ((2, 3, 5, 6), (4, 7, 9)), ((1, 2, 3, 4, 5, 6, 7, 8), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))
        # alternate the rule at every step of `sws`, starting with `first`
        ev(first, other) = Dict(n => sw(isodd(k) ? first : other) for (k, n) in enumerate(sws))
        ref, _ = _du_arrivals(_du_mk(0.5), dts, fire_at)
        r1, _ = _du_arrivals(_du_mk(0.5), dts, fire_at; events = ev(_DU_STDP, _DU.NoLTP()))
        r2, _ = _du_arrivals(_du_mk(0.5; LTPParam = _DU_STDP), dts, fire_at; events = ev(_DU.NoLTP(), _DU_STDP))
        @test r1 == ref && r2 == ref
        @test ref == sort(unique(fire_at .+ 5))
    end
end

@testset "spike-list delay: dt change with spikes in flight, as the general path" begin
    for (d, dts, fire_at, expected) in (
            (0.5, [fill(0.1f0, 3); fill(0.11f0, 17)], (2, 10), [7, 15]),   # same delay in steps (5): in-flight spike delivered
            (0.8, [fill(0.1f0, 3); fill(0.125f0, 17)], (2, 10), [17]))     # 8 -> 7 steps: ring and buffer rebuilt, in-flight spike dropped
        a, _ = _du_arrivals(_du_mk(d), dts, fire_at)
        # general path: the delay stored per synapse from the first step
        b, _ = _du_arrivals(_du_general!(_du_mk(d)), dts, fire_at)
        @test a == b == expected
    end
end

@testset "spike-list delay: rewiring keeps the spikes in flight (delivered through the new connectivity)" begin
    c = _du_mk(0.5)
    ev = Dict(4 => (c -> (_DU.update_sparse_matrix!(c, SparseArrays.sparse([2, 3], [1, 2], Float32[5, 7], 3, 2)); c)))
    pre = c.fireJ; T = Time(); got = Pair{Int,Vector{Float32}}[]
    for n in 1:12
        haskey(ev, n) && ev[n](c)
        pre .= (n == 2)
        update_time!(T, 0.1f0)
        _DU.forward!(c, c.param, 0.1f0, T)
        any(!iszero, c.g) && (push!(got, n => copy(c.g)); c.g .= 0)
    end
    @test got == [7 => Float32[0, 5, 7]]
end

# recurrent network: spike-list path (A) against the general path (B) from the same state; equal up to the rounding of the per-step sums
@testset "spike-list delay against the general path, recurrent network, $FT" for FT in (Float32, Float64)
    Random.seed!(17)
    pre = _DU.Poisson(N = 60, param = PoissonParameter(60Hz))
    Random.seed!(23); postA, inA, recA = _du_branch(pre, FT, _DUD.Dirac(0.8), _DU.NoLTP(), _DU.NoSTP(), "A")
    Random.seed!(23); postB, inB, recB = _du_branch(pre, FT, _DUD.Dirac(0.8), _DU.NoLTP(), _DU.NoSTP(), "B")
    for c in (inB, recB)
        _du_general!(c)
    end
    P = [pre, postA, postB]; C = [inA, recA, inB, recB]; T = Time{FT}()
    maxdv = 0.0; samefire = true; nsp = 0
    for n in 1:400
        _DU.train!(P, C, _DU.AbstractStimulus[], FT(0.1), T)
        maxdv = max(maxdv, maximum(abs.(Float64.(postA.v) .- Float64.(postB.v))))
        samefire &= postA.fire == postB.fire
        nsp += count(postA.fire)
    end
    @test _DU._use_spike_ring(inA) && _DU._use_spike_ring(recA) && !_DU._use_spike_ring(inB) && !_DU._use_spike_ring(recB)
    @test nsp > 0 && samefire
    @test maxdv <= 1e-3
end
