# Constant delays (the delay is stored once for all synapses, `_dconst(param) >= 0`): `forward!` delivers through one ring-buffer
# column per step. It must be bit-identical to the general per-synapse path (the same delay stored per synapse, `_dconst == -1`): same
# `g`, buffer, ring position, weights, STP state and post membrane potential after every step.
using Test, Random, SNNModels
using SNNModels: SparseArrays
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
        (pname, ltp, stp) in (("fixed", _DU.NoLTP(), _DU.NoSTP()),
                              ("STDP", _DU.STDPGerstner(A_pre = 5e-3, A_post = -5e-3), _DU.NoSTP()),
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
