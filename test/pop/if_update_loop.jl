# Bit-identity of the branchless `update_neuron!` of `IF` against the original branching loop, kept here as the reference.
using Random

# reference: the loop of SNNModels <= 1.9 (refractory `continue` branch, `round(Int, τabs/dt)` evaluated at every spike)
function _ref_update_neuron!(p::IF, param, dt)
    (; N, v, w, I, tabs, fire, syn_curr) = p
    (; τm, El, R, Vt, Vr) = param
    τabs = p.spike.τabs
    for i = 1:N
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end
        v[i] += dt / τm * (-(v[i] - El) + R * (-w[i] + I[i]) - R * syn_curr[i])
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
    if (hasfield(typeof(param), :τw) && param.τw > 0)
        (; a, b, τw) = param
        for i = 1:N
            w[i] = ifelse(fire[i], w[i] + param.b, w[i])
            (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
        end
    end
end

function _ref_integrate!(p::IF, param, dt)
    SNNModels.update_synapses!(p, p.synapse, p.receptors, p.synvars, dt)
    SNNModels.synaptic_current!(p, p.synapse, p.synvars)
    _ref_update_neuron!(p, param, dt)
end

_same(x::AbstractVector, y::AbstractVector) = isequal(x, y) && all(i -> isequal(x[i], y[i]) && (signbit(x[i]) == signbit(y[i]) || x[i] isa Bool || x[i] isa Integer), eachindex(x))

function _state_equal(p, q)
    all(f -> _same(getfield(p, f), getfield(q, f)), (:v, :w, :fire, :tabs, :I, :syn_curr)) &&
    all(f -> _same(getfield(p.synvars, f), getfield(q.synvars, f)), (:ge, :gi))
end

function _run_pair(::Type{FT}, N, synapse; adapt, τabs, nsteps = 400, seed = 1) where {FT}
    Random.seed!(seed)
    param = adapt ? IFParameter(FT = FT, a = 4, b = 80, τw = 144) : IFParameter(FT = FT)
    p = IF(; N, param, synapse, spike = PostSpike(FT = FT, τabs = τabs), VFT = Vector{FT})
    p.tabs .= rand(0:4, N)                       # mixed refractory state from the first step
    p.w .= FT.(10 .* rand(N))
    q = deepcopy(p)
    dt = FT(0.1)
    nspk = 0; nref = 0
    for n in 1:nsteps
        Ivec = FT.(1000 .* rand(N))               # strong drive: many spikes
        p.I .= Ivec; q.I .= Ivec
        for r in keys(p.receptors)
            x = FT.(30 .* rand(N) .* (rand(N) .< 0.1))
            getfield(p.receptors, r) .= x; getfield(q.receptors, r) .= x
        end
        SNNModels.integrate!(p, p.param, dt)
        _ref_integrate!(q, q.param, dt)
        _state_equal(p, q) || return (false, n, nspk, nref)
        nspk += count(p.fire); nref += count(>(0), p.tabs)
    end
    return (true, nsteps, nspk, nref)
end

@testset "IF update_neuron! branchless loop is bit-identical to the reference" begin
    for FT in (Float32, Float64), mk in (() -> DoubleExpSynapse(FT = FT), () -> CurrentSynapse())
        FT == Float64 && mk() isa CurrentSynapse && continue       # Float64 populations: DoubleExpSynapse
        for N in (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1001), adapt in (false, true)
            ok, n, nspk, nref = _run_pair(FT, N, mk(); adapt, τabs = 2)
            @test ok
            if N >= 100                                  # the cases really exercised spikes and refractoriness
                @test nspk > 0
                @test nref > 0
            end
        end
    end
    # a population whose neurons never fire, and one with τabs = 0 (no refractory period)
    @test _run_pair(Float32, 64, DoubleExpSynapse(); adapt = true, τabs = 0)[1]
    @test _run_pair(Float32, 64, DoubleExpSynapse(); adapt = false, τabs = 0)[1]
    # refractory neurons keep v, spikes reset v and set tabs, tabs is decremented
    let p = IF(N = 4, param = IFParameter(), spike = PostSpike(τabs = 1))
        p.v .= [-55, -55, -49.9999, 0]; p.tabs .= [3, 0, 0, 0]; p.I .= 0; p.syn_curr .= 0
        v0 = copy(p.v)
        SNNModels.update_neuron!(p, p.param, 0.1f0)
        @test p.v[1] == v0[1] && p.tabs[1] == 2 && !p.fire[1]       # refractory: unchanged, decremented
        @test !p.fire[2] && p.tabs[2] == 0
        @test p.fire[4] && p.v[4] == p.param.Vr && p.tabs[4] == 10   # round(Int, 1 / 0.1) steps
    end
    # a state vector shorter than N is a clear error, not an out-of-bounds write
    let p = IF(N = 8)
        resize!(p.syn_curr, 5)
        @test_throws DimensionMismatch SNNModels.update_neuron!(p, p.param, 0.1f0)
    end
    let p = IF(N = 8)
        resize!(p.tabs, 3)
        @test_throws DimensionMismatch SNNModels.update_neuron!(p, p.param, 0.1f0)
    end
    # the allocation of a step does not grow with N (the loop itself allocates nothing)
    let p = IF(N = 10), q = IF(N = 10000)
        SNNModels.update_neuron!(p, p.param, 0.1f0); SNNModels.update_neuron!(q, q.param, 0.1f0)
        @test (@allocated SNNModels.update_neuron!(q, q.param, 0.1f0)) == (@allocated SNNModels.update_neuron!(p, p.param, 0.1f0))
    end
end

true
