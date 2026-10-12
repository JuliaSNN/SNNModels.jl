using SNNModels
using Test
using Random
@load_units

# Lazily stored plasticity state. The event-driven STDP rules (pair, Confavreux2025, weight-dependent,
# triplet) store a trace as `stored * scale` (one scalar per trace type, decayed once per step), and the
# event-driven Markram STP rules update `u`, `x` only at presynaptic spikes. Whatever reads this state
# (`monitor!`, `trace_values`, `plasticity_value`, `sync_traces!`) must see the true value, compared here
# with an independent dense computation (every neuron decayed at every step), and recording must not
# change the dynamics.

const _K = SNNModels
const _DT_LZ = 0.125f0

# ---- independent dense references (Float64, every neuron updated at every step) ----

# traces after each step: x <- (x + spike) * exp(-dt/τ)
function _dense_trace(spikes::AbstractMatrix{Bool}, τ, dt)
    N, nsteps = size(spikes)
    f = exp(-Float64(dt) / Float64(τ))
    x = zeros(N)
    out = zeros(N, nsteps)
    for n in 1:nsteps
        x .= (x .+ spikes[:, n]) .* f
        out[:, n] = x
    end
    return out
end

# event-driven Markram STP, advanced densely: relax over every step, then the spike jumps
function _dense_stp(spikes::AbstractMatrix{Bool}, U, τF, τD, dt)
    N, nsteps = size(spikes)
    fF, fD = exp(-Float64(dt) / Float64(τF)), exp(-Float64(dt) / Float64(τD))
    u, x = fill(Float64(U), N), ones(N)
    uo, xo = zeros(N, nsteps), zeros(N, nsteps)
    for n in 1:nsteps
        u .= U .- (U .- u) .* fF
        x .= 1 .- (1 .- x) .* fD
        for j in 1:N
            spikes[j, n] || continue
            u[j] += U * (1 - u[j])
            x[j] -= u[j] * x[j]
        end
        uo[:, n] = u; xo[:, n] = x
    end
    return uo, xo
end

# clock-driven Markram STP: jump at the spike (update_traces!), Euler relaxation after forward! (plasticity!)
function _dense_stp_timestep(spikes::AbstractMatrix{Bool}, U, τF, τD, dt)
    N, nsteps = size(spikes)
    u, x = fill(Float64(U), N), ones(N)
    uo, xo = zeros(N, nsteps), zeros(N, nsteps)
    for n in 1:nsteps
        for j in 1:N
            spikes[j, n] || continue
            u[j] += U * (1 - u[j])
            x[j] -= u[j] * x[j]
        end
        u .+= dt .* (U .- u) ./ τF
        x .+= dt .* (1 .- x) ./ τD
        uo[:, n] = u; xo[:, n] = x
    end
    return uo, xo
end

# ---- a driven pre (Poisson) -> post (IF) connection, spikes of both populations logged ----

const _NPRE_LZ, _NPOST_LZ = 12, 6

# Returns (syn, pre, post, pf, qf): pf[:, n], qf[:, n] are the spikes of step n. `monitor` is a function
# called on the synapse before the run (nothing: unmonitored).
function _lz_run(; ltp = NoLTP(), stp = NoSTP(), nsteps = 600, seed = 4, monitor = nothing, μ = 40.0)
    Random.seed!(seed)
    pre = SNNModels.Poisson(N = _NPRE_LZ, param = SNNModels.PoissonParameter(80Hz))
    post = SNNModels.IF(N = _NPOST_LZ)
    Random.seed!(seed + 1)
    syn = SpikingSynapse(pre, post, :ge; conn = (p = 0.6, μ = μ), LTPParam = ltp, STPParam = stp)
    isnothing(monitor) || monitor(syn)
    pf, qf = falses(_NPRE_LZ, nsteps), falses(_NPOST_LZ, nsteps)
    k = Ref(0)
    # called before each step: the flags are those of the previous step
    spy(; kw...) = (k[] > 0 && (pf[:, k[]] .= pre.fire; qf[:, k[]] .= post.fire); k[] += 1)
    Random.seed!(seed + 2)
    train!([pre, post], [syn], SNNModels.AbstractStimulus[]; duration = nsteps * _DT_LZ, dt = _DT_LZ, perturbation! = spy)
    @assert k[] == nsteps
    pf[:, nsteps] .= pre.fire; qf[:, nsteps] .= post.fire
    return syn, pre, post, pf, qf
end

# recorded samples of a plasticity variable: drop the sample at t = 0, one column per step
_samples(syn, group, key) = Float64.(SNNModels.getvariable(syn, Symbol(group, "_", key)))[:, 2:end]

_rules = (
    "STDPGerstner" => (() -> STDPGerstner(A_pre = 0.05f0, A_post = -0.04f0, Wmax = 80.0f0, Wmin = 0.0f0),
                       ((:tpre, :pre, 20.0), (:tpost, :post, 20.0))),
    "STDPConfavreux2025" => (() -> STDPConfavreux2025(η = 0.01f0, κ = -1.0f0, γ = 1.0f0, α = 0.001f0, τpre = 17.0f0, τpost = 31.0f0, Wmax = 80.0f0, Wmin = 0.0f0),
                       ((:tpre, :pre, 17.0), (:tpost, :post, 31.0))),
    "STDPWeightDependent" => (() -> STDPWeightDependent(η = 0.02f0, α = 1.05f0, μ_plus = 0.5f0, μ_minus = 1.0f0, τpre = 23.0f0, τpost = 12.0f0, Wmax = 80.0f0, Wmin = 0.0f0),
                       ((:tpre, :pre, 23.0), (:tpost, :post, 12.0))),
    "STDPTriplet" => (() -> STDPTriplet(A2_plus = 0.01f0, A3_plus = 0.02f0, A2_minus = 0.01f0, A3_minus = 0.005f0, τ_x = 200.0f0, Wmax = 80.0f0, Wmin = 0.0f0),
                       ((:r1, :pre, 16.8), (:r2, :pre, 200.0), (:o1, :post, 33.7), (:o2, :post, 40.0))),
)

@testset "Lazily stored plasticity state" begin

    @testset "there is no opt-in any more" begin
        for (name, (mk, _)) in _rules
            @test !hasfield(typeof(mk()), :lazy_decay)
            @test !hasfield(typeof(plasticityvariables(mk(), 5, 4)), :lazy)
        end
        @test_throws MethodError STDPGerstner(lazy_decay = true)
    end

    for (name, (mk, traces)) in _rules
        @testset "$name: monitor! records the trace values" begin
            keys = first.(traces)
            syn, pre, post, pf, qf = _lz_run(ltp = mk(), monitor = s -> monitor!(s, collect(keys), :LTPVars, sr = 10kHz))
            @test sum(pf) > 50 && sum(qf) > 20                         # both sides spiked
            @test any(!=(1.0), syn.LTPVars.scale)                      # the state is stored scaled
            for (key, side, τ) in traces
                ref = _dense_trace(side === :pre ? pf : qf, τ, _DT_LZ)
                rec = _samples(syn, :LTPVars, key)
                @test size(rec) == size(ref)
                @test maximum(rec) > 0.1
                @test rec ≈ ref rtol = 1.0e-5 atol = 1.0e-12
                # the stored values are not the values (the recorder did not record them)
                stored = Float64.(getfield(syn.LTPVars, key))
                @test !(stored ≈ ref[:, end])
            end
            @test syn.W != _lz_run(ltp = mk(), nsteps = 1)[1].W        # plasticity acted
        end
    end

    @testset "STP: monitor! records the values of u and x" begin
        U, τF, τD = 0.2f0, 300.0f0, 150.0f0
        mk() = MarkramSTPParameter(U = U, τF = τF, τD = τD)
        syn, pre, post, pf, qf = _lz_run(stp = mk(), monitor = s -> monitor!(s, [:u, :x, :_ρ, :last_spike], :STPVars, sr = 10kHz))
        uref, xref = _dense_stp(pf, U, τF, τD, _DT_LZ)
        u, x = _samples(syn, :STPVars, :u), _samples(syn, :STPVars, :x)
        @test u ≈ uref rtol = 1.0e-5
        @test x ≈ xref rtol = 1.0e-5
        @test maximum(u) > 1.5 * U && minimum(x) < 0.9                    # facilitation and depression happened
        # the stored values are the values only at a spike: at the end the quiet neurons differ
        @test !(Float64.(syn.STPVars.u) ≈ uref[:, end])
        # the accessor and the state
        T = Time(); T.t[1] = Float32(600 * _DT_LZ)
        @test Float64.(plasticity_value(syn, :STPVars, :u, T)) ≈ uref[:, end] rtol = 1.0e-5
        @test Float64.(plasticity_value(syn, :STPVars, :x, T)) ≈ xref[:, end] rtol = 1.0e-5
        @test plasticity_value(syn, :STPVars, :_ρ, T) === syn.STPVars._ρ
        @test plasticity_value(syn, :STPVars, :last_spike, T) === syn.STPVars.last_spike
        # a synapse that has not spiked yet: u = U, x = 1 at any time
        v = plasticityvariables(mk(), 4, 3)
        @test SNNModels.current_value(v, mk(), :u, 100.0f0) == fill(U, 4)
        @test SNNModels.current_value(v, mk(), :x, 100.0f0) == ones(Float32, 4)
        # heterogeneous parameters
        het = MarkramSTPParameterHet(τD = fill(150.0f0, _NPRE_LZ), τF = fill(300.0f0, _NPRE_LZ), U = fill(U, _NPRE_LZ))
        synh, _, _, pfh, _ = _lz_run(stp = het, monitor = s -> monitor!(s, [:u, :x], :STPVars, sr = 10kHz))
        uh, xh = _dense_stp(pfh, U, τF, τD, _DT_LZ)
        @test _samples(synh, :STPVars, :u) ≈ uh rtol = 1.0e-5
        @test _samples(synh, :STPVars, :x) ≈ xh rtol = 1.0e-5
    end

    @testset "STP, clock-driven rule: stored values are the values" begin
        U, τF, τD = 0.2f0, 300.0f0, 150.0f0
        syn, pre, post, pf, qf = _lz_run(stp = MarkramSTPParameterTimestep(U = U, τF = τF, τD = τD),
                                         monitor = s -> monitor!(s, [:u, :x], :STPVars, sr = 10kHz))
        uref, xref = _dense_stp_timestep(pf, U, τF, τD, _DT_LZ)
        @test _samples(syn, :STPVars, :u) ≈ uref rtol = 1.0e-4
        @test _samples(syn, :STPVars, :x) ≈ xref rtol = 1.0e-4
        @test plasticity_value(syn, :STPVars, :u, 1.0f0) === syn.STPVars.u
    end

    @testset "recording does not change the dynamics (bit-identical weights)" begin
        for (name, (mk, traces)) in _rules
            keys = collect(first.(traces))
            a = _lz_run(ltp = mk())[1]
            b = _lz_run(ltp = mk(), monitor = s -> monitor!(s, keys, :LTPVars, sr = 10kHz))[1]
            @test a.W == b.W
            for k in propertynames(a.LTPVars)
                @test getfield(a.LTPVars, k) == getfield(b.LTPVars, k)     # stored values and scales as well
            end
        end
        # STP (the efficacy of the synapses) and STP together with LTP
        for stp in (MarkramSTPParameter(U = 0.2f0, τF = 300.0f0, τD = 150.0f0),
                    MarkramSTPParameterTimestep(U = 0.2f0, τF = 300.0f0, τD = 150.0f0))
            ltp = STDPGerstner(A_pre = 0.05f0, A_post = -0.04f0, Wmax = 80.0f0, Wmin = 0.0f0)
            a = _lz_run(ltp = ltp, stp = stp)[1]
            b = _lz_run(ltp = ltp, stp = stp, monitor = s -> (monitor!(s, [:u, :x, :_ρ], :STPVars, sr = 10kHz);
                                                              monitor!(s, [:tpre, :tpost], :LTPVars, sr = 10kHz);
                                                              monitor!(s, [:W, :ρ], sr = 10kHz)))[1]
            @test a.W == b.W
            @test a.ρ == b.ρ
            @test a.STPVars.u == b.STPVars.u && a.STPVars.x == b.STPVars.x
            @test a.LTPVars.tpre == b.LTPVars.tpre && a.LTPVars.scale == b.LTPVars.scale
        end
    end

    @testset "sync_traces! gives the values of trace_values" begin
        for (name, (mk, traces)) in _rules
            syn = _lz_run(ltp = mk())[1]
            v = syn.LTPVars
            keys = first.(traces)
            stored, scale = [copy(getfield(v, k)) for k in keys], copy(v.scale)
            vals = [trace_values(syn, k) for k in keys]
            @test [copy(getfield(v, k)) for k in keys] == stored && v.scale == scale   # reading left the state alone
            @test any(!=(1.0), v.scale)
            @test trace_values(syn, :last_pre) === v.last_pre                            # other fields as they are
            @test sync_traces!(syn) === syn
            @test all(==(1.0), v.scale)
            @test [getfield(v, k) for k in keys] == vals                                 # same values, bit for bit
            @test all(k -> plasticity_value(syn, :LTPVars, k, 0.0f0) == getfield(v, k), keys)
            v0 = [copy(getfield(v, k)) for k in keys]
            sync_traces!(v)                                                              # idempotent
            @test [getfield(v, k) for k in keys] == v0
        end
        # variables without lazy traces
        @test sync_traces!(SNNModels.NoVariables()) isa SNNModels.NoVariables
        @test sync_traces!(SpikingSynapse(Identity(N = 3), Identity(N = 3), :g; conn = (p = 1.0, μ = 1.0))) isa SpikingSynapse
        syn = _lz_run(ltp = iSTDPPotential(η = 1.0f-3), nsteps = 100)[1]                 # tpost driven by v_post: dense, stored as it is
        @test all(==(1.0), syn.LTPVars.scale)
        @test trace_values(syn, :tpre) === syn.LTPVars.tpre
        @test plasticity_value(syn, :LTPVars, :tpre, 0.0f0) === syn.LTPVars.tpre
    end

    @testset "monitor! of a lazily stored variable without accessor fails loudly" begin
        struct _LazyNoAccessor <: SNNModels.PlasticityVariables
            a::Vector{Float32}
            active::Vector{Bool}
        end
        SNNModels.lazy_variables(::_LazyNoAccessor) = (:a,)
        v = _LazyNoAccessor(ones(Float32, 3), [true])
        @test_throws ArgumentError SNNModels.current_value(v, nothing, :a, 0.0f0)
        @test SNNModels.current_value(v, nothing, :active, 0.0f0) === v.active                # not lazy: as it is
        struct _ConnWithLazy <: SNNModels.AbstractConnection
            name::String
            records::Dict{Symbol,Any}
            LTPVars::_LazyNoAccessor
        end
        c = _ConnWithLazy("c", Dict{Symbol,Any}(), v)
        @test_throws ArgumentError monitor!(c, [:a], :LTPVars)
        @test !haskey(c.records, :LTPVars_a)
    end

    @testset "the stored traces are rescaled before the scale underflows" begin
        # τ = 1 ms at dt = 0.125 ms: the factor is 0.88, the scale reaches 1e-6 after about 110 steps
        rule = STDPGerstner(τpre = 1.0f0, τpost = 1.5f0, A_pre = 0.05f0, A_post = -0.04f0, Wmax = 80.0f0, Wmin = 0.0f0)
        syn, pre, post, pf, qf = _lz_run(ltp = rule, nsteps = 2000, monitor = s -> monitor!(s, [:tpre, :tpost], :LTPVars, sr = 10kHz))
        @test all(s -> _K._SCALE_MIN <= s <= 1.0, syn.LTPVars.scale)
        @test _samples(syn, :LTPVars, :tpre) ≈ _dense_trace(pf, 1.0, _DT_LZ) rtol = 1.0e-5 atol = 1.0e-12
        @test _samples(syn, :LTPVars, :tpost) ≈ _dense_trace(qf, 1.5, _DT_LZ) rtol = 1.0e-5 atol = 1.0e-12
        # traces of neurons that stopped firing flush to zero instead of becoming subnormal
        s2 = SpikingSynapse(Identity(N = 6), Identity(N = 5), :g; conn = (p = 1.0, μ = 1.0), LTPParam = rule)
        s2.LTPVars.tpre .= 1.0f0
        T = Time()
        for n in 1:3000
            update_time!(T, _DT_LZ)
            plasticity!(s2, s2.param, _DT_LZ, T)
        end
        @test all(x -> x == 0 || abs(x) >= floatmin(Float32), s2.LTPVars.tpre)
        @test all(trace_values(s2, :tpre) .< 1.0f-30)
    end

    @testset "a change of dt or of a time constant keeps the decay of the steps already run" begin
        Random.seed!(5)
        pre, post = Identity(N = 20), Identity(N = 15)
        rule = STDPGerstner(A_pre = 0.05f0, A_post = -0.04f0, Wmax = 20.0f0, Wmin = 0.0f0)
        syn = SpikingSynapse(pre, post, :g; conn = (p = 0.5f0, μ = 5.0f0), LTPParam = rule)
        dts = n -> n <= 300 ? 0.125f0 : n <= 450 ? 0.25f0 : 0.0625f0
        T = Time()
        xpre, xpost = zeros(20), zeros(15)
        Random.seed!(9)
        for n in 1:700
            pf, qf = rand(20) .< 0.05, rand(15) .< 0.05
            pre.fire .= pf; post.fire .= qf
            dt = dts(n)
            update_time!(T, dt)
            plasticity!(syn, syn.param, dt, T)
            xpre .= (xpre .+ pf) .* exp(-Float64(dt) / 20.0)
            xpost .= (xpost .+ qf) .* exp(-Float64(dt) / 20.0)
        end
        @test Float64.(trace_values(syn, :tpre)) ≈ xpre rtol = 1.0e-5 atol = 1.0e-12
        @test Float64.(trace_values(syn, :tpost)) ≈ xpost rtol = 1.0e-5 atol = 1.0e-12
    end

    @testset "weights agree with the dense Float32 decay of SNNModels 1.9.1 to rounding" begin
        # the dense kernels of the old path (traces multiplied at every step), run by hand
        Random.seed!(7)
        Npre, Npost = 50, 40
        p = STDPGerstner(A_pre = 0.05f0, A_post = -0.04f0, Wmax = 20.0f0, Wmin = 0.0f0)
        pre, post = Identity(N = Npre), Identity(N = Npost)
        syn = SpikingSynapse(pre, post, :g; conn = (p = 0.3f0, μ = 5.0f0, σ = 1.0f0), LTPParam = p)
        Wd = copy(syn.W)
        tpre, tpost = zeros(Float32, Npre), zeros(Float32, Npost)
        lp, lq = zeros(Float32, Npre), zeros(Float32, Npost)
        T = Time()
        fpre, fpost = Float32(exp(-_DT_LZ / p.τpre)), Float32(exp(-_DT_LZ / p.τpost))
        Random.seed!(13)
        for n in 1:1500
            pre.fire .= rand(Npre) .< 0.05; post.fire .= rand(Npost) .< 0.05
            update_time!(T, _DT_LZ)
            plasticity!(syn, syn.param, _DT_LZ, T)
            _K._pre_spike_pass!((w, i, j) -> p.A_post * tpost[i], Wd, syn.colptr, syn.I, syn.fireJ, p.Wmin, p.Wmax)
            _K._post_spike_pass!((w, i, j) -> p.A_pre * tpre[j], Wd, syn.rowptr, syn.index, syn.J, syn.fireI, p.Wmin, p.Wmax)
            _K._spike_increment!(tpre, lp, syn.fireJ, get_time(T))
            _K._spike_increment!(tpost, lq, syn.fireI, get_time(T))
            _K._rescale!(tpre, fpre); _K._rescale!(tpost, fpost)
        end
        @test maximum(abs.(syn.W .- Wd)) > 0                      # not bit-identical to the dense path ...
        @test maximum(abs.(syn.W .- Wd)) < 5.0f-5               # ... but equal to Float32 rounding
        @test maximum(abs.(Wd .- 5.0f0)) > 0.1                   # plasticity acted
        @test trace_values(syn, :tpre) ≈ tpre rtol = 1.0f-5 atol = 1.0f-12
    end
end
