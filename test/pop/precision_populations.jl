# Float64 populations (SNNModels 1.10): every population model and every population synapse model runs in Float64 when the
# population is built with `VFT = Vector{Float64}`; the parameters, spike, synapse, synaptic variables, receptor buffers and
# dendrites follow `VFT`. (`SNNModels.Poisson`: the suite also loads Distributions, which exports a `Poisson`.) The Float32 default is unchanged (bit-identical, checked by the *_update_loop.jl tests and by the J3
# identity hashes of the validation campaign).
using SNNModels, Test, Random
import SNNModels: integrate!, convert_precision, synaptic_variables, synaptic_receptors
@load_units

const _PP_SKIP = (:records, :targets, :id, :name, :spikes, :Vspike)   # Vspike: Float32 threshold field by design (documented)

# every floating-point value reachable from a population: (path, type) for arrays (element type) and scalars
function _pp_floats!(out, x, path)
    if x isa AbstractArray{<:AbstractFloat}
        push!(out, path => eltype(x))
    elseif x isa AbstractFloat
        push!(out, path => typeof(x))
    elseif x isa NamedTuple
        for (k, v) in pairs(x); _pp_floats!(out, v, "$path.$k"); end
    elseif x isa AbstractVector && !isempty(x) && !(eltype(x) <: Number)
        for (i, v) in enumerate(x); _pp_floats!(out, v, "$path[$i]"); end
    elseif x isa Union{SNNModels.AbstractSynapseVariable,SNNModels.AbstractSynapseParameter,SNNModels.AbstractParameter,
                       SNNModels.Dendrite,PostSpike,Receptor,NMDAVoltageDependency}
        x isa DendNeuronParameter && return out    # dendritic geometry parameters: precision-agnostic (Float32 lengths)
        for f in fieldnames(typeof(x)); _pp_floats!(out, getfield(x, f), "$path.$f"); end
    end
    return out
end
function _pp_floats(p)
    out = Pair{String,DataType}[]
    for f in fieldnames(typeof(p))
        f in _PP_SKIP && continue
        _pp_floats!(out, getfield(p, f), String(f))
    end
    return out
end

# copy every float state array of `src` into the matching array of `dst` (converted): same initial state in both precisions
function _pp_copy!(dst, src)
    for f in fieldnames(typeof(src))
        f in (:records, :targets, :param, :spikes) && continue
        a, b = getfield(dst, f), getfield(src, f)
        if a isa AbstractArray{<:AbstractFloat}
            a .= b
        elseif a isa AbstractArray{Bool} || a isa AbstractArray{<:Integer}
            a .= b
        elseif a isa NamedTuple
            for k in keys(a); getfield(a, k) .= getfield(b, k); end
        elseif a isa Union{SNNModels.AbstractSynapseVariable,SNNModels.Dendrite}
            _pp_copy!(a, b)
        end
    end
    return dst
end

_pp_main(p) = hasfield(typeof(p), :v) ? p.v : hasfield(typeof(p), :v_s) ? p.v_s : hasfield(typeof(p), :x) ? p.x :
              hasfield(typeof(p), :r) ? p.r : p.fire
_pp_alloc(p, dt) = (integrate!(p, p.param, dt); integrate!(p, p.param, dt); @allocated integrate!(p, p.param, dt))
function _pp_run!(p, dt, n; drive = nothing)
    fires = Vector{Vector{Bool}}()
    for _ in 1:n
        drive === nothing || drive(p)
        integrate!(p, p.param, dt)
        hasfield(typeof(p), :fire) && push!(fires, copy(p.fire))
    end
    return fires
end
_pp_state(p) = [copy(getfield(p, f)) for f in fieldnames(typeof(p)) if getfield(p, f) isa AbstractArray && !(f in (:records,))]

const _PP_N = 37
_pp_rv(FT, n, s) = FT.(s .* (0.5 .+ rand(Xoshiro(7), n)))
# name => (builder(FT; quiet), dt, synapse target symbol and compartment, or nothing)
const _PP_MODELS = [
    "IF" => ((FT; quiet = false) -> (p = IF(N = _PP_N, VFT = Vector{FT}); p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 300); p), 0.1, (:ge,)),
    "AdEx" => ((FT; quiet = false) -> (p = AdEx(N = _PP_N, VFT = Vector{FT}); p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 800); p), 0.1, (:ge,)),
    "AdExHet" => ((FT; quiet = false) -> begin
        n = _PP_N
        pa = AdExParameter(FT = Vector{FT}, C = _pp_rv(FT, n, 281), gl = _pp_rv(FT, n, 40), Vt = fill(FT(-50), n), Vr = fill(FT(-70.6), n),
                           El = fill(FT(-70.6), n), ΔT = _pp_rv(FT, n, 2), τw = _pp_rv(FT, n, 144), a = _pp_rv(FT, n, 4), b = _pp_rv(FT, n, 80))
        p = AdEx(N = n, VFT = Vector{FT}, param = pa); p.I .= quiet ? 0 : _pp_rv(FT, n, 800); p
    end, 0.1, (:ge,)),
    "ExtendedIF" => ((FT; quiet = false) -> (p = ExtendedIF(N = _PP_N, VFT = Vector{FT}); p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 400); p), 0.1, (:ge,)),
    "HH" => ((FT; quiet = false) -> (p = HH(N = _PP_N, VFT = Vector{FT}); p.ge .= 0; p.gi .= 0; p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 100); p), 0.02, (:ge,)),
    "IZ" => ((FT; quiet = false) -> (p = IZ(N = _PP_N, VFT = Vector{FT}); p.ge .= 0; p.gi .= 0; p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 12); p), 0.1, (:ge,)),
    "MorrisLecar" => ((FT; quiet = false) -> (p = MorrisLecar(N = _PP_N, VFT = Vector{FT}); p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 40); p), 0.05, (:ge,)),
    # rate models: their connections (RateSynapse, FORCE) are Float32 only (not in the scope of the Float64 populations)
    "Rate" => ((FT; quiet = false) -> (p = Rate(N = _PP_N, VFT = Vector{FT}); p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 1); p), 0.1, nothing),
    "WilsonCowan" => ((FT; quiet = false) -> (p = WilsonCowan(N = _PP_N, VFT = Vector{FT}); p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 1); p), 0.1, nothing),
    "Tripod" => ((FT; quiet = false) -> (p = Tripod(N = _PP_N, VFT = Vector{FT}); p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 800); p), 0.1, (:glu, :d1)),
    "BallAndStick" => ((FT; quiet = false) -> (p = BallAndStick(N = _PP_N, VFT = Vector{FT}); p.Is .= quiet ? 0 : _pp_rv(FT, _PP_N, 800); p), 0.1, (:glu, :d)),
    "PoissonHomo" => ((FT; quiet = false) -> SNNModels.Poisson(N = _PP_N, VFT = Vector{FT}, param = PoissonParameter(quiet ? 1Hz : 20Hz; FT)), 0.1, nothing),
    "PoissonDense" => ((FT; quiet = false) -> SNNModels.Poisson(N = _PP_N, VFT = Vector{FT}, param = PoissonParameter(500Hz; FT)), 0.1, nothing),
    "PoissonHet" => ((FT; quiet = false) -> SNNModels.Poisson(N = _PP_N, VFT = Vector{FT}, param = PoissonParameter(FT.(range(1, 50, length = _PP_N) .* Hz); FT)), 0.1, nothing),
    "VariablePoisson" => ((FT; quiet = false) -> VariablePoisson(N = _PP_N, VFT = Vector{FT}, param = VariablePoissonParameter(FT = FT, β = 2.0)), 0.1, nothing),
    "InhomogeneousPoisson" => ((FT; quiet = false) -> InhomogeneousPoisson(N = _PP_N, VFT = Vector{FT},
                                param = InhomogeneousPoissonParam(FT = FT, β = 2.0, r0 = 20Hz)), 0.1, nothing),
]
const _PP_POISSON = ("PoissonHomo", "PoissonDense", "PoissonHet", "VariablePoisson", "InhomogeneousPoisson")

@testset "Float64 populations" verbose = true begin
    @testset "$name" for (name, (mk, dt, tgt)) in _PP_MODELS
        Random.seed!(11)
        p = mk(Float64)
        # construction from VFT only: every float of the state, parameters, spike, synapse and dendrites is Float64
        fl = _pp_floats(p)
        @test !isempty(fl)
        bad = filter(x -> x.second !== Float64, fl)
        @test isempty(bad) || (@info "Float32 leak in $name" bad; false)
        @test float_type(p) === Float64
        # one step with a Float64 dt is inferred and allocates no more than the Float32 step
        @test (@inferred(integrate!(p, p.param, 0.1)); true)
        # (a population that allocates in Float32, through the chunk closure of the threaded step, may allocate a few bytes more:
        # the closure captures `dt`)
        q = mk(Float32)
        a64, a32 = _pp_alloc(p, dt), _pp_alloc(q, Float32(dt))
        @test a32 == 0 ? a64 == 0 : a64 <= 2 * a32
        @test all(isfinite, _pp_main(p))

        # Float64 against Float32 on a short horizon, from the same state with the same drive
        if !(name in _PP_POISSON)
            for quiet in (true, false)
                Random.seed!(3); p32 = mk(Float32; quiet)
                Random.seed!(3); p64 = mk(Float64; quiet)
                _pp_copy!(p64, p32)
                f32 = _pp_run!(p32, Float32(dt), 20)
                f64 = _pp_run!(p64, dt, 20)
                m32, m64 = Float64.(_pp_main(p32)), _pp_main(p64)
                @test f32 == f64                                     # same spikes (none in the quiet regime of the spiking models)
                @test maximum(abs.(m64 .- m32)) <= 1e-4 * max(1.0, maximum(abs, m32))
            end
        else
            # Poisson: the sparse regime draws the same Float64 uniforms in both precisions; the dense regime draws uniforms of the
            # population type (different streams): same rate
            Random.seed!(5); p32 = mk(Float32); f32 = _pp_run!(p32, 0.1f0, 2000)
            Random.seed!(5); p64 = mk(Float64); f64 = _pp_run!(p64, 0.1, 2000)
            n32, n64 = sum(count, f32), sum(count, f64)
            @test abs(n64 - n32) <= 5 * sqrt(max(n32, 1)) + 2
            name in ("PoissonHomo", "PoissonHet") && @test f32[1:20] == f64[1:20]
        end

        # self-consistency: two Float64 runs from the same seed are identical
        Random.seed!(21); a = mk(Float64); fa = _pp_run!(a, dt, 200)
        Random.seed!(21); b = mk(Float64); fb = _pp_run!(b, dt, 200)
        @test fa == fb && isequal(_pp_state(a), _pp_state(b))

        # a SpikingSynapse onto the Float64 population is Float64
        if tgt !== nothing
            pre = Identity(N = 5, VFT = Vector{Float64})
            syn = SpikingSynapse(pre, p, tgt...; conn = (p = 1.0, μ = 1.0))
            @test float_type(syn) === Float64 && eltype(syn.W) === Float64
        end
    end

    @testset "Float64 step halves agree to O(dt)" begin
        # quiet regime, 5 ms: dt against dt / 2 (forward Euler / Heun error), and Float64 against Float32 much closer than that
        for (name, (mk, dt, _)) in _PP_MODELS
            name in _PP_POISSON && continue
            Random.seed!(3); a = mk(Float64; quiet = true)
            Random.seed!(3); b = mk(Float64; quiet = true); _pp_copy!(b, a)
            n = round(Int, 5 / dt)
            _pp_run!(a, dt, n); _pp_run!(b, dt / 2, 2n)
            d = maximum(abs.(_pp_main(a) .- _pp_main(b)))
            @test all(isfinite, _pp_main(a)) && all(isfinite, _pp_main(b))
            @test d <= 0.1 * max(1.0, maximum(abs, _pp_main(a)))
        end
    end

    @testset "population synapse models in Float64 on $P" for P in (IF, AdEx)
        syns = [SingleExpSynapse(FT = Float64), DeltaSynapse(), CurrentSynapse(FT = Float64), DoubleExpCurrentSynapse(FT = Float64),
                DoubleExpSynapse(FT = Float64), convert_precision(ReceptorSynapse(), Float64),
                convert_precision(MultiReceptorSynapse(), Float64), Confavreux2025Synapse(FT = Float64)]
        for s in syns
            Random.seed!(2)
            p = P(N = 9, VFT = Vector{Float64}, synapse = s)
            @test isempty(filter(x -> x.second !== Float64, _pp_floats(p)))
            for b in values(p.receptors); b .= 0.5; end
            integrate!(p, p.param, 0.1)
            @test all(isfinite, p.v) && all(isfinite, p.syn_curr)
            @test (@inferred(integrate!(p, p.param, 0.1)); true)
            # against Float32 from the same state
            s32 = s isa DeltaSynapse ? s : convert_precision(s, Float32)
            Random.seed!(2); q = P(N = 9, synapse = s32); Random.seed!(2); p = P(N = 9, VFT = Vector{Float64}, synapse = s)
            _pp_copy!(p, q); q.I .= 300; p.I .= 300
            for k in 1:20
                for b in values(q.receptors); b .= 0.2f0; end
                for b in values(p.receptors); b .= 0.2f0; end
                integrate!(q, q.param, 0.1f0); integrate!(p, p.param, 0.1)
            end
            @test q.fire == p.fire
            @test maximum(abs.(p.v .- q.v)) <= 1e-3
        end
    end

    @testset "synaptic_variables / synaptic_receptors with a float type" begin
        for s in (SingleExpSynapse(), DeltaSynapse(), CurrentSynapse(), DoubleExpCurrentSynapse(), DoubleExpSynapse(),
                  ReceptorSynapse(), MultiReceptorSynapse(), Confavreux2025Synapse(), TripodSomaSynapse, TripodDendSynapse)
            for FT in (Float32, Float64)
                sv = synaptic_variables(s, 4, FT)
                @test all(f -> !(getfield(sv, f) isa AbstractArray) || eltype(getfield(sv, f)) === FT, fieldnames(typeof(sv)))
                @test all(b -> eltype(b) === FT, values(synaptic_receptors(s, 4, FT)))
            end
            # the two-argument forms keep the precision of the synapse model (Float32 by default)
            @test typeof(synaptic_variables(s, 4)) == typeof(synaptic_variables(s, 4, Float32))
        end
    end

    @testset "convert_precision round trip" begin
        for s in (SingleExpSynapse(), CurrentSynapse(), DoubleExpCurrentSynapse(), DoubleExpSynapse(), ReceptorSynapse(),
                  MultiReceptorSynapse(), Confavreux2025Synapse(), TripodSomaSynapse, TripodDendSynapse,
                  AdExParameter(), PostSpike(), HHParameter(), IZParameter(), SNNModels.MorrisLecarParameter(), ExtendedIFParameter(),
                  PoissonHomoParameter(), InhomogeneousPoissonParam(), VariablePoissonParameter())
            @test convert_precision(s, Float32) === s                      # Float32 objects are used as they are
            s64 = convert_precision(s, Float64)
            @test float_type(s64) === Float64
            back = convert_precision(s64, Float32)
            @test typeof(back) == typeof(s)
            @test all(f -> isequal(getfield(back, f), getfield(s, f)), fieldnames(typeof(s)))
        end
        @test convert_precision(DeltaSynapse(), Float64) === DeltaSynapse()  # no float field: precision-agnostic
    end

    @testset "refusals: one precision per population" begin
        @test_throws ArgumentError AdEx(N = 3, VFT = Vector{Float64}, param = AdExParameter())
        @test_throws ArgumentError AdEx(N = 3, VFT = Vector{Float64}, synapse = SingleExpSynapse())
        @test_throws ArgumentError AdEx(N = 3, VFT = Vector{Float64}, spike = PostSpike())
        @test_throws ArgumentError AdEx(N = 3, synapse = SingleExpSynapse(FT = Float64))
        @test_throws ArgumentError IF(N = 3, VFT = Vector{Float64}, synapse = ReceptorSynapse())
        @test_throws ArgumentError HH(N = 3, VFT = Vector{Float64}, param = HHParameter())
        @test_throws ArgumentError IZ(N = 3, VFT = Vector{Float64}, param = IZParameter())
        @test_throws ArgumentError MorrisLecar(N = 3, VFT = Vector{Float64}, param = SNNModels.MorrisLecarParameter())
        @test_throws ArgumentError ExtendedIF(N = 3, VFT = Vector{Float64}, param = ExtendedIFParameter())
        @test_throws ArgumentError Tripod(N = 3, VFT = Vector{Float64}, adex = AdExParameter())
        @test_throws ArgumentError Tripod(N = 3, VFT = Vector{Float64}, soma_syn = TripodSomaSynapse)
        @test_throws ArgumentError BallAndStick(N = 3, VFT = Vector{Float64}, dend_syn = TripodDendSynapse)
        @test_throws ArgumentError SNNModels.Poisson(N = 3, VFT = Vector{Float64}, param = PoissonParameter(10Hz))
        @test_throws ArgumentError InhomogeneousPoisson(N = 3, VFT = Vector{Float64}, param = InhomogeneousPoissonParam())
        @test_throws ArgumentError VariablePoisson(N = 3, VFT = Vector{Float64}, param = VariablePoissonParameter())
        # DeltaSynapse has no float field: accepted in both precisions
        @test float_type(IF(N = 3, VFT = Vector{Float64}, synapse = DeltaSynapse()).synvars.ge) === Float64
    end

    @testset "network of Float64 Tripod and IF" begin
        Random.seed!(4)
        E = SNNModels.Poisson(N = 20, VFT = Vector{Float64}, param = PoissonParameter(50Hz; FT = Float64))
        T = Tripod(N = 10, VFT = Vector{Float64})
        I = IF(N = 10, VFT = Vector{Float64})
        s1 = SpikingSynapse(E, T, :glu, :d1; conn = (p = 0.5, μ = 2.0))
        s2 = SpikingSynapse(E, I, :ge; conn = (p = 0.5, μ = 2.0))
        s3 = SpikingSynapse(T, I, :ge; conn = (p = 0.5, μ = 2.0))
        model = compose(; E, T, I, s1, s2, s3, silent = true)
        @test typeof(model.time) == Time{Float64}
        @test all(s -> eltype(s.W) === Float64, (s1, s2, s3))
        monitor!(T, [:fire])
        sim!(; model, duration = 100ms)
        @test all(isfinite, T.v_s) && all(isfinite, T.v_d1) && all(isfinite, I.v)
        @test eltype(T.v_s) === Float64
    end
end
