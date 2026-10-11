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
# main state and spike steps of every neuron over `n` steps
function _pp_trace!(p, dt, n)
    main = Vector{Vector{Float64}}()
    spikes = [Int[] for _ in 1:(hasfield(typeof(p), :fire) ? length(p.fire) : 0)]
    for k in 1:n
        integrate!(p, p.param, dt)
        push!(main, Float64.(_pp_main(p)))
        for i in eachindex(spikes); p.fire[i] && push!(spikes[i], k); end
    end
    return (; main, spikes)
end
# drive gain of the long driven comparison (HH and MorrisLecar need a stronger current than the short tests to fire)
const _PP_DRIVE_GAIN = Dict("HH" => 3, "MorrisLecar" => 2)
_pp_scale_drive!(p, g) = (hasfield(typeof(p), :Is) ? (p.Is .*= g) : hasfield(typeof(p), :I) && (p.I .*= g); p)
_pp_state(p) = [copy(getfield(p, f)) for f in fieldnames(typeof(p)) if getfield(p, f) isa AbstractArray && !(f in (:records,))]

# a synapse model written for SNNModels <= 1.9: only the two-argument `synaptic_variables` / `synaptic_receptors`
struct _PPLegacySyn{FT} <: SNNModels.AbstractSynapseParameter
    τ::FT
end
SNNModels.synaptic_variables(::_PPLegacySyn, N::Int) = SNNModels.SingleExpSynapseVars(N = N)
SNNModels.synaptic_receptors(::_PPLegacySyn, N::Int) = (exc = zeros(Float32, N),)

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
    # non-default synapse models of the dendritic populations (conductance-based single exponential on soma and dendrites)
    "TripodSingleExp" => ((FT; quiet = false) -> (p = Tripod(N = _PP_N, VFT = Vector{FT}, soma_syn = SingleExpSynapse(FT = FT),
                          dend_syn = SingleExpSynapse(FT = FT)); p.I .= quiet ? 0 : _pp_rv(FT, _PP_N, 800); p), 0.1, (:glu, :d1)),
    "BallAndStickSingleExp" => ((FT; quiet = false) -> (p = BallAndStick(N = _PP_N, VFT = Vector{FT}, soma_syn = SingleExpSynapse(FT = FT),
                                dend_syn = SingleExpSynapse(FT = FT)); p.Is .= quiet ? 0 : _pp_rv(FT, _PP_N, 800); p), 0.1, (:glu, :d)),
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

        # Float64 against Float32 from the same state with the same drive: quiet regime 200 steps (no spike), driven regime 500
        # steps (1000 for HH at dt = 0.02 ms) with a drive that makes every spiking model fire several times
        if !(name in _PP_POISSON)
            Random.seed!(3); p32 = mk(Float32; quiet = true)
            Random.seed!(3); p64 = mk(Float64; quiet = true)
            _pp_copy!(p64, p32)
            f32 = _pp_run!(p32, Float32(dt), 200)
            f64 = _pp_run!(p64, dt, 200)
            m32, m64 = Float64.(_pp_main(p32)), _pp_main(p64)
            @test f32 == f64 && !any(any, f32)
            @test maximum(abs.(m64 .- m32)) <= 1e-4 * max(1.0, maximum(abs, m32))

            gain, nsteps = get(_PP_DRIVE_GAIN, name, 1), name == "HH" ? 1000 : 500
            Random.seed!(3); p32 = mk(Float32); _pp_scale_drive!(p32, gain)
            Random.seed!(3); p64 = mk(Float64); _pp_scale_drive!(p64, gain)
            _pp_copy!(p64, p32)
            t32, t64 = _pp_trace!(p32, Float32(dt), nsteps), _pp_trace!(p64, dt, nsteps)
            if hasfield(typeof(p32), :fire)
                s32, s64 = t32.spikes, t64.spikes                    # spike steps per neuron
                n32, n64 = sum(length, s32), sum(length, s64)
                @test n32 >= 20                                      # every spiking model fires (20 to 200 spikes of 37 neurons)
                @test abs(n64 - n32) <= max(2, 0.02 * n32)
                @test all(abs(length(a) - length(b)) <= 1 for (a, b) in zip(s32, s64))
                @test all(abs(a[1] - b[1]) <= 2 for (a, b) in zip(s32, s64) if !isempty(a) && !isempty(b))
                # state before the first spike of either run
                k0 = min(minimum(first, filter(!isempty, s32)), minimum(first, filter(!isempty, s64))) - 1
                @test k0 >= 5
                ref = max(1.0, maximum(abs, t32.main[1]))
                @test maximum(maximum(abs.(t64.main[k] .- t32.main[k])) for k in 1:k0) <= 1e-4 * ref
            else
                ref = max(1.0, maximum(abs, t32.main[end]))
                @test maximum(maximum(abs.(t64.main[k] .- t32.main[k])) for k in 1:nsteps) <= 1e-4 * ref
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

    @testset "synapse models with only the two-argument forms (written for 1.9)" begin
        s = _PPLegacySyn(1.0f0)
        @test synaptic_variables(s, 4, Float32) isa SNNModels.SingleExpSynapseVars{Vector{Float32}}
        @test keys(synaptic_receptors(s, 4, Float32)) == (:exc,)
        # another precision needs the three-argument method: the error names it
        @test_throws ErrorException synaptic_variables(s, 4, Float64)
        @test_throws ErrorException synaptic_receptors(s, 4, Float64)
        err = try synaptic_receptors(s, 4, Float64) catch e; sprint(showerror, e) end
        @test occursin("synaptic_receptors(synapse, N, FT)", err)
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

    @testset "Float64 defaults carry the decimal values" begin
        # receptors of the predefined synapses: the decimal values of their definition, derived fields computed in Float64
        r_exact(r, E, τr, τd, g0) = r.E_rev === E && r.τr === τr && r.τd === τd && r.g0 === g0 &&
                                    r.gsyn === g0 * norm_synapse(τr, τd) && r.α === SNNModels.α_synapse(τr, τd) &&
                                    r.τr⁻ === 1 / τr && r.τd⁻ === 1 / τd
        soma = convert_precision(TripodSomaSynapse, Float64)
        @test r_exact(soma.syn[1], 0.0, 0.26, 2.0, 0.73) && r_exact(soma.syn[2], -70.0, 0.1, 15.0, 0.38)
        dend = convert_precision(TripodDendSynapse, Float64)
        @test r_exact(dend.syn[1], 0.0, 0.26, 2.0, 0.73) && r_exact(dend.syn[2], 0.0, 8.0, 35.0, 1.31) && dend.syn[2].nmda === 1.0
        @test r_exact(dend.syn[3], -70.0, 4.8, 29.0, 0.27) && r_exact(dend.syn[4], -90.0, 30.0, 400.0, 0.006)
        for n in (soma.NMDA, dend.NMDA, convert_precision(EyalNMDA, Float64), NMDAVoltageDependency(T = Float64))
            @test n.b === 3.36 && n.k === -0.077 && n.mg === 1.0
        end
        rs = convert_precision(ReceptorSynapse(), Float64)
        @test r_exact(rs.syn[1], 0.0, 1.0, 6.0, 0.7) && r_exact(rs.syn[2], 0.0, 1.0, 100.0, 0.15)
        @test r_exact(rs.syn[3], -70.0, 0.5, 10.0, 2.0) && r_exact(rs.syn[4], -90.0, 30.0, 400.0, 0.006)
        @test convert_precision(MultiReceptorSynapse(), Float64).syn == rs.syn
        # a receptor that is not predefined is converted value by value (its Float32 rounding is kept)
        @test convert_precision(Receptor(τr = 0.3, τd = 3.0, g0 = 0.9), Float64).τr === Float64(0.3f0)
        for P in (Tripod, BallAndStick)
            p = P(N = 3, VFT = Vector{Float64})
            @test p.soma_syn.syn == soma.syn && p.dend_syn.syn == dend.syn && p.soma_syn.NMDA === soma.NMDA
            @test p.adex.τm === 281.0 / 40.0 && p.adex.R === 1 / 40.0 && p.adex.C === 281.0 && p.adex.gl === 40.0
        end
        # synapse models and neuron parameters
        @test Confavreux2025Synapse(FT = Float64).α === 0.23
        @test SingleExpSynapse(FT = Float64).gsyn_e === 1.0 && DoubleExpSynapse(FT = Float64).gsyn_i === 1.0
        a = AdExParameter(FT = Float64)
        @test a.τm === 7.025 && a.R === 0.025
        i = IFParameter(FT = Float64)
        @test i.τm === 15.0 && i.R === 0.06 && i.gl === 1 / 0.06 && i.C === 15.0 * (1 / 0.06)
        h = HHParameter(FT = Float64)
        @test h.Cm === 200.0 && h.gl === 10.0 && h.gn === 20000.0 && h.gk === 6000.0
        @test SNNModels.MorrisLecarParameter(FT = Float64).ϕ === 0.025
        @test PoissonHomoParameter(FT = Float64).rate === 0.001
        # dendritic geometry: lengths on the 1 um grid and the 4 um diameter as decimals, cable constants of `human_dend` as
        # decimals (Ri = 200 Ω cm, Rd = 38907 Ω cm², Cd = 0.5 μF/cm²), conductances and capacitance computed in Float64
        dec(x) = x === parse(Float64, string(round(Int, x * 1e6)) * "e-6")
        for (P, ds) in ((Tripod, (:d1, :d2)), (BallAndStick, (:d,)))
            Random.seed!(9); p = P(N = 20, VFT = Vector{Float64})
            for f in ds
                d = getfield(p, f)
                @test all(dec, d.l) && all(==(4e-4), d.d)
                @test d.gm == [SNNModels.G_mem(Rd = 3.8907e-5, d = 4e-4, l = l) for l in d.l]
                @test d.gax == [SNNModels.G_axial(Ri = 2e-7, d = 4e-4, l = l) for l in d.l]
                @test d.C == [SNNModels.C_mem(Cd = 5e5, d = 4e-4, l = l) for l in d.l]
            end
            # same lengths (in whole micrometres) as the Float32 population built from the same seed: one draw per dendrite
            # in both, of the same grid index (the Float32 grid points are not the Float32 roundings of the decimals: 0.038799997f0)
            Random.seed!(9); q = P(N = 20)
            for f in ds
                @test round.(Int, getfield(p, f).l .* 1e4) == round.(Int, getfield(q, f).l .* 1e4)
            end
        end
        # no default of a Float64 population is a Float32-rounded value (a Float32-representable Float64 with a long decimal form)
        f32_rounded(x) = isfinite(x) && Float64(Float32(x)) == x && length(string(x)) > 10
        function rounded!(out, x, path, seen)
            if x isa Float64
                f32_rounded(x) && push!(out, path => x)
            elseif x isa AbstractArray{Float64}
                for v in x; f32_rounded(v) && (push!(out, path => v); break); end
            elseif x isa AbstractArray && !(eltype(x) <: Number)
                for (k, v) in enumerate(x); rounded!(out, v, "$path[$k]", seen); end
            elseif x isa NamedTuple
                for (k, v) in pairs(x); rounded!(out, v, "$path.$k", seen); end
            elseif isstructtype(typeof(x)) && parentmodule(typeof(x)) === SNNModels && !(x in seen)
                push!(seen, x)
                for f in fieldnames(typeof(x)); isdefined(x, f) && rounded!(out, getfield(x, f), "$path.$f", seen); end
            end
            return out
        end
        for (name, (mk, _, _)) in _PP_MODELS
            name in ("AdExHet", "PoissonHet", "PoissonHomo", "PoissonDense", "InhomogeneousPoisson") && continue  # user-given rates/parameters
            Random.seed!(1); p = mk(Float64; quiet = true)
            r = rounded!(Pair{String,Float64}[], p, name, Base.IdSet{Any}())
            @test isempty(r) || (@info "Float32-rounded defaults in $name" r; false)
        end
        @test isempty(rounded!(Pair{String,Float64}[], SNNModels.Poisson(N = 3, VFT = Vector{Float64}), "Poisson", Base.IdSet{Any}()))
        for s in (SingleExpSynapse(FT = Float64), CurrentSynapse(FT = Float64), DoubleExpCurrentSynapse(FT = Float64),
                  DoubleExpSynapse(FT = Float64), convert_precision(ReceptorSynapse(), Float64), Confavreux2025Synapse(FT = Float64),
                  convert_precision(MultiReceptorSynapse(), Float64), PostSpike(FT = Float64), ExtendedIFParameter(FT = Float64),
                  IZParameter(FT = Float64), VariablePoissonParameter(FT = Float64), InhomogeneousPoissonParam(FT = Float64))
            r = rounded!(Pair{String,Float64}[], s, string(nameof(typeof(s))), Base.IdSet{Any}())
            @test isempty(r) || (@info "Float32-rounded defaults" r; false)
        end

        # Float32 defaults unchanged
        @test TripodSomaSynapse.syn[1].τr === 0.26f0 && TripodSomaSynapse.syn[1].gsyn === Float32(0.73 * norm_synapse(0.26, 2.0))
        @test Tripod(N = 2).soma_syn === TripodSomaSynapse && BallAndStick(N = 2).dend_syn === TripodDendSynapse
        @test HHParameter().Cm === 1uF * cm^(-2) * 20000um^2 && AdExParameter().τm === 281f0 / 40f0 && IFParameter().R === 0.06f0
        @test SNNModels.MorrisLecarParameter().ϕ === 25Hz && PoissonHomoParameter().rate === 1Hz
        @test Confavreux2025Synapse().α === 0.23f0 && EyalNMDA.b === 3.36f0 && NMDAVoltageDependency().k === -0.077f0
    end

    @testset "Float64 dendrite lengths: same draw and grid as Float32 on any 1 um range" begin
        # Every integer-um range (lo, hi) on a 5 um grid in 0-500 um (including 0-5, 10-500, 35-85, whose Float32 bounds are not
        # the Float32 roundings of the decimals), several seeds: the Float64 lengths are the decimal whole micrometres of the
        # Float32 lengths (k / 1e4), the Float32 lengths are on the integer-um grid, and the global RNG is left in the same state.
        um64 = 1e-4
        nbad_len = nbad_rng = nbad_dec = 0
        for lo in 0:5:495, hi in lo+5:5:500, seed in (1, 7)
            Random.seed!(seed); d32 = create_dendrite(3, (lo * um, hi * um)); s32 = copy(Random.default_rng())
            Random.seed!(seed); d64 = create_dendrite(3, (lo * um, hi * um); FT = Float64); s64 = copy(Random.default_rng())
            s32 == s64 || (nbad_rng += 1)
            for (a, b) in zip(d32.l, d64.l)
                a <= 0 && (b == -1 || (nbad_len += 1); continue)
                k32 = Float64(a) / Float64(um)
                k = round(Int, k32)
                abs(k32 - k) <= 1e-6 * k || (nbad_len += 1)
                abs(b / um64 - k) <= 1e-9 * k || (nbad_len += 1)
                b === parse(Float64, "$(k)e-4") || (nbad_dec += 1)
            end
        end
        @test nbad_len == 0
        @test nbad_rng == 0
        @test nbad_dec == 0
        # the examples of review round 2: same lengths as Float32 where the decimal range has another length
        for (lo, hi) in ((0, 5), (10, 500), (35, 85), (40, 65))
            Random.seed!(3); a = create_dendrite(50, (lo * um, hi * um)).l
            Random.seed!(3); b = create_dendrite(50, (lo * um, hi * um); FT = Float64).l
            @test round.(Int, a ./ um) == round.(Int, b ./ um64)
        end
        # fixed lengths and diameters given with the Float32 unit constants: the decimal whole micrometres
        for x in (5, 10, 35, 40, 65, 85, 200, 500)
            d = SNNModels._create_dendrite_decimal(x * um; d = x * um)
            @test d.l === parse(Float64, "$(x)e-4") && d.d === d.l
        end
        # Float32 path unchanged: the drawn value is an element of the Float32 range lo:1um:hi
        Random.seed!(4); a = create_dendrite(20, (35um, 85um)).l
        @test all(in(35um:1um:85um), a) && eltype(a) === Float32
        # physiologies that are not predefined: values given with the Float32 unit constants are read as their decimals
        # (a `Physiology` equal to a predefined one is that one: isbits, `===` compares the values)
        p = Physiology(210 * Ω * cm, 1700Ω * cm^2, 2μF / cm^2 |> Float32)
        @test SNNModels._decimal_physiology(p) == Physiology(2.1e-7, 1.7e-6, 2e6)
        q = Physiology(150 * Ω * cm, 38907 * Ω * cm^2, 0.5μF / cm^2 |> Float32)
        @test SNNModels._decimal_physiology(q) == Physiology(1.5e-7, 3.8907e-5, 5e5)
        r = Physiology(123.4f0 * Ω * cm, 2500.5f0 * Ω * cm^2, 0.75f0 * μF / cm^2)
        @test SNNModels._decimal_physiology(r) == Physiology(1.234e-7, 2.5005e-6, 7.5e5)
        # predefined: 200 Ω cm, 38907 Ω cm², 0.5 μF/cm² (human); 200 Ω cm, 1700 Ω cm², 1 μF/cm² (mouse); Ω = 1e-9, μF = 1e6, cm = 1
        @test SNNModels._decimal_physiology(human_dend) == Physiology(2e-7, 3.8907e-5, 5e5)
        @test SNNModels._decimal_physiology(mouse_dend) == Physiology(2e-7, 1.7e-6, 1e6)
    end

    @testset "integrate! takes the step in the precision of the population" begin
        # a step of another precision is converted to `float_type(p)` (as `sim!` does): same state as the converted step
        for (name, (mk, dt, _)) in _PP_MODELS, FT in (Float32, Float64)
            OT = FT === Float32 ? Float64 : Float32
            # (one population after the other: the Poisson models draw from the global RNG at every step)
            Random.seed!(8); a = mk(FT); for _ in 1:5; integrate!(a, a.param, FT(OT(dt))); end
            Random.seed!(8); b = mk(FT); for _ in 1:5; integrate!(b, b.param, OT(dt)); end
            @test isequal(_pp_state(a), _pp_state(b)) || (@info "dt conversion differs" name FT; false)
        end
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
