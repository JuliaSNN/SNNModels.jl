# Acceptance test of `Multipod`: with two dendrites it reproduces `Tripod` (HeunScheme) and `TripodPaper` (PaperScheme) bit for
# bit (`==`) in the membrane potentials, the adaptation current, the threshold, the refractory counters and the spike times.
#
# Protocols (those of the Tripod audit, see test/pop/tripod_21.jl):
#   - subthreshold current step, 300 pA from 50 to 250 ms (300 ms);
#   - single EPSP on d1 (event of weight 1 at step 500, 200 ms);
#   - suprathreshold current step, 1200 pA from 50 to 450 ms (500 ms; 15 spikes in the published code);
#   - frozen Poisson input on the six receptor buffers (glu/gaba of s, d1, d2): the audit events (500 ms) followed by a seeded
#     frozen Poisson train (500 ms), 1 s in total;
#   - Fig. 8 plateau: one event of weight 100 on d1, both dendrites of length 400, 250, 150 um (1 s).
# Every protocol runs on the two schemes; HeunScheme with the default receptors (exact kinetics) and with the receptors of the
# published code (`exact = false`, NMDA block exp32), so that the receptors, the NMDA gating, the backpropagation and refractory
# clamp and both spike rules are exercised.
using Test
using Random
using SNNModels
@load_units

const _MPEQ_DT = 0.1f0

_mpeq_data(name) = joinpath(@__DIR__, "..", "data", name)
function _mpeq_matrix(name)
    rows = [parse.(Float64, split(l)) for l in eachline(_mpeq_data(name)) if !startswith(l, "#") && !isempty(strip(l))]
    return reduce(vcat, (permutedims(r) for r in rows))
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Inputs of the protocols: a function `(row, step) -> weight` for the six receptor buffers and a current `t (ms) -> pA`
# ---------------------------------------------------------------------------------------------------------------------------------

# frozen Poisson events: the audit events (first 5000 steps) followed by a seeded train (rates chosen to make the neuron fire)
function _mpeq_poisson_events(nsteps)
    X = zeros(Float32, 6, nsteps)
    ev = _mpeq_matrix("tripod_poisson_events.txt")
    for k in axes(ev, 1)
        X[Int(ev[k, 1]), Int(ev[k, 2])] = Float32(ev[k, 3])
    end
    rng = Xoshiro(2024)
    rates = (2.0, 4.0, 4.0, 1.0, 1.0, 1.0) .* 1e-3 .* 10      # events per step on each row (kHz x dt)
    for tt in 5001:nsteps, r in 1:6
        rand(rng) < rates[r] && (X[r, tt] = 1.5f0)
    end
    return X
end

const _MPEQ_PROTOCOLS = (
    step_sub = (n = 3000, I = t -> 50 <= t < 250 ? 300f0 : 0f0, inputs = nothing),
    epsp_d1 = (n = 2000, I = nothing, inputs = (r, tt) -> (r == 2 && tt == 500) ? 1f0 : 0f0),
    step_supra = (n = 5000, I = t -> 50 <= t < 450 ? 1200f0 : 0f0, inputs = nothing),
    poisson_1s = (n = 10000, I = nothing, inputs = let X = _mpeq_poisson_events(10000); (r, tt) -> X[r, tt]; end),
)

# ---------------------------------------------------------------------------------------------------------------------------------
# Runner: drives one population and records the full state at every step
# ---------------------------------------------------------------------------------------------------------------------------------

# receptor buffers in the order of the event rows: glu s, d1, d2, gaba s, d1, d2
_mpeq_buffers(P) = (P.receptors_s.glu, P.receptors_d1.glu, P.receptors_d2.glu,
                    P.receptors_s.gaba, P.receptors_d1.gaba, P.receptors_d2.gaba)

function _mpeq_run(P, n; inputs = nothing, I = nothing, plateau_d1 = nothing)
    N = Int(P.N)
    trace = Vector{NTuple{7,Vector}}(undef, n)
    spikes = Tuple{Int,Int}[]
    for tt in 1:n
        if inputs !== nothing
            for (r, buf) in enumerate(_mpeq_buffers(P))
                w = inputs(r, tt)
                w != 0 && (buf .= w)
            end
        end
        plateau_d1 === nothing || (P.receptors_d1.glu .= tt == 150 ? Float32(plateau_d1) : 0f0)
        I === nothing || (P.I .= I(tt * 0.1f0))
        SNNModels.integrate!(P, P.param, convert(eltype(P.v_s), _MPEQ_DT))
        for i in 1:N
            P.fire[i] && push!(spikes, (tt, i))
        end
        trace[tt] = (copy(P.v_s), copy(P.v_d1), copy(P.v_d2), copy(P.w_s), copy(P.θ), copy(P.tabs), copy(P.fire))
    end
    return trace, spikes
end

# largest absolute difference over all steps of the potentials and w (0 when bit-identical; NaN-safe through `==` below)
_mpeq_maxdiff(a, b) = maximum(maximum(maximum(abs.(Float64.(x) .- Float64.(y))) for (x, y) in zip(sa[1:4], sb[1:4])) for (sa, sb) in zip(a, b))

# ---------------------------------------------------------------------------------------------------------------------------------
# Matched constructors: the Tripod reference and the Multipod with the same two lengths and the same keywords
# ---------------------------------------------------------------------------------------------------------------------------------

const _MPEQ_PAPER_SOMA = ReceptorSynapse(glu_receptors = [1], gaba_receptors = [2], NMDA = EyalNMDA32,
                                         syn = TripodSomaSynapsePaper.syn, decay = 32, exact = false)

function _mpeq_pair(scheme::Symbol, receptors::Symbol, lengths; N = 1, VFT = Vector{Float32})
    FT = eltype(VFT)
    if scheme === :paper
        config = lengths == [400um, 150um] ? :distal_proximal : nothing
        T = config === nothing ? TripodPaper(; N, VFT, param = TripodParameter(ds = lengths, El_dend = -70.6mV)) :
            TripodPaper(; N, VFT, config)
        M = MultipodPaper(; N, VFT, lengths)
        return T, M
    end
    # HeunScheme: library defaults (exact kinetics) or the receptors of the published code (`exact = false`)
    kw = receptors === :paper ?
         (soma_syn = convert_precision(_MPEQ_PAPER_SOMA, FT), dend_syn = convert_precision(TripodDendSynapsePaper, FT),
          adex = AdExParameter(FT = FT, Vt = -50.4mV), spike = PostSpike(FT = FT, AP_membrane = 20mV, up = 1ms, τabs = 2ms, At = 0mV),
          Vspike = -50.4mV) :
         (; )
    Random.seed!(77)
    T = Tripod(; N, VFT, param = TripodParameter(ds = lengths), kw...)
    Random.seed!(77)
    M = Multipod(; N, VFT, param = MultipodParameter(ds = lengths), kw...)
    # identical initial state (the constructors draw the same random numbers, checked here)
    @test M.v_s == T.v_s && M.v_d1 == T.v_d1 && M.v_d2 == T.v_d2
    return T, M
end

function _mpeq_compare(T, M, n; kwargs...)
    tr_T, sp_T = _mpeq_run(T, n; kwargs...)
    tr_M, sp_M = _mpeq_run(M, n; kwargs...)
    @test sp_T == sp_M
    @test all(tr_T[k] == tr_M[k] for k in 1:n)          # v_s, v_d1, v_d2, w_s, θ, tabs, fire: bit for bit
    return _mpeq_maxdiff(tr_T, tr_M), length(sp_T)
end

# ---------------------------------------------------------------------------------------------------------------------------------
# The acceptance matrix
# ---------------------------------------------------------------------------------------------------------------------------------

const MPEQ_RESULTS = Dict{String,Tuple{Float64,Int}}()

@testset "Multipod N_d = 2 == Tripod / TripodPaper (bit for bit)" begin
    for (scheme, receptors) in ((:heun, :default), (:heun, :paper), (:paper, :paper))
        tag = "$(scheme)/$(receptors)"
        @testset "$tag" begin
            for (pname, prot) in pairs(_MPEQ_PROTOCOLS)
                T, M = _mpeq_pair(scheme, receptors, [400um, 150um])
                d, nsp = _mpeq_compare(T, M, prot.n; inputs = prot.inputs, I = prot.I)
                MPEQ_RESULTS["$tag/$pname"] = (d, nsp)
                @test d == 0
            end
            for l in (400, 250, 150)
                T, M = _mpeq_pair(scheme, receptors, [l * um, l * um])
                d, nsp = _mpeq_compare(T, M, 10000; plateau_d1 = 100)
                MPEQ_RESULTS["$tag/plateau_$l"] = (d, nsp)
                @test d == 0
            end
        end
    end

    # the published-code values hold for the Multipod preset (original code: 15 spikes, plateaus 71.1 / 58.4 / 28.7 ms)
    @testset "MultipodPaper reproduces the audit numbers" begin
        @test MPEQ_RESULTS["paper/paper/step_supra"][2] == 15
        for (l, orig) in ((400, 71.1), (250, 58.4), (150, 28.7))
            M = MultipodPaper(N = 1, lengths = [l * um, l * um])
            cnt = 0
            for tt in 1:10000
                M.receptors_d1.glu[1] = tt == 150 ? 100f0 : 0f0
                SNNModels.integrate!(M, M.param, 0.1f0)
                tt >= 150 && M.v_d1[1] >= -60f0 && (cnt += 1)
            end
            @test abs(0.1 * cnt - orig) <= 2.0
        end
    end

    # many neurons (vectorised loops, several chunks), random lengths, Float32 and Float64, mouse physiology
    @testset "population of 37 neurons, random lengths, $(eltype(VFT))" for VFT in (Vector{Float32}, Vector{Float64})
        FT = eltype(VFT)
        Random.seed!(5)
        T = Tripod(N = 37; VFT, param = TripodParameter(ds = [(150um, 400um), (200um, 300um)], physiology = mouse_dend),
                   dend_syn = convert_precision(TripodDendSynapseMouse, FT))
        Random.seed!(5)
        M = Multipod(N = 37; VFT, param = MultipodParameter(ds = [(150um, 400um), (200um, 300um)], physiology = mouse_dend),
                     dend_syn = convert_precision(TripodDendSynapseMouse, FT))
        @test M.d1.l == T.d1.l && M.d2.gax == T.d2.gax && M.v_s == T.v_s
        rng = Xoshiro(9)
        X = [rand(rng) < 0.02 ? 10f0 * rand(rng, Float32) : 0f0 for _ in 1:6, _ in 1:4000]
        drive = t -> 50 <= t < 350 ? 1500f0 : 0f0        # pA: strong enough for the Heun rule (Vspike 0 mV) with mouse dendrites
        d, nsp = _mpeq_compare(T, M, 4000; inputs = (r, tt) -> X[r, tt], I = drive)
        MPEQ_RESULTS["heun/mouse/N37/$(FT)"] = (d, nsp)
        @test d == 0 && nsp > 0
        T, M = _mpeq_pair(:paper, :paper, [300um, 200um]; N = 37, VFT)
        d, nsp = _mpeq_compare(T, M, 4000; inputs = (r, tt) -> X[r, tt], I = drive)
        MPEQ_RESULTS["paper/N37/$(FT)"] = (d, nsp)
        @test d == 0 && nsp > 0
    end
end

# summary printed with the test output (protocol => (max |difference|, number of spikes))
for k in sort(collect(keys(MPEQ_RESULTS)))
    println("  multipod==tripod  ", rpad(k, 34), " max|Δ| = ", MPEQ_RESULTS[k][1], "  spikes = ", MPEQ_RESULTS[k][2])
end
