# Benchmark of Multipod against Tripod (not part of the test suite).
#
# Usage (any env that develops SNNModels):
#   julia --project=<env> -t 1 bench/multipod_bench.jl
#   julia --project=<env> -t 4 bench/multipod_bench.jl
#
# Two measurements, Float32, N = 2000 neurons, dt = 0.1 ms:
#   1. sim! of 1 s (10000 steps) of a model with Poisson input (PoissonLayer stimuli, 20 Hz x 100 sources) onto every
#      compartment (:s and each dendrite), for Tripod and Multipod with 2, 4 and 6 dendrites. Reported: ns per neuron-step of
#      the whole sim! (stimuli included) and the bytes allocated by the timed sim!.
#   2. integrate! of the population alone (receptor buffers filled from pre-drawn Poisson events), for the same models and for
#      an array-layout variant of Multipod (dendrites in Vectors, number of dendrites known at run time only; the layout that
#      was the fallback of the design). The array variant is checked to give the same state bit for bit before it is timed.
# With -t 4 the populations run in threaded chunks (`neurons_per_thread = 250`); with -t 1 on the serial path.

using SNNModels
using Random
using Printf
using SNNModels.UnPack: @unpack
@load_units

const N = 2000
const DT = 0.1f0
const STEPS = 10_000
const NPT = Threads.nthreads() > 1 ? 250 : 0

# ---------------------------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------------------------

# dendrite lengths: those of Tripod (400, 150 um) first, so that Multipod D=2 is the Tripod of the comparison
const LENGTHS = [400um, 150um, 300um, 250um, 200um, 350um]

# a population with D dendrites (D = 0: Tripod) of fixed lengths; `paper`: the published-code preset and PaperScheme
function make_population(D::Int; paper = false)
    Random.seed!(1)
    if D == 0
        return paper ? TripodPaper(N = N, name = "P") : Tripod(N = N, name = "P", param = TripodParameter(ds = LENGTHS[1:2]))
    end
    return paper ? MultipodPaper(N = N, name = "P", lengths = LENGTHS[1:D]) :
           Multipod(N = N, name = "P", param = MultipodParameter(ds = LENGTHS[1:D]))
end

ndend(P) = P isa Tripod ? 2 : ndendrites(P)
targets(P) = (:s, (Symbol(:d, k) for k in 1:ndend(P))...)

# Poisson input onto every compartment, weights large enough to make the neurons fire at a few Hz
function make_model(P)
    stims = NamedTuple(Symbol(:noise_, t) => Stimulus(PoissonLayer(20Hz, N = 100), P, :glu, t; conn = (μ = 4.0, ρ = 0.5))
                       for t in targets(P))
    return compose(; P, stims..., silent = true)
end

label(D) = D == 0 ? "Tripod" : "Multipod D=$D"

# ---------------------------------------------------------------------------------------------------------------------------------
# 1. sim! of 1 s
# ---------------------------------------------------------------------------------------------------------------------------------

function bench_sim(D; reps = 3)
    model = make_model(make_population(D))
    sim!(; model, duration = 20ms, dt = DT, neurons_per_thread = NPT)          # compile
    times = Float64[]
    bytes = 0
    for _ in 1:reps
        stats = @timed sim!(; model, duration = STEPS * DT, dt = DT, neurons_per_thread = NPT)
        push!(times, stats.time)
        bytes = stats.bytes
    end
    rate = count_rate(model)
    return minimum(times) / (STEPS * N) * 1e9, bytes, rate
end

# mean firing rate (Hz) in a short extra run with the spikes monitored
function count_rate(model)
    P = model.pop.P
    monitor!(P, [:fire]; monitor_rate = 100Hz)
    sim!(; model, duration = 500ms, dt = DT, neurons_per_thread = NPT)
    r = sum(length, spiketimes(P)) / (N * 0.5)
    clear_records!(P)
    return r
end

# ---------------------------------------------------------------------------------------------------------------------------------
# 2. integrate! alone, with the array-layout variant
# ---------------------------------------------------------------------------------------------------------------------------------

# Array layout: the dendritic quantities in Vectors indexed by a run-time k, and the axial currents summed in a scratch vector
# (k outer, neurons inner, so that the neuron loops still vectorise). The arithmetic and its order are those of Multipod.
struct ArrayPod{P,VV,DV,SV,RV}
    base::P            # the Multipod whose soma, synapse models and work arrays are used
    D::Int
    v_d::VV            # Vector of the D potential vectors
    dends::DV          # Vector of the D Dendrites
    synvars::SV        # Vector of the D synaptic variable sets
    receptors::RV      # Vector of the D receptor buffers
    acc::Vector{Float32}
end

function ArrayPod(M::Multipod)
    D = ndendrites(M)
    get(pre) = [getfield(M, Symbol(pre, k)) for k in 1:D]
    return ArrayPod(M, D, get(:v_d), get(:d), get(:synvars_d), get(:receptors_d), zeros(Float32, Int(M.N)))
end

function array_integrate!(A::ArrayPod, dt::Float32)
    p = A.base
    n_abs = max(1, round(Int, p.spike.τabs / dt))
    n_ref = max(1, round(Int, p.spike.up / dt)) + n_abs
    SNNModels.foreach_chunk_c(Int(p.N)) do c, rng
        array_chunk!(A, dt, rng, n_abs, n_ref)
    end
end

function array_chunk!(A::ArrayPod, dt, rng, n_abs, n_ref)
    p = A.base
    D = A.D
    @unpack Δv, Δv_temp, fire = p
    SNNModels.update_synapses!(p, p.soma_syn, p.receptors_s, p.synvars_s, dt, rng)
    for k in 1:D
        SNNModels.update_synapses!(p, p.dend_syn, A.receptors[k], A.synvars[k], dt, rng)
    end
    @inbounds for k in 1:(D + 2), i in rng
        Δv[i, k] = 0f0
        Δv_temp[i, k] = 0f0
    end
    @inbounds for i in rng
        fire[i] = false
    end
    SNNModels._tripod_clamp_soma!(p, rng, n_abs)
    array_stage!(A, Δv, dt, rng)
    @inbounds for k in 1:(D + 2), i in rng
        Δv_temp[i, k] = Δv[i, k]
    end
    SNNModels._tripod_hold_soma!(p, rng, n_abs)
    array_stage!(A, Δv, dt, rng)
    array_state!(A, dt, rng, n_abs, n_ref)
end

function array_stage!(A::ArrayPod, Δv, dt, rng)
    p = A.base
    D = A.D
    @unpack v_s, I_d, I, w_s, θ, is, v_pred, adex, soma_syn, dend_syn, synvars_s = p
    @unpack C, gl, El, ΔT, a, τw = adex
    acc = A.acc
    @inbounds @simd ivdep for i in rng
        v_pred[i, 1] = v_s[i] + Δv[i, 1] * dt
    end
    for k in 1:D
        vd = A.v_d[k]
        @inbounds @simd ivdep for i in rng
            v_pred[i, k + 1] = vd[i] + Δv[i, k + 1] * dt
        end
    end
    @views SNNModels._synaptic_current!(p, soma_syn, synvars_s, v_pred[:, 1], is[:, 1], rng)
    for k in 1:D
        @views SNNModels._synaptic_current!(p, dend_syn, A.synvars[k], v_pred[:, k + 1], is[:, k + 1], rng)
    end
    @inbounds for k in 1:(D + 1)
        @simd ivdep for i in rng
            is[i, k] = clamp(is[i, k], -1500f0, 1500f0)
        end
    end
    # soma: start of the left-to-right sum (adaptation + synapses), then one pass per dendrite
    @inbounds @simd ivdep for i in rng
        acc[i] = muladd(Δv[i, D + 2], dt, w_s[i]) + is[i, 1]
    end
    for k in 1:D
        d = A.dends[k]
        gax, gm, Eld, Cd = d.gax, d.gm, d.El, d.C
        @inbounds @simd ivdep for i in rng
            vs = v_pred[i, 1]
            vd = v_pred[i, k + 1]
            ic = (vs - vd) * gax[i]
            Δv[i, k + 1] = ((muladd(Eld[i] - vd, gm[i], -is[i, k + 1]) + ic) + I_d[i]) / Cd[i]
            acc[i] = acc[i] + ic
        end
    end
    @inbounds @simd ivdep for i in rng
        vs = v_pred[i, 1]
        ws = muladd(Δv[i, D + 2], dt, w_s[i])
        e = SNNModels.exp256((vs - θ[i]) / ΔT)
        Δv[i, 1] = muladd(gl, muladd(ΔT, e, El - vs), I[i] - acc[i]) / C
        Δv[i, D + 2] = muladd(a, vs - El, -ws) / τw
    end
end

function array_state!(A::ArrayPod, dt, rng, n_abs, n_ref)
    p = A.base
    D = A.D
    @unpack v_s, w_s, fire, θ, tabs, Δv, Δv_temp, spike, adex, Vspike = p
    @unpack AP_membrane, At, τA = spike
    @unpack Vr, Vt, b = adex
    nabs_f = Float32(n_abs)
    nref_f = Float32(n_ref)
    @inbounds @simd ivdep for i in rng
        t1 = tabs[i] - 1
        th1 = θ[i] + dt * (Vt - θ[i]) / τA
        bp = t1 > nabs_f
        rf = !bp & (t1 > 0)
        act = t1 <= 0
        vs0 = v_s[i]
        ws0 = w_s[i]
        dv1 = Δv[i, 1]
        vsc = ifelse(bp, AP_membrane, Vr)
        spk = act & (vs0 + dv1 * dt >= Vspike)
        keep = act & !spk
        vs_h = Float32(vs0 + 0.5 * dt * (Δv_temp[i, 1] + dv1))
        ws_h = Float32(ws0 + 0.5 * dt * (Δv_temp[i, D + 2] + Δv[i, D + 2]))
        fire[i] = spk
        Δv[i, 1] = ifelse(spk, AP_membrane - vs0, dv1)
        v_s[i] = ifelse(bp | rf, vsc, ifelse(spk, AP_membrane, ifelse(keep, vs_h, vs0)))
        w_s[i] = ifelse(spk, ws0 + b, ifelse(keep, ws_h, ws0))
        θ[i] = ifelse(spk, th1 + At, th1)
        tabs[i] = ifelse(spk, nref_f, t1)
    end
    for k in 1:D
        vdk = A.v_d[k]
        @inbounds @simd ivdep for i in rng
            vd = vdk[i]
            vd_h = Float32(vd + 0.5 * dt * (Δv_temp[i, k + 1] + Δv[i, k + 1]))
            vdk[i] = ifelse(fire[i], vd, vd_h)
        end
    end
end

# pre-drawn Poisson events (20 Hz x 100 sources x weight 2 per compartment, as a per-neuron Bernoulli draw per step)
function draw_events(ncomp, steps; seed = 3)
    rng = Xoshiro(seed)
    p = 100 * 20e-3 * DT * 0.5      # expected events per neuron and step (ρ = 0.5 of 100 sources at 20 Hz)
    return [findall(rand(rng, N) .< p) for _ in 1:ncomp, _ in 1:steps]
end

buffers(P) = (P.receptors_s.glu, (getfield(P, Symbol(:receptors_d, k)).glu for k in 1:ndend(P))...)
buffers(A::ArrayPod) = (A.base.receptors_s.glu, (r.glu for r in A.receptors)...)

function drive!(bufs, ev, tt)
    for (c, buf) in enumerate(bufs)
        @inbounds for i in ev[c, tt]
            buf[i] += 4f0
        end
    end
end

function run_integrate!(P, ev, steps)
    bufs = buffers(P)
    for tt in 1:steps
        drive!(bufs, ev, tt)
        P isa ArrayPod ? array_integrate!(P, DT) : SNNModels.integrate!(P, P.param, DT)
    end
end

function bench_integrate(make; steps = 2000, reps = 3)
    P = make()
    ev = draw_events(length(buffers(P)), steps)
    SNNModels.set_neurons_per_thread!(NPT)
    run_integrate!(P, ev, 50)                                            # compile
    times = Float64[]
    bytes = 0
    for _ in 1:reps
        stats = @timed run_integrate!(P, ev, steps)
        push!(times, stats.time)
        bytes = stats.bytes
    end
    SNNModels.set_neurons_per_thread!(0)
    return minimum(times) / (steps * N) * 1e9, bytes / steps
end

# the array variant gives the state of Multipod bit for bit (so that its timing is a like-for-like comparison)
function check_array_variant(D)
    M = make_population(D)
    A = ArrayPod(make_population(D))
    ev = draw_events(D + 1, 1000)
    SNNModels.set_neurons_per_thread!(NPT)
    run_integrate!(M, ev, 1000)
    run_integrate!(A, ev, 1000)
    SNNModels.set_neurons_per_thread!(0)
    return M.v_s == A.base.v_s && all(getfield(M, Symbol(:v_d, k)) == A.v_d[k] for k in 1:D) && M.w_s == A.base.w_s
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------------------------

println("Multipod benchmark: N = $N, Float32, dt = $DT ms, threads = $(Threads.nthreads()), neurons_per_thread = $NPT")
println()
println("1. sim! of 1 s, Poisson input on every compartment")
@printf("   %-16s %14s %14s %12s %10s\n", "model", "ns/neuron-step", "per dendrite", "alloc (B)", "rate (Hz)")
sim_results = Dict{Int,Float64}()
for D in (0, 2, 4, 6)
    t, bytes, rate = bench_sim(D)
    sim_results[D] = t
    nd = D == 0 ? 2 : D
    @printf("   %-16s %14.2f %14.2f %12d %10.2f\n", label(D), t, t / (nd + 1), bytes, rate)
end
@printf("   Multipod D=2 / Tripod: %.3f\n", sim_results[2] / sim_results[0])

println()
println("2. integrate! alone (2000 steps), generated types vs array layout")
@printf("   %-24s %14s %16s\n", "population", "ns/neuron-step", "alloc/step (B)")
int_results = Dict{String,Float64}()
for D in (0, 2, 4, 6)
    t, a = bench_integrate(() -> make_population(D))
    int_results[label(D)] = t
    @printf("   %-24s %14.2f %16.1f\n", label(D), t, a)
end
for D in (2, 4, 6)
    ok = check_array_variant(D)
    t, a = bench_integrate(() -> ArrayPod(make_population(D)))
    int_results["array D=$D"] = t
    @printf("   %-24s %14.2f %16.1f   (bit-identical to Multipod: %s)\n", "array layout D=$D", t, a, ok)
end
for D in (0, 2, 4)
    t, a = bench_integrate(() -> make_population(D; paper = true))
    int_results[label(D) * " paper"] = t
    @printf("   %-24s %14.2f %16.1f\n", label(D) * " PaperScheme", t, a)
end
@printf("   Multipod D=2 / Tripod: %.3f (HeunScheme), %.3f (PaperScheme)\n", int_results["Multipod D=2"] / int_results["Tripod"],
        int_results["Multipod D=2 paper"] / int_results["Tripod paper"])
for D in (2, 4, 6)
    @printf("   array D=%d / Multipod D=%d: %.3f\n", D, D, int_results["array D=$D"] / int_results["Multipod D=$D"])
end
