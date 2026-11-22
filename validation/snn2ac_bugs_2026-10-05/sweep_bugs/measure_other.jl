# Before/after measurements for the fixes that change numbers, other than Tripod/BallAndStick.
#
# Usage:  julia --project=<env> measure_other.jl <variant>     (variant = unfixed | fixed)
# Output: data/other_<variant>.jld2
# Each block is wrapped in try/catch: code paths that fail in the unfixed version store the
# error message instead of numbers.

using SpikingNeuralNetworks
using SNNModels
using JLD2
using Statistics
using Random
SNN.@load_units
const SM = SNNModels

const VARIANT = ARGS[1]
const OUT = joinpath(@__DIR__, "data", "other_$(VARIANT).jld2")
data = Dict{String,Any}("variant" => VARIANT)

function guarded(f, key)
    try
        data[key] = f()
    catch e
        data[key] = Dict("error" => sprint(showerror, e)[1:min(end, 300)])
        @warn "failed" key data[key]["error"]
    end
end

# ---- HH and MorrisLecar: spike flags per action potential, f-I ----------------------------
upcross(v, θ) = count(k -> v[k-1] <= θ < v[k], 2:length(v))
guarded("HH") do
    I = collect(Float32, 0:50:1500)
    H = HH(N = length(I)); H.ge .= 0; H.gi .= 0; H.I .= I
    dt = 0.01f0; n = round(Int, 1000 / dt)
    flags = zeros(Int, length(I)); v = zeros(Float32, length(I), n)
    for k = 1:n
        SM.integrate!(H, H.param, dt); flags .+= H.fire; v[:, k] .= H.v
    end
    aps = [upcross(v[i, :], -20.0f0) for i in eachindex(I)]
    Dict("I" => I, "flags" => flags, "aps" => aps, "v_example" => v[findfirst(==(500.0f0), I), 1:10:20000])
end
guarded("MorrisLecar") do
    M_ = MorrisLecar(N = 1)
    dt = 0.05f0; flags = 0; v = Float32[]
    for rep = 1:3, (I, T) in ((100.0f0, 50.0f0), (0.0f0, 300.0f0))
        M_.I .= I
        for _ = 1:round(Int, T / dt)
            SM.integrate!(M_, M_.param, dt); flags += M_.fire[1]; push!(v, M_.v[1])
        end
    end
    Dict("flags" => flags, "aps" => upcross(v, 20.0f0), "v" => v[1:10:end])
end

# readers of the stored fields of an event-driven rule (SNNModels >= 1.10); the older variants
# (unfixed, 1.8.4) have no accessor and store the values
wval(syn) = isdefined(SM, :plasticity_value) ? SM.plasticity_value(syn, :W) : syn.W
xval(syn, T) = isdefined(SM, :plasticity_value) ? SM.plasticity_value(syn, :LTPVars, :x, T)[1] : syn.LTPVars.x[1]

# ---- vSTDP: dt (in)dependence and the 0 mV initial traces ----------------------------------
function vstdp_pair(dt; Tsim = 30.0f0, vpost = -40.0f0)
    E = Identity(N = 1); P = AdEx(N = 1)
    syn = SpikingSynapse(E, P, :glu; conn = (p = 1.0, μ = 1.0), LTPParam = vSTDPParameter())
    W0 = wval(syn)[1]; T = SM.Time()
    xs = Float32[]
    for k = 1:round(Int, Tsim / dt)
        E.fire .= (k == 1); P.v .= vpost
        SM.plasticity!(syn, syn.LTPParam, syn.LTPVars, Float32(dt), T)
        push!(xs, xval(syn, T))
    end
    return wval(syn)[1] - W0, maximum(xs)
end
guarded("vSTDP") do
    res = Dict{String,Any}()
    for dt in (0.125, 0.0625, 0.025)
        dw, xmax = vstdp_pair(dt); dwr, _ = vstdp_pair(dt; vpost = -70.6f0)
        res["dt_$(dt)"] = Dict("dw_depolarised" => dw, "x_peak" => xmax, "dw_at_rest" => dwr)
    end
    # silent AdEx target, 20 Hz Poisson input, 50 ms of train!: mean weight change
    Random.seed!(7)
    Pre = SNN.Poisson(N = 100, param = PoissonParameter(20Hz))
    Post = AdEx(N = 20)
    Post.v .= -70.6f0
    syn = SpikingSynapse(Pre, Post, :glu; conn = (p = 0.2, μ = 1.0), LTPParam = vSTDPParameter())
    W0 = copy(wval(syn))
    train!([Pre, Post], [syn]; duration = 50ms)
    res["silent_target_mean_dw"] = mean(wval(syn) .- W0)
    res
end

# ---- AggregateScaling at two dt --------------------------------------------------------------
function aggregate(dt; Tsim = 2000.0f0)
    P = Identity(N = 5); E = IF(N = 2)
    S = SpikingSynapse(P, E, :ge; conn = (p = 1.0, μ = 2.0))
    A = AggregateScaling(E, [S]; param = AggregateScalingParameter(E.N, 50Hz))
    T = SM.Time(); nper = round(Int, 10.0 / dt)
    y, WT, Wsum = Float32[], Float32[], Float32[]
    for k = 1:round(Int, Tsim / dt)
        update_time!(T, Float32(dt))
        E.fire .= (k % nper == 0)                       # 100 Hz postsynaptic firing
        SM.forward!(A, A.param, Float32(dt), T)
        SM.plasticity!(A, A.param, Float32(dt), T)
        if k % round(Int, 10 / dt) == 0
            push!(y, A.y[1]); push!(WT, A.WT[1])
            push!(Wsum, sum(S.W[S.index[S.rowptr[1]:(S.rowptr[2]-1)]]))
        end
    end
    Dict("y" => y, "WT" => WT, "Wsum" => Wsum, "t" => collect(10.0:10.0:Tsim))
end
guarded("AggregateScaling") do
    Dict("dt_0.125" => aggregate(0.125), "dt_0.05" => aggregate(0.05))
end

# ---- HetRec at two dt (no spikes: deterministic) ----------------------------------------------
function hetrec(dt)
    p = HetRecParameter(Nd = 2, overlap = 0.5f0, rate = SM.Uniform(0.0f0, 1.0f-9))
    Random.seed!(3)
    pop = Population(p; N = 5)
    pop.v_d .= range(1.0f0, 10.0f0, length = length(pop.v_d)); pop.τd .= 1.0f6
    trace = Float32[]
    for k = 1:round(Int, 300 / dt)
        SM.integrate!(pop, pop.param, Float32(dt))
        k % round(Int, 1 / dt) == 0 && push!(trace, pop.trace[1])
    end
    Dict("trace" => trace, "v_s" => copy(pop.v_s), "trace_end" => copy(pop.trace))
end
guarded("HetRec") do
    Dict("dt_0.125" => hetrec(0.125), "dt_0.05" => hetrec(0.05))
end

# ---- Confavreux2025Synapse: conductance jump and PSP at two dt ------------------------------
guarded("Confavreux") do
    res = Dict{String,Any}()
    for dt in (0.125f0, 0.05f0)
        E = IF(N = 1, synapse = Confavreux2025Synapse())
        E.v .= -70.0f0
        E.receptors.glu .= 1.0f0
        vmax = -100.0f0; g0 = NaN32
        for k = 1:round(Int, 100 / dt)
            SM.integrate!(E, E.param, dt)
            k == 1 && (g0 = E.synvars.gAMPA[1])
            vmax = max(vmax, E.v[1])
        end
        res["dt_$(dt)"] = Dict("gAMPA_after_spike" => g0, "psp_peak" => vmax + 70.0f0)
    end
    res
end

# ---- interpolated_record time axis -----------------------------------------------------------
guarded("record_axis") do
    E = IF(N = 5)
    stim = Stimulus(PoissonFixed(rate = 2kHz), E, :ge)
    model = compose(; E, stim, silent = true)
    sim!(model, 2s)
    monitor!(E, [:v], sr = 10Hz)
    sim!(model, 2s)
    _, r = SM.interpolated_record(E, :v)
    F = IF(N = 2)
    monitor!(F, [:v], sr = 1kHz)
    sim!([F]; duration = 10.5ms)
    _, r2 = SM.interpolated_record(F, :v)
    Dict("r_10Hz_2to4s" => collect(r), "n_10Hz" => size(getvariable(E, :v), 2),
         "r_1kHz_10.5ms" => collect(r2), "n_1kHz" => size(getvariable(F, :v), 2))
end

# ---- STDP kernel measured by SNNPlots.stdp_test ------------------------------------------------
guarded("stdp_kernel") do
    ΔT = [-50.0, -30.0, -20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    p = STDPGerstner()
    Dict("ΔT" => ΔT, "dw" => [SNNPlots.stdp_test(p; ΔT = d) for d in ΔT],
         "A_pre" => p.A_pre, "A_post" => p.A_post, "τpre" => p.τpre, "τpost" => p.τpost)
end

# ---- Rate input accumulation under RateSynapse ------------------------------------------------
guarded("Rate_g") do
    Random.seed!(5)
    R = Rate(N = 50)
    RR = RateSynapse(R, R; μ = 1.0, p = 0.5)
    monitor!(R, [:x, :r], sr = 1kHz)
    sim!([R], [RR]; duration = 220ms)
    x = getvariable(R, :x); r = getvariable(R, :r)
    Dict("mean_abs_x_20ms" => mean(abs.(x[:, 21])), "mean_abs_x_220ms" => mean(abs.(x[:, end])),
         "saturated_220ms" => mean(abs.(r[:, end]) .> 0.99), "saturated_20ms" => mean(abs.(r[:, 21]) .> 0.99))
end

# ---- AdditiveNorm: summed input weight after one normalisation --------------------------------
guarded("AdditiveNorm") do
    Random.seed!(9)
    E = IF(N = 20)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.5, μ = 2.0))
    N_ = SynapseNormalization([EE]; param = AdditiveNorm(τ = 10ms))
    W0 = copy(N_.W0)
    EE.W .*= 1.5f0
    SM.plasticity!(N_, N_.param)
    sums = zeros(Float32, 20)
    for i = 1:20, k = EE.rowptr[i]:(EE.rowptr[i+1]-1)
        sums[i] += EE.W[EE.index[k]]
    end
    Dict("W0" => W0, "after" => sums)
end

# ---- BalancedStimulus default configuration --------------------------------------------------
guarded("Balanced") do
    P = IF(N = 50)
    stim = Stimulus(BalancedParameter(r0 = 1kHz, kIE = 0.5), P, :ge)
    T = SM.Time(); dt = 0.125f0
    for _ = 1:4000
        SM.stimulate!(stim, stim.param, T, dt)
    end
    Dict("exc_rate_per_ms" => mean(stim.ge) / (4000 * dt), "inh_rate_per_ms" => mean(stim.gi) / (4000 * dt),
         "neurons_driven" => count(>(0), stim.ge), "same_buffer" => stim.ge === stim.gi)
end

mkpath(dirname(OUT))
jldsave(OUT; data)
@info "saved" OUT
