# Single-neuron validation of the Tripod / BallAndStick fixes (branch fix/sweep-bugs of SNNModels).
#
# Usage:  julia --project=<env> measure_multicompartment.jl <variant>
#   <variant> is a label stored with the data: unfixed, dtonly, glonly, fixed (Tripod)
#   and unfixedIs, dtonlyIs, glonlyIs, fixed (BallAndStick; see README.md).
#   The script only uses API that exists in every variant (integrate! on a population,
#   I / I_d (Tripod) and Is (BallAndStick) external currents, receptor input buffers).
# Output: data/mc_<variant>.jld2
#
# Measures, for two configurations ("default": library defaults, "paper": the parameter set of
# the SNNModels Tripod tests, which follows the published Tripod model):
#   f-I curve (1 s current steps, 0..3000 pA, dt = 0.1 ms), rheobase (1 pA resolution),
#   spike waveform and adaptation current (dt = 0.025 ms), ISI adaptation,
#   dendritic EPSP attenuation and NMDA plateau (single input of weight w on a dendrite),
#   back-propagated AP in the dendrite, dt dependence (dt = 0.0125, 0.025, 0.05, 0.1, 0.125 ms).

using SNNModels
using JLD2
using Statistics
@load_units

const VARIANT = ARGS[1]
const OUT = joinpath(@__DIR__, "data", "mc_$(VARIANT).jld2")

# ---------------------------------------------------------------- neuron construction
function adex_cfg(cfg)
    cfg == :default && return (AdExParameter(), PostSpike())
    return (
        AdExParameter(C = 281pF, gl = 40nS, Vr = -55.6mV, El = -70.6mV, ΔT = 2mV,
                      Vt = -50.4mV, a = 4nS, b = 80.5pA, τw = 144ms),
        PostSpike(At = 10.0mV, τA = 30.0ms, τabs = 0.1ms, up = 0.1ms),
    )
end

function make_neuron(model, cfg, N)
    adex, spike = adex_cfg(cfg)
    if model == :Tripod
        ds = cfg == :default ? [300um, 300um] : [160um, 200um]
        p = Tripod(N = N, param = TripodParameter(ds = ds), adex = adex, spike = spike)
    else
        ds = cfg == :default ? [300um] : [200um]
        p = BallAndStick(N = N, param = BallAndStickParameter(ds = ds), adex = adex, spike = spike)
    end
    reset_state!(p)
    return p
end

dendrites(p::Tripod) = (p.v_d1, p.v_d2)
dendrites(p::BallAndStick) = (p.v_d,)
dend1(p::Tripod) = p.v_d1
dend1(p::BallAndStick) = p.v_d
dend1_glu(p::Tripod) = p.receptors_d1.glu
dend1_glu(p::BallAndStick) = p.receptors_d.glu
soma_current(p::Tripod) = p.I
soma_current(p::BallAndStick) = p.Is

function reset_state!(p)
    El = p.adex.El
    p.v_s .= El
    for v in dendrites(p)
        v .= El
    end
    p.w_s .= 0
    p.θ .= p.adex.Vt
    p.tabs .= 0
    p.fire .= false
    return p
end

# ---------------------------------------------------------------- protocols
"Current step `I` (vector, one per neuron) for `T` ms at `dt`; returns spike times per neuron."
function current_step(model, cfg, I::Vector{Float32}, T, dt; record = false)
    p = make_neuron(model, cfg, length(I))
    soma_current(p) .= I
    st = [Float32[] for _ in I]
    n = round(Int, T / dt)
    vs = record ? zeros(Float32, length(I), n) : nothing
    vd = record ? zeros(Float32, length(I), n) : nothing
    w = record ? zeros(Float32, length(I), n) : nothing
    for k = 1:n
        SNNModels.integrate!(p, p.param, Float32(dt))
        for i in eachindex(I)
            p.fire[i] && push!(st[i], k * dt)
        end
        if record
            vs[:, k] .= p.v_s
            vd[:, k] .= dend1(p)
            w[:, k] .= p.w_s
        end
    end
    return st, vs, vd, w
end

function rheobase(model, cfg; dt = 0.1, T = 1000.0)
    lo, hi = 0.0f0, 4000.0f0
    # coarse scan with many neurons at once, then bisection
    grid = collect(Float32, 0:10:4000)
    st, = current_step(model, cfg, grid, T, dt)
    k = findfirst(s -> !isempty(s), st)
    isnothing(k) && return NaN32
    lo, hi = grid[max(k - 1, 1)], grid[k]
    fine = collect(Float32, lo:1:hi)
    st, = current_step(model, cfg, fine, T, dt)
    return fine[findfirst(s -> !isempty(s), st)]
end

"Single glutamatergic input of weight `w` on dendrite 1 at t = 20 ms, no current; 300 ms at dt."
function epsp(model, cfg, ws::Vector{Float32}; dt = 0.1, T = 300.0)
    p = make_neuron(model, cfg, length(ws))
    n = round(Int, T / dt)
    vs = zeros(Float32, length(ws), n)
    vd = zeros(Float32, length(ws), n)
    st = [Float32[] for _ in ws]
    for k = 1:n
        k == round(Int, 20 / dt) && (dend1_glu(p) .+= ws)
        SNNModels.integrate!(p, p.param, Float32(dt))
        vs[:, k] .= p.v_s
        vd[:, k] .= dend1(p)
        for i in eachindex(ws)
            p.fire[i] && push!(st[i], k * dt)
        end
    end
    return vs, vd, st
end

isis(st) = length(st) > 1 ? diff(st) : Float32[]

function measure(model, cfg)
    res = Dict{String,Any}()
    # f-I curve, dt = 0.1 ms
    Igrid = collect(Float32, 0:25:3000)
    st, = current_step(model, cfg, Igrid, 1000.0, 0.1)
    res["fI_I"] = Igrid
    res["fI_rate"] = Float32[length(s) for s in st]                   # Hz (1 s)
    res["fI_first_isi"] = Float32[length(s) > 1 ? s[2] - s[1] : NaN32 for s in st]
    res["fI_latency"] = Float32[isempty(s) ? NaN32 : s[1] for s in st]
    res["rheobase"] = rheobase(model, cfg)
    # waveform, adaptation, ISI adaptation at fixed currents, dt = 0.025 ms
    Iw = Float32[1000, 1500, 2000]
    st, vs, vd, w = current_step(model, cfg, Iw, 500.0, 0.025; record = true)
    res["wave_I"] = Iw
    res["wave_dt"] = 0.025f0
    res["wave_vs"] = vs[:, 1:4:end]          # 0.1 ms resolution
    res["wave_vd"] = vd[:, 1:4:end]
    res["wave_w"] = w[:, 1:4:end]
    res["wave_st"] = st
    res["isi"] = [isis(s) for s in st]
    res["adaptation_index"] = Float32[length(s) > 2 ? isis(s)[end] / isis(s)[1] : NaN32 for s in st]
    # back-propagated AP: peak dendritic depolarisation within 3 ms after each spike
    bap = Float32[]
    for (i, s) in enumerate(st), t in s
        k = round(Int, t / 0.025)
        k + 120 <= size(vd, 2) || continue
        push!(bap, maximum(vd[i, k:(k+120)]) - vd[i, max(k - 1, 1)])
    end
    res["bap_amplitude"] = bap
    # dendritic EPSPs and NMDA plateau
    ws = Float32[0.5, 1, 2, 4, 8, 16, 32, 64]
    vs, vd, st = epsp(model, cfg, ws)
    rest_s, rest_d = vs[:, round(Int, 19 / 0.1)], vd[:, round(Int, 19 / 0.1)]
    res["epsp_w"] = ws
    res["epsp_soma"] = vec(maximum(vs, dims = 2)) .- rest_s
    res["epsp_dend"] = vec(maximum(vd, dims = 2)) .- rest_d
    res["epsp_spikes"] = Float32[length(s) for s in st]
    res["plateau_ms"] = Float32[count(>(-50.0f0), vd[i, :]) * 0.1 for i in eachindex(ws)]
    res["epsp_vs"] = vs[:, 1:5:end]          # 0.5 ms resolution
    res["epsp_vd"] = vd[:, 1:5:end]
    # dt dependence: spike times of 500 ms current steps
    dts = [0.0125, 0.025, 0.05, 0.1, 0.125]
    Idt = Float32[800, 1000, 1200, 1500, 2000, 2500]
    res["dt_values"] = dts
    res["dt_I"] = Idt
    res["dt_st"] = [first(current_step(model, cfg, Idt, 500.0, dt)) for dt in dts]
    Ifi = collect(Float32, 0:50:3000)
    res["dt_fI_I"] = Ifi
    res["dt_fI_rate"] = [Float32[length(s) for s in first(current_step(model, cfg, Ifi, 1000.0, dt))] for dt in dts]
    res["dt_rheobase"] = [rheobase(model, cfg; dt) for dt in (0.05, 0.1, 0.125)]
    return res
end

data = Dict{String,Any}("variant" => VARIANT)
for model in (:Tripod, :BallAndStick), cfg in (:default, :paper)
    @info "measuring" VARIANT model cfg
    data["$(model)_$(cfg)"] = measure(model, cfg)
end
mkpath(dirname(OUT))
jldsave(OUT; data)
@info "saved" OUT
