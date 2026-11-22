# Figures of the Tripod / BallAndStick comparison, regenerated from data/mc_<variant>.jld2.
# Usage: julia --project=<env with SNNPlots, CairoMakie, JLD2> plot_multicompartment.jl
# Output: figures/mc_fI.png, figures/mc_traces.png, figures/mc_epsp.png, figures/mc_dt.png

using SNNPlots, CairoMakie, JLD2, Statistics
SNNPlots.@makie_default

const D = joinpath(@__DIR__, "data")
const F = joinpath(@__DIR__, "figures")
mkpath(F)
variants(model) = model == "Tripod" ? ["unfixed", "dtonly", "glonly", "fixed"] :
                                      ["unfixedIs", "dtonly", "glonly", "fixed"]
label(v) = Dict("unfixed" => "unfixed (1.8.4)", "unfixedIs" => "unfixed (1.8.4, Is on)",
                "dtonly" => "Heun fix only", "glonly" => "gl fix only", "fixed" => "all fixes")[v]
cols = Dict("unfixed" => okabe_ito_10[1], "unfixedIs" => okabe_ito_10[1], "dtonly" => okabe_ito_10[2],
            "glonly" => okabe_ito_10[3], "fixed" => okabe_ito_10[6])
styles = Dict("unfixed" => :solid, "unfixedIs" => :solid, "dtonly" => :dash, "glonly" => :solid, "fixed" => :dash)
data = Dict(v => JLD2.load(joinpath(D, "mc_$(v).jld2"), "data") for v in ("unfixed", "unfixedIs", "dtonly", "glonly", "fixed"))
panels = [("Tripod", "default"), ("Tripod", "paper"), ("BallAndStick", "default"), ("BallAndStick", "paper")]

# ---------------------------------------------------------------- f-I curves
fig = Figure(size = (900, 650))
for (k, (model, cfg)) in enumerate(panels)
    ax = Axis(fig[(k - 1) ÷ 2 + 1, (k - 1) % 2 + 1], xlabel = "somatic current (pA)",
              ylabel = "rate (Hz)", title = "$model, $cfg parameters")
    for v in variants(model)
        d = data[v]["$(model)_$(cfg)"]
        lines!(ax, d["fI_I"], d["fI_rate"], color = cols[v], linestyle = styles[v], label = label(v), linewidth = 2)
        vlines!(ax, [d["rheobase"]], color = cols[v], linestyle = :dot, linewidth = 1)
    end
    k == 1 && axislegend(ax, position = :lt)
end
save(joinpath(F, "mc_fI.png"), fig)

# ---------------------------------------------------------------- traces: v_s, v_d1, w_s, ISIs
fig = Figure(size = (1000, 800))
for (k, (model, cfg)) in enumerate([("Tripod", "default"), ("Tripod", "paper")])
    ax1 = Axis(fig[1, k], ylabel = "V soma (mV)", title = "$model, $cfg, 1500 pA")
    ax2 = Axis(fig[2, k], ylabel = "V dendrite 1 (mV)")
    ax3 = Axis(fig[3, k], ylabel = "w (pA)", xlabel = "time (ms)")
    ax4 = Axis(fig[4, k], ylabel = "ISI (ms)", xlabel = "spike index")
    for v in ("unfixed", "fixed")
        d = data[v]["$(model)_$(cfg)"]
        t = (1:size(d["wave_vs"], 2)) .* 0.1
        sel = t .<= 150
        lines!(ax1, t[sel], d["wave_vs"][2, sel], color = cols[v], label = label(v))
        lines!(ax2, t[sel], d["wave_vd"][2, sel], color = cols[v])
        lines!(ax3, t, d["wave_w"][2, :], color = cols[v])
        isi = d["isi"][2]
        scatterlines!(ax4, 1:length(isi), isi, color = cols[v], markersize = 5)
    end
    k == 1 && axislegend(ax1, position = :rb)
end
save(joinpath(F, "mc_traces.png"), fig)

# ---------------------------------------------------------------- EPSPs and NMDA plateau
fig = Figure(size = (1000, 600))
for (k, model) in enumerate(("Tripod", "BallAndStick"))
    ax = Axis(fig[1, k], xscale = log2, xlabel = "input weight", ylabel = "peak depolarisation (mV)",
              title = "$model (default): single dendritic input")
    for v in (variants(model)[1], "fixed")
        d = data[v]["$(model)_default"]
        scatterlines!(ax, d["epsp_w"], d["epsp_dend"], color = cols[v], label = "$(label(v)), dendrite")
        scatterlines!(ax, d["epsp_w"], d["epsp_soma"], color = cols[v], linestyle = :dash, marker = :rect,
                      label = "$(label(v)), soma")
    end
    axislegend(ax, position = :lt)
    ax2 = Axis(fig[2, k], xlabel = "time (ms)", ylabel = "V dendrite (mV)", title = "w = 32 and 64")
    for v in (variants(model)[1], "fixed"), (i, w) in ((7, 32), (8, 64))
        d = data[v]["$(model)_default"]
        t = (1:size(d["epsp_vd"], 2)) .* 0.5
        lines!(ax2, t, d["epsp_vd"][i, :], color = cols[v], linestyle = w == 32 ? :solid : :dash)
    end
    hlines!(ax2, [-50], color = :gray, linestyle = :dot)
end
save(joinpath(F, "mc_epsp.png"), fig)

# ---------------------------------------------------------------- dt dependence
devmax(st, ref) = maximum([isempty(a) || isempty(b) ? 0.0 : maximum(abs.(a[1:min(end, length(b))] .- b[1:min(end, length(a))]))
                           for (a, b) in zip(st, ref)])
fig = Figure(size = (1500, 650))
for (k, (model, cfg)) in enumerate(panels)
    axB = Axis(fig[1, k], xlabel = "dt (ms)", ylabel = "max spike-time error (ms)", yscale = log10,
               title = "$model, $cfg\nvs fixed code at dt = 0.0125 ms")
    axA = Axis(fig[2, k], xlabel = "dt (ms)", ylabel = "max spike-time change (ms)", yscale = log10,
               title = "vs the same code at dt = 0.0125 ms")
    ref = data["fixed"]["$(model)_$(cfg)"]["dt_st"][1]
    for v in variants(model)
        d = data[v]["$(model)_$(cfg)"]
        dts = d["dt_values"][1:4]           # dt = 0.125 shown in the text (refractory artefact)
        errB = [max(devmax(d["dt_st"][j], ref), 1e-3) for j in 1:4]
        errA = [max(devmax(d["dt_st"][j], d["dt_st"][1]), 1e-3) for j in 2:4]
        scatterlines!(axB, dts, errB, color = cols[v], linestyle = styles[v], label = label(v))
        scatterlines!(axA, dts[2:4], errA, color = cols[v], linestyle = styles[v], label = label(v))
    end
    k == 1 && axislegend(axB, position = :rc)
end
save(joinpath(F, "mc_dt.png"), fig)
@info "figures written" F
