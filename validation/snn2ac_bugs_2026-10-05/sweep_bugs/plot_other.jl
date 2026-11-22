# Before/after figure for the other fixes, from data/other_<variant>.jld2.
# Usage: julia --project=<env with SNNPlots, CairoMakie, JLD2> plot_other.jl
# Output: figures/other_fixes.png

using SNNPlots, CairoMakie, JLD2, Statistics
SNNPlots.@makie_default
const D = joinpath(@__DIR__, "data")
const F = joinpath(@__DIR__, "figures")
u = JLD2.load(joinpath(D, "other_unfixed.jld2"), "data")
f = JLD2.load(joinpath(D, "other_fixed.jld2"), "data")
cu, cf = okabe_ito_10[1], okabe_ito_10[6]

fig = Figure(size = (1300, 800))
# HH f-I from spike flags
ax = Axis(fig[1, 1], title = "HH: rate from spike flags", xlabel = "current (pA)", ylabel = "rate (Hz)", yscale = log10)
lines!(ax, u["HH"]["I"], max.(u["HH"]["flags"], 0.5), color = cu, label = "unfixed")
lines!(ax, f["HH"]["I"], max.(f["HH"]["flags"], 0.5), color = cf, label = "fixed")
scatter!(ax, f["HH"]["I"], max.(f["HH"]["aps"], 0.5), color = :black, markersize = 5, label = "action potentials")
axislegend(ax, position = :rb)
# STDP kernel
ax = Axis(fig[1, 2], title = "stdp_test, STDPGerstner", xlabel = "Δt = t_post - t_pre (ms)", ylabel = "ΔW")
d = f["stdp_kernel"]
t = range(-50, 50, 201)
lines!(ax, t, [x > 0 ? d["A_pre"] * exp(-x / d["τpre"]) : d["A_post"] * exp(x / d["τpost"]) for x in t], color = :gray, label = "pair rule")
scatterlines!(ax, u["stdp_kernel"]["ΔT"], u["stdp_kernel"]["dw"], color = cu, label = "unfixed")
scatterlines!(ax, d["ΔT"], d["dw"], color = cf, label = "fixed")
hlines!(ax, [0], color = :black, linewidth = 0.5)
axislegend(ax, position = :lt)
# AggregateScaling target weight
ax = Axis(fig[1, 3], title = "AggregateScaling: WT (post 100 Hz, target 50 Hz)", xlabel = "time (ms)", ylabel = "WT", yscale = Makie.pseudolog10)
for (dtk, ls) in (("dt_0.125", :solid), ("dt_0.05", :dash))
    lines!(ax, u["AggregateScaling"][dtk]["t"], u["AggregateScaling"][dtk]["WT"], color = cu, linestyle = ls, label = "unfixed $(dtk[4:end])")
    lines!(ax, f["AggregateScaling"][dtk]["t"], f["AggregateScaling"][dtk]["WT"], color = cf, linestyle = ls, label = "fixed $(dtk[4:end])")
end
axislegend(ax, position = :lb)
# vSTDP trace x peak and dW at rest
ax = Axis(fig[2, 1], title = "vSTDP: peak presynaptic trace x", xlabel = "dt (ms)", ylabel = "x peak")
dts = [0.125, 0.0625, 0.025]
scatterlines!(ax, dts, [u["vSTDP"]["dt_$(x)"]["x_peak"] for x in dts], color = cu, label = "unfixed")
scatterlines!(ax, dts, [f["vSTDP"]["dt_$(x)"]["x_peak"] for x in dts], color = cf, label = "fixed")
axislegend(ax, position = :rc)
# Confavreux PSP
ax = Axis(fig[2, 2], title = "Confavreux2025Synapse: PSP of a w = 1 spike", xticks = (1:2, ["dt 0.125", "dt 0.05"]), ylabel = "peak (mV)")
barplot!(ax, [0.85, 1.85], [u["Confavreux"]["dt_0.125"]["psp_peak"], u["Confavreux"]["dt_0.05"]["psp_peak"]], width = 0.3, color = cu)
barplot!(ax, [1.15, 2.15], [f["Confavreux"]["dt_0.125"]["psp_peak"], f["Confavreux"]["dt_0.05"]["psp_peak"]], width = 0.3, color = cf)
# record time axis
ax = Axis(fig[2, 3], title = "interpolated_record axis (:v at 10 Hz from 2 s)", xlabel = "sample", ylabel = "axis - true time (ms)")
truth = 2100.0:100.0:4000.0
scatterlines!(ax, 1:20, u["record_axis"]["r_10Hz_2to4s"] .- truth, color = cu, label = "unfixed")
scatterlines!(ax, 1:20, f["record_axis"]["r_10Hz_2to4s"] .- truth, color = cf, label = "fixed")
axislegend(ax, position = :rc)
save(joinpath(F, "other_fixes.png"), fig)
@info "saved" joinpath(F, "other_fixes.png")
