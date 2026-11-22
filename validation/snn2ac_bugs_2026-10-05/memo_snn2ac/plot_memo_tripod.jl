# Tripod figure for the SNN2AC memo: SNNModels <= 1.8.4 vs 1.9.0.
# Reads ../sweep_bugs/data/mc_{unfixed,fixed}.jld2 (measure_multicompartment.jl),
# net_{unfixed,fixed}.jld2 (measure_network.jl, 400-cell Tripod E/I network) and
# membrane_pair.jld2 (membrane_pair.jl). Writes figures/tripod_snn2ac.png.
using SNNPlots, CairoMakie, JLD2, Statistics
SNNPlots.@makie_default

const DIR = @__DIR__
const SB = joinpath(DIR, "..", "sweep_bugs", "data")
const OLD, NEW = colorant"#D55E00", colorant"#0072B2"
mc = Dict(v => load(joinpath(SB, "mc_$v.jld2"), "data") for v in ("unfixed", "fixed"))
net = Dict(v => load(joinpath(SB, "net_$v.jld2"), "data")["dt_0.1"] for v in ("unfixed", "fixed"))
mp = load(joinpath(SB, "membrane_pair.jld2"), "res")
rates(st, T) = [length(s) / (T / 1000) for s in st]

fig = Figure(size = (1350, 820))

# A: single-neuron f-I
ax = Axis(fig[1, 1], title = "A  single Tripod: somatic f-I curve", xlabel = "somatic current (pA)", ylabel = "rate (Hz)")
for (cfg, ls, lab) in (("paper", :solid, "published parameters"), ("default", :dash, "library defaults"))
    for (v, c, vl) in (("unfixed", OLD, "≤ 1.8.4"), ("fixed", NEW, "1.9.0"))
        d = mc[v]["Tripod_$cfg"]
        lines!(ax, d["fI_I"], d["fI_rate"], color = c, linestyle = ls, linewidth = 2, label = "$vl, $lab")
    end
end
axislegend(ax, position = :lt, labelsize = 10)
r = (mc["unfixed"]["Tripod_paper"]["rheobase"], mc["fixed"]["Tripod_paper"]["rheobase"])
text!(ax, 0.98, 0.04, space = :relative, align = (:right, :bottom), fontsize = 10,
      text = "rheobase (published parameters) $(round(Int, r[1])) → $(round(Int, r[2])) pA")

# B: network mean rates
ax = Axis(fig[1, 2], title = "B  Tripod E/I network (400 E, 2 × 50 I): mean rates", ylabel = "rate (Hz)",
          xticks = (1:3, ["E (Tripod)", "I1", "I2"]))
pops = ("st_E", "st_I1", "st_I2")
mo = [mean(rates(net["unfixed"][p], net["unfixed"]["T"])) for p in pops]
mn = [mean(rates(net["fixed"][p], net["fixed"]["T"])) for p in pops]
barplot!(ax, repeat(1:3, 2), vcat(mo, mn), dodge = repeat([1, 2], inner = 3), color = repeat([OLD, NEW], inner = 3))
for (i, (a, b)) in enumerate(zip(mo, mn))
    text!(ax, i, max(a, b), text = "$(round(a, digits = 1)) → $(round(b, digits = 1))", align = (:center, :bottom), fontsize = 10)
end
ylims!(ax, 0, 1.15 * maximum(mn))
elem = [PolyElement(color = OLD), PolyElement(color = NEW)]
axislegend(ax, elem, ["≤ 1.8.4", "1.9.0"], position = :lt, labelsize = 10)

# C: E rate distribution
ax = Axis(fig[2, 1], title = "C  network: firing rate of each excitatory Tripod", xlabel = "rate (Hz)", ylabel = "neurons")
stephist!(ax, rates(net["unfixed"]["st_E"], net["unfixed"]["T"]), bins = 0:2:70, color = OLD, linewidth = 2, label = "≤ 1.8.4")
stephist!(ax, rates(net["fixed"]["st_E"], net["fixed"]["T"]), bins = 0:2:70, color = NEW, linewidth = 2, label = "1.9.0")
axislegend(ax, position = :rt, labelsize = 10)

# D: τm set by @update!
ax = Axis(fig[2, 2], title = "D  @update! adex.τm = x (gl = 40 nS): Tripod rate at 1500 pA",
          xlabel = "τm (ms)", ylabel = "rate (Hz)")
τms = mp["τm"]; k = findfirst(==(1500f0), mp["I"])
scatterlines!(ax, τms, [mp["tripod_old"][t][k] for t in τms], color = OLD, label = "≤ 1.8.x: τm ignored (C stays 281 pF)")
scatterlines!(ax, τms, [mp["tripod_new"][t][k] for t in τms], color = NEW, label = "1.9.0: C = τm · gl (200 to 800 pF)")
axislegend(ax, position = :rt, labelsize = 10)
ylims!(ax, 0, 60)

Label(fig[0, 1:2], "Tripod neuron: SNNModels ≤ 1.8.4 vs 1.9.0 (gl in the somatic exponential term, consistent Heun step, refractoriness, membrane pair rule)",
      fontsize = 13, halign = :left)
save(joinpath(DIR, "figures", "tripod_snn2ac.png"), fig, px_per_unit = 2)
println("E ", mo[1], " -> ", mn[1], "; I1 ", mo[2], " -> ", mn[2], "; I2 ", mo[3], " -> ", mn[3], "; rheobase ", r)
