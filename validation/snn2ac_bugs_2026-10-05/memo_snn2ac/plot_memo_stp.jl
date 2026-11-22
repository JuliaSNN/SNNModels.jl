# STP update-order figure for the SNN2AC memo: SNN2AC's two thalamic STP synapses (configuration
# STPinput) under SNNModels <= 1.8.4 (mixed order) and >= 1.9.0 (Mongillo 2008).
# Reads ../sweep_bugs/data/stp_order_{mixed,mongillo}.jld2 (stp_order.jl, simulator output) and
# ../sweep_bugs/data/stp_effect.jld2 (stp_effect_analysis.jl, recursions checked against the old
# simulator output). Writes figures/stp_snn2ac.png.
using SNNPlots, CairoMakie, JLD2
SNNPlots.@makie_default

const DIR = @__DIR__
const SB = joinpath(DIR, "..", "sweep_bugs", "data")
const OLD, NEW = colorant"#D55E00", colorant"#0072B2"
old = load(joinpath(SB, "stp_order_mixed.jld2"))["res"]
new = load(joinpath(SB, "stp_order_mongillo.jld2"))["res"]
eff = load(joinpath(SB, "stp_effect.jld2"))
Us, rates = eff["Us"], eff["rates"]
syn = [("ThExt → Exc", "SNN2AC ThExt_to_Exc", "SNN2AC ThExt→Exc"),
       ("ThExt → PV", "SNN2AC ThExt_to_PV", "SNN2AC ThExt→PV")]

fig = Figure(size = (1150, 820))
for (k, (title, korder, keff)) in enumerate(syn)
    o, n = old[korder], new[korder]
    ax = Axis(fig[1, k], title = "$title  (U = $(o.U), τF = $(Int(o.τF)) ms, τD = $(Int(o.τD)) ms)",
              xlabel = "spike number (20 Hz train; 11 = probe 500 ms later)", ylabel = k == 1 ? "efficacy u·x" : "",
              xticks = 1:11)
    s = 1:length(o.event)
    scatterlines!(ax, s, o.event, color = OLD, label = "≤ 1.8.4 (v5)")
    scatterlines!(ax, s, n.event, color = NEW, label = "1.9.0")
    ylims!(ax, 0, 0.8)
    k == 1 && axislegend(ax, position = :rt, labelsize = 11)

    P = eff["res"][keff].P
    ax2 = Axis(fig[2, k], title = "facilitating vs depressing (paired-pulse ratio = 1)", xlabel = "presynaptic rate (Hz)",
               ylabel = k == 1 ? "U" : "")
    contourf!(ax2, rates, Us, (log2.(P[:mongillo]))', levels = [-10, 0, 10], colormap = [colorant"#dbe7f1", colorant"#f6dccb"])
    contour!(ax2, rates, Us, (log2.(P[:mixed]))', levels = [0.0], color = OLD, linewidth = 2.5)
    contour!(ax2, rates, Us, (log2.(P[:mongillo]))', levels = [0.0], color = NEW, linewidth = 2.5)
    scatter!(ax2, [20.0], [o.U], color = :black, marker = :star5, markersize = 16)
    text!(ax2, 21.5, o.U, text = "SNN2AC, 20 Hz", fontsize = 11, align = (:left, :center))
    text!(ax2, 0.97, 0.95, space = :relative, align = (:right, :top), fontsize = 11,
          text = "below the line: facilitating\nabove: depressing\nshading: 1.9.0")
end
Label(fig[0, 1:2], "Thalamic input STP of SNN2AC (STPinput): same parameters, old and new update order. " *
      "Orange line: PPR = 1 for ≤ 1.8.4; blue: 1.9.0.", fontsize = 13, halign = :left)
save(joinpath(DIR, "figures", "stp_snn2ac.png"), fig, px_per_unit = 2)
println("wrote figures/stp_snn2ac.png")
