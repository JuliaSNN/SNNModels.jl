# Efficacy per spike before (mixed u-/u+ order, SNNModels <= 1.8.4) and after (Mongillo 2008)
# the STP fix, for a 20 Hz train of 10 spikes plus a recovery probe 500 ms later.
# Reads data/stp_order_{mixed,mongillo}.jld2 written by stp_order.jl; writes figures/stp_order.png.
using SNNPlots, CairoMakie, JLD2
SNNPlots.@makie_default

const OLD, NEW, REF = colorant"#D55E00", colorant"#0072B2", colorant"#1f1f1e"
dir = @__DIR__
old = load(joinpath(dir, "data", "stp_order_mixed.jld2"))["res"]
new = load(joinpath(dir, "data", "stp_order_mongillo.jld2"))["res"]
names = ["depressing", "facilitating", "facilitating, U = 0.05", "SNN2AC ThExt_to_Exc", "SNN2AC ThExt_to_PV"]

fig = Figure(size = (1500, 420))
for (k, n) in enumerate(names)
    o, m = old[n], new[n]
    ax = Axis(fig[1, k], title = "$(n)\nU = $(o.U), τF = $(Int(o.τF)) ms, τD = $(Int(o.τD)) ms", titlesize = 12,
              xlabel = "spike (11 = probe after 500 ms)", ylabel = k == 1 ? "efficacy u·x" : "")
    s = 1:length(o.event)
    scatterlines!(ax, s, o.event, color = OLD, label = "≤ 1.8.4 (mixed order)")
    scatterlines!(ax, s, m.event, color = NEW, label = "fixed (Mongillo 2008)")
    scatter!(ax, s, m.reference, color = REF, marker = :xcross, markersize = 8, label = "analytic Mongillo 2008")
    k == 1 && axislegend(ax, position = :rt, labelsize = 10)
end
save(joinpath(dir, "figures", "stp_order.png"), fig, px_per_unit = 2)
println("wrote ", joinpath(dir, "figures", "stp_order.png"))
