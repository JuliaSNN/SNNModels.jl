# Figure of the membrane pair rule effect, regenerated from data/membrane_pair.jld2
# (written by membrane_pair.jl). Output: figures/membrane_pair.png
using SNNPlots, CairoMakie, JLD2
SNNPlots.@makie_default

const D = @__DIR__
res = load(joinpath(D, "data", "membrane_pair.jld2"), "res")
I, τms = res["I"], res["τm"]
seq = cgrad(:viridis, length(τms) + 1, categorical = true)   # one hue ramp for the ordered τm values
lab(τm) = "τm = $(round(τm, digits = 3)) ms (C = $(round(Int, res["C_new"][τm])) pF)"

fig = Figure(size = (1350, 420))
panels = (("Tripod, ≤ 1.8.x: @update! τm ignored", "tripod_old"),
          ("Tripod, 1.9.0: @update! τm sets C = τm gl", "tripod_new"),
          ("AdEx (unchanged: reads τm in both versions)", "adex"))
for (k, (title, key)) in enumerate(panels)
    ax = Axis(fig[1, k], title = title, titlesize = 13, xlabel = "somatic current (pA)",
              ylabel = k == 1 ? "rate (Hz)" : "")
    for (j, τm) in enumerate(τms)
        lines!(ax, I, res[key][τm], color = seq[j], linewidth = 2, label = lab(τm))
    end
    k == 2 && axislegend(ax, position = :lt, labelsize = 10)
    ylims!(ax, 0, 95)
end
Label(fig[0, 1:3], "@update! adex.τm = x with gl = 40 nS, τm in SNN2AC's optimisation range. 1 s current steps, dt = 0.1 ms.",
      fontsize = 13, halign = :left)
save(joinpath(D, "figures", "membrane_pair.png"), fig, px_per_unit = 2)
println("wrote figures/membrane_pair.png")
