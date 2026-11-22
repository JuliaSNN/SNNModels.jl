# Network-level figure from data/net_*.jld2 (Tripod E/I network of network_model.jl).
# Usage: julia --project=<env with SNNPlots, CairoMakie, JLD2> plot_network.jl
# Output: figures/network.png

using SNNPlots, CairoMakie, JLD2, Statistics
SNNPlots.@makie_default
const D = joinpath(@__DIR__, "data")
const F = joinpath(@__DIR__, "figures")
const T0 = 500.0f0
load(f) = JLD2.load(joinpath(D, f), "data")
runs = [("unfixed", "net_unfixed.jld2"), ("Heun fix only", "net_dtonly.jld2"),
        ("gl fix only", "net_glonly.jld2"), ("all fixes", "net_fixed.jld2")]
rate(st, T) = mean(length.([filter(>(T0), s) for s in st])) / ((T - T0) / 1000)
cv(st) = mean([std(diff(s)) / mean(diff(s)) for s in st if length(s) > 3])

fig = Figure(size = (1100, 750))
for (j, (title, f)) in enumerate([("E rate (Hz)", x -> rate(x["st_E"], x["T"])),
                                  ("I1 rate (Hz)", x -> rate(x["st_I1"], x["T"])),
                                  ("I2 rate (Hz)", x -> rate(x["st_I2"], x["T"])),
                                  ("E ISI CV", x -> cv(x["st_E"])),
                                  ("E: g_exc / g_inh", x -> mean(x["ge"][51:end]) / mean(x["gi"][51:end]))])
    ax = Axis(fig[1, j], title = title, xticks = (1:4, first.(runs)), xticklabelrotation = π / 4)
    for (k, (lab, file)) in enumerate(runs), (o, dt) in ((-0.15, "0.1"), (0.15, "0.125"))
        barplot!(ax, [k + o], [f(load(file)["dt_$(dt)"])], width = 0.3,
                 color = dt == "0.1" ? okabe_ito_10[6] : okabe_ito_10[2])
    end
    # seed-to-seed variability (seed 4321) as black markers, dt = 0.1 ms
    for (k, file) in ((1, "net_unfixed_seed4321.jld2"), (4, "net_fixed_seed4321.jld2"))
        scatter!(ax, [k - 0.15], [f(load(file)["dt_0.1"])], color = :black, markersize = 7)
    end
end
Label(fig[0, :], "bars: dt = 0.1 ms (left), 0.125 ms (right); dots: second seed, dt = 0.1 ms", fontsize = 12)
for (k, (lab, file)) in enumerate([("unfixed (1.8.4)", "net_unfixed.jld2"), ("all fixes", "net_fixed.jld2")])
    ax = Axis(fig[2, k == 1 ? (1:2) : (3:5)], title = "$lab, dt = 0.1 ms",
              xlabel = "time (ms)", ylabel = "E neuron")
    st = load(file)["dt_0.1"]["st_E"]
    raster!(ax, SNNPlots.SNNModels.Spiketimes(st[1:150]), [4000, 5000])
end
save(joinpath(F, "network.png"), fig)
@info "saved" joinpath(F, "network.png")
