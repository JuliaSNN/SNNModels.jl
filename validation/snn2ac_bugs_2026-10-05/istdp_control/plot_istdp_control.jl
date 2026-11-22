# Figures for the iSTDPRate fixed-vs-buggy control check.
# Reads the JLD2 files written by istdp_cmp.jl and istdp_structured_input.jl (run once in the
# fixed and once in the buggy environment) and writes:
#   istdp_fixed_vs_buggy.png     kernel test + unstructured E-I network
#   istdp_structured_input.png   localized ensembles and input gradient
# Style: SNNPlots' Makie theme (loaded by `using SNNPlots`: no grid, frameless legends,
# Okabe-Ito palette). SNNPlots has no comparison-panel plot types, so panels are plain Makie.
# Run from the publication environment:
#   julia --project=<env with SNNModels >= 1.9, SNNPlots, CairoMakie, JLD2> scripts/validation/snnmodels_bugs_2026-10-05/istdp_control/plot_istdp_control.jl
using SNNPlots, CairoMakie, JLD2, Statistics
# SNNPlots <= 0.2.9 set its theme only at precompile time, so apply it explicitly
# (from 0.2.10 `using SNNPlots` already does this; the call is harmless).
SNNPlots.@makie_default

const DIR = @__DIR__
const FIXED, BUGGY, STATIC = colorant"#0072B2", colorant"#D55E00", colorant"#8c8c8c"   # Okabe-Ito blue / vermillion, neutral grey
const INK = colorant"#1f1f1e"

binmean(x, k) = [mean(x[i:min(i + k - 1, end)]) for i = 1:k:length(x)]
bincent(n, k) = [(i + min(i + k - 1, n)) / 2 for i = 1:k:n]
r2(x) = round(x, digits = 2)
note!(ax, x, y, s; align = (:left, :top)) =
    text!(ax, x, y, space = :relative, align = align, fontsize = 11, color = INK, text = s)

# ---------------------------------------------------------------------------------------------
# Figure 1: kernel test (same forced spikes) and unstructured E-I network
# ---------------------------------------------------------------------------------------------
function figure_kernel_network(f, b)
    fig = Figure(size = (1150, 820))
    Label(fig[0, 1:2], "iSTDPRate: registered SNNModels 1.8.2 (fixed) vs control/istdp-bug-1.8.2 (buggy), identical connectivity and seeds",
          fontsize = 14, halign = :left, color = INK)

    ax = Axis(fig[1, 1], title = "A  Same forced spikes: final I→E weights", titlealign = :left,
              xlabel = "fixed 1.8.2 weight (pF)  (= plain reference, max err 1e-5)", ylabel = "weight (pF)")
    lims = extrema(vcat(f["W_kernel"], b["W_kernel"]))
    lines!(ax, [lims...], [lims...], color = STATIC, linewidth = 1, linestyle = :dash)
    scatter!(ax, f["W_kernel"], b["W_kernel"], color = (BUGGY, 0.35), markersize = 4, label = "buggy (control branch)")
    scatter!(ax, f["W_kernel"], f["W_kernel"], color = FIXED, markersize = 3, label = "fixed (registered 1.8.2)")
    axislegend(ax, position = :lt)

    dW = b["W_kernel"] .- f["W_kernel"]
    ax = Axis(fig[1, 2], title = "B  Same forced spikes: buggy − fixed per synapse", titlealign = :left,
              xlabel = "synapse (CSC order, grouped by inhibitory neuron)", ylabel = "ΔW (pF)")
    hlines!(ax, [0], color = STATIC, linewidth = 1)
    scatter!(ax, 1:length(dW), dW, color = (BUGGY, 0.5), markersize = 3)
    note!(ax, 0.02, 0.97, "$(count(!=(0), dW)) of $(length(dW)) synapses differ\nmax |ΔW| = $(round(maximum(abs.(dW)), digits = 1)) pF")

    ax = Axis(fig[2, 1], title = "C  E–I network, 5 s train!: I→E weights", titlealign = :left,
              xlabel = "fixed 1.8.2 weight (pF)", ylabel = "buggy weight (pF)")
    lims = extrema(vcat(f["WIE"], b["WIE"]))
    lines!(ax, [lims...], [lims...], color = STATIC, linewidth = 1, linestyle = :dash)
    scatter!(ax, f["WIE"], b["WIE"], color = (BUGGY, 0.25), markersize = 3)
    note!(ax, 0.02, 0.97, "same initial weights in both\ncorr = $(r2(cor(f["WIE"], b["WIE"]))), mean |ΔW| = $(round(mean(abs.(f["WIE"] .- b["WIE"])), digits = 1)) pF")

    # rates are spike counts over 3 s, i.e. multiples of 1/3 Hz: align bins to that grid
    ax = Axis(fig[2, 2], title = "D  E–I network: excitatory firing rates (last 3 s)", titlealign = :left,
              xlabel = "rate (Hz; spike count over 3 s, bins of 1/3 Hz)", ylabel = "neurons")
    edges = range(-1 / 6, maximum(vcat(f["rates"], b["rates"])) + 1 / 6, step = 1 / 3)
    stephist!(ax, f["rates"], bins = edges, color = FIXED, linewidth = 2, label = "fixed  mean $(r2(mean(f["rates"]))) Hz, sd $(r2(std(f["rates"])))")
    stephist!(ax, b["rates"], bins = edges, color = BUGGY, linewidth = 2, linestyle = :dash, label = "buggy  mean $(r2(mean(b["rates"]))) Hz, sd $(r2(std(b["rates"])))")
    axislegend(ax, position = :rt)
    return fig
end

# ---------------------------------------------------------------------------------------------
# Figure 2: structured external input (two localized ensembles, gradient)
# ---------------------------------------------------------------------------------------------
function figure_structured(F, B)
    fig = Figure(size = (1300, 1250))
    Legend(fig[0, 1:2],
           [LineElement(color = STATIC, linewidth = 2), LineElement(color = FIXED, linewidth = 2),
            LineElement(color = BUGGY, linewidth = 2, linestyle = :dash)],
           ["no plasticity (static I→E)", "fixed iSTDPRate (registered 1.8.2)", "buggy iSTDPRate (control branch)"],
           orientation = :horizontal, tellwidth = false, labelsize = 13)
    for (col, kind, name) in ((1, "ensembles", "Two localized ensembles (E 151–250, 551–650)"),
                              (2, "gradient", "Input gradient along E index"))
        fs, fp, bp = F["$(kind)_static"], F["$(kind)_plastic"], B["$(kind)_plastic"]
        NE, NI = length(fs.Iext), length(fp.wout)
        x = bincent(NE, 20)

        ax = Axis(fig[1, col], title = "$(name)\nexternal current per E neuron", titlealign = :left,
                  ylabel = "pA", xticklabelsvisible = false)
        band!(ax, 1:NE, zeros(NE), fs.Iext, color = (STATIC, 0.35))
        lines!(ax, 1:NE, fs.Iext, color = STATIC, linewidth = 1.5)

        ax = Axis(fig[2, col], title = "E firing rate, last 5 s (20-neuron bins); dotted: 5 Hz target", titlealign = :left,
                  ylabel = "Hz", xticklabelsvisible = false)
        hlines!(ax, [5], color = STATIC, linestyle = :dot, linewidth = 1)
        lines!(ax, x, binmean(fs.rE, 20), color = STATIC, linewidth = 2)
        lines!(ax, x, binmean(fp.rE, 20), color = FIXED, linewidth = 2)
        lines!(ax, x, binmean(bp.rE, 20), color = BUGGY, linewidth = 2, linestyle = :dash)
        note!(ax, 0.01, 0.40, "corr(rate, input): static $(r2(cor(fs.rE, fs.Iext))), fixed $(r2(cor(fp.rE, fs.Iext))), buggy $(r2(cor(bp.rE, fs.Iext)))";
              align = (:left, :center))

        ax = Axis(fig[3, col], title = "Mean incoming I→E weight per E neuron", titlealign = :left,
                  ylabel = "pF", xticklabelsvisible = false)
        lines!(ax, x, binmean(fp.win, 20), color = FIXED, linewidth = 2)
        lines!(ax, x, binmean(bp.win, 20), color = BUGGY, linewidth = 2, linestyle = :dash)
        note!(ax, 0.99, 0.03, "corr(weight, input): fixed $(r2(cor(fp.win, fs.Iext))), buggy $(r2(cor(bp.win, fs.Iext)))";
              align = (:right, :bottom))

        # Under the bug, inhibitory neuron j's outgoing synapses are potentiated by the E neurons
        # whose row ranges cover j's CSC block, i.e. E index ≈ j * NE / NI.
        k = NE ÷ NI
        ax = Axis(fig[4, col], title = "Mean outgoing weight per I neuron j, plotted at E index $(k)j", titlealign = :left,
                  ylabel = "pF", xlabel = "E neuron index (row 4: $(k) × inhibitory index j)")
        xi = k .* bincent(NI, 5)
        lo, hi = extrema(vcat(binmean(fp.wout, 5), binmean(bp.wout, 5)))
        band!(ax, 1:NE, fill(lo, NE), lo .+ fs.Iext ./ maximum(fs.Iext) .* (hi - lo) .* 0.3, color = (STATIC, 0.25))
        lines!(ax, xi, binmean(fp.wout, 5), color = FIXED, linewidth = 2)
        lines!(ax, xi, binmean(bp.wout, 5), color = BUGGY, linewidth = 2, linestyle = :dash)
        drive = fs.Iext[k .* (1:NI)]
        note!(ax, 0.01, 0.97, "corr(weight, input at E index $(k)j): fixed $(r2(cor(fp.wout, drive))), buggy $(r2(cor(bp.wout, drive)))\ngrey: input profile (scaled)")
    end
    rowsize!(fig.layout, 1, Relative(0.13))
    colgap!(fig.layout, 40)
    return fig
end

f, b = load(joinpath(DIR, "result_fixed.jld2")), load(joinpath(DIR, "result_buggy.jld2"))
save(joinpath(DIR, "istdp_fixed_vs_buggy.png"), figure_kernel_network(f, b), px_per_unit = 2)
F, B = load(joinpath(DIR, "structured_fixed.jld2"))["results"], load(joinpath(DIR, "structured_buggy.jld2"))["results"]
save(joinpath(DIR, "istdp_structured_input.png"), figure_structured(F, B), px_per_unit = 2)
println("wrote ", joinpath(DIR, "istdp_fixed_vs_buggy.png"), " and ", joinpath(DIR, "istdp_structured_input.png"))
