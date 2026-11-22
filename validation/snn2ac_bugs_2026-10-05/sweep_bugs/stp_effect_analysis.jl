# Effect of the Tsodyks-Markram STP update order (SNNModels <= 1.8.4 "mixed" vs 1.9.0 Mongillo 2008)
# over parameter space. Pure recursions (no simulator), checked against the saved output of
# the old code (data/stp_order_mixed.jld2). Writes data/stp_effect.jld2 and figures/stp_effect.png.
using SNNPlots, CairoMakie, JLD2, Statistics
SNNPlots.@makie_default
dir = @__DIR__

# efficacies of a spike train under the two conventions
function efficacies(U, τF, τD, times; order::Symbol)
    u, x, t0 = U, 1.0, -Inf; out = Float64[]
    for t in times
        Δ = t - t0
        u = U - (U - u) * exp(-Δ / τF); x = 1 - (1 - x) * exp(-Δ / τD)
        if order === :mixed            # efficacy with u before the jump, depletion with u after
            push!(out, u * x); u += U * (1 - u); x -= u * x
        else                           # Mongillo 2008: jump first, same u+ for both
            u += U * (1 - u); push!(out, u * x); x -= u * x
        end
        t0 = t
    end
    return out
end
regular(rate_Hz; n = 20, t0 = 100.0) = t0 .+ (0:n-1) .* (1000.0 / rate_Hz)

# 1. check the mixed recursion reproduces the old simulator output
old = load(joinpath(dir, "data", "stp_order_mixed.jld2")); probe = old["probe"]
maxerr = maximum(maximum(abs.(efficacies(r.U, r.τF, r.τD, probe; order = :mixed) .- r.event)) for r in values(old["res"]))
println("mixed recursion vs old SNNModels output: max |Δ| = ", maxerr)

# 2. parameter sweeps
Us = range(0.02, 0.9, length = 90); rates = range(2, 60, length = 59)
syn = [("SNN2AC ThExt→Exc", 500.0, 70.0), ("SNN2AC ThExt→PV", 200.0, 110.0),
       ("facilitating (τF 1500, τD 200)", 1500.0, 200.0), ("depressing (τF 50, τD 800)", 50.0, 800.0)]
ppr(e) = e[2] / e[1]; steady(e) = mean(e[end-4:end])
res = Dict{String,Any}()
for (name, τF, τD) in syn
    P = Dict(o => [ppr(efficacies(U, τF, τD, regular(r); order = o)) for U in Us, r in rates] for o in (:mixed, :mongillo))
    S = Dict(o => [steady(efficacies(U, τF, τD, regular(r); order = o)) for U in Us, r in rates] for o in (:mixed, :mongillo))
    res[name] = (; τF, τD, P, S)
end
first_ratio = [U * (2 - U) / U for U in Us]          # first spike: new/old = 2 - U
@save joinpath(dir, "data", "stp_effect.jld2") Us rates res maxerr

# fraction of (U, rate) where the paired-pulse character flips (PPR>1 facilitating)
for (name, _, _) in syn
    P = res[name].P; flip = mean((P[:mixed] .> 1) .!= (P[:mongillo] .> 1))
    println(rpad(name, 34), " PPR flips facilitating<->depressing on ", round(100flip, digits = 1), "% of (U, rate) grid; ",
            "steady-state new/old median ", round(median(res[name].S[:mongillo] ./ res[name].S[:mixed]), digits = 2))
end

# 3. figure
fig = Figure(size = (1500, 1050))
cm = :balance
for (k, (name, τF, τD)) in enumerate(syn)
    P = res[name].P
    for (row, o, lab) in ((1, :mixed, "≤ 1.8.4 (mixed)"), (2, :mongillo, "1.9.0 (Mongillo 2008)"))
        ax = Axis(fig[row, k], title = row == 1 ? "$(name)\npaired-pulse ratio, $(lab)" : "paired-pulse ratio, $(lab)",
                  titlesize = 12, xlabel = row == 2 ? "presynaptic rate (Hz)" : "", ylabel = k == 1 ? "U" : "")
        hm = heatmap!(ax, rates, Us, log2.(P[o])', colormap = cm, colorrange = (-1.5, 1.5))
        contour!(ax, rates, Us, log2.(P[o])', levels = [0.0], color = :black, linewidth = 1.5)
        (k == 4 && row == 1) && Colorbar(fig[1:2, 5], hm, label = "log₂ PPR  (red: facilitating, blue: depressing)")
    end
    ax = Axis(fig[3, k], title = "steady-state efficacy, new / old", titlesize = 12, xlabel = "presynaptic rate (Hz)", ylabel = k == 1 ? "U" : "")
    hm2 = heatmap!(ax, rates, Us, (res[name].S[:mongillo] ./ res[name].S[:mixed])', colormap = :viridis, colorrange = (1, 2.2))
    k == 4 && Colorbar(fig[3, 5], hm2, label = "ratio")
end
Label(fig[0, 1:4], "Tsodyks-Markram STP: old mixed update order vs Mongillo 2008, 20-spike regular trains. Black line: PPR = 1.",
      fontsize = 14, halign = :left)
save(joinpath(dir, "figures", "stp_effect.png"), fig, px_per_unit = 2)
println("wrote figures/stp_effect.png")
