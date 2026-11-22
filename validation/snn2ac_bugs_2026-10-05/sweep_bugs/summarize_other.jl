# Print the before/after numbers of data/other_<variant>.jld2 and write data/other_summary.txt.
# Usage: julia --project=<env with JLD2> summarize_other.jl

using JLD2, Statistics, Printf
const D = joinpath(@__DIR__, "data")
u = JLD2.load(joinpath(D, "other_unfixed.jld2"), "data")
f = JLD2.load(joinpath(D, "other_fixed.jld2"), "data")
io = IOBuffer()
out(args...) = (print(io, args...); print(args...))
err(d) = d isa Dict && haskey(d, "error")

out("== HH (1 s, dt = 0.01 ms): spike flags / upward crossings of -20 mV ==\n")
for (lab, d) in (("unfixed", u["HH"]), ("fixed", f["HH"]))
    sel = findall(>(0), d["aps"])
    ratio = mean(d["flags"][sel] ./ d["aps"][sel])
    out(@sprintf("%-8s flags/AP = %.2f (range %.2f-%.2f); rate at 500/1000/1500 pA from flags: %s; true AP rate: %s\n", lab, ratio,
        minimum(d["flags"][sel] ./ d["aps"][sel]), maximum(d["flags"][sel] ./ d["aps"][sel]),
        join(d["flags"][[findfirst(==(Float32(x)), d["I"]) for x in (500, 1000, 1500)]], "/"),
        join(d["aps"][[findfirst(==(Float32(x)), d["I"]) for x in (500, 1000, 1500)]], "/")))
end
out("== MorrisLecar (3 pulses of 100 pA) ==\n")
for (lab, d) in (("unfixed", u["MorrisLecar"]), ("fixed", f["MorrisLecar"]))
    out(@sprintf("%-8s flags = %d, action potentials = %d\n", lab, d["flags"], d["aps"]))
end
out("== vSTDP: one pre spike, post clamped at -40 mV for 30 ms ==\n")
for (lab, d) in (("unfixed", u["vSTDP"]), ("fixed", f["vSTDP"]))
    for dt in (0.125, 0.0625, 0.025)
        x = d["dt_$(dt)"]
        out(@sprintf("%-8s dt=%-6g dW = %.5f   peak x = %.5f   dW (post at rest) = %.5f\n", lab, dt,
            x["dw_depolarised"], x["x_peak"], x["dw_at_rest"]))
    end
    out(@sprintf("%-8s silent AdEx target, 20 Hz input, 50 ms: mean dW = %.5f\n", lab, d["silent_target_mean_dw"]))
end
out("== AggregateScaling (post at 100 Hz, target 50 Hz, 2 s) ==\n")
for (lab, d) in (("unfixed", u["AggregateScaling"]), ("fixed", f["AggregateScaling"]))
    for dt in ("0.125", "0.05")
        x = d["dt_$(dt)"]
        out(@sprintf("%-8s dt=%-5s y(end) = %.4f /ms   WT(0.5 s) = %.3f   WT(2 s) = %.3f   sum W(2 s) = %.3f\n", lab, dt,
            x["y"][end], x["WT"][50], x["WT"][end], x["Wsum"][end]))
    end
end
out("== HetRec (no spikes, 300 ms): adaptive baseline of neuron 1 ==\n")
for (lab, d) in (("unfixed", u["HetRec"]), ("fixed", f["HetRec"]))
    a, b = d["dt_0.125"], d["dt_0.05"]
    out(@sprintf("%-8s trace(300 ms) dt=0.125: %.3f  dt=0.05: %.3f  (ratio %.2f); v_s: %.3f vs %.3f\n", lab,
        a["trace_end"][1], b["trace_end"][1], a["trace_end"][1] / b["trace_end"][1], a["v_s"][1], b["v_s"][1]))
end
out("== Confavreux2025Synapse (spike of weight 1 at rest) ==\n")
for (lab, d) in (("unfixed", u["Confavreux"]), ("fixed", f["Confavreux"]))
    for dt in ("0.125", "0.05")
        x = d["dt_$(dt)"]
        out(@sprintf("%-8s dt=%-5s gAMPA after the spike = %.4f nS   PSP peak = %.4f mV\n", lab, dt, x["gAMPA_after_spike"], x["psp_peak"]))
    end
end
out("== interpolated_record time axis ==\n")
for (lab, d) in (("unfixed", u["record_axis"]), ("fixed", f["record_axis"]))
    r = d["r_10Hz_2to4s"]; r2 = d["r_1kHz_10.5ms"]
    out(@sprintf("%-8s 10 Hz from 2 s: %d samples, axis %.3f:%.3f:%.3f ms (true 2100:100:4000)\n", lab, d["n_10Hz"], r[1], r[2] - r[1], r[end]))
    out(@sprintf("%-8s 1 kHz for 10.5 ms: %d samples, axis %.3f:%.3f:%.3f ms (true 0:1:10)\n", lab, d["n_1kHz"], r2[1], r2[2] - r2[1], r2[end]))
end
out("== stdp_kernel (STDPGerstner, SNNPlots.stdp_test) ==\n")
for (lab, d) in (("unfixed", u["stdp_kernel"]), ("fixed", f["stdp_kernel"]))
    out(@sprintf("%-8s ", lab), join([@sprintf("%g: %.2e", t, w) for (t, w) in zip(d["ΔT"], d["dw"])], "  "), "\n")
end
d = f["stdp_kernel"]
out("expected  ", join([@sprintf("%g: %.2e", t, t > 0 ? d["A_pre"] * exp(-t / d["τpre"]) : d["A_post"] * exp(t / d["τpost"])) for t in d["ΔT"]], "  "),
    "   (pair rule with the STDPGerstner amplitudes and time constants)\n")
out("== Rate population under RateSynapse (input g reset each step) ==\n")
for (lab, d) in (("unfixed", u["Rate_g"]), ("fixed", f["Rate_g"]))
    out(@sprintf("%-8s mean |x| at 20 ms = %.3f, at 220 ms = %.3f; fraction |r| > 0.99 at 20 / 220 ms = %.2f / %.2f\n", lab,
        d["mean_abs_x_20ms"], d["mean_abs_x_220ms"], d["saturated_20ms"], d["saturated_220ms"]))
end
out("== AdditiveNorm (weights x1.5, one normalisation) ==\n")
for (lab, d) in (("unfixed", u["AdditiveNorm"]), ("fixed", f["AdditiveNorm"]))
    out(@sprintf("%-8s mean W0 = %.2f, mean sum after = %.2f, max |sum - W0| = %.3f\n", lab, mean(d["W0"]), mean(d["after"]), maximum(abs.(d["after"] .- d["W0"]))))
end
out("== BalancedStimulus default ==\n")
for (lab, d) in (("unfixed", u["Balanced"]), ("fixed", f["Balanced"]))
    if err(d)
        out(@sprintf("%-8s error: %s\n", lab, first(split(d["error"], '\n'))))
    else
        out(@sprintf("%-8s exc %.3f /ms, inh %.3f /ms per neuron, neurons driven %d, same buffer %s\n", lab,
            d["exc_rate_per_ms"], d["inh_rate_per_ms"], d["neurons_driven"], d["same_buffer"]))
    end
end
write(joinpath(D, "other_summary.txt"), String(take!(io)))
