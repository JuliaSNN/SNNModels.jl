# Print the numbers of the Tripod / BallAndStick comparison from data/mc_<variant>.jld2 and write
# them to data/mc_summary.txt (used in the report).
# Usage: julia --project=<any env with JLD2> summarize_multicompartment.jl

using JLD2, Statistics, Printf

const D = joinpath(@__DIR__, "data")
load_variant(v) = JLD2.load(joinpath(D, "mc_$(v).jld2"), "data")

# baseline label per model: BallAndStick needs the variant in which Is is applied
variants(model) = model == "Tripod" ? ["unfixed", "dtonly", "glonly", "fixed"] :
                                      ["unfixedIs", "dtonly", "glonly", "fixed"]

io = IOBuffer()
out(args...) = (print(io, args...); print(args...))

"Spike-time deviation (ms) of `st` from the reference `ref`: max |Δt| over matched spikes, and Δcount."
function st_dev(st, ref)
    devs = Float64[]
    dn = 0
    for (a, b) in zip(st, ref)
        n = min(length(a), length(b))
        n > 0 && push!(devs, maximum(abs.(a[1:n] .- b[1:n])))
        dn = max(dn, abs(length(a) - length(b)))
    end
    return isempty(devs) ? NaN : maximum(devs), dn
end

ref_fixed = load_variant("fixed")
for model in ("Tripod", "BallAndStick"), cfg in ("default", "paper")
    key = "$(model)_$(cfg)"
    out("\n=== $key ===\n")
    out(@sprintf("%-10s %9s %8s %8s %8s %9s %8s %8s %8s %8s %8s\n", "variant", "rheo(pA)",
        "r@1000", "r@1500", "r@2500", "ISI1@1500", "AI@1500", "EPSPs@1", "EPSPd@1", "att@1", "plat@32"))
    for v in variants(model)
        d = load_variant(v)[key]
        I = d["fI_I"]; r = d["fI_rate"]
        rate_at(x) = r[findfirst(==(Float32(x)), I)]
        isi1 = d["isi"][2]
        k1 = findfirst(==(1.0f0), d["epsp_w"]); k32 = findfirst(==(32.0f0), d["epsp_w"])
        out(@sprintf("%-10s %9.0f %8.0f %8.0f %8.0f %9.2f %8.2f %8.3f %8.3f %8.2f %8.1f\n", v,
            d["rheobase"], rate_at(1000), rate_at(1500), rate_at(2500),
            isempty(isi1) ? NaN : isi1[1], d["adaptation_index"][2],
            d["epsp_soma"][k1], d["epsp_dend"][k1], d["epsp_dend"][k1] / d["epsp_soma"][k1],
            d["plateau_ms"][k32]))
    end
    out("EPSP vs weight (soma / dend mV; plateau ms > -50 mV; spikes):\n")
    for v in variants(model)
        d = load_variant(v)[key]
        out(@sprintf("  %-10s ", v), join([@sprintf("w=%g: %.2f/%.2f/%.0f/%d", w, s, dd, p, n) for (w, s, dd, p, n) in
            zip(d["epsp_w"], d["epsp_soma"], d["epsp_dend"], d["plateau_ms"], d["epsp_spikes"])], "  "), "\n")
    end
    out("bAP dendritic amplitude (mean, mV): ", join([@sprintf("%s %.2f", v, mean(load_variant(v)[key]["bap_amplitude"])) for v in variants(model)], ", "), "\n")
    out("spike times of the 1500 pA step (first 6, ms):\n")
    for v in variants(model)
        st = load_variant(v)[key]["wave_st"][2]
        out(@sprintf("  %-10s ", v), join([@sprintf("%.2f", t) for t in st[1:min(6, end)]], " "), "\n")
    end
    # dt dependence
    rf = ref_fixed[key]
    dts = rf["dt_values"]
    out("dt dependence (500 ms steps at $(rf["dt_I"]) pA): max |spike-time deviation| (ms) / max |Δ spike count|\n")
    out("  reference A = same variant at dt = 0.0125 ms; reference B = fixed code at dt = 0.0125 ms\n")
    for v in variants(model)
        d = load_variant(v)[key]
        refA = d["dt_st"][1]; refB = rf["dt_st"][1]
        cells = String[]
        for (k, dt) in enumerate(dts)
            k == 1 && continue
            a, na = st_dev(d["dt_st"][k], refA); b, nb = st_dev(d["dt_st"][k], refB)
            push!(cells, @sprintf("dt=%g: A %.2f/%d  B %.2f/%d", dt, a, na, b, nb))
        end
        a, na = st_dev(d["dt_st"][1], refB)
        push!(cells, @sprintf("dt=0.0125 vs B: %.2f/%d", a, na))
        out(@sprintf("  %-10s ", v), join(cells, " | "), "\n")
    end
    out("rheobase at dt = 0.05 / 0.1 / 0.125 ms: ", join([@sprintf("%s %s", v, string(round.(load_variant(v)[key]["dt_rheobase"]))) for v in variants(model)], "; "), "\n")
    out("rate (Hz) at 1500 pA for dt = ", join(dts, "/"), ": ")
    for v in variants(model)
        d = load_variant(v)[key]
        k = findfirst(==(1500.0f0), d["dt_fI_I"])
        out(@sprintf("%s %s; ", v, join([string(Int(rr[k])) for rr in d["dt_fI_rate"]], "/")))
    end
    out("\n")
end
write(joinpath(D, "mc_summary.txt"), String(take!(io)))
