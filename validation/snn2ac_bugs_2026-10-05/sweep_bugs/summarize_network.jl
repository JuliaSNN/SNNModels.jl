# Network statistics from data/net_<variant>.jld2: population rates, ISI CV, population Fano
# factor (5 ms bins), fraction of active E cells, excitatory/inhibitory conductance ratio of the
# E cells. The first 500 ms are discarded. Writes data/net_summary.txt.
# Usage: julia --project=<env with JLD2> summarize_network.jl

using JLD2, Statistics, Printf
const D = joinpath(@__DIR__, "data")
const T0 = 500.0f0

function stats(res)
    T = res["T"]
    out = Dict{String,Float64}()
    for k in ("E", "I1", "I2")
        st = [filter(>(T0), s) for s in res["st_$(k)"]]
        rates = length.(st) ./ ((T - T0) / 1000)
        out["rate_$(k)"] = mean(rates)
        cvs = [std(diff(s)) / mean(diff(s)) for s in st if length(s) > 3]
        out["cv_$(k)"] = isempty(cvs) ? NaN : mean(cvs)
        allst = reduce(vcat, st; init = Float32[])
        counts = [count(t -> e <= t < e + 5, allst) for e in T0:5:(T-5)]
        out["fano_$(k)"] = var(counts) / mean(counts)
        k == "E" && (out["active_E"] = mean(rates .> 1.0))
    end
    n = length(res["ge"]); keep = round(Int, n * T0 / T)+1:n
    out["ge"] = mean(res["ge"][keep]); out["gi"] = mean(res["gi"][keep])
    out["ge_gi"] = out["ge"] / out["gi"]
    return out
end

io = IOBuffer()
pr(args...) = (print(io, args...); print(args...))
cols = ["rate_E", "rate_I1", "rate_I2", "cv_E", "cv_I1", "cv_I2", "fano_E", "active_E", "ge", "gi", "ge_gi"]
pr(@sprintf("%-22s", "variant / dt"), join([@sprintf("%9s", c) for c in cols]), "\n")
files = [("unfixed", "net_unfixed.jld2"), ("unfixed seed 4321", "net_unfixed_seed4321.jld2"),
         ("dtonly", "net_dtonly.jld2"), ("glonly", "net_glonly.jld2"), ("fixed", "net_fixed.jld2"),
         ("fixed seed 4321", "net_fixed_seed4321.jld2")]
for (lab, f) in files
    isfile(joinpath(D, f)) || continue
    d = JLD2.load(joinpath(D, f), "data")
    for dt in ("0.1", "0.125")
        s = stats(d["dt_$(dt)"])
        pr(@sprintf("%-22s", "$lab / $dt"), join([@sprintf("%9.3f", s[c]) for c in cols]), "\n")
    end
end
write(joinpath(D, "net_summary.txt"), String(take!(io)))
