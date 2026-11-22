# Network-level comparison: the Tripod E/I network of network_model.jl, identical seeds, run
# with sim! (no plasticity) for 5 s at dt = 0.1 and 0.125 ms.
#
# Usage:  julia --project=<env> measure_network.jl <variant> [seed]
# Output: data/net_<variant>.jld2 (seed 1234) or data/net_<variant>_seed<seed>.jld2, with spike
# times and conductance snapshots. A second seed measures the seed-to-seed variability.

using SNNModels
using JLD2
using Statistics
@load_units
include(joinpath(@__DIR__, "network_model.jl"))

const VARIANT = ARGS[1]
const SEED = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1234
const OUT = joinpath(@__DIR__, "data", SEED == 1234 ? "net_$(VARIANT).jld2" : "net_$(VARIANT)_seed$(SEED).jld2")

# total excitatory / inhibitory conductance (nS) of the E cells, summed over the compartments
function conductances(E)
    ge = zeros(Float32, E.N)
    gi = zeros(Float32, E.N)
    for (syn, vars) in ((E.soma_syn, E.synvars_s), (E.dend_syn, E.synvars_d1), (E.dend_syn, E.synvars_d2))
        for (n, rec) in enumerate(syn.syn)
            g = rec.gsyn .* vars.g[:, n]
            n in syn.glu_receptors ? (ge .+= g) : (gi .+= g)
        end
    end
    return ge, gi
end

data = Dict{String,Any}("variant" => VARIANT)
for dt in (0.1f0, 0.125f0)
    model = tripod_network(; seed = SEED)
    monitor!(model.pop, [:fire], monitor_rate = 100Hz)
    T, chunk = 5000.0f0, 10.0f0
    ge_t, gi_t = Float32[], Float32[]
    for _ = 1:round(Int, T / chunk)
        sim!(; model, duration = chunk, dt = dt)
        ge, gi = conductances(model.pop.E)
        push!(ge_t, mean(ge)); push!(gi_t, mean(gi))
    end
    res = Dict{String,Any}()
    for k in (:E, :I1, :I2)
        res["st_$(k)"] = spiketimes(getfield(model.pop, k))
    end
    res["ge"] = ge_t
    res["gi"] = gi_t
    res["T"] = T
    data["dt_$(dt)"] = res
    @info "done" VARIANT dt mean(length.(res["st_E"])) / (T / 1000)
end
mkpath(dirname(OUT))
jldsave(OUT; data)
@info "saved" OUT
