# Efficacy per spike of the Tsodyks-Markram STP rules for a regular presynaptic train,
# before and after switching to the Mongillo 2008 convention (u jumps first; release and
# depletion both use u+). Run once per code version with LABEL set ("mixed" or "mongillo").
# The analytic Mongillo 2008 recursion is computed independently in Julia as the reference.
using SNNModels, JLD2
@load_units

# one presynaptic neuron, one synapse; spikes forced at the given times
function efficacies(param, spike_times; dt = 0.1f0, T_end = maximum(spike_times) + 1)
    pre, post = Identity(N = 1), Identity(N = 1)
    syn = SpikingSynapse(pre, post, :ge; conn = ones(Float32, 1, 1), STPParam = param)
    T = Time(); vars = syn.STPVars
    steps = round(Int, T_end / dt); spike_steps = Set(round.(Int, spike_times ./ dt))
    out = Float32[]
    for k = 1:steps
        update_time!(T, dt)
        pre.fire[1] = k in spike_steps
        SNNModels.update_traces!(syn, param, vars, dt, T)       # train! order: traces, forward, plasticity
        pre.fire[1] && push!(out, syn.ρ[1])                     # efficacy used by forward! this step
        SNNModels.plasticity!(syn, param, vars, dt, T)
    end
    return out
end

# Mongillo, Barak & Tsodyks 2008: u jumps first, release u+ x-, depletion u+ x-
function mongillo(U, τF, τD, spike_times)
    u, x, t0 = U, 1.0, -Inf; out = Float64[]
    for t in spike_times
        Δ = t - t0
        u = U - (U - u) * exp(-Δ / τF); x = 1 - (1 - x) * exp(-Δ / τD)
        u += U * (1 - u); push!(out, u * x); x -= u * x; t0 = t
    end
    return out
end

train = collect(100.0:50.0:550.0)          # 10 spikes at 20 Hz
probe = vcat(train, 1050.0)                # plus a recovery probe 500 ms after the train
cases = [("depressing", 0.5, 50.0, 800.0), ("facilitating", 0.2, 1500.0, 200.0), ("facilitating, U = 0.05", 0.05, 1500.0, 200.0),
         ("SNN2AC ThExt_to_Exc", 0.3, 500.0, 70.0), ("SNN2AC ThExt_to_PV", 0.2, 200.0, 110.0)]
res = Dict{String,Any}()
for (name, U, τF, τD) in cases
    ev = efficacies(MarkramSTPParameterEvent(U = Float32(U), τF = Float32(τF), τD = Float32(τD)), probe)
    ts = efficacies(MarkramSTPParameterTimestep(U = Float32(U), τF = Float32(τF), τD = Float32(τD)), probe)
    res[name] = (; U, τF, τD, event = ev, timestep = ts, reference = mongillo(U, τF, τD, probe))
    println(LABEL, " ", rpad(name, 24), " event ", round.(ev[[1, 2, 10, 11]], digits = 4),
            " timestep ", round.(ts[[1, 2, 10, 11]], digits = 4),
            " mongillo ", round.(res[name].reference[[1, 2, 10, 11]], digits = 4))
end
@save joinpath(@__DIR__, "data", "stp_order_$(LABEL).jld2") res probe
