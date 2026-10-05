using SNNModels
using Test
@load_units

# Tsodyks-Markram STP in the Mongillo, Barak & Tsodyks (2008) convention: at a presynaptic
# spike u jumps first, and the same u+ sets the efficacy and the depletion of x.
function _mongillo_ref(U, τF, τD, spike_times)
    u, x, t0 = U, 1.0, -Inf; out = Float64[]
    for t in spike_times
        Δ = t - t0
        u = U - (U - u) * exp(-Δ / τF); x = 1 - (1 - x) * exp(-Δ / τD)
        u += U * (1 - u); push!(out, u * x); x -= u * x; t0 = t
    end
    return out
end

function _stp_efficacies(param, spike_times; dt = 0.1f0)
    pre, post = Identity(N = 1), Identity(N = 1)
    syn = SpikingSynapse(pre, post, :ge; conn = ones(Float32, 1, 1), STPParam = param)
    T = Time(); vars = syn.STPVars
    spike_steps = Set(round.(Int, spike_times ./ dt)); out = Float32[]
    for k = 1:round(Int, (maximum(spike_times) + 1) / dt)
        update_time!(T, dt)
        pre.fire[1] = k in spike_steps
        SNNModels.update_traces!(syn, param, vars, dt, T)
        pre.fire[1] && push!(out, syn.ρ[1])
        SNNModels.plasticity!(syn, param, vars, dt, T)
    end
    return out
end

@testset "Markram STP, Mongillo 2008 order" begin
    times = vcat(collect(100.0:50.0:550.0), 1050.0)
    for (U, τF, τD) in ((0.5, 50.0, 800.0), (0.2, 1500.0, 200.0), (0.3, 500.0, 70.0))
        ref = _mongillo_ref(U, τF, τD, times)
        ev = _stp_efficacies(MarkramSTPParameterEvent(U = Float32(U), τF = Float32(τF), τD = Float32(τD)), times)
        ts = _stp_efficacies(MarkramSTPParameterTimestep(U = Float32(U), τF = Float32(τF), τD = Float32(τD)), times)
        @test ev[1] ≈ U * (2 - U) atol = 1e-6       # first spike from rest
        @test maximum(abs.(ev .- ref)) < 1e-4        # exact between spikes
        @test maximum(abs.(ts .- ref)) < 1e-3        # forward-Euler relaxation
    end
    # heterogeneous variant with equal parameters reproduces the homogeneous one
    het = MarkramSTPParameterHet(U = [0.3f0], τF = [500f0], τD = [70f0])
    @test _stp_efficacies(het, times) ≈ _stp_efficacies(MarkramSTPParameterEvent(U = 0.3f0, τF = 500f0, τD = 70f0), times)
end
