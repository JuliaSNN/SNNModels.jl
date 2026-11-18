#=
`delay_uv` of vSTDPParameter / vSTDPDenseParameter: the low-pass filters u, v are driven by the potential
round(delay_uv/dt) steps earlier. Closed-form LTD after a potential step, event-driven == dense, dt change.
=#

using SNNModels
using Test
using Random
@load_units

const _PDT = 0.1f0     # time step (ms)

# Presynaptic spikes (list per step) and postsynaptic potentials (npost x nsteps): a slow sinusoid around
# -70 mV and, at Poisson postsynaptic spikes, +100 mV for 1 ms then a -8 mV afterhyperpolarisation, so the
# potential crosses θ_LTP (peak about +30 mV).
function _pp_inputs(; npre = 20, npost = 3, nsteps = 20000, dt = 0.1, seed = 7, rate = 10.0, post_rate = 8.0)
    rng = MersenneTwister(seed)
    spikes = [Int[] for _ in 1:nsteps]
    for j in 1:npre, n in 1:nsteps
        rand(rng) < rate * dt * 1.0e-3 && push!(spikes[n], j)
    end

    V = zeros(Float32, npost, nsteps)
    for i in 1:npost
        post = [n for n in 1:nsteps if rand(rng) < post_rate * dt * 1.0e-3]
        for n in 1:nsteps
            t = (n - 1) * dt
            v = -70 + 4sin(2π * t / (200 + 37i))
            for m in post
                τ = t - (m - 1) * dt
                (0 <= τ <= 150) || continue
                v += τ < 1 ? 100.0 : -8exp(-(τ - 1) / 15)
            end
            V[i, n] = v
        end
    end
    return spikes, V
end

# Step a plastic synapse through `train!` with prescribed presynaptic spikes and postsynaptic potential:
# call n sees the potential V[:, n] (written into `vpost`) and the spikes of step n. The synapse does not
# drive its target (`syn.g` is replaced by a scratch vector).
function _pp_run!(syn, pre, vpost, spikes, V; dt = _PDT)
    syn.g = zeros(Float32, length(vpost))
    T = Time()
    for n in eachindex(spikes)
        vpost .= view(V, :, n)
        fill!(pre.fire, false)
        pre.fire[spikes[n]] .= true
        train!(SNNModels.AbstractPopulation[], [syn], SNNModels.AbstractStimulus[], dt, T)
    end
    return syn
end

@testset "vSTDP delay_uv: closed form" begin
    pre, post = IF(N = 1), IF(N = 1)
    # potential step V0 -> V1 at step m, LTD only (θ_LTP above the potential), presynaptic spike at step s:
    # the drive of u at step n is V(n - D), so u(s) = V1 + (V0 - V1) q_u^(s - m - D + 1) if s >= m + D, else V0
    for D in (0, 10, 25)
        rule = vSTDPParameter(A_LTD = 2.0e-4, θ_LTD = -69.0, θ_LTP = 100.0, τu = 35.0, delay_uv = D * _PDT,
                              Wmin = -100.0, Wmax = 100.0)
        V0, V1, m, s, N = -60.0f0, -40.0f0, 50, 70, 100
        V = reshape([n < m ? V0 : V1 for n in 1:N], 1, N)
        spikes = [n == s ? [1] : Int[] for n in 1:N]
        syn = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, 1, 1), LTPParam = rule)
        _pp_run!(syn, pre, post.v, spikes, V)
        qu = 1 - Float64(_PDT) / rule.τu
        u = s >= m + D ? V1 + (V0 - V1) * qu^(s - m - D + 1) : V0
        @test plasticity_value(syn, :W)[1] - 1 ≈ -rule.A_LTD * (u - rule.θ_LTD) rtol = 1.0e-4
        @test syn.LTPVars.vslot[2] == D
    end
end

@testset "vSTDP delay_uv: event-driven equals dense, dt change" begin
    npre, npost = 20, 3
    spikes, V = _pp_inputs(; npre, npost, seed = 3)
    for D in (1, 7)
        kw = (A_LTD = 2.0e-3, A_LTP = 5.0e-4, θ_LTP = -20.0, Wmin = 0.0, Wmax = 2.0, delay_uv = D * _PDT)
        pre, post = IF(N = npre), IF(N = npost)
        se = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, npost, npre), LTPParam = vSTDPParameter(; kw...))
        sd = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, npost, npre), LTPParam = vSTDPDenseParameter(; kw...))
        s0 = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, npost, npre),
                            LTPParam = vSTDPParameter(; merge(kw, (delay_uv = 0.0,))...))
        foreach(s -> _pp_run!(s, pre, post.v, spikes, V), (se, sd, s0))
        change = maximum(abs.(sd.W .- 1))
        @test change > 1.0e-2
        @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 2.0e-5 * change
        @test maximum(abs.(plasticity_value(s0, :W) .- sd.W)) > 1.0e-3 * change   # the delay changes the result
    end

    # the number of delay steps follows dt: a dt change refills the delay line and keeps event == dense
    kw = (A_LTD = 2.0e-3, A_LTP = 5.0e-4, θ_LTP = -20.0, Wmin = 0.0, Wmax = 2.0, delay_uv = 1.0)
    pre, post = IF(N = npre), IF(N = npost)
    se = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, npost, npre), LTPParam = vSTDPParameter(; kw...))
    sd = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, npost, npre), LTPParam = vSTDPDenseParameter(; kw...))
    for s in (se, sd)
        _pp_run!(s, pre, post.v, spikes[1:10000], V[:, 1:10000])
        @test s.LTPVars.vslot[2] == 10
        _pp_run!(s, pre, post.v, spikes[10001:end], V[:, 10001:end]; dt = 0.05f0)
        @test s.LTPVars.vslot[2] == 20
    end
    @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 2.0e-5 * maximum(abs.(sd.W .- 1))
end
