using SNNModels
using Test
using Random
@load_units

# Event-driven STDP: equivalence with the old clock-driven kernels, analytic
# single-pair / kernel tests, sign convention.

include(joinpath(@__DIR__, "stdp_reference.jl"))

const _DT = 0.125f0

# Drive a synapse between two Identity populations with Bernoulli spikes set
# directly in the fire vectors, and apply the reference kernel in lockstep to a
# copy of the weights.
function _lockstep(param; Npre = 200, Npost = 150, steps = 4000, rate = 0.02,
                   seed = 1, ref = true, kw...)
    Random.seed!(seed)
    pre, post = Identity(N = Npre), Identity(N = Npost)
    syn = SpikingSynapse(pre, post, :g; conn = (p = 0.1f0, μ = 5.0f0, σ = 1.0f0),
                         LTPParam = param)
    r = (; syn.rowptr, syn.colptr, syn.I, syn.J, syn.index, W = copy(syn.W),
         syn.fireI, syn.fireJ)
    st = ref_state(Npre, Npost)
    T = Time()
    for _ = 1:steps
        pre.fire .= rand(Npre) .< rate
        post.fire .= rand(Npost) .< rate
        update_time!(T, _DT)
        plasticity!(syn, syn.param, _DT, T)
        param isa STDPMexicanHat ? ref_plasticity!(r, param, st, _DT) :
                                   ref_plasticity!(r, param, st, get_time(T); kw...)
    end
    return syn.W, r.W
end

# One synapse, pre and post spikes forced at the given step indices.
function _pair_dw(param, pre_steps, post_steps; nsteps = maximum(vcat(pre_steps, post_steps)) + 10,
                  w0 = 1.0f-3)  # small w0: dw = W - w0 keeps full Float32 precision
    pre, post = Identity(N = 1), Identity(N = 1)
    syn = SpikingSynapse(pre, post, :g; conn = fill(w0, 1, 1), LTPParam = param)
    T = Time()
    for n = 1:nsteps
        pre.fire[1] = n in pre_steps
        post.fire[1] = n in post_steps
        update_time!(T, _DT)
        plasticity!(syn, syn.param, _DT, T)
    end
    return syn.W[1] - w0
end

@testset "event-driven STDP" begin

    @testset "default sign convention" begin
        p = STDPGerstner()
        @test p.A_pre > 0
        @test p.A_post < 0
    end

    @testset "equivalence with clock-driven reference" begin
        # bounds far away: the reference clamps once per step, the event-driven
        # code after each pass (identical unless a bound is hit in between)
        pg = STDPGerstner(A_pre = 0.05, A_post = -0.04, Wmax = 1.0f3, Wmin = -1.0f3)
        W, R = _lockstep(pg; amplitude_quirk = false, coincidence_quirk = false)
        @test isapprox(W, R; rtol = 1e-4)
        @test maximum(abs.(W .- 5.0f0)) > 0.1  # plasticity actually happened
        # the old code (both quirks) is a different rule
        W, Rold = _lockstep(pg; amplitude_quirk = true, coincidence_quirk = true)
        @test !isapprox(W, Rold; rtol = 1e-2)

        pc = STDPConfavreux2025(η = 0.01, κ = -1.0f0, γ = 1.0f0, α = 0.001f0,
                                β = -0.002f0, Wmax = 1.0f3, Wmin = -1.0f3)
        W, R = _lockstep(pc; coincidence_quirk = false)
        @test isapprox(W, R; rtol = 1e-4)

        # MexicanHat: same rule, only the loops changed
        pm = STDPMexicanHat(A = 0.01)
        W, R = _lockstep(pm)
        @test isapprox(W, R; rtol = 1e-5)
    end

    @testset "bounds: initial clamp and touched clamp" begin
        p = STDPGerstner(A_pre = 1.0, A_post = -1.0, Wmax = 6.0f0, Wmin = 4.0f0)
        W, R = _lockstep(p; steps = 2000, amplitude_quirk = false, coincidence_quirk = false)
        @test all(4.0f0 .<= W .<= 6.0f0)
        @test any(W .== 6.0f0) && any(W .== 4.0f0)
    end

    @testset "single pair, both signs" begin
        τpre, τpost = 15ms, 25ms
        Δ = 80  # steps, 10 ms
        for (Ap, Am) in ((0.01, -0.012), (-0.01, 0.012))
            p = STDPGerstner(A_pre = Ap, A_post = Am, τpre = τpre, τpost = τpost,
                             Wmax = 100.0f0, Wmin = -100.0f0)
            # pre before post: LTP branch, A_pre * exp(-Δt/τpre)
            @test _pair_dw(p, [20], [20 + Δ]) ≈ Float32(Ap) * exp(-Δ * _DT / τpre) rtol = 1e-4
            # post before pre: A_post * exp(-Δt/τpost)
            @test _pair_dw(p, [20 + Δ], [20]) ≈ Float32(Am) * exp(-Δ * _DT / τpost) rtol = 1e-4
            # same step: no interaction
            @test _pair_dw(p, [20], [20]) == 0
        end
    end

    @testset "kernel sweep" begin
        p = STDPGerstner(A_pre = 0.01, A_post = -0.012, τpre = 17ms, τpost = 34ms,
                         Wmax = 100.0f0, Wmin = -100.0f0)
        t0 = 1000
        for d in -800:20:800
            d == 0 && continue
            dw = _pair_dw(p, [t0], [t0 + d])
            Δt = d * _DT
            expected = Δt > 0 ? p.A_pre * exp(-Δt / p.τpre) : p.A_post * exp(Δt / p.τpost)
            @test dw ≈ expected rtol = 1e-4
        end
    end

    @testset "all-to-all summation of pairs" begin
        # two pre spikes then one post: LTP sums both pairings; the second pre
        # spike also reads nothing from post (post fired later)
        p = STDPGerstner(A_pre = 0.01, A_post = -0.01, Wmax = 100.0f0, Wmin = -100.0f0)
        dw = _pair_dw(p, [10, 50], [90])
        @test dw ≈ 0.01f0 * (exp(-80 * _DT / 20) + exp(-40 * _DT / 20)) rtol = 1e-4
    end

    @testset "train! with forced spikes (SpikeTimeStimulusIdentity)" begin
        p = STDPGerstner(A_pre = 0.01, A_post = -0.01, Wmax = 100.0f0, Wmin = -100.0f0)
        for ΔT in (-10ms, 10ms)
            inputs = SpikeTimeParameter([100ms, 100ms + ΔT], [1, 2])
            st = Identity(N = 2)
            stim = SpikeTimeStimulusIdentity(st, :g, param = inputs)
            # negative weight: Identity fires on any g > 0 (whatever `sym`), so a
            # positive weight would make neuron 2 fire one step after neuron 1
            # and add a spurious pre-post pairing.
            w = zeros(Float32, 2, 2)
            w[2, 1] = -1.0f0
            syn = SpikingSynapse(st, st, :h, conn = w, LTPParam = p)
            model = compose(; st, stim, syn, silent = true)
            train!(model = model, duration = 200ms, dt = 0.125ms)
            expected = ΔT > 0 ? p.A_pre * exp(-ΔT / p.τpre) : p.A_post * exp(ΔT / p.τpost)
            @test syn.W[1] + 1 ≈ expected rtol = 1e-4
        end
    end
end
true
