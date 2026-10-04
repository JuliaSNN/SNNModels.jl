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
        ref || continue
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

@testset "triplet STDP (Pfister & Gerstner 2006)" begin
    p = STDPTriplet(A3_minus = 2.0e-3, Wmax = 100.0f0, Wmin = -100.0f0)
    e(n, τ) = exp(-n * _DT / τ)
    (; A2_plus, A3_plus, A2_minus, A3_minus, τ_plus, τ_minus, τ_x, τ_y) = p
    a, Δ, Tp = 100, 80, 120   # steps (10 ms, 15 ms)

    # pre-post-post: pre at a, posts at a+Δ and a+Δ+Tp
    expected = A2_plus * e(Δ, τ_plus) + e(Δ + Tp, τ_plus) * (A2_plus + A3_plus * e(Tp, τ_y))
    @test _pair_dw(p, [a], [a + Δ, a + Δ + Tp]) ≈ expected rtol = 1e-4

    # post-pre-post: post at a, pre at a+Δ, post at a+Δ+Tp
    expected = -A2_minus * e(Δ, τ_minus) + e(Tp, τ_plus) * (A2_plus + A3_plus * e(Δ + Tp, τ_y))
    @test _pair_dw(p, [a + Δ], [a, a + Δ + Tp]) ≈ expected rtol = 1e-4

    # post-pre-pre (triplet LTD term): post at a, pres at a+Δ and a+Δ+Tp
    expected = -A2_minus * e(Δ, τ_minus) - e(Δ + Tp, τ_minus) * (A2_minus + A3_minus * e(Tp, τ_x))
    @test _pair_dw(p, [a + Δ, a + Δ + Tp], [a]) ≈ expected rtol = 1e-4

    # single pairs reduce to pair STDP with A2 amplitudes
    @test _pair_dw(p, [a], [a + Δ]) ≈ A2_plus * e(Δ, τ_plus) rtol = 1e-4
    @test _pair_dw(p, [a + Δ], [a]) ≈ -A2_minus * e(Δ, τ_minus) rtol = 1e-4

    # defaults: Table 4, all-to-all minimal model
    d = STDPTriplet()
    @test (d.A2_plus, d.A3_plus, d.A2_minus, d.A3_minus) == (5.3f-3, 8.0f-3, 3.5f-3, 0.0f0)
    @test (d.τ_plus, d.τ_minus, d.τ_y) == (16.8f0ms, 33.7f0ms, 40.0f0ms)
end

@testset "weight-dependent STDP" begin
    Δ = 80
    for (μp, μm, α) in ((1.0, 1.0, 1.0), (0.5, 0.0, 1.2), (0.0, 0.0, 1.0))
        p = STDPWeightDependent(η = 0.01, α = α, μ_plus = μp, μ_minus = μm,
                                Wmax = 20.0f0, Wmin = 2.0f0)
        W̃ = p.Wmax - p.Wmin
        for w0 in (4.0f0, 11.0f0, 17.0f0)
            ltp = p.η * W̃^(1 - p.μ_plus) * (p.Wmax - w0)^p.μ_plus * exp(-Δ * _DT / p.τpre)
            ltd = -p.η * p.α * W̃^(1 - p.μ_minus) * (w0 - p.Wmin)^p.μ_minus * exp(-Δ * _DT / p.τpost)
            @test _pair_dw(p, [100], [100 + Δ]; w0 = w0) ≈ ltp rtol = 1e-3
            @test _pair_dw(p, [100 + Δ], [100]; w0 = w0) ≈ ltd rtol = 1e-3
        end
    end

    # μ = 0 is additive STDP: identical to STDPGerstner(A_pre = ηW̃, A_post = -ηαW̃)
    # here W̃ = Wmax - Wmin = 2000, η = 5e-6: A_pre = 0.01, A_post = -0.012
    Random.seed!(3)
    pre, post = Identity(N = 100), Identity(N = 80)
    s1 = SpikingSynapse(pre, post, :g; conn = (p = 0.2f0, μ = 5.0f0, σ = 1.0f0),
        LTPParam = STDPGerstner(A_pre = 0.01, A_post = -0.012, Wmax = 1.0f3, Wmin = -1.0f3))
    s2 = SpikingSynapse(pre, post, :g; conn = matrix(s1),
        LTPParam = STDPWeightDependent(η = 5.0e-6, α = 1.2, μ_plus = 0, μ_minus = 0,
                                       Wmax = 1.0f3, Wmin = -1.0f3))
    @test s1.W == s2.W
    T = Time()
    for _ = 1:3000
        pre.fire .= rand(100) .< 0.02
        post.fire .= rand(80) .< 0.02
        update_time!(T, _DT)
        plasticity!(s1, s1.param, _DT, T)
        plasticity!(s2, s2.param, _DT, T)
    end
    @test isapprox(s1.W, s2.W; rtol = 1e-5)
    @test maximum(abs.(s1.W .- 5.0f0)) > 0.1

    # multiplicative bounds hold under strong driving
    p = STDPWeightDependent(η = 0.5, Wmax = 6.0f0, Wmin = 4.0f0)
    W, _ = _lockstep(p; steps = 2000, ref = false)
    @test all(4.0f0 .<= W .<= 6.0f0)
end
true
