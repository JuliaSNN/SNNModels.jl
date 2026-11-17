#=
Presets of the Tripod plasticity code and the features they use
================================================================

Tests of
  - the voltage-STDP presets (lkd_vstdp, clopath_vstdp_visualcortex, bono_vstdp): values, the :tripod
    conversion against an independent per-step implementation of the Tripod-code update, closed-form
    single-synapse updates, the `delay_uv` delay of the filtered potentials;
  - the triplet presets (pfister_visualcortex, four variants) and nearest-spike traces: closed-form
    updates and equality of the event-driven rule with a dense per-step Float64 reference;
  - gutig_nltah: the mapping to STDPWeightDependent and closed-form pair updates.
=#

using SNNModels
using Test
using Random
@load_units

include(joinpath(@__DIR__, "plasticity_presets_ref.jl"))

# ── Voltage STDP: preset values ───────────────────────────────────────────────────────────────────

@testset "vSTDP presets: values" begin
    p = lkd_vstdp(convention = :verbatim)
    @test (p.A_LTD, p.A_LTP, p.θ_LTD, p.θ_LTP) == (8.0f-4, 1.4f-3, -70.0f0, -49.0f0)
    @test (p.τu, p.τv, p.τx, p.Wmin, p.Wmax, p.delay_uv) == (10.0f0, 7.0f0, 15.0f0, 0.78f0, 11.4f0, 0.0f0)
    @test p.τu != vSTDPParameter().τu                 # the library default is not the LKD set

    p = clopath_vstdp_visualcortex(convention = :verbatim)
    @test p.A_LTD ≈ 14.0f-5 / 2.5f0 && p.A_LTP ≈ 8.0f-5 / 2.5f0
    @test (p.θ_LTD, p.θ_LTP, p.delay_uv, p.Wmin, p.Wmax) == (-70.6f0, 25.3f0, 1.0f0, 0.5f0, 1.5f0)

    p = bono_vstdp(convention = :verbatim)
    @test p.A_LTD ≈ 2.0f-4 && p.A_LTP ≈ 6.0f-4
    @test (p.θ_LTD, p.θ_LTP, p.τu, p.τv, p.τx, p.delay_uv) == (-69.0f0, -15.0f0, 35.0f0, 35.0f0, 15.0f0, 1.0f0)

    # :tripod conversion at dt: A_LTD = dt a⁻, A_LTP = a⁺ (τx - dt)
    for (f, set) in _PRESETS, dt in (0.1, 0.05)
        p = f(dt = dt)
        @test p isa vSTDPParameter
        @test p.A_LTD ≈ Float32(dt * set.a_ltd) rtol = 1.0e-6
        @test p.A_LTP ≈ Float32(set.a_ltp * (set.τx - dt)) rtol = 1.0e-6
        @test f(dense = true) isa vSTDPDenseParameter
    end

    # overrides and errors
    @test bono_vstdp(Wmax = 3.0).Wmax == 3.0f0
    @test bono_vstdp(FT = Float64).A_LTD isa Float64
    @test_throws ArgumentError bono_vstdp(convention = :paper)
end

# ── Voltage STDP: closed-form updates of one synapse ──────────────────────────────────────────────

@testset "vSTDP presets: closed-form single synapse" begin
    pre, post = IF(N = 1), IF(N = 1)

    # constant potential V above both thresholds, one presynaptic spike at step 1, N steps:
    # LTD = A_LTD (V - θ_LTD); LTP = Σ_n dt A_LTP x_n g = A_LTP g (1 - q^N), q = 1 - dt/τx,
    # g = (V - θ_LTD)(V - θ_LTP) (u = v = V at all steps)
    for (f, _) in _PRESETS
        rule = f(Wmin = -100.0, Wmax = 100.0)
        V = rule.θ_LTP + 5.0f0
        N = 400
        syn = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, 1, 1), LTPParam = rule)
        spikes = [n == 1 ? [1] : Int[] for n in 1:N]
        _pp_run!(syn, pre, post.v, spikes, fill(V, 1, N))
        q = 1 - Float64(_PDT) / rule.τx
        g = (V - rule.θ_LTD) * (V - rule.θ_LTP)
        expected = -rule.A_LTD * (V - rule.θ_LTD) + rule.A_LTP * g * (1 - q^N)
        @test plasticity_value(syn, :W)[1] - 1 ≈ expected rtol = 1.0e-4
    end

    # potential step V0 -> V1 at step m, LTD only (θ_LTP above the potential), presynaptic spike at step s:
    # the drive of u at step n is V(n - D), so u(s) = V1 + (V0 - V1) q_u^(s - m - D + 1) if s >= m + D, else V0
    for D in (0, 10, 25)
        rule = bono_vstdp(delay_uv = D * _PDT, θ_LTP = 100.0, Wmin = -100.0, Wmax = 100.0)
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

# ── Voltage STDP: library rule with the :tripod convention == update of the Tripod code ───────────

@testset "vSTDP presets: equal to the update of the Tripod code" begin
    npre, npost = 20, 3
    spikes, V = _pp_inputs(; npre, npost)
    @test count(>(25.3f0), V) > 100                   # every θ_LTP is crossed
    for (f, set) in _PRESETS, dense in (false, true)
        # far bounds: the Tripod code clamps after the whole step, the library after LTD and after LTP
        rule = f(; dense, Wmin = -100.0, Wmax = 100.0)
        D = round(Int, set.ε / _PDT)
        pre, post = IF(N = npre), IF(N = npost)
        syn = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, npost, npre), LTPParam = rule)
        _pp_run!(syn, pre, post.v, spikes, V)
        W = _pp_mat(syn)
        R = _tripod_code_vstdp(spikes, V, set; D, w0 = 1.0, npre)
        change = maximum(abs.(R .- 1))
        @test change > 1.0e-3                         # the weights moved
        # absolute term: Float32 rounding of about 2e4 additions to a weight of 1 (dense rule)
        @test maximum(abs.(W .- R)) < 1.0e-4 * change + 3.0e-6
    end

    # event-driven == dense with a delay, preset bounds active
    pre, post = IF(N = npre), IF(N = npost)
    se = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, npost, npre), LTPParam = bono_vstdp(A_LTP = 1.0))
    sd = SpikingSynapse(pre, post, :ge; conn = fill(1.0f0, npost, npre), LTPParam = bono_vstdp(A_LTP = 1.0, dense = true))
    _pp_run!(se, pre, post.v, spikes, V)
    _pp_run!(sd, pre, post.v, spikes, V)
    @test count(==(1.5f0), sd.W) > 0                  # some weights reach Wmax
    @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 1.0e-4
    @test all(0.5f0 .<= plasticity_value(se, :W) .<= 1.5f0)
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

# ── Triplet STDP ──────────────────────────────────────────────────────────────────────────────────

# trace value at step n of a neuron that fired at `steps` (all-to-all: sum, nearest: last spike only),
# read before the spikes of step n
function _trace(steps, n, τ, nearest; dt = _PDT)
    past = filter(<(n), steps)
    isempty(past) && return 0.0
    nearest && return exp(-(n - maximum(past)) * dt / τ)
    return sum(exp(-(n - m) * dt / τ) for m in past)
end

# closed-form total weight change of one synapse for given pre and post spike steps
function _triplet_dw(p, pre_steps, post_steps)
    dw = 0.0
    for a in pre_steps         # depression at presynaptic spikes
        dw -= _trace(post_steps, a, p.τ_minus, p.nearest) * (p.A2_minus + p.A3_minus * _trace(pre_steps, a, p.τ_x, p.nearest))
    end
    for b in post_steps        # potentiation at postsynaptic spikes
        dw += _trace(pre_steps, b, p.τ_plus, p.nearest) * (p.A2_plus + p.A3_plus * _trace(post_steps, b, p.τ_y, p.nearest))
    end
    return dw
end

# one synapse between Identity populations, spikes forced at the given steps
function _triplet_run(p, pre_steps, post_steps; nsteps = maximum(vcat(pre_steps, post_steps)) + 5, w0 = 1.0f-3)
    pre, post = Identity(N = 1), Identity(N = 1)
    syn = SpikingSynapse(pre, post, :g; conn = fill(w0, 1, 1), LTPParam = p)   # small w0: dw keeps the Float32 digits
    T = Time()
    for n in 1:nsteps
        pre.fire[1] = n in pre_steps
        post.fire[1] = n in post_steps
        update_time!(T, _PDT)
        plasticity!(syn, syn.param, _PDT, T)
    end
    return Float64(syn.W[1]) - w0
end

# Dense Float64 reference: every trace decays exactly every step; weights read the traces before the
# spikes of the step; then the traces of the spiking neurons are increased by 1 or set to 1.
function _triplet_dense(p, pre_fire, post_fire, I, J, W0)
    W = Float64.(W0)
    Npre, Npost = size(pre_fire, 1), size(post_fire, 1)
    r1, r2, o1, o2 = zeros(Npre), zeros(Npre), zeros(Npost), zeros(Npost)
    f = (r1 = exp(-_PDT / p.τ_plus), r2 = exp(-_PDT / p.τ_x), o1 = exp(-_PDT / p.τ_minus), o2 = exp(-_PDT / p.τ_y))
    for n in 1:size(pre_fire, 2)
        for s in eachindex(W)
            i, j = I[s], J[s]
            pre_fire[j, n] && (W[s] = clamp(W[s] - o1[i] * (p.A2_minus + p.A3_minus * r2[j]), p.Wmin, p.Wmax))
        end
        for s in eachindex(W)
            i, j = I[s], J[s]
            post_fire[i, n] && (W[s] = clamp(W[s] + r1[j] * (p.A2_plus + p.A3_plus * o2[i]), p.Wmin, p.Wmax))
        end
        for (x, fire) in ((r1, view(pre_fire, :, n)), (r2, view(pre_fire, :, n)), (o1, view(post_fire, :, n)), (o2, view(post_fire, :, n)))
            for k in eachindex(x)
                fire[k] && (x[k] = p.nearest ? 1.0 : x[k] + 1.0)
            end
        end
        r1 .*= f.r1; r2 .*= f.r2; o1 .*= f.o1; o2 .*= f.o2
    end
    return W
end

@testset "pfister_visualcortex: values" begin
    p = pfister_visualcortex()
    @test (p.A2_plus, p.A3_plus, p.A2_minus, p.A3_minus) == (5.0f-10, 6.2f-3, 7.0f-3, 2.3f-4)
    @test (p.τ_x, p.τ_y, p.τ_plus, p.τ_minus, p.nearest) == (101.0f0, 125.0f0, 16.8f0, 33.7f0, false)
    p = pfister_visualcortex(true, false)
    @test (p.A2_plus, p.A3_plus, p.A2_minus, p.A3_minus, p.τ_y, p.nearest) == (0.0f0, 6.5f-3, 7.1f-3, 0.0f0, 125.0f0, false)
    p = pfister_visualcortex(false, true)
    @test (p.A2_plus, p.A3_plus, p.A2_minus, p.A3_minus) == (8.8f-11, 5.3f-2, 6.6f-3, 3.1f-3)
    @test (p.τ_x, p.τ_y, p.nearest) == (714.0f0, 40.0f0, true)
    p = pfister_visualcortex(false, false)
    @test (p.A2_plus, p.A3_plus, p.A2_minus, p.A3_minus, p.τ_y, p.nearest) == (0.0f0, 5.2f-2, 8.0f-3, 0.0f0, 40.0f0, true)
    @test pfister_visualcortex(Wmax = 5.0).Wmax == 5.0f0
    @test STDPTriplet().nearest == false
    @test convert_precision(pfister_visualcortex(false, true), Float64).nearest
end

@testset "triplet STDP: closed-form updates, all-to-all and nearest-spike" begin
    # two presynaptic spikes, three postsynaptic spikes, then two presynaptic spikes:
    # pairs, post-post triplets (o2) and pre-pre triplets (r2) all contribute
    pre_steps, post_steps = [10, 30, 120, 125], [60, 75, 90]
    for alltoall in (true, false), full in (true, false)
        p = pfister_visualcortex(alltoall, full; Wmin = -1.0, Wmax = 1.0)
        expected = _triplet_dw(p, pre_steps, post_steps)
        @test abs(expected) > 1.0e-4
        @test _triplet_run(p, pre_steps, post_steps) ≈ expected rtol = 1.0e-4
    end
    # nearest-spike and all-to-all differ only through the earlier spikes
    p_all = STDPTriplet(A2_plus = 1.0e-2, A3_plus = 1.0e-2, A2_minus = 1.0e-2, A3_minus = 1.0e-2, Wmin = -1.0, Wmax = 1.0)
    p_near = STDPTriplet(A2_plus = 1.0e-2, A3_plus = 1.0e-2, A2_minus = 1.0e-2, A3_minus = 1.0e-2, Wmin = -1.0, Wmax = 1.0, nearest = true)
    @test _triplet_run(p_all, [10], [20]) ≈ _triplet_run(p_near, [10], [20]) rtol = 1.0e-6
    @test !(_triplet_run(p_all, [5, 10], [20]) ≈ _triplet_run(p_near, [5, 10], [20]))
    @test _triplet_run(p_near, [5, 10], [20]) ≈ _triplet_dw(p_near, [5, 10], [20]) rtol = 1.0e-5
end

@testset "triplet STDP: event-driven equals the dense reference" begin
    Npre, Npost, nsteps = 60, 40, 6000
    for alltoall in (true, false), full in (true, false)
        # amplitudes x 20 so the weights move well above the Float32 resolution
        base = pfister_visualcortex(alltoall, full)
        p = pfister_visualcortex(alltoall, full; A2_plus = 20 * base.A2_plus, A3_plus = 20 * base.A3_plus,
                                 A2_minus = 20 * base.A2_minus, A3_minus = 20 * base.A3_minus, Wmin = 0.0, Wmax = 2.0)
        Random.seed!(11)
        pre, post = Identity(N = Npre), Identity(N = Npost)
        syn = SpikingSynapse(pre, post, :g; conn = (p = 0.2f0, μ = 1.0f0, σ = 0.2f0), LTPParam = p)
        W0 = copy(syn.W)
        pre_fire = rand(Npre, nsteps) .< 0.004
        post_fire = rand(Npost, nsteps) .< 0.004
        T = Time()
        for n in 1:nsteps
            pre.fire .= view(pre_fire, :, n)
            post.fire .= view(post_fire, :, n)
            update_time!(T, _PDT)
            plasticity!(syn, syn.param, _PDT, T)
        end
        R = _triplet_dense(p, pre_fire, post_fire, syn.I, syn.J, W0)
        change = maximum(abs.(R .- W0))
        @test change > 1.0e-2
        @test maximum(abs.(syn.W .- R)) < 1.0e-4 * change
    end
end

# ── Gütig et al. 2003 ─────────────────────────────────────────────────────────────────────────────

@testset "gutig_nltah" begin
    g = gutig_nltah(τ = 20ms, λ = 0.01, μ = 0.4, α = 1.05)
    @test g isa STDPWeightDependent
    @test (g.η, g.α, g.μ_plus, g.μ_minus, g.τpre, g.τpost, g.Wmin, g.Wmax) ==
          (0.01f0, 1.05f0, 0.4f0, 0.4f0, 20.0f0, 20.0f0, 0.0f0, 1.0f0)
    @test_throws UndefKeywordError gutig_nltah(τ = 20ms, λ = 0.01)

    # one pair, w0 = 0.3: pre at step a, post at b > a -> λ (1 - w)^μ e^{-Δt/τ}; reversed -> -λ α w^μ e^{-Δt/τ}
    function pair(rule, pre_steps, post_steps; w0 = 0.3f0)
        pre, post = Identity(N = 1), Identity(N = 1)
        syn = SpikingSynapse(pre, post, :g; conn = fill(w0, 1, 1), LTPParam = rule)
        T = Time()
        for n in 1:maximum(vcat(pre_steps, post_steps)) + 3
            pre.fire[1] = n in pre_steps
            post.fire[1] = n in post_steps
            update_time!(T, _PDT)
            plasticity!(syn, syn.param, _PDT, T)
        end
        return Float64(syn.W[1]) - w0
    end
    Δt = 50 * _PDT
    for μ in (0.0, 0.4, 1.0)
        g = gutig_nltah(τ = 20ms, λ = 0.01, μ = μ, α = 1.05)
        @test pair(g, [10], [60]) ≈ 0.01 * (1 - 0.3)^μ * exp(-Δt / 20) rtol = 1.0e-4
        @test pair(g, [60], [10]) ≈ -0.01 * 1.05 * 0.3^μ * exp(-Δt / 20) rtol = 1.0e-4
    end
    # other bounds: the same rule on the normalised weight (w - Wmin)/(Wmax - Wmin)
    g = gutig_nltah(τ = 20ms, λ = 0.01, μ = 0.4, Wmin = 0.5, Wmax = 2.5)
    ŵ = (1.1 - 0.5) / 2.0
    @test pair(g, [10], [60]; w0 = 1.1f0) ≈ 2.0 * 0.01 * (1 - ŵ)^0.4 * exp(-Δt / 20) rtol = 1.0e-4
end
