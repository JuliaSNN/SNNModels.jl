# Bit-identity of the branchless `update_neuron!` of `AdEx` (scalar and per-neuron parameters) against the original branching
# loops of SNNModels <= 1.9, kept here as the reference.
using Random
include(joinpath(@__DIR__, "_loop_equiv.jl"))

# reference: scalar parameters
function _adex_ref!(p::AdEx, param, dt)
    (; N, v, w, fire, θ, I, tabs, syn_curr) = p
    (; τm, Vt, Vr, El, R, ΔT, τw, a, b) = param
    (; At, τA, τabs) = p.spike
    @inbounds for i ∈ 1:N
        v[i] = ifelse(fire[i], Vr, v[i])
        fire[i] = false
        tabs[i] -= 1
        tabs[i] > 0 && continue
        w[i] += dt * (a * (v[i] - El) - w[i]) / τw
        v[i] +=
            dt * (
                -(v[i] - El)
                + (ΔT < 0.0f0 ? 0.0f0 : ΔT * exp((v[i] - θ[i]) / ΔT))
                - R * syn_curr[i]
                - R * w[i]
                + R * I[i]
            ) / τm
        θ[i] += dt * (Vt - θ[i]) / τA
        fire[i] = v[i] >= 0mV
        v[i] = ifelse(fire[i], 20.0f0, v[i])
        w[i] = ifelse(fire[i], w[i] + b, w[i])
        θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
end

# reference: per-neuron parameters
function _adex_ref_het!(p::AdEx, param, dt)
    (; N, v, w, fire, θ, I, tabs, syn_curr) = p
    (; τm, Vt, Vr, El, R, ΔT, τw, a, b) = param
    (; At, τA, τabs) = p.spike
    @inbounds for i ∈ 1:N
        v[i] = ifelse(fire[i], Vr[i], v[i])
        fire[i] = false
        tabs[i] -= 1
        tabs[i] > 0 && continue
        w[i] += dt * (a[i] * (v[i] - El[i]) - w[i]) / τw[i]
        v[i] +=
            dt * (
                -(v[i] - El[i])
                + (ΔT[i] < 0.0f0 ? 0.0f0 : ΔT[i] * exp((v[i] - θ[i]) / ΔT[i]))
                - R[i] * syn_curr[i]
                - R[i] * w[i]
                + R[i] * I[i]
            ) / (τm[i])
        θ[i] += dt * (Vt[i] - θ[i]) / τA
        fire[i] = v[i] >= 0mV
        v[i] = ifelse(fire[i], 20.0f0, v[i])
        w[i] = ifelse(fire[i], w[i] + b[i], w[i])
        θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
end

const _ADEX_FIELDS = (:v, :w, :θ, :fire, :tabs, :I, :syn_curr)

function _adex_pair(N; het, ΔT = 2mV, τabs = 2ms, nsteps = 300, seed = 1, At = 3mV)
    Random.seed!(seed)
    param = het ? make_heterogeneous(AdExParameter(), N) : AdExParameter(ΔT = ΔT)
    if het && N > 0
        param.ΔT[1:3:end] .= -1.0f0                     # some neurons without exponential term
    end
    p = AdEx(; N, param, spike = PostSpike(τabs = τabs, At = At, τA = 10ms))
    p.v .= -75f0 .+ 40f0 .* rand(Float32, N)             # up to -35 mV: the exponential term spikes some of them
    p.w .= 100f0 .* rand(Float32, N)
    p.θ .= p.θ .+ 4f0 .* randn(Float32, N)
    p.tabs .= rand(0:4, N)
    p.fire .= rand(Bool, N)
    q = deepcopy(p)
    dt = 0.1f0
    nspk = 0; nref = 0
    ref! = het ? _adex_ref_het! : _adex_ref!
    for n in 1:nsteps
        Iv = 2000f0 .* rand(Float32, N); sc = 500f0 .* randn(Float32, N)
        p.I .= Iv; q.I .= Iv; p.syn_curr .= sc; q.syn_curr .= sc
        update_neuron!(p, p.param, dt)
        ref!(q, q.param, dt)
        loop_state_equal(p, q, _ADEX_FIELDS) || return (false, n, loop_first_diff(p, q, _ADEX_FIELDS), nspk, nref)
        nspk += count(p.fire); nref += count(>(0), p.tabs)
    end
    return (true, nsteps, nothing, nspk, nref)
end

@testset "AdEx update_neuron! branchless loops are bit-identical to the reference" begin
    for het in (false, true), N in (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1001)
        het && N == 0 && continue
        ok, n, f, nspk, nref = _adex_pair(N; het)
        @test ok
        if N >= 100
            @test nspk > 0
            @test nref > 0
        end
    end
    # negative slope factor (no exponential term), no refractory period, no threshold adaptation
    @test _adex_pair(64; het = false, ΔT = -1mV)[1]
    @test _adex_pair(64; het = false, τabs = 0ms)[1]
    @test _adex_pair(64; het = true, τabs = 0ms)[1]
    @test _adex_pair(64; het = false, At = 0mV)[1]
    # refractory neurons keep v = Vr (after the reset), spikes set v = 20 mV, w += b, tabs = round(τabs / dt)
    let p = AdEx(N = 3, spike = PostSpike(τabs = 1ms, At = 5mV))
        p.v .= [-30, -60, 5]; p.fire .= [true, false, false]; p.tabs .= [3, 0, 0]; p.I .= 0; p.syn_curr .= 0; p.w .= 0
        θ0 = copy(p.θ)
        update_neuron!(p, p.param, 0.1f0)
        @test p.v[1] == p.param.Vr && p.tabs[1] == 2 && !p.fire[1] && p.θ[1] == θ0[1]   # refractory: reset, frozen
        @test !p.fire[2] && p.tabs[2] == -1
        @test p.fire[3] && p.v[3] == 20f0 && p.tabs[3] == 10 && p.w[3] > p.param.b - 1 && p.θ[3] > θ0[3] + 4
    end
    # a state vector shorter than N is a clear error, not an out-of-bounds write
    let p = AdEx(N = 8)
        resize!(p.syn_curr, 5)
        @test_throws DimensionMismatch update_neuron!(p, p.param, 0.1f0)
    end
    let p = AdEx(N = 8)
        resize!(p.tabs, 3)
        @test_throws DimensionMismatch update_neuron!(p, p.param, 0.1f0)
    end
    let p = AdEx(N = 8, param = make_heterogeneous(AdExParameter(), 8))
        resize!(p.param.R, 3)
        @test_throws DimensionMismatch update_neuron!(p, p.param, 0.1f0)
    end
    # the allocation of a step does not grow with N
    let p = AdEx(N = 10), q = AdEx(N = 10000)
        update_neuron!(p, p.param, 0.1f0); update_neuron!(q, q.param, 0.1f0)
        @test (@allocated update_neuron!(q, q.param, 0.1f0)) == (@allocated update_neuron!(p, p.param, 0.1f0))
    end
end

true
