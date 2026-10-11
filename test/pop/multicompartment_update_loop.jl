# The vectorised / branchless `integrate!` of `Tripod` and `BallAndStick` (and the receptor synapse stage they use) against the original
# loops of SNNModels <= 1.9, kept here as the reference. Synapse and state stages are bit-identical; the Heun stage (`update_neuron!`,
# vectorised since 1.10 without `@fastmath`) differs from the `@fastmath` reference by a few ulp, so it is compared with a tolerance,
# both populations restarted from the same state at every step.
using Random
include(joinpath(@__DIR__, "_loop_equiv.jl"))

# reference: synaptic current of ReceptorSynapse / MultiReceptorSynapse
function _ref_receptor_current!(p, synapse, synvars, v, syncurr)
    (; N) = p
    (; g, h) = synvars
    (; syn, NMDA) = synapse
    (; mg, b, k) = NMDA
    fill!(syncurr, 0.0f0)
    for n in eachindex(syn)
        (; gsyn, E_rev, nmda) = syn[n]
        for neuron ∈ 1:N
            syncurr[neuron] +=
                gsyn *
                g[neuron, n] *
                (v[neuron] - E_rev) *
                (nmda == 0.0f0 ? 1.0f0 : 1 / (1.0f0 + (mg / b) * exp256(k * v[neuron])))
        end
    end
end


# reference: MultiReceptorSynapse input stage of SNNModels <= 1.9
function _ref_update_synapses!(p, synapse::MultiReceptorSynapse, receptors, synvars, dt)
    for name in keys(synapse.receptors)
        @inbounds for n in synapse.receptors[name]
            (; N, g, h) = synvars
            (; τr⁻, τd⁻, α) = synapse.syn[n]
            target = getfield(receptors, name)
            for i ∈ 1:N
                h[i, n] += target[i] * α
                g[i, n] = SNNModels.exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
                h[i, n] = SNNModels.exp64(-dt * τr⁻) * (h[i, n])
            end
        end
    end
    for name in keys(receptors)
        fill!(getfield(receptors, name), 0.0f0)
    end
end

# reference: Tripod, SNNModels <= 1.9
function _ref_update_neuron!(p::Tripod, param, Δv, dt)
    (; v_d1, v_d2, v_s, I_d, I, w_s, θ) = p
    (; d1, d2) = p
    (; is, ic, v_pred) = p
    (; adex, soma_syn, dend_syn) = p
    (; C, gl, El, ΔT, a, τw) = adex
    (; synvars_s, synvars_d1, synvars_d2) = p
    @inbounds for i ∈ 1:p.N
        v_pred[i, 1] = v_s[i] + Δv[i, 1] * dt
        v_pred[i, 2] = v_d1[i] + Δv[i, 2] * dt
        v_pred[i, 3] = v_d2[i] + Δv[i, 3] * dt
    end
    @views _ref_receptor_current!(p, soma_syn, synvars_s, v_pred[:, 1], is[:, 1])
    @views _ref_receptor_current!(p, dend_syn, synvars_d1, v_pred[:, 2], is[:, 2])
    @views _ref_receptor_current!(p, dend_syn, synvars_d2, v_pred[:, 3], is[:, 3])
    clamp!(is, -1500, 1500)
    @fastmath @inbounds for i ∈ 1:p.N
        vs = v_pred[i, 1]
        vd1 = v_pred[i, 2]
        vd2 = v_pred[i, 3]
        ws = w_s[i] + Δv[i, 4] * dt
        ic[1] = (vs - vd1) * d1.gax[i]
        ic[2] = (vs - vd2) * d2.gax[i]
        Δv[i, 2] = ((d1.El[i] - vd1) * d1.gm[i] - is[i, 2] + ic[1] + I_d[i]) / d1.C[i]
        Δv[i, 3] = ((d2.El[i] - vd2) * d2.gm[i] - is[i, 3] + ic[2] + I_d[i]) / d2.C[i]
        Δv[i, 1] =
            (
                gl * (El - vs) +
                gl * ΔT * exp256((vs - θ[i]) / ΔT) - ws
                - is[i, 1]
                - ic[1] - ic[2]
                + I[i]
            ) / C
        Δv[i, 4] = (a * (vs - El) - ws) / τw
    end
end

function _ref_integrate!(p::Tripod, param, dt)
    (; τabs, up) = p.spike
    n_abs = max(1, round(Int, τabs / dt))
    n_ref = max(1, round(Int, up / dt)) + n_abs
    _ref_heun!(p, param, dt)
    _ref_state!(p, dt, n_abs, n_ref)
end

# reference: synapse stages and the two Heun stages (the first part of `_ref_integrate!`)
function _ref_heun!(p::Tripod, param, dt)
    (; Δv, Δv_temp, fire, synvars_s, synvars_d1, synvars_d2, receptors_d1, receptors_d2, receptors_s, soma_syn, dend_syn) = p
    SNNModels.update_synapses!(p, soma_syn, receptors_s, synvars_s, dt)
    SNNModels.update_synapses!(p, dend_syn, receptors_d1, synvars_d1, dt)
    SNNModels.update_synapses!(p, dend_syn, receptors_d2, synvars_d2, dt)
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)
    _ref_update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    _ref_update_neuron!(p, param, Δv, dt)
end

# reference: state stage (spike, refractoriness, Heun update; the second part of `_ref_integrate!`)
function _ref_state!(p::Tripod, dt, n_abs, n_ref)
    (; N, v_s, w_s, v_d1, v_d2, fire, θ, tabs, Δv, Δv_temp, d1, d2, spike, adex, Vspike) = p
    (; AP_membrane, At, τA) = spike
    (; Vr, Vt, b) = adex
    @inbounds for i ∈ 1:N
        tabs[i] -= 1
        θ[i] += dt * (Vt - θ[i]) / τA
        if tabs[i] > n_abs
            v_s[i] = AP_membrane
            v_d1[i] += dt * (v_s[i] - v_d1[i]) * d1.gax[i] / d1.C[i]
            v_d2[i] += dt * (v_s[i] - v_d2[i]) * d2.gax[i] / d2.C[i]
        elseif tabs[i] > 0
            v_s[i] = Vr
            v_d1[i] += dt * (v_s[i] - v_d1[i]) * d1.gax[i] / d1.C[i]
            v_d2[i] += dt * (v_s[i] - v_d2[i]) * d2.gax[i] / d2.C[i]
        elseif tabs[i] <= 0
            fire[i] = v_s[i] + Δv[i, 1] * dt >= Vspike
            Δv[i, 1] = ifelse(fire[i], AP_membrane - v_s[i], Δv[i, 1])
            v_s[i] = ifelse(fire[i], AP_membrane, v_s[i])
            w_s[i] = ifelse(fire[i], w_s[i] + b, w_s[i])
            θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
            tabs[i] = ifelse(fire[i], n_ref, tabs[i])
            fire[i] && continue
            v_s[i] += 0.5 * dt * (Δv_temp[i, 1] + Δv[i, 1])
            v_d1[i] += 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2])
            v_d2[i] += 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3])
            w_s[i] += 0.5 * dt * (Δv_temp[i, 4] + Δv[i, 4])
        end
    end
end

# reference: BallAndStick, SNNModels <= 1.9
function _ref_update_neuron!(p::BallAndStick, param, Δv, dt)
    (; v_d, v_s, w_s, θ, Is, Id) = p
    (; d) = p
    (; is, ic, v_pred) = p
    (; adex, soma_syn, dend_syn) = p
    (; C, gl, El, ΔT, a, τw) = adex
    (; synvars_s, synvars_d) = p
    @inbounds for i ∈ 1:p.N
        v_pred[i, 1] = v_s[i] + Δv[i, 1] * dt
        v_pred[i, 2] = v_d[i] + Δv[i, 2] * dt
    end
    @views _ref_receptor_current!(p, soma_syn, synvars_s, v_pred[:, 1], is[:, 1])
    @views _ref_receptor_current!(p, dend_syn, synvars_d, v_pred[:, 2], is[:, 2])
    clamp!(is, -1000, 1000)
    @fastmath @inbounds for i ∈ 1:p.N
        vs = v_pred[i, 1]
        vd = v_pred[i, 2]
        ws = w_s[i] + Δv[i, 3] * dt
        ic[1] = (vs - vd) * d.gax[i]
        Δv[i, 1] =
            (
                gl * (El - vs) +
                gl * ΔT * exp256((vs - θ[i]) / ΔT) - ws
                - is[i, 1]
                - ic[1]
                + Is[i]
            ) / C
        Δv[i, 2] = ((d.El[i] - vd) * d.gm[i] - is[i, 2] + ic[1] + Id[i]) / d.C[i]
        Δv[i, 3] = (a * (vs - El) - ws) / τw
    end
end

function _ref_integrate!(p::BallAndStick, param, dt)
    (; τabs, up) = p.spike
    n_abs = max(1, round(Int, τabs / dt))
    n_ref = max(1, round(Int, up / dt)) + n_abs
    _ref_heun!(p, param, dt)
    _ref_state!(p, dt, n_abs, n_ref)
end

function _ref_heun!(p::BallAndStick, param, dt)
    (; Δv, Δv_temp, fire, synvars_s, synvars_d, receptors_d, receptors_s, soma_syn, dend_syn) = p
    SNNModels.update_synapses!(p, soma_syn, receptors_s, synvars_s, dt)
    SNNModels.update_synapses!(p, dend_syn, receptors_d, synvars_d, dt)
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)
    _ref_update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    _ref_update_neuron!(p, param, Δv, dt)
end

function _ref_state!(p::BallAndStick, dt, n_abs, n_ref)
    (; N, v_s, w_s, v_d, fire, θ, tabs, Δv, Δv_temp, d, spike, adex, Vspike) = p
    (; AP_membrane, At, τA) = spike
    (; Vr, Vt, b) = adex
    @inbounds for i ∈ 1:N
        tabs[i] -= 1
        θ[i] += dt * (Vt - θ[i]) / τA
        if tabs[i] > n_abs
            v_s[i] = AP_membrane
            v_d[i] += dt * (v_s[i] - v_d[i]) * d.gax[i] / d.C[i]
        elseif tabs[i] > 0
            v_s[i] = Vr
            v_d[i] += dt * (v_s[i] - v_d[i]) * d.gax[i] / d.C[i]
        elseif tabs[i] <= 0
            fire[i] = v_s[i] + Δv[i, 1] * dt >= Vspike
            Δv[i, 1] = ifelse(fire[i], AP_membrane - v_s[i], Δv[i, 1])
            v_s[i] = ifelse(fire[i], AP_membrane, v_s[i])
            w_s[i] = ifelse(fire[i], w_s[i] + b, w_s[i])
            θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
            tabs[i] = ifelse(fire[i], n_ref, tabs[i])
            fire[i] && continue
            v_s[i] += 0.5 * dt * (Δv_temp[i, 1] + Δv[i, 1])
            v_d[i] += 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2])
            w_s[i] += 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3])
        end
    end
end

const _TRIPOD_STATE = (:v_s, :w_s, :v_d1, :v_d2, :fire, :tabs, :θ, :Δv, :Δv_temp, :is, :v_pred, :I, :I_d)
const _BAS_STATE = (:v_s, :w_s, :v_d, :fire, :tabs, :θ, :Δv, :Δv_temp, :is, :v_pred, :Is, :Id)
const _TRIPOD_EXACT = (:fire, :tabs, :I, :I_d)                  # must stay bit-identical
const _BAS_EXACT = (:fire, :tabs, :Is, :Id)
const _TRIPOD_POT = (:v_s, :w_s, :v_d1, :v_d2, :θ)              # state after the step: within `_STATE_RTOL`
const _BAS_POT = (:v_s, :w_s, :v_d, :θ)
_mc_synvars(T) = T === Tripod ? (:synvars_s, :synvars_d1, :synvars_d2) : (:synvars_s, :synvars_d)
const _STAGE_ULPS = 4          # Heun stage: |Δv - Δv_ref| <= 4 ulp of the magnitude of the terms of the derivative
const _STATE_RTOL = 3f-5       # one whole step from the same state: potentials within 3e-5 relative (measured max 7e-6, near-divergent neurons)

# Error of a Heun-stage derivative in units of the last place of its terms. The derivative is a sum of terms (leak, exponential,
# adaptation, synaptic, axial, external) that cancel: an element-wise ulp distance of the result is unbounded near zero, while
# reassociation or contraction changes it by a few ulp of the largest term. `S` = sum of the absolute terms (Float64).
function _term_ulps(a::Float32, b::Float32, S::Float64)
    (a == b || (isnan(a) && isnan(b))) && return 0.0
    (isfinite(a) && isfinite(b) && isfinite(S)) || return Inf
    S == 0 && return Inf
    return abs(Float64(a) - Float64(b)) / ldexp(1.0, exponent(S) - 23)     # Float32 ulp at S (also above floatmax(Float32))
end

# term magnitudes of the derivatives (column k of Δv) of neuron i, from the inputs of the stage
function _term_scale(p::Tripod, Δvin, i, k, dt)
    (; v_pred, w_s, is, I, I_d, d1, d2, adex) = p
    (; C, gl, El, ΔT, a, τw) = adex
    vs, vd1, vd2 = Float64(v_pred[i, 1]), Float64(v_pred[i, 2]), Float64(v_pred[i, 3])
    ws = Float64(w_s[i] + Δvin[i, 4] * dt)
    ic1 = abs((vs - vd1) * d1.gax[i]); ic2 = abs((vs - vd2) * d2.gax[i])
    k == 1 && return (abs(gl * (El - vs)) + abs(gl * ΔT * exp((vs - p.θ[i]) / ΔT)) + abs(ws) + abs(is[i, 1]) + ic1 + ic2 + abs(I[i])) / C
    k == 2 && return (abs((d1.El[i] - vd1) * d1.gm[i]) + abs(is[i, 2]) + ic1 + abs(I_d[i])) / d1.C[i]
    k == 3 && return (abs((d2.El[i] - vd2) * d2.gm[i]) + abs(is[i, 3]) + ic2 + abs(I_d[i])) / d2.C[i]
    return (abs(a * (vs - El)) + abs(ws)) / τw
end
function _term_scale(p::BallAndStick, Δvin, i, k, dt)
    (; v_pred, w_s, is, Is, Id, d, adex) = p
    (; C, gl, El, ΔT, a, τw) = adex
    vs, vd = Float64(v_pred[i, 1]), Float64(v_pred[i, 2])
    ws = Float64(w_s[i] + Δvin[i, 3] * dt)
    icd = abs((vs - vd) * d.gax[i])
    k == 1 && return (abs(gl * (El - vs)) + abs(gl * ΔT * exp((vs - p.θ[i]) / ΔT)) + abs(ws) + abs(is[i, 1]) + icd + abs(Is[i])) / C
    k == 2 && return (abs((d.El[i] - vd) * d.gm[i]) + abs(is[i, 2]) + icd + abs(Id[i])) / d.C[i]
    return (abs(a * (vs - El)) + abs(ws)) / τw
end

# The two Heun stages in isolation, from the state of `q` (inputs already set): SNNModels `update_neuron!` and the reference get
# identical inputs at both stages (stage 2 starts from the reference stage-1 derivatives). Returns (max ulp stage 1, stage 2), in
# ulp of the terms (`_term_ulps`), and whether `v_pred` and `is` (computed before the loop) are bit-identical.
function _stage_ulps(q, T, dt)
    a = deepcopy(q); b = deepcopy(q)
    for x in (a, b)
        if T === Tripod
            SNNModels.update_synapses!(x, x.soma_syn, x.receptors_s, x.synvars_s, dt)
            SNNModels.update_synapses!(x, x.dend_syn, x.receptors_d1, x.synvars_d1, dt)
            SNNModels.update_synapses!(x, x.dend_syn, x.receptors_d2, x.synvars_d2, dt)
        else
            SNNModels.update_synapses!(x, x.soma_syn, x.receptors_s, x.synvars_s, dt)
            SNNModels.update_synapses!(x, x.dend_syn, x.receptors_d, x.synvars_d, dt)
        end
        fill!(x.Δv, 0f0)
    end
    us = [0.0, 0.0]; same = true
    for stage in 1:2
        Δvin = copy(b.Δv)
        SNNModels.update_neuron!(a, a.param, a.Δv, dt)
        _ref_update_neuron!(b, b.param, b.Δv, dt)
        same &= loop_same(a.v_pred, b.v_pred) && loop_same(a.is, b.is)
        for k in axes(b.Δv, 2), i in 1:b.N
            us[stage] = max(us[stage], _term_ulps(a.Δv[i, k], b.Δv[i, k], _term_scale(b, Δvin, i, k, dt)))
        end
        copyto!(a.Δv, b.Δv)        # stage 2 from the same stage-1 derivatives
    end
    return us[1], us[2], same
end

_rclose(x, y, rtol) = size(x) == size(y) && all(i -> x[i] == y[i] || abs(x[i] - y[i]) <= rtol * max(abs(x[i]), abs(y[i])), eachindex(x))
_maxrel(x, y) = isempty(x) ? 0.0 : maximum(i -> x[i] == y[i] ? 0.0 : Float64(abs(x[i] - y[i]) / max(abs(x[i]), abs(y[i]))), eachindex(x))

# copy the state of `q` into `p` (state fields and synaptic variables)
function _mc_sync!(p, q, T)
    for f in (T === Tripod ? _TRIPOD_STATE : _BAS_STATE)
        copyto!(getfield(p, f), getfield(q, f))
    end
    for sv in _mc_synvars(T), f in (:g, :h)
        copyto!(getfield(getfield(p, sv), f), getfield(getfield(q, sv), f))
    end
end

# Every step: `p` (SNNModels) restarts from the state of `q` (reference), both get the same input, and
#  - the two Heun stages in isolation (`_stage_ulps`) agree within `_STAGE_ULPS` ulp of the terms, `v_pred` and `is` bit-identical;
#  - the state stage alone is bit-identical: `r` = reference Heun stages, then `_tripod_state!`/`_bas_state!` of SNNModels, equals `q`;
#  - after the whole step: synaptic variables, `fire`, `tabs` bit-identical, potentials within `_STATE_RTOL`.
# Returns (ok, step, what, nspk, nfree, nbp, nrf, max ulp stage 1, max ulp stage 2, max relative difference of the potentials)
function _mc_pair(T, N; nsteps = 150, seed = 1, dt = 0.1f0, spike = PostSpike(τabs = 2ms, up = 1ms, At = 2mV))
    Random.seed!(seed)
    p = T(; N, spike)
    adex = p.adex
    exact, pot = T === Tripod ? (_TRIPOD_EXACT, _TRIPOD_POT) : (_BAS_EXACT, _BAS_POT)
    for f in (:v_s, :v_d1, :v_d2, :v_d)
        hasfield(T, f) && (getfield(p, f) .= adex.El .+ 35f0 .* rand(Float32, N))
    end
    p.w_s .= 20f0 .* rand(Float32, N)
    p.tabs .= Float32.(rand(0:12, N))                # backpropagation, refractory and free neurons from the start
    q = deepcopy(p)
    nspk = 0; nfree = 0; nbp = 0; nrf = 0; u1 = 0.0; u2 = 0.0; rel = 0.0
    n_abs = max(1, round(Int, spike.τabs / dt))
    n_ref = max(1, round(Int, spike.up / dt)) + n_abs
    state! = T === Tripod ? SNNModels._tripod_state! : SNNModels._bas_state!
    fail(s, what) = (false, s, what, nspk, nfree, nbp, nrf, u1, u2, rel)
    for s in 1:nsteps
        _mc_sync!(p, q, T)
        for f in (:I, :Is)
            hasfield(T, f) && (x = 800f0 .* rand(Float32, N); getfield(p, f) .= x; getfield(q, f) .= x)
        end
        for f in (:I_d, :Id)
            hasfield(T, f) && (x = 100f0 .* rand(Float32, N); getfield(p, f) .= x; getfield(q, f) .= x)
        end
        for (rp, rq) in zip(T === Tripod ? (p.receptors_s, p.receptors_d1, p.receptors_d2) : (p.receptors_s, p.receptors_d),
                            T === Tripod ? (q.receptors_s, q.receptors_d1, q.receptors_d2) : (q.receptors_s, q.receptors_d))
            for k in keys(rp)
                x = 40f0 .* rand(Float32, N) .* (rand(Float32, N) .< 0.1)
                getfield(rp, k) .= x; getfield(rq, k) .= x
            end
        end
        s1, s2, same = _stage_ulps(q, T, dt)
        u1 = max(u1, s1); u2 = max(u2, s2)
        same || return fail(s, :stage_inputs)
        (s1 <= _STAGE_ULPS && s2 <= _STAGE_ULPS) || return fail(s, :heun_stage)
        r = deepcopy(q)
        SNNModels.integrate!(p, p.param, dt)
        _ref_integrate!(q, q.param, dt)
        _ref_heun!(r, r.param, dt)
        state!(r, dt, 1:N, n_abs, n_ref)
        loop_state_equal(r, q, (T === Tripod ? _TRIPOD_STATE : _BAS_STATE)) || return fail(s, :state_stage)
        all(sv -> loop_state_equal(getfield(p, sv), getfield(q, sv), (:g, :h)), _mc_synvars(T)) || return fail(s, :synvars)
        loop_state_equal(p, q, exact) || return fail(s, loop_first_diff(p, q, exact))
        for f in pot
            _rclose(getfield(p, f), getfield(q, f), _STATE_RTOL) || return fail(s, f)
            rel = max(rel, _maxrel(getfield(p, f), getfield(q, f)))
        end
        nspk += count(p.fire); nbp += count(>(n_abs), p.tabs); nrf += count(t -> 0 < t <= n_abs, p.tabs); nfree += count(<=(0), p.tabs)
    end
    return (true, nsteps, nothing, nspk, nfree, nbp, nrf, u1, u2, rel)
end

@testset "Tripod and BallAndStick vectorised loops agree with the reference to a few ulp" begin
    ulps = Dict{Any,Any}()
    for T in (Tripod, BallAndStick), N in (0, 1, 2, 3, 7, 8, 9, 16, 17, 100, 501)
        r = _mc_pair(T, N)
        @test r[1]
        r[1] || @info "Tripod/BallAndStick comparison failed" T N step = r[2] what = r[3] stage_ulps = r[8:9]
        ulps[(T, N)] = r[8:10]
        if N >= 100
            @test r[4] > 0                     # spikes
            @test r[5] > 0                     # free neurons
            @test r[6] > 0                     # backpropagation period
            @test r[7] > 0                     # refractory period
        end
    end
    @test _mc_pair(Tripod, 64; dt = 0.05f0)[1]
    @test _mc_pair(BallAndStick, 64; spike = PostSpike(τabs = 1ms, up = 0.1ms))[1]
    for T in (Tripod, BallAndStick)
        rs = [v for (k, v) in ulps if k[1] === T]
        @info "Heun stage vs the @fastmath reference (ulp of the terms of Δv; relative difference of the potentials after one step)" T stage1 = maximum(first, rs) stage2 = maximum(r -> r[2], rs) state_rel = maximum(last, rs)
    end
    # the axial currents live in registers: `ic` is no longer written (field kept for compatibility)
    for T in (Tripod, BallAndStick)
        let p = T(N = 300)
            n = length(p.ic)
            hasfield(T, :I) ? (p.I .= 600f0) : (p.Is .= 600f0)
            for _ in 1:20
                SNNModels.integrate!(p, p.param, 0.1f0)
            end
            @test length(p.ic) == n && all(iszero, p.ic)
        end
    end
    # a state array shorter than N is a clear error
    for T in (Tripod, BallAndStick)
        let p = T(N = 8)
            resize!(p.tabs, 3)
            @test_throws DimensionMismatch SNNModels.integrate!(p, p.param, 0.1f0)
        end
        let p = T(N = 8, Δv = zeros(Float32, 3, T === Tripod ? 4 : 3))
            @test_throws DimensionMismatch SNNModels.integrate!(p, p.param, 0.1f0)
        end
    end
end

@testset "ReceptorSynapse and MultiReceptorSynapse stages are bit-identical to the reference" begin
    for N in (0, 1, 3, 8, 9, 100, 501), mk in (() -> ReceptorSynapse(), () -> MultiReceptorSynapse())
        Random.seed!(5)
        syn = mk()
        p = IF(; N, synapse = syn)
        for f in (:g, :h); getfield(p.synvars, f) .= 3f0 .* rand(Float32, size(getfield(p.synvars, f))...); end
        p.v .= -80f0 .+ 70f0 .* rand(Float32, N)
        q = deepcopy(p)
        for s in 1:100
            for k in keys(p.receptors)
                x = 30f0 .* rand(Float32, N) .* (rand(Float32, N) .< 0.2)
                getfield(p.receptors, k) .= x; getfield(q.receptors, k) .= x
            end
            SNNModels.update_synapses!(p, p.synapse, p.receptors, p.synvars, 0.1f0)
            if syn isa MultiReceptorSynapse
                _ref_update_synapses!(q, q.synapse, q.receptors, q.synvars, 0.1f0)
            else
                SNNModels.update_synapses!(q, q.synapse, q.receptors, q.synvars, 0.1f0)
            end
            @test loop_state_equal(p.synvars, q.synvars, (:g, :h))
            SNNModels.synaptic_current!(p, p.synapse, p.synvars)
            _ref_receptor_current!(q, q.synapse, q.synvars, q.v, q.syn_curr)
            @test loop_same(p.syn_curr, q.syn_curr)
        end
    end
    # a state or input shorter than N is a clear error
    let p = IF(N = 8, synapse = MultiReceptorSynapse())
        resize!(p.receptors.glu, 3)
        @test_throws DimensionMismatch SNNModels.update_synapses!(p, p.synapse, p.receptors, p.synvars, 0.1f0)
    end
    let p = IF(N = 8, synapse = ReceptorSynapse())
        resize!(p.syn_curr, 3)
        @test_throws DimensionMismatch SNNModels.synaptic_current!(p, p.synapse, p.synvars)
    end
end

true
