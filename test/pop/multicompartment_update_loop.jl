# Bit-identity of the vectorised / branchless `integrate!` of `Tripod` and `BallAndStick` (and of the receptor synapse stage they use)
# against the original loops of SNNModels <= 1.9, kept here as the reference.
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
    (; N, v_s, w_s, v_d1, v_d2) = p
    (; fire, θ, tabs) = p
    (; Δv, Δv_temp, is) = p
    (; synvars_s, synvars_d1, synvars_d2, d1, d2, I_d) = p
    (; receptors_d1, receptors_d2, receptors_s) = p
    (; spike, adex, soma_syn, dend_syn, Vspike) = p
    (; AP_membrane, up, τabs, At, τA) = spike
    (; Vr, Vt, b) = adex
    n_abs = max(1, round(Int, τabs / dt))
    n_ref = max(1, round(Int, up / dt)) + n_abs
    SNNModels.update_synapses!(p, soma_syn, receptors_s, synvars_s, dt)
    SNNModels.update_synapses!(p, dend_syn, receptors_d1, synvars_d1, dt)
    SNNModels.update_synapses!(p, dend_syn, receptors_d2, synvars_d2, dt)
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)
    _ref_update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    _ref_update_neuron!(p, param, Δv, dt)
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
    (; N, v_s, w_s, v_d) = p
    (; fire, θ, tabs) = p
    (; Δv, Δv_temp, is) = p
    (; synvars_s, synvars_d, d) = p
    (; receptors_d, receptors_s) = p
    (; spike, adex, soma_syn, dend_syn, Vspike) = p
    (; AP_membrane, up, τabs, At, τA) = spike
    (; Vr, Vt, b) = adex
    n_abs = max(1, round(Int, τabs / dt))
    n_ref = max(1, round(Int, up / dt)) + n_abs
    SNNModels.update_synapses!(p, soma_syn, receptors_s, synvars_s, dt)
    SNNModels.update_synapses!(p, dend_syn, receptors_d, synvars_d, dt)
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)
    _ref_update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    _ref_update_neuron!(p, param, Δv, dt)
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

const _TRIPOD_FIELDS = (:v_s, :w_s, :v_d1, :v_d2, :fire, :tabs, :θ, :Δv, :Δv_temp, :is, :ic, :v_pred, :I, :I_d)
const _BAS_FIELDS = (:v_s, :w_s, :v_d, :fire, :tabs, :θ, :Δv, :Δv_temp, :is, :ic, :v_pred, :Is, :Id)

function _mc_pair(T, N; nsteps = 150, seed = 1, dt = 0.1f0, spike = PostSpike(τabs = 2ms, up = 1ms, At = 2mV))
    Random.seed!(seed)
    p = T(; N, spike)
    adex = p.adex
    fld = T === Tripod ? _TRIPOD_FIELDS : _BAS_FIELDS
    for f in (:v_s, :v_d1, :v_d2, :v_d)
        hasfield(T, f) && (getfield(p, f) .= adex.El .+ 35f0 .* rand(Float32, N))
    end
    p.w_s .= 20f0 .* rand(Float32, N)
    p.tabs .= Float32.(rand(0:12, N))                # backpropagation, refractory and free neurons from the start
    q = deepcopy(p)
    nspk = 0; nfree = 0; nbp = 0; nrf = 0
    n_abs = max(1, round(Int, spike.τabs / dt))
    for s in 1:nsteps
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
        SNNModels.integrate!(p, p.param, dt)
        _ref_integrate!(q, q.param, dt)
        loop_state_equal(p, q, fld) || return (false, s, loop_first_diff(p, q, fld), nspk, nfree)
        syn_ok = T === Tripod ?
                 all(sv -> loop_state_equal(getfield(p, sv), getfield(q, sv), (:g, :h)), (:synvars_s, :synvars_d1, :synvars_d2)) :
                 all(sv -> loop_state_equal(getfield(p, sv), getfield(q, sv), (:g, :h)), (:synvars_s, :synvars_d))
        syn_ok || return (false, s, :synvars, nspk, nfree)
        nspk += count(p.fire); nbp += count(>(n_abs), p.tabs); nrf += count(t -> 0 < t <= n_abs, p.tabs); nfree += count(<=(0), p.tabs)
    end
    return (true, nsteps, nothing, nspk, nfree, nbp, nrf)
end

@testset "Tripod and BallAndStick vectorised loops are bit-identical to the reference" begin
    for T in (Tripod, BallAndStick), N in (0, 1, 2, 3, 7, 8, 9, 16, 17, 100, 501)
        r = _mc_pair(T, N)
        @test r[1]
        if N >= 100
            @test r[4] > 0                     # spikes
            @test r[5] > 0                     # free neurons
            @test r[6] > 0                     # backpropagation period
            @test r[7] > 0                     # refractory period
        end
    end
    @test _mc_pair(Tripod, 64; dt = 0.05f0)[1]
    @test _mc_pair(BallAndStick, 64; spike = PostSpike(τabs = 1ms, up = 0.1ms))[1]
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
