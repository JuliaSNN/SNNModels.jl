# Reference copies of the per-synapse loops of the event-driven vSTDP of SNNModels <= 2.3.0 (`@simd` without `ivdep`: the
# loops do not vectorise), used by test/syn/vstdp_ivdep.jl and bench/vstdp_event_bench.jl to check that the vectorised loops
# give the same bits. Test-only: not part of the library.
#
#   old_current_weights   the weights with the pending potentiation (`plasticity_value(syn, :W)`)
#   old_ev_spikes!        materialisation of the outgoing synapses of the presynaptic spikes
#   old_ltd!              depression at the presynaptic spikes (loop of `plasticity!`)
#
# `OldKernelvSTDP` has the fields of `vSTDPParameter` and the same state (`vSTDPVariables`); its `update_traces!` and
# `plasticity!` are the library ones with the three loops above, so that a whole `train!` can run the old loops. Build it from
# a library rule: `OldKernelvSTDP(vSTDPParameter(...))`.

using SNNModels
using SNNModels: vSTDPVariables, LTPParameter, AbstractSparseSynapse, presynaptic_spikes

# the weights are written to `Wcur` (a new vector by default)
function old_current_weights(c, vars::vSTDPVariables, Wcur = similar(c.W))
    isnan(vars.cache[1]) && return c.W
    colptr, I, W = c.colptr, c.I, c.W
    x, K, snap, cache = vars.x, vars.K, vars.snap, vars.cache
    T = eltype(W)
    dtA, Wmax = cache[7], cache[4]
    @inbounds for j in 1:length(colptr)-1
        a = dtA * x[j]
        @simd for s in colptr[j]:(colptr[j+1]-1)
            Wcur[s] = T(min(Float64(W[s]) + a * (K[I[s]] - snap[s]), Wmax))
        end
    end
    return Wcur
end

function old_ev_spikes!(c, vars::vSTDPVariables, param, spikes)
    colptr, I, W = c.colptr, c.I, c.W
    x, K, snap, pw, cache = vars.x, vars.K, vars.snap, vars.pw, vars.cache
    T = eltype(W)
    dtA, Wmax, qi, itx = cache[7], cache[4], cache[9], 1 / Float64(param.τx)
    qn = pw[2] * qi
    @inbounds for j in spikes
        a = dtA * x[j]
        @simd for s in colptr[j]:(colptr[j+1]-1)
            k = K[I[s]]
            W[s] = T(min(Float64(W[s]) + a * (k - snap[s]), Wmax))
            snap[s] = k
        end
        x[j] += qn * itx
    end
    vars.matstep[1] = vars.tick[1]
    return nothing
end

function old_ltd!(W, I, colptr, u, A_LTD, θ_LTD, Wmin, spikes)
    z = zero(eltype(W))
    @inbounds for j in spikes
        @simd for s in colptr[j]:(colptr[j+1]-1)
            w = W[s] - A_LTD * max(u[I[s]] - θ_LTD, z)
            W[s] = w < Wmin ? Wmin : w
        end
    end
    return nothing
end

struct OldKernelvSTDP{FT} <: LTPParameter
    A_LTD::FT
    A_LTP::FT
    θ_LTD::FT
    θ_LTP::FT
    τu::FT
    τv::FT
    τx::FT
    Wmax::FT
    Wmin::FT
    epoch::FT
    delay_uv::FT
    active::Vector{Bool}
end

# conversions from and to the library rule (same field values)
_lib_rule(p::OldKernelvSTDP{FT}) where {FT} = vSTDPParameter{FT}((getfield(p, k) for k in fieldnames(OldKernelvSTDP))...)
OldKernelvSTDP(p::vSTDPParameter{FT}) where {FT} = OldKernelvSTDP{FT}((getfield(p, k) for k in fieldnames(vSTDPParameter))...)

SNNModels.needs_transpose(::OldKernelvSTDP) = false
SNNModels.plasticityvariables(p::OldKernelvSTDP, Npre, Npost) = SNNModels.plasticityvariables(_lib_rule(p), Npre, Npost)

function SNNModels.update_traces!(c::AbstractSparseSynapse, p::OldKernelvSTDP, vars::vSTDPVariables, dt::AbstractFloat, T::Time)
    param = _lib_rule(p)
    SNNModels._ev_check!(c, vars, param, dt)
    old_ev_spikes!(c, vars, param, presynaptic_spikes(c))
    return nothing
end

# copy of the library `plasticity!` of vSTDPParameter with the old loops
function SNNModels.plasticity!(c::AbstractSparseSynapse, p::OldKernelvSTDP, vars::vSTDPVariables, dt::AbstractFloat, T::Time)
    param = _lib_rule(p)
    colptr, I, W, v_post = c.colptr, c.I, c.W, c.v_post
    u, v, K, pw, steps, tick, matstep, cache, initialized =
        vars.u, vars.v, vars.K, vars.pw, vars.steps, vars.tick, vars.matstep, vars.cache, vars.initialized
    A_LTD, θ_LTD, θ_LTP, τu, τv, Wmin = param.A_LTD, param.θ_LTD, param.θ_LTP, param.τu, param.τv, param.Wmin
    z = zero(eltype(W))
    if !initialized[1]
        u .= v_post
        v .= v_post
        initialized[1] = true
    end
    SNNModels._ev_check!(c, vars, param, dt)
    spikes = presynaptic_spikes(c)
    matstep[1] == tick[1] || old_ev_spikes!(c, vars, param, spikes)
    q, qi = cache[6], cache[9]
    pq = pw[1] * q
    pw[1] = pq
    pw[2] = pw[2] * qi
    D = SNNModels._uv_delay_steps(param, dt)
    if D == 0
        @inbounds @fastmath for i in eachindex(u)
            u[i] += dt * (-u[i] + v_post[i]) / τu
            v[i] += dt * (-v[i] + v_post[i]) / τv
            g = max(v[i] - θ_LTD, z) * max(v_post[i] - θ_LTP, z)
            K[i] += pq * Float64(g)
        end
    else
        off = SNNModels._uv_delay_slot!(vars, v_post, D)
        vbuf = vars.vbuf
        @inbounds @fastmath for i in eachindex(u)
            Vd = vbuf[off+i]
            vbuf[off+i] = v_post[i]
            u[i] += dt * (-u[i] + Vd) / τu
            v[i] += dt * (-v[i] + Vd) / τv
            g = max(v[i] - θ_LTD, z) * max(v_post[i] - θ_LTP, z)
            K[i] += pq * Float64(g)
        end
    end
    old_ltd!(W, I, colptr, u, A_LTD, θ_LTD, Wmin, spikes)
    tick[1] += 1
    steps[1] += 1
    steps[1] >= cache[8] && SNNModels._ev_rebase!(c, vars)
    return nothing
end
