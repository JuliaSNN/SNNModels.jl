#=
Helpers of the voltage-STDP preset tests (`plasticity_presets.jl`, `vstdp_tripod.jl`)
=====================================================================================

Inputs with prescribed presynaptic spikes and postsynaptic potentials, a driver that steps a plastic
synapse through `train!`, and an independent Float64 implementation of the update of the Tripod
plasticity code. Not part of the package.
=#

using SNNModels
using Random

# time step of the Tripod code (ms); the constants are defined once when both test files are included
if !@isdefined(_PDT)
    const _PDT = 0.1f0
end

# Presynaptic spikes (list per step) and postsynaptic potentials (npost x nsteps): a slow sinusoid around
# -70 mV and, at Poisson postsynaptic spikes, +100 mV for 1 ms then a -8 mV afterhyperpolarisation, so the
# potential crosses every θ_LTP of the presets (peak about +30 mV).
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

# weights as a (post, pre) matrix
function _pp_mat(syn)
    SNNModels.ensure_transpose!(syn)
    W = plasticity_value(syn, :W)
    M = zeros(Float64, length(syn.fireI), length(syn.fireJ))
    for s in eachindex(W)
        M[syn.I[s], syn.J[s]] = W[s]
    end
    return M
end

# Independent Float64 implementation of the update of the Tripod code (`learning_functions.jl`,
# `bonoclopath_tripod.jl`), with the filters advanced once per step and a delay of D steps:
#   x += spike; x -= dt x / τx                          (trace jumps by 1, decays in the same step)
#   u += dt (V(n - D) - u) / τu, v += dt (V(n - D) - v) / τv
#   at a presynaptic spike: if u > θ⁻, w -= dt a⁻ (u - θ⁻)
#   if v > θ⁻ and V > θ⁺: w += dt x a⁺ (V - θ⁺) (v - θ⁻)
# u, v and the delay line start at the potential of the first step (as the library rule).
function _tripod_code_vstdp(spikes, V, set; D, w0, npre, dt = Float64(_PDT))
    npost, nsteps = size(V)
    W = fill(Float64(w0), npost, npre)
    x = zeros(npre)
    u = Float64.(V[:, 1])
    v = copy(u)
    for n in 1:nsteps
        Vn = Float64.(view(V, :, n))
        Vd = Float64.(view(V, :, max(1, n - D)))

        # presynaptic trace of the Tripod code
        for j in spikes[n]
            x[j] += 1
        end
        @. x -= dt * x / set.τx

        # filters of the delayed potential
        @. u += dt * (Vd - u) / set.τu
        @. v += dt * (Vd - v) / set.τv

        # depression at the presynaptic spikes, potentiation of all synapses
        for i in 1:npost
            if u[i] > set.θ_ltd
                for j in spikes[n]
                    W[i, j] -= dt * set.a_ltd * (u[i] - set.θ_ltd)
                end
            end
            if v[i] > set.θ_ltd && Vn[i] > set.θ_ltp
                for j in 1:npre
                    W[i, j] += dt * x[j] * set.a_ltp * (Vn[i] - set.θ_ltp) * (v[i] - set.θ_ltd)
                end
            end
        end
    end
    return W
end

if !@isdefined(_PRESETS)
    const _PRESETS = (
        (lkd_vstdp, SNNModels._LKD_VSTDP),
        (clopath_vstdp_visualcortex, SNNModels._CLOPATH_VC_VSTDP),
        (bono_vstdp, SNNModels._BONO_VSTDP),
    )
end
