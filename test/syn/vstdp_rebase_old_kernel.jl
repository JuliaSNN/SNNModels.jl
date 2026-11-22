# Reference copy of the epoch restart `_ev_rebase!` of the event-driven vSTDP of SNNModels <= 2.3.0 (one serial pass over
# the weights, then `fill!` of the snapshots), used by test/syn/vstdp_rebase.jl and bench/vstdp_rebase_bench.jl to check
# that the fused, vectorised and threaded rebase gives the same bits. Test-only: not part of the library.
#
# `OldRebaseNorm` is a NormParam whose step first rebases every vSTDP synapse with this copy and then runs the library
# normalisation step, so that a whole `train!` can use the old rebase.

using SNNModels
using SNNModels: NormParam, vSTDPVariables

function old_ev_rebase!(c, vars::vSTDPVariables)
    colptr, I, W = c.colptr, c.I, c.W
    x, K, snap, pw, steps, cache = vars.x, vars.K, vars.snap, vars.pw, vars.steps, vars.cache
    T = eltype(W)
    dtA, Wmax, qp = cache[7], cache[4], pw[1]
    @inbounds for j in 1:length(colptr)-1
        a = dtA * x[j]
        @simd for s in colptr[j]:(colptr[j+1]-1)
            W[s] = T(min(Float64(W[s]) + a * (K[I[s]] - snap[s]), Wmax))
        end
        x[j] = x[j] * qp
    end
    fill!(K, 0.0)
    fill!(snap, 0.0)
    steps[1] = 0
    pw[1] = 1.0
    pw[2] = 1.0
    return nothing
end

# `sync_weights!` with the old rebase (a no-op before the first step, as the library method)
old_sync_weights!(c) = (v = c.LTPVars; v isa vSTDPVariables && !isnan(v.cache[1]) && old_ev_rebase!(c, v); c)

@snn_kw struct OldRebaseNorm{FT = Float32} <: NormParam
    τ::FT
end

# The old rebase on every synapse, then the library step on the rebased state (its own rebase finds K = 0 and the
# snapshots at 0 and leaves the weights as they are).
function SNNModels.plasticity!(c::SynapseNormalization, param::OldRebaseNorm)
    foreach(old_sync_weights!, c.synapses)
    SNNModels.plasticity!(c, MultiplicativeNorm(τ = param.τ))
end
