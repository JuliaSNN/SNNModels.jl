# Reference copy of the SynapseNormalization kernels of SNNModels <= 2.3.0 (row by row through the transposed index
# `index`), used by test/syn/normalization_csc.jl and bench/normalisation_bench.jl to check that the CSC-order kernels give
# the same bits. Test-only: not part of the library.
#
# `OldNorm` is a NormParam whose step runs these kernels, so that a whole `train!` can use them.

using SNNModels
using SNNModels: NormParam

@snn_kw struct OldNorm{FT = Float32} <: NormParam
    τ::FT
    additive::Bool = false
end

SNNModels.plasticity!(c::SynapseNormalization, param::OldNorm) = old_norm_step!(c, param.additive)

# One normalization step with the old kernels (multiplicative or additive).
function old_norm_step!(c::SynapseNormalization, additive::Bool)
    W1, W0, μ, synapses = c.W1, c.W0, c.μ, c.synapses
    foreach(SNNModels.sync_weights!, synapses)
    foreach(SNNModels.ensure_transpose!, synapses)
    fill!(W1, zero(eltype(W1)))
    nsyn = zeros(Int, length(W1))
    for syn in synapses
        _old_count_inputs!(nsyn, syn.rowptr)
    end
    for syn in synapses
        _old_sum_inputs!(W1, syn.rowptr, syn.W, syn.index)
    end
    @inbounds for i in eachindex(μ)
        if additive
            μ[i] = nsyn[i] > 0 ? (W0[i] - W1[i]) / nsyn[i] : zero(eltype(μ))
        else
            μ[i] = W1[i] > 0 ? W0[i] / W1[i] : one(eltype(μ))
        end
    end
    for syn in synapses
        _old_apply_norm!(syn.W, syn.rowptr, syn.index, μ, additive)
    end
end

function _old_count_inputs!(nsyn, rowptr)
    @inbounds for i = 1:(length(rowptr)-1)
        nsyn[i] += rowptr[i+1] - rowptr[i]
    end
    return nsyn
end

function _old_sum_inputs!(W1, rowptr, W, index)
    @inbounds for i = 1:(length(rowptr)-1)
        s = W1[i]
        for j = rowptr[i]:(rowptr[i+1]-1)
            s += W[index[j]]
        end
        W1[i] = s
    end
    return W1
end

function _old_apply_norm!(W, rowptr, index, μ, additive::Bool)
    @inbounds for i = 1:(length(rowptr)-1)
        m = μ[i]
        if additive
            for j = rowptr[i]:(rowptr[i+1]-1)
                W[index[j]] += m
            end
        else
            for j = rowptr[i]:(rowptr[i+1]-1)
                W[index[j]] *= m
            end
        end
    end
    return W
end

# ── AggregateScaling: the step of SNNModels <= 2.3.0 ──────────────────────────

function old_aggregate_step!(c::AggregateScaling, param::AggregateScalingParameter)
    Wt, WT, μ, synapses = c.Wt, c.WT, c.μ, c.synapses
    Wmin = param.Wmin
    foreach(SNNModels.sync_weights!, synapses)
    foreach(SNNModels.ensure_transpose!, synapses)
    fill!(Wt, zero(eltype(Wt)))
    nsyn = zeros(Int, length(Wt))
    for syn in synapses
        rowptr = syn.rowptr
        for i = 1:(length(rowptr)-1)
            nsyn[i] += rowptr[i+1] - rowptr[i]
        end
    end
    for syn in synapses
        rowptr, W, index = syn.rowptr, syn.W, syn.index
        for i = 1:(length(rowptr)-1)
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1)
                Wt[i] += W[index[j]]
            end
        end
    end
    @inbounds for i in eachindex(μ)
        μ[i] = Wt[i] > 0 ? max(WT[i] - nsyn[i] * Wmin, zero(eltype(Wt))) / Wt[i] : one(eltype(μ))
    end
    for syn in synapses
        rowptr, W, index = syn.rowptr, syn.W, syn.index
        for i = 1:(length(rowptr)-1)
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1)
                Wt[i] > 0 && (W[index[j]] = W[index[j]] * μ[i] + Wmin)
            end
        end
    end
end
