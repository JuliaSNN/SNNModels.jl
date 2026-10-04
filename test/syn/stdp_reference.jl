# Reference (pre event-driven) STDP kernels, kept ONLY for regression tests.
#
# These are verbatim ports of the clock-driven implementations of STDPGerstner,
# STDPConfavreux2025 and STDPMexicanHat that lived in
# src/connections/sparse_plasticity/STDP_traces.jl before the event-driven rewrite
# (SNNModels dev @ 614cfca), with the threading removed (it does not change the
# arithmetic: every synapse was updated independently).
#
# They operate on a plain synapse-like object (anything with rowptr, colptr, I, J,
# index, W, fireI, fireJ) and on a reference state created by `ref_state`.
# Not exported, not part of the package.
#
# Two switches reproduce the two behaviours that the rewrite changed on purpose:
#   amplitude_quirk   = true : STDPGerstner traces are incremented by A_pre/A_post
#                              and the weight update multiplies by A_pre/A_post again
#                              (effective amplitude A^2, sign of A lost).
#   coincidence_quirk = true : a neuron that fires in the current step contributes
#                              0 (its whole trace, including earlier spikes, is
#                              ignored in this step). With `false`, the trace value
#                              before this step's increment is used (Auryn/Brian2).
# With both switches `true` the functions reproduce the old code exactly.

ref_state(Npre, Npost) = (
    tpre = zeros(Float32, Npre),
    tpost = zeros(Float32, Npost),
    last_pre = zeros(Float32, Npre),
    last_post = zeros(Float32, Npost),
    Δpre = zeros(Float32, Npre),
    Δpost = zeros(Float32, Npost),
)

# Lazy-trace update of the old implementation. `inc` is the trace increment.
function _ref_traces!(tr, last, Δ, fire, t, τ, inc, coincidence_quirk)
    @inbounds for j in eachindex(fire)
        if coincidence_quirk
            if fire[j]
                tr[j] = tr[j] * exp(-(t - last[j]) / τ) + inc
                last[j] = t
            end
            Δ[j] = t > last[j] ? tr[j] * exp(-(t - last[j]) / τ) : 0.0f0
        else
            # value of the trace before this step's spike is added
            Δ[j] = tr[j] * exp(-(t - last[j]) / τ)
            if fire[j]
                tr[j] = Δ[j] + inc
                last[j] = t
            end
        end
    end
end

function ref_plasticity!(
    c,
    param::STDPGerstner,
    st,
    t::Float32;
    amplitude_quirk = true,
    coincidence_quirk = true,
)
    (; I, J, W, fireI, fireJ) = c
    (; A_pre, A_post, τpre, τpost, Wmax, Wmin) = param
    inc_pre = amplitude_quirk ? A_pre : 1.0f0
    inc_post = amplitude_quirk ? A_post : 1.0f0
    _ref_traces!(st.tpre, st.last_pre, st.Δpre, fireJ, t, τpre, inc_pre, coincidence_quirk)
    _ref_traces!(st.tpost, st.last_post, st.Δpost, fireI, t, τpost, inc_post, coincidence_quirk)
    @inbounds for s in eachindex(W)
        i, j = I[s], J[s]
        if fireI[i]
            W[s] += A_pre * st.Δpre[j]
        end
        if fireJ[j]
            W[s] += A_post * st.Δpost[i]
        end
        W[s] = clamp(W[s], Wmin, Wmax)
    end
end

function ref_plasticity!(
    c,
    param::STDPConfavreux2025,
    st,
    t::Float32;
    coincidence_quirk = true,
)
    (; I, J, W, fireI, fireJ) = c
    (; η, α, β, κ, γ, τpre, τpost, Wmin, Wmax) = param
    _ref_traces!(st.tpre, st.last_pre, st.Δpre, fireJ, t, τpre, 1.0f0, coincidence_quirk)
    _ref_traces!(st.tpost, st.last_post, st.Δpost, fireI, t, τpost, 1.0f0, coincidence_quirk)
    @inbounds for s in eachindex(W)
        i, j = I[s], J[s]
        if fireI[i]
            W[s] += η * (γ * st.Δpre[j] + β)
        end
        if fireJ[j]
            W[s] += η * (κ * st.Δpost[i] + α)
        end
        W[s] = clamp(W[s], Wmin, Wmax)
    end
end

# Old MexicanHat: Euler traces, O(nnz) scans of all rows and columns, clamp all W.
_ref_mexican_hat(x::Float32) = (1 - x) * exp(-x / sqrt(2)) |> x -> isnan(x) ? 0 : x

function ref_plasticity!(c, param::STDPMexicanHat, st, dt::Float32)
    (; rowptr, colptr, I, J, index, W, fireI, fireJ) = c
    (; A, τ, Wmax, Wmin) = param
    tpre, tpost = st.tpre, st.tpost
    @inbounds @fastmath begin
        for i in eachindex(fireI)
            tpost[i] += dt * (-tpost[i]) / τ
        end
        for i in findall(fireI)
            tpost[i] += 1
        end
        for j in eachindex(fireJ)
            tpre[j] += dt * (-tpre[j]) / τ
        end
        for j in findall(fireJ)
            tpre[j] += 1
        end
        for i = 1:(length(rowptr)-1)
            for st = rowptr[i]:(rowptr[i+1]-1)
                s = index[st]
                if fireJ[J[s]] && abs(tpost[i] * tpre[J[s]]) > 0.0f0
                    W[s] += A * _ref_mexican_hat((log(tpre[J[s]] / tpost[i]))^2)
                end
            end
        end
        for j = 1:(length(colptr)-1)
            for s = colptr[j]:(colptr[j+1]-1)
                if fireI[I[s]] && abs(tpost[I[s]] * tpre[j]) > 0.0f0
                    W[s] += A * _ref_mexican_hat(log(tpre[j] / tpost[I[s]])^2)
                end
            end
        end
    end
    @inbounds for i in eachindex(W)
        W[i] = clamp(W[i], Wmin, Wmax)
    end
end
