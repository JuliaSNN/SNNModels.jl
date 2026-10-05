@doc raw"""
    vSTDPParameter(; A_LTD = 8e-4, A_LTP = 1.4e-3, θ_LTD = -70mV, θ_LTP = -49mV,
                     τu = 20ms, τv = 7ms, τx = 15ms, Wmax = 30pF, Wmin = 0.1pF, active = [true])

Voltage-based STDP in the form of Clopath et al. (2010): depression is triggered by a
presynaptic spike and depends on a low-pass filtered postsynaptic membrane potential;
potentiation depends on a presynaptic spike trace, on the instantaneous postsynaptic
potential and on a second low-pass filter of it. It is a subtype of `LTPParameter`; pass it
as `LTPParam` to a `SpikingSynapse` that targets a population with a membrane potential
(`v_post` of the synapse). Its state is `vSTDPVariables`.

# Equations
With ``V_i`` the postsynaptic potential (`v_post`), ``u_i``, ``v_i`` its low-pass filters,
``x_j`` the presynaptic trace and ``[z]_+ = \max(z, 0)``:
```math
\begin{aligned}
τ_x \frac{dx_j}{dt} &= -x_j + S_j, \qquad
τ_u \frac{du_i}{dt} = -u_i + V_i, \qquad
τ_v \frac{dv_i}{dt} = -v_i + V_i,\\
Δw_{ij}^{LTD} &= -A_{LTD}\,[u_i - θ_{LTD}]_+ \quad \text{at each presynaptic spike of } j,\\
Δw_{ij}^{LTP} &= A_{LTP}\, x_j\, [v_i - θ_{LTD}]_+\, [V_i - θ_{LTP}]_+ \quad \text{at every time step}.
\end{aligned}
```
``S_j`` is 1 in the step in which ``j`` fires and 0 otherwise, so a spike increases ``x_j``
by ``dt/τ_x``. Note that both thresholds of the LTP term are as in the code: the filtered
potential ``v`` is compared with `θ_LTD`, the instantaneous one with `θ_LTP`.

# Integration
`plasticity!` (only under `train!`), per step: (1) forward Euler for ``x`` (all presynaptic
neurons, including this step's spikes), (2) forward Euler for ``u`` and ``v`` (all
postsynaptic neurons, driven by the current `v_post`), (3) for each presynaptic neuron ``j``
(threaded over chunks of ``j``): if ``j`` fired, LTD on its outgoing synapses and clamp at
`Wmin`; then, whether or not ``j`` fired, the LTP increment on all its outgoing synapses and
clamp at `Wmax`. The LTP increment is added per step and is not multiplied by `dt`.
The traces start at 0 mV (`vSTDPVariables` defaults), not at the resting potential.

# Fields
- `A_LTD::FT = 8 * 10e-5pA / mV` (= 8e-4): LTD amplitude per mV.
- `A_LTP::FT = 14 * 10e-5pA / (mV * mV)` (= 1.4e-3): LTP amplitude per mV².
- `θ_LTD::FT = -70mV`: LTD threshold (also used for the filtered potential in the LTP term).
- `θ_LTP::FT = -49mV`: LTP threshold on the instantaneous potential.
- `τu::FT = 20ms`: time constant of ``u`` (LTD filter).
- `τv::FT = 7ms`: time constant of ``v`` (LTP filter).
- `τx::FT = 15ms`: time constant of the presynaptic trace ``x``.
- `Wmax::FT = 30pF`, `Wmin::FT = 0.1pF`: weight bounds.
- `active::VBT = [true]`: present in the parameter type; activation is controlled by the
  `active` field of `vSTDPVariables` (see `set_LTP!`).

# References
Clopath, C., Büsing, L., Vasilaki, E. & Gerstner, W. (2010). Connectivity reflects coding: a
model of voltage-based STDP with homeostasis. Nature Neuroscience 13, 344-352. The default
parameter values are not referenced in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.AdEx(N = 10, param = SNN.AdExParameter())
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 1.0), LTPParam = SNN.vSTDPParameter())
SNN.train!(model = SNN.compose(; E, P, syn), duration = 200ms)
```
"""
vSTDPParameter

@snn_kw struct vSTDPParameter{FT = Float32} <: LTPParameter
    A_LTD::FT = 8 * 10e-5pA / mV
    A_LTP::FT = 14 * 10e-5pA / (mV * mV)
    θ_LTD::FT = -70mV
    θ_LTP::FT = -49mV
    τu::FT = 20ms
    τv::FT = 7ms
    τx::FT = 15ms
    Wmax::FT = 30.0pF
    Wmin::FT = 0.1pF
    active::VBT = [true]
end

"""
    vSTDPVariables(; Npre, Npost, u = zeros(Npost), v = zeros(Npost), x = zeros(Npre), active = [true])

State of `vSTDPParameter`: `u`, `v` (low-pass filters of the postsynaptic potential with
time constants `τu`, `τv`, length `Npost`, mV), `x` (presynaptic spike trace, length `Npre`)
and the `active` flag. All traces start at 0.
"""
vSTDPVariables

@snn_kw struct vSTDPVariables{VFT = Vector{Float32},IT = Int} <: LTPVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npost) # postsynaptic potential filtered with τu (LTD)
    v::VFT = zeros(Npost) # postsynaptic potential filtered with τv (LTP)
    x::VFT = zeros(Npre) # presynaptic spike trace
    active::VBT = [true]
end

function plasticityvariables(param::T, Npre, Npost) where {T<:vSTDPParameter}
    return vSTDPVariables(Npre = Npre, Npost = Npost)
end

"""
    plasticity!(c::AbstractSparseSynapse, param::vSTDPParameter, variables::vSTDPVariables, dt::Float32, T::Time)

One step of the voltage-based STDP rule `vSTDPParameter` (see its docstring for the
equations and the update order): updates the traces in `variables` and the weights `c.W` in
place, clamping them to `[Wmin, Wmax]`. Called by `train!`, never by `sim!`.
"""
function plasticity!(
    c::PT,
    param::vSTDPParameter,
    plasticity::vSTDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, index = c
    @unpack u, v, x = plasticity
    @unpack A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin = param
    # R(x::Float32) = x < 0.0f0 ? 0.0f0 : x

    # update pre-synaptic spike trace
    @fastmath @inbounds begin
        for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
            x[j] += dt * (-x[j] + fireJ[j]) / τx
        end

        Is = 1:(length(rowptr)-1)
        @turbo for i in eachindex(Is) # Iterate over postsynaptic neurons
            u[i] += dt * (-u[i] + v_post[i]) / τu # postsynaptic neuron
            v[i] += dt * (-v[i] + v_post[i]) / τv # postsynaptic neuron
        end
        # @simd for s = colptr[j]:(colptr[j+1]-1) 
        chunks = Iterators.partition(eachindex(fireJ), cld(length(fireJ), Threads.nthreads())) |> collect
        Threads.@threads for c in eachindex(chunks) # Iterate over presynaptic neurons
            @simd for j in chunks[c]
                if fireJ[j]
                    @turbo for s = colptr[j]:(colptr[j+1]-1)
                        W[s] += - A_LTD * clamp(u[I[s]] - θ_LTD, 0.0f0, Inf)
                        W[s] < Wmin && (W[s] = Wmin)
                    end
                end
                @turbo for s = colptr[j]:(colptr[j+1]-1)
                    W[s] += A_LTP *
                        x[j] *
                        clamp(v[I[s]] - θ_LTD, 0.0f0, Inf) *
                        clamp(v_post[I[s]] - θ_LTP, 0.0f0, Inf)
                    W[s] > Wmax && (W[s] = Wmax)
                end
            end
        end

    end
end

export vSTDPParameter, vSTDPVariables, plasticityvariables, plasticity!


# @inbounds @fastmath @simd for i in eachindex(fireI) # Iterate over postsynaptic neurons
#     u[i] += dt * (-u[i] + v_post[i]) / τu # postsynaptic neuron
#     v[i] += dt * (-v[i] + v_post[i]) / τv # postsynaptic neuron
# end

# @inbounds @fastmath  for i in eachindex(Is) # Iterate over postsynaptic neurons
#     ltd_v = (v[i] - θ_LTD)
#     ltp = (v_post[i] - θ_LTP)
#     @simd for s = rowptr[i]:(rowptr[i+1]-1)
#         j = J[index[s]]
#         if fireJ[j] && (u[i] - θ_LTD) > 0.0f0
#             W[index[s]] += -A_LTD * (u[i] - θ_LTD)
#         end
#         if ltp > 0.0f0 && ltd_v > 0.0f0
#             W[index[s]] += A_LTP * x[j] * ltp * ltd_v
#         end
#     end
# end
