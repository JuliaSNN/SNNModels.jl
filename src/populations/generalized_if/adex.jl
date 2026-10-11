@doc raw"""
    AdExParameter{FT = Float32}(; C, gl, τm, R, Vt = -50mV, Vr = -70.6mV, El = -70.6mV,
                                 ΔT = 2mV, τw = 144ms, a = 4nS, b = 80.5pA)

Neuron parameters of the adaptive exponential integrate-and-fire model (`AdEx`), in the
parameterisation of Brette and Gerstner (2005). The code comment notes that the original paper
uses a leak conductance of 30 nS; the default here is 40 nS.

`AdExParameter` is mutable. With `FT = Vector{Float32}` (see `make_heterogeneous`) every
parameter is per neuron and `AdEx` uses the heterogeneous method of `update_neuron!`.

# Membrane parameters
`C`, `gl`, `R` and `τm` satisfy `τm = C / gl` and `R = 1nS / gl`. `AdEx` integrates with `τm`
and `R`; `Tripod` and `BallAndStick`, which hold an `AdExParameter` in their `adex` field,
integrate with `C` and `gl`. Give them as a pair of independent values, any of (C, gl), (C, R),
(C, τm), (gl, τm), (R, τm), and the other two are derived; give none and the default pair
`C = 281pF`, `gl = 40nS` is used. A single value is an error: a given value is never combined with
a default. More values are accepted if consistent. Changing one of them later (`@update!`,
property assignment `p.τm = x`, [`with_membrane`](@ref), `make_heterogeneous`) keeps the
others consistent: `τm` keeps `gl` (changes `C`), `C` keeps `gl` (changes `τm`), `gl` or `R`
keeps `C` (changes `τm`). See [`resolve_membrane`](@ref) and [`membrane_update`](@ref).
Up to SNNModels 1.8.x the four fields were independent after construction: setting `τm` had no
effect on Tripod and BallAndStick, and `τm` alone was combined with the default `C`, `gl`.

# Fields
- `C::FT`: membrane capacitance (pF); default 281 pF.
- `gl::FT`: leak conductance (nS); default 40 nS.
- `Vt::FT = -50mV`: threshold of the exponential term (rheobase threshold), also the resting
  value of the adaptive threshold ``θ`` (mV). It is not the spike-detection threshold (0 mV).
- `Vr::FT = -70.6mV`: reset potential (mV).
- `El::FT = -70.6mV`: leak reversal potential (mV).
- `τm::FT`: membrane time constant (ms), `C / gl`, 7.025 ms with the defaults.
- `R::FT`: membrane resistance (GΩ), `1nS / gl`, 0.025 GΩ with the defaults.
- `ΔT::FT = 2mV`: slope factor of the exponential term (mV); a negative value removes the
  exponential term.
- `τw::FT = 144ms`: adaptation time constant (ms).
- `a::FT = 4nS`: subthreshold adaptation conductance (nS).
- `b::FT = 80.5pA`: spike-triggered adaptation increment (pA).

# References
Brette R., Gerstner W. (2005). Adaptive exponential integrate-and-fire model as an effective
description of neuronal activity. J. Neurophysiol. 94:3637-3642.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
param = SNN.AdExParameter(a = 0nS, b = 0pA)   # exponential IF without adaptation
E = SNN.AdEx(N = 10, param = param)
```
"""
AdExParameter
@snn_kw mutable struct AdExParameter{FT = Float32} <: AbstractGeneralizedIFParameter
    C::FT = NaN32 # Membrane capacitance (pF); default pair C = 281pF, gl = 40nS, see resolve_membrane
    gl::FT = NaN32 # (nS) leak conductance #BretteGerstner2005 says 30 nS
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = -70.6mV # Resting membrane potential 
    τm::FT = NaN32 # Membrane time constant (ms), τm = C / gl
    R::FT = NaN32 # Resistance (GΩ), R = 1nS / gl
    ΔT::FT = 2mV # Slope factor
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
end



@doc raw"""
    AdEx(; N = 100, param = AdExParameter(), synapse = DoubleExpSynapse(), spike = PostSpike(),
           name = "AdEx", kwargs...)

Population of adaptive exponential integrate-and-fire neurons with an adaptive spike threshold,
combining `param::AdExParameter`, any synapse model and `spike::PostSpike`.

# Equations
```math
\begin{aligned}
\tau_m \frac{dv}{dt} &= -(v - E_l) + \Delta_T \exp\!\left(\frac{v - \theta}{\Delta_T}\right)
                        - R\, I_{syn} - R\, w + R\, I \\
\tau_w \frac{dw}{dt} &= a\,(v - E_l) - w \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```
A spike is emitted when ``v \geq 0`` mV; then ``v`` is set to 20 mV for that step,
``w \leftarrow w + b``, ``\theta \leftarrow \theta + A_t``, and the neuron is refractory.
``I_{syn}`` is the synaptic current of the synapse model (`syn_curr`), ``τ_A``, ``A_t`` and
`τabs` come from `spike` (with the default `At = 0mV` the threshold stays at `Vt`).

# Integration
Forward Euler, per neuron and per step (`update_neuron!`):
1. if the neuron fired in the previous step, `v = Vr` (reset);
2. `fire = false`; `tabs -= 1`; if `tabs > 0` the neuron is refractory and nothing else is
   updated (so ``v`` stays at ``V_r`` and ``w``, ``θ`` are frozen);
3. `w += dt * (a * (v - El) - w) / τw` (with the old ``v``);
4. `v += dt * (-(v - El) + ΔT * exp((v - θ) / ΔT) - R * syn_curr - R * w + R * I) / τm`
   (exponential term omitted if `ΔT < 0`);
5. `θ += dt * (Vt - θ) / τA`;
6. if `v >= 0mV`: `fire = true`, `v = 20mV`, `w += b`, `θ += At`,
   `tabs = round(Int, τabs / dt)`.
After a spike the neuron is therefore clamped for `round(Int, τabs / dt) - 1` steps.

# Fields
- `name::String = "AdEx"`, `id::String = randstring(12)`.
- `param::AdExParameter = AdExParameter()`, `synapse = DoubleExpSynapse()`,
  `spike::PostSpike = PostSpike()`.
- `N::Int32 = 100`: number of neurons.
- `v::VFT`: membrane potential (mV), uniform in `[Vr, Vt]`. `VFT = Vector{Float32}` by default; `VFT = Vector{Float64}` gives a
  `Float64` population (`param`, `synapse`, `spike`, the synaptic variables and receptor buffers follow; see `float_type`).
- `w::VFT`: adaptation current (pA), zeros.
- `fire::Vector{Bool}`: spike flags.
- `θ::VFT`: adaptive threshold of the exponential term (mV), initialised to `Vt`.
- `tabs::Vector{Int}`: refractory counters (steps), ones.
- `I::VFT`: external current (pA).
- `syn_curr::VFT`: synaptic current (pA).
- `synvars`: synaptic state; `receptors::NamedTuple`: spike input buffers (see `IF`).
- `records::Dict`.

# References
Brette R., Gerstner W. (2005). Adaptive exponential integrate-and-fire model as an effective
description of neuronal activity. J. Neurophysiol. 94:3637-3642.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.AdEx(N = 10, param = SNN.AdExParameter(), spike = SNN.PostSpike(τabs = 2ms))
E.I .= 1nA
SNN.monitor!(E, [:v, :w, :fire])
SNN.sim!([E]; duration = 500ms)
```
"""
AdEx

@snn_kw struct AdEx{
    IT = Int32,
    VFT = Vector{Float32},
    PST<:PostSpike,
    SYNT<:AbstractSynapseParameter,
    SYNV<:AbstractSynapseVariable,
    AdExt<:AdExParameter,
    RECT<:NamedTuple,
} <: AbstractGeneralizedIF

    name::String = "AdEx"
    id::String = randstring(12)

    param::AdExt = AdExParameter(FT = eltype(VFT))
    synapse::SYNT = DoubleExpSynapse(FT = eltype(VFT))
    spike::PST = PostSpike(FT = eltype(VFT))

    N::IT = 100 # Number of neurons
    v::VFT = param.Vr .+ rand(eltype(VFT), N) .* (param.Vt - param.Vr)
    w::VFT = zeros(eltype(VFT), N) # Adaptation current

    fire::VBT = zeros(Bool, N) # Store spikes
    θ::VFT = ones(eltype(VFT), N) .* param.Vt # Array with membrane potential thresholds
    tabs::VIT = ones(Int, N) # Membrane time constant
    I::VFT = zeros(eltype(VFT), N) # Current

    # Two receptors synaptic conductance
    syn_curr::VFT = zeros(eltype(VFT), N)
    synvars::SYNV = synaptic_variables(synapse, N, eltype(VFT)) # Synaptic variables for receptor model
    receptors::RECT = synaptic_receptors(synapse, N, eltype(VFT))

    records::Dict = Dict()
end


function synaptic_target(targets::Dict, post::T, sym::Symbol, target) where {T<:AdEx}
    syn = get_synapse_symbol(post.synapse, sym)
    sym = Symbol(syn)
    g = getfield(post.receptors, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end


function Population(
    param::AdExParameter;
    synapse::AbstractSynapseParameter,
    N,
    spike = PostSpike(),
    kwargs...,
)
    return AdEx(; N, param, synapse, spike, SYNT = typeof(synapse), kwargs...)
end


# Membrane update of `AdEx` (scalar parameters); see the `AdEx` docstring for the step order.
#
# Branchless and vectorised (`@inbounds @simd`): the update is evaluated for every neuron and discarded for the
# refractory ones (`tabs > 0` after the decrement) with `ifelse`, which is bit-identical to the branching loop of
# SNNModels <= 1.9 (every value that is kept goes through the same floating-point operations in the same order).
# The exponential term does not depend on the neuron through anything but `v` and `θ`; for a refractory neuron it may overflow to `Inf`,
# which is discarded. Hoisted out of the loop: `round(Int, τabs / dt)` and the test `ΔT < 0`.
function update_neuron!(p::P, param::T, dt::AbstractFloat) where {P<:AdEx,T<:AdExParameter}
    update_neuron!(p, param, dt, 1:Int(p.N))
end

function update_neuron!(
    p::P,
    param::T,
    dt::FT,
    rng::UnitRange{Int},
) where {P<:AdEx,FT<:AbstractFloat,T<:AdExParameter{FT}}
    @unpack N, v, w, fire, θ, I, tabs, syn_curr = p
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b = param
    @unpack At, τA, τabs = p.spike
    _check_state_lengths(N, v, w, fire, θ, I, tabs, syn_curr)
    nabs = round(Int, τabs / dt)    # refractory steps, loop invariant
    useexp = !(ΔT < zero(FT))       # the exponential term is removed by a negative slope factor

    @inbounds @simd for i ∈ rng
        vi = ifelse(fire[i], Vr, v[i])      # reset after the spike of the previous step
        ti = tabs[i] - 1
        act = ti <= 0                       # not refractory
        wi = w[i]
        thi = θ[i]

        w1 = wi + dt * (a * (vi - El) - wi) / τw
        ex = ifelse(useexp, ΔT * @inline(exp((vi - thi) / ΔT)), zero(FT))
        v1 = vi +
             dt * (
            -(vi - El)  # leakage
            + ex        # exponential term
            - R * syn_curr[i] # synapses
            - R * w1    # adaptation
            + R * I[i]  # external current
        ) / τm
        th1 = thi + dt * (Vt - thi) / τA
        spk = act & (v1 >= 0mV)

        fire[i] = spk
        v[i] = ifelse(act, ifelse(spk, FT(20), v1), vi)
        w[i] = ifelse(act, ifelse(spk, w1 + b, w1), wi)
        θ[i] = ifelse(act, ifelse(spk, th1 + At, th1), thi)
        tabs[i] = ifelse(spk, nabs, ti)
    end
end


# Heterogeneous version: every parameter of `AdExParameter{Vector{Float32}}` is per neuron.
function update_neuron!(
    p::P,
    param::T,
    dt::FT,
    rng::UnitRange{Int},
) where {P<:AdEx,FT<:AbstractFloat,T<:AdExParameter{Vector{FT}}}
    @unpack N, v, w, fire, θ, I, tabs, syn_curr = p
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b = param
    @unpack At, τA, τabs = p.spike
    _check_state_lengths(N, v, w, fire, θ, I, tabs, syn_curr, τm, Vt, Vr, El, R, ΔT, τw, a, b)
    nabs = round(Int, τabs / dt)

    @inbounds @simd for i ∈ rng
        vi = ifelse(fire[i], Vr[i], v[i])
        ti = tabs[i] - 1
        act = ti <= 0
        wi = w[i]
        thi = θ[i]
        Ti = ΔT[i]

        w1 = wi + dt * (a[i] * (vi - El[i]) - wi) / τw[i]
        ex = ifelse(Ti < zero(FT), zero(FT), Ti * @inline(exp((vi - thi) / Ti)))
        v1 = vi +
             dt * (
            -(vi - El[i])  # leakage
            + ex           # exponential term
            - R[i] * syn_curr[i] # synapses
            - R[i] * w1    # adaptation
            + R[i] * I[i]  # external current
        ) / τm[i]
        th1 = thi + dt * (Vt[i] - thi) / τA
        spk = act & (v1 >= 0mV)

        fire[i] = spk
        v[i] = ifelse(act, ifelse(spk, FT(20), v1), vi)
        w[i] = ifelse(act, ifelse(spk, w1 + b[i], w1), wi)
        θ[i] = ifelse(act, ifelse(spk, th1 + At, th1), thi)
        tabs[i] = ifelse(spk, nabs, ti)
    end
end


_chunk_population(::AdEx) = true

# one precision per population (see `IF`)
function snn_kw_finalize(::Type{T}, f::NamedTuple) where {T<:AdEx}
    _check_precision("AdEx", _state_float(T, :v), "param" => f.param, "synapse" => f.synapse, "spike" => f.spike)
    return f
end

export AdEx, AdExParameter
