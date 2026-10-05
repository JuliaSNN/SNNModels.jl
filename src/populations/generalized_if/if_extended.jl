@doc raw"""
    ExtendedIFParameter{FT = Float32}(; Cm = 250pF, Vt = -40mV, Vr = -65mV, El = -70mV,
                                       gl = 10nS, τe = 6ms, τi = 20ms, E_i = -75mV,
                                       E_e = 0mV, τabs = 5ms, α = 0)

Parameters of `ExtendedIF`, a conductance-based integrate-and-fire neuron with three synaptic
conductances (excitatory, PV-like and SST-like inhibition) and a multiplicative
excitation-SST interaction term.

# Fields
- `Cm::FT = 250pF`: membrane capacitance (pF).
- `Vt::FT = -40mV`: spike threshold (mV).
- `Vr::FT = -65mV`: reset potential (mV).
- `El::FT = -70mV`: leak reversal potential (mV).
- `gl::FT = 10nS`: leak conductance (nS); the code marks this value as arbitrary.
- `τe::FT = 6ms`: decay time constant of `g_Exc` (ms).
- `τi::FT = 20ms`: decay time constant of `g_PV` and `g_SST` (ms).
- `E_i::FT = -75mV`: inhibitory reversal potential (mV), used by both PV and SST conductances.
- `E_e::FT = 0mV`: excitatory reversal potential (mV).
- `τabs::FT = 5ms`: absolute refractory period (ms).
- `α::FT = 0.0`: strength of the interaction term ``-α\, g_E\, g_{SST} (E_e - v)``
  (1/nS); `0` disables it ("dendritic interaction term" in the code).

Reference not given in the code.
"""
ExtendedIFParameter

@snn_kw struct ExtendedIFParameter{FT = Float32} <: AbstractGeneralizedIFParameter
    Cm::FT = 250pF
    Vt::FT = -40mV
    Vr::FT = -65mV
    El::FT = -70mV
    gl::FT = 10.0nS # ! THIS PARAMETER IS ARBITRARY
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 20ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 5ms # Absolute refractory period
    α::FT = 0.0 # Dendritic interaction term
end

@doc raw"""
    ExtendedIF(; N = 100, param = ExtendedIFParameter(), name = "ExtendedIF", kwargs...)

Conductance-based integrate-and-fire population with excitatory (`g_Exc`), PV (`g_PV`) and SST
(`g_SST`) conductances and an optional multiplicative interaction between excitation and SST
inhibition. Unlike `IF` and `AdEx`, it has no `synapse` field: the conductances are fields of the
population and decay exponentially. Connections
target one of the conductances: `SpikingSynapse(pre, post, sym; conn)` with `sym` in `:g_Exc`,
`:g_PV`, `:g_SST`; `:ge`/`:glu` are mapped to `:g_Exc` and `:gi`/`:gaba` to `:g_PV`. A spike of
weight ``w`` (nS) increments the conductance by ``w``. (Up to SNNModels 1.8.4 there was no
`synaptic_target` method and this raised a `MethodError`.)

# Equations
```math
\begin{aligned}
C_m \frac{dv}{dt} &= g_l (E_l - v) + g_E (E_e - v) + g_{PV} (E_i - v) + g_{SST} (E_i - v)
                    - \alpha\, g_E\, g_{SST} (E_e - v) + I \\
\frac{dg_E}{dt} &= -\frac{g_E}{\tau_e}, \qquad
\frac{dg_{PV}}{dt} = -\frac{g_{PV}}{\tau_i}, \qquad
\frac{dg_{SST}}{dt} = -\frac{g_{SST}}{\tau_i}
\end{aligned}
```
Spike when ``v > V_t``, then ``v \leftarrow V_r`` and refractoriness for `τabs`.

# Integration
`integrate!` first decays the three conductances (forward Euler), then updates each neuron:
if `tabs > 0` the neuron is refractory (`fire = false`, `tabs -= 1`, ``v`` unchanged); otherwise
one forward-Euler step of ``v``, threshold test, reset to `Vr` and `tabs = round(Int, τabs / dt)`.

# Fields
- `id::String = randstring(12)`, `name::String = "ExtendedIF"`.
- `param::ExtendedIFParameter = ExtendedIFParameter()`.
- `N::Int32 = 100`.
- `v`: membrane potential (mV), uniform in `[Vr, Vt]`.
- `g_Exc`, `g_PV`, `g_SST`: conductances (nS), zeros.
- `tabs`: refractory counters (steps, stored as `Float32`), zeros.
- `w`: unused by the dynamics, zeros.
- `fire::Vector{Bool}`; `I`: external current (pA); `records::Dict`.
- `Δv`, `Δv_temp`: unused buffers.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.ExtendedIF(N = 10, param = SNN.ExtendedIFParameter(α = 0.01))
E.I .= 400pA
SNN.monitor!(E, [:v, :fire])
SNN.sim!([E]; duration = 100ms)
```
"""
ExtendedIF

@snn_kw mutable struct ExtendedIF{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    IFT<:AbstractGeneralizedIFParameter,
} <: AbstractGeneralizedIF
    id::String = randstring(12)
    name::String = "ExtendedIF"
    param::IFT = ExtendedIFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    g_Exc::VFT = zeros(N)
    g_PV::VFT = zeros(N)
    g_SST::VFT = zeros(N)
    tabs::VFT = zeros(N)
    w::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
    Δv::VFT = zeros(Float32, N)
    Δv_temp::VFT = zeros(Float32, N)
end

function integrate!(p::ExtendedIF, param::ExtendedIFParameter, dt::Float32)
    update_synapses!(p, param, dt)
    update_neuron!(p, param, dt)
end

function update_neuron!(p::ExtendedIF, param::ExtendedIFParameter, dt::Float32)
    @unpack N, v, g_Exc, g_PV, g_SST, w, I, tabs, fire = p
    @unpack El, E_i, E_e, τabs, gl, α = param
    @unpack N, v, w, tabs, fire = p
    @unpack Vt, Vr, τabs = param
    @inbounds for i = 1:N
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end
        # Membrane potential
        dv =
            (
                gl * (El - v[i]) +
                g_Exc[i] * (E_e - v[i]) +
                g_PV[i] * (E_i - v[i]) +
                g_SST[i] * (E_i - v[i]) +
                -α * g_Exc[i] * g_SST[i] * (E_e - v[i]) +
                + I[i] # synaptic term
                # 0
            ) / param.Cm
        v[i] += dt * dv
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        # Absolute refractory period
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
end

function update_synapses!(p::ExtendedIF, param::ExtendedIFParameter, dt::Float32)
    @unpack N, g_Exc, g_PV, g_SST = p
    @unpack τe, τi = param
    @inbounds for i = 1:N
        g_Exc[i] += dt * (-g_Exc[i] / τe)
        g_PV[i] += dt * (-g_PV[i] / τi)
        g_SST[i] += dt * (-g_SST[i] / τi)
    end
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target = nothing,
) where {T<:ExtendedIF}
    sym = sym in (:ge, :glu) ? :g_Exc : sym in (:gi, :gaba) ? :g_PV : sym
    sym in (:g_Exc, :g_PV, :g_SST) ||
        throw(ArgumentError("ExtendedIF connections target :g_Exc, :g_PV or :g_SST, got :$sym"))
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end

export ExtendedIF, ExtendedIFParameter, update_neuron!, update_synapses!
