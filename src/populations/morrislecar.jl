# Parameters from : https://github.com/nest/ode-toolbox/blob/master/tests/morris_lecar.json
@doc raw"""
    MorrisLecarParameter{FT = Float32}(; Cm = 6.69pF, El = -50mV, EK = -70mV, ECa = 100mV,
                                        gl = 0.5nS, gK = 2nS, gCa = 1.1nS, τe = 5ms, τi = 10ms,
                                        V1 = 30mV, V2 = 15mV, V3 = 0mV, V4 = 30mV, ϕ = 25Hz,
                                        Ee = 0mV, Ei = -75mV)

Parameters of the Morris-Lecar population `MorrisLecar`. The values are taken from the
NEST ode-toolbox test file `morris_lecar.json` (cited in the code).

# Fields
- `Cm::FT = 6.69pF`: membrane capacitance (pF).
- `El::FT = -50mV`, `EK::FT = -70mV`, `ECa::FT = 100mV`: leak, potassium and calcium reversal
  potentials (mV).
- `gl::FT = 0.5nS`, `gK::FT = 2nS`, `gCa::FT = 1.1nS`: leak, maximal potassium and maximal
  calcium conductances (nS).
- `τe::FT = 5ms`, `τi::FT = 10ms`: decay time constants of `ge`, `gi` (ms).
- `V1::FT = 30mV`, `V2::FT = 15mV`: midpoint and slope of the calcium activation ``m_\infty``.
- `V3::FT = 0mV`, `V4::FT = 30mV`: midpoint and slope of the potassium activation ``w_\infty``.
- `ϕ::FT = 25Hz`: rate scale of the potassium gating (0.025 per ms).
- `Ee::FT = 0mV`, `Ei::FT = -75mV`: synaptic reversal potentials (mV).

`MorrisLecarParameter <: AbstractPopulationParameter`, so `MorrisLecar` runs under `sim!` and
`train!`. Up to SNNModels 1.8.4 `train!` raised a `MethodError` (no `update_traces!` method).
"""
MorrisLecarParameter

@snn_kw struct MorrisLecarParameter{FT = Float32} <: AbstractPopulationParameter
    Cm::FT = 6.69pF
    El::FT = -50mV
    EK::FT = -70mV
    ECa::FT = 100mV
    gl::FT = 0.5nS
    gK::FT = 2nS
    gCa::FT = 1.1nS
    τe::FT = 5ms
    τi::FT = 10ms
    V1::FT = 30mV
    V2::FT = 15mV
    V3::FT = 0mV
    V4::FT = 30mV
    ϕ::FT = 25Hz
    Ee::FT = 0mV
    Ei::FT = -75mV
end


@snn_kw mutable struct MorrisLecar{VFT = Vector{Float32},VBT = Vector{Bool}} <:
                       AbstractPopulation
    name::String = "MorrisLecar"
    id::String = randstring(12)
    param::MorrisLecarParameter = MorrisLecarParameter()
    N::Int32 = 100
    v::VFT = -52.14 .+ zeros(N)
    w::VFT = 0.2 .+ zeros(N)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

@doc raw"""
    MorrisLecar(; N = 100, param = MorrisLecarParameter(), name = "MorrisLecar", kwargs...)

Population of Morris-Lecar neurons (instantaneous calcium activation, slow potassium
activation ``w``) with conductance-based exponential synapses `ge`, `gi`.
Connections target `ge` or `gi`: `SpikingSynapse(pre, post, :ge; conn)` (`:glu` is mapped to
`:ge`, `:gaba` to `:gi`); a presynaptic spike of weight ``w`` (nS) increments the conductance by
``w``. (Up to SNNModels 1.8.4 there was no `synaptic_target` method and this raised a
`MethodError`.)

# Equations
```math
\begin{aligned}
C_m \frac{dv}{dt} &= I + g_l (E_l - v) + g_{Ca}\, m_\infty(v) (E_{Ca} - v) + g_K\, w\, (E_K - v)
                    + g_e (E_e - v) + g_i (E_i - v) \\
\frac{dw}{dt} &= \frac{w_\infty(v) - w}{\tau_w(v)} \\
m_\infty(v) &= \tfrac12 \left(1 + \tanh\frac{v - V_1}{V_2}\right), \quad
w_\infty(v) = \tfrac12 \left(1 + \tanh\frac{v - V_3}{V_4}\right), \quad
\tau_w(v) = \frac{1}{\phi \cosh\left(\frac{v - V_3}{2 V_4}\right)} \\
\frac{dg_e}{dt} &= -\frac{g_e}{\tau_e}, \qquad \frac{dg_i}{dt} = -\frac{g_i}{\tau_i}
\end{aligned}
```

# Integration
Forward Euler, sequential per neuron: ``v`` is advanced with the intrinsic and external
currents (old ``w``), then ``w`` with the new ``v``, then the synaptic term
`dt / Cm * (ge * (Ee - v) + gi * (Ei - v))` is added, then `ge`, `gi` decay.
A spike is flagged in the step in which ``v`` crosses 20 mV upwards (one flag per action
potential); there is no reset. (Up to SNNModels 1.8.4 `fire` was the level test `v > 20mV`.)

# Fields
- `name::String = "MorrisLecar"`, `id::String = randstring(12)`,
  `param::MorrisLecarParameter = MorrisLecarParameter()`, `N::Int32 = 100`.
- `v = -52.14 .+ zeros(N)` (mV), `w = 0.2 .+ zeros(N)`.
- `ge`, `gi`: synaptic conductances (nS), zeros.
- `fire::Vector{Bool}`; `I`: external current (pA); `records::Dict`.

# References
Morris C., Lecar H. (1981). Voltage oscillations in the barnacle giant muscle fiber.
Biophys. J. 35:193-213 (linked in the code). Parameters from the NEST ode-toolbox test
`morris_lecar.json`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.MorrisLecar(N = 5)
E.I .= 100pA
SNN.monitor!(E, [:v, :w])
SNN.sim!([E]; duration = 200ms, dt = 0.05ms)
```
"""
MorrisLecar

function integrate!(p::MorrisLecar, param::MorrisLecarParameter, dt::Float32)
    @unpack N, v, w, ge, gi, fire, I = p
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    @unpack Ee, Ei = param
    @inbounds for i = 1:N
        v_old = v[i]
        m_ss = 0.5*(1+tanh((v[i]-V1)/V2))
        n_ss = 0.5*(1+tanh((v[i]-V3)/V4))
        τ = 1 / (ϕ * cosh((v[i]-V3)/(2V4)))

        # v[i] += dt / Cm * (
        #         I[i] +
        #         gl  * (El  - v[i]) +
        #         gCa * (ECa - v[i]) * m_ss +
        #         gK  * (EK  - v[i]) * w[i] +
        #         0
        #     )
        # w[i]  += dt * (n_ss - w[i])/τ

        v[i] += dt / Cm * MorrisLecar_dv(v[i], w[i], I[i], param)
        w[i] += dt * MorrisLecar_dw(v[i], w[i], param)

        v[i] += dt/Cm * (ge[i] * (Ee - v[i]) + gi[i] * (Ei - v[i]))
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        # spike = upward crossing of 20 mV (one flag per action potential)
        fire[i] = (v_old <= 20.0f0) & (v[i] > 20.0f0)
    end
end


# Intrinsic + external current of the Morris-Lecar membrane equation (pA), i.e. `Cm * dv/dt`
# without the synaptic term.
function MorrisLecar_dv(v::Float32, w::Float32, I::Float32, param::MorrisLecarParameter)
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    m_ss = 0.5*(1+tanh((v-V1)/V2))
    return I + gl * (El - v) + gCa * (ECa - v) * m_ss + gK * (EK - v) * w
end


# Right-hand side of the potassium gating equation, `(w_inf(v) - w) / τw(v)` (1/ms).
function MorrisLecar_dw(v::Float32, w::Float32, param::MorrisLecarParameter)
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    n_ss = 0.5*(1+tanh((v-V3)/V4))
    τ = 1 / (ϕ * cosh((v-V3)/(2V4)))
    return (n_ss - w)/τ
end


# w-nullcline helper: the w-nullcline is `w = w_inf(v)` (returned `-w_inf(v)` up to 1.8.4).
function MorrisLecar_w_nullcline(v::Float32, param::MorrisLecarParameter)
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    n_ss = 0.5*(1+tanh((v-V3)/V4))
    return n_ss
end


# v-nullcline helper: value of `w` for which `dv/dt = 0` (synaptic input excluded).
function MorrisLecar_v_nullcline(v::Float32, I::Float32, param::MorrisLecarParameter)
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    m_ss = 0.5*(1+tanh((v-V1)/V2))
    return -(I + gl * (El - v) + gCa * (ECa - v) * m_ss)/(gK * (EK - v))
end

function plasticity!(p::MorrisLecar, param::MorrisLecarParameter, dt::Float32, T::Time) end

function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target = nothing,
) where {T<:MorrisLecar}
    sym = sym == :glu ? :ge : sym == :gaba ? :gi : sym
    sym in (:ge, :gi) || throw(ArgumentError("MorrisLecar connections target :ge or :gi, got :$sym"))
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end

export MorrisLecar

# function HH_spike_count(p::HH, dt = 0.01)
#     neurons = hcat(p.records[:fire]...)
#     spike_count = zeros(size(neurons, 1))
#     for (n, fires) in enumerate(eachrow(neurons))
#         r = length(findall(x -> fires[x] > 0 && fires[x+1] == 0, eachindex(fires[1:end-1])))
#         spike_count[n] = r / 1000
#     end
#     return spike_count
# end
