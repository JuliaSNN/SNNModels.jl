@doc raw"""
    HHParameter{FT = Float32}(; Cm, gl, El = -65mV, Ek = -90mV, En = 50mV, gn, gk, Vt = -63mV,
                               τe = 5ms, τi = 10ms, Ee = 0mV, Ei = -80mV)

Parameters of the Hodgkin-Huxley population `HH`. Capacitance and maximal conductances are
specified as densities times a membrane area of 20000 μm²:

- `Cm::FT = 1uF * cm^(-2) * 20000um^2`: membrane capacitance, 200 pF.
- `gl::FT = 5e-5siemens * cm^(-2) * 20000um^2`: leak conductance, 10 nS.
- `El::FT = -65mV`: leak reversal potential (mV).
- `Ek::FT = -90mV`: potassium reversal potential (mV).
- `En::FT = 50mV`: sodium reversal potential (mV).
- `gn::FT = 100msiemens * cm^(-2) * 20000um^2`: maximal sodium conductance, 20000 nS.
- `gk::FT = 30msiemens * cm^(-2) * 20000um^2`: maximal potassium conductance, 6000 nS.
- `Vt::FT = -63mV`: offset of the gating kinetics (mV); it is not a spike threshold.
- `τe::FT = 5ms`, `τi::FT = 10ms`: decay time constants of `ge`, `gi` (ms).
- `Ee::FT = 0mV`, `Ei::FT = -80mV`: synaptic reversal potentials (mV).

`HHParameter` is not a subtype of `AbstractPopulationParameter`, so `train!` raises a
`MethodError` for `HH` populations in SNNModels 1.8.4 (no `update_traces!` fallback);
use `sim!`.
"""
HHParameter

@snn_kw struct HHParameter{FT = Float32}
    Cm::FT = 1uF * cm^(-2) * 20000um^2
    gl::FT = 5e-5siemens * cm^(-2) * 20000um^2
    El::FT = -65mV
    Ek::FT = -90mV
    En::FT = 50mV
    gn::FT = 100msiemens * cm^(-2) * 20000um^2
    gk::FT = 30msiemens * cm^(-2) * 20000um^2
    Vt::FT = -63mV
    τe::FT = 5ms
    τi::FT = 10ms
    Ee::FT = 0mV
    Ei::FT = -80mV
end

@snn_kw mutable struct HH{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractPopulation
    name::String = "HH"
    id::String = randstring(12)
    param::HHParameter = HHParameter()
    N::Int32 = 100
    v::VFT = param.El .+ 5(randn(N) .- 1)
    m::VFT = zeros(N)
    n::VFT = zeros(N)
    h::VFT = ones(N)
    ge::VFT = (1.5randn(N) .+ 4) .* 10nS
    gi::VFT = (12randn(N) .+ 20) .* 10nS
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym = nothing,
    target = nothing,
) where {T<:HH}
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end

@doc raw"""
    HH(; N = 100, param = HHParameter(), name = "HH", kwargs...)

Population of single-compartment Hodgkin-Huxley neurons (sodium, delayed-rectifier potassium
and leak currents) with conductance-based exponential synapses `ge`, `gi`.
Connections target `:ge` or `:gi`.

# Equations
```math
\begin{aligned}
C_m \frac{dv}{dt} &= I + g_l (E_l - v) + g_e (E_e - v) + g_i (E_i - v)
                    + g_n m^3 h (E_n - v) + g_k n^4 (E_k - v) \\
\frac{dx}{dt} &= \alpha_x(v)\,(1 - x) - \beta_x(v)\, x, \qquad x \in \{m, n, h\} \\
\frac{dg_e}{dt} &= -\frac{g_e}{\tau_e}, \qquad \frac{dg_i}{dt} = -\frac{g_i}{\tau_i}
\end{aligned}
```
with ``u = v - V_t`` (mV) and rates in 1/ms:
```math
\begin{aligned}
\alpha_m &= \frac{0.32\,(13 - u)}{e^{(13 - u)/4} - 1}, &
\beta_m &= \frac{0.28\,(u - 40)}{e^{(u - 40)/5} - 1}, \\
\alpha_n &= \frac{0.032\,(15 - u)}{e^{(15 - u)/5} - 1}, &
\beta_n &= 0.5\, e^{(10 - u)/40}, \\
\alpha_h &= 0.128\, e^{(17 - u)/18}, &
\beta_h &= \frac{4}{1 + e^{(40 - u)/5}}.
\end{aligned}
```

# Integration
Forward Euler, sequential: `m`, `n`, `h` are updated first, then ``v`` with the new gating
variables, then `ge`, `gi` decay. `fire[i] = v[i] > -20mV` is evaluated after the update; it is a
level test, so `fire` stays `true` for every step the membrane is above -20 mV (one action
potential usually produces several consecutive `true` steps). There is no reset and no
refractory period. A small `dt` (e.g. 0.01-0.05 ms) is needed for stability.

# Fields
- `name::String = "HH"`, `id::String = randstring(12)`, `param::HHParameter = HHParameter()`.
- `N::Int32 = 100`.
- `v = El .+ 5(randn(N) .- 1)`: membrane potential (mV).
- `m = zeros(N)`, `n = zeros(N)`, `h = ones(N)`: gating variables.
- `ge = (1.5randn(N) .+ 4) .* 10nS`, `gi = (12randn(N) .+ 20) .* 10nS`: synaptic conductances
  (nS), random initial values.
- `fire::Vector{Bool}`; `I`: external current (pA); `records::Dict`.

# References
The code links only the Wikipedia page of the Hodgkin-Huxley model. Canonical model:
Hodgkin A. L., Huxley A. F. (1952). A quantitative description of membrane current and its
application to conduction and excitation in nerve. J. Physiol. 117:500-544.
The rate functions and conductance densities above have the Traub-Miles form used in the
COBAHH benchmark of Brette et al. (2007, J. Comput. Neurosci.); the code does not
cite this source.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.HH(N = 5)
E.ge .= 0; E.gi .= 0
E.I .= 500pA
SNN.monitor!(E, [:v])
SNN.sim!([E]; duration = 50ms, dt = 0.01ms)
```
"""
HH

function integrate!(p::HH, param::HHParameter, dt::Float32)
    @unpack N, v, m, n, h, ge, gi, fire, I = p
    @unpack Cm, gl, El, Ek, En, gn, gk, Vt, τe, τi, Ee, Ei = param
    @inbounds for i = 1:N
        fire[i] = false
        m[i] +=
            dt * (
                0.32f0 * (13.0f0 - v[i] + Vt) /
                (exp((13.0f0 - v[i] + Vt) / 4.0f0) - 1.0f0) * (1.0f0 - m[i]) -
                0.28f0 * (v[i] - Vt - 40.0f0) /
                (exp((v[i] - Vt - 40.0f0) / 5.0f0) - 1.0f0) * m[i]
            )
        n[i] +=
            dt * (
                0.032f0 * (15.0f0 - v[i] + Vt) /
                (exp((15.0f0 - v[i] + Vt) / 5.0f0) - 1.0f0) * (1.0f0 - n[i]) -
                0.5f0 * exp((10.0f0 - v[i] + Vt) / 40.0f0) * n[i]
            )
        h[i] +=
            dt * (
                0.128f0 * exp((17.0f0 - v[i] + Vt) / 18.0f0) * (1.0f0 - h[i]) -
                4.0f0 / (1.0f0 + exp((40.0f0 - v[i] + Vt) / 5.0f0)) * h[i]
            )
        v[i] +=
            dt / Cm * (
                I[i] +
                gl * (El - v[i]) +
                ge[i] * (Ee - v[i]) +
                gi[i] * (Ei - v[i]) +
                gn * m[i]^3 * h[i] * (En - v[i]) +
                gk * n[i]^4 * (Ek - v[i])
            )
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > -20.0f0
    end
end

export HH, HHParameter

# function HH_spike_count(p::HH, dt = 0.01)
#     neurons = hcat(p.records[:fire]...)
#     spike_count = zeros(size(neurons, 1))
#     for (n, fires) in enumerate(eachrow(neurons))
#         r = length(findall(x -> fires[x] > 0 && fires[x+1] == 0, eachindex(fires[1:end-1])))
#         spike_count[n] = r / 1000
#     end
#     return spike_count
# end
