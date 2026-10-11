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

The four area-scaled defaults are computed with the `Float32` unit constants (`um` is not exact in
`Float32`), so in `Float32` they are 199.99998 pF, 9.999999 nS, 19999.998 nS and 5999.9995 nS (unchanged
since 1.8); with `FT = Float64` they are the decimal values 200, 10, 20000 and 6000.

`HHParameter <: AbstractPopulationParameter`, so `HH` runs under `sim!` and `train!` (no-op
`update_traces!`/`plasticity!` fallbacks). Up to SNNModels 1.8.4 `train!` raised a `MethodError`.
"""
HHParameter

@snn_kw struct HHParameter{FT = Float32} <: AbstractPopulationParameter
    # area-scaled values: in Float32 as computed with the Float32 unit constants (199.99998, 9.999999, ...), as before
    # 1.10; in Float64 the decimal values (200 pF, 10 nS, 20000 nS, 6000 nS), see `_unit_default`
    Cm::FT = _unit_default(FT, 1uF * cm^(-2) * 20000um^2, 200.0)
    gl::FT = _unit_default(FT, 5e-5siemens * cm^(-2) * 20000um^2, 10.0)
    El::FT = -65mV
    Ek::FT = -90mV
    En::FT = 50mV
    gn::FT = _unit_default(FT, 100msiemens * cm^(-2) * 20000um^2, 20000.0)
    gk::FT = _unit_default(FT, 30msiemens * cm^(-2) * 20000um^2, 6000.0)
    Vt::FT = -63mV
    τe::FT = 5ms
    τi::FT = 10ms
    Ee::FT = 0mV
    Ei::FT = -80mV
end

@snn_kw mutable struct HH{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractPopulation
    name::String = "HH"
    id::String = randstring(12)
    param::HHParameter = HHParameter(FT = eltype(VFT))
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

# `x^4` of a `Float32`, written out: Base evaluates `Float32 ^ 4` in `Float64` by `power_by_squaring` (an out-of-line call that blocks vectorisation);
# this is the same arithmetic, `x2 = x * x; x2 * x2` in `Float64`, rounded to `Float32`, hence the same bits.
@inline function _pow4(x::Float32)
    y = Float64(x)
    y2 = y * y
    return Float32(y2 * y2)
end
@inline _pow4(x::Float64) = (x2 = x * x; x2 * x2)

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
variables, then `ge`, `gi` decay. A spike is flagged (`fire[i] = true`) in the step in which
``v`` crosses -20 mV upwards (`v_old <= -20mV < v`), so each action potential gives exactly one
flag. There is no reset and no refractory period. (Up to SNNModels 1.8.4 `fire` was the level
test `v > -20mV`, true for every step above -20 mV, i.e. several flags per action potential.) A small `dt` (e.g. 0.01-0.05 ms) is needed for stability.

# Fields
- `name::String = "HH"`, `id::String = randstring(12)`, `param::HHParameter = HHParameter()`.
- `N::Int32 = 100`.
- `VFT = Vector{Float32}`: type of the state vectors; `VFT = Vector{Float64}` gives a `Float64` population (the parameters
  follow; see `float_type`).
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

function integrate!(p::HH, param::HHParameter, dt_::AbstractFloat)
    dt = _step_dt(p, dt_)   # a step of another precision is converted to that of the population (as `sim!` does)
    @unpack N, v, m, n, h, ge, gi, fire, I = p
    @unpack Cm, gl, El, Ek, En, gn, gk, Vt, τe, τi, Ee, Ei = param
    _check_state_lengths(N, v, m, n, h, ge, gi, fire, I)
    T = eltype(v)    # literals in the precision of the state (`T(0.32) === 0.32f0` in Float32)
    # `@inline exp`: the exponentials are inlined so that the loop vectorises (same bits as the `exp` call)
    @inbounds @simd for i = 1:N
        v_old = v[i]
        m[i] +=
            dt * (
                T(0.32) * (T(13.0) - v[i] + Vt) /
                (@inline(exp((T(13.0) - v[i] + Vt) / T(4.0))) - T(1.0)) * (T(1.0) - m[i]) -
                T(0.28) * (v[i] - Vt - T(40.0)) /
                (@inline(exp((v[i] - Vt - T(40.0)) / T(5.0))) - T(1.0)) * m[i]
            )
        n[i] +=
            dt * (
                T(0.032) * (T(15.0) - v[i] + Vt) /
                (@inline(exp((T(15.0) - v[i] + Vt) / T(5.0))) - T(1.0)) * (T(1.0) - n[i]) -
                T(0.5) * @inline(exp((T(10.0) - v[i] + Vt) / T(40.0))) * n[i]
            )
        h[i] +=
            dt * (
                T(0.128) * @inline(exp((T(17.0) - v[i] + Vt) / T(18.0))) * (T(1.0) - h[i]) -
                T(4.0) / (T(1.0) + @inline(exp((T(40.0) - v[i] + Vt) / T(5.0)))) * h[i]
            )
        v[i] +=
            dt / Cm * (
                I[i] +
                gl * (El - v[i]) +
                ge[i] * (Ee - v[i]) +
                gi[i] * (Ei - v[i]) +
                gn * m[i]^3 * h[i] * (En - v[i]) +
                gk * _pow4(n[i]) * (Ek - v[i])
            )
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        # spike = upward crossing of -20 mV (one flag per action potential)
        fire[i] = (v_old <= -T(20.0)) & (v[i] > -T(20.0))
    end
end

# one precision per population (see `IF`)
function snn_kw_finalize(::Type{T}, f::NamedTuple) where {T<:HH}
    _check_precision("HH", _state_float(T, :v), "param" => f.param)
    return f
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
