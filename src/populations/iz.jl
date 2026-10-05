@doc raw"""
    IZParameter{FT = Float32}(; a = 0.01, b = 0.2, c = -65, d = 2, τe = 5ms, τi = 10ms,
                               Ee = 0mV, Ei = -80mV)

Parameters of the Izhikevich neuron model (`IZ`) with two conductance-based, exponentially
decaying synapses.

`IZParameter` is not a subtype of `AbstractPopulationParameter`, so the `train!` fallbacks
(`update_traces!`, `plasticity!`) are not defined for `IZ`: an `IZ` population can be simulated
with `sim!` but `train!` raises a `MethodError` in SNNModels 1.8.4.

# Fields
- `a::FT = 0.01`: time scale of the recovery variable ``u`` (1/ms).
- `b::FT = 0.2`: sensitivity of ``u`` to ``v``.
- `c::FT = -65`: after-spike reset value of ``v`` (mV).
- `d::FT = 2`: after-spike increment of ``u``.
- `τe::FT = 5ms`: decay time constant of the excitatory conductance `ge` (ms).
- `τi::FT = 10ms`: decay time constant of the inhibitory conductance `gi` (ms).
- `Ee::FT = 0mV`: excitatory reversal potential (mV).
- `Ei::FT = -80mV`: inhibitory reversal potential (mV).

The defaults `a = 0.01, b = 0.2, c = -65, d = 2` are not one of the named firing classes of
Izhikevich (2003) (regular spiking is `a = 0.02, b = 0.2, c = -65, d = 8`).

# References
Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural
Networks, 14(6), 1569-1572.
"""
IZParameter
@snn_kw struct IZParameter{FT = Float32}
    a::FT = 0.01
    b::FT = 0.2
    c::FT = -65
    d::FT = 2
    τe::FT = 5ms
    τi::FT = 10ms
    Ee::FT = 0mV
    Ei::FT = -80mV
end

@doc raw"""
    IZ(; N = 100, param = IZParameter(), name = "IZ", kwargs...)

Population of Izhikevich neurons with conductance-based excitatory (`ge`) and inhibitory
(`gi`) synapses. Connections target `:ge` or `:gi` directly.

# Equations
```math
\begin{aligned}
\frac{dv}{dt} &= 0.04 v^2 + 5 v + 140 - u + I + g_e (E_e - v) + g_i (E_i - v) \\
\frac{du}{dt} &= a\,(b v - u) \\
\frac{dg_e}{dt} &= -\frac{g_e}{\tau_e}, \qquad \frac{dg_i}{dt} = -\frac{g_i}{\tau_i}
\end{aligned}
```
with ``v`` in mV and ``t`` in ms. When ``v > 30`` mV: ``v \leftarrow c``, ``u \leftarrow u + d``.
There is no capacitance: ``I`` and ``g (E - v)`` enter directly in mV/ms.

# Integration
Per step: the conductances decay (forward Euler); ``v`` is advanced with two forward-Euler
half steps of `dt/2` of the quadratic part (as in Izhikevich 2003), ``u`` with one Euler step
using the new ``v``, then the synaptic term `dt * (ge * (Ee - v) + gi * (Ei - v))` is added;
finally spikes are detected (`v > 30`) and reset. No refractory period.

# Fields
- `id::String = randstring(12)`, `name::String = "IZ"`, `param::IZParameter = IZParameter()`.
- `N::Int32 = 100`.
- `v::Vector{Float32} = fill(-65, N)`: membrane potential (mV).
- `u::Vector{Float32} = param.b * v`: recovery variable.
- `fire::Vector{Bool}`; `I::Vector{Float32}`: external input (mV/ms), zeros.
- `ge::Vector{Float32} = (1.5randn(N) .+ 4) .* 10nS`: excitatory conductance, random initial
  values (mean 40 nS).
- `gi::Vector{Float32} = (12randn(N) .+ 20) .* 10nS`: inhibitory conductance, random initial
  values (mean 200 nS, can be negative).
- `records::Dict`.

# References
Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural
Networks, 14(6), 1569-1572.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IZ(N = 10, param = SNN.IZParameter(a = 0.02, b = 0.2, c = -65, d = 8))
E.I .= 10
SNN.monitor!(E, [:v, :fire])
SNN.sim!([E]; duration = 200ms, dt = 0.5ms)
```
"""
IZ

@snn_kw mutable struct IZ{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractPopulation
    id::String = randstring(12)
    name::String = "IZ"
    param::IZParameter = IZParameter()
    N::Int32 = 100
    v::VFT = fill(-65.0, N)
    u::VFT = param.b * v
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
    ge::VFT = (1.5randn(N) .+ 4) .* 10nS
    gi::VFT = (12randn(N) .+ 20) .* 10nS
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym = nothing,
    target = nothing,
) where {T<:IZ}
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end

"""
    integrate!(p::IZ, param::IZParameter, dt::Float32)

One step of the Izhikevich population: exponential decay of `ge`, `gi` (forward Euler), two
half steps of the quadratic membrane equation, one step of `u`, the synaptic term, then spike
detection (`v > 30`) with reset `v = c`, `u += d`. See `IZ`.
"""
function integrate!(p::IZ, param::IZParameter, dt::Float32)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d = param
    @unpack ge, gi = p
    @inbounds for i = 1:N
        ge[i] += dt * -ge[i] / param.τe
        gi[i] += dt * -gi[i] / param.τi
    end
    @inbounds for i = 1:N
        v[i] += 0.5f0 * dt * (0.04f0 * v[i]^2 + 5.0f0 * v[i] + 140.0f0 - u[i] + I[i])
        v[i] += 0.5f0 * dt * (0.04f0 * v[i]^2 + 5.0f0 * v[i] + 140.0f0 - u[i] + I[i])
        u[i] += dt * (a * (b * v[i] - u[i]))
        v[i] += dt * (ge[i] * (param.Ee - v[i]) + gi[i] * (param.Ei - v[i]))
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > 30.0f0
        v[i] = ifelse(fire[i], c, v[i])
        u[i] += ifelse(fire[i], d, 0.0f0)
    end
end


export IZ, IZParameter
