@doc raw"""
    IFParameter{FT = Float32}(; C, gl, τm, R, Vt = -50mV, Vr = -60mV, El = -70mV,
                               ΔT = 2mV, a = 0, b = 0, τw = 0)

Neuron parameters of the leaky integrate-and-fire model `IF`, with optional subthreshold and
spike-triggered adaptation (adaptive IF when `τw > 0`).

# Membrane parameters
`C`, `gl`, `R` and `τm` satisfy `τm = C / gl` and `R = 1nS / gl`; the model integrates with
`τm` and `R`. Give them as a pair of independent values, any of (C, gl), (C, R), (C, τm),
(gl, τm), (R, τm), and the other two are derived; give none and the default pair `τm = 15ms`,
`R = 0.06` (GΩ, i.e. 60 MΩ; then `gl ≈ 16.7nS`, `C = 250pF`) is used. A single value is an error:
a given value is never combined with a default. More values are accepted if consistent. Changing
one of them later (`@update!`, [`with_membrane`](@ref)) keeps the others consistent: `τm` keeps
`gl` (changes `C`), `C` keeps `gl` (changes `τm`), `gl` or `R` keeps `C` (changes `τm`). See
[`resolve_membrane`](@ref) and [`membrane_update`](@ref).
Up to SNNModels 1.8.x a single value was combined with the defaults (e.g. `C` alone was ignored
and `τm = 15ms`); since 1.9.0 it is an error.

# Equations
```math
\begin{aligned}
\tau_m \frac{dv}{dt} &= -(v - E_l) + R\,(I - w) - R\, I_{syn} \\
\tau_w \frac{dw}{dt} &= a\,(v - E_l) - w
\end{aligned}
```
with ``I_{syn}`` the synaptic current computed by the synapse model of the population
(`syn_curr`, e.g. ``I_{syn} = g_E (v - E_E) + g_I (v - E_I)`` for `DoubleExpSynapse`) and ``I``
the external current (field `I` of the population). When ``v > V_t``: ``v \leftarrow V_r``,
``w \leftarrow w + b`` and the neuron is refractory for `spike.τabs`.
The adaptation variable is integrated only if `τw > 0`.

# Fields
- `C::FT`: membrane capacitance (pF); default from the pair, 250 pF.
- `gl::FT`: leak conductance (nS); default from the pair, 16.7 nS.
- `τm::FT`: membrane time constant (ms); default 15 ms.
- `Vt::FT = -50mV`: spike threshold (mV).
- `Vr::FT = -60mV`: reset potential (mV).
- `El::FT = -70mV`: leak reversal potential (mV).
- `R::FT`: membrane resistance (GΩ); default 0.06.
- `ΔT::FT = 2mV`: slope factor (mV); not used by `IF`.
- `a::FT = 0.0`: subthreshold adaptation conductance (nS).
- `b::FT = 0.0`: spike-triggered adaptation increment (pA).
- `τw::FT = 0.0`: adaptation time constant (ms); `0` disables adaptation.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
param = SNN.IFParameter(C = 281pF, gl = 40nS, a = 4nS, b = 80.5pA, τw = 144ms)
E = SNN.IF(N = 10, param = param)
```
"""
IFParameter

@snn_kw struct IFParameter{FT = Float32} <: AbstractGeneralizedIFParameter
    C::FT = NaN32        #(pF)
    gl::FT = NaN32         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = NaN32 # Membrane time constant (ms), τm = C / gl
    Vt::FT = -50mV # Membrane threshold potential
    Vr::FT = -60mV # Membrane reset potential
    El::FT = -70mV    # Membrane leak potential
    R::FT = NaN32 # Resistance (GΩ), R = 1nS / gl
    ΔT::FT = 2mV # Slope factor
    a::FT = 0.0 # Subthreshold adaptation parameter
    b::FT = 0.0 #80.5pA # 'sra' current increment
    τw::FT = 0.0 #144ms # adaptation time constant (~Ca-activated K current inactivation)
end

@doc raw"""
    IF(; N = 100, param = IFParameter(), synapse = DoubleExpSynapse(), spike = PostSpike(),
         name = "IF", kwargs...)

Population of leaky integrate-and-fire neurons (optionally adaptive), combining the neuron
model `param::IFParameter`, any synapse model `synapse::AbstractSynapseParameter` and the spike
parameters `spike::PostSpike`.

# Equations
```math
\begin{aligned}
\tau_m \frac{dv}{dt} &= -(v - E_l) + R\,(I - w) - R\, I_{syn} \\
\tau_w \frac{dw}{dt} &= a\,(v - E_l) - w, \qquad w \leftarrow w + b \text{ at each spike}
\end{aligned}
```
See `IFParameter` for the parameters and the synapse model for ``I_{syn}``.

# Integration
One call of `integrate!` (see `AbstractGeneralizedIF`) updates the synapses, computes
`syn_curr`, then `update_neuron!` performs, for each neuron:
- if the refractory counter `tabs > 0`: `fire = false`, `tabs -= 1`, ``v`` is not integrated
  (it stays at ``V_r``);
- otherwise one forward-Euler step
  `v += dt / τm * (-(v - El) + R * (-w + I) - R * syn_curr)`;
  if `v > Vt` the neuron fires: `v = Vr`, `tabs = round(Int, spike.τabs / dt)`.
Then, only if `τw > 0`, for every neuron `w += b` if it fired and
`w += dt * (a * (v - El) - w) / τw` (forward Euler, using the updated ``v``).
`tabs` is initialised to 1, so the first step is skipped by every neuron.

# Fields
- `param::IFParameter = IFParameter()`: neuron parameters.
- `synapse::SYNT = DoubleExpSynapse()`: synapse model.
- `spike::PST = PostSpike()`: spike parameters (only `τabs` is used).
- `id::String = randstring(12)`, `name::String = "IF"`.
- `N::Int32 = 100`: number of neurons.
- `v::VFT`: membrane potential (mV), initialised uniformly in `[Vr, Vt]`. `VFT = Vector{Float32}` by default;
  `IF(N = 10, VFT = Vector{Float64})` is a `Float64` population: `param`, `synapse` (`DoubleExpSynapse`), `spike`, the
  synaptic variables and the receptor buffers default to `Float64` too, and the connections and stimuli built on it
  follow (see `float_type`). Mixing precisions in one population is an `ArgumentError`.
- `w::VFT`: adaptation current (pA), zeros.
- `fire::Vector{Bool}`: spike flags of the last step.
- `tabs::Vector{Int}`: refractory counters (steps), ones.
- `I::VFT`: external current (pA), zeros.
- `syn_curr::VFT`: total synaptic current (pA).
- `synvars`: synaptic state, `synaptic_variables(synapse, N)`.
- `receptors::NamedTuple`: spike input buffers, `synaptic_receptors(synapse, N)`
  (`(glu, gaba)` for the two-receptor synapse models). Connections target them with the
  symbols `:ge`/`:glu`/`:he` (excitatory) and `:gi`/`:gaba`/`:hi` (inhibitory).
- `records::Dict`: recordings.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100, param = SNN.IFParameter(C = 281pF, gl = 40nS))
E.I .= 1nA                       # constant external current
SNN.monitor!(E, [:v, :fire])
SNN.sim!([E]; duration = 200ms)
```
"""
IF

@snn_kw struct IF{
    IT = Int32,
    VFT = Vector{Float32},
    PST = PostSpike{eltype(VFT)},
    SYNT<:AbstractSynapseParameter,
    SYNV<:AbstractSynapseVariable,
    RECT<:NamedTuple,
} <: AbstractGeneralizedIF

    param::IFParameter = IFParameter(FT = eltype(VFT))
    synapse::SYNT = DoubleExpSynapse(FT = eltype(VFT))
    spike::PST = PostSpike(FT = eltype(VFT))

    id::String = randstring(12)
    name::String = "IF"

    N::IT = 100 # Number of neurons
    v::VFT = param.Vr .+ rand(eltype(VFT), N) .* (param.Vt - param.Vr)
    w::VFT = zeros(eltype(VFT), N) # Adaptation current
    fire::VBT = zeros(Bool, N) # Store spikes
    tabs::VIT = ones(Int, N) # Membrane time constant
    I::VFT = zeros(eltype(VFT), N) # Current

    # Two receptors synaptic conductance
    syn_curr::VFT = zeros(eltype(VFT), N)
    synvars::SYNV = synaptic_variables(synapse, N) # Synaptic variables for receptor model
    receptors::RECT = synaptic_receptors(synapse, N)
    # Synaptic targets


    records::Dict = Dict()
end

function Population(
    param::IFParameter;
    synapse::AbstractSynapseParameter,
    spike::PostSpike,
    N,
    kwargs...,
)
    return IF(; N, param, synapse, spike, SYNT = typeof(synapse), kwargs...)
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target = nothing,
) where {T<:IF}
    syn = get_synapse_symbol(post.synapse, sym)
    sym = Symbol(syn)
    g = getfield(post.receptors, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end


"""
    update_neuron!(p::AbstractGeneralizedIF, param::AbstractGeneralizedIFParameter, dt)

Third stage of `integrate!` for generalized integrate-and-fire populations: integrate the
membrane (and adaptation) equations over `dt` using the synaptic current `p.syn_curr`, detect
spikes, apply reset and refractoriness. Each model has its own method (`IF`, `AdEx` with scalar
or per-neuron parameters, `ExtendedIF`, multicompartment models); see their docstrings for the
update order.

This method (`IF`): forward Euler, threshold `Vt`, reset `Vr`, refractory counter `tabs`;
adaptation `w` only if `τw > 0`.
"""
function update_neuron!(
    p::IF,
    param::T,
    dt::AbstractFloat,
) where {T<:AbstractGeneralizedIFParameter}
    @unpack N, v, w, I, tabs, fire, syn_curr = p
    @unpack τm, El, R, Vt, Vr = param
    @unpack τabs = p.spike
    _check_state_lengths(N, v, w, I, tabs, fire, syn_curr)
    nabs = round(Int, τabs / dt)    # refractory steps, loop invariant
    FT = eltype(v)

    # Branchless: the membrane update is evaluated for every neuron and discarded for the refractory ones,
    # which keeps the loop free of control flow so that it vectorises.
    @inbounds @simd for i = 1:N
        ti = tabs[i]
        vi = v[i]
        refr = ti > 0
        vn = convert(FT, vi + dt/τm * (-(vi - El) + R*(-w[i] + I[i]) - R*syn_curr[i]))
        spk = !refr & (vn > Vt)
        fire[i] = spk
        v[i] = ifelse(refr, vi, ifelse(spk, Vr, vn))
        tabs[i] = ifelse(refr, ti - 1, ifelse(spk, nabs, ti))
    end


    # Adaptation current
    if (hasfield(typeof(param), :τw) && param.τw > 0)
        @unpack a, b, τw = param
        @inbounds @simd for i = 1:N
            w[i] = ifelse(fire[i], w[i] + b, w[i])
            (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
        end
    end
end

# Guard for the `@inbounds` neuron loops: every state vector must hold at least `N` entries
# (unrolled at compile time over the argument tuple, no allocation).
@inline _lengths_ok(N, ::Tuple{}) = true
@inline _lengths_ok(N, t::Tuple) = length(first(t)) >= N && _lengths_ok(N, Base.tail(t))
@inline function _check_state_lengths(N, vs::Vararg{AbstractVector})
    _lengths_ok(N, vs) || _throw_state_length(N, vs)
    return nothing
end
@noinline function _throw_state_length(N, vs)
    L = minimum(length, vs)
    throw(DimensionMismatch("population state vector of length $L is shorter than N = $N"))
end

_float64_supported(p::IF) = p.synapse isa DoubleExpSynapse

# one precision per population: the parameter objects must have the type of the state arrays (`float_type`)
function snn_kw_finalize(::Type{<:IF}, f::NamedTuple)
    FT = eltype(f.v)
    for (what, obj) in (("param", f.param), ("synapse", f.synapse), ("spike", f.spike))
        float_type(obj) === FT || throw(ArgumentError(
            "IF mixes precisions: the state is $FT but `$what` is a $(typeof(obj)) ($(float_type(obj))). " *
            "Build all with the same floating-point type, e.g. IF(VFT = Vector{Float64}, param = IFParameter(FT = Float64), " *
            "synapse = DoubleExpSynapse(FT = Float64), spike = PostSpike(FT = Float64)); the defaults follow `VFT`."))
    end
    return f
end

export IF, IFParameter


# function Heun_update_neuron!(p::IF, param::T, dt::Float32) where {T<:AbstractIFParameter}
#     function _update_neuron!(
#         Δv::Vector{Float32},
#         p::IF,
#         param::T,
#         dt::Float32,
#     ) where {T<:AbstractIFParameter}
#         @unpack N, v, ge, gi, w, I, tabs, fire = p
#         @unpack τm, Vr, El, R, E_i, E_e, τabs, gsyn_e, gsyn_i = param
#         @inbounds for i = 1:N
#             if tabs[i] > 0
#                 v[i] = Vr
#                 fire[i] = false
#                 tabs[i] -= 1
#                 continue
#             end
#             Δv[i] =
#                 (
#                     -(v[i] + Δv[i] * dt - El) / R +# leakage
#                     -ge[i] * (v[i] + Δv[i] * dt - E_e) * gsyn_e +
#                     -gi[i] * (v[i] + Δv[i] * dt - E_i) * gsyn_i +
#                     -w[i] # adaptation
#                     +
#                     I[i] #synaptic term
#                 ) * R / τm
#         end
#     end
#     @unpack Δv_temp, Δv = p
#     _update_neuron!(Δv, p, param, dt)
#     @turbo for i = 1:p.N
#         Δv_temp[i] = Δv[i]
#     end
#     _update_neuron!(Δv, p, param, dt)
#     @turbo for i = 1:p.N
#         p.v[i] += 0.5f0 * (Δv_temp[i] + Δv[i]) * dt
#     end
#     !(hasfield(typeof(param), :τw) && param.τw > 0.0f0) && (return)
#     @unpack a, b, τw = param
#     @inbounds for i = 1:N
#         (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
#     end
# end
