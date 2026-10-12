"""
    AbstractMarkramSTPParameter <: STPParameter

Abstract supertype of the Tsodyks-Markram short-term plasticity rules:
`MarkramSTPParameterEvent` (alias `MarkramSTPParameter`) and `MarkramSTPParameterHet` (both
event-driven, subtypes of `AbstractMarkramSTPParameterEvent`) and the clock-driven
`MarkramSTPParameterTimestep`. All use `MarkramSTPVariables`.
"""
abstract type AbstractMarkramSTPParameter <: STPParameter end
abstract type AbstractMarkramSTPParameterEvent <: AbstractMarkramSTPParameter end

@doc raw"""
    MarkramSTPParameterTimestep(; τD = 200ms, τF = 1500ms, U = 0.2, Wmax = 1pF, Wmin = 0pF)

Tsodyks-Markram short-term plasticity integrated at every time step (clock-driven variant of
[`MarkramSTPParameterEvent`](@ref)).

Each presynaptic neuron ``j`` has a utilisation ``u_j`` and a fraction of available
resources ``x_j``. The efficacy of all its outgoing synapses is ``ρ_s = u_j x_j`` and a spike
adds ``W_s ρ_s`` to the target conductance.

# Equations
```math
\frac{du}{dt} = \frac{U - u}{τ_F}, \qquad \frac{dx}{dt} = \frac{1 - x}{τ_D}
```
and, at a presynaptic spike (Mongillo, Barak & Tsodyks 2008), ``u \leftarrow u + U (1 - u)``,
then ``ρ_s = u\, x`` for every outgoing synapse, then ``x \leftarrow x - u\, x`` (both with
the facilitated ``u``).

# Integration
The spike jumps and the efficacy are computed in `update_traces!`, which `train!` calls before
`forward!`, so the spike of step ``n`` is transmitted with the efficacy ``u^+ x`` of that step.
`plasticity!` (after `forward!`) relaxes ``u`` and ``x`` of all neurons by one forward-Euler
step. `sim!` never calls either. Up to SNNModels 1.8.4 the spike was transmitted with the
efficacy of the previous step (``u^-``) while the depletion used ``u^+``.

# Fields
- `τD::FT = 200ms`: recovery (depression) time constant of ``x`` (ms).
- `τF::FT = 1500ms`: facilitation time constant of ``u`` (ms).
- `U::FT = 0.2`: baseline utilisation (dimensionless).
- `Wmax::FT = 1pF`, `Wmin::FT = 0pF`: unused by the update (kept for compatibility).

# References
Tsodyks, M. V. & Markram, H. (1997). PNAS 94, 719-723; Markram, H., Wang, Y. & Tsodyks, M.
(1998). PNAS 95, 5323-5328; Mongillo, G., Barak, O. & Tsodyks, M. (2008). Science 319,
1543-1546 (the formulation named in the code).
"""
MarkramSTPParameterTimestep

@snn_kw struct MarkramSTPParameterTimestep{FT = Float32} <: AbstractMarkramSTPParameter
    τD::FT = 200ms # τx
    τF::FT = 1500ms # τu
    U::FT = 0.2
    Wmax::FT = 1.0pF
    Wmin::FT = 0.0pF
end

@doc raw"""
    MarkramSTPParameter(; τD = 200ms, τF = 1500ms, U = 0.2, Wmax = 1pF, Wmin = 0pF)
    MarkramSTPParameterEvent(; τD = 200ms, τF = 1500ms, U = 0.2, Wmax = 1pF, Wmin = 0pF)

Tsodyks-Markram short-term plasticity (depression and facilitation), event-driven.
`MarkramSTPParameter` is an alias (a non-constant global) of `MarkramSTPParameterEvent`.
Pass it to `SpikingSynapse` with `STPParam = MarkramSTPParameter()`.

The model describes the refractoriness of release: a spike uses the fraction ``u`` of the
available resources ``x``, which recover with time constant ``τ_D`` (depression); each spike
increases ``u``, which relaxes to ``U`` with time constant ``τ_F`` (facilitation). The
efficacy of every outgoing synapse ``s`` of the presynaptic neuron ``j`` is ``ρ_s = u_j x_j``
and the spike adds ``W_s ρ_s`` to the target conductance.

# Equations
Between spikes
```math
\frac{du}{dt} = \frac{U - u}{τ_F}, \qquad \frac{dx}{dt} = \frac{1 - x}{τ_D}.
```
At a presynaptic spike at time ``t_n``, with ``Δ = t_n - t_{n-1}`` the interval from the
previous spike of the same neuron:
```math
\begin{aligned}
u^- &= U - (U - u)\, e^{-Δ/τ_F}, &\qquad x^- &= 1 - (1 - x)\, e^{-Δ/τ_D},\\
u^+ &= u^- + U (1 - u^-), & & \\
ρ &= u^+ x^-, & x &\leftarrow x^- - u^+ x^- ,\qquad u \leftarrow u^+ .
\end{aligned}
```
This is the formulation of Mongillo, Barak & Tsodyks (2008): the utilisation jumps first, and
the same ``u^+`` sets both the transmitted efficacy and the depletion of resources. From rest
(``u = U``, ``x = 1``) the first spike is transmitted with ``ρ = U (2 - U)``.

!!! note "Changed in SNNModels 1.9.0"
    Up to 1.8.4 the efficacy used ``u^-`` (before the jump) while the depletion used ``u^+``,
    a mix of the Markram et al. (1998) and Mongillo et al. (2008) conventions that depressed
    more than either. Efficacies are now higher, e.g. ``0.51`` instead of ``0.30`` for the first
    spike with ``U = 0.3``.

# Integration
Exact solution between spikes, evaluated only at presynaptic spikes (no per-step work for
silent neurons). The update is done in `update_traces!`, which `train!` calls *before*
`forward!`, so the spike of the current step is transmitted with the new efficacy.
`plasticity!` does nothing for this rule. Under `sim!` neither is called and `ρ` keeps its
value (1 for a new synapse).

# Fields
- `τD::FT = 200ms`: recovery (depression) time constant of ``x`` (ms).
- `τF::FT = 1500ms`: facilitation time constant of ``u`` (ms).
- `U::FT = 0.2`: baseline utilisation (dimensionless).
- `Wmax::FT = 1pF`, `Wmin::FT = 0pF`: unused by the update (kept for compatibility).

# References
Tsodyks, M. V. & Markram, H. (1997). PNAS 94, 719-723; Markram, H., Wang, Y. & Tsodyks, M.
(1998). PNAS 95, 5323-5328; Mongillo, G., Barak, O. & Tsodyks, M. (2008). Science 319,
1543-1546 (the formulation named in the code).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 1.0),
                         STPParam = SNN.MarkramSTPParameter(τD = 200ms, τF = 1500ms, U = 0.2))
SNN.train!(model = SNN.compose(; E, P, syn), duration = 500ms)  # STP needs train!
extrema(syn.ρ)
```
"""
MarkramSTPParameterEvent

@snn_kw struct MarkramSTPParameterEvent{FT = Float32} <: AbstractMarkramSTPParameterEvent
    τD::FT = 200ms # τx
    τF::FT = 1500ms # τu
    U::FT = 0.2
    Wmax::FT = 1.0pF
    Wmin::FT = 0.0pF
end
@doc raw"""
    MarkramSTPParameterHet(; τD::Vector{Float32}, τF::Vector{Float32}, U::Vector{Float32})

Event-driven Tsodyks-Markram short-term plasticity with one parameter set per presynaptic
neuron. The equations and the integration are those of [`MarkramSTPParameterEvent`](@ref), with
``U``, ``τ_F`` and ``τ_D`` replaced by `U[j]`, `τF[j]` and `τD[j]` of the presynaptic neuron
``j``.

# Fields (no defaults, all required; vectors of length `Npre`)
- `τD::VFT`: recovery (depression) time constants (ms).
- `τF::VFT`: facilitation time constants (ms).
- `U::VFT`: baseline utilisations.

Unlike the homogeneous rules there are no `Wmax`/`Wmin` fields.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.IF(N = 10)
stp = SNN.MarkramSTPParameterHet(τD = fill(200f0, 50), τF = fill(1500f0, 50), U = rand(Float32, 50))
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 1.0), STPParam = stp)
```
"""
MarkramSTPParameterHet

@snn_kw struct MarkramSTPParameterHet{VFT = Vector{Float32}} <: AbstractMarkramSTPParameterEvent
    τD::VFT
    τF::VFT
    U::VFT
end

MarkramSTPParameter =  MarkramSTPParameterEvent 
needs_transpose(::AbstractMarkramSTPParameter) = false   # the Markram rules walk the outputs of the presynaptic neurons only
"""
    MarkramSTPParameter

Alias of [`MarkramSTPParameterEvent`](@ref) (event-driven Tsodyks-Markram short-term
plasticity); `MarkramSTPParameter(; τD = 200ms, τF = 1500ms, U = 0.2)` builds a
`MarkramSTPParameterEvent`.
"""
MarkramSTPParameter

"""
    MarkramSTPVariables(; Npre, Npost, u = zeros(Npre), x = ones(Npre), _ρ = ones(Npre),
                        last_spike = fill(-Inf, Npre), active = [true])

State of the Markram STP rules, one entry per presynaptic neuron. Created by
`plasticityvariables(param, Npre, Npost)`, which sets `u .= U`, `x .= 1` and `_ρ .= U`
(for `MarkramSTPParameterHet`, `U` is a vector and is broadcast elementwise).

# Fields
- `Npost`, `Npre`: number of postsynaptic and presynaptic neurons.
- `u`: utilisation of each presynaptic neuron.
- `x`: fraction of available resources of each presynaptic neuron.
- `_ρ`: efficacy ``u x`` that the last spike of each presynaptic neuron transmitted, copied to the synapse vector `c.ρ`.
- `last_spike`: time of the previous spike (ms), used by the event-driven rules.

!!! warning "Stored `u` and `x` are the values at the last spike"
    With the event-driven rules (`MarkramSTPParameterEvent`, `MarkramSTPParameterHet`) `u` and `x` are
    updated only at a presynaptic spike (to ``u^+`` and the depleted ``x``). Their value at time `t` is
    `U - (U - u) exp(-(t - last_spike)/τF)` and `1 - (1 - x) exp(-(t - last_spike)/τD)`; read it with
    `plasticity_value(syn, :STPVars, :u, t)`. `MarkramSTPParameterTimestep` stores the values.
- `active`: the rule is applied only if `any(active)`; see `set_STP!`.
"""
MarkramSTPVariables

@snn_kw struct MarkramSTPVariables{VFT = Vector{Float32},IT = Int} <: STPVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npre) # presynaptic state
    x::VFT = ones(Npre)  # presynaptic state
    _ρ::VFT = ones(Npre) # presynaptic state
    last_spike::VFT = fill(-Inf, Npre)
    active::VBT = [true]
end

function plasticityvariables(param::T, Npre, Npost) where {T<:AbstractMarkramSTPParameter}
    variables = MarkramSTPVariables(Npre = Npre, Npost = Npost, VFT = _vft(param))
    ## initialize variables
    variables.u .= param.U
    variables.x .= 1.0
    variables._ρ .= param.U
    return variables
end

lazy_variables(::MarkramSTPVariables) = (:u, :x)

# value at time t of the event-updated u, x: relaxation from the last spike (U, τF, τD scalars or per-neuron vectors)
function current_value(v::MarkramSTPVariables, param::AbstractMarkramSTPParameterEvent, key::Symbol, t::Real)
    key === :u && return _relax_event(v.u, v.last_spike, param.U, param.τF, t)
    key === :x && return _relax_event(v.x, v.last_spike, one(eltype(v.x)), param.τD, t)
    return getfield(v, key)
end
# the clock-driven rule relaxes u and x at every step: the stored values are the values
current_value(v::MarkramSTPVariables, param::MarkramSTPParameterTimestep, key::Symbol, t::Real) = getfield(v, key)

_relax_event(y, last_spike, target, τ, t::Real) =
    target .- (target .- y) .* exp.(-max.(t .- last_spike, zero(eltype(y))) ./ τ)

# `ρ[s] = _ρ[j]` for every outgoing synapse of each spiking `j`, partitioned by postsynaptic range like `forward!`
# (the per-neuron state is updated serially by the callers).
function _set_efficacy!(c, spikes, _ρ)
    @unpack colptr, I, ρ = c
    Npost = length(c.fireI)
    ptr = _syn_chunk_ptr(_chunkptr(c), Npost, colptr, I, spikes)
    if ptr === nothing
        @inbounds for j in spikes
            @simd for s = colptr[j]:(colptr[j+1]-1)
                ρ[s] = _ρ[j]
            end
        end
    else
        foreach_chunk_c(Npost) do ch, rng
            _set_efficacy_chunk!(ρ, _ρ, ptr, ch, spikes)
        end
    end
    return nothing
end
@inline function _set_efficacy_chunk!(ρ, _ρ, ptr, ch, spikes)
    @inbounds for j in spikes
        @simd for s = ptr[ch, j]:(ptr[ch+1, j]-1)
            ρ[s] = _ρ[j]
        end
    end
end

function update_traces!(
    c::PT,
    param::MarkramSTPParameterEvent,
    variables::MarkramSTPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, ρ, index = c
    @unpack u, x, _ρ = variables
    @unpack U, τF, τD = param

    # @inbounds @simd 
    FT = eltype(u)
    tnow = FT(get_time(T))
    ΔT = zero(FT)
    for j in presynaptic_spikes(c) # Iterate over the presynaptic neurons that fired
        ΔT = tnow > variables.last_spike[j] ? tnow - variables.last_spike[j] : zero(FT)
        variables.last_spike[j] = tnow
        # relax u and x over the interval since the previous spike (exact solution)
        u[j] = U - (U - u[j]) * exp(-ΔT / τF)
        x[j] = 1 - (1 - x[j]) * exp(-ΔT / τD)
        # Mongillo, Barak & Tsodyks 2008: u jumps first; release and depletion both use u+
        u[j] += U * (1 - u[j])
        _ρ[j] = u[j] * x[j]
        x[j] -= u[j] * x[j]
    end
    _set_efficacy!(c, presynaptic_spikes(c), _ρ)
end

function update_traces!(
    c::PT,
    param::MarkramSTPParameterHet,
    variables::MarkramSTPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, ρ, index = c
    @unpack u, x, _ρ = variables
    @unpack U, τF, τD = param

    # @inbounds @simd 
    FT = eltype(u)
    tnow = FT(get_time(T))
    ΔT = zero(FT)
    for j in presynaptic_spikes(c) # Iterate over the presynaptic neurons that fired
        ΔT = tnow > variables.last_spike[j] ? tnow - variables.last_spike[j] : zero(FT)
        variables.last_spike[j] = tnow
        # relax u and x over the interval since the previous spike (exact solution)
        u[j] = U[j] - (U[j] - u[j]) * exp(-ΔT / τF[j])
        x[j] = 1 - (1 - x[j]) * exp(-ΔT / τD[j])
        # Mongillo, Barak & Tsodyks 2008: u jumps first; release and depletion both use u+
        u[j] += U[j] * (1 - u[j])
        _ρ[j] = u[j] * x[j]
        x[j] -= u[j] * x[j]
    end
    _set_efficacy!(c, presynaptic_spikes(c), _ρ)
end


function plasticity!(
    c::PT,
    param::MET,
    variables::MarkramSTPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse, MET<:AbstractMarkramSTPParameterEvent}
end

function plasticity!(
    c::PT,
    param::MarkramSTPParameterTimestep,
    plasticity::MarkramSTPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, ρ, index = c
    @unpack u, x, _ρ = plasticity
    @unpack U, τF, τD, Wmax, Wmin = param

    # relaxation between spikes (forward Euler); the spike jumps are applied in update_traces!
    @turbo for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
        @fastmath u[j] += dt * (U - u[j]) / τF # facilitation
        @fastmath x[j] += dt * (1 - x[j]) / τD # depression
    end
end

# Mongillo, Barak & Tsodyks 2008 order at a presynaptic spike: u jumps first, the efficacy
# u+ x- is used by forward! in the same step, then x is depleted by u+ x-.
function update_traces!(
    c::PT,
    param::MarkramSTPParameterTimestep,
    variables::MarkramSTPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack colptr, ρ = c
    @unpack u, x, _ρ = variables
    @unpack U = param
    @inbounds for j in presynaptic_spikes(c)
        u[j] += U * (1 - u[j])
        _ρ[j] = u[j] * x[j]
        x[j] -= u[j] * x[j]
    end
    _set_efficacy!(c, presynaptic_spikes(c), _ρ)
end


export MarkramSTPParameter, MarkramSTPVariables, plasticityvariables, plasticity! , update_traces! , MarkramSTPParameterEvent, MarkramSTPParameterTimestep, MarkramSTPParameterHet
