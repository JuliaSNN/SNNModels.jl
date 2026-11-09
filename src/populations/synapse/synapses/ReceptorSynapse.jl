abstract type AbstractReceptorParameter <: AbstractSynapseParameter end
abstract type AbstractReceptorVariable <: AbstractSynapseVariable end

@doc raw"""
    ReceptorSynapse(; syn = SomaReceptors, NMDA = NMDAVoltageDependency(),
                      glu_receptors = [1, 2], gaba_receptors = [3, 4], decay = 64, exact = true)
    ReceptorSynapse(syn::ReceptorArray, NMDA::NMDAVoltageDependency{Float32}; kwargs...)

Conductance-based synapse built from an arbitrary list of receptors (`Receptor`), each with its
own rise and decay time constants, peak conductance and reversal potential, and optional NMDA
magnesium block. Spikes arriving on the `glu` input drive the receptors listed in
`glu_receptors`, spikes on the `gaba` input drive those in `gaba_receptors`.

# Equations
Each receptor ``r`` (an entry of `syn`, see `Receptor`) has a conductance ``g_r`` and an
auxiliary rise variable ``h_r``. A spike of weight ``w`` arriving on the receptor group of
``r`` increments ``h_r`` by ``\alpha_r w`` with ``\alpha_r = 1/\tau_{r} - 1/\tau_{d}``:
```math
\frac{dh_r}{dt} = -\frac{h_r}{\tau_{r}} + \alpha_r \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_r}{dt} = -\frac{g_r}{\tau_{d}} + h_r
```
With this choice a unit weight gives ``g_r(t) = e^{-t/\tau_d} - e^{-t/\tau_r}``, whose peak is
``1/\mathrm{norm\_synapse}(\tau_r, \tau_d)``; since ``g_{syn} = g_0\,\mathrm{norm\_synapse}(\tau_r,\tau_d)``
the peak conductance of a unit weight is exactly ``g_0`` (nS), for any `dt` (the integration below is exact between spikes; the
sampled peak on the time grid is at most slightly below ``g_0``). Before 2.2.0 the peak was ``c(dt)\,g_0`` with
``c = \alpha\,dt / (1 - e^{-\alpha\,dt}) > 1`` (about 1.23 for AMPA at `dt = 0.125` ms); see the CHANGELOG for the migration rule. The synaptic current is
```math
I_{syn} = \sum_r g_{syn,r}\, g_r\,(V - E_{rev,r})\, B_r(V), \qquad
B_r(V) = \begin{cases} 1 & \text{if } nmda_r = 0 \\
\left(1 + \frac{[\mathrm{Mg}]}{b}\, e^{k V}\right)^{-1} & \text{otherwise}\end{cases}
```
where ``b``, ``k`` and ``[\mathrm{Mg}]`` are the fields of the `NMDAVoltageDependency` stored in `NMDA`.

# Integration
Exact integration of the linear system over one step, receptor by receptor (the input arrives at the beginning of the step):
`h += α w`; `g = e_d g + K h`; `h = e_r h`, with `e_d = exp(-dt/τd)`, `e_r = exp(-dt/τr)` and
`K = (e_d - e_r)/α` (`K = dt e_d` when `α = 0`). The coefficients are evaluated in `Float64` from `dt` at every call, outside the
loop over neurons, and rounded to the precision of the population. A unit weight gives, at every grid point, the exact kernel
`e^{-t/τd} - e^{-t/τr}` up to rounding. With `exact = false` (legacy, used by `PaperScheme`/`TripodPaper` and by releases up to
2.1) the update is the exponential-Euler one, `h += α w`; `g = E(-dt/τd) (g + dt h)`; `h = E(-dt/τr) h`, where `E` is the
power-law approximation ``(1 + x/64)^{64}`` of the exponential (`exp64`), or the one selected by `decay`; its sampled kernel is
`c(dt)` times the exact one, `c = α dt / (1 - e^{-α dt})`, so the peak conductance is `c g_0` instead of `g_0`.
The receptor buffers are then zeroed. The current is evaluated by the dendritic and point models with
the potential of the integration stage being computed: for `Tripod` and `BallAndStick` the stage potentials of the Heun method (the
Euler-predicted potential at the second stage), for the point-neuron models the potential at the beginning of the step.

# Fields
- `syn::ST = SomaReceptors`: vector of `Receptor` (a `ReceptorArray`); the default is the
  somatic AMPA, NMDA, GABAa, GABAb set `SomaReceptors`.
- `NMDA::NMDAT = NMDAVoltageDependency()`: parameters of the magnesium block (``b = 3.36``,
  ``k = -0.077`` 1/mV, ``[\mathrm{Mg}] = 1`` mM).
- `glu_receptors::VIT = [1, 2]`: indices in `syn` driven by the `glu` input.
- `gaba_receptors::VIT = [3, 4]`: indices in `syn` driven by the `gaba` input.
- `decay::Int = 64`: exponential used in the conductance update: `64` (default, as in every release up to 2.0), `32`
  (`exp32o`, the `exp32` of the published Tripod code, which maps inputs below -10 to 0), `256`, or `0` (exact `exp`).
  Used only when `exact = false`. Not available in `MultiReceptorSynapse`.
- `exact::Bool = true`: exact integration of the kinetics (peak conductance `g0` for any `dt`). `false` selects the legacy
  exponential-Euler update (peak `c(dt) g0`), kept for `TripodPaper`; do not use it elsewhere.

Type parameters: `VIT = Vector{Int}`, `ST = Vector{Receptor{Float32}}`,
`NMDAT = NMDAVoltageDependency{Float32}`. State variables: `ReceptorSynapseVars`. Input
buffers: `(glu, gaba)`, so connections target `:glu`/`:ge`/`:he` or `:gaba`/`:gi`/`:hi`.

Predefined instances: `SomaSynapse`, `TripodSomaSynapse`, `TripodDendSynapse`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.ReceptorSynapse())          # AMPA + NMDA + GABAa + GABAb
syn = SNN.ReceptorSynapse(SNN.SomaReceptors, SNN.SomaNMDA; glu_receptors = [1], gaba_receptors = [3])
```
"""
ReceptorSynapse

@snn_kw struct ReceptorSynapse{
    VIT = Vector{Int},
    ST = Vector{Receptor{Float32}},
    NMDAT = NMDAVoltageDependency{Float32},
} <: AbstractReceptorParameter
    ## Synapses
    syn::ST=SomaReceptors
    NMDA::NMDAT = NMDAVoltageDependency()
    glu_receptors::VIT = [1, 2]
    gaba_receptors::VIT = [3, 4]
    decay::Int = 64
    exact::Bool = true
end

snn_kw_finalize(::Type{<:ReceptorSynapse}, f::NamedTuple) = (_check_exp_approx("ReceptorSynapse decay", f.decay); f)

ReceptorSynapse(syn::ReceptorArray, NMDA::NMDAVoltageDependency{Float32}; kwargs...) =
    ReceptorSynapse(; kwargs..., syn = syn, NMDA = NMDA)

function synaptic_receptors(synapse::ReceptorSynapse, N::Int, ::Type{FT}) where {FT<:AbstractFloat}
    return (glu = zeros(FT, N), gaba = zeros(FT, N))
end

# Copy in precision `T` (the object itself when already `T`); predefined receptors and NMDA blocks are rebuilt from their
# decimal definition (see `_PREDEFINED`).
_convert_receptors(syn::AbstractVector{<:Receptor}, ::Type{T}) where {T} = Receptor{T}[convert_precision(r, T) for r in syn]
function convert_precision(r::Receptor{S}, ::Type{T}) where {S,T<:AbstractFloat}
    S === T && return r
    d = _from_definition(r, T)
    d === nothing || return d::Receptor{T}
    return Receptor{T}(map(f -> _convert_value(getfield(r, f), T), fieldnames(typeof(r)))...)
end
function convert_precision(n::NMDAVoltageDependency{S}, ::Type{T}) where {S,T<:AbstractFloat}
    S === T && return n
    d = _from_definition(n, T)
    d === nothing || return d::NMDAVoltageDependency{T}
    return NMDAVoltageDependency{T}(T(n.b), T(n.k), T(n.mg), n.approx)
end
function convert_precision(x::ReceptorSynapse, ::Type{T}) where {T<:AbstractFloat}
    _float_type(typeof(x)) === T && return x
    syn, NMDA = _convert_receptors(x.syn, T), convert_precision(x.NMDA, T)
    return ReceptorSynapse{typeof(x.glu_receptors),typeof(syn),typeof(NMDA)}(syn, NMDA, x.glu_receptors, x.gaba_receptors, x.decay, x.exact)
end


@doc raw"""
    MultiReceptorSynapse(; syn = SomaReceptors, NMDA = NMDAVoltageDependency())
    MultiReceptorSynapse(syn::ReceptorArray; kwargs...)

Receptor-based conductance synapse in which every receptor is driven by the input buffer named
by its `target` field, instead of the fixed `glu`/`gaba` split of `ReceptorSynapse`. The
receptors are grouped by `target` (`infer_receptors`) and the population gets one input
buffer per distinct target, so a connection can address, for example, an `:AMPA` or a
`:GABAb` group directly (`SpikingSynapse(pre, post, :AMPA; conn)`). All receptors must have
`target` different from `:none`.

The kinetics, the NMDA block and the integration scheme are the same as `ReceptorSynapse`:

# Equations
Each receptor ``r`` (an entry of `syn`, see `Receptor`) has a conductance ``g_r`` and an
auxiliary rise variable ``h_r``. A spike of weight ``w`` arriving on the receptor group of
``r`` increments ``h_r`` by ``\alpha_r w`` with ``\alpha_r = 1/\tau_{r} - 1/\tau_{d}``:
```math
\frac{dh_r}{dt} = -\frac{h_r}{\tau_{r}} + \alpha_r \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_r}{dt} = -\frac{g_r}{\tau_{d}} + h_r
```
With this choice a unit weight gives ``g_r(t) = e^{-t/\tau_d} - e^{-t/\tau_r}``, whose peak is
``1/\mathrm{norm\_synapse}(\tau_r, \tau_d)``; since ``g_{syn} = g_0\,\mathrm{norm\_synapse}(\tau_r,\tau_d)``
the peak conductance of a unit weight is exactly ``g_0`` (nS), for any `dt` (the integration below is exact between spikes; the
sampled peak on the time grid is at most slightly below ``g_0``). Before 2.2.0 the peak was ``c(dt)\,g_0`` with
``c = \alpha\,dt / (1 - e^{-\alpha\,dt}) > 1`` (about 1.23 for AMPA at `dt = 0.125` ms); see the CHANGELOG for the migration rule. The synaptic current is
```math
I_{syn} = \sum_r g_{syn,r}\, g_r\,(V - E_{rev,r})\, B_r(V), \qquad
B_r(V) = \begin{cases} 1 & \text{if } nmda_r = 0 \\
\left(1 + \frac{[\mathrm{Mg}]}{b}\, e^{k V}\right)^{-1} & \text{otherwise}\end{cases}
```
where ``b``, ``k`` and ``[\mathrm{Mg}]`` are the fields of the `NMDAVoltageDependency` stored in `NMDA`.

# Integration
Exact integration over one step, receptor by receptor: `h += α w`; `g = e_d g + K h`; `h = e_r h` with
`e_d = exp(-dt/τd)`, `e_r = exp(-dt/τr)`, `K = (e_d - e_r)/α` (see `ReceptorSynapse`). The receptor buffers are then zeroed.
The current is evaluated with the potential of the integration stage being computed (see `ReceptorSynapse`).

# Fields
- `syn::ST = SomaReceptors`: vector of `Receptor`.
- `NMDA::NMDAT = NMDAVoltageDependency()`: magnesium block parameters.
- `receptors::REC = infer_receptors(syn)`: `NamedTuple` mapping each target symbol to the
  indices of the receptors in `syn` (computed, normally not given).

The positional form `MultiReceptorSynapse(syn; kwargs...)` is equivalent to
`MultiReceptorSynapse(; syn, kwargs...)`. State variables: `ReceptorSynapseVars`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
recs = SNN.Receptors(SNN.Receptor(E_rev = 0mV, τr = 0.5ms, τd = 3ms, g0 = 1nS, target = :AMPA),
                     SNN.Receptor(E_rev = -70mV, τr = 0.5ms, τd = 6ms, g0 = 1nS, target = :GABA))
syn = SNN.MultiReceptorSynapse(recs)
P = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS), synapse = syn)
keys(P.receptors)   # (:GABA, :AMPA)
```
"""
MultiReceptorSynapse

@snn_kw struct MultiReceptorSynapse{
    ST = Vector{Receptor{Float32}},
    NMDAT = NMDAVoltageDependency{Float32},
    REC<:NamedTuple,
} <: AbstractReceptorParameter
    syn::ST=SomaReceptors
    NMDA::NMDAT = NMDAVoltageDependency()
    receptors::REC = infer_receptors(syn)
end

# positional form, equivalent to the keyword form `MultiReceptorSynapse(; syn)`
MultiReceptorSynapse(syn::ReceptorArray; kwargs...) = MultiReceptorSynapse(; syn = syn, kwargs...)

function convert_precision(x::MultiReceptorSynapse, ::Type{T}) where {T<:AbstractFloat}
    _float_type(typeof(x)) === T && return x
    syn, NMDA = _convert_receptors(x.syn, T), convert_precision(x.NMDA, T)
    return MultiReceptorSynapse{typeof(syn),typeof(NMDA),typeof(x.receptors)}(syn, NMDA, x.receptors)
end

function synaptic_receptors(synapse::MultiReceptorSynapse, N::Int, ::Type{FT}) where {FT<:AbstractFloat}

    receptors = Dict{Symbol,Vector{FT}}()
    for rec in keys(synapse.receptors)
        receptors[rec] = zeros(FT, N)
    end
    return (; receptors...)
end

"""
    ReceptorSynapseVars{MFT} <: AbstractReceptorVariable

State variables of `ReceptorSynapse` and `MultiReceptorSynapse`, created by
`synaptic_variables(synapse, N)` with one column per receptor of `synapse.syn`.

# Fields
- `N::Int = 100`: number of neurons.
- `g::MFT = zeros(Float32, N, 4)`: conductances (before scaling by `gsyn`), `N x n_receptors`.
- `h::MFT = zeros(Float32, N, 4)`: auxiliary rise variables, `N x n_receptors`.
"""
ReceptorSynapseVars
@snn_kw struct ReceptorSynapseVars{MFT = Matrix{Float32}} <: AbstractReceptorVariable
    N::Int = 100
    g::MFT = zeros(eltype(MFT), N, 4)
    h::MFT = zeros(eltype(MFT), N, 4)
end


function synaptic_variables(synapse::T, N::Int, ::Type{FT}) where {T<:AbstractReceptorParameter,FT<:AbstractFloat}
    num_receptors = length(synapse.syn)
    return ReceptorSynapseVars(;
        N = N,
        MFT = Matrix{FT},
        g = zeros(FT, N, num_receptors),
        h = zeros(FT, N, num_receptors),
    )
end

@inline function update_synapses!(
    p::P,
    synapse::ReceptorSynapse,
    receptors::RECT,
    synvars::ReceptorSynapseVars,
    dt::AbstractFloat,
) where {P<:AbstractPopulation,RECT<:NamedTuple}
    update_synapses!(p, synapse, receptors, synvars, dt, 1:Int(synvars.N))
    fill!(receptors.glu, zero(eltype(receptors.glu)))     # the whole buffers (the chunk method clears its neurons only)
    fill!(receptors.gaba, zero(eltype(receptors.gaba)))
end

# Chunk method: the neurons `rng` only, and the input buffers `glu`, `gaba` are cleared on `rng` only.
@inline function update_synapses!(
    p::P,
    synapse::ReceptorSynapse,
    receptors::RECT,
    synvars::ReceptorSynapseVars,
    dt::AbstractFloat,
    rng::UnitRange{Int},
) where {P<:AbstractPopulation,RECT<:NamedTuple}
    @unpack glu_receptors, gaba_receptors, decay, exact = synapse
    @unpack glu, gaba = receptors
    if exact
        _exact_receptors!(synapse, synvars, glu, gaba, dt, rng)
    # legacy exponential-Euler update; the exponential is chosen by `decay`: one compiled method per approximation, selected by a branch (no dynamic dispatch)
    elseif decay == 64
        _decay_receptors!(Val(64), synapse, synvars, glu, gaba, dt, rng)
    elseif decay == 32
        _decay_receptors!(Val(32), synapse, synvars, glu, gaba, dt, rng)
    elseif decay == 256
        _decay_receptors!(Val(256), synapse, synvars, glu, gaba, dt, rng)
    else
        _decay_receptors!(Val(0), synapse, synvars, glu, gaba, dt, rng)
    end
    z = zero(eltype(glu))
    @inbounds for i in rng
        glu[i] = z
        gaba[i] = z
    end
end

# Coefficients of the exact one-step update of a receptor, `g' = ed g + K h`, `h' = er h`, evaluated in Float64 and rounded to `FT`.
# `K = (ed - er)/α` is written with `expm1` (ed - er = -ed expm1(-α dt)) to avoid the cancellation for small `dt`; the
# limit `K = dt ed` is used when `α == 0`. The stored `α` is used in both the spike increment and `K`, so that the kernel is exact
# whatever the rounding of `α`.
@inline function _exact_coeffs(::Type{FT}, dt, τr⁻, τd⁻, α) where {FT<:AbstractFloat}
    dt64, a = Float64(dt), Float64(α)
    ed = exp(-dt64 * Float64(τd⁻))
    er = exp(-dt64 * Float64(τr⁻))
    K = a == 0.0 ? dt64 * ed : -ed * expm1(-a * dt64) / a
    return FT(ed), FT(er), FT(K)
end

# One receptor `n` on the neurons `rng`: spike input `inp`, then the exact update of `g` and `h`.
@inline function _exact_receptor!(g, h, inp, n::Int, ed, er, K, α, rng)
    @turbo for i ∈ rng
        hn = h[i, n] + inp[i] * α
        g[i, n] = ed * g[i, n] + K * hn
        h[i, n] = er * hn
    end
end

@inline function _exact_receptors!(synapse, synvars, glu, gaba, dt, rng)
    @unpack glu_receptors, gaba_receptors = synapse
    @unpack g, h = synvars
    FT = eltype(g)
    @inbounds for n in glu_receptors
        r = synapse.syn[n]
        ed, er, K = _exact_coeffs(FT, dt, r.τr⁻, r.τd⁻, r.α)
        _exact_receptor!(g, h, glu, n, ed, er, K, FT(r.α), rng)
    end
    @inbounds for n in gaba_receptors
        r = synapse.syn[n]
        ed, er, K = _exact_coeffs(FT, dt, r.τr⁻, r.τd⁻, r.α)
        _exact_receptor!(g, h, gaba, n, ed, er, K, FT(r.α), rng)
    end
end

# Legacy exponential-Euler update (`exact = false`)
@inline function _decay_receptors!(::Val{N}, synapse, synvars, glu, gaba, dt, rng) where {N}
    @unpack glu_receptors, gaba_receptors = synapse
    @unpack g, h = synvars
    @inbounds for n in glu_receptors
        @unpack τr⁻, τd⁻, α = synapse.syn[n]
        @turbo for i ∈ rng
            h[i, n] += glu[i] * α
            g[i, n] = _expapprox(Val(N), -dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = _expapprox(Val(N), -dt * τr⁻) * (h[i, n])
        end
    end
    @simd for n in gaba_receptors
        @unpack τr⁻, τd⁻, α = synapse.syn[n]
        @turbo for i ∈ rng
            h[i, n] += gaba[i] * α
            g[i, n] = _expapprox(Val(N), -dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = _expapprox(Val(N), -dt * τr⁻) * (h[i, n])
        end
    end
end

@inline function update_synapses!(
    p::P,
    synapse::MultiReceptorSynapse,
    receptors::RECT,
    synvars::ReceptorSynapseVars,
    dt::AbstractFloat,
) where {P<:AbstractPopulation,RECT<:NamedTuple}
    update_synapses!(p, synapse, receptors, synvars, dt, 1:Int(synvars.N))
    for name in keys(receptors)
        fill!(getfield(receptors, name), zero(eltype(getfield(receptors, name))))
    end
end

# Chunk method: the neurons `rng` only (the input buffers are cleared on `rng` only).
@inline function update_synapses!(
    p::P,
    synapse::MultiReceptorSynapse,
    receptors::RECT,
    synvars::ReceptorSynapseVars,
    dt::AbstractFloat,
    rng::UnitRange{Int},
) where {P<:AbstractPopulation,RECT<:NamedTuple}
    for name in keys(synapse.receptors)
        @inbounds for n in synapse.receptors[name]
            update_receptor!(synvars, synapse.syn[n], getfield(receptors, name), n, dt, rng)
        end
    end
    for name in keys(receptors)
        buf = getfield(receptors, name)
        @inbounds for i in rng
            buf[i] = zero(eltype(buf))
        end
    end
end

@inline function update_receptor!(
    synvars::T,
    receptor::Receptor{FT},
    target::Vector{FT},
    n::Int,
    dt::FT,
    rng::UnitRange{Int} = 1:Int(synvars.N),
) where {T<:AbstractReceptorVariable,FT<:AbstractFloat}
    @unpack N, g, h = synvars
    @unpack τr⁻, τd⁻, α = receptor
    (size(g, 1) >= N && size(h, 1) >= N && n <= size(g, 2) && n <= size(h, 2) && length(target) >= N) ||
        throw(DimensionMismatch("receptor state of $(size(g)) / $(size(h)) or input buffer of length $(length(target)) too small for N = $N receptors $n"))
    ed, er, K = _exact_coeffs(FT, dt, τr⁻, τd⁻, α)    # loop invariants
    _exact_receptor!(g, h, target, n, ed, er, K, α, rng)
end

_chunk_synapse(::AbstractReceptorParameter) = true

@inline function synaptic_current!(
    p::T,
    synapse::P,
    synvars::S,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {
    T<:AbstractGeneralizedIF,
    P<:AbstractReceptorParameter,
    S<:AbstractReceptorVariable,
    VT1<:AbstractVector,
    VT2<:AbstractVector,
}
    @unpack N = p
    @unpack g, h = synvars
    @unpack syn, NMDA = synapse
    @unpack mg, b, k = NMDA
    (length(v) >= N && length(syncurr) >= N && size(g, 1) >= N && size(g, 2) >= length(syn)) ||
        throw(DimensionMismatch("membrane potential, current or conductance matrix too small for N = $N and $(length(syn)) receptors"))
    fill!(syncurr, zero(eltype(syncurr)))
    _synaptic_current_rng!(synapse, synvars, v, syncurr, 1:Int(N))
end

# Chunk method of the six-argument form: the neurons `rng` only (clears `syncurr` on `rng` only).
@inline function synaptic_current!(
    p::T,
    synapse::P,
    synvars::S,
    v::VT1,
    syncurr::VT2,
    rng::UnitRange{Int},
) where {
    T<:AbstractGeneralizedIF,
    P<:AbstractReceptorParameter,
    S<:AbstractReceptorVariable,
    VT1<:AbstractVector,
    VT2<:AbstractVector,
}
    @inbounds for i in rng
        syncurr[i] = zero(eltype(syncurr))
    end
    _synaptic_current_rng!(synapse, synvars, v, syncurr, rng)
end

@inline function _synaptic_current_rng!(synapse, synvars, v, syncurr, rng::UnitRange{Int})
    @unpack g, h = synvars
    @unpack syn, NMDA = synapse
    @unpack mg, b, k, approx = NMDA
    mgb = mg / b    # loop invariant
    # Receptor by receptor, so every neuron accumulates in receptor order; the NMDA factor is evaluated only for NMDA receptors.
    for n in eachindex(syn)
        @unpack gsyn, E_rev, nmda = syn[n]
        if nmda == zero(nmda)
            @inbounds @simd for neuron ∈ rng
                syncurr[neuron] += gsyn * g[neuron, n] * (v[neuron] - E_rev)
            end
        elseif approx == 256
            _nmda_current!(Val(256), syncurr, g, v, n, gsyn, E_rev, k, mgb, rng)
        elseif approx == 32
            _nmda_current!(Val(32), syncurr, g, v, n, gsyn, E_rev, k, mgb, rng)
        elseif approx == 64
            _nmda_current!(Val(64), syncurr, g, v, n, gsyn, E_rev, k, mgb, rng)
        else
            _nmda_current!(Val(0), syncurr, g, v, n, gsyn, E_rev, k, mgb, rng)
        end
    end
end

# current of one NMDA receptor (with the magnesium block, exponential `_expapprox(Val(N), ·)`) added to `syncurr` on `rng`
@inline function _nmda_current!(::Val{N}, syncurr, g, v, n, gsyn, E_rev, k, mgb, rng) where {N}
    @inbounds @simd for neuron ∈ rng
        syncurr[neuron] +=
            gsyn *
            g[neuron, n] *
            (v[neuron] - E_rev) *
            (1 / (one(mgb) + mgb * _expapprox(Val(N), k * v[neuron])))
    end
end

export ReceptorSynapse, MultiReceptorSynapse
