abstract type AbstractReceptorParameter <: AbstractSynapseParameter end
abstract type AbstractReceptorVariable <: AbstractSynapseVariable end

@doc raw"""
    ReceptorSynapse(; syn = SomaReceptors, NMDA = NMDAVoltageDependency(),
                      glu_receptors = [1, 2], gaba_receptors = [3, 4])
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
the peak conductance of a unit weight is ``g_0`` (nS). The synaptic current is
```math
I_{syn} = \sum_r g_{syn,r}\, g_r\,(V - E_{rev,r})\, B_r(V), \qquad
B_r(V) = \begin{cases} 1 & \text{if } nmda_r = 0 \\
\left(1 + \frac{[\mathrm{Mg}]}{b}\, e^{k V}\right)^{-1} & \text{otherwise}\end{cases}
```
where ``b``, ``k`` and ``[\mathrm{Mg}]`` are the fields of the `NMDAVoltageDependency` stored in `NMDA`.

# Integration
Exact exponential decay over one step (exponential Euler), receptor by receptor:
`h += α w`; `g = exp(-dt/τd) (g + dt h)`; `h = exp(-dt/τr) h`. The receptor buffers are then
zeroed. The current is computed with the membrane potential at the beginning of the step.

# Fields
- `syn::ST = SomaReceptors`: vector of `Receptor` (a `ReceptorArray`); the default is the
  somatic AMPA, NMDA, GABAa, GABAb set `SomaReceptors`.
- `NMDA::NMDAT = NMDAVoltageDependency()`: parameters of the magnesium block (``b = 3.36``,
  ``k = -0.077`` 1/mV, ``[\mathrm{Mg}] = 1`` mM).
- `glu_receptors::VIT = [1, 2]`: indices in `syn` driven by the `glu` input.
- `gaba_receptors::VIT = [3, 4]`: indices in `syn` driven by the `gaba` input.

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
end

ReceptorSynapse(syn::ReceptorArray, NMDA::NMDAVoltageDependency{Float32}; kwargs...) =
    ReceptorSynapse(; kwargs..., syn = syn, NMDA = NMDA)

function synaptic_receptors(synapse::ReceptorSynapse, N::Int, ::Type{FT}) where {FT<:AbstractFloat}
    return (glu = zeros(FT, N), gaba = zeros(FT, N))
end

# Copy in the precision `T`: every receptor converted to `Receptor{T}`, the NMDA block to `NMDAVoltageDependency{T}` (the object
# itself when it is already `T`, so the Float32 defaults `TripodSomaSynapse`, `TripodDendSynapse` are used as they are).
_convert_receptors(syn::AbstractVector{<:Receptor}, ::Type{T}) where {T} = Receptor{T}[convert_precision(r, T) for r in syn]
function convert_precision(r::Receptor{S}, ::Type{T}) where {S,T<:AbstractFloat}
    S === T && return r
    return Receptor{T}(map(f -> _convert_value(getfield(r, f), T), fieldnames(typeof(r)))...)
end
function convert_precision(n::NMDAVoltageDependency{S}, ::Type{T}) where {S,T<:AbstractFloat}
    S === T && return n
    return NMDAVoltageDependency{T}(T(n.b), T(n.k), T(n.mg))
end
function convert_precision(x::ReceptorSynapse, ::Type{T}) where {T<:AbstractFloat}
    _float_type(typeof(x)) === T && return x
    syn, NMDA = _convert_receptors(x.syn, T), convert_precision(x.NMDA, T)
    return ReceptorSynapse{typeof(x.glu_receptors),typeof(syn),typeof(NMDA)}(syn, NMDA, x.glu_receptors, x.gaba_receptors)
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
the peak conductance of a unit weight is ``g_0`` (nS). The synaptic current is
```math
I_{syn} = \sum_r g_{syn,r}\, g_r\,(V - E_{rev,r})\, B_r(V), \qquad
B_r(V) = \begin{cases} 1 & \text{if } nmda_r = 0 \\
\left(1 + \frac{[\mathrm{Mg}]}{b}\, e^{k V}\right)^{-1} & \text{otherwise}\end{cases}
```
where ``b``, ``k`` and ``[\mathrm{Mg}]`` are the fields of the `NMDAVoltageDependency` stored in `NMDA`.

# Integration
Exact exponential decay over one step (exponential Euler), receptor by receptor:
`h += α w`; `g = exp(-dt/τd) (g + dt h)`; `h = exp(-dt/τr) h`. The receptor buffers are then
zeroed. The current is computed with the membrane potential at the beginning of the step.

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
    @unpack glu_receptors, gaba_receptors = synapse
    @unpack g, h = synvars
    @unpack glu, gaba = receptors
    @inbounds for n in glu_receptors
        @unpack τr⁻, τd⁻, α = synapse.syn[n]
        @turbo for i ∈ rng
            h[i, n] += glu[i] * α
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end
    @simd for n in gaba_receptors
        @unpack τr⁻, τd⁻, α = synapse.syn[n]
        @turbo for i ∈ rng
            h[i, n] += gaba[i] * α
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end
    z = zero(eltype(glu))
    @inbounds for i in rng
        glu[i] = z
        gaba[i] = z
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
    ed = exp64(-dt * τd⁻)    # loop invariants
    er = exp64(-dt * τr⁻)
    @inbounds @simd for i ∈ rng
        h[i, n] += target[i] * α
        g[i, n] = ed * (g[i, n] + dt * h[i, n])
        h[i, n] = er * (h[i, n])
    end
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
    @unpack mg, b, k = NMDA
    mgb = mg / b    # loop invariant
    # Receptor by receptor, so every neuron accumulates in the same order as the scalar loop of SNNModels <= 1.9 (bit-identical).
    # The voltage-dependent factor is only evaluated for the NMDA receptors (the multiplication by 1 of the others is exact).
    for n in eachindex(syn)
        @unpack gsyn, E_rev, nmda = syn[n]
        if nmda == zero(nmda)
            @inbounds @simd for neuron ∈ rng
                syncurr[neuron] += gsyn * g[neuron, n] * (v[neuron] - E_rev)
            end
        else
            @inbounds @simd for neuron ∈ rng
                syncurr[neuron] +=
                    gsyn *
                    g[neuron, n] *
                    (v[neuron] - E_rev) *
                    (1 / (one(mgb) + mgb * exp256(k * v[neuron])))
            end
        end
    end
end

export ReceptorSynapse, MultiReceptorSynapse
