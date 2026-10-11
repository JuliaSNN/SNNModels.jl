"""
    AbstractConnectionParameter <: AbstractParameter

Supertype of the `param` field of every connection (`AbstractConnection`). The concrete
parameter type selects, by dispatch, the `forward!`, `update_traces!` and `plasticity!`
methods that the simulation loop calls for the connection.
"""
abstract type AbstractConnectionParameter <: AbstractParameter end

include("empty.jl")
include("rate_synapse.jl")
include("fl_synapse.jl")
include("fl_sparse_synapse.jl")
include("pinning_synapse.jl")
include("pinning_sparse_synapse.jl")
include("spike_rate_synapse.jl")

@doc raw"""
    forward!(c::AbstractConnection, param::AbstractConnectionParameter, dt::AbstractFloat, T::Time)
    forward!(c, param)

Propagate the activity of the presynaptic population of connection `c` to its
postsynaptic target for one time step.

`sim!` and `train!` call the four-argument form once per step for every connection, after
all stimuli and populations have been updated. The generic four-argument method defined
here forwards to the two-argument form `forward!(c, param)`, which connection types that do
not need the time implement. `SpikingSynapse` implements the four-argument form directly.

What is added to the target depends on the connection type:
- `SpikingSynapse`: for every presynaptic neuron ``j`` that fired in this step,
  ``g_i \leftarrow g_i + W_{ij}\,\rho_{ij}`` for all its targets ``i`` (``g`` is the
  postsynaptic variable selected by `sym`, ``\rho`` the short-term efficacy). With
  delays, the increment is queued and added once the delay has elapsed.
- `RateSynapse`: ``g_i \leftarrow g_i + \sum_j W_{ij} r_j``.
- `FLSynapse`, `PINningSynapse`: ``g = W r`` (overwrite), plus the FORCE feedback term.
- metaplasticity objects (`SynapseNormalization`, `Turnover`): no-op, except
  `AggregateScaling`, which updates its activity trace and target in `forward!`.

The parameter type of a connection must be a subtype of `AbstractConnectionParameter` for the
generic four-argument method to apply.
"""
forward!(c::C, param::P, dt::AbstractFloat, T::Time) where {C<:AbstractConnection, P<:AbstractConnectionParameter} = forward!(c, param)

update_traces!(
    p::C,
    param::P,
    dt::AbstractFloat,
    T::Time,
) where {C<:AbstractConnection, P<:AbstractConnectionParameter} = nothing



"""
    AbstractSparseSynapse <: AbstractConnection

Connections that store their weights in the double sparse layout produced by `dsparse`
(fields `rowptr`, `colptr`, `I`, `J`, `index`, `W`). Required by the connectivity helpers
(`matrix`, `presynaptic`, `postsynaptic`, ...) and by the metaplasticity objects.
"""
abstract type AbstractSparseSynapse <: AbstractConnection end
"""
    AbstractSpikingSynapse <: AbstractSparseSynapse

Sparse connections driven by presynaptic spikes (`SpikingSynapse`).
"""
abstract type AbstractSpikingSynapse <: AbstractSparseSynapse end

"""
    prepare_spikes!(c::AbstractConnection)
    release_spikes!(c::AbstractConnection)

Called by `train!` and `sim!` around the update of every connection: `prepare_spikes!` lets
the connection reuse one index list of the spiking neurons for all its consumers during the
step, `release_spikes!` drops it. No-op for connections without spike index lists.
"""
prepare_spikes!(c::AbstractConnection) = nothing
release_spikes!(c::AbstractConnection) = nothing

"""
    presynaptic_spikes(c)
    postsynaptic_spikes(c)

Ascending `Int32` indices of the presynaptic (`c.fireJ`) and postsynaptic (`c.fireI`) neurons
that fired in this step. `SpikingSynapse` serves them from its `SpikeIndex` lists; the generic
methods (synapse types without the lists) scan the firing vector and allocate.
"""
presynaptic_spikes(c::AbstractSparseSynapse) = Int32[j for j in eachindex(c.fireJ) if c.fireJ[j]]
postsynaptic_spikes(c::AbstractSparseSynapse) = Int32[i for i in eachindex(c.fireI) if c.fireI[i]]

"""
    AbstractSpikingSynapseParameter <: AbstractConnectionParameter

Parameter types of `SpikingSynapse`: `SpikingSynapseParameter` (no delays) and the internal
`SpikingSynapseDelayParameter` (per-synapse delays and delivery queues).
"""
abstract type AbstractSpikingSynapseParameter <: AbstractConnectionParameter end

"""
    PlasticityVariables

Supertype of the per-connection state of a plasticity rule (traces, last spike times, STP
variables). Created by `plasticityvariables(rule, Npre, Npost)`; every concrete subtype has
an `active` flag vector that switches the rule on or off.
"""
abstract type PlasticityVariables end
"""
    PlasticityParameter

Supertype of plasticity rules: long-term rules (`LTPParameter`, passed to `SpikingSynapse`
with the `LTPParam` keyword) and short-term rules (`STPParameter`, keyword `STPParam`).
Plasticity is applied only by `train!`, never by `sim!`.
"""
abstract type PlasticityParameter end
"""
    AbstractConnectivity

Abstract placeholder type for connectivity specifications. No concrete subtype is defined in
SNNModels 1.8.4; connectivity is passed as `conn::Union{NamedTuple,AbstractMatrix}` (see
`sparse_matrix`).
"""
abstract type AbstractConnectivity end
Connectivity = Union{NamedTuple,AbstractMatrix}

"""
    convert_precision(x, ::Type{T})

Copy of the parameter object `x` (a plasticity rule, a metaplasticity parameter, a stimulus parameter, ...) with its floating
point fields and type parameters converted to `T`; `x` itself if it already has type `T`. Used by the connection and
stimulus constructors to bring a rule to the precision of the population it acts on, so a rule built with the default
`Float32` parameters works on a `Float64` population (the values are those of the `Float32` numbers: build the rule with
`FT = Float64`, for example `STDPGerstner(FT = Float64, A_pre = 1e-4)`, to have the `Float64` values) and a `Float64`
rule on a `Float32` population is rounded to `Float32`. Fields that are not floating point (flags, integers) are kept
(shared, not copied).
"""
convert_precision(x, ::Type{T}) where {T<:AbstractFloat} = x
function convert_precision(x::P, ::Type{T}) where {P<:Union{AbstractParameter,PlasticityParameter},T<:AbstractFloat}
    ps = P.parameters
    isempty(ps) && return x
    any(_is_float_param, ps) || return x
    _float_type(P) === T && return x
    newps = map(p -> _convert_param(p, T), ps)
    vals = map(f -> _convert_value(getfield(x, f), T), fieldnames(P))
    return Base.typename(P).wrapper{newps...}(vals...)
end
_is_float_param(p) = p isa Type && (p <: AbstractFloat || p <: AbstractArray{<:AbstractFloat} || p <: AbstractVector{<:AbstractVector{<:AbstractFloat}})
_convert_param(p, ::Type{T}) where {T} = !(p isa Type) ? p : p <: AbstractFloat ? T : p <: AbstractArray{<:AbstractFloat} ? Vector{T} :
    p <: AbstractVector{<:AbstractVector{<:AbstractFloat}} ? Vector{Vector{T}} : p
_convert_value(v::AbstractFloat, ::Type{T}) where {T} = T(v)
_convert_value(v::AbstractArray{<:AbstractFloat}, ::Type{T}) where {T} = convert(Vector{T}, v)
_convert_value(v::AbstractVector{<:AbstractVector{<:AbstractFloat}}, ::Type{T}) where {T} = Vector{T}[convert(Vector{T}, x) for x in v]
_convert_value(v, ::Type{T}) where {T} = v

include("sparse_plasticity.jl")
include("spiking_synapse.jl")
include("delay_kernels.jl")

"""
    MetaPlasticityParameter <: AbstractConnectionParameter

Supertype of the parameters of metaplasticity objects: `NormParam` (synaptic normalization,
aggregate scaling) and `TurnoverParam` (structural turnover).
"""
abstract type MetaPlasticityParameter <: AbstractConnectionParameter end
"""
    AbstractMetaPlasticity <: AbstractConnection

Connections that do not transmit activity but act on the weights of other connections
(`SynapseNormalization`, `AggregateScaling`, `Turnover`). They are added to the model like
any connection, and their `plasticity!` runs only under `train!`.
"""
abstract type AbstractMetaPlasticity <: AbstractConnection end
"""
    AbstractNormalization <: AbstractMetaPlasticity

Metaplasticity objects that rescale the total input weight of each postsynaptic neuron
(`SynapseNormalization`, `AggregateScaling`).
"""
abstract type AbstractNormalization <: AbstractMetaPlasticity end
include("metaplasticity/normalization.jl")
include("metaplasticity/aggregate_scaling.jl")
include("metaplasticity/turnover.jl")
