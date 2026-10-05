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
    forward!(c::AbstractConnection, param::AbstractConnectionParameter, dt::Float32, T::Time)
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
forward!(c::C, param::P, dt::Float32, T::Time) where {C<:AbstractConnection, P<:AbstractConnectionParameter} = forward!(c, param)

update_traces!(
    p::C,
    param::P,
    dt::Float32,
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

include("sparse_plasticity.jl")
include("spiking_synapse.jl")

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
