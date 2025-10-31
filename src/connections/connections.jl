"""
    AbstractConnectionParameter <: AbstractParameter
"""
abstract type AbstractConnectionParameter <: AbstractParameter end

"""
    AbstractConnection

An abstract type representing a connection. Any struct inheriting from this type must implement:

# Methods
- `forward!(c::Receptors, param::SynapseParameter)`: Propagates the signal through the synapse.
- `plasticity!(c::Receptors, param::SynapseParameter, dt::Float32, T::Time)`: Updates the synapse parameters based on plasticity rules.
"""
abstract type AbstractConnection <: AbstractComponent end

include("empty.jl")
include("rate_synapse.jl")
include("fl_synapse.jl")
include("fl_sparse_synapse.jl")
include("pinning_synapse.jl")
include("pinning_sparse_synapse.jl")
include("spike_rate_synapse.jl")

"""
    AbstractSparseSynapse <: AbstractConnection
"""
abstract type AbstractSparseSynapse <: AbstractConnection end
"""
    AbstractSpikingSynapse <: AbstractSparseSynapse
"""
abstract type AbstractSpikingSynapse <: AbstractSparseSynapse end
"""
    SpikingSynapseParameter <: AbstractConnectionParameter
"""
struct SpikingSynapseParameter <: AbstractConnectionParameter end
"""
    PlasticityVariables
"""
abstract type PlasticityVariables end
"""
    PlasticityParameter
"""
abstract type PlasticityParameter end
"""
    AbstractConnectivity
"""
abstract type AbstractConnectivity end
Connectivity = Union{NamedTuple,AbstractMatrix}

include("sparse_plasticity.jl")
include("spiking_synapse.jl")

abstract type MetaPlasticityParameter <: AbstractConnectionParameter end
abstract type AbstractMetaPlasticity <: AbstractConnection end
"""
    AbstractNormalization <: AbstractConnection
"""
abstract type AbstractNormalization <: AbstractMetaPlasticity end
include("metaplasticity/normalization.jl")
include("metaplasticity/aggregate_scaling.jl")
include("metaplasticity/turnover.jl")