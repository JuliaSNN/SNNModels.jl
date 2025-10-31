"""
    AbstractPopulationParameter <: AbstractParameter
"""
abstract type AbstractPopulationParameter <: AbstractParameter end

"""
    AbstractPopulation

An abstract type representing a population. Any struct inheriting from this type must implement:

# Methods
- `integrate!(p::NeuronModel, param::NeuronModelParam, dt::Float32)`: Integrates the neuron model over a time step `dt` using the given parameters.
- `plasticity!(p::NeuronModel, param::NeuronModelParam, dt::Float32, T::Time)`: Updates the neuron model parameters based on plasticity rules.
"""
abstract type AbstractPopulation <: AbstractComponent end

integrate!(p::AbstractPopulation, param::AbstractPopulationParameter, dt::Float32) = nothing
plasticity!(
    p::AbstractPopulation,
    param::AbstractPopulationParameter,
    dt::Float32,
    T::Time,
) = nothing

## Spikes
abstract type AbstractSpikeParameter end
include("spike/postspike.jl")

## Neurons
include("poisson.jl")
include("iz.jl")
include("hh.jl")
include("morrislecar.jl")
include("rate.jl")
include("identity.jl")

## gIF and synapses
abstract type AbstractGeneralizedIFParameter <: AbstractPopulationParameter end
abstract type AbstractGeneralizedIF <: AbstractPopulation end

## Synapses
include("synapse/receptors.jl")
include("synapse/synapses.jl")
include("synapse/receptor_types.jl")
include("synapse/synaptic_targets.jl")

include("generalized_if/gif.jl")
include("generalized_if/if.jl")
include("generalized_if/adex.jl")
include("generalized_if/if_extended.jl")
# include("generalized_if/if_CANAHP.jl")
# include("adex/adex_multitimescale.jl")

## Multicompartment
abstract type AbstractDendriteIF <: AbstractGeneralizedIF end
include("multicompartment/dendrite.jl")
include("multicompartment/dendneuron_parameter.jl")
include("multicompartment/tripod.jl")
include("multicompartment/ballandstick.jl")
# include("multicompartment/multipod.jl")

## Population constructor
Population(; param, kwargs...) = Population(param; kwargs...)

## Heterogeneous populations
function heterogeneous(
    param::T,
    N::Int;
    kwargs...,
) where {T<:AbstractGeneralizedIFParameter}
    # ξ_het = ones(Float32, N)
    _type = typeof(param)
    het_dict = Dict{Symbol,Vector{Float32}}()
    for fields in fieldnames(_type)
        if haskey(kwargs, fields)
            het_dict[fields] = rand(kwargs[fields], N)
        else
            het_dict[fields] = fill(getfield(param, fields), N)
        end
    end
    # het_dict = het_dict |> dict2ntuple
    return getfield(SNNModels, nameof(_type))(; het_dict..., FT = Vector{Float32})
end


export AbstractDendriteIF,
    AbstractGeneralizedIF,
    AbstractGeneralizedIFParameter,
    Population,
    integrate!,
    plasticity!,
    heterogeneous
