abstract type AbstractGeneralizedIFParameter <: AbstractPopulationParameter end
abstract type AbstractGeneralizedIF <: AbstractPopulation end

integrate!(p::AbstractPopulation, param::AbstractPopulationParameter, dt::Float32) = nothing
plasticity!(
    p::AbstractPopulation,
    param::AbstractPopulationParameter,
    dt::Float32,
    T::Time,
) = nothing

include("synapse/synapse.jl")
include("synapse/synapse_parameters.jl")

## Neurons
include("poisson.jl")
include("iz.jl")
include("hh.jl")
include("morrislecar.jl")
include("rate.jl")
include("identity.jl")

## IF

include("generalized_if/gif.jl")
include("generalized_if/synapses.jl")
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


export AbstractDendriteIF, AbstractGeneralizedIF, AbstractGeneralizedIFParameter
