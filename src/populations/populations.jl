abstract type AbstractGeneralizedIFParameter <: AbstractPopulationParameter end
abstract type AbstractGeneralizedIF <: AbstractPopulation end

integrate!(p::AbstractPopulation, param::AbstractPopulationParameter, dt::Float32) = nothing
plasticity!(
    p::AbstractPopulation,
    param::AbstractPopulationParameter,
    dt::Float32,
    T::Time,
) = nothing

include("synapse.jl")
include("synapse_parameters.jl")

## Neurons
include("poisson.jl")
include("iz.jl")
include("hh.jl")
include("morrislecar.jl")
include("rate.jl")
include("identity.jl")

## IF
abstract type AbstractIFParameter <: AbstractGeneralizedIFParameter end
include("if/if.jl")
include("if/if_CANAHP.jl")
include("if/if_extended.jl")

## AdEx
abstract type AbstractAdExParameter <: AbstractGeneralizedIFParameter end
include("adex/adex_parameter.jl")
include("adex/adex.jl")
include("adex/adex_multitimescale.jl")

## Multicompartment
abstract type AbstractDendriteIF <: AbstractGeneralizedIF end
include("multicompartment/dendrite.jl")
include("multicompartment/dendneuron_parameter.jl")
include("multicompartment/tripod.jl")
include("multicompartment/ballandstick.jl")
# include("multicompartment/multipod.jl")


export AbstractDendriteIF, AbstractGeneralizedIF, AbstractGeneralizedIFParameter
