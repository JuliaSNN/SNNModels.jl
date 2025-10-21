abstract type AbstractGeneralizedIFParameter <: AbstractPopulationParameter end
abstract type AbstractGeneralizedIF <: AbstractPopulation end

integrate!(p::AbstractPopulation, param::AbstractPopulationParameter, dt::Float32) = nothing
plasticity!(
    p::AbstractPopulation,
    param::AbstractPopulationParameter,
    dt::Float32,
    T::Time,
) = nothing

function heterogeneous(param::T, N::Int; kwargs...) where {T<:AbstractGeneralizedIFParameter}
    # ξ_het = ones(Float32, N)
    _type = typeof(param)
    het_dict = Dict{Symbol, Vector{Float32}}()
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

include("synapse/synapse.jl")
include("synapse/synapses.jl")
include("synapse/synapse_parameters.jl")
include("synapse/synaptic_targets.jl")

## Neurons
include("poisson.jl")
include("iz.jl")
include("hh.jl")
include("morrislecar.jl")
include("rate.jl")
include("identity.jl")

## IF

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

Population(; param, kwargs...) = Population(param; kwargs...)
export AbstractDendriteIF, AbstractGeneralizedIF, AbstractGeneralizedIFParameter, Population, integrate!, plasticity!, heterogeneous
