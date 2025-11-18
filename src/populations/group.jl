@snn_kw struct GroupPopulationParameter{
    T<:Type
} <: AbstractPopulationParameter 
    param::T = NoPopulationParameter
end

@snn_kw struct PopulationGroup{
    PT <: Vector,
    GPT <: AbstractPopulationParameter,
} <: AbstractPopulation
    populations::PT = PT[]
    N::Int = length(populations)
    param::GPT = GroupPopulationParameter()
    name::String = "PopulationGroup"
    id::String = randstring(12)
end

function integrate!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:PopulationGroup,T<:AbstractPopulationParameter}
    for pop in p.populations
        integrate!(pop, pop.param, dt)
    end
end

function plasticity!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:PopulationGroup,T<:AbstractPopulationParameter}
    for pop in p.populations
        plasticity!(pop, pop.param, dt)
    end
end

function record!(
    p::P,
    T::Time,
) where {P<:PopulationGroup}
    for pop in p.populations
        record!(pop, T)
    end
end

function PopulationGroup(
    populations::Vector;
    name="PopulationGroup",
)
    [@assert pop isa AbstractPopulation for pop in populations]
    return PopulationGroup(;populations, name=name, 
                param  = GroupPopulationParameter(typeof(populations[1].param)),
                PT= Vector{typeof(populations[1])})
end

monitor!(group::Item, keys::Vector; kwargs...) where Item<:SNNModels.PopulationGroup = [monitor!(pop, keys; kwargs...) for pop in group.populations]

record(p::PopulationGroup, args...; kwargs...) =
    [record(pop, args...; kwargs...) for pop in p.populations]
    



export Group, GroupParameter, PopulationGroup