# @eval SNNModels begin
@snn_kw struct StimulusGroup{ST=Vector{AbstractStimulus}, } <: AbstractStimulusGroup
    id::String = randstring(12)
    name::String = "StimulusGroup"
    param::PoissonStimulusParameter
    elements::ST
    targets::Dict = Dict()
    records::Dict = Dict()
end

function StimulusGroup(param::P, 
                        post::T,  
                        sym::Symbol, 
                        comps::Vector{Symbol};
                        name = "StimulusGroup",
                        kwargs...
                        ) where {T<: AbstractPopulation, P<:AbstractStimulusParameter}
    elements = Vector{AbstractStimulus}()
    for comp in comps
        push!(elements, Stimulus(param, post, sym; comp=comp, name, kwargs...))
    end
    targets = Dict(:pre => :StimulusGroup, :post => post.id, :sym => comps)
    StimulusGroup(;name, param, elements, targets)
end

set_variable!(stim::StimulusGroup, var::Symbol, value) = map(s -> set_variable!(s, var, value), stim.elements)

set_intervals!(stim::StimulusGroup, intervals) = map(s -> set_intervals!(s, intervals), stim.elements)

record(stim::StimulusGroup, args...) = map(s -> record(s, args...), stim.elements)

stimulate!(stim::StimulusGroup, param::P, time::Time, dt::Float32) where {P<:AbstractStimulusParameter} = map(s -> stimulate!(s, param, time, dt), stim.elements)

set_active!(stim::StimulusGroup, active::Bool) = map(s -> set_active!(s, active), stim.elements)

neurons(stim::StimulusGroup) = vcat(map(s -> neurons(s), stim.elements))

export  StimulusGroup, set_variable!, set_intervals!, stimulate!, set_active!, neurons
# end