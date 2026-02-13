# @eval SNNModels begin
@snn_kw struct StimulusGroup{ST=Vector{AbstractStimulus}, } <: AbstractStimulus
    id::String = randstring(12)
    name::String = "StimulusGroup"
    param::PoissonStimulusParameter
    stimuli::ST
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
    stimuli = Vector{AbstractStimulus}()
    for comp in comps
        push!(stimuli, Stimulus(param, post, sym; comp=comp, name, kwargs...))
    end
    targets = Dict(:pre => :StimulusGroup, :post => post.id, :sym => comps)
    StimulusGroup(;name, param, stimuli, targets)
end

set_variable!(stim::StimulusGroup, var::Symbol, value) = map(s -> set_variable!(s, var, value), stim.stimuli)

set_intervals!(stim::StimulusGroup, intervals) = map(s -> set_intervals!(s, intervals), stim.stimuli)

stimulate!(stim::StimulusGroup, param::P, time::Time, dt::Float32) where {P<:AbstractStimulusParameter} = map(s -> stimulate!(s, param, time, dt), stim.stimuli)

set_active!(stim::StimulusGroup, active::Bool) = map(s -> set_active!(s, active), stim.stimuli)

neurons(stim::StimulusGroup) = vcat(map(s -> neurons(s), stim.stimuli))

export  StimulusGroup, set_variable!, set_intervals!, stimulate!, set_active!, neurons
# end