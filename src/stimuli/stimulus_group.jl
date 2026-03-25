# @eval SNNModels begin
"""
    StimulusGroup

A container to group multiple stimuli together. This is useful for managing and applying operations to a set of related stimuli simultaneously.

An example is the `MultiCompartmentStimulusGroup` function, which creates a `StimulusGroup` to deliver stimuli to multiple compartments of a multi-compartment neuron model.

# Fields
- `id::String`: A unique identifier for the stimulus group.
- `name::String`: A name for the stimulus group.
- `param::PoissonStimulusParameter`: The parameter for the Poisson stimulus. This seems to be a common parameter for the group, though individual elements can have their own variations.
- `elements::Vector{AbstractStimulus}`: A vector of `AbstractStimulus` objects that belong to this group.
- `targets::Dict`: A dictionary specifying the targets of the stimuli in the group.
- `records::Dict`: A dictionary to store recorded data from the stimuli.
"""
StimulusGroup

@snn_kw struct StimulusGroup{ST=Vector{AbstractStimulus}, } <: AbstractStimulusGroup
    id::String = randstring(12)
    name::String = "StimulusGroup"
    param::PoissonStimulusParameter
    elements::ST
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
    MultiCompartmentStimulusGroup(param, post, sym, comps; name="StimulusGroup", kwargs...)

A constructor function that creates a `StimulusGroup` for delivering stimuli to multiple compartments of a population of neurons (e.g., a multi-compartment model).

# Arguments
- `param::AbstractStimulusParameter`: The parameter object for the stimuli to be created.
- `post::AbstractPopulation`: The target population for the stimuli.
- `sym::Symbol`: The symbol representing the synaptic variable to be targeted (e.g., `:ge` for excitatory conductance).
- `comps::Vector{Symbol}`: A vector of symbols representing the names of the compartments to be stimulated.
- `name::String`: An optional name for the `StimulusGroup`.
- `kwargs...`: Additional keyword arguments passed to the `Stimulus` constructor.

# Returns
- `StimulusGroup`: A `StimulusGroup` containing a `Stimulus` for each specified compartment.
"""
function MultiCompartmentStimulusGroup(param::P, 
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

"""
    set_variable!(stim::StimulusGroup, var::Symbol, value)

Sets the value of a variable for all stimuli within a `StimulusGroup`. This is a convenience function to broadcast the operation to all elements of the group.
"""
set_variable!(stim::StimulusGroup, var::Symbol, value) = map(s -> set_variable!(s, var, value), stim.elements)

"""
    set_intervals!(stim::StimulusGroup, intervals)

Sets the activity intervals for all stimuli within a `StimulusGroup`. This is a convenience function to broadcast the operation to all elements of the group.
"""
set_intervals!(stim::StimulusGroup, intervals) = map(s -> set_intervals!(s, intervals), stim.elements)

"""
    record(stim::StimulusGroup, args...)

Applies the `record` function to all stimuli within a `StimulusGroup`. This is a convenience function to broadcast the operation to all elements of the group.
"""
record(stim::StimulusGroup, args...) = map(s -> record(s, args...), stim.elements)

"""
    stimulate!(stim::StimulusGroup, param, time, dt)

Applies the `stimulate!` function to all stimuli within a `StimulusGroup`, delivering the stimulation for the current time step.
"""
stimulate!(stim::StimulusGroup, param::P, time::Time, dt::Float32) where {P<:AbstractStimulusParameter} = map(s -> stimulate!(s, param, time, dt), stim.elements)

"""
    set_active!(stim::StimulusGroup, active::Bool)

Sets the active state for all stimuli within a `StimulusGroup`.
"""
set_active!(stim::StimulusGroup, active::Bool) = map(s -> set_active!(s, active), stim.elements)

"""
    neurons(stim::StimulusGroup)

Returns a concatenated vector of the neuron indices targeted by all stimuli within the `StimulusGroup`.
"""
neurons(stim::StimulusGroup) = vcat(map(s -> neurons(s), stim.elements))

export  StimulusGroup, set_variable!, set_intervals!, stimulate!, set_active!, neurons, MultiCompartmentStimulusGroup
# end