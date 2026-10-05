# @eval SNNModels begin
"""
    StimulusGroup(; id = randstring(12), name = "StimulusGroup", param::PoissonStimulusParameter,
                  elements::Vector{AbstractStimulus}, targets = Dict(), records = Dict())

A container of stimuli handled as one model component. It is a subtype of
`AbstractStimulusGroup`, not of `AbstractStimulus`: when a model is simulated, `sim!` and
`train!` unpack the group and run each element as an ordinary stimulus. The helper methods
`set_variable!`, `set_intervals!`, `set_active!`, `record` and `stimulate!` broadcast to all
elements.

`param` must be a `PoissonStimulusParameter` (the field type is restricted), which limits
groups to Poisson stimuli. The usual way to build a group is
[`MultiCompartmentStimulusGroup`](@ref).

# Fields
- `id::String`, `name::String = "StimulusGroup"`
- `param::PoissonStimulusParameter`: parameter shared by the elements.
- `elements::Vector{AbstractStimulus}`: the grouped stimuli.
- `targets::Dict`, `records::Dict`
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
    MultiCompartmentStimulusGroup(param::AbstractStimulusParameter, post::AbstractPopulation,
                                  sym::Symbol, comps::Vector{Symbol}; name = "StimulusGroup", kwargs...)

Build a [`StimulusGroup`](@ref) with one stimulus per compartment in `comps`, each created
with `Stimulus(param, post, sym; comp, name, kwargs...)`. All elements share the same
parameter object, so changing it (for instance with `set_variable!`) affects all
compartments.

Because the compartment is passed as the keyword `comp` and the group field `param` is a
`PoissonStimulusParameter`, this works with `PoissonFixed`, `PoissonInterval` and
`PoissonVariable` parameters.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Tripod(N = 10)
group = SNN.MultiCompartmentStimulusGroup(SNN.PoissonFixed(rate = 10Hz), E, :glu, [:d1, :d2])
model = SNN.compose(; E, group)
SNN.sim!(; model, duration = 10ms)
```
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

Return the neuron indices targeted by the stimuli of the group. In SNNModels 1.8.4 the result
is a vector with one index vector per element (`vcat` is applied to a single vector of
vectors, so the lists are not concatenated).
"""
neurons(stim::StimulusGroup) = vcat(map(s -> neurons(s), stim.elements))

export  StimulusGroup, set_variable!, set_intervals!, stimulate!, set_active!, neurons, MultiCompartmentStimulusGroup
# end