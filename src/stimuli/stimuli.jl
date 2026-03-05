"""
    AbstractStimulusParameter <: AbstractParameter
"""
abstract type AbstractStimulusParameter <: AbstractParameter end

include("empty.jl")
include("poisson.jl")
include("poisson_layer.jl")
include("current.jl")
include("timed.jl")
include("balanced.jl")
include("stimulus_group.jl")


function neurons(stim::G) where {G<:AbstractStimulus}
    if hasfield(typeof(stim), :neurons) 
        return stim.neurons
    else
        @warn "Stimulus: $stim does not have a :neurons field."
        return nothing
    end
end

function set_variable!(stim::G, var::Symbol, value) where {G<:AbstractStimulus}
    if hasfield(typeof(stim), :param) && hasfield(typeof(stim.param), :variables)
        @info "Setting variable $var to $value for stimulus $(stim.name)"
        stim.param.variables[var] = value
    elseif  hasfield(typeof(stim), :param) && hasfield(typeof(stim.param), var)
        @info "Setting variable $var to $value for stimulus $(stim.name)"
        getfield(stim.param, var) .= value
    else
        @warn "Stimulus: $(stim.name) (type: $(typeof(stim)) does not have a param with variables $var. Cannot set variable."
    end
end


function set_intervals!(stim::G, intervals) where {G<:AbstractStimulus}
    if hasfield(typeof(stim), :param) && hasfield(typeof(stim.param), :intervals)
        empty!(stim.param.intervals)
        append!(stim.param.intervals, intervals)
    else
        @warn "Stimulus: $(stim.name) (type: $(typeof(stim))) does not have a param with variables containing intervals. Cannot set intervals."
    end
end 

function set_active!(stim::G, active::Bool) where {G<:AbstractStimulus}
    if hasfield(typeof(stim), :param) && hasfield(typeof(stim.param), :active)
        stim.param.active[1] = active
    else
        @warn "Stimulus: $stim does not have a param with active field. Cannot set active state."
    end
end

export set_variable!, set_active!, set_intervals!, neurons, AbstractStimulus, AbstractStimulusParameter, Stimulus, StimulusGroup, stimulate!