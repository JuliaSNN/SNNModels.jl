"""
    AbstractStimulusParameter <: AbstractParameter

Supertype of all stimulus parameter types. The parameter object stored in the `param`
field of a stimulus selects, by dispatch, the `stimulate!` method that is called once per
time step by `sim!` and `train!`, and the generic [`Stimulus`](@ref) constructor that builds
the stimulus.

Loaded subtypes: `PoissonStimulusParameter` (`PoissonFixed`, `PoissonInterval`,
`PoissonVariable`), `PoissonLayerParameter` (`PoissonLayer`, `PoissonLayerHet`),
`CurrentParameter` (`CurrentNoise`), `SpikeTimeStimulusParameter`, `BalancedParameter`.
"""
abstract type AbstractStimulusParameter <: AbstractParameter end

include("empty.jl")
include("poisson.jl")
include("poisson_layer.jl")
include("current.jl")
include("timed.jl")
include("balanced.jl")
include("stimulus_group.jl")

"""
    stimulate!(stim, param, time::Time, dt::Float32)

Advance the stimulus `stim` by one time step. Called by `sim!` and `train!` for every
stimulus of the model, after the clock has been advanced (`update_time!`) and before the
populations are integrated. The method is selected by the type of `param` (normally
`stim.param`). Each stimulus type documents what its `stimulate!` method writes into the
target population (conductance increments, current values).
"""
stimulate!

"""
    neurons(stim::AbstractStimulus)

Return the indices of the postsynaptic neurons targeted by `stim` (its `neurons` field).
Stimuli without a `neurons` field (`PoissonStimulusLayer`, `SpikeTimeStimulus`,
`BalancedStimulus`) return `nothing` with a warning; for those the targets are defined by
the connectivity matrix or by the whole population.
"""
function neurons(stim::G) where {G<:AbstractStimulus}
    if hasfield(typeof(stim), :neurons) 
        return stim.neurons
    else
        @warn "Stimulus: $(typeof(stim)) does not have a :neurons field."
        return nothing
    end
end

"""
    set_variable!(stim::AbstractStimulus, var::Symbol, value)

Change a parameter of a stimulus at runtime.

- If `stim.param` has a `variables` dictionary (e.g. `PoissonVariable`), sets
  `stim.param.variables[var] = value`.
- Otherwise, if `stim.param` has a field `var`: an array field (e.g. `active`, `I_base`,
  `rates`) is updated in place (`getfield(stim.param, var) .= value`); a scalar field of a
  mutable parameter (`PoissonFixed.rate`, `PoissonInterval.rate`, `PoissonLayer.rate`) is
  replaced (`setproperty!`); a scalar field of an immutable parameter raises an
  `ArgumentError`. (Up to SNNModels 1.8.4 the scalar case threw a broadcast error; the
  Poisson parameter types were immutable.)
- Otherwise a warning is emitted and nothing changes.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
f(t, v) = v[:r]
stim = SNN.Stimulus(SNN.PoissonVariable(variables = Dict{Symbol,Any}(:r => 5Hz), rate = f), E, :ge)
SNN.set_variable!(stim, :r, 20Hz)
```
"""
function set_variable!(stim::G, var::Symbol, value) where {G<:AbstractStimulus}
    if hasfield(typeof(stim), :param) && hasfield(typeof(stim.param), :variables)
        @info "Setting variable $var to $value for stimulus $(stim.name)"
        stim.param.variables[var] = value
    elseif  hasfield(typeof(stim), :param) && hasfield(typeof(stim.param), var)
        @info "Setting variable $var to $value for stimulus $(stim.name)"
        field = getfield(stim.param, var)
        if field isa AbstractArray
            field .= value
        elseif ismutable(stim.param)
            setproperty!(stim.param, var, value)
        else
            throw(ArgumentError("the field $var of $(typeof(stim.param)) is a scalar of an immutable parameter and cannot be changed in place"))
        end
    else
        @warn "Stimulus: $(stim.name) (type: $(typeof(stim)) does not have a param with variables $var. Cannot set variable."
    end
end


"""
    set_intervals!(stim::AbstractStimulus, intervals)

Replace the activity intervals of a stimulus whose parameter has an `intervals` field
(`PoissonInterval`): the vector is emptied and `intervals` (a vector of `[start, end]`
vectors, in ms) is appended. Warns and does nothing otherwise.
"""
function set_intervals!(stim::G, intervals) where {G<:AbstractStimulus}
    if hasfield(typeof(stim), :param) && hasfield(typeof(stim.param), :intervals)
        empty!(stim.param.intervals)
        append!(stim.param.intervals, intervals)
    else
        @warn "Stimulus: $(stim.name) (type: $(typeof(stim))) does not have a param with variables containing intervals. Cannot set intervals."
    end
end 

"""
    set_active!(stim::AbstractStimulus, active::Bool)

Switch a stimulus on or off by writing `stim.param.active[1] = active`. Warns and does
nothing if the parameter has no `active` field.

Only the `stimulate!` methods of `PoissonStimulus` (all `PoissonStimulusParameter`s) read
the flag. `PoissonLayer` and `PoissonLayerHet` have an `active` field but their
`stimulate!` ignores it (SNNModels 1.8.4), so `set_active!(stim, false)` does not silence a
`PoissonStimulusLayer`.
"""
function set_active!(stim::G, active::Bool) where {G<:AbstractStimulus}
    if hasfield(typeof(stim), :param) && hasfield(typeof(stim.param), :active)
        stim.param.active[1] = active
    else
        @warn "Stimulus: $stim does not have a param with active field. Cannot set active state."
    end
end

export set_variable!, set_active!, set_intervals!, neurons, AbstractStimulus, AbstractStimulusParameter, Stimulus, StimulusGroup, stimulate!