
"""
    AbstractStimulus

An abstract type representing a stimulus. Any struct inheriting from this type must implement:

# Methods
- `stimulate!(p::Stimulus, param::StimulusParameter, time::Time, dt::Float32)`: Applies the stimulus to the population.
"""
abstract type AbstractStimulus <: AbstractComponent end

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

export Stimulus, stimulate!
