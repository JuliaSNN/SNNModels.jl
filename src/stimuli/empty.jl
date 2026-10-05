"""
    EmptyStimulus(; param = EmptyParam(), records = Dict())

Placeholder stimulus that does nothing. `sim!` and `train!` use `[EmptyStimulus()]` as the
default stimulus vector when none is given; its `stimulate!` method is a no-op.
"""
EmptyStimulus

@snn_kw struct EmptyStimulus <: AbstractStimulus
    param::EmptyParam = EmptyParam()
    records::Dict = Dict()
end

function stimulate!(p::EmptyStimulus, param::EmptyParam, T::Time, dt::Float32) end

export EmptyStimulus
