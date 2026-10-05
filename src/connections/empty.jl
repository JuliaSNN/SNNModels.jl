"""
    EmptySynapse(; id = randstring(12), param = EmptyParam(), targets = Dict(), records = Dict())

Connection that does nothing: its `forward!` is a no-op. It is the default connection list
of `sim!` and `train!` (`C = [EmptySynapse()]`) when a network has no connections.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
SNN.sim!([E], [SNN.EmptySynapse()]; duration = 10ms)
```
"""
EmptySynapse

@snn_kw struct EmptySynapse <: AbstractConnection
    id::String = randstring(12)
    param::EmptyParam = EmptyParam()
    targets::Dict = Dict()
    records::Dict = Dict()
end

function forward!(p::EmptySynapse, param::EmptyParam) end
function forward!(
    p::EmptySynapse,
    param::EmptyParam,
    dt::Float32,
    T::Time,
) 
end

export EmptySynapse
