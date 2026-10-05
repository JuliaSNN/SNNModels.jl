"""
    EmptySynapse(; id = randstring(12), param = EmptyParam(), targets = Dict(), records = Dict())

Connection that does nothing: its `forward!`, `update_traces!` and `plasticity!` are no-ops. It
is the default connection list of `sim!` and `train!` (`C = [EmptySynapse()]`) when a network
has no connections, so both `sim!([pop]; duration)` and `train!([pop]; duration)` work.

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
update_traces!(p::EmptySynapse, param::EmptyParam, dt::Float32, T::Time) = nothing
plasticity!(p::EmptySynapse, param::EmptyParam, dt::Float32, T::Time) = nothing

export EmptySynapse
