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
