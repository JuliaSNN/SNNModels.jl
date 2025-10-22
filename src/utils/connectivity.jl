abstract type AbstractConnectivity end

Connectivity = Union{NamedTuple,AbstractMatrix}

function NormalConnection(μ = 1.0f0, σ = 0.0f0; kwargs...)
    return (; dist = Distribution.Normal, μ = μ, σ = σ, kwargs...)
end

# @snn_kw struct SynapticConn{T} <: AbstractConnectivity
#         ρ::T = 0.1f0  # Connection probability
#         μ::T = 1.0f0
#         σ::T = 0.0f0
#         dist::Symbol = :Normal
#         rule::Symbol = :Fixed
#         γ::T = -1.0f0
#         kmin::T = -1.0f0
# end

# @snn_kw struct MatrixConn{T} <: AbstractConnectivity
#         w::Matrix{T}
# end

# export ConnParam, SynapticConn, MatrixConn
