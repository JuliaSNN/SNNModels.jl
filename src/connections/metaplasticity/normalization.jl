"""
    NormParam <: MetaPlasticityParameter

Abstract type of the normalization parameters: `MultiplicativeNorm`, `AdditiveNorm`
(used by `SynapseNormalization`) and `AggregateScalingParameter` (used by
`AggregateScaling`).
"""
abstract type NormParam <: MetaPlasticityParameter end

"""
    MultiplicativeNorm(; τ, operator = *)

Multiplicative synaptic normalization, to be passed to `SynapseNormalization` (or
`MetaPlasticity`). Every `τ`, the incoming weights of each postsynaptic neuron are scaled by a
common factor so that their sum returns to its value at construction time (see
`SynapseNormalization`).

# Fields
- `τ::Float32`: interval between two normalizations (ms). Required, no default.
- `operator::Function = *`: combination operator; do not change.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
norm = SNN.MultiplicativeNorm(τ = 20ms)
```
"""
MultiplicativeNorm

@snn_kw struct MultiplicativeNorm{FT = Float32} <: NormParam
    τ::FT
    operator::Function = *
end

"""
    AdditiveNorm(; τ, operator = +)

Additive synaptic normalization, to be passed to `SynapseNormalization` (or
`MetaPlasticity`). Every `τ`, a common offset is added to the incoming weights of each
postsynaptic neuron (see `SynapseNormalization` for the exact update, which does not
restore the initial sum exactly).

# Fields
- `τ::Float32`: interval between two normalizations (ms). Required, no default.
- `operator::Function = +`: combination operator; do not change.
"""
AdditiveNorm

@snn_kw struct AdditiveNorm{FT = Float32} <: NormParam
    τ::FT
    operator::Function = +
end

@doc raw"""
    SynapseNormalization{VFT, VIT, VST} <: AbstractNormalization

Metaplasticity object that keeps the total excitatory input weight of each postsynaptic
neuron close to its initial value. It acts on one or more sparse synapses (`synapses`) that
share the same postsynaptic population, and is added to the model as a connection (it
transmits nothing: its `forward!` is a no-op).

# Update
At construction ``W^0_i = \sum_{s \in \text{in}(i)} W_s`` (sum over all synapses of all
`synapses` onto neuron ``i``). Under `train!` only, when the step counter is a multiple of
`round(Int, τ / dt)`, with ``W^1_i`` the current sum:
- `MultiplicativeNorm`: ``\mu_i = W^0_i / W^1_i`` and ``W_s \leftarrow W_s\,\mu_i``, which
  restores ``\sum_s W_s = W^0_i`` exactly.
- `AdditiveNorm`: ``\mu_i = (W^0_i - W^1_i)/W^1_i`` and ``W_s \leftarrow W_s + \mu_i``. The
  offset is not divided by the number of inputs, so the sum becomes
  ``W^1_i + n_i (W^0_i - W^1_i)/W^1_i`` (``n_i`` inputs of neuron ``i``), which equals
  ``W^0_i`` only if ``n_i = W^1_i``.

The normalization runs at its position in the connection list, after the `forward!` and
`plasticity!` of the connections listed before it.

# Fields
- `param::NormParam`: `MultiplicativeNorm` or `AdditiveNorm`.
- `synapses::Vector{<:AbstractSparseSynapse}`: normalized connections.
- `W0::Vector{Float32}`: initial summed input weight per postsynaptic neuron.
- `W1::Vector{Float32}`: summed input weight at the last normalization.
- `μ::Vector{Float32}`: last normalization factor/offset per neuron.
- `t::Vector{Int32} = [0, 1]`: unused.
- `id`, `name = "SynapseNormalization"`, `targets` (`:post` id and `:synapses` ids),
  `records`.

# References
Reference not given in the code.
"""
SynapseNormalization

@snn_kw struct SynapseNormalization{
    VFT = Vector{Float32},
    VIT = Vector{Int32},
    VST = Vector{<:AbstractSparseSynapse},
} <: AbstractNormalization
    id::String = randstring(12)
    param::NormParam = MultiplicativeNorm()
    name::String = "SynapseNormalization"
    synapses::VST
    t::VIT = [0, 1]
    W0::VFT = [0.0f0]
    W1::VFT = [0.0f0]
    μ::VFT = [0.0f0]
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
    SynapseNormalization(synapses; param::NormParam, kwargs...)

Build a `SynapseNormalization` acting on the vector `synapses` (all `AbstractSparseSynapse`
with the same postsynaptic population; asserted). Computes `W0` from the current weights.
`kwargs...` are forwarded to the struct (e.g. `name`).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 2.0), LTPParam = SNN.STDPGerstner())
norm = SNN.SynapseNormalization([EE]; param = SNN.MultiplicativeNorm(τ = 20ms))
model = SNN.compose(; E, EE, norm)
SNN.train!(model; duration = 100ms)
```
"""
function SynapseNormalization(synapses; param::NormParam, kwargs...)
    @assert length(synapses) > 0
    EE = synapses[1]
    for syn in synapses
        @assert isa(syn, AbstractSparseSynapse)
        @assert syn.fireI === EE.fireI "Synapses must have the same postsynaptic population"
    end
    N = length(synapses[1].fireI)

    W0 = zeros(Float32, N)
    W1 = zeros(Float32, N)
    μ = zeros(Float32, N)

    # Test that all synapses have the same postsynaptic neurons

    targets = Dict()
    posts = [syn.targets[:post] for syn in synapses]
    @assert length(unique(posts)) == 1
    targets[:post] = unique(posts)[1]
    targets[:synapses] = [syn.id for syn in synapses]
    for syn in synapses
        @assert isa(syn, AbstractSparseSynapse)
        @unpack rowptr, W, index = syn
        Is = 1:(length(rowptr)-1)
        @assert length(Is) == N
        for i in eachindex(Is)
            @simd for j ∈ rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron 
                W0[i] += W[index[j]]
            end
        end
    end
    SynapseNormalization(; @symdict(param, W0, W1, μ, synapses)..., targets, kwargs...)
end

"""
    MetaPlasticity(param::NormParam, synapses; kwargs...)

Same as `SynapseNormalization(synapses; param, kwargs...)`.
"""
function MetaPlasticity(param::NormParam, synapses; kwargs...)
    SynapseNormalization(synapses; param, kwargs...)
end



function forward!(c::SynapseNormalization, param::NormParam) end

"""
    plasticity!(c::SynapseNormalization, param::NormParam, dt::Float32, T::Time)

Apply the normalization (see `SynapseNormalization`) when `get_step(T)` is a multiple of
`round(Int, param.τ / dt)`; otherwise do nothing. Called only by `train!`.
"""
function plasticity!(c::SynapseNormalization, param::NormParam, dt::Float32, T::Time)
    tt = get_step(T)
    @unpack τ = param
    if ((tt) % round(Int, τ / dt)) < dt
        plasticity!(c, param)
    end
end

"""
    plasticity!(c::SynapseNormalization, param::NormParam)

Apply one normalization step immediately: recompute `W1`, the factors/offsets `μ`, and
update the weights of all `c.synapses`.
"""
function plasticity!(c::SynapseNormalization, param::NormParam)
    @unpack W1, W0, μ, synapses = c
    @unpack operator = param
    fill!(W1, 0.0f0)
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons of i
                W1[i] += W[index[j]]
            end
        end
    end
    # normalize
    # @fastmath @inbounds @simd 
    @turbo for i in eachindex(μ)
        μ[i] = (W0[i] - operator(W1[i], 0.0f0)) / W1[i] #operator defines additive or multiplicative norm
    end
    # apply
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron i
                W[index[j]] = operator(W[index[j]], μ[i])
            end
        end
    end
end

export MultiplicativeNorm, AdditiveNorm, SynapseNormalization, NormParam
