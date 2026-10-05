"""
    AggregateScalingParameter(; τ = 10ms, τa, τe, Y, Wmin = 0.5pF, Wmax = 250pF)
    AggregateScalingParameter(N, rate = 10Hz; τ = 10ms, τa = 100ms, τe = 100ms, Wmin = 0.5pF)

Parameters of `AggregateScaling` (homeostatic scaling of the summed input weight towards a
target firing rate).

# Fields
- `τ::Float32 = 10ms`: interval between two rescalings of the weights (ms).
- `τa::Float32`: time constant of the rate estimate `y` (ms); positional default `100ms`.
- `τe::Float32`: time constant of the target total weight `WT` (ms); positional default
  `100ms`.
- `Y::Vector{Float32}`: target firing rate of each postsynaptic neuron, in library units
  (1/ms; write it with the `Hz` unit, e.g. `5Hz`). The positional constructor fills it with
  `rate` (default `10Hz`).
- `Wmin::Float32 = 0.5pF`: offset added to every weight after rescaling. The `pF` unit in the
  code is a numerical factor (1); the weight unit is that of the synapse target.
- `Wmax::Float32 = 250pF`: soft upper bound of `WT`.

The positional form builds the vector `Y` for `N` neurons and does not take `Wmax`.
(Up to SNNModels 1.8.4 its `Wmin` default was `0.05`, against `0.5pF` in the keyword form.)
"""
AggregateScalingParameter

@snn_kw struct AggregateScalingParameter{FT = Float32,VFT = Vector{Float32}} <: NormParam
    τ::FT = 10ms
    τa::FT
    τe::FT
    Y::VFT
    Wmin::FT = 0.5pF
    Wmax::FT = 250pF
end

function AggregateScalingParameter(
    N,
    rate = 10Hz;
    τ = 10ms,
    τa = 100ms,
    τe = 100ms,
    Wmin = 0.5pF,
)
    AggregateScalingParameter(; τ = τ, τa = τa, τe = τe, Y = fill(Float32(rate), N), Wmin = Float32(Wmin))
end

@doc raw"""
    AggregateScaling{VFT, VST} <: AbstractNormalization

Homeostatic aggregate scaling of the excitatory input of each postsynaptic neuron: a rate
estimate ``y_i`` drives a target total weight ``W^T_i``, and the incoming weights are
periodically rescaled so that their sum follows ``W^T_i``.

# Equations
```math
\tau_a \frac{dy_i}{dt} = -y_i + \sum_f \delta(t - t_i^f), \qquad
\tau_e \frac{dW^T_i}{dt} = \left(1 - \frac{W^T_i}{W_{max}}\right)\left(1 - \frac{y_i}{Y_i}\right)
```
``y_i`` is an estimate of the firing rate of neuron ``i`` (1/ms; a spike increases it by
``1/\tau_a``), compared with the target rate ``Y_i``. ``W^T_i`` starts from the summed input
weight at construction.

# Update
- `forward!` (every step of `sim!` and `train!`): forward Euler step of ``y``.
- `plasticity!` (every step of `train!` only): forward Euler step of ``W^T``; every
  `max(1, round(Int, τ / dt))` steps, with ``W^t_i`` the current summed input weight,
```math
\mu_i = \frac{\max(W^T_i - n_i W_{min}, 0)}{W^t_i}, \qquad W_s \leftarrow W_s\,\mu_i + W_{min}
```
  for every synapse ``s`` onto neuron ``i`` (``n_i`` incoming synapses), so that the summed
  weight equals ``W^T_i`` (neurons with no input weight are skipped).

!!! note "Changed after SNNModels 1.8.4"
    Up to 1.8.4 ``y`` and ``W^T`` were updated per step without `dt` (effective time
    constants ``\tau_a\,dt`` and ``\tau_e\,dt`` in ms), ``y`` counted spikes (increment 1)
    while ``Y`` is a rate, and ``W^T`` also evolved under `sim!`. With `dt = 0.125ms` and
    `τa = 100ms` the homeostatic fixed point was at a rate of ``Y / 12.5\,\mathrm{ms}`` instead of
    ``Y``. The rescaling used ``\mu_i = (W^T_i - W_{min})/W^t_i``, so the summed weight became
    ``W^T_i + (n_i - 1) W_{min}`` instead of ``W^T_i``. The `N` field was always 0.

# Fields
- `param::AggregateScalingParameter`; `synapses`: the scaled sparse synapses (same
  postsynaptic population).
- `Wt::Vector{Float32}`: summed input weight at the last rescaling.
- `WT::Vector{Float32}`: target summed input weight.
- `y::Vector{Float32}`: rate estimate (1/ms); `μ::Vector{Float32}`: last scaling factors.
- `fire`: reference to the postsynaptic `fire` vector.
- `N::Int32`: number of postsynaptic neurons.
- `id`, `targets`, `records`.

# References
Reference not given in the code.
"""
AggregateScaling

@snn_kw struct AggregateScaling{
    VFT = Vector{Float32},
    VST = Vector{<:AbstractSparseSynapse},
} <: AbstractNormalization
    N::Int32 = 0
    id::String = randstring(12)
    param::NormParam = MultiplicativeNorm()
    synapses::VST
    Wt::VFT
    WT::VFT
    fire::VBT
    y::VFT = zeros(Float32, N)
    μ::VFT = zeros(Float32, N)
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
    AggregateScaling(N, synapses; param::AggregateScalingParameter, kwargs...)

Build an `AggregateScaling` for the vector `synapses` (sparse synapses with the same
postsynaptic population, asserted). `N` is the number of postsynaptic neurons, either an
`Int` or any object with a field `N` (e.g. the postsynaptic population). `WT` is initialised
with the current summed input weight of each neuron.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 2.0))
AS = SNN.AggregateScaling(E, [EE]; param = SNN.AggregateScalingParameter(E.N, 5Hz))
model = SNN.compose(; E, EE, AS)
SNN.train!(model; duration = 100ms)
```
"""
function AggregateScaling(N, synapses; param::AggregateScalingParameter, kwargs...)
    # Set the target and verify is the same population for all synapses
    targets = Dict()
    posts = [syn.targets[:post] for syn in synapses]
    @assert length(unique(posts)) == 1
    targets[:post] = unique(posts)[1]
    targets[:synapses] = [syn.id for syn in synapses]

    # retrierbr 
    if !isa(N, Int)
        @unpack N = N
    end
    WT = zeros(Float32, N)
    Wt = zeros(Float32, N)
    y = zeros(Float32, N)
    μ = zeros(Float32, N)
    # populate the post-synaptic weight array
    fire = synapses[1].fireI
    for syn in synapses
        @assert isa(syn, AbstractSparseSynapse)
        @unpack rowptr, W, index, fireI = syn
        Is = 1:(length(rowptr)-1)
        @assert length(Is) == N
        for i in eachindex(Is)
            @simd for j ∈ rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron 
                WT[i] += W[index[j]]
            end
        end
    end
    AggregateScaling(; N = Int32(N), @symdict(param, Wt, WT, y, μ, fire, synapses)..., targets, kwargs...)
end



"""
    forward!(c::AggregateScaling, param::AggregateScalingParameter, dt::Float32, T::Time)

Update the rate estimate `y` (see `AggregateScaling`). Runs at every step of `sim!` and
`train!`.
"""
function forward!(c::AggregateScaling, param::AggregateScalingParameter, dt::Float32, T::Time)
    @unpack y, fire = c
    @unpack τa = param
    @inbounds @simd for i in eachindex(fire)
        y[i] += -dt * y[i] / τa + fire[i] / τa
    end
end

"""
    plasticity!(c::AggregateScaling, param::AggregateScalingParameter, dt::Float32, T::Time)

Update the target summed weight `WT` and, when `get_step(T)` is a multiple of
`max(1, round(Int, param.τ / dt))`, rescale the incoming weights (see `AggregateScaling`).
Called only by `train!`.
"""
function plasticity!(
    c::AggregateScaling,
    param::AggregateScalingParameter,
    dt::Float32,
    T::Time,
)
    @unpack y, WT = c
    @unpack Y, τe, Wmax, τ = param
    @inbounds @simd for i in eachindex(WT)
        WT[i] += dt * (1 - WT[i] / Wmax) * (1 - y[i] / Y[i]) / τe
    end
    if get_step(T) % max(1, round(Int, τ / dt)) == 0
        plasticity!(c, param)
    end
end

"""
    plasticity!(c::AggregateScaling, param::AggregateScalingParameter)

Rescale the incoming weights immediately: `W ← W μ + Wmin` with
`μ = max(WT - n Wmin, 0) / Wt` (`n` incoming synapses), so that the summed weight becomes `WT`
(or `n Wmin` if `WT < n Wmin`). Neurons whose summed input weight `Wt` is 0 are left unchanged.
"""
function plasticity!(c::AggregateScaling, param::AggregateScalingParameter)
    @unpack Wt, WT, μ, synapses, y = c
    @unpack τe, Y, Wmin = param
    fill!(Wt, 0.0f0)
    nsyn = zeros(Int, length(Wt))
    for syn in synapses
        @unpack rowptr = syn
        for i = 1:(length(rowptr)-1)
            nsyn[i] += rowptr[i+1] - rowptr[i]
        end
    end
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons of i
                Wt[i] += W[index[j]]
            end
        end
    end
    # normalize

    @inbounds for i in eachindex(μ)
        # neurons without input weight cannot be rescaled
        μ[i] = Wt[i] > 0 ? max(WT[i] - nsyn[i] * Wmin, 0.0f0) / Wt[i] : 1.0f0
    end
    # apply
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron i
                Wt[i] > 0 && (W[index[j]] = W[index[j]] * μ[i] + Wmin)
            end
        end
    end
end

export AggregateScaling, AggregateScalingParameter
