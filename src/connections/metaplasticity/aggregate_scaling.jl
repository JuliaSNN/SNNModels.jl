"""
    AggregateScalingParameter(; τ = 10ms, τa, τe, Y, Wmin = 0.5pF, Wmax = 250pF)
    AggregateScalingParameter(N, rate = 10Hz; τ = 10ms, τa = 100ms, τe = 100ms, Wmin = 0.05)

Parameters of `AggregateScaling` (homeostatic scaling of the summed input weight towards a
target activity).

# Fields
- `τ::Float32 = 10ms`: interval between two rescalings of the weights (ms).
- `τa::Float32`: decay constant of the activity trace `y`, applied per step (see
  `AggregateScaling`); positional constructor default `100ms`.
- `τe::Float32`: time constant of the target total weight `WT`, applied per step; positional
  constructor default `100ms`.
- `Y::Vector{Float32}`: target value of the activity trace, one per postsynaptic neuron;
  the positional constructor fills it with `rate` (default `10Hz`, i.e. `0.01` in the
  library units of 1/ms).
- `Wmin::Float32 = 0.5pF` (keyword constructor) or `0.05` (positional constructor): offset
  added to every weight after rescaling. The `pF` unit in the code is a numerical factor
  (1); the weight unit is that of the synapse target.
- `Wmax::Float32 = 250pF`: soft upper bound of `WT`.

The positional form builds the vector `Y` for `N` neurons and does not take `Wmax`.
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
    Wmin = 0.05,
)
    AggregateScalingParameter(; τ = τ, τa = τa, τe = τe, Y = fill(Float32(rate), N), Wmin = Float32(Wmin))
end

@doc raw"""
    AggregateScaling{VFT, VST} <: AbstractNormalization

Homeostatic aggregate scaling of the excitatory input of each postsynaptic neuron: an
activity trace ``y_i`` drives a target total weight ``W^T_i``, and the incoming weights are
periodically rescaled so that their sum follows ``W^T_i``.

# Update
`forward!` (called at every step by both `sim!` and `train!`), for every postsynaptic neuron
``i`` (per step, without `dt`):
```math
y_i \leftarrow y_i - \frac{y_i}{\tau_a} + \delta_i, \qquad
W^T_i \leftarrow W^T_i + \frac{1}{\tau_e}\left(1 - \frac{W^T_i}{W_{max}}\right)
\left(1 - \frac{y_i}{Y_i}\right)
```
with ``\delta_i = 1`` if neuron ``i`` fired in this step. ``W^T_i`` starts from the summed
input weight at construction.

`plasticity!` (only under `train!`), every `round(Int, τ / dt)` steps, with ``W^t_i`` the
current summed input weight:
```math
\mu_i = \frac{W^T_i - W_{min}}{W^t_i}, \qquad W_s \leftarrow W_s\,\mu_i + W_{min}
```
for every synapse ``s`` onto neuron ``i``.

Note that ``\tau_a`` and ``\tau_e`` are used per integration step (their effective time
constants are ``\tau_a\,dt`` and ``\tau_e\,dt`` in ms), that ``y`` counts spikes while ``Y`` is
given as a rate, and that `WT` evolves also under `sim!`.

# Fields
- `param::AggregateScalingParameter`; `synapses`: the scaled sparse synapses (same
  postsynaptic population).
- `Wt::Vector{Float32}`: summed input weight at the last rescaling.
- `WT::Vector{Float32}`: target summed input weight.
- `y::Vector{Float32}`: activity trace; `μ::Vector{Float32}`: last scaling factors.
- `fire`: reference to the postsynaptic `fire` vector.
- `N::Int32 = 0`: not set by the constructor (stays 0).
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
    AggregateScaling(; @symdict(param, Wt, WT, y, μ, fire, synapses)..., targets, kwargs...)
end



"""
    forward!(c::AggregateScaling, param::AggregateScalingParameter)

Update the activity trace `y` and the target summed weight `WT` (see `AggregateScaling`).
Runs at every step of both `sim!` and `train!`.
"""
function forward!(c::AggregateScaling, param::AggregateScalingParameter)
    @unpack y, fire, WT = c
    @unpack Y, τa, τe, Wmax = param

    @inbounds @simd for i in eachindex(fire)
        y[i] -= y[i]/τa
    end
    @inbounds @simd for i in eachindex(fire)
        fire[i] && (y[i] += 1)
    end
    @inbounds @simd for i in eachindex(fire)
        WT[i] += (1-WT[i]/Wmax)*(1 - y[i]/Y[i])/τe
    end


end

"""
    plasticity!(c::AggregateScaling, param::AggregateScalingParameter, dt::Float32, T::Time)

Rescale the incoming weights (see `AggregateScaling`) when `get_step(T)` is a multiple of
`round(Int, param.τ / dt)`. Called only by `train!`.
"""
function plasticity!(
    c::AggregateScaling,
    param::AggregateScalingParameter,
    dt::Float32,
    T::Time,
)
    tt = get_step(T)
    @unpack τ = param
    if ((tt) % round(Int, τ / dt)) < dt
        plasticity!(c, param)
    end

end

"""
    plasticity!(c::AggregateScaling, param::AggregateScalingParameter)

Rescale the incoming weights immediately: `W ← W μ + Wmin` with `μ = (WT - Wmin) / Wt`.
"""
function plasticity!(c::AggregateScaling, param::AggregateScalingParameter)
    @unpack Wt, WT, μ, synapses, y = c
    @unpack τe, Y, Wmin = param
    fill!(Wt, 0.0f0)
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons of i
                Wt[i] += W[index[j]]
            end
        end
    end
    # normalize

    @turbo for i in eachindex(μ)
        μ[i] = (WT[i] - Wmin) / Wt[i] #operator defines additive or multiplicative norm
    end
    # apply
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron i
                W[index[j]] = W[index[j]] * μ[i] + Wmin
            end
        end
    end
end

export AggregateScaling, AggregateScalingParameter
