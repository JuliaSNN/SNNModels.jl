"""
    SpikeTimeStimulusParameter(; spiketimes = [], neurons = [])
    SpikeTimeStimulusParameter(spiketimes::Vector{Float32}, neurons::Vector{Int})

List of input spikes for a [`SpikeTimeStimulus`](@ref): spike `k` is emitted by input
neuron `neurons[k]` at time `spiketimes[k]` (ms).

`SpikeTimeStimulus` walks through the list in order, so `spiketimes` must be sorted in
increasing order. The direct constructors do not sort; prefer [`SpikeTimeParameter`](@ref),
which sorts the spikes by time and converts the times to `Float32`.

# Fields
- `spiketimes::Vector{Float32} = []`: spike times (ms), sorted.
- `neurons::Vector{Int} = []`: index of the input neuron of each spike (same length).

# Related functions
- `SpikeTimeParameter(spiketimes, neurons)`, `SpikeTimeParameter(; spiketimes, neurons)`,
  `SpikeTimeParameter(st::Spiketimes)`: build a sorted parameter.
- `shift_spikes!(param, delay)`: add `delay` to all spike times.
- `max_neuron(param)`: largest input neuron index (0 if empty).
"""
SpikeTimeStimulusParameter

@snn_kw struct SpikeTimeStimulusParameter{VFT = Vector{Float32},VIT = Vector{Int}} <:
               AbstractStimulusParameter
    spiketimes::VFT=[]
    neurons::VIT=[]
end

"""
    SpikeTimeParameter(spiketimes::Vector, neurons::Vector{Int})
    SpikeTimeParameter(; neurons = Int[], spiketimes = Float32[])
    SpikeTimeParameter(st::Spiketimes)
    SpikeTimeParameter(st::Vector{Vector{Float64}})

Build a [`SpikeTimeStimulusParameter`](@ref).

- `SpikeTimeParameter(spiketimes, neurons)`: asserts equal lengths, sorts the pairs by spike
  time and converts the times to `Float32`.
- `SpikeTimeParameter(st::Spiketimes)` (and `Vector{Vector{Float64}}`): flattens one vector of
  spike times per input neuron (`st[i]` are the times of neuron `i`) and sorts by time.
- The keyword form `SpikeTimeParameter(; neurons, spiketimes)` passes the vectors to the
  struct unchanged: it does NOT sort, so the spike times must already be sorted.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
param = SNN.SpikeTimeParameter([30ms, 10ms, 20ms], [1, 2, 3])
param.spiketimes, param.neurons   # (Float32[10, 20, 30], [2, 3, 1])
```
"""
SpikeTimeParameter(; neurons = Int[], spiketimes = Float32[]) =
    SpikeTimeStimulusParameter(spiketimes, neurons)

function SpikeTimeParameter(spiketimes::VFT, neurons::Vector{Int}) where {VFT<:Vector}
    @assert length(spiketimes) == length(neurons) "spiketimes and neurons must have the same length"
    order = sort(1:length(spiketimes), by = x -> spiketimes[x])
    return SpikeTimeStimulusParameter(Float32.(spiketimes[order]), neurons[order])
end


function SpikeTimeParameter(spiketimes::Spiketimes)
    neurons = Int[]
    times = Float32[]
    for i in eachindex(spiketimes)
        for t in spiketimes[i]
            push!(neurons, i)
            push!(times, t)
        end
    end
    order = sort(1:length(times), by = x -> times[x])
    return SpikeTimeStimulusParameter(Float32.(times[order]), neurons[order])
end

function SpikeTimeParameter(spiketimes::Vector{Vector{Float64}})
    _spiketimes = [Float32.(t) for t in spiketimes]
    return SpikeTimeParameter(_spiketimes)
end

## SpikeTimeStimulus
@doc raw"""
    SpikeTimeStimulus(post::AbstractPopulation, sym::Symbol, comp = nothing;
                      conn, param::SpikeTimeStimulusParameter, N = nothing, name = "SpikeTime")
    Stimulus(param::SpikeTimeStimulusParameter, post, sym, comp = nothing; conn, kwargs...)

Deliver a predefined list of spikes from `N` virtual input neurons to `post` through a
sparse weight matrix.

`conn` is a connectivity NamedTuple (e.g. `(p = 0.1, μ = 1.0)`, see `sparse_matrix`) or a
weight matrix of size `post.N x N`. `N` defaults to `max_neuron(param)`, the largest input
neuron index in `param`. For one-to-one input use [`SpikeTimeStimulusIdentity`](@ref).

# Equations
At each step, every spike ``k`` with ``t_k \le t`` that has not been delivered yet (spikes
are consumed in order) makes its input neuron ``j = `` `neurons[k]` fire and increments
its targets ``i``:
```math
g_i \leftarrow g_i + W_{ij}
```
Since the clock is advanced before stimuli are called, a spike at ``t_k`` is delivered in
the first step whose end time is ``\ge t_k``; the timing resolution is `dt`.

# Fields
- `N::Int`: number of input neurons; `name = "SpikeTime"`; `id`
- `param::SpikeTimeStimulusParameter`
- `rowptr`, `colptr`, `I`, `J`, `index`, `W`: sparse connectivity from `dsparse`.
- `g::Vector{Float32}`: target variable of `post` (shared).
- `next_spike::Vector{Float32}`: time of the next spike (`Inf` when exhausted).
- `next_index::Vector{Int}`: index of the next spike in `param` (`-1` when exhausted).
- `fire::Vector{Bool}`: input neurons that fired in the last step (recordable with
  `monitor!(stim, :fire)`).
- `records::Dict`, `targets::Dict`

To reuse the stimulus with new spikes call `update_spikes!` or `shift_spikes!`, which also
rewind `next_index`/`next_spike`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
param = SNN.SpikeTimeParameter([10ms, 20ms, 30ms], [1, 2, 1])
stim = SNN.SpikeTimeStimulus(E, :ge; param, conn = (p = 0.5, μ = 2.0))
model = SNN.compose(; E, stim)
SNN.sim!(; model, duration = 50ms)
```
"""
SpikeTimeStimulus

@snn_kw struct SpikeTimeStimulus{VFT = Vector{Float32}} <: AbstractStimulus
    N::Int
    name::String = "SpikeTime"
    id::String = randstring(12)
    param::SpikeTimeStimulusParameter
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    g::VFT  # rise conductance
    next_spike::VFT = [0]
    next_index::VIT = [0]
    fire::VBT = falses(N)
    records::Dict = Dict()
    targets::Dict = Dict()
end

function SpikeTimeStimulus(
    post::T,
    sym::Symbol,
    comp = nothing;
    conn::Connectivity,
    N = nothing,
    param::SpikeTimeStimulusParameter,
    name::String = "SpikeTime",
) where {T<:AbstractPopulation}

    # set the synaptic weight matrix
    N = isnothing(N) ? max_neuron(param) : N

    w = sparse_matrix(N, post.N, conn)
    rowptr, colptr, I, J, index, W = dsparse(w)

    targets = Dict(:pre => :SpikeTimeStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, comp)

    next_spike = zeros(Float32, 1)
    next_index = zeros(Int, 1)
    next_spike[1] = isempty(param.spiketimes) ? Inf : param.spiketimes[1]
    next_index[1] = isempty(param.spiketimes) ? -1 : 1

    return SpikeTimeStimulus(;
        N = N,
        param = param,
        next_spike = next_spike,
        next_index = next_index,
        g = g,
        targets = targets,
        @symdict(rowptr, colptr, I, J, index, W)...,
        name,
    )
end

"""
    SpikeTimeStimulusIdentity(post::AbstractPopulation, sym::Symbol, comp = nothing;
                              param::SpikeTimeStimulusParameter, kwargs...)

[`SpikeTimeStimulus`](@ref) with one input neuron per neuron of `post` and one-to-one
connections of weight 1: input neuron `j` excites only neuron `j` of `post`, adding 1 to its
target variable per spike. `N = post.N`; the connectivity is a sparse identity matrix
(no dense `N x N` matrix is built). Extra keywords (e.g. `name`) are forwarded.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
stim = SNN.SpikeTimeStimulusIdentity(E, :ge; param = SNN.SpikeTimeParameter([5ms, 10ms], [3, 7]))
```
"""
function SpikeTimeStimulusIdentity(
    post::T,
    sym::Symbol,
    comp = nothing;
    param::SpikeTimeStimulusParameter,
    kwargs...,
) where {T<:AbstractPopulation}
    conn = sparse(1.0f0 * LinearAlgebra.I, post.N, post.N)  # sparse identity, no N x N dense
    return SpikeTimeStimulus(post, sym, comp; conn, N = post.N, param = param, kwargs...)
end

"""
    Stimulus(param::SpikeTimeStimulusParameter, post::AbstractPopulation, sym::Symbol, comp = nothing; kwargs...)

Build a [`SpikeTimeStimulus`](@ref); equivalent to
`SpikeTimeStimulus(post, sym, comp; param, kwargs...)` (the keyword `conn` is required).
"""
function Stimulus(
    param::SpikeTimeStimulusParameter,
    post::T,
    sym::Symbol,
    comp = nothing;
    kwargs...,
) where {T<:AbstractPopulation}
    return SpikeTimeStimulus(post, sym, comp; param, kwargs...)
end

"""
    stimulate!(s::SpikeTimeStimulus, param::SpikeTimeStimulusParameter, time::Time, dt::Float32)

Clear `s.fire`, then deliver every pending spike with time `<= get_time(time)`: mark its
input neuron in `s.fire`, add the weights of its outgoing connections to `s.g`, and advance
`next_index`/`next_spike` (set to `-1`/`Inf` after the last spike). `dt` is unused.
"""
function stimulate!(
    s::SpikeTimeStimulus,
    param::SpikeTimeStimulusParameter,
    time::Time,
    dt::Float32,
)
    @unpack colptr, I, W, fire, g, next_spike, next_index = s
    @unpack spiketimes, neurons = param
    fill!(fire, false)
    while next_spike[1] <= get_time(time)
        j = neurons[next_index[1]] # loop on presynaptic neurons
        fire[j] = true
        @inbounds @simd for s ∈ colptr[j]:(colptr[j+1]-1)
            g[I[s]] += W[s]
        end
        if next_index[1] < length(spiketimes)
            next_index[1] += 1
            next_spike[1] = spiketimes[next_index[1]]
        else
            next_spike[1] = Inf
            next_index[1] = -1
        end
    end
end
"""
    next_neuron(p::SpikeTimeStimulus)

Return the input neuron of the next pending spike, `param.neurons[next_index]`; return `[]`
once all spikes have been delivered (`next_index == -1`). (Up to SNNModels 1.8.4 it returned
`[]` while the last spike was still pending and threw a `BoundsError` after the last spike.)
"""
function next_neuron(p::SpikeTimeStimulus)
    @unpack next_spike, next_index, param = p
    if 1 <= next_index[1] <= length(param.spiketimes)
        return param.neurons[next_index[1]]
    else
        return []
    end
end

"""
    shift_spikes!(spiketimes::Vector{Float32}, delay::Number)

Add `Float32(delay)` (ms, may be negative) to every element of `spiketimes`, in place.
"""
function shift_spikes!(param::Vector{Float32}, delay::Number)
    @. param += Float32(delay)
end

"""
    shift_spikes!(param::SpikeTimeStimulusParameter, delay::Number)

Add `delay` (ms) to all spike times of `param`, in place. The state of a stimulus that uses
`param` is not rewound; use `shift_spikes!(stimulus, delay)` for that.
"""
function shift_spikes!(param::SpikeTimeStimulusParameter, delay::Number)
    shift_spikes!(param.spiketimes, delay)
end

"""
    shift_spikes!(stimulus::SpikeTimeStimulus, delay::Number)

Add `delay` (ms) to all spike times of `stimulus.param` and rewind the stimulus to the first
spike (`next_index = 1`, `next_spike = spiketimes[1]`). Spikes shifted to times earlier than
the current simulation time are all delivered at the next step. With an empty spike list the
stimulus is marked as exhausted.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
stim = SNN.SpikeTimeStimulusIdentity(E, :ge; param = SNN.SpikeTimeParameter([5ms, 10ms], [3, 7]))
SNN.shift_spikes!(stim, 100ms)
stim.param.spiketimes  # Float32[105, 110]
```
"""
function shift_spikes!(stimulus::SpikeTimeStimulus, delay::Number)
    shift_spikes!(stimulus.param.spiketimes, delay)
    _rewind!(stimulus)
end

# Point the stimulus at its first spike (or mark it as exhausted if there is none).
function _rewind!(stim)
    if isempty(stim.param.spiketimes)
        stim.next_index[1] = -1
        stim.next_spike[1] = Inf
    else
        stim.next_index[1] = 1
        stim.next_spike[1] = stim.param.spiketimes[1]
    end
    return stim
end

"""
    update_spikes!(stim, spikes, start_time = 0.0f0)

Replace the spike list of a [`SpikeTimeStimulus`](@ref) with `spikes` and rewind it.

`spikes` must have fields `spiketimes` and `neurons` (e.g. a `SpikeTimeStimulusParameter`).
The new times are `spikes.spiketimes .+ start_time`, sorted by time (with their neurons).
`stim.next_index` is set to 1 and `stim.next_spike` to the first new spike time (an empty list
marks the stimulus as exhausted). (Up to SNNModels 1.8.4 the spikes were not sorted and an
empty list raised a `BoundsError`.) The input-neuron count `stim.N` and the weight matrix are
not changed, so the new neuron indices must be `<= stim.N`. Returns `stim`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
stim = SNN.SpikeTimeStimulusIdentity(E, :ge; param = SNN.SpikeTimeParameter([5ms], [1]))
SNN.update_spikes!(stim, SNN.SpikeTimeParameter([1ms, 2ms], [4, 5]), 1000ms)
```
"""
function update_spikes!(stim, spikes, start_time = 0.0f0)
    empty!(stim.param.spiketimes)
    empty!(stim.param.neurons)
    order = sortperm(spikes.spiketimes)
    append!(stim.param.spiketimes, spikes.spiketimes[order] .+ start_time)
    append!(stim.param.neurons, spikes.neurons[order])
    _rewind!(stim)
    return stim
end

"""
    max_neuron(param::SpikeTimeStimulusParameter)

Largest input neuron index in `param.neurons`, or `0` if there are no spikes. Used as the
default number of input neurons `N` of a `SpikeTimeStimulus`.
"""
max_neuron(param::SpikeTimeStimulusParameter) =
    isempty(param.neurons) ? 0 : maximum(param.neurons)



export SpikeTimeStimulusParameter,
    SpikeTimeStimulus,
    SpikeTimeStimulusIdentity,
    SpikeTimeParameter,
    stimulate!,
    next_neuron,
    max_neuron,
    shift_spikes!,
    update_spikes!
