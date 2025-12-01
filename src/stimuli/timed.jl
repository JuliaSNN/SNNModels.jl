"""
    SpikeTimeStimulusParameter{VFT, VIT} <: AbstractStimulusParameter

A parameter structure for spike time stimulus in spiking neural networks. Users are encouraged to create instances of this struct using the provided constructors rather than directly instantiating it. This will ensure that the spike times and neuron indices are properly sorted and with the correct types.


# Fields
- `spiketimes::VFT`: Vector of spike times (default: `Float32[]`)
- `neurons::VIT`: Vector of neuron indices corresponding to each spike time (default: `Int[]`)

# Constructors
- `SpikeTimeStimulusParameter(spiketimes, neurons)`: Creates a parameter structure with given spike times and neuron indices.
- `SpikeTimeStimulusParameter()`: Creates an empty parameter structure with default empty vectors.
- `SpikeTimeParameter(spiketimes, neurons)`: Alternative constructor that sorts spike times and neuron indices by spike time.
- `SpikeTimeParameter(;spiketimes, neurons)`: Keyword argument constructor that sorts spike times and neuron indices by spike time.
- `SpikeTimeParameter(spiketimes::Spiketimes)`: Converts a `Spiketimes` object to a `SpikeTimeStimulusParameter` by flattening the spike times and neuron indices.

# Notes
- The spike times and neuron indices are automatically sorted by spike time in the constructors that accept them.
- The `Spiketimes` type is expected to be a collection of spike times for each neuron.
"""
SpikeTimeStimulusParameter

@snn_kw struct SpikeTimeStimulusParameter{VFT = Vector{Float32},VIT = Vector{Int}} <:
               AbstractStimulusParameter
    spiketimes::VFT=[]
    neurons::VIT=[]
end

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
"""
    SpikeTimeStimulus{FT, VFT, VBT, DT, VIT} <: AbstractStimulus

A spike time stimulus structure for spiking neural networks. This stimulus type delivers spikes to postsynaptic neurons at specified times.

# Fields
- `N::Int`: Number of presynaptic neurons
- `name::String`: Name of the stimulus (default: "SpikeTime")
- `id::String`: Unique identifier for the stimulus (default: random 12-character string)
- `param::SpikeTimeStimulusParameter`: Parameter structure containing spike times and neuron indices
- `rowptr::VIT`: Row pointer of sparse weight matrix
- `colptr::VIT`: Column pointer of sparse weight matrix
- `I::VIT`: Postsynaptic indices of weight matrix
- `J::VIT`: Presynaptic indices of weight matrix
- `index::VIT`: Index mapping for weight matrix
- `W::VFT`: Synaptic weights
- `g::VFT`: Rise conductance
- `next_spike::VFT`: Next spike time (default: [0])
- `next_index::VIT`: Index of next spike (default: [0])
- `fire::VBT`: Boolean vector indicating which neurons fired (default: falses(N))
- `records::Dict`: Dictionary for recording data
- `targets::Dict`: Dictionary specifying stimulus targets

# Constructors
- `SpikeTimeStimulus(post::AbstractPopulation, sym::Symbol, target; kwargs...)`: Creates a spike time stimulus with specified parameters.
- `SpikeTimeStimulusIdentity(post::AbstractPopulation, sym::Symbol, target; kwargs...)`: Creates an identity spike time stimulus where each presynaptic neuron connects to a corresponding postsynaptic neuron.

# Keyword Arguments
- `p::Real`: Connection probability (default: 0.05)
- `μ`: Mean of synaptic weight distribution (default: 1.0)
- `σ`: Standard deviation of synaptic weight distribution (default: 0.0)
- `w`: Synaptic weight matrix (default: generated based on distribution parameters)
- `dist::Symbol`: Distribution type for synaptic weights (default: :Normal)
- `rule::Symbol`: Connection rule (default: :Fixed)
- `N::Int`: Number of presynaptic neurons (default: determined from parameters)
- `param::SpikeTimeStimulusParameter`: Spike time parameters (required)

# Notes
- The stimulus delivers spikes to postsynaptic neurons at times specified in the `param` field.
- The synaptic weight matrix can be specified directly or generated based on distribution parameters.
- The `SpikeTimeStimulusIdentity` constructor creates a 1-to-1 connection between presynaptic and postsynaptic neurons.
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
    SpikeTimeStimulusIdentity(post::AbstractPopulation, sym::Symbol, comp::AbstractCompartment; param::SpikeTimeStimulusParameter, kwargs...)

Create an identity spike time stimulus where each presynaptic neuron connects to a corresponding postsynaptic neuron.

This constructor creates a 1-to-1 connection between presynaptic and postsynaptic neurons, with each neuron in the presynaptic population connecting to the same neuron in the postsynaptic population. The synaptic weight matrix is set to an identity matrix.

# Arguments
- `post::AbstractPopulation`: The postsynaptic population to which the stimulus will be applied
- `sym::Symbol`: The symbol representing the synaptic connection
- `target`: The target of the stimulus (optional)
- `param::SpikeTimeStimulusParameter`: The spike time parameters for the stimulus
- `kwargs...`: Additional keyword arguments to pass to the `SpikeTimeStimulus` constructor

# Returns
- `SpikeTimeStimulus`: A spike time stimulus with identity connections

# Notes
- The number of presynaptic neurons (N) is set to the size of the postsynaptic population (post.N)
- The synaptic weight matrix is set to an identity matrix (LinearAlgebra.I(post.N))
- This is useful for creating direct connections between corresponding neurons in presynaptic and postsynaptic populations
"""
function SpikeTimeStimulusIdentity(
    post::T,
    sym::Symbol,
    comp = nothing;
    param::SpikeTimeStimulusParameter,
    kwargs...,
) where {T<:AbstractPopulation}
    conn = LinearAlgebra.I(post.N) |> Matrix
    return SpikeTimeStimulus(post, sym, comp; conn, N = post.N, param = param, kwargs...)
end

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

Update the synaptic conductances of a `SpikeTimeStimulus` based on the current simulation time.

This function processes all spikes that have occurred up to the current simulation time, updating the synaptic conductances of the postsynaptic neurons accordingly. It also advances the spike index to the next unprocessed spike.

# Arguments
- `s::SpikeTimeStimulus`: The spike time stimulus to update
- `param::SpikeTimeStimulusParameter`: The spike time parameters containing spike times and neuron indices
- `time::Time`: The current simulation time
- `dt::Float32`: The time step of the simulation (unused in this function)

# Details
- The function first resets the `fire` vector to indicate no neurons have fired
- It then processes all spikes that have occurred up to the current time:
  - For each spike, it marks the presynaptic neuron as fired
  - It updates the synaptic conductances of all postsynaptic neurons connected to the firing neuron
  - It advances to the next spike in the spike train
- If there are no more spikes, the next spike time is set to infinity and the next index is set to -1

# Notes
- The function modifies the `fire`, `g`, `next_spike`, and `next_index` fields of the stimulus
- The synaptic conductances are updated by adding the synaptic weights of the connected neurons
- This function is typically called during each time step of a simulation
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

Get the index of the next neuron that will fire in the spike time stimulus.

This function returns the index of the presynaptic neuron that will fire next in the stimulus. If there are no more spikes to process, it returns an empty array.

# Arguments
- `p::SpikeTimeStimulus`: The spike time stimulus to query

# Returns
- The index of the next neuron to fire, or an empty array if there are no more spikes

# Notes
- This function does not modify the stimulus state
- The returned neuron index corresponds to the presynaptic neuron in the stimulus
"""
function next_neuron(p::SpikeTimeStimulus)
    @unpack next_spike, next_index, param = p
    if next_index[1] < length(param.spiketimes)
        return param.neurons[next_index[1]]
    else
        return []
    end
end

"""
    shift_spikes!(param::Vector{Float32}, delay::Number)

Shift all spike times in a vector by a specified delay.

This function modifies the spike times in the vector by adding the specified delay to each time. This effectively shifts all spikes forward or backward in time.

# Arguments
- `param::Vector{Float32}`: Vector of spike times to be shifted
- `delay::Number`: The amount of time to shift each spike (can be positive or negative)

# Notes
- The function modifies the input vector in-place
- The delay is converted to Float32 before being added to the spike times
"""
function shift_spikes!(param::Vector{Float32}, delay::Number)
    @. param += Float32(delay)
end

"""
    shift_spikes!(param::SpikeTimeStimulusParameter, delay::Number)

Shift all spike times in a SpikeTimeStimulusParameter by a specified delay.

This function modifies the spike times in the parameter structure by adding the specified delay to each time. This effectively shifts all spikes forward or backward in time.

# Arguments
- `param::SpikeTimeStimulusParameter`: The parameter structure containing spike times to be shifted
- `delay::Number`: The amount of time to shift each spike (can be positive or negative)

# Notes
- The function modifies the spike times in the parameter structure in-place
- The delay is converted to Float32 before being added to the spike times
"""
function shift_spikes!(param::SpikeTimeStimulusParameter, delay::Number)
    shift_spikes!(param.spiketimes, delay)
end

"""
    shift_spikes!(stimulus::SpikeTimeStimulus, delay::Number)

Shift all spike times in a SpikeTimeStimulus by a specified delay.

This function modifies the spike times in the stimulus by adding the specified delay to each time. It also resets the stimulus to start processing spikes from the beginning of the shifted spike train.

# Arguments
- `stimulus::SpikeTimeStimulus`: The stimulus containing spike times to be shifted
- `delay::Number`: The amount of time to shift each spike (can be positive or negative)

# Notes
- The function modifies the spike times in the stimulus in-place
- The stimulus is reset to start processing spikes from the first spike in the shifted spike train
- The delay is converted to Float32 before being added to the spike times
"""
function shift_spikes!(stimulus::SpikeTimeStimulus, delay::Number)
    shift_spikes!(stimulus.param.spiketimes, delay)
    stimulus.next_index[1] = 1
    stimulus.next_spike[1] = stimulus.param.spiketimes[1]
end

"""
    update_spikes!(stim, spikes, start_time = 0.0f0)

Update the spike times in a stimulus with new spike data.

This function replaces the existing spike times and neuron indices in the stimulus with new data from the provided spikes object. The new spike times are offset by the specified start time.

# Arguments
- `stim`: The stimulus to update (must have a `param` field with `spiketimes` and `neurons` vectors)
- `spikes`: An object containing new spike times and neuron indices (must have `spiketimes` and `neurons` fields)
- `start_time::Float32`: The time offset to apply to the new spike times (default: 0.0f0)

# Returns
- The updated stimulus

# Notes
- The function clears the existing spike data and replaces it with the new data
- The stimulus is reset to start processing spikes from the first spike in the new spike train
- The new spike times are offset by the specified start time
"""
function update_spikes!(stim, spikes, start_time = 0.0f0)
    empty!(stim.param.spiketimes)
    empty!(stim.param.neurons)
    append!(stim.param.spiketimes, spikes.spiketimes .+ start_time)
    append!(stim.param.neurons, spikes.neurons)
    stim.next_index[1] = 1
    stim.next_spike[1] = stim.param.spiketimes[1]
    return stim
end

"""
    max_neuron(param::SpikeTimeStimulusParameter)

Get the maximum neuron index in a SpikeTimeStimulusParameter.

This function returns the highest neuron index present in the parameter structure. If there are no neurons, it returns 0.

# Arguments
- `param::SpikeTimeStimulusParameter`: The parameter structure containing neuron indices

# Returns
- The maximum neuron index, or 0 if there are no neurons

# Notes
- This function does not modify the parameter structure
- The maximum neuron index is determined by finding the maximum value in the `neurons` vector
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
    update_spikes!,
    SpikeTime
