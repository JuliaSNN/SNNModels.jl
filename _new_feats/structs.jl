Spiketimes = Vector{Vector{Float32}}
FT = Float32
VFT = Vector{Float32}
IntervalsSpiketimes = Vector{Vector{Vector{Float32}}}

abstract type SNNDataTypes end
abstract type AbstractStimParams end
abstract type AbstractStoreParams end
abstract type AbstractNetParams end
abstract type AbstractLearnParams end
abstract type AbstractEncoding end

abstract type AbstractPopulationParams end
abstract type AbstractSynParams end
abstract type AbstractConnMap end
abstract type AbstractConnections end

struct States <: SNNDataTypes end
struct Spikes <: SNNDataTypes end
struct Rates <: SNNDataTypes end

# function Base.getproperty(mnt::Weights, sym::Symbol)
#     return getfield(getfield(mnt, :data), sym)
# end
# function Base.fieldnames(mnt::Weights)
#     return fieldnames(typeof(getfield(mnt, :data)))
# end

export SNNData, States, Spikes, Rates, Spiketimes
export SNNDataTypes, Tracker

# export AbstractStimParams, AbstractStoreParams, AbstractNetParams, AbstractLearnParams

abstract type NNParams end
abstract type NNChunks end

@with_kw struct NeuronModels{T<:SNN.AbstractPopulationParameter} <: AbstractPopulationParams
    AdEx::T
    LIF_sst::T
    LIF_pv::T
end

@with_kw struct SynapseModels{T<:SNN.Synapse} <: AbstractSynParams
    Esyn_dend::T
    Esyn_soma::T
    Isyn_sst::T
    Isyn_pv::T
end

@with_kw struct NetParams <: AbstractNetParams
    neurons::Int64 = 1000
    types::Vector
    model::String
    conn::AbstractConnMap
    symmetrical::Bool = false

    tripod::Int64 = round(Int, types[1] * neurons)
    sst::Int64 = round(Int, types[2] * neurons)
    pv::Int64 = round(Int, types[3] * neurons)

    exc_noise::Float32 = 4.0f0
    sst_noise::Float32 = 2.5f0
    pv_noise::Float32 = 2.5f0
    network::Function = null #Simulation gets very slow if network model included
end

@with_kw mutable struct LearnParams <: AbstractLearnParams
    learn_exc::Bool = true
    learn_sst::Bool = true
    learn_pv::Bool = true
    stdp::SNN.AbstractConnectionParameter
    istdp::SNN.AbstractConnectionParameter
    nmda_weights::Bool = true
end


@with_kw mutable struct StoreParams <: AbstractStoreParams
    id::String
    root::String
    path::String = joinpath(root, id)
    params::String = joinpath(path, "params")
    data::String = joinpath(path, "data")
    interval::Int64 = 5000
    store_weights::Bool = true
    store_membrane::Bool = true
    store_spikes::Bool = true
end


@with_kw mutable struct StimParams <: AbstractStimParams
    simtime::Int64   # simulation time (ms)
    dictionary::Union{String,Dict} #define used dictionary
    rate::Float32 ## input rate
    density::Float32 = 0.05f0 # density of inputs
    symbols::Int64 = -1   # number of symbols
    duration::Int64 = 50  # stimulus duration (ms)
    mask_words::Bool = false # don't stimulate the words populations
    seq_length::Int64 = simtime ÷ duration # number of presented stimuli
    input::String = "asymmetric"    # input type
    strength::Float32 = 1.78f0 ## input magnitude
    silence::Int64 = 1 # silence between stimuli (number of phonemes)
end

@with_kw struct DendParams{T<:SNN.Dendrite}
    ds::Matrix{T}
end


MFT = Array{Float32,3}
@with_kw struct Tracker <: NNChunks
    track_neurons::Vector{Int64} = [1]
    interval_steps::Int64 = 1
    interval::Vector{Float32} = [0.0f0, 0.0f0]
    voltage::MFT = zeros(Float32, length(track_neurons), 4, interval_steps)
    currents::MFT = zeros(Float32, length(track_neurons), 6, interval_steps)
    stimuli::MFT = zeros(Float32, length(track_neurons), 6, interval_steps)
    gs::MFT = zeros(Float32, length(track_neurons), 4, interval_steps)
    hs::MFT = zeros(Float32, length(track_neurons), 4, interval_steps)
    g1::MFT = zeros(Float32, length(track_neurons), 4, interval_steps)
    h1::MFT = zeros(Float32, length(track_neurons), 4, interval_steps)
    g2::MFT = zeros(Float32, length(track_neurons), 4, interval_steps)
    h2::MFT = zeros(Float32, length(track_neurons), 4, interval_steps)
end

@with_kw mutable struct NNSpikes <: NNChunks
    exc::Spiketimes = Spiketimes()
    pv::Spiketimes = Spiketimes()
    sst::Spiketimes = Spiketimes()
    tt::Float32 = -1.0f0
    file::String = ""
    _read::Bool = false
end

@with_kw mutable struct NNStates <: NNChunks
    mem::Matrix{Float32} = zeros(Float32, 1, 1)
    cur::Matrix{Float32} = zeros(Float32, 1, 1)
    labels::Matrix{Int64} = zeros(Int, 1, 1)
    timestamps::Vector{Float32} = zeros(Float32, 1)
    tt::Float32 = -1.0f0
    file::String = ""
    _read::Bool = false
end

# @with_kw mutable struct NNWeights <: NNChunks
#     w::Weights = Weights()
#     tt::Float32 = 0.0f0
#     file::String = ""
#     _read::Bool = false
# end

@with_kw mutable struct NNTracker <: NNChunks
    tracker::Tracker = Tracker()
    tt::Float32 = 0.0f0
    file::String = ""
    _read::Bool = false
    _read_fields::Vector{Symbol} = []
end



struct Params
    stim::AbstractStimParams
    seq::AbstractEncoding
    net::AbstractNetParams
    dend::DendParams
    learn::AbstractLearnParams
    store::AbstractStoreParams
    W::AbstractConnections
    neurons::AbstractPopulationParams
    synapses::AbstractSynParams
end
