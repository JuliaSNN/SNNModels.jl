"""
    BalancedStimulusParameter{VFT} <: AbstractParameter

A parameter struct for the BalancedStimulus, containing parameters for the balanced input distribution.
The balanced stimulus generates both excitatory and inhibitory inputs to a postsynaptic population, maintaining a balance between excitation and inhibition. The balance is controlled by two parameters that define the characteristics of the input.
kIE: Scaling factor for inhibitory rate.
wIE: Weight for inhibitory connections.

The parameter β controls the noise in the firing rate, with higher values leading to more variability. The time constant τ determines how quickly the noise decays over time. The baseline firing rate r0 sets the average rate of input spikes.

# Fields
- `kIE::Float32`: Scaling factor for inhibitory rate (default: 1.0)
- `β::Float32`: Noise parameter (default: 0.0)
- `τ::Float32`: Time constant for noise (default: 50.0 ms)
- `r0::Float32`: Baseline firing rate (default: 1kHz)
- `wIE::Float32`: Weight for inhibitory connections (default: 1.0)
- `same_input::Bool`: Whether to use same input for all neurons (default: false)
"""
BalancedParameter

@snn_kw struct BalancedParameter{VFT}
    kIE::Float32 = 1.0
    β::Float32 = 0.0
    τ::Float32 = 50.0ms
    r0::Float32 = 1kHz
    wIE::Float32 = 1.0
    same_input::Bool = false
end

"""
    BalancedStimulus{
        VFT = Vector{Float32},
        VBT = Vector{Bool},
        VIT = Vector{Int},
        IT = Int32,
    } <: AbstractStimulus
A stimulus that generates balanced excitatory and inhibitory inputs to a postsynaptic population.
# Fields
- `param::BalancedStimulusParameter`: Parameters for the balanced stimulus.
- `N::IT`: Number of neurons in the stimulus.
- `N_pre::IT`: Number of presynaptic neurons connected to each postsynaptic
- `neurons::VIT`: Indices of neurons in the postsynaptic population receiving the stimulus.
- `ge::VFT`: Target excitatory conductance for each neuron.
- `gi::VFT`: Target inhibitory conductance for each neuron.
- `colptr::VIT`: Column pointers for sparse connectivity matrix.
- `rowptr::VIT`: Row pointers for sparse connectivity matrix.
- `I::VIT`: Row indices for sparse connectivity matrix.
- `J::VIT`: Column indices for sparse connectivity matrix.
- `index::VIT`: Indices for non-zero entries in sparse connectivity matrix.
- `W::VFT`: Weights for connections in sparse connectivity matrix.
- `r::VFT`: Firing rates for each neuron.
- `noise::VFT`: Noise values for each neuron.
- `fire::VBT`: Boolean array indicating if presynaptic neurons have fired.
- `randcache::VFT`: Cache for random values used in spike generation.
- `randcache_β::VFT`: Cache for random values used in noise generation.
- `records::Dict`: Dictionary for recording variables during simulation.
- `targets::Dict`: Dictionary specifying the target populations and synaptic variables.
"""
BalancedStimulus

@snn_kw struct BalancedStimulus{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    VIT = Vector{Int},
    IT = Int32,
} <: AbstractStimulus
    id::String = randstring(12)
    param::BalancedParameter
    name::String = "Balanced"
    N::IT = 100
    N_pre::IT = 5
    neurons::VIT
    ##
    ge::VFT # target conductance for exc
    gi::VFT # target conductance for inh
    colptr::VIT
    rowptr::VIT
    I::VIT
    J::VIT
    index::VIT
    r::VFT
    noise::VFT
    W::VFT
    fire::VBT = zeros(Bool, N_pre)
    ##
    randcache::VFT = rand(N_pre) # random cache
    randcache_β::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
end


"""
    BalancedStimulus(post::T, sym::Symbol, r::Union{Function, Float32}, neurons=[]; N_pre::Int=50, p_post::R=0.05f0, μ::R=1.f0, param=BalancedParameter()) where {T <: AbstractPopulation, R <: Number}

Constructs a BalancedStimulus object for a spiking neural network.

# Arguments
- `post::T`: The target population for the stimulus.
- `sym::Symbol`: The symbol representing the synaptic conductance or current.
- `r::Union{Function, Float32}`: The firing rate of the stimulus. Can be a constant value or a function of time.
- `neurons=[]`: The indices of the neuronsin the target population that receive the stimulus. If empty, neuronsare randomly selected based on the probability `p_post`.
- `N::Int=200`: The number of Balanced neurons neurons.
- `N_pre::Int=5`: The number of presynaptic connected.
- `p_post::R=0.05f0`: The probability of connection between presynaptic and postsynaptic neurons.
- `μ::R=1.f0`: The scaling factor for the synaptic weights.
- `param=BalancedParameter()`: The parameters for the Balanced distribution.

# Returns
A `BalancedStimulus` object.
"""
function BalancedStimulus(
    post::T,
    sym_e::Symbol,
    sym_i::Symbol,
    target = nothing;
    neurons = :ALL,
    μ = 1.0f0,
    param::Union{BalancedParameter,R},
    kwargs...,
) where {T<:AbstractPopulation,R<:Real}

    neurons = neurons == :ALL ? (1:post.N) : neurons
    w = zeros(Float32, length(neurons), length(neurons))
    w = μ * sparse(w)
    rowptr, colptr, I, J, index, W = dsparse(w)

    targets = Dict(:pre => :BalancedStim, :post => post.id)
    ge, _ = synaptic_target(targets, post, sym_e, target)
    gi, _ = synaptic_target(targets, post, sym_i, target)

    if typeof(param) <: Real
        r = param
        param = BSParam(rate = (x, y) -> r, r * param.kIE)
    end

    r = ones(Float32, post.N) * param.r0
    noise = zeros(Float32, post.N)

    N_pre = ceil(Int, param.r0 * maximum([1, param.β / 100]))

    return BalancedStimulus(;
        param = param,
        N = length(neurons),
        N_pre = N_pre,
        neurons = neurons,
        targets = targets,
        r = r,
        noise = noise,
        ge = ge,
        gi = gi,
        @symdict(rowptr, colptr, I, J, index, W)...,
        kwargs...,
    )
end


function Stimulus(param::BalancedParameter, post::T, sym::Symbol, target = nothing; kwargs...) where {T<:AbstractPopulation}
    return BalancedStimulus(post, sym, sym, target; param, kwargs...)
end


"""
    stimulate!(p::BalancedStimulus, param::BalancedParameter, time::Time, dt::Float32)

Generate a Balanced stimulus for a postsynaptic population.
"""
function stimulate!(
    p::BalancedStimulus,
    param::BalancedParameter,
    time::Time,
    dt::Float32,
)
    @unpack N, N_pre, randcache, randcache_β, fire, neurons, colptr, W, I, ge, gi = p

    ## Inhomogeneous Poisson process
    @unpack r0, β, τ, kIE, wIE, same_input = param
    @unpack noise, r = p
    # Irate::Float32 = r0 * kIE
    R(x::Float32, v0::Float32 = 0.0f0) = x > 0.0f0 ? x : v0

    # Inhibitory spike
    rand!(randcache)
    for i = 1:N
        @simd for j = 1:N_pre # loop on presynaptic neurons
            if randcache[j] < r0 * kIE / N_pre * dt
                gi[i] += 1 * wIE
            end
        end
    end

    # Excitatory spike
    re::Float32 = 0.0f0
    cc::Float32 = 0.0f0
    Erate::Float32 = 0.0f0
    rand!(randcache_β)
    rand!(randcache)
    if same_input
        i = 1
        re = randcache_β[i] - 0.5f0
        cc = 1.0f0 - dt / τ
        noise[i] = (noise[i] - re) * cc + re
        Erate = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
        r[i] += (r0 - Erate) / 400ms * dt
        @assert Erate >= 0
        @inbounds @fastmath for i = 1:N
            rand!(randcache)
            @simd for j = 1:N_pre # loop on presynaptic neurons
                if randcache[j] < Erate / N_pre * dt
                    ge[i] += 1.0
                end
            end
        end
    else
        @inbounds @fastmath for i = 1:N
            re = randcache_β[i] - 0.5f0
            cc = 1.0f0 - dt / τ
            noise[i] = (noise[i] - re) * cc + re
            Erate = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
            r[i] += (r0 - Erate) / 400ms * dt
            @assert Erate >= 0
            rand!(randcache)
            @simd for j = 1:N_pre # loop on presynaptic neurons
                if randcache[j] < Erate / N_pre * dt
                    ge[i] += 1.0
                end
            end
        end
    end

end

export BalancedStimulus, stimulate!, BSParam, BalancedParameter

