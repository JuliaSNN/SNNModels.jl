abstract type PoissonLayerParameter <: AbstractStimulusParameter end

"""
    PoissonLayer

    Poisson stimulus with rate defined for each cell in the layer. Each neuron of the 'N' Poisson population fires with 'rate'.
    The connectivity is defined by the parameter 'ϵ'. Thus, the number of presynaptic neuronsconnected to the postsynaptic neuronsis 'N*ϵ'. Each post-synaptic cell receives rate: 'rate * N * ϵ'.

    # Fields
    - `rate::Vector{R}`: A vector containing the rate of the Poisson stimulus.
    - `N::Int32`: The number of neuronsin the layer.
    - `ϵ::Float32`: The fraction of presynaptic neuronsconnected to the postsynaptic neurons.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonLayer

@snn_kw struct PoissonLayer{FT = Float32} <: PoissonLayerParameter
    rate::FT  # Default rate in Hz
    N::Int32 = 1
    active::VBT = [true]
end

@snn_kw struct PoissonLayerHet{VFT = Vector{Float32}} <: PoissonLayerParameter
    N::Int32 = 1
    rates::VFT
    active::VBT = [true]
end

function PoissonLayer(rate::R; kwargs...) where {R<:Real}
    N = kwargs[:N]
    return PoissonLayer(; N = N, rate = rate)
end

function PoissonLayerHet(rate::R; kwargs...) where {R<:Real}
    N = kwargs[:N]
    return PoissonLayerHet(; N = N, rates = fill(rate, N))
end

"""
    PoissonStimulusLayer{VFT, VBT, VIT, IT}

A layer representing a Poisson stimulus applied to a postsynaptic population.

# Fields
- `N::Int`: Number of neurons in the stimulus layer.
- `id::String`: Unique identifier for the layer.
- `name::String`: Name of the layer.
- `param::PoissonLayer`: Parameters for the Poisson stimulus.
- `g::VFT`: Target conductance for the soma.
- `colptr::VIT`: Column pointer for sparse connectivity.
- `rowptr::VIT`: Row pointer for sparse connectivity.
- `I::VIT`: Row indices for sparse connectivity.
- `J::VIT`: Column indices for sparse connectivity.
- `index::VIT`: Indices for sparse connectivity.
- `W::VFT`: Weights for synaptic connections.
- `fire::VBT`: Boolean vector indicating which neurons fired.
- `randcache::VFT`: Random cache for Poisson spike generation.
- `records::Dict`: Dictionary for storing simulation records.
- `targets::Dict`: Dictionary specifying target populations.

This layer implements a Poisson stimulus where each neuron fires independently with a given rate,
and the connectivity is defined by sparse matrix representations.
"""
PoissonStimulusLayer
@snn_kw struct PoissonStimulusLayer{VFT = Vector{Float32},PT<:PoissonLayerParameter} <:
               AbstractStimulus
    N::Int
    id::String = randstring(12)
    name::String = "Poisson"
    param::PT = PoissonLayer(-1)
    ##
    g::VFT # target conductance for soma
    colptr::VIT
    rowptr::VIT
    I::VIT
    J::VIT
    index::VIT
    W::VFT
    fire::VBT = zeros(Bool, N)
    ##
    randcache::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
end


function PoissonStimulusLayer(
    post::T,
    sym::Symbol,
    comp = nothing;
    conn::Connectivity,
    param::PoissonLayerParameter,
    name::String = "Poisson",
) where {T<:AbstractPopulation}
    # @warn "PoissonStimulusLayer is deprecated. Please use Stimulus(param, post, sym, comp; conn) instead."

    w = sparse_matrix(param.N, post.N, conn)
    rowptr, colptr, I, J, index, W = dsparse(w)
    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, comp)

    # Construct the SpikingSynapse instance
    return PoissonStimulusLayer(;
        param = param,
        N = param.N,
        targets = targets,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
        name = name,
    )
end

function Stimulus(
    param::PoissonLayerParameter,
    post::T,
    sym::Symbol,
    comp = nothing;
    conn::NamedTuple,
    name::String = "Poisson",
) where {T<:AbstractPopulation}

    w = sparse_matrix(param.N, post.N; conn...)
    rowptr, colptr, I, J, index, W = dsparse(w)
    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, comp)

    # Construct the SpikingSynapse instance
    return PoissonStimulusLayer(;
        param = param,
        N = param.N,
        targets = targets,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
        name = name,
    )
end


"""
    stimulate!(p::PoissonStimulus, param::PoissonLayer, time::Time, dt::Float32)

Generate a Poisson stimulus for a postsynaptic population.

# Arguments
- `p`: Poisson stimulus.
- `param`: Parameters for the stimulus.
- `time`: Current time.
- `dt`: Time step.
"""

function stimulate!(p::PoissonStimulusLayer, param::PoissonLayer, time::Time, dt::Float32)
    @unpack N, randcache, fire, colptr, W, I, g = p
    @unpack rate = param
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < rate * dt
            fire[j] = true
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s]
            end
        else
            fire[j] = false
        end
    end
end

function stimulate!(
    p::PoissonStimulusLayer,
    param::PoissonLayerHet,
    time::Time,
    dt::Float32,
)
    @unpack N, randcache, fire, colptr, W, I, g = p
    @unpack rates = param
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < rates[j] * dt
            fire[j] = true
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s]
            end
        else
            fire[j] = false
        end
    end
end

export PoissonLayer, stimulate!, PoissonStimulusLayer, Stimulus, PoissonLayerHet
