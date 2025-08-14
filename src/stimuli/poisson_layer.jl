"""
    PoissonLayerParameter

    Poisson stimulus with rate defined for each cell in the layer. Each neuron of the 'N' Poisson population fires with 'rate'.
    The connectivity is defined by the parameter 'ϵ'. Thus, the number of presynaptic neuronsconnected to the postsynaptic neuronsis 'N*ϵ'. Each post-synaptic cell receives rate: 'rate * N * ϵ'.

    # Fields
    - `rate::Vector{R}`: A vector containing the rate of the Poisson stimulus.
    - `N::Int32`: The number of neuronsin the layer.
    - `ϵ::Float32`: The fraction of presynaptic neuronsconnected to the postsynaptic neurons.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonLayerParameter

@snn_kw struct PoissonLayerParameter{R = Float32} <: PoissonStimulusParameter
    rate::Float32
    N::Int32
    rates::Vector{R} = fill(Float32.(rate), N)
    p::Float32
    μ::Float32
    σ::Float32 = 0
    active::Vector{Bool} = [true]
end

function PoissonLayerParameter(rate::R; kwargs...) where {R<:Real}
    N = kwargs[:N]
    rates = fill(Float32.(rate), N)
    return PoissonLayerParameter(; N=N, kwargs..., rate = rate, rates = rates)
end

function PoissonStimulusLayer(rate::R; kwargs...) where {R<:Real}
    @warn "PoissonStimulusLayer is deprecated, use PoissonLayer instead."
    return PoissonLayerParameter(rate; kwargs...)
end



function PoissonLayer(
    post::T,
    sym::Symbol,
    target = nothing;
    w = nothing,
    param::P,
    dist::Symbol = :Normal,
    kwargs...,
) where {T<:AbstractPopulation, P<:PoissonStimulusParameter}

    w = sparse_matrix(w, param.N, post.N, dist, param.μ, param.σ, param.p)

    ## select a subset of neuronsthat receive the stimulus
    rowptr, colptr, I, J, index, W = dsparse(w)

    targets = Dict(:pre => :PoissonStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, target)

    # Construct the SpikingSynapse instance
    return PoissonStimulus(;
        param = param,
        N = param.N,
        N_pre = 0,
        neurons = unique(J),
        targets = targets,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
        kwargs...,
    )
end


function stimulate!(
    p::PoissonStimulus,
    param::PoissonLayerParameter,
    time::Time,
    dt::Float32,
)
    @unpack N, randcache, fire, neurons, colptr, W, I, g = p
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

export PoissonLayerParameter, PoissonLayer, stimulate!
