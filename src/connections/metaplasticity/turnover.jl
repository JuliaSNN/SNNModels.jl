"""
    Abstract type for turnover parameters.
"""
abstract type TurnoverParam <: MetaPlasticityParameter end

"""
    RandomTurnover{FT = Float32} <: TurnoverParam
"""
RandomTurnoverParameter

@snn_kw struct RandomTurnover{FT <: AbstractFloat} <: TurnoverParam
    rate::FT = -1f0
    τ::FT = 1/rate
    threshold::FT = 0.1f0
    μ::FT = 3.0f0
end

"""
    ActivityDependentTurnover{VFT <: Vector{Float32}} <: TurnoverParam
"""
ActivityDependentTurnoverParameter

@snn_kw struct ActivityDependentTurnover{FT} <: TurnoverParam
    rate ::FT = -1
    τ::FT = 1/rate
    fraction::FT = 0.1f0
    τpre::FT = 250f0ms
    τpost::FT = 250f0ms
    μ::FT = 3.0f0
end


@snn_kw struct Turnover{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    ST <: AbstractConnection,
} <: AbstractMetaPlasticity
    id::String = randstring(12)
    name::String = "Turnover"
    param::TurnoverParam = RandomTurnover(0)
    synapse::ST = SpikingSynapse()
    pre::VFT = zeros(Float32, length(synapse.fireJ))
    post::VFT = zeros(Float32, length(synapse.fireI))
    p::MFT = ones(Float32, length(synapse.fireI), length(synapse.fireJ))
    p_rewire::VFT = zeros(Float32, 1)
    p_values::VFT = zeros(Float32, length(synapse.W))
    targets::Dict = Dict()
    records::Dict = Dict()
end


function MetaPlasticity(param::T, synapse;  kwargs...) where {T<:TurnoverParam}
    targets = Dict(
            :synapses => [synapse.id],
            :post => synapse.targets[:post]
    )
    Turnover(;param, kwargs..., synapse, targets)
end

function forward!(c::Turnover, param::T) where {T<:TurnoverParam} end

function plasticity!(c::Turnover, param::ActivityDependentTurnover, dt::Float32, T::Time) where {TT<:TurnoverParam}
    @unpack synapse, p, pre, post, p_values = c
    @unpack fraction, μ, τpre, τpost = param
    @turbo for i in eachindex(synapse.fireJ)
        pre[i] += (-pre[i]*dt + synapse.fireJ[i])/τpre 
    end
    @turbo for i in eachindex(synapse.fireI)
        post[i] += (-post[i]*dt + synapse.fireI[i])/τpost
    end

    ##
    @unpack τ = param
    tt = get_step(T)
    if ((tt) % round(Int, τ / dt)) < dt
        @simd for j in eachindex(synapse.fireJ)
            for s in postsynaptic_idxs(synapse, j)
                p_values[s] = pre[j] * post[synapse.I[s]]
            end
        end
        c.p_rewire[1] = quantile(p_values, fraction)
        # @show mean(p_values)
        # @show c.p_rewire[1]
        # @show quantile(p_values, fraction)
        # @show minimum(p_values)
        # @show maximum(p_values)
        plasticity!(c, param)
    end
end

function plasticity!(c::Turnover, param::TT) where {TT<:TurnoverParam}
    @unpack synapse, p = c
    @unpack μ = param

    synaptic_turnover!(
        c.synapse, 
        p_rewire = c.p_rewire[1], 
        p_new = (post, pre)->p[post, pre],
        p_values = c.p_values,
        μ = μ,
    ) 
end 

export TurnoverParam, RandomTurnover, ActivityDependentTurnover, Turnover, MetaPlasticity


"""
    synaptic_turnover!(C::SpikingSynapse; p_rewire=0.05, p_pre = x->rand(), p_new = x->rand(), μ = 3.0)

Perform synaptic turnover on a spiking synapse connection matrix.

# Arguments
- `C::SpikingSynapse`: The spiking synapse connection to modify
- `p_rewire::Float64=0.05`: Probability threshold for rewiring existing connections
- `p_pre::Function=x->rand()`: Function that returns probability for each presynaptic connection `s` to be rewired 
- `p_new::Function=x->rand()`: Function that returns probability for selecting new postsynaptic neurons
- `μ::Float64=3.0`: Weight value for new connections

# Description
This function implements synaptic turnover by:
1. Generating thresholds for selecting connections to rewire. 
2. Identifying plausible new connections for each presynaptic neuron
3. Selecting connections to rewire based on the probability thresholds
4. Replacing the selected connections with new ones
5. Updating the sparse matrix structure

The function modifies the connection matrix in-place and updates its sparse matrix representation.
"""
function synaptic_turnover!(
    C::S;
    p_rewire = 0.05,
    p_new = x->rand(),
    μ = 3.0,
    p_values = nothing,
) where {S<:AbstractConnection}
    # @info "Performing synaptic turnover on $(C.name)"
    p_values = isnothing(p_values) ? rand(Uniform(0, 1), length(C.W)) : p_values
    all_post = Set(1:length(C.fireI))
    all_posts = postsynaptic(C)
    new_connections = map(eachindex(all_posts)) do pre
        my_post = all_posts[pre]
        plausible_post = setdiff(all_post, my_post) |> collect
        # plausible_post[sortperm([p_new(post, pre) for post in plausible_post])]
        (plausible_post, Weights([p_new(post, pre) for post in plausible_post]))
    end

    rep_connections = Int[]
    rep_neurons = Int[]
    # @show typeof(p_values), size(p_values), p_values[1]
    @unpack rowptr, colptr, I, J, index, W, fireJ = C
    for j in eachindex(fireJ)
        post_n = 0
        for s in postsynaptic_idxs(C, j)
            p_values[s] > p_rewire && continue
            push!(rep_connections, s)
            post_n += 1
        end
        # @show "Changing $(post_n), p_rewire=$(p_rewire)"
        post_n == 0 && continue
        plausible_post, weights = new_connections[j]
        append!(rep_neurons, sample(plausible_post, weights, post_n; replace=false))
    end
    for (s, new_post) in zip(rep_connections, rep_neurons)
        C.I[s] = new_post
        C.W[s] = rand(Normal(μ, sqrt(μ)))
    end
    update_sparse_matrix!(C)
end



