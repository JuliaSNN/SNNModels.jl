"""
    population_indices(P)

Global neuron indices of each population when the populations of `P` (a `NamedTuple`, e.g.
`model.pop`) are concatenated in the order of `keys(P)`, as in `spiketimes(model.pop)`.

# Returns
A `NamedTuple` (sorted by key) mapping each key to the `Vector{Int}` of its indices.

# Example
```julia
using SpikingNeuralNetworks
pops = (E = SNN.IF(N = 4), I = SNN.IF(N = 2))
SNN.population_indices(pops)   # (E = [1, 2, 3, 4], I = [5, 6])
```
"""
function population_indices(P)
    n = 1
    indices = Dict{Symbol,Vector{Int}}()
    for k in keys(P)
        p = getfield(P, k)
        indices[k] = n:(n+p.N-1)
        n += p.N
    end
    return dict2ntuple(sort(indices))
end

no_noise(p) = !occursin(string("noise"), string(p.name))

"""
    filter_items(P; condition::Function = no_noise)

Keep the items of `P` (a `NamedTuple` of components) that have a `name` field and for which
`condition(item)` is true. The default condition drops items whose name contains `"noise"`.

# Returns
A `NamedTuple` of the selected items, sorted by their `name`.
"""
function filter_items(P; condition::Function = no_noise)
    populations = Dict{Symbol,Any}()
    for k in keys(P)
        p = getfield(P, k)
        hasfield(typeof(p), :name) || continue
        condition(p) || continue
        p = getfield(P, k)
        push!(populations, k => p)
    end
    return dict2ntuple(sort(populations, by = x -> getfield(P, x).name))
end



"""
    subpopulations(stim, subset = nothing)

Neurons targeted by each stimulus of `stim` (a `NamedTuple` of stimuli, e.g. `model.stim`).

# Arguments
- `stim`: `NamedTuple` of stimuli; `neurons(s)` gives the target neurons of each.
- `subset`: optional collection of stimulus names (strings); other stimuli are skipped.

# Returns
- A `NamedTuple`, sorted by name, mapping each stimulus `name` to the unique ids of the neurons
  it targets.
"""
function subpopulations(stim, subset=nothing)
    populations = Dict{String,Vector{Int}}()
    my_keys = collect(keys(stim))
    for key in my_keys
        name = getfield(stim, key).name
        !isnothing(subset) && !(string(name) ∈ subset) && continue
        populations[name] = vcat(neurons(getfield(stim, key))...) |> unique |> collect
    end
    return dict2ntuple(sort(populations))
end

"""
    target_neurons(stim, targets)

For each key in `targets` (strings or symbols), the unique neuron ids targeted by
`stim[key]` (see `neurons`). Throws if a key is not in `stim`. Returns a `Vector{Vector{Int}}`.
"""
function target_neurons(stim, targets=nothing)
    t_neurons = Vector{Int}[]
    for key in targets
        haskey(stim, Symbol(key)) || throw("Stimulus does not contain target: $key")
        name = getfield(stim, Symbol(key)).name
        push!(t_neurons, vcat(neurons(getfield(stim, Symbol(key)))...) |> unique |> collect)
    end
    return t_neurons
end

"""
    average_conn_strength(M::AbstractMatrix, pops::Vector{Vector{Int}}, sparsity = 0.2)

Matrix of mean connection strengths between groups of neurons: entry `(i, j)` is
`mean(M[pops[i], pops[j]]) / sparsity` (post group `i`, pre group `j`), i.e. the mean weight of
the existing synapses if `M` has connection density `sparsity`.
"""
function average_conn_strength(M::T, pops::Vector{Vector{Int}}, sparsity=0.2) where {T<:AbstractMatrix}
    pre = pops
    post = pops
    ave_conn = zeros(Float32, length(post), length(pre))
    for i in eachindex(post)
        for j in eachindex(pre)
            ave_conn[i, j] = mean(M[post[i], pre[j]])/sparsity
        end
    end
    return ave_conn
end

export population_indices, target_neurons, subpopulations, filter_items, average_conn_strength
