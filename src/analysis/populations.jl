"""
    population_indices(P, type = "ˆ")

Given a dictionary `P` containing population names as keys and population objects as values, this function returns a named tuple `indices` that maps each population name to a range of indices. The range represents the indices of the neurons belonging to that population.

# Arguments
- `P`: A dictionary containing population names as keys and population objects as values.
- `type`: A string specifying the type of population to consider. Only population names that contain the specified type will be included in the output. Defaults to "ˆ".

# Returns
A named tuple `indices` where each population name is mapped to a range of indices.
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

"""
    filter_items(P, regex)

Filter populations in dictionary `P` based on a regular expression `regex`.
Returns a named tuple of populations that match the regex.

# Arguments
- `P`: Container of items.
- `regex`: Regular expression to match population names.

# Returns
A named tuple of populations that match the regex.

# Examples
"""

no_noise(p) = !occursin(string("noise"), string(p.name))

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
    subpopulations(stim)

Extracts the names and the neuron ids projected from a given set of stimuli.

# Arguments
- `stim`: A dictionary containing stimulus information.

# Returns
- `names`: A vector of strings representing the names of the subpopulations.
- `pops`: A vector of arrays representing the populations of the subpopulations.

# Example
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

function target_neurons(stim, targets=nothing)
    t_neurons = Vector{Int}[]
    for key in targets
        haskey(stim, Symbol(key)) || throw("Stimulus does not contain target: $key")
        name = getfield(stim, Symbol(key)).name
        push!(t_neurons, vcat(neurons(getfield(stim, Symbol(key)))...) |> unique |> collect)
    end
    return t_neurons
end

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

export population_indices, target_neurons, filter_populations, subpopulations, filter_items, average_conn_strength
