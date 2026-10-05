

"""
    graph(model)

Build a `MetaGraphs.MetaDiGraph` describing the connectivity of a model created by `compose`.

# Vertices
One per population and one per stimulus, with properties `:name` (component name), `:id`
(component `id`) and `:key` (key in `model.pop` or `model.stim`).

# Edges
One edge per ordered pair of vertices; several connections between the same pair are stored as
vectors in the same edge. Edge properties (vectors with one entry per connection): `:type`
(`syn.targets[:type]`, `:fire_to_g` for stimuli), `:name` (connection name), `:pop`
(`"pre -> post.sym"`), `:key`, `:id`, `:meta` (key of the metaplasticity operator acting on the
connection, or `:none`), `:target` (`targets[:sym]`) and `:count`; `:multi` is the number of
connections on the edge.

- Connections are added from `syn.targets[:fire]` (presynaptic id) to `syn.targets[:post]`.
- Stimuli are drawn as an edge from the stimulus vertex to `stim.targets[:post]`.
- `AbstractMetaPlasticity` connections (normalization, scaling, turnover) add no edge; their key
  is written in the `:meta` property of the edges of the synapses listed in
  `targets[:synapses]`.

# Errors
Throws `ArgumentError` for a connection whose `targets` has no `:type` entry, and an error if a
target population id is not in the model.

# Example
```julia
using SpikingNeuralNetworks
@load_units
E = SNN.IF(N = 10, name = "E")
EE = SNN.SpikingSynapse(E, E, :ge, conn = (μ = 1.0, p = 0.2))
g = SNN.graph(SNN.compose(; E, EE, silent = true))   # 1 vertex, 1 self-loop edge
```
"""
function graph(model)
    graph = MetaGraphs.MetaDiGraph()
    @unpack pop, syn, stim = model
    meta_plast = Dict()
    for (k, pop) in pairs(pop)
        name = pop.name
        id = pop.id
        add_vertex!(graph, Dict(:name => name, :id => id, :key => k))
    end
    for (k, syn) in pairs(syn)
        if typeof(syn) <: AbstractMetaPlasticity
            push!(meta_plast, k => syn)
        elseif haskey(syn.targets, :type)
            pre_id = syn.targets[:fire]
            post_id = syn.targets[:post]
            type = syn.targets[:type]
            add_connection!(graph, pre_id, post_id, k, syn, type)
        else
            throw(ArgumentError("Only SpikingSynapse is supported"))
        end
    end
    for (k, stim) in pairs(stim)
        # verterx and edge for the stimulus have the same id
        pre_id = stim.id
        post_id = stim.targets[:post]
        add_vertex!(graph, Dict(:name => stim.name, :id => pre_id, :key => k))
        type = :fire_to_g
        add_connection!(graph, pre_id, post_id, k, stim, type)
    end
    for (k, v) in meta_plast
        ids =
            v.targets[:synapses] isa Vector ? v.targets[:synapses] : [v.targets[:synapses]]
        for id in ids
            _edges, _ids = filter_edge_props(graph, :id, id)
            for (e, i) in zip(_edges, _ids)
                props(graph, e.src, e.dst)[:meta][i] = k
            end
        end
    end
    return graph
end

function add_connection!(graph, pre_id, post_id, k, syn, type)
    pre_node = find_id_vertex(graph, pre_id)
    post_node = find_id_vertex(graph, post_id)
    pre_name = get_prop(graph, pre_node, :name)
    post_name = get_prop(graph, post_node, :name)
    sym = haskey(syn.targets, :sym) ? syn.targets[:sym] : "missing"
    syn_name = "$(syn.name)"
    syn_pop = "$(pre_name) -> $(post_name).$sym"
    id = syn.id
    if !has_edge(graph, pre_node, post_node)
        add_edge!(
            graph,
            pre_node,
            post_node,
            Dict(
                :type => [type],
                :name => [syn_name],
                :pop => [syn_pop],
                :key => [k],
                :id => [id],
                :meta => [:none],
                :target => [sym],
                :count => [1],
                :multi => 1,
            ),
        )
    else
        multi_dict = props(graph, pre_node, post_node)
        _multi = multi_dict[:multi] + 1
        multi_dict[:multi] = _multi
        push!(multi_dict[:name], syn_name)
        push!(multi_dict[:pop], syn_pop)
        push!(multi_dict[:type], type)
        push!(multi_dict[:key], k)
        push!(multi_dict[:id], id)
        push!(multi_dict[:meta], :none)
        push!(multi_dict[:target], sym)
        push!(multi_dict[:count], _multi)
        set_props!(graph, pre_node, post_node, multi_dict)
    end
end

function filter_first_vertex(g::AbstractMetaGraph, fn::Function)
    for v in vertices(g)
        fn(g, v) && return v
    end
    # error("No vertex matching conditions found")
    return nothing
end

function filter_edge_props(g::AbstractMetaGraph, key, value)
    _edges = []
    _ids = []
    for e in edges(g)
        prop = props(g, e)
        for i in eachindex(prop[key])
            if prop[key][i] == value
                push!(_edges, e)
                push!(_ids, i)
            end
        end
    end
    if isempty(_edges)
        # error("No edge matching conditions found")
        return []
    end
    return _edges, _ids
end

function find_id_vertex(g::AbstractMetaGraph, id)
    v = filter_first_vertex(g, (g, v) -> get_prop(g, v, :id) == id)
    isnothing(v) && error("Population $id not found")
    return v
end


function find_key_graph(g::AbstractMetaGraph, id)
    v = filter_first_vertex(g, (g, v) -> get_prop(g, v, :key) == id)
    isnothing(v) && isnothing(e) && error("Vertex or edge not found")
    return insothing(v) ? e : v
end


export graph
