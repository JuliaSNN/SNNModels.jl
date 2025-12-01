
rand_value(N, p1, p2) = minimum([p1, p2]) .+ rand(N) .* abs(p1-p2)

@inline function exp32(x::R) where {R<:Real}
    x = ifelse(x < -10.0f0, -32.0f0, x)
    x = 1.0f0 + x / 32.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end

@inline function exp64(x::R) where {R<:Real}
    x = ifelse(x < -10.0f0, -64.0f0, x)
    x = 1.0f0 + x / 64.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end

@inline function exp256(x::Float32)
    x = ifelse(x < -10.0f0, -256.0f0, x)
    x = 1.0f0 + x / 256.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end

name(pre, post, k = nothing) =
    isnothing(k) ? Symbol("$(pre)_to_$(post)") : Symbol("$(pre)_to_$(post)_$(k)")
str_name(pre, post, k = nothing) =
    isnothing(k) ? "$(pre)_to_$(post)" : "$(pre)_to_$(post)_$(k)"
str_name(pre::String, k = nothing) = isnothing(k) ? "$pre" : "$(pre)_$(k)"



"""

    compose(kwargs...; syn=nothing, pop=nothing)

Merge multiple models into a single model.

## Arguments
- `kwargs...`: List of `kwarg` elements, i.e., dictionary or named tuples, containing the models to be merged.
    - if `kwarg` has elements with `:pop` and `:syn` entries, the function copies them into the merged model.
    - if `kwarg` has no `:pop` and `:syn` entries, the function iterates over all the elements contained in `kwarg` and merge them into the model.
- `syn`: Optional dictionary of synapses to be merged.
- `pop`: Optional dictionary of populations to be merged.
- `stim`: Optional dictionary of stimuli to be merged.

## Returns
A tuple `(pop, syn)` representing the merged populations and synapses.

## Details
This function takes in multiple models represented as keyword arguments and merges them into a single model. The models can be specified using the `pop` and `syn` fields in the keyword arguments. If the `pop` and `syn` fields are not present, the function expects the keyword arguments to have elements with `:pop` or `:syn` fields.

The merged populations and synapses are stored in dictionaries `populations` and `synapses`, respectively. The function performs type assertions to ensure that the elements being merged are of the correct types (`AbstractPopulation` for populations and `AbstractConnection` for synapses).

If `syn` and/or `pop` and/or `stim` arguments are provided, they are merged into the respective dictionaries.

## Example
"""
function compose(args...; name = randstring(10), silent = false, time = Time(), kwargs...)
    pop = Dict{Symbol,Any}()
    syn = Dict{Symbol,Any}()
    stim = Dict{Symbol,Any}()
    for v in args
        v isa String && continue
        v isa Time && continue
        extract_items(Symbol(""), v, pop = pop, syn = syn, stim = stim, time = time)
    end
    for (k, v) in kwargs
        v isa String && continue
        v isa Time && continue
        extract_items(k, v, pop = pop, syn = syn, stim = stim, time = time)
    end
    pop = DrWatson.dict2ntuple(sort(pop, by = x -> x))
    syn = DrWatson.dict2ntuple(sort(syn, by = x -> x))
    stim = DrWatson.dict2ntuple(sort(stim, by = x -> stim[x].name))
    name = haskey(kwargs, :name) ? args.name : name
    model = (pop = pop, syn = syn, stim = stim, name = name, time = time)
    if !silent
        print_model(model)
    end
    return model
end


function f2l(s, l = 10)
    s = string(s)
    if length(s) < l
        return s * repeat(" ", l - length(s))
    else
        return s[1:l]
    end
end


"""
    print_model(model)

Prints the details of the given model. 
The model is expected to have three components: `pop` (populations), `syn` (synapses), and `stim` (stimuli).

The function displays a graph representation of the model, followed by detailed information about each component.

# Arguments
- `model`: The model containing populations, synapses, and stimuli to be printed.

# Outputs
Prints the graph of the model, along with the name, key, type, and parameters of each component in the populations, synapses, and stimuli.

# Exception
Raises an assertion error if any component in the populations is not a subtype of `AbstractPopulation`, if any component in the synapses is not a subtype of `AbstractConnection`, or if any component in the stimuli is not a subtype of `AbstractStimulus`.

"""
function print_model(model, get_keys = false)
    model_graph = graph(model)
    @unpack pop, syn, stim = model
    populations = Vector{String}()
    for k in keys(pop)
        v = filter_first_vertex(model_graph, (g, v) -> get_prop(model_graph, v, :key) == k)
        name = props(model_graph, v)[:name]
        _k = get_keys ? "($k)" : ""
        @assert typeof(getfield(pop, k)) <: AbstractPopulation "Expected neuron, got $(typeof(getfield(network.pop,k)))"
        push!(
            populations,
            "$(f2l(name)): $(f2l(nameof(typeof(getfield(pop,k))))):  $(f2l(getfield(pop,k).N)) $(f2l((nameof(typeof(getfield(pop,k).param)))))",
        )

    end
    synapses = Vector{String}()
    for k in keys(syn)
        typeof(syn[k]) <: AbstractMetaPlasticity && continue
        _edges, _ids = filter_edge_props(model_graph, :key, k)
        for (e, i) in zip(_edges, _ids)
            name = props(model_graph, e)[:name][i]
            syn_pop = props(model_graph, e)[:pop][i]
            _k = get_keys ? "($k)" : ""
            meta =
                props(model_graph, e)[:meta][i] !== :none ?
                "($(props(model_graph, e)[:meta][i]))" : ""
            # @info "$name $(_k) $meta: \n $(nameof(typeof(getfield(syn,k)))): $(nameof(typeof(getfield(syn,k).param)))"
            @assert typeof(getfield(syn, k)) <: AbstractConnection "Expected synapse, got $(typeof(getfield(network.syn,k)))"
            if hasfield(typeof(getfield(syn, k)), :LTPParam)
                ltp_name = nameof(typeof(getfield(syn, k).LTPParam))
                stp_name = nameof(typeof(getfield(syn, k).STPParam))
            else
                ltp_name = nothing
                stp_name = nothing
            end
            push!(
                synapses,
                "$(f2l(name, 18)) : $(f2l(syn_pop, 30)):$(f2l(meta)) : $(f2l(ltp_name)) : $(f2l(stp_name))",
            )
        end
    end
    stimuli = Vector{String}()
    for k in keys(stim)
        _edges, _ids = filter_edge_props(model_graph, :key, k)
        for (e, i) in zip(_edges, _ids)
            name = props(model_graph, e)[:name][i]
            syn_pop = props(model_graph, e)[:pop][i]
            _k = get_keys ? "($k)" : ""
            # @info "$name $(_k): $(nameof(typeof(getfield(stim,k)))): $(nameof(typeof(getfield(stim,k).param)))"
            @assert typeof(getfield(stim, k)) <: AbstractStimulus "Expected stimulus, got $(typeof(getfield(network.stim,k)))"
            push!(
                stimuli,
                "$(f2l(name)) $(_k): $(f2l(syn_pop, 30)) $(nameof(typeof(getfield(stim,k))))",
            )
        end
    end
    sort!(stimuli)
    sort!(synapses)
    sort!(populations)

    @info "================"
    @info "Model: $(model.name)"
    @info "Time: $(get_time(model.time)/1000) s"
    @info "----------------"
    @info "Populations ($(length(populations))):"
    for p in populations
        @info p
    end
    @info "----------------"
    @info "Synapses ($(length(synapses))): "
    for s in synapses
        @info s
    end
    @info "----------------"
    @info "Stimuli ($(length(stimuli))):"
    for s in stimuli
        @info s
    end
    @info "================"
end

"""
    extract_items(root::Symbol, container; pop::Dict{Symbol,Any}, syn::Dict{Symbol, Any}, stim::Dict{Symbol,Any})

Extracts items from a container and adds them to the corresponding dictionaries based on their type.

## Arguments
- `root::Symbol`: The root symbol for the items being extracted.
- `container`: The container from which to extract items.
- `pop::Dict{Symbol,Any}`: The dictionary to store population items.
- `syn::Dict{Symbol, Any}`: The dictionary to store synapse items.
- `stim::Dict{Symbol,Any}`: The dictionary to store stimulus items.

## Returns
- `true`: Always returns true.

## Details
- If the type of the item in the container is `AbstractPopulation`, it is added to the `pop` dictionary.
- If the type of the item in the container is `AbstractConnection`, it is added to the `syn` dictionary.
- If the type of the item in the container is `AbstractStimulus`, it is added to the `stim` dictionary.
- If the type of the item in the container is none of the above, the function is recursively called to extract items from the nested container.
"""
function extract_items(
    root::Symbol,
    container;
    pop::Dict{Symbol,Any},
    syn::Dict{Symbol,Any},
    stim::Dict{Symbol,Any},
    time::Time,
)
    function special_key(k)
        k == :pop || k == :syn || k == :stim
    end

    v = container
    if typeof(v) <: AbstractPopulation
        @assert !haskey(pop, root) "Population $(root) already exists"
        push!(pop, root => v)
    elseif typeof(v) <: AbstractConnection
        @assert !haskey(syn, root) "Receptors $(root) already exists"
        push!(syn, root => v)
    elseif typeof(v) <: AbstractStimulus
        @assert !haskey(stim, root) "Stimulus $(root) already exists"
        push!(stim, root => v)
    elseif typeof(v) <: Time
        # update_time!(time, v)
    else
        for k in keys(container)
            k == :name && continue
            v = getindex(container, k)
            if special_key(k)
                extract_items(root, v; pop, syn, stim, time)
                continue
            end
            new_key = k
            if !isempty(String(root)) && !special_key(root)
                new_key = Symbol(string(root) * "_" * string(k))
            end
            if typeof(v) <: AbstractPopulation
                @assert !haskey(pop, new_key) "Population $(new_key) already exists"
                push!(pop, new_key => v)
            elseif typeof(v) <: AbstractConnection
                @assert !haskey(syn, new_key) "Receptors $(new_key) already exists"
                push!(syn, new_key => v)
            elseif typeof(v) <: AbstractStimulus
                @assert !haskey(stim, new_key) "Stimulus $(new_key) already exists"
                push!(stim, new_key => v)
            else
                extract_items(new_key, v; pop, syn, stim, time)
            end
        end
    end
    return true
end

function remove_element(model, key)
    pop = Dict(pairs(model.pop))
    syn = Dict(pairs(model.syn))
    stim = Dict(pairs(model.stim))
    if haskey(model.pop, key)
        delete!(pop, key)
    elseif haskey(model.syn, key)
        delete!(syn, key)
    elseif haskey(model.stim, key)
        delete!(stim, key)
    else
        throw(ArgumentError("Element not found"))
    end
    compose(pop, syn, stim)
end


function merge_models(args...; kwargs...)
    @warn "merge_models is deprecated, use `compose` instead"
    compose(args...; kwargs...)
end

export compose,
    merge_models, # compose
    remove_element,
    print_model,
    exp64,
    exp256,
    name,
    str_name
