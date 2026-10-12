"""
    rand_value(N, p1, p2)

Generate `N` random values (`Float64`) uniformly distributed between `p1` and `p2`.

# Arguments
- `N`: Number of random values to generate
- `p1`: First boundary value
- `p2`: Second boundary value

# Returns
- Vector of N random values between min(p1,p2) and max(p1,p2)
"""
rand_value(N, p1, p2) = minimum([p1, p2]) .+ rand(N) .* abs(p1-p2)

"""
    exp32(x::R) where {R<:Real}

Fast approximation of `exp(x)` as ``(1 + x/32)^{32}``, computed with five squarings.
Inputs below `-10` are clamped to `-10`.

# Returns
- `Float32` approximation of `exp(x)`; accurate only for small `|x|`.
"""
@inline function exp32(x::R) where {R<:Real}
    x = ifelse(x < -10.0f0, -10.0f0, x)
    x = 1.0f0 + x / 32.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end

"""
    exp64(x::AbstractFloat)

Fast approximation of `exp(x)` as ``(1 + x/64)^{64}``, computed with six squarings.
Inputs below `-10` are replaced by `-64`, so that the result is exactly `0`.

# Returns
- approximation of `exp(x)` in the type of `x` (`Float32` for an integer argument); accurate only for small `|x|`.

# Example
```julia
using SpikingNeuralNetworks
SNNModels.exp64(-1.0f0)   # 0.3650, exp(-1) = 0.3679
```
"""
@inline function exp64(x::T) where {T<:AbstractFloat}
    x = ifelse(x < T(-10), T(-64), x)
    x = one(T) + x / T(64)
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end
# other real arguments (integers, rationals) are computed in Float32
@inline exp64(x::Real) = exp64(Float32(x))

"""
    exp256(x::AbstractFloat)

Fast approximation of `exp(x)` as ``(1 + x/256)^{256}``, computed with eight squarings.
Inputs below `-10` are replaced by `-256`, so that the result is exactly `0`.

# Returns
- approximation of `exp(x)` in the type of `x` (`Float32` or `Float64`).
"""
@inline function exp256(x::T) where {T<:AbstractFloat}
    x = ifelse(x < T(-10), T(-256), x)
    x = one(T) + x / T(256)
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

# `Base.tanh` / `Base.cosh` of a `Float32` (same algorithm, same bits) with `exp` inlined, so that loops using them
# vectorise; the `Base` functions call an out-of-line `exp`.
@inline function _tanh_inl(x::Float32)
    abs2x = abs(2x)
    abs2x >= Base.Math.TANH_LARGE_X(Float32) && return copysign(one(Float32), x)
    abs2x <= Base.Math.TANH_SMALL_X(Float32) && return x * Base.Math.tanh_kernel(x * x)
    k = @inline exp(abs2x)
    return copysign(1 - 2 / (k + 1), x)
end

@inline _tanh_inl(x::AbstractFloat) = tanh(x)     # other precisions: Base
@inline _cosh_inl(x::AbstractFloat) = cosh(x)

@inline function _cosh_inl(x::Float32)
    absx = abs(x)
    if absx <= Base.Math.COSH_SMALL_X(Float32)
        return Base.Math.cosh_kernel(x * x)
    elseif absx >= Base.Math.H_LARGE_X(Float32)
        E = @inline exp(0.5f0 * absx)
        return 0.5f0 * E * E
    end
    E = @inline exp(absx)
    return 0.5f0 * (E + 1 / E)
end

"""
    name(pre, post, k=nothing)

Generate a Symbol name for connections between populations.

# Arguments
- `pre`: Pre-synaptic population name
- `post`: Post-synaptic population name
- `k`: Optional suffix for the name

# Returns
- Symbol in format `:pre_to_post` or `:pre_to_post_k` if k is provided
"""
name(pre, post, k = nothing) =
    isnothing(k) ? Symbol("$(pre)_to_$(post)") : Symbol("$(pre)_to_$(post)_$(k)")

"""
    str_name(pre, post, k=nothing)

Generate a String name for connections between populations.

# Arguments
- `pre`: Pre-synaptic population name (String or Symbol)
- `post`: Post-synaptic population name
- `k`: Optional suffix for the name

# Returns
- String in format "pre_to_post" or "pre_to_post_k" if k is provided
"""
str_name(pre, post, k = nothing) =
    isnothing(k) ? "$(pre)_to_$(post)" : "$(pre)_to_$(post)_$(k)"

"""
    str_name(pre::String, k=nothing)

Generate a String name with optional suffix.

# Arguments
- `pre::String`: Base name
- `k`: Optional suffix

# Returns
- String in format "pre" or "pre_k" if k is provided
"""
str_name(pre::String, k = nothing) = isnothing(k) ? "$pre" : "$(pre)_$(k)"



"""
    sorted_ntuple(d::AbstractDict; by = identity)

`NamedTuple` with the entries of `d` ordered by `sort(keys; by)`; `String` keys become `Symbol`s.
Replaces `DrWatson.dict2ntuple(sort(d; by))`: OrderedCollections 2 removed `sort(::Dict)`.
"""
function sorted_ntuple(d::AbstractDict; by = identity)
    ks = sort!(collect(keys(d)); by = by)
    return NamedTuple{Tuple(Symbol.(ks))}(Tuple(d[k] for k in ks))
end

"""
    compose(args...; name = randstring(10), silent = false, time = nothing, kwargs...)

Assemble populations, connections and stimuli (or whole models) into one network model.

Returns the `NamedTuple` `(pop = ..., syn = ..., stim = ..., name = name, time = time)`, where
`pop`, `syn` and `stim` are `NamedTuple`s of components. This is the object accepted by `sim!`,
`train!`, `monitor!`, `save_model`, etc.

# Arguments
- `args...`: models or containers (`NamedTuple`/`Dict`) to merge. Their components keep their own
  keys. `String` and `Time` positional arguments are ignored.
- `kwargs...`: components or containers given by key. A component passed as `E = pop` is stored
  under `:E`. A container passed as `exc = (E = ..., EE = ...)` is flattened with the key as
  prefix (`:exc_E`, `:exc_EE`). Inside containers, the keys `:pop`, `:syn`, `:stim` are descended
  into without prefixing and the key `:name` is skipped.
- `name`: model name (default: random 10-character string).
- `silent = false`: if `false`, print the model summary with `print_model`.
- `time = nothing`: the clock of the new model; by default a new `Time{FT}()` with `FT` the promotion of the `float_type`s
  of the populations. The clocks of merged models are not reused: pass `time = model.time` to keep one.

Components are classified by type (`AbstractPopulation`, `AbstractConnection`,
`AbstractStimulus` or `AbstractStimulusGroup`); a duplicated key is an `AssertionError` (to replace a synapse of an
existing model use [`change_plasticity`](@ref)).
`pop` and `syn` are sorted by key, `stim` by the stimuli `name`.

# Example
```julia
using SpikingNeuralNetworks
@load_units
E = SNN.IF(N = 10, name = "E")
I = SNN.IF(N = 5, name = "I")
EI = SNN.SpikingSynapse(E, I, :ge, conn = (μ = 1.0, p = 0.2))
model = SNN.compose(; E, I, EI, name = "toy", silent = true)
model2 = SNN.compose(model, extra = (A = SNN.IF(N = 3, name = "A"),), silent = true)
keys(model2.pop)   # (:E, :I, :extra_A)
sim!(model, 100ms)
```
"""
function compose(args...; name = randstring(10), silent = false, time = nothing, kwargs...)
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
    pop = sorted_ntuple(pop)
    syn = sorted_ntuple(syn)
    stim = sorted_ntuple(stim, by = x -> stim[x].name)
    time = isnothing(time) ? Time{_model_float(values(pop))}() : time
    model = (pop = pop, syn = syn, stim = stim, name = name, time = time)
    if !silent
        print_model(model)
    end
    return model
end


"""
    f2l(s, l=10)

Format string to fixed length by padding or truncating.

# Arguments
- `s`: Input value to convert to string
- `l`: Target length (default: 10)

# Returns
- String of exactly length `l`, padded with spaces or truncated
"""
function f2l(s, l = 10)
    s = string(s)
    if length(s) < l
        return s * repeat(" ", l - length(s))
    else
        return s[1:l]
    end
end


"""
    print_model(model, get_keys = false)

Log (with `@info`) a summary of a model built by `compose`: name, current time (s), and one line
per population (name, type, `N`, parameter type), per synapse (name, pre/post populations,
long-term plasticity type, metaplasticity, short-term plasticity type; `AbstractMetaPlasticity`
connections are skipped in the list) and per stimulus (name, target, type). The connectivity is
taken from `graph(model)`. `get_keys` is accepted but does not change the output.

# Arguments
- `model`: The model containing populations, synapses, and stimuli to be printed.

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
                "$(f2l(name, 18)) : $(f2l(syn_pop, 30)) || $(f2l(ltp_name))- $(f2l(meta)) : $(f2l(stp_name))",
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
            @assert (typeof(getfield(stim, k)) <: AbstractStimulus || typeof(getfield(stim, k)) <: AbstractStimulusGroup) "Expected stimulus, got $(typeof(getfield(network.stim,k)))"
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

# Message of the duplicate-key assertion of `extract_items` (`root`: the `compose` keyword that delivered `key`, empty if positional).
function _dup_msg(kind, key, root)
    where = isempty(String(root)) ? "a positional argument of `compose`" : "the container `$(root)` given to `compose`"
    hint = kind == "synapse" ?
        " To replace a synapse of a model (e.g. with another plasticity rule), do not pass `compose(; model..., $(key) = new)` " *
        "(`model` already holds `$(key)`); use `change_plasticity(model, :$(key); LTP, STP)`." : ""
    return "Duplicate $(kind) key `$(key)`: a $(kind) with this name was already collected before $(where).$(hint)"
end

"""
    extract_items(root::Symbol, container; pop::Dict{Symbol,Any}, syn::Dict{Symbol,Any}, stim::Dict{Symbol,Any}, time::Time)

Recursively collect the components found in `container` into `pop`, `syn` and `stim`
(helper of `compose`). Keys of nested containers are prefixed with `root` (`root_key`) unless
`root` is empty or one of `:pop`, `:syn`, `:stim`; `:name` entries and `Time` values are skipped.

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
    time,
)
    function special_key(k)
        k == :pop || k == :syn || k == :stim
    end

    v = container
    if typeof(v) <: AbstractPopulation
        @assert !haskey(pop, root) _dup_msg("population", root, root)
        push!(pop, root => v)
    elseif typeof(v) <: AbstractConnection
        @assert !haskey(syn, root) _dup_msg("synapse", root, root)
        push!(syn, root => v)
    elseif typeof(v) <: AbstractStimulus || typeof(v) <: AbstractStimulusGroup
        @assert !haskey(stim, root) _dup_msg("stimulus", root, root)
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
                @assert !haskey(pop, new_key) _dup_msg("population", new_key, root)
                push!(pop, new_key => v)
            elseif typeof(v) <: AbstractConnection
                @assert !haskey(syn, new_key) _dup_msg("synapse", new_key, root)
                push!(syn, new_key => v)
            elseif typeof(v) <: AbstractStimulus || typeof(v) <: AbstractStimulusGroup
                @assert !haskey(stim, new_key) _dup_msg("stimulus", new_key, root)
                push!(stim, new_key => v)
            else
                extract_items(new_key, v; pop, syn, stim, time)
            end
        end
    end
    return true
end

"""
    remove_element(model, key)

Remove an element (population, synapse, or stimulus) from a model by its key.

# Arguments
- `model`: The network model
- `key`: Symbol key of the element to remove

# Returns
- New model with the element removed

# Throws
- Nothing: if `key` is not found, an `@info` message is logged and the model is returned unchanged.

The new model is built with `compose(; model..., silent = true)`, so it keeps the `name` and
`time` of `model`. `remove_element(model, keys::Vector)` removes several elements.
"""
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
        # throw(ArgumentError("Element not found"))
        @info "Element with key $(key) not found in model. No element removed."
    end
    compose(;model..., pop, syn, stim, silent = true)
end

function remove_element(model, key::Vector)
    for k in key
        model = remove_element(model, k)
    end
    return model
end

"""
    change_plasticity(model, syn_name::Symbol; LTP = nothing, STP = nothing)

A new model in which the synapse `syn_name` is replaced by `with_plasticity(model.syn[syn_name]; LTP, STP)` (see
[`with_plasticity`](@ref)): a rule left to `nothing` is kept, `NoLTP()` / `NoSTP()` switch that part off. Everything
else in the result is the same object as in `model`, which is not modified. The rule types are type parameters of the
synapse, so there is no in-place model-level variant.

The new synapse shares `W`, the connectivity, the delay queue, the records and any kept rule state with the old one:
simulate only the new model (or `deepcopy(model)` first). A `SynapseNormalization` or `AggregateScaling` in the model keeps
acting on the old synapse.

# Throws
`ArgumentError` if `syn_name` is not a synapse of the model; the message lists the available names.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100); E.I .= 300.0
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5))
model = SNN.compose(; E, EE, silent = true)
model_p = SNN.change_plasticity(model, :EE; LTP = SNN.STDPGerstner())
SNN.train!(; model = model_p, duration = 100ms)
model_f = SNN.change_plasticity(model_p, :EE; LTP = SNN.NoLTP(), STP = SNN.NoSTP())   # weights frozen
```
"""
function change_plasticity(model, syn_name::Symbol; LTP = nothing, STP = nothing)
    if !haskey(model.syn, syn_name)
        throw(ArgumentError("model has no synapse `:$(syn_name)`; available synapses: " *
                            join(map(k -> ":" * String(k), keys(model.syn)), ", ")))
    end
    old = model.syn[syn_name]
    old isa SpikingSynapse || throw(ArgumentError("`$(syn_name)` is a $(nameof(typeof(old))), not a SpikingSynapse; " *
                                                  "only SpikingSynapse has switchable plasticity"))
    new = if LTP isa NoLTP && STP isa NoSTP
        without_plasticity(old)
    else
        with_plasticity(old; LTP, STP)
    end
    syn = merge(model.syn, NamedTuple{(syn_name,)}((new,)))
    return merge(model, (; syn))
end

"""
    merge_models(args...; kwargs...)

Deprecated alias of `compose` (emits a warning).
"""
function merge_models(args...; kwargs...)
    @warn "merge_models is deprecated, use `compose` instead"
    compose(args...; kwargs...)
end

export compose,
    change_plasticity,
    merge_models, # compose
    remove_element,
    print_model,
    exp64,
    exp256,
    name,
    str_name


# precision of a set of populations: the promotion of their float types (Float32 for none)
_model_float(pops) = reduce(promote_type, (float_type(p) for p in pops); init = Float32)
