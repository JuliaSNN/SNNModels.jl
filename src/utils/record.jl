import Interpolations: scale, interpolate, BSpline, Linear, NoInterp
"""
    get_time(T::Time)

Get the current time.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The current time.

"""
get_time(T::Time)::Float32 = T.t[1]

get_time(model::NamedTuple)::Float32 = model.time.t[1]

"""
    get_step(T::Time)

Get the current time step.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The current time step.

"""
get_step(T::Time)::Float32 = T.tt[1]

"""
    get_dt(T::Time)

Get the time step size.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The time step size.

"""
get_dt(T::Time)::Float32 = T.dt

"""
    get_interval(T::Time)

Get the time interval from 0 to the current time.

# Arguments
- `T::Time`: The Time object.

# Returns
- `StepRange{Float32}`: The time interval.

"""
get_interval(T::Time) = Float32(T.dt):Float32(T.dt):get_time(T)

"""
    update_time!(T::Time, dt::Float32)

Update the current time and time step.

# Arguments
- `T::Time`: The Time object.
- `dt::Float32`: The time step size.

"""
function update_time!(T::Time, dt::Float32)
    T.t[1] += dt
    T.tt[1] += 1
end

function update_time!(T::Time, myT::Time)
    T.t[1] = myT.t[1]
    T.tt[1] = myT.tt[1]
end

function reset_time!(T::Time)
    T.t[1] = 0.0f0
    T.tt[1] = 0
end

function reset_time!(model::NamedTuple)
    model.time.t[1] = 0.0f0
    model.time.tt[1] = 0
end

"""
    record_fire!(obj::PT, T::Time, indices::Dict{Symbol,Vector{Int}}) where {PT <: Union{AbstractPopulation, AbstractStimulus}}

Record the firing activity of the `obj` object into the `obj.records[:fire]` array.

# Arguments
- `obj::PT`: The object to record the firing activity from.
- `T::Time`: The time at which the recording is happening.
- `indices::Dict{Symbol,Vector{Int}}`: A dictionary containing indices for each variable to record.

"""
function record_fire!(
    fire::Vector{Bool},
    record::Dict{Symbol,AbstractVector},
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
)
    # @unpack fire = obj
    # @unpack records = obj
    sum(fire) == 0 && return
    ind::Vector{Int} = haskey(indices, :fire) ? indices[:fire] : collect(eachindex(fire))
    t::Float32 = get_time(T)
    push!(record[:time], t)
    push!(record[:neurons], findall(fire[ind]))
end

"""
    record_sym!(obj, key::Symbol, T::Time, indices::Dict{Symbol,Vector{Int}})

Record the variable `key` of the `obj` object into the `obj.records[key]` array.

# Arguments
- `obj`: The object to record the variable from.
- `key::Symbol`: The key of the variable to record.
- `T::Time`: The time at which the recording is happening.
- `indices::Dict{Symbol,Vector{Int}}`: A dictionary containing indices for each variable to record.

"""
function record_sym!(
    my_record,
    obj,
    key::Symbol,
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
    sr::Float32,
)
    !record_step(T, sr) && return
    ind::Vector{Int} = haskey(indices, key) ? indices[key] : axes(my_record, 1)
    @inbounds _record_sym(my_record, obj.records[key], ind)
end

@inline function _record_sym(
    my_record::Vector{T},
    records::Vector{Vector{T}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind])
end

@inline function _record_sym(
    my_record::T,
    records::Vector{T},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record)
end

@inline function _record_sym(
    my_record::Array{T,3},
    records::Vector{Array{T,3}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind, :, :])
end

@inline function _record_sym(
    my_record::Vector{Vector{T}},
    records::Vector{Vector{Vector{T}}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, deepcopy(my_record[ind]))
end

@inline function _record_sym(
    my_record::Matrix{T},
    records::Vector{Matrix{T}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind, :])
end

@inline function record_step(T, sr)
    period = max(1, floor(Int, 1.0f0 / sr / get_dt(T)))
    (get_step(T) % period) == 0
end

"""
    record!(obj, T::Time)

Record the state of the object `obj` at the current time `T`.
"""
@inline function get_model_field(obj, key::Symbol, var_map::Dict{Symbol,Tuple{Symbol,Symbol}})
    if haskey(var_map, key)
        parent, sub = var_map[key]
        return getfield(getfield(obj, parent), sub)
    else
        return getfield(obj, key)
    end
end

function record!(obj, T::Time)
    @unpack records = obj
    !haskey(records, :data) && return
    meta = records[:meta]
    time = get_time(T)
    for key::Symbol in records[:data]
        isnan(get(records[:start_time], key, NaN32)) && (records[:start_time][key] = time)
        records[:end_time][key] = time
        if key === :fire
            record_fire!(obj.fire, records[:fire], T, records[:indices])
        else
            record_sym!(
                get_model_field(obj, key, meta[:var_map]),
                obj, key, T, records[:indices], records[:sr][key],
            )
        end
    end
end

"""
    monitor!(obj::Item, keys::Vector; sr = 1000Hz, variables::Symbol = :none) where {Item<:Union{AbstractPopulation,AbstractStimulus,AbstractConnection}}

Initialize monitoring for specified variables in an object.

# Arguments
- `obj::Item`: The object to monitor (must be a population, stimulus, or connection).
- `keys::Vector`: A vector of symbols or tuples specifying variables to monitor.
  - If a symbol is provided, it specifies the variable to monitor.
  - If a tuple is provided, the first element is the variable symbol and the second is a list of indices to monitor.
- `sr::Float32`: The sampling rate for recording (default: 1000Hz).
- `variables::Symbol`: The variable group to monitor (default: :none). If specified, monitors variables within this group.

# Details
This function sets up monitoring for the specified variables in the object. It initializes necessary recording structures if they don't exist, and configures the sampling rate and indices for each variable to be monitored.

For firing activity (:fire), it creates a dictionary to store spike times and neuron indices. For other variables, it determines the appropriate data type and creates a vector to store the recorded values.

# Notes
- If a variable is not found in the object, a warning is issued.
- If a variable is already being monitored, a warning is issued.
- The function handles both direct object fields and nested fields within variable groups.
"""
# Keys created by _init_records!. _clear skips these (they are schema, not data).
# Add here when adding a new metadata key to _init_records!.
const _RECORD_META_KEYS = (:indices, :sr, :variables, :data, :meta)

function _init_records!(records::Dict)
    haskey(records, :indices)    || (records[:indices]    = Dict{Symbol,Vector{Int}}())
    haskey(records, :sr)         || (records[:sr]         = Dict{Symbol,Float32}())
    haskey(records, :variables)  || (records[:variables]  = Vector{Symbol}())
    haskey(records, :start_time) || (records[:start_time] = Dict{Symbol,Float32}())
    haskey(records, :end_time)   || (records[:end_time]   = Dict{Symbol,Float32}())
    haskey(records, :data)       || (records[:data]       = Symbol[])
    haskey(records, :meta)       || (records[:meta]       = Dict{Symbol,Any}(
        :var_map => Dict{Symbol,Tuple{Symbol,Symbol}}(),
    ))
    return records
end

function monitor!(
    obj::Item,
    keys::Vector;
    sr = 1000Hz,
    variables::Symbol = :none,
    verbose = false
) where {Item<:Union{AbstractPopulation,AbstractStimulus,AbstractConnection}}
    _init_records!(obj.records)
    ## If the key is a tuple, then the first element is the symbol and the second element is the list of neurons to record.
    for key in keys
        sym, ind = isa(key, Tuple) ? key : (key, [])
        if sym == :fire
            if haskey(obj.records, :fire)
                verbose && @warn "Field :fire already being monitored in $(obj.name)"
                :fire ∉ obj.records[:data] && push!(obj.records[:data], :fire)
                continue
            end
            obj.records[:fire] = Dict{Symbol,AbstractVector}(
                :time => Vector{Float32}(),
                :neurons => Vector{Vector{Int}}(),
            )
            obj.records[:start_time][:fire] = NaN32
            @debug "Monitoring :fire in $(obj.name)"
            :fire ∉ obj.records[:data] && push!(obj.records[:data], :fire)
            continue
        end
        if variables == :none
            if hasfield(typeof(obj), sym)
                typ = typeof(getfield(obj, sym))
                key = sym
            else
                verbose && @warn "Field $sym not found in $(nameof(typeof(obj)))"
                continue 
            end
        else
            if hasproperty(obj, variables)
                if hasproperty(getfield(obj, variables), sym)
                    typ = typeof(getfield(getfield(obj, variables), sym))
                    key = Symbol(variables, "_", sym)
                    if !(variables ∈ obj.records[:variables])
                        @debug "Monitoring $(variables)"
                        push!(obj.records[:variables], variables)
                    end
                else
                    verbose && @warn "Field $sym not found in $(nameof(typeof(getfield(obj, variables))))"
                    continue
                end
            else
                verbose && @warn "Field $variables not found in $(nameof(typeof(obj)))"
                continue
            end
        end
        @debug "Monitoring :$(key) in $(obj.name)"

        if haskey(obj.records, key)
            verbose && @warn "Key $key already being monitored in $(obj.name)"
            continue
        end
        !isempty(ind) && (obj.records[:indices][key] = ind)
        obj.records[:sr][key] = sr
        obj.records[key] = Vector{typ}()
        push!(obj.records[:data], key)
        obj.records[:start_time][key] = NaN32
        if variables != :none
            obj.records[:meta][:var_map][key] = (variables, sym)
        end
    end
end


function monitor!(objs::Array, keys::Vector; sr = 200Hz, kwargs...)
    for obj in objs
        monitor!(obj, keys, sr = sr; kwargs...)
    end
end

function monitor!(objs::NamedTuple, keys::Vector; sr = 200Hz, kwargs...)
    for obj in values(objs)
        monitor!(obj, keys, sr = sr; kwargs...)
    end
end

monitor!(obj, keys::Symbol; kwargs...) = monitor!(obj, [keys]; kwargs...)

monitor!(obj, keys::Tuple; kwargs...) = monitor!(obj, [keys]; kwargs...)

monitor!(objs, keys, variables::Symbol; kwargs...) =
    monitor!(objs, keys; variables = variables, kwargs...)

"""
    interpolated_record(p, sym)

    Returns the recording with interpolated time values and the extrema of the recorded time points.

    N.B. 
    ----
    The element can be accessed at whichever time point by using the index of the array. The time point must be within the range of the recorded time points, in r_v.
"""
function interpolated_record(p, sym, τ = 20ms)
    if sym == :fire
        return firing_rate(p, τ = 20ms)
    end
    v_dt = getvariable(p, sym)

    # Set NoInterp in the singleton dimensions:
    interp = get_interpolator(v_dt)
    v = interpolate(v_dt, interp)
    # Last dimension is time
    ax = map(1:(length(size(v_dt))-1)) do i
        axes(v_dt, i)
    end

    r_v = get_measure_interval(p, sym, size(v_dt, ndims(v_dt)))
    y = scale(v, ax..., r_v)
    return y, r_v
end

function get_measure_interval(p::AbstractComponent, sym::Symbol, steps::Int)
    _start = p.records[:start_time][sym]
    _end = p.records[:end_time][sym]
    return range(_start, _end, steps)
end

function get_measure_interval(
    p::AbstractComponent,
    sym::Symbol,
    step::R,
) where {R<:Union{Float32,Float64}}
    _start = p.records[:start_time][sym]
    _end = p.records[:end_time][sym]
    return range(_start, _end, step = step)
end



function add_endtime!(model::NamedTuple)
    @assert isa_model(model) "Model is not a valid NetworkModel"
    time = model.time
    for obj in values(model)
        obj isa String && continue
        obj isa Time && continue
        for v in obj
            if v isa AbstractPopulation ||
               v isa AbstractStimulus ||
               v isa AbstractConnection
                # @info "Adding end time for $(v.name)"
                !haskey(v.records, :end_time) &&
                    (v.records[:end_time] = Dict{Symbol,Float32}())
                for key in get(v.records, :data, Symbol[])
                    if !haskey(v.records[:end_time], key)
                        v.records[:end_time][key] = get_time(time)
                    end
                end
            end
        end
    end
end

function add_starttime!(model::NamedTuple)
    @assert isa_model(model) "Model is not a valid NetworkModel"
    for obj in values(model)
        obj isa String && continue
        obj isa Time && continue
        for v in obj
            if v isa AbstractPopulation ||
               v isa AbstractStimulus ||
               v isa AbstractConnection
                # @info "Adding start time for $(v.name)"
                !haskey(v.records, :start_time) &&
                    (v.records[:start_time] = Dict{Symbol,Float32}())
                for key in get(v.records, :data, Symbol[])
                    if !haskey(v.records[:start_time], key)
                        v.records[:start_time][key] = 0
                    end
                end
            end
        end
    end
end


function squeeze(A::AbstractArray)
    singleton_dims = tuple((d for d = 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims = singleton_dims)
end

function get_interpolator(A::AbstractArray)
    singleton_dims = tuple((d for d = 1:ndims(A) if size(A, d) == 1)...)
    interp = repeat(Vector{Any}([BSpline(Linear())]), ndims(A))
    for d in singleton_dims
        interp[d] = NoInterp()
    end
    return Tuple(interp)
end

"""
    record(p, sym::Symbol; range = false, interval = nothing, kwargs...)

Record data from a population `p` based on the specified symbol `sym`.

# Arguments
- `p`: The population from which to record data.
- `sym::Symbol`: The type of data to record. Valid options are `:fire` for firing rate, `:spiketimes` or `:spikes` for spike times.
- `range::Bool=false`: If `true`, return both the recorded data and the range. Default is `false`.
- `interval`: The time interval for recording. Required for firing rate recording (`sym = :fire`).
- `kwargs...`: Additional keyword arguments to pass to the recording function.

# Returns
- If `sym = :fire` and `range = true`, returns a tuple `(v, r)` where `v` is the firing rate and `r` is the range.
- If `sym = :fire` and `range = false`, returns the firing rate `v`.
- If `sym = :spiketimes` or `sym = :spikes`, returns the spike times.
- For other symbols, returns a tuple `(v, r)` if `range = true`, or `v` if `range = false`.

# Examples
```julia
# Record firing rate for a population p over a specific interval
v = record(p, :fire; interval = (0.0, 1.0))

# Record firing rate and range for a population p over a specific interval
v, r = record(p, :fire; range = true, interval = (0.0, 1.0))

# Record spike times for a population p
spikes = record(p, :spiketimes)
```
"""
function record(
    p::C,
    sym::Symbol;
    range = false,
    interval = nothing,
    interpolate = true,
    variables=  nothing,
    kwargs...,
) where {C<:Component}
    if sym == :fire
        @assert !isnothing(interval) "Range must be provided for firing rate recording"
        v, r = firing_rate(p, interval; interpolate, kwargs...)
    elseif sym == :spiketimes || sym == :spikes
        return spiketimes(p)
    else
        sym = isnothing(variables) ? sym : Symbol(variables, "_", sym)
        # not interpolate
        if !interpolate
            v = getvariable(p, sym)
            r = interval
            # interpolate
        else
            v, r = interpolated_record(p, sym)
            if !isnothing(interval)
                @assert interval[1] .>= r[1] "Interval start $(interval[1]) is out of bounds $(r_v[1])"
                @assert interval[end] .<= r[end] "Interval end $(interval[end]) is out of bounds $(r[end])"
                v_dt = v(axes(v, 1), interval)
                r = interval
                ax = map(i -> axes(v_dt, i), 1:(length(size(v))-1))
                v = scale(
                    Interpolations.interpolate(v_dt, get_interpolator(v_dt)),
                    ax...,
                    r,
                )
            end
        end
    end
    if range
        return v, r
    else
        return v
    end
end

function record(pops, sym::Symbol; interval=nothing, interpolate=true, kwargs...)
    @assert !isnothing(interval) "Interval must be provided for recording of multiple populations"
    v_dt = map(pops) do pop
        !haskey(pop.records, sym) && return fill(0.0f0, (0, length(interval)))
        rr = record(pop, sym; interval, range = false, interpolate, kwargs...)
        rr(axes(rr, 1), interval)
    end |> x-> vcat(x...)
    r = interval
    ax = map(i -> axes(v_dt, i), 1:(length(size(v_dt))-1))
    scale(
        Interpolations.interpolate(v_dt, get_interpolator(v_dt)),
        ax...,
        r,
    )
end


record(p::T, sym::Symbol, interval::R; kwargs...) where {R<:AbstractRange, T<:Union{Component, Vector{<:Component}}} =
    record(p, sym; interval, kwargs...)



"""
getvariable(obj, key, id=nothing)

Returns the recorded values for a given object and key. If an id is provided, returns the recorded values for that specific id.
"""
function getvariable(obj, key, id = nothing)
    rec = getrecord(obj, key)
    if isa(rec[1], Matrix)
        @debug "Matrix recording"
        array = zeros(size(rec[1])..., length(rec))
        for i in eachindex(rec)
            array[:, :, i] = rec[i]
        end
        return array
    elseif typeof(rec[1]) <: Vector{Vector{typeof(rec[1][1][1])}} # it is a multipod
        @debug "Multipod recording"
        i = length(rec)
        n = length(rec[1])
        d = length(rec[1][1])
        array = zeros(d, n, i)
        for i in eachindex(rec)
            for n in eachindex(rec[i])
                array[:, n, i] = rec[i][n]
            end
        end
        return array
    else
        @debug "Vector recording"
        isnothing(id) && return hcat(rec...)
        return hcat(rec...)[id, :]
    end
end

"""
getrecord(p, sym)

Returns the recorded values for a given object and symbol. If the symbol is not found in the object's records, it checks the records of the object's plasticity and returns the values for the matching symbol.
"""
function getrecord(p, sym)
    haskey(p.records, sym) && return p.records[sym]
    throw(ArgumentError("The record $sym is not found"))
end

"""
clear_records!(obj)

Clears all the records of a given object.
"""
function clear_records!(obj)
    if obj isa AbstractPopulation || obj isa AbstractStimulus || obj isa AbstractConnection
        _clear(obj.records)
    else
        for v in obj
            if v isa AbstractPopulation ||
               v isa AbstractStimulus ||
               v isa AbstractConnection 
                @debug "Removing records from $(v.name)"
                _clear(v.records)
            elseif v isa AbstractGroup
                for g in v.elements
                    clear_records!(g)
                end
            elseif v isa String
                continue
            elseif v isa Time
                continue
            else
                clear_records!(v)
            end
        end
    end

end

function _clear(z)
    for (key, val) in z
        key ∈ _RECORD_META_KEYS && continue
        if key == :start_time || key == :end_time
            empty!(val)  # reset time bounds so next sim uses fresh start/end
            continue
        end
        if isa(val, Dict)
            _clear(val)
        else
            try
                empty!(val)
            catch e
                # @warn "Could not clear records for $key"
            end
        end
    end
end

"""
clear_records!(obj, sym::Symbol)

Clears the records of a given object for a specific symbol.
"""
function clear_records!(obj, sym::Symbol)
    for (key, val) in obj.records
        (key == sym) && (empty!(val))
    end
end

"""
clear_records!(objs::AbstractArray)

Clears the records of multiple objects.
"""
function clear_records!(objs::AbstractArray)
    for obj in objs
        clear_records!(obj)
    end
end


"""
clear_monitor!(obj)

Clears all the records of a given object.
"""
function clear_monitor!(obj)
    for (k, val) in obj.records
        delete!(obj.records, k)
    end
end

function clear_monitor!(objs::NamedTuple)
    for obj in values(objs)
        try
            clear_monitor!(obj)
        catch
            typeof(obj) == String && continue
            typeof(obj) == Time && continue
            @warn "Could not clear monitor for $(nameof(typeof(obj))): $obj"
        end
    end
end


export Time,
    get_time,
    get_step,
    get_dt,
    get_interval,
    update_time!,
    record_plast!,
    record_fire!,
    record_sym!,
    record!,
    monitor!,
    getvariable,
    getrecord,
    clear_records!,
    clear_monitor!,
    record,
    reset_time!,
    interpolated_record,
    get_measure_interval,
    add_endtime!,
    add_starttime!
