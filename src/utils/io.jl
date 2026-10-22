import DrWatson: save, load

"""
    SNNfolder(path, name, info)

Folder in which `SNNsave` stores a model: `joinpath(path, savename(name, info, connector = "-"))`
(DrWatson `savename`, so the folder name encodes the `info` parameters).

# Arguments
- `path`: Base directory path
- `name`: Model name
- `info`: NamedTuple with model metadata

# Returns
- String path to the model folder
"""
function SNNfolder(path, name, info)
    return joinpath(path, savename(name, info, connector = "-"))
end

"""
    SNNfile(type, count::Int, suffix = "")

File name used by `SNNsave`/`SNNload`: `"\$(type)-\$(count)-\$(suffix).jld2"` for `count > 0`,
`"\$(type)-\$(suffix).jld2"` for `count == 0` (e.g. `"model-.jld2"`, `"data-2-trial.jld2"`).
"""
function SNNfile(type, count::Int, suffix="")
    count_string = count > 0 ? "-$(count)" : ""
    return "$(type)$(count_string)-$(suffix).jld2"
end

"""
    SNNpath(path, name, info, type, count)

Full path `joinpath(SNNfolder(path, name, info), SNNfile(type, count))` of a stored file
(empty suffix).

# Arguments
- `path`: Base directory path
- `name`: Model name
- `info`: NamedTuple with model metadata
- `type`: Type of file (:model, :data, etc.)
- `count`: File counter

# Returns
- Complete file path string
"""
function SNNpath(path, name, info, type, count)
    return joinpath(SNNfolder(path, name, info), SNNfile(type, count))
end

"""
    check_loaded_model(x; what = "file")

Throw an `ArgumentError` if `x` (a loaded model, params or `NamedTuple`/`Dict` of them) contains
a `JLD2.ReconstructedMutable` or `JLD2.ReconstructedStatic` placeholder, searching recursively
through tuples, dictionaries, arrays of non-bits elements and struct fields. JLD2 builds such
placeholders when the file holds a type whose layout no longer exists, e.g. a `model` saved with SNNModels <= 1.9.1;
such a model loads without error and then fails later with an unrelated message. No migration is attempted. Returns `x`.
"""
function check_loaded_model(x; what::AbstractString = "file")
    seen = Base.IdSet{Any}()
    _check_reconstructed(x, seen, what, "")
    return x
end

# the deepest placeholder below `x`: names the layout that changed rather than its container
function _innermost_placeholder(x)
    for v in getfield(x, :fields)
        (v isa JLD2.ReconstructedMutable || v isa JLD2.ReconstructedStatic) && return _innermost_placeholder(v)
    end
    return x
end

function _check_reconstructed(x, seen, what, path)
    if x isa JLD2.ReconstructedMutable || x isa JLD2.ReconstructedStatic
        x = _innermost_placeholder(x)
        tn = string(typeof(x).parameters[1])
        length(tn) > 90 && (tn = first(tn, 87) * "...")
        throw(ArgumentError(
            "The $what holds an object of the type `$tn` that JLD2 could not rebuild" *
            (isempty(path) ? "" : " (at $path)") * ": it was saved with an older SNNModels " *
            "(<= 1.9.1) layout that this version cannot read. Rebuild the model from its " *
            "configuration, or open the file with SNNModels 1.9.1.",
        ))
    end
    (x === nothing || x isa Union{Number,AbstractString,Symbol,Function,Type,Module}) && return
    x isa AbstractArray && isbitstype(eltype(x)) && return
    ismutable(x) && (x in seen ? (return) : push!(seen, x))
    if x isa AbstractDict
        for (k, v) in x
            _check_reconstructed(v, seen, what, string(path, "[", repr(k), "]"))
        end
    elseif x isa AbstractArray || x isa Tuple
        for (i, v) in enumerate(x)
            _check_reconstructed(v, seen, what, string(path, "[", i, "]"))
        end
    elseif isstructtype(typeof(x))
        for f in fieldnames(typeof(x))
            isdefined(x, f) || continue
            _check_reconstructed(getfield(x, f), seen, what, string(path, ".", f))
        end
    end
    return
end

"""
    SNNload(; path, name = "", info = nothing, count = 0, suffix = "", type = :model)
    SNNload(path, name = "", info = nothing)

Load a file written by `SNNsave`.

If `path` is a file, it is loaded directly. Otherwise the file
`joinpath(SNNfolder(path, name, info), SNNfile(type, count, suffix))` is loaded.

# Arguments
- `path::String`: file, or base directory used in `SNNsave`.
- `name::String`, `info`: model name and metadata `NamedTuple` (required if `path` is not a file).
- `count::Int = 0`: file counter.
- `suffix = ""`: file suffix.
- `type::Symbol = :model`: `:model` (records cleared) or `:data` (with records).

# Returns
- The stored dictionary as a `NamedTuple` (`(model = ..., config = ..., extra keys...)`), or
  `nothing` (with an `@error` message) if the file does not exist.

# Throws
- `ArgumentError` if `path` is not a file and `name` or `info` is missing.

# Example
```julia
using SpikingNeuralNetworks
@load_units
E = SNN.IF(N = 10, name = "E")
model = SNN.compose(; E, silent = true)
dir = mktempdir()
SNN.save_model(; model, path = dir, name = "toy", info = (seed = 1,))
loaded = SNN.load_model(dir, "toy", (seed = 1,))
loaded.model.pop.E.N   # 10
```
"""
function SNNload(;
    path::String,
    name::String = "",
    info = nothing,
    count::Int = 0,
    suffix = "",
    type::Symbol = :model,
)
    ## Check if path is a directory
    if isfile(path)
        @info "Loading $(path)"
        return dict2ntuple(check_loaded_model(DrWatson.load(path); what = "file $(path)"))
    else
        if isempty(name) || isnothing(info)
            throw(
                ArgumentError(
                    "If path is not file, `name::String`` and `info::NamedTuple` are required",
                ),
            )
        end
        root = path
    end

    root = SNNfolder(path, name, info)
    file = joinpath(root, SNNfile(type, count, suffix))
    if !isfile(file)
        @error "File not found: $file"
        return nothing
    end


    tic = time()
    DATA = JLD2.load(file)
    @info "$type $(name)"
    @info "Loading time:  $(round(time()-tic, digits=2)) seconds"
    check_loaded_model(DATA; what = "$type file $(file)")
    return dict2ntuple(DATA)
end

SNNload(path::String, name::String = "", info = nothing; kwargs...) =
    SNNload(; path = path, name = name, info = info, type = :model, kwargs...)

"""
    load_model(path, name, info; kwargs...)

Load the `:model` file (records cleared) stored by `SNNsave`; same as
`SNNload(; path, name, info, kwargs..., type = :model)`. Returns `nothing` if the file is missing.

# Arguments
- `path::String`: Directory path
- `name::String`: Model name
- `info::NamedTuple`: Model metadata
- `kwargs...`: Additional arguments passed to SNNload

# Returns
- Loaded model as NamedTuple
"""
load_model(path::String, name::String, info::NamedTuple; kwargs...) =
    SNNload(; path = path, name = name, info = info, kwargs..., type = :model)

"""
    load_data(path, name, info; kwargs...)

Load the `:data` file (model with its records) stored by `SNNsave`; same as
`SNNload(; path, name, info, kwargs..., type = :data)`.

# Arguments
- `path::String`: Directory path  
- `name::String`: Model name
- `info::NamedTuple`: Model metadata
- `kwargs...`: Additional arguments passed to SNNload

# Returns
- Loaded data as NamedTuple
"""
load_data(path::String, name::String, info::NamedTuple; kwargs...) =
    SNNload(; path = path, name = name, info = info, kwargs..., type = :data)
load_data(path, name, info; kwargs...) = SNNload(; path, name, info, kwargs..., type = :data)

"""
    load_or_run(f; path, name, info, exp_config...)

Return `load_model(path, name, info)` if that file exists; otherwise call `f(info)`, save the
result with `save_model` and return it.

The model is saved in the folder that `load_model(path, name, info)` reads, so a second call
loads it. (Up to SNNModels 1.8.4 `name` was replaced by `savename(name, info)` before saving,
so the model was written to another folder and never found.)

# Arguments
- `f::Function`: Function to run if model doesn't exist (receives `info` as argument)
- `path`: Directory path
- `name`: Model name  
- `info`: Model metadata
- `exp_config...`: Configuration passed to save_model if running

# Returns
- Loaded or newly generated model
"""
function load_or_run(f::Function; path, name, info, exp_config...)
    loaded = load_model(path, name, info)
    if isnothing(loaded)
        @info "Running simulation for: $(savename(name, info, connector = "-"))"
        produced = f(info)
        save_model(model = produced, path = path, name = name, info = info, exp_config...)
        return produced
    end
    return loaded
end




"""
    SNNsave(model; path, name, info, suffix = "", config = nothing, type = :all, count = 0, kwargs...)

Save a model to `SNNfolder(path, name, info)` with JLD2 (through `DrWatson.save`).

# Arguments
- `model`: the model to save.
- `path`, `name`, `info`: base directory, model name and metadata `NamedTuple`; the folder name is
  `savename(name, info, connector = "-")`.
- `suffix = ""`, `count = 0`: file suffix and counter, see `SNNfile`.
- `config = nothing`: configuration stored with the model and written in `config.jl`.
- `type = :all`: `:all` writes the `:data` file (model with records) and the `:model` file
  (a `deepcopy` with `clear_records!` applied); `:model` writes only the `:model` file. Any other
  value logs an error and saves nothing.
- `kwargs...`: additional entries stored in the same file.

When `count < 2`, the folder also receives a human-readable `config.jl` written by
`write_config`, with a timestamp and the git commit hash (`"unknown"` outside a git repository).

# Returns
- The path of the `:model` file.
"""
function SNNsave(
    model;
    path,
    name,
    info,
    suffix="",
    config = nothing,
    type = :all,
    count = 0,
    kwargs...,
)

    function store_data(filename, data)
        Logging.LogLevel(0) == Logging.Error
        @time DrWatson.save(filename, data)
        Logging.LogLevel(0) == Logging.Info
        @info "$type stored. It occupies $(filesize(filename) |> Base.format_bytes)"
    end

    @info "Storing $(type)-$count of `$(savename(name, info, connector="-"))`
    at $(path) \n"

    # saved weights are the current ones: materialise the deferred weight updates (vSTDPParameter; changes the later
    # trajectory at rounding level)
    _sync_model_weights!(model)

    ## Create directory if it does not exist
    root = SNNfolder(path, name, info)
    isdir(root) || mkpath(root)

    ## Write config file
    if count < 2
        write_config(joinpath(root, "config.jl"), info; config, kwargs...)
    end

    if type == :all
        type = :data
        filename = joinpath(root, SNNfile(type, count, suffix))
        data = merge((@strdict model = model config = config), kwargs)
        store_data(filename, data)

        type = :model
        _model = deepcopy(model)
        clear_records!(_model)
        filename = joinpath(root, SNNfile(type, count, suffix))
        data = merge((@strdict model = _model config = config), kwargs)
        store_data(filename, data)
        return filename
    elseif type == :model
        _model = deepcopy(model)
        clear_records!(_model)
        filename = joinpath(root, SNNfile(type, count, suffix))
        data = merge((@strdict model = _model config = config), kwargs)
        store_data(filename, data)
        return filename
    else
        @error "Unknown type: $type. Use :all, :model, or :data."
    end
end

# materialise the deferred weights of every sparse synapse of a model (see `sync_traces!`)
function _sync_model_weights!(model)
    hasproperty(model, :syn) || return model
    for c in values(model.syn)
        c isa AbstractSparseSynapse && sync_weights!(c)
    end
    return model
end

export check_loaded_model, load, save, load_model, load_data, SNNload, SNNsave, SNNpath, SNNfolder, savename

"""
    save_model(; model, path, name, info, config = nothing, kwargs...)

Save both the `:data` and the `:model` file, i.e. `SNNsave(model; path, name, info, config,
type = :all, kwargs...)`. See `SNNsave` for the file layout and `load_model`/`load_data`.

# Arguments
- `model`: The model to save
- `path`: Directory path
- `name`: Model name
- `info`: Model metadata
- `config`: Optional configuration
- `kwargs...`: Additional data to save

# Returns
- Path to the saved files
"""
save_model(; model, path, name, info, config = nothing, kwargs...) = SNNsave(
    model;
    path = path,
    name = name,
    info = info,
    config = config,
    type = :all,
    kwargs...,
)
save_model

"""
    data2model(; path, name=randstring(10), info=nothing, count = 0, suffix = "", kwargs...)

Create the `:model` file (records cleared) from an existing `:data` file, using the layout of
`SNNsave`: `SNNfolder(path, name, info)/SNNfile(:data, count, suffix)` and
`.../SNNfile(:model, count, suffix)`. (Up to SNNModels 1.8.4 it checked
`joinpath(path, savename(name, info, "data.jld2"))`, which `SNNsave` never writes.)

# Arguments
- `path`: Directory path
- `name`: Model name (default: random string)
- `info`: Model metadata

# Returns
- `true` if model file exists or was created, `false` if data file doesn't exist
"""
function data2model(; path, name = randstring(10), info = nothing, count = 0, suffix = "", kwargs...)
    root = SNNfolder(path, name, info)
    # Does data file exist? If no return false
    data_path = joinpath(root, SNNfile(:data, count, suffix))
    !isfile(data_path) && return false
    # Does model file exist? If yes return true
    model_path = joinpath(root, SNNfile(:model, count, suffix))
    isfile(model_path) && return true
    data = load_data(path, name, info; count, suffix)
    clear_records!(data.model)
    # If model file does not exist, save model file
    # Logging.LogLevel(0) == Logging.Error
    @time DrWatson.save(model_path, ntuple2dict(data))

    isfile(model_path) && return true
    @error "Model file not saved"
end

function model_path_name(; path, name = randstring(10), info = nothing, kwargs...)
    @warn " `model_path_name` is deprecated, use `SNNpath` instead"
    return SNNpath(path, name, info, :model, 0)
end

"""
    save_config(; path, name=randstring(10), config, info=nothing)

Save `config` as `joinpath(path, savename(name, info, "config.jld2", connector = "-"))`
(JLD2, key `"config"`), creating `path` if needed.

# Arguments
- `path`: Directory path
- `name`: Config name (default: random string)
- `config`: Configuration data to save
- `info`: Optional metadata

# Returns
- Nothing
"""
function save_config(; path, name = randstring(10), config, info = nothing)
    @info "Parameters: `$(savename(name, info, connector="-"))` \nsaved at $(path)"

    isdir(path) || mkpath(path)

    params_path = joinpath(path, savename(name, info, "config.jld2", connector = "-"))
    DrWatson.save(params_path, @strdict config)  # Here you are saving a Julia object to a file

    return
end

"""
    get_timestamp()

Return the current date and time (`Dates.now()`).
"""
function get_timestamp()
    return now()
end

"""
    get_git_commit_hash(; dir = pwd(), warn = true)

Get the git commit hash (`git rev-parse HEAD`) of the repository containing `dir`
(default: the working directory).

# Arguments
- `dir`: directory to run `git` in (any directory inside the work tree).
- `warn = true`: emit the one-time warning when no repository is found.

# Returns
- String containing the full commit hash, or `"unknown"` when `dir` is not inside a git repository,
  does not exist, or git is not available (e.g. cluster jobs run from a copied directory).

# Note
- Uses `git` from PATH; honours `GIT_DIR`/`GIT_WORK_TREE` if set.
- `write_config` looks in the active project directory first, then in the working directory.
"""
function get_git_commit_hash(; dir::AbstractString = pwd(), warn::Bool = true)
    try
        isdir(dir) || error("not a directory: $dir")
        return readchomp(pipeline(setenv(`git rev-parse HEAD`; dir = dir); stderr = devnull))
    catch
        warn && @warn "get_git_commit_hash: no git repository found from $dir (or git unavailable); recording \"unknown\"" maxlog = 1
        return "unknown"
    end
end

# Commit hash recorded by `write_config`: the repository of the active project (the code that produced the
# run), else the one of the working directory, else "unknown" with a warning.
function _provenance_commit_hash(commit_dir = nothing)
    dirs = String[]
    commit_dir === nothing || push!(dirs, String(commit_dir))
    proj = Base.active_project()
    proj === nothing || push!(dirs, dirname(proj))
    push!(dirs, pwd())
    for d in dirs
        h = get_git_commit_hash(; dir = d, warn = false)
        h != "unknown" && return h
    end
    return get_git_commit_hash(; dir = last(dirs), warn = true)
end

"""
    write_value(file, key, value, indent="", equal_sign="=")

Write `key = value,` to `file` as Julia source (helper of `write_config`).

# Arguments
- `file`: IO stream to write to
- `key`: Key name (empty string for array elements)
- `value`: Value to write (supports Number, String, Symbol, Tuple, Array, Dict, NamedTuple, etc.)
- `indent`: Indentation string (default: "")
- `equal_sign`: Assignment operator (default: "=")

# Details
- Recursively handles nested structures; other structs are written as `TypeName(field = ..., ...)`.
- Formats different types appropriately (quoted strings, symbols with :, ranges as `a:s:b`).
- In a `Dict`, non-numeric values are written as quoted strings.
"""
function write_value(file, key, value, indent = "", equal_sign = "=")
    if isa(value, Number)
        println(file, "$indent$key $(equal_sign) $value,")
    elseif isa(value, String)
        println(file, "$indent$key $(equal_sign) \"$value\",")
    elseif isa(value, Symbol)
        println(file, "$indent$key $(equal_sign) :$value,")
    elseif isa(value, Tuple)
        println(file, "$indent$key $(equal_sign) (")
        for v in value
            write_value(file, "", v, indent * "    ", "")
        end
        println(file, "$indent),")
    elseif typeof(value) <: AbstractRange || isa(value, StepRange{Int64,Int64})
        _s = step(value)
        _end = last(value)
        _start = first(value)
        println(file, "$indent$key $(equal_sign) $(_start):$(_s):$(_end),")
    elseif isa(value, Bool)
        println(file, "$indent$key $(equal_sign) $value,")
    elseif isa(value, Array)
        println(file, "$indent$key $(equal_sign) [")
        for v in value
            write_value(file, "", v, indent * "    ", "")
        end
        println(file, "$indent],")
    elseif isa(value, Dict)
        println(file, "$indent$key = Dict(")
        for (k, v) in value
            if isa(v, Number)
                println(file, "$indent    :$k => $v")#$(write_value(file,"",v,"", ""))")
            else
                isa(v, String)
                println(file, "$indent    :$k => \"$v\",")
            end
            # else
            #     # println(file, "$indent    $k => $v,")
            #     write_value(file, k, v, indent * "    ")
            # end
        end
        println(file, "$indent),")
    else
        isa(value, NamedTuple)
        name = isa(value, NamedTuple) ? "" : nameof(typeof(value))
        println(file, "$indent$key $equal_sign $(name)(")
        for field in fieldnames(typeof(value))
            field_value = getfield(value, field)
            write_value(file, field, field_value, indent * "    ")
        end
        println(file, "$indent),")
    end
end

"""
    write_config(path, info; config, name="", commit_dir=nothing, kwargs...)

Write `info` (and `config` if not `nothing`) as Julia source to a text file, preceded by the
generation timestamp and the git commit hash (`get_git_commit_hash`, `"unknown"` outside a git
repository).

# Arguments
- `path::String`: file path; if `name` is given, the file is
  `joinpath(path, savename(name, info, "config", connector = "-"))` instead.
- `info`: `NamedTuple` written as `info = (...)`.
- `config`: `NamedTuple` written as `config = (...)`, or `nothing`.
- `commit_dir = nothing`: directory searched first for the git repository whose commit is recorded;
  then the directory of the active project, then the working directory. `"unknown"` (with a warning) if none is in a repository.
- `kwargs...`: accepted and ignored.

# Returns
- The path of the written file.

# Details
- Entries named `study` or `models` are skipped. (Up to SNNModels 1.8.4 only `models` was
  skipped, because of operator precedence.)
"""
function write_config(path::String, info; config, name = "", commit_dir = nothing, kwargs...)
    timestamp = get_timestamp()
    commit_hash = _provenance_commit_hash(commit_dir)

    if name !== ""
        config_path = joinpath(path, savename(name, info, "config", connector = "-"))
    else
        config_path = path
    end

    file = open(config_path, "w")

    println(file, "# Configuration file generated on: $timestamp")
    println(file, "# Corresponding Git commit hash: $commit_hash")
    println(file, "")
    println(file, "info = (")
    for (key, value) in pairs(info)
        (String(key) == "study" || String(key) == "models") && continue
        write_value(file, key, value, "    ")
    end
    println(file, ")")
    if !isnothing(config)
        println(file, "config = (")
        for (key, value) in pairs(config)
            (String(key) == "study" || String(key) == "models") && continue
            write_value(file, key, value, "    ")
        end
        println(file, ")")
    end
    # for (info_name, info_value) in pairs(kwargs)
    #     String(info_name) == "sequence" && continue
    #     if isa(info_value, NamedTuple)
    #         println(file, "$(info_name) = (")
    #         for (key, value) in pairs(info_value)
    #             write_value(file, key, value, "        ")
    #         end
    #         println(file, "    )")
    #     end
    # end
    close(file)
    @info "Config file saved"
    return config_path
end

"""
    print_summary(p)

Print the type, parameter type, `name`, `N` and every parameter field of a population `p`.
"""
function print_summary(p)
    println("Type: $(nameof(typeof(p))) $(nameof(typeof(p.param)))")
    println("  Name: ", p.name)
    println("  Number of Neurons: ", p.N)
    for k in fieldnames(typeof(p.param))
        println("   $k: $(getfield(p.param,k))")
    end
end


"""
    read_folder(path, files=nothing; my_filter = <SNNfile names of `type`>, type=:model, name=nothing)

List the files of the directory `path` for which `my_filter(file, type)` is true (logging
each match) and append their full paths to `files`.

# Arguments
- `path`: Directory path to read from
- `files`: Optional vector to append results to (default: creates new vector)
- `my_filter`: Filter function `(file, type) -> Bool`; the default matches the files written by
  `SNNsave` (`SNNfile` names `"\$(type)-....jld2"`, e.g. `model-.jld2`, `data-2-trial.jld2`) and
  the older `"...\$(type).jld2"` names. (Up to SNNModels 1.8.4 the default matched only names
  ending in `"\$(type).jld2"`, so it missed the files of `SNNsave`.)
- `type`: File type to match (default: :model)
- `name`: accepted and ignored.

# Returns
- Vector of file paths matching the filter
"""
function read_folder(
    path,
    files = nothing;
    my_filter = (file, _type) -> endswith(file, ".jld2") &&
        (startswith(file, "$(_type)-") || endswith(file, "$(_type).jld2")),
    type = :model,
    name = nothing,
)
    if isnothing(files)
        files = []
    end
    n = 0
    for file in readdir(path)
        if my_filter(file, type)
            n+=1
            @info n, file
            push!(files, joinpath(path, file))
        end
    end
    return files
end

"""
    read_folder!(df, path; type=:model, name=nothing)

Same as `read_folder(path, df; type, name)`: append the matching file paths to `df`.

# Arguments
- `df`: Vector to append results to
- `path`: Directory path to read from
- `type`: File type to match (default: :model)
- `name`: Optional name filter

# Returns
- The modified df vector
"""
function read_folder!(df, path; type = :model, name = nothing)
    read_folder(path, df; type = type, name = name)
end




export save_model,
    load_model,
    load_data,
    save_config,
    data2model,
    write_config,
    print_summary,
    load_or_run,
    read_folder,
    read_folder!
