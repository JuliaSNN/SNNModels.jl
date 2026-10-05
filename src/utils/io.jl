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
        return dict2ntuple(DrWatson.load(path))
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
    return dict2ntuple(DATA)
end

SNNload(path::String, name::String = "", info = nothing, kwargs...) =
    SNNload(; path = path, name = name, info = info, kwargs..., type = :model)

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
load_data(path, name, info) = SNNload(; path, name, info, type = :data, kwargs...)

"""
    load_or_run(f; path, name, info, exp_config...)

Return `load_model(path, name, info)` if that file exists; otherwise call `f(info)`, save the
result with `save_model` and return it.

Note: before saving, `name` is replaced by `savename(name, info, connector = "-")`, so the file is
written to a different folder from the one `load_model(path, name, info)` looks in.

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
        name = savename(name, info, connector = "-")
        @info "Running simulation for: $name"
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

export load, save, load_model, load_data, SNNload, SNNsave, SNNpath, SNNfolder, savename

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
    data2model(; path, name=randstring(10), info=nothing, kwargs...)

Create the `:model` file (records cleared) from an existing `:data` file.

Note: the paths it checks, `joinpath(path, savename(name, info, "data.jld2"))` and
`... "model.jld2"`, do not follow the folder layout written by `SNNsave`
(`SNNfolder(path, name, info)/data-.jld2`).

# Arguments
- `path`: Directory path
- `name`: Model name (default: random string)
- `info`: Model metadata

# Returns
- `true` if model file exists or was created, `false` if data file doesn't exist
"""
function data2model(; path, name = randstring(10), info = nothing, kwargs...)
    # Does data file exist? If no return false
    data_path = joinpath(path, savename(name, info, "data.jld2", connector = "-"))
    !isfile(data_path) && return false
    # Does model file exist? If yes return true
    data = load_data(path, name, info)
    clear_records!(data.model)

    model_path = joinpath(path, savename(name, info, "model.jld2", connector = "-"))
    isfile(model_path) && return true
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
    get_git_commit_hash()

Get current git commit hash of the repository containing the working directory.

# Returns
- String containing the full commit hash, or `"unknown"` when the working directory is not inside
  a git repository or git is not available (e.g. cluster jobs run from a copied directory).
  `write_config` records this string, so a missing repository no longer aborts the run.

# Note
- Uses `git` from PATH; honours `GIT_DIR`/`GIT_WORK_TREE` if set.
"""
function get_git_commit_hash()
    try
        return readchomp(pipeline(`git rev-parse HEAD`; stderr = devnull))
    catch
        @warn "get_git_commit_hash: not inside a git repository (or git unavailable); recording \"unknown\"" maxlog = 1
        return "unknown"
    end
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
    write_config(path, info; config, name="", kwargs...)

Write `info` (and `config` if not `nothing`) as Julia source to a text file, preceded by the
generation timestamp and the git commit hash (`get_git_commit_hash`, `"unknown"` outside a git
repository).

# Arguments
- `path::String`: file path; if `name` is given, the file is
  `joinpath(path, savename(name, info, "config", connector = "-"))` instead.
- `info`: `NamedTuple` written as `info = (...)`.
- `config`: `NamedTuple` written as `config = (...)`, or `nothing`.
- `kwargs...`: accepted and ignored.

# Returns
- The path of the written file.

# Details
- Entries named `models` are skipped. (The intended skip of `study` is not effective: because of
  operator precedence, `String(key) == "study" || String(key) == "models" && continue` only skips
  `models`.)
"""
function write_config(path::String, info; config, name = "", kwargs...)
    timestamp = get_timestamp()
    commit_hash = get_git_commit_hash()

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
        String(key) == "study" || String(key)=="models" && continue
        write_value(file, key, value, "    ")
    end
    println(file, ")")
    if !isnothing(config)
        println(file, "config = (")
        for (key, value) in pairs(config)
            String(key) == "study" || String(key)=="models" && continue
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
    read_folder(path, files=nothing; my_filter=(file,_type)->endswith(file,"type.jld2"), type=:model, name=nothing)

List the files of the directory `path` for which `my_filter(file, type)` is true (logging
each match) and append their full paths to `files`.

# Arguments
- `path`: Directory path to read from
- `files`: Optional vector to append results to (default: creates new vector)
- `my_filter`: Filter function `(file, type) -> Bool`; the default matches names ending in
  `"\$(type).jld2"`, e.g. `model.jld2` (note that `SNNsave` with the default empty suffix writes
  `model-.jld2`, which the default filter does not match).
- `type`: File type to match (default: :model)
- `name`: accepted and ignored.

# Returns
- Vector of file paths matching the filter
"""
function read_folder(
    path,
    files = nothing;
    my_filter = (file, _type)->endswith(file, "$(_type).jld2"),
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
    get_path,
    data2model,
    write_config,
    print_summary,
    load_or_run,
    read_folder,
    read_folder!
