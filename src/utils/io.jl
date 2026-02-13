import DrWatson: save, load

"""
    SNNfolder(path, name, info)

Generate folder path for SNN model storage using DrWatson's savename convention.

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
    SNNfile(type, count)

Generate filename for SNN data files.

# Arguments
- `type`: Type of file (:model, :data, etc.)
- `count`: File counter (0 for base file, >0 for numbered versions)

# Returns
- Filename string with .jld2 extension
"""
function SNNfile(type, count::Int, suffix="")
    count_string = count > 0 ? "-$(count)" : ""
    return "$(type)$(count_string)-$(suffix).jld2"
end

"""
    SNNpath(path, name, info, type, count)

Generate complete file path for SNN data files.

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
    SNNload(; path, name="", info=nothing, count=1, type=:model)

Load SNN model or data from disk.

# Arguments
- `path::String`: File path or directory
- `name::String`: Model name (required if path is directory)
- `info`: NamedTuple with model metadata (required if path is directory)
- `count::Int`: File version number (default: 1)
- `type::Symbol`: Type to load (:model or :data, default: :model)

# Returns
- Loaded data as NamedTuple

# Throws
- `ArgumentError` if path is directory but name or info is missing
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
    @info "Loading time:  $(time()-tic) seconds"
    return dict2ntuple(DATA)
end

SNNload(path::String, name::String = "", info = nothing, kwargs...) =
    SNNload(; path = path, name = name, info = info, kwargs..., type = :model)

"""
    load_model(path, name, info; kwargs...)

Load a saved model from disk.

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

Load saved data from disk.

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

Load model from disk if available, otherwise run function to generate it.

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
    SNNsave(model; path, name, info, config=nothing, type=:all, count=1, kwargs...)

Save SNN model and/or data to disk.

# Arguments
- `model`: The model to save
- `path`: Base directory path
- `name`: Model name
- `info`: NamedTuple with model metadata
- `config`: Optional configuration to save alongside model
- `type`: What to save - :all (both model and data), :model (model only), :data (data only)
- `count`: Version number for the file
- `kwargs...`: Additional data to save

# Returns
- Path to the saved file

# Details
- When type=:all, saves both model (with cleared records) and full data
- Creates config.jl file with metadata and git commit hash
- Model records are cleared before saving to reduce file size
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
    save_model(; model, path, name, info, config=nothing, kwargs...)

Convenience function to save both model and data.

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

Convert data file to model file by clearing records.

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

Save configuration to disk as JLD2 file.

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

Get current timestamp.

# Returns
- Current date and time
"""
function get_timestamp()
    return now()
end

"""
    get_git_commit_hash()

Get current git commit hash of the repository.

# Returns
- String containing the full commit hash

# Note
- Requires git to be available in PATH
"""
function get_git_commit_hash()
    return readchomp(`git rev-parse HEAD`)
end

"""
    write_value(file, key, value, indent="", equal_sign="=")

Write a value to a configuration file with proper formatting.

# Arguments
- `file`: IO stream to write to
- `key`: Key name (empty string for array elements)
- `value`: Value to write (supports Number, String, Symbol, Tuple, Array, Dict, NamedTuple, etc.)
- `indent`: Indentation string (default: "")
- `equal_sign`: Assignment operator (default: "=")

# Details
- Recursively handles nested structures
- Formats different types appropriately (quoted strings, symbols with :, etc.)
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

Write configuration and metadata to a Julia config file.

# Arguments
- `path::String`: File path or directory for config file
- `info`: NamedTuple with model metadata
- `config`: Configuration to save
- `name`: Optional name for the config file
- `kwargs...`: Additional named tuples to save

# Details
- Generates timestamped config file with git commit hash
- Creates human-readable Julia syntax output
- Skips "study" and "models" fields
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
    println(file, "config = (")
    for (key, value) in pairs(config)
        String(key) == "study" || String(key)=="models" && continue
        write_value(file, key, value, "    ")
    end
    println(file, ")")
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

    Prints a summary of the given element.
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

Read all matching files from a folder.

# Arguments
- `path`: Directory path to read from
- `files`: Optional vector to append results to (default: creates new vector)
- `my_filter`: Filter function (file, type) -> Bool (default: matches .jld2 files)
- `type`: File type to match (default: :model)
- `name`: Optional name filter

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

Read matching files from folder and append to existing vector.

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
