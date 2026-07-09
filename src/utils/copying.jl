# This file is a part of Julia. License is MIT: https://julialang.org/license

# deep copying

# Note: modelcopy_internal(::Any, ::IdDict) is
#       only exposed for specialization by libraries

"""
    modelcopy(x)

Create a deep copy of `x`: everything is copied recursively, resulting in a fully
independent object. For example, deep-copying an array creates deep copies of all
the objects it contains and produces a new array with the consistent relationship
structure (e.g., if the first two elements are the same object in the original array,
the first two elements of the new array will also be the same `modelcopy`ed object).
Calling `modelcopy` on an object should generally
have the same effect as serializing and then deserializing it.

While it isn't normally necessary, user-defined types can override the default `modelcopy`
behavior by defining a specialized version of the function
`modelcopy_internal(x::T, dict::IdDict)` (which shouldn't otherwise be used),
where `T` is the type to be specialized for, and `dict` keeps track of objects copied
so far within the recursion. Within the definition, `modelcopy_internal` should be used
in place of `modelcopy`, and the `dict` variable should be
updated as appropriate before returning.

!!! warning
    It is better to avoid this function in favor of custom `copy` methods or use-case-specific
    copying functions. `modelcopy` is slow and can easily copy too many objects, or generate an
    object that violates invariants, since it does not respect abstraction boundaries.
"""
function modelcopy(@nospecialize x)
    isbitstype(typeof(x)) && return x
    return modelcopy_internal(x, IdDict())::typeof(x)
end

modelcopy_internal(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,UnionAll,Task,Regex},
                  stackdict::IdDict) = x
modelcopy_internal(x::Tuple, stackdict::IdDict) =
    ntuple(i->modelcopy_internal(x[i], stackdict), length(x))
modelcopy_internal(x::Module, stackdict::IdDict) = error("modelcopy of Modules not supported")

function modelcopy_internal(x::Base.SimpleVector, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]::typeof(x)
    end
    y = Core.svec(Any[modelcopy_internal(x[i], stackdict) for i = 1:length(x)]...)
    stackdict[x] = y
    return y
end

function modelcopy_internal(x::String, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]::typeof(x)
    end
    y = GC.@preserve x unsafe_string(pointer(x), sizeof(x))
    stackdict[x] = y
    return y
end

function modelcopy_internal(@nospecialize(x), stackdict::IdDict)
    T = typeof(x)::DataType
    nf = nfields(x)
    if ismutable(x)
        if haskey(stackdict, x)
            return stackdict[x]::typeof(x)
        end
        y = ccall(:jl_new_struct_uninit, Any, (Any,), T)
        stackdict[x] = y
        for i in 1:nf
            if isdefined(x, i)
                xi = getfield(x, i)
                if !isbits(xi)
                    xi = modelcopy_internal(xi, stackdict)::typeof(xi)
                end
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i-1, xi)
            end
        end
    elseif nf == 0 || isbitstype(T)
        y = x
    else
        flds = Vector{Any}(undef, nf)
        for i in 1:nf
            if isdefined(x, i)
                xi = getfield(x, i)
                if !isbits(xi)
                    xi = modelcopy_internal(xi, stackdict)::typeof(xi)
                end
                flds[i] = xi
            else
                nf = i - 1 # rest of tail must be undefined values
                break
            end
        end
        y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, flds, nf)
    end
    return y::T
end

function modelcopy_internal(x::Memory, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]::typeof(x)
    end
    _modelcopy_memory_t(x, eltype(x), stackdict)
end

function _modelcopy_memory_t(@nospecialize(x::Memory), T, stackdict::IdDict)
    if isbitstype(T)
        return (stackdict[x]=copy(x))
    end
    dest = typeof(x)(undef, length(x))
    stackdict[x] = dest
    xr = memoryref(x)
    dr = memoryref(dest)
    for i = 1:length(x)
        xi = Core.memoryrefnew(xr, i, false)
        if Core.memoryref_isassigned(xi, :not_atomic, false)
            xi = Core.memoryrefget(xi, :not_atomic, false)
            if !isbits(xi)
                xi = modelcopy_internal(xi, stackdict)::typeof(xi)
            end
            di = Core.memoryrefnew(dr, i, false)
            Core.memoryrefset!(di, xi, :not_atomic, false)
        end
    end
    return dest
end
function modelcopy_internal(x::Array{T, N}, stackdict::IdDict) where {T, N}
    if haskey(stackdict, x)
        return stackdict[x]::typeof(x)
    end
    y = stackdict[x] = Array{T, N}(undef, ntuple(Returns(0), Val{N}()))
    setfield!(y, :ref, modelcopy_internal(x.ref, stackdict))
    setfield!(y, :size, x.size)
    y
end
function modelcopy_internal(x::GenericMemoryRef, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]::typeof(x)
    end
    mem = getfield(x, :mem)
    dest = memoryref(modelcopy_internal(mem, stackdict)::typeof(mem))
    i = Base.memoryrefoffset(x)
    i == 1 || (dest = Core.memoryrefnew(dest, i, true))
    return dest
end


function modelcopy_internal(x::Union{Dict,IdDict}, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]::typeof(x)
    end

    if isbitstype(eltype(x))
        return (stackdict[x] = copy(x))
    end

    dest = empty(x)
    stackdict[x] = dest
    for (k, v) in x
        if haskey(x, :data)
            # Records dict: preserve schema keys, wipe timing and data buffers.
            if k === :start_time || k === :end_time
                # Empty timing dicts — stale warmup timestamps must not survive into the copy.
                dest[k] = Dict{Symbol,Float32}()
                continue
            end
            if k in x[:data]
                if k === :fire
                    dest[modelcopy_internal(k, stackdict)] = Dict{Symbol,AbstractVector}(
                        :time        => Vector{Float32}(),
                        :neurons     => Vector{Vector{Int}}(),
                        :times_buf   => Vector{Float32}(),
                        :neurons_buf => Vector{Int}(),
                    )
                else
                    dest[modelcopy_internal(k, stackdict)] = Vector{Float32}()
                end
                continue
            end
        end
        dest[modelcopy_internal(k, stackdict)] = modelcopy_internal(v, stackdict)
    end
    dest
end

function modelcopy_internal(x::Base.AbstractLock, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]::typeof(x)
    end
    y = typeof(x)()
    stackdict[x] = y
    return y
end

function modelcopy_internal(x::Base.GenericCondition, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]::typeof(x)
    end
    y = typeof(x)(modelcopy_internal(x.lock, stackdict))
    stackdict[x] = y
    return y
end

export modelcopy    