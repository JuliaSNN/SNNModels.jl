# Flush-to-zero during `sim!` / `train!`. A state decaying by a multiplicative or forward-Euler step never reaches 0: it
# ends in the subnormal range, where x86 arithmetic is about 100x slower. With flush-to-zero a would-be subnormal result
# is 0, so the variable stops at a small normal value; arithmetic above `floatmin` is unchanged. The mode is per thread.

"""
    with_flush_subnormals(f, flush::Bool = true)

Run `f()` with flush-to-zero (`set_zero_subnormals(true)`) on the calling thread and, when threading is on (see
[`neurons_per_thread`](@ref)), on the Polyester workers, then restore the previous mode of every thread. `flush = false`
runs `f()` unchanged. Used by `sim!` and `train!` (keyword `flush_subnormals`).
"""
function with_flush_subnormals(f::F, flush::Bool = true) where {F}
    flush || return f()
    old = get_zero_subnormals()
    workers = _BPC[] > 0 && Threads.nthreads() > 1
    olds = workers ? _set_ftz_workers!() : nothing
    set_zero_subnormals(true)
    try
        return f()
    finally
        set_zero_subnormals(old)
        olds === nothing || _restore_ftz_workers!(olds)
    end
end

# set the mode on every thread (the persistent workers keep it between `@batch` loops); returns the previous modes by `threadid`
function _set_ftz_workers!()
    n = Threads.nthreads()
    olds = zeros(Bool, Threads.maxthreadid())
    @batch per = thread for k in 1:n
        olds[Threads.threadid()] = get_zero_subnormals()
        set_zero_subnormals(true)
    end
    return olds
end
function _restore_ftz_workers!(olds::Vector{Bool})
    n = Threads.nthreads()
    @batch per = thread for k in 1:n
        set_zero_subnormals(olds[Threads.threadid()])
    end
    return nothing
end
