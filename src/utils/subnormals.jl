# Flush-to-zero during `sim!` / `train!`.
#
# Every exponentially decaying state integrated with a multiplicative or forward-Euler step (`g += -dt g / τ`, `x += dt (-x)`)
# ends in the subnormal range when its input stops, and stays there: once `dt g / τ` is below half an ulp of `g` the
# update rounds to `g` itself, so the value never reaches 0. Arithmetic on subnormal operands takes a microcode assist
# on x86 (about 100 cycles per instruction instead of 1), so a few silent synapses make the whole vector loop slow:
# the T4 balanced network at p = 0.02 and N = 500 ran at 0.051 s per simulated second instead of 0.009, and the quiet
# `Rate` population at 41 ns per neuron-step instead of 0.66 (perf/pre110_diag/job2_subnormal.csv).
# The hardware flush-to-zero / denormals-are-zero mode (`set_zero_subnormals(true)`, what Auryn gets from `-ffast-math`)
# removes the assist: a result that would be subnormal is 0 and a subnormal operand reads as 0. A decaying variable then stops
# at a small normal value (where its decrement `dt g / τ` falls below `floatmin`, i.e. |g| <= floatmin τ / dt: 5.9e-37 for
# τ = 5 ms, dt = 0.1 ms in Float32) instead of drifting through the subnormal range. Only arithmetic below `floatmin`
# (1.2e-38 in Float32, 2.2e-308 in Float64) is affected; the T4 network gives bit-identical spikes and membrane potentials.
# The mode is per thread: it is set on the calling thread and, when the population updates run threaded
# (`neurons_per_thread > 0` and more than one Julia thread), on every Polyester worker, and restored afterwards.

"""
    with_flush_subnormals(f, flush::Bool = true)

Run `f()` with the floating-point flush-to-zero mode (`set_zero_subnormals(true)`) on the calling thread and, if the
population updates are threaded (see [`neurons_per_thread`](@ref)), on the Polyester worker threads, then restore the
previous mode of every thread. `flush = false` runs `f()` unchanged. Used by `sim!` and `train!` (keyword
`flush_subnormals = true`): a result that would be subnormal is 0 and a subnormal operand reads as 0, so only
arithmetic below `floatmin` of the type changes. A decaying variable stops at a small normal value (|g| <= `floatmin` τ / dt)
instead of a subnormal one; spikes and states above that level are unchanged.
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

# one iteration per Julia thread of the default pool (`per = thread`): the persistent workers keep the mode between `@batch` loops
# returns the previous mode of every thread (indexed by `threadid`)
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
