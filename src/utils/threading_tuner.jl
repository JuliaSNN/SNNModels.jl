using Statistics: median

"""
    set_optimal_neurons_per_thread!(model; candidates = [128, 256, 512, 1024, 2048, Inf], steps = 200,
                                    repeats = 5, dt = 0.125f0, set = true, verbose = true)

Time a short `sim!` of a copy of `model` (`modelcopy`; `model` is not touched) for every candidate `neurons_per_thread`
(`Inf` = threading off) and, with `set = true`, make the fastest the current setting. Returns a vector of named tuples
`(neurons_per_thread, nchunks, us_per_step, speedup)` (`nchunks`: the largest over the populations; `speedup` relative to
threading off). With one Julia thread nothing is measured and threading stays off. The optimum depends on the machine,
the thread count and the model; short runs measure the initial, not the stationary, activity.
"""
function set_optimal_neurons_per_thread!(model; candidates = [128, 256, 512, 1024, 2048, Inf], steps = 200,
                                         repeats = 5, dt = 0.125f0, set = true, verbose = true)
    if Threads.nthreads() == 1
        verbose && @info "one Julia thread: threading stays off"
        set && set_neurons_per_thread!(0)
        return [(neurons_per_thread = 0, nchunks = 1, us_per_step = NaN, speedup = 1.0)]
    end
    work = modelcopy(model)
    Ns = [Int(p.N) for p in values(work.pop)]
    duration = steps * dt
    sim!(work, duration; dt = dt, neurons_per_thread = 0)                       # compile
    rows = NamedTuple[]
    for c in candidates
        sim!(work, duration; dt = dt, neurons_per_thread = c)                   # warm-up of the setting
        ts = [(@elapsed sim!(work, duration; dt = dt, neurons_per_thread = c)) / steps * 1e6 for _ in 1:repeats]
        bpc = _bpc_value(c)
        push!(rows, (neurons_per_thread = bpc * BATCH, nchunks = maximum(nchunks.(Ns, bpc)), us_per_step = median(ts), speedup = NaN))
    end
    off = rows[findfirst(r -> r.neurons_per_thread == 0, rows)].us_per_step
    rows = [merge(r, (speedup = off / r.us_per_step,)) for r in rows]
    best = rows[argmin([r.us_per_step for r in rows])]
    if verbose
        @info "neurons_per_thread tuning ($(Threads.nthreads()) threads, $steps steps, median of $repeats)"
        for r in rows
            println("  ", lpad(r.neurons_per_thread == 0 ? "off" : string(r.neurons_per_thread), 5), "  chunks ", r.nchunks,
                    "  ", round(r.us_per_step, digits = 2), " us/step  x", round(r.speedup, digits = 2),
                    r === best ? "   <- best" : "")
        end
    end
    set && set_neurons_per_thread!(best.neurons_per_thread)
    return rows
end
export set_optimal_neurons_per_thread!
