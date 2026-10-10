using Statistics: median

"""
    set_optimal_neurons_per_thread!(model; candidates = [128, 256, 512, 1024, 2048, Inf], steps = 200,
                                    repeats = 5, dt = 0.125f0, set = true, verbose = true)

Time a short `sim!` of a copy of `model` (`modelcopy`: the model itself is not touched) for every candidate
`neurons_per_thread` with the threads that are available, report the median time per step for each, and (with `set = true`)
make the fastest the current setting (`set_neurons_per_thread!`). `Inf` means threading off. Returns a vector of
named tuples `(neurons_per_thread, nchunks, us_per_step, speedup)`, `nchunks` being the largest number of chunks of any
population, `speedup` relative to threading off. The setting is global: if the populations of a model differ strongly in cost,
tune a model that contains the dominant population type. With one Julia thread nothing is measured and threading stays off.

The optimum depends on the machine, on the number of threads and on the model; record it with the results
(`Threads.nthreads()`, the table). Short runs measure the firing activity at the start of the run, which can differ from the
stationary one; run the model to its stationary state first, or increase `steps`.
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
