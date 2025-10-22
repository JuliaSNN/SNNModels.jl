if_types = [IFParameter, IFCurrentParameter, IFCurrentDeltaParameter, IFSinExpParameter]

# for if_type in if_types
plots = map(if_types) do if_type
    E = IF(; N = 1, param = if_type())
    Se = Stimulus(PoissonFixed(rate = 1kHz), E, :ge)
    Si = Stimulus(PoissonFixed(rate = 1kHz), E, :gi)
    model = compose(; E, Se, Si, silent = true)
    monitor!(E, [:v, :fire, :syn_curr])
    sim!(model; duration = 300ms)
    # vecplot(E, :syn_curr, title=string(if_type), xlabel = "Time (ms)", ylabel = "Membrane Potential (mV)")
end

# plot(plots...)

true
