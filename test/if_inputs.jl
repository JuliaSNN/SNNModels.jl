if_types = [
    IFParameter,
    IFCurrentParameter,
    IFCurrentDeltaParameter,
    IFSinExpParameter
    ]

# for if_type in if_types
plots = map(if_types) do if_type
    E = IF(; N = 1, param=if_type())
    Se = PoissonStimulus(E, :ge, p_post = 1, N_pre = 100, param = 1kHz, μ=1 )
    Si = PoissonStimulus(E, :gi, p_post = 1, N_pre = 100, param = 1kHz, μ=1 )
    model = compose(; E, Se, Si, silent=true)
    monitor!(E, [:v, :fire, :syn_curr])
    sim!(model; duration = 300ms)
    # vecplot(E, :syn_curr, title=string(if_type), xlabel = "Time (ms)", ylabel = "Membrane Potential (mV)")
end

# plot(plots...)