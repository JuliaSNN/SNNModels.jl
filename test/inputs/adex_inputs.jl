adex_types = [AdExParameter, AdExSinExpParameter, AdExReceptorParameter]

# for if_type in if_types
plots = map(adex_types) do adex_type
    E = AdEx(; N = 1, param = adex_type())
    Se = Stimulus(PoissonFixed(rate=1kHz), E, :he, μ = 6)
    Si = Stimulus(PoissonFixed(rate=1kHz), E, :hi, μ = 3)
    model = compose(; E, Se, Si, silent = true)
    monitor!(E, [:v, :fire, :syn_curr])
    sim!(model; duration = 300ms)
    vecplot(
        E,
        :v,
        title = string(adex_type),
        xlabel = "Time (ms)",
        ylabel = "Membrane Potential (mV)",
        add_spikes = true,
    )
end

plot(plots...)
true
