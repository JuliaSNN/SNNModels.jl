# Tripod E/I network used for the network-level comparison.
#
# Structure of SNNModels test/network/tripod_network.jl (400 Tripod E cells, two IF inhibitory
# populations of 50 cells: I1 fast (PV-like, onto the soma), I2 slow (SST-like, onto dendrite d1),
# recurrent E-to-E on both dendrites), ported to the current SNNModels API and run without
# plasticity (sim!), driven by Poisson background on both dendrites of the E cells (the
# interneurons, with El = -55 mV, are driven by E only). The weights were set once, on the
# unfixed code, for irregular activity (E about 12 Hz, I1 about 48 Hz, I2 about 24 Hz) and are
# identical for every variant.

using SNNModels
@load_units

function tripod_network(; seed = 1234, NE = 400, NI = 100, w_EI = 0.3, w_IE = 4.0, w_EE = 2.0,
                        w_bgE = 20.0)
    SNNModels.Random.seed!(seed)
    E = Tripod(N = NE, name = "E")
    I1 = IF(N = NI ÷ 2, name = "I1", param = IFParameter(τm = 7ms, R = 0.06, El = -55mV))
    I2 = IF(N = NI ÷ 2, name = "I2", param = IFParameter(τm = 20ms, R = 0.06, El = -55mV))
    E_to_I1 = SpikingSynapse(E, I1, :ge; conn = (p = 0.2, μ = w_EI))
    E_to_I2 = SpikingSynapse(E, I2, :ge; conn = (p = 0.2, μ = w_EI))
    I1_to_E = SpikingSynapse(I1, E, :gaba, :s; conn = (p = 0.2, μ = w_IE))
    I2_to_E = SpikingSynapse(I2, E, :gaba, :d1; conn = (p = 0.2, μ = w_IE))
    E_to_E_d1 = SpikingSynapse(E, E, :glu, :d1; conn = (p = 0.2, μ = w_EE))
    E_to_E_d2 = SpikingSynapse(E, E, :glu, :d2; conn = (p = 0.2, μ = w_EE))
    bg_d1 = Stimulus(PoissonLayer(rate = 10Hz, N = 1000), E, :glu, :d1; conn = (p = 0.2, μ = w_bgE))
    bg_d2 = Stimulus(PoissonLayer(rate = 10Hz, N = 1000), E, :glu, :d2; conn = (p = 0.2, μ = w_bgE))
    model = compose(; E, I1, I2, E_to_I1, E_to_I2, I1_to_E, I2_to_E, E_to_E_d1, E_to_E_d2,
        bg_d1, bg_d2, silent = true)
    return model
end
