E = let
    NMDA = let
        Mg_mM = 1mM
        nmda_b = 3.36f0   # voltage dependence of nmda channels
        nmda_k = -0.077f0    # Eyal 2018
        NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
    end

    SomaSynapse = ReceptorSynapse(
        glu_receptors = [1],
        gaba_receptors = [2],
        syn = Receptors(
            AMPA = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
            GABAa = Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38),
        ),
    )


    DendSynapse = ReceptorSynapse(
        glu_receptors = [1, 2],
        gaba_receptors = [3, 4],
        syn = Receptors(
            AMPA = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
            NMDA = Receptor(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
            GABAa = Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27),
            GABAb = Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.0006),
        ),
    )

    dend_neuron = (
        param = TripodParameter(ds = [160um, 200um], physiology = human_dend),
        # adex parameters
        adex = AdExParameter(
            C = 281pF,
            gl = 40nS,
            Vr = -55.6,
            El = -70.6,
            ΔT = 2,
            Vt = -50.4,
            a = 4,
            b = 80.5pA,
            τw = 144,
        ),

        # post-spike adaptation
        spike = PostSpike(At = 10.0, τA = 30.0, τabs = 0.1ms, up = 0.1ms),

        # synaptic properties
        soma_syn = SomaSynapse,
        dend_syn = DendSynapse,
    )

    E = SNNModels.Population(; N = 100, dend_neuron...)
end

poisson_exc = PoissonLayer(
    10.2Hz,    # Mean firing rate (Hz) 
    N = 1000, # Neurons in the Poisson Layer
)

poisson_inh = PoissonLayer(
    3Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)

stim_exc = Stimulus(poisson_exc, E, :glu, :d1, conn = (μ = 0.1, ρ = 1), name = "noiseE")
stim_inh = Stimulus(poisson_inh, E, :gaba, :d1, conn = (μ = 0.1, ρ = 1), name = "noiseI")

model = compose(; E, stim_exc, stim_inh, silent = true)

train!(model, 1s)

true
