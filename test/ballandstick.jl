SomaSynapse = Synapse(
    AMPA = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    GABAa = Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38),
)

DendSynapse = Synapse(
    AMPA = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    NMDA = Receptor(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
    GABAa = Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27),
    GABAb = Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.0006),
)

NMDA = let
    Mg_mM = 1.0mM
    nmda_b = 3.36   # voltage dependence of nmda channels
    nmda_k = -0.077     # Eyal 2018
    NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end

dend_neuron = DendNeuronParameter(
    # adex parameters
    C = 281pF,
    gl = 40nS,
    Vr = -55.6,
    Er = -70.6,
    ΔT = 2,
    Vt = -50.4,
    a = 4,
    b = 80.5pA,
    τw = 144,
    up = 0.1ms,
    τabs = 0.1ms,

    # post-spike adaptation
    postspike = PostSpike(A = 10.0, τA = 30.0),

    # synaptic properties
    soma_syn = SomaSynapse,
    dend_syn = DendSynapse,
    NMDA = NMDA,

    # dendrite
    ds = [160um],
    physiology = human_dend,
)

E = SNNModels.BallAndStick(N = 1, param = dend_neuron)

poisson_exc = PoissonLayerParameter(
    10.2Hz,    # Mean firing rate (Hz) 
    p = 1.0f0,  # Probability of connecting to a neuron
    μ = 1.0,  # Synaptic strength (nS)
    N = 1000, # Neurons in the Poisson Layer
)

poisson_inh = PoissonLayerParameter(
    3Hz,       # Mean firing rate (Hz)
    p = 1.0f0,   # Probability of connecting to a neuron
    μ = 4.0,   # Synaptic strength (nS)
    N = 1000,  # Neurons in the Poisson Layer
)

# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc = PoissonLayer(E, :glu, :d, param = poisson_exc, name = "noiseE")
stim_inh = PoissonLayer(E, :gaba, :d, param = poisson_inh, name = "noiseI")

model = compose(; E, stim_exc, stim_inh, silent=true)
sim!(model, 1s)

true
