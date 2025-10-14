## Soma synapse parameters

EyalNMDA = let
    Mg_mM = Float32(1.0mM)
    nmda_b = 3.36f0   # voltage dependence of nmda channels
    nmda_k = -0.077f0     # Eyal 2018
    NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end


## Tripod
MilesGabaSoma =
    GABAergic(Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38), Receptor())

DuarteGluSoma = Glutamatergic(
    AMPA = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    NMDA = ReceptorVoltage(),
)

EyalGluDend = Glutamatergic(
    AMPA = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    NMDA = ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
)
MilesGabaDend = GABAergic(
    Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27),
    Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006), # τd = 100.0
)

TripodSomaReceptors = Receptors(DuarteGluSoma, MilesGabaSoma)
TripodDendReceptors = Receptors(EyalGluDend, MilesGabaDend)

TripodSomaSynapse = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = TripodSomaReceptors,
    NMDA = EyalNMDA,
)

TripodDendSynapse = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = TripodDendReceptors,
    NMDA = EyalNMDA,
)

## Soma parameters

SomaNMDA = let
    Mg_mM = 1.0mM |> Float32
    nmda_b = 3.57f0   # voltage dependence of nmda channels
    nmda_k = -0.062f0     # Eyal 2018
    NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end
SomaGlu = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 1ms, τd = 6.0ms, g0 = 0.7),
    ReceptorVoltage(
        # name = "NMDA",
        E_rev = 0.0,
        τr = 1ms,
        τd = 100.0,
        g0 = 0.15,
        nmda = 1.0f0,
    ),
)
SomaGABA = GABAergic(
    Receptor(E_rev = -70.0, τr = 0.5, τd = 10.0, g0 = 2.0),
    Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006), # τd = 100.0
)

SomaNMDA = NMDAVoltageDependency()
SomaReceptors = Receptors(SomaGlu, SomaGABA)
SomaSynapse = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = SomaReceptors,
    NMDA = SomaNMDA,
)

## CAN_AHP parameters

Glu_CANAHP = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 1, τd = 2.5ms, g0 = 0.2mS/cm^2),
    ReceptorVoltage(E_rev = 0.0, τr = 4.65ms, τd = 75ms, g0 = 0.3mS/cm^2, nmda = 1.0f0),
)
Gaba_CANAHP = GABAergic(
    Receptor(E_rev = -70.0, τr = 1, τd = 10ms, g0 = 0.35mS/cm^2),
    Receptor(E_rev = -90.0, τr = 90ms, τd = 160ms, g0 = 5e-4mS/cm^2), # τd = 100.0
)
Synapse_CANAHP = Receptors(Glu_CANAHP, Gaba_CANAHP)
αs_CANAHP = [1.0, 0.275/ms, 1.0, 0.015/ms]
NMDA_CANAHP = let
    Mg_mM = 1.5mM |> Float32
    nmda_b = 3.57f0
    nmda_k = -0.063f0
    NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end


export SomaNMDA,
    SomaSynapse, TripodSomaSynapse, TripodDendSynapse, EyalNMDA, NMDA_CANAHP, Synapse_CANAHP
