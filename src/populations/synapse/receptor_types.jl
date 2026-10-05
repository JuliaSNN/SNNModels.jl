## Soma synapse parameters

"""
    EyalNMDA :: NMDAVoltageDependency{Float32}

NMDA magnesium block with `mg = 1` mM, `b = 3.36` mM, `k = -0.077` 1/mV (values attributed in
the code to Eyal et al. 2018). Numerically identical to `SomaNMDA`.
"""
EyalNMDA = let
    Mg_mM = Float32(1.0mM)
    nmda_b = 3.36f0   # voltage dependence of nmda channels
    nmda_k = -0.077f0     # Eyal 2018
    NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end


# Predefined receptors (E_rev in mV, τr/τd in ms, g0 in nS). The names refer to the source of
# the parameters; the references are cited in SNNUtils/src/models/quaresima2022.jl:
#   Miles R. et al. (1996), Differences between somatic and dendritic inhibition in the
#     hippocampus, Neuron 16(4):815-823, doi:10.1016/S0896-6273(00)80101-4 (GABA receptors).
#   Eyal G. et al. (2018), Human cortical pyramidal neurons: from spines to spikes via models,
#     Front. Cell. Neurosci. 12, doi:10.3389/fncel.2018.00181 (dendritic AMPA/NMDA).
#   Duarte: somatic AMPA; reference not given in the code.
## Tripod
MilesGabaSoma = Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38, target = :gaba)

DuarteGluSoma = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73, target = :glu)

EyalGluDend = Glutamatergic(
    AMPA = Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73, target = :glu),
    NMDA = ReceptorVoltage(
        E_rev = 0.0,
        τr = 8,
        τd = 35.0,
        g0 = 1.31,
        nmda = 1.0f0,
        target = :glu,
    ),
)
MilesGabaDend = GABAergic(
    Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27, target = :gaba),
    Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006, target = :gaba), # τd = 100.0
)

TripodSomaReceptors = Receptors(DuarteGluSoma, MilesGabaSoma)
TripodDendReceptors = Receptors(EyalGluDend, MilesGabaDend)

"""
    TripodSomaSynapse :: ReceptorSynapse

Default somatic synapse of `Tripod` and `BallAndStick`: a `ReceptorSynapse` with two receptors,
`glu_receptors = [1]`, `gaba_receptors = [2]`, `NMDA = EyalNMDA`.

| Receptor | E_rev (mV) | τr (ms) | τd (ms) | g0 (nS) |
|:---------|-----------:|--------:|--------:|--------:|
| AMPA (`DuarteGluSoma`) | 0 | 0.26 | 2.0 | 0.73 |
| GABAa (`MilesGabaSoma`) | -70 | 0.1 | 15.0 | 0.38 |

The GABA parameters are attributed to Miles et al. (1996), Neuron 16(4):815-823; the AMPA
parameters to "Duarte" (reference not given in the code).
"""
TripodSomaSynapse = ReceptorSynapse(
    glu_receptors = [1],
    gaba_receptors = [2],
    syn = TripodSomaReceptors,
    NMDA = EyalNMDA,
)

"""
    TripodDendSynapse :: ReceptorSynapse

Default dendritic synapse of `Tripod` and `BallAndStick`: a `ReceptorSynapse` with four
receptors, `glu_receptors = [1, 2]`, `gaba_receptors = [3, 4]`, `NMDA = EyalNMDA`.

| Receptor | E_rev (mV) | τr (ms) | τd (ms) | g0 (nS) | NMDA block |
|:---------|-----------:|--------:|--------:|--------:|:----------:|
| AMPA  | 0   | 0.26 | 2.0   | 0.73  | no  |
| NMDA  | 0   | 8    | 35.0  | 1.31  | yes |
| GABAa | -70 | 4.8  | 29.0  | 0.27  | no  |
| GABAb | -90 | 30   | 400.0 | 0.006 | no  |

AMPA/NMDA parameters attributed to Eyal et al. (2018), Front. Cell. Neurosci. 12,
doi:10.3389/fncel.2018.00181; GABA parameters to Miles et al. (1996), Neuron 16(4):815-823
(references as cited in SNNUtils/src/models/quaresima2022.jl).
"""
TripodDendSynapse = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = TripodDendReceptors,
    NMDA = EyalNMDA,
)

## Soma parameters

SomaGlu = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 1ms, τd = 6.0ms, g0 = 0.7, target = :glu),
    ReceptorVoltage(
        # name = "NMDA",
        E_rev = 0.0,
        τr = 1ms,
        τd = 100.0,
        g0 = 0.15,
        nmda = 1.0f0,
        target = :glu,
    ),
)
SomaGABA = GABAergic(
    Receptor(E_rev = -70.0, τr = 0.5, τd = 10.0, g0 = 2.0, target = :gaba),
    Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006, target = :gaba),
)

# Somatic NMDA Mg block: the NMDAVoltageDependency defaults (b = 3.36, k = -0.077).
# (An earlier definition with b = 3.57, k = -0.062 was always overridden by this line and has
# been removed in 1.8.3; the effective value is unchanged.)
"""
    SomaNMDA :: NMDAVoltageDependency{Float32}

Somatic NMDA magnesium block, equal to the `NMDAVoltageDependency` defaults
(`b = 3.36` mM, `k = -0.077` 1/mV, `mg = 1` mM).
"""
SomaNMDA = NMDAVoltageDependency()
"""
    SomaReceptors :: ReceptorArray

Somatic receptor set `[AMPA, NMDA, GABAa, GABAb]`, the default `syn` of `ReceptorSynapse` and
`MultiReceptorSynapse`.

| Receptor | E_rev (mV) | τr (ms) | τd (ms) | g0 (nS) | NMDA block | target |
|:---------|-----------:|--------:|--------:|--------:|:----------:|:------:|
| AMPA  | 0   | 1   | 6.0   | 0.7   | no  | `:glu`  |
| NMDA  | 0   | 1   | 100.0 | 0.15  | yes | `:glu`  |
| GABAa | -70 | 0.5 | 10.0  | 2.0   | no  | `:gaba` |
| GABAb | -90 | 30  | 400.0 | 0.006 | no  | `:gaba` |
"""
SomaReceptors = Receptors(SomaGlu, SomaGABA)
"""
    SomaSynapse :: ReceptorSynapse

`ReceptorSynapse` with `syn = SomaReceptors`, `glu_receptors = [1, 2]`, `gaba_receptors = [3, 4]`,
`NMDA = SomaNMDA`; equal to `ReceptorSynapse()`.
"""
SomaSynapse = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = SomaReceptors,
    NMDA = SomaNMDA,
)

# ## CAN_AHP parameters

# Glu_CANAHP = Glutamatergic(
#     Receptor(E_rev = 0.0, τr = 1, τd = 2.5ms, g0 = 0.2mS/cm^2),
#     ReceptorVoltage(E_rev = 0.0, τr = 4.65ms, τd = 75ms, g0 = 0.3mS/cm^2, nmda = 1.0f0),
# )
# Gaba_CANAHP = GABAergic(
#     Receptor(E_rev = -70.0, τr = 1, τd = 10ms, g0 = 0.35mS/cm^2),
#     Receptor(E_rev = -90.0, τr = 90ms, τd = 160ms, g0 = 5e-4mS/cm^2), # τd = 100.0
# )
# Synapse_CANAHP = Receptors(Glu_CANAHP, Gaba_CANAHP)
# αs_CANAHP = [1.0, 0.275/ms, 1.0, 0.015/ms]
# NMDA_CANAHP = let
#     Mg_mM = 1.5mM |> Float32
#     nmda_b = 3.57f0
#     nmda_k = -0.063f0
#     NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
# end


# NOTE: `NMDA_CANAHP` and `Synapse_CANAHP` are exported below but their definitions are
# commented out above, so they are not defined.
export SomaNMDA,
    SomaSynapse,
    TripodSomaSynapse,
    TripodDendSynapse,
    EyalNMDA,
    NMDA_CANAHP,
    Synapse_CANAHP,
    SomaReceptors
