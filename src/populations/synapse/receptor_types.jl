## Soma synapse parameters

"""
    EyalNMDA :: NMDAVoltageDependency{Float32}

NMDA magnesium block with `mg = 1` mM, `b = 3.36` mM, `k = -0.077` 1/mV (Eyal et al. 2018).
Numerically identical to `SomaNMDA`.
"""
EyalNMDA = _predefined(NMDAVoltageDependency; mg = 1.0, b = 3.36, k = -0.077)   # b: voltage dependence of NMDA channels; k: Eyal 2018

"""
    EyalNMDA32 :: NMDAVoltageDependency{Float32}

`EyalNMDA` (`b = 3.36` mM, `k = -0.077` 1/mV, `mg = 1` mM) with the exponential of the block evaluated as in the published
Tripod code, ``(1 + x/32)^{32}`` (`approx = 32`; `equations.jl:17-20` of `aquaresima/tripod_neuron`, commit 8698e1a). Block
at -70 mV: 0.0225 (0.0160 with `EyalNMDA`). Used by `TripodDendSynapsePaper`.
"""
EyalNMDA32 = _predefined(NMDAVoltageDependency; mg = 1.0, b = 3.36, k = -0.077, approx = 32)

"""
    MouseNMDA :: NMDAVoltageDependency{Float32}

Magnesium block with the mouse values printed in the Tripod paper (Quaresima et al. 2023, bioRxiv 10.1101/2022.09.05.506197):
`b = 3.57` mM, `k = -0.062` 1/mV (the paper's "gamma"), `mg = 1` mM. The published Tripod code does not use them: it
runs the mouse model with the human block (`EyalNMDA`, 3.36 / -0.077), which is what `TripodDendSynapseMouse` does.
`MouseNMDA` is provided for runs that follow the printed parameters.
"""
MouseNMDA = _predefined(NMDAVoltageDependency; mg = 1.0, b = 3.57, k = -0.062)


# Predefined receptors (E_rev in mV, τr/τd in ms, g0 in nS). The names refer to the source of
# the parameters; the references are (as cited in the code of the published Tripod model, aquaresima/tripod_neuron):
#   Miles R. et al. (1996), Differences between somatic and dendritic inhibition in the
#     hippocampus, Neuron 16(4):815-823, doi:10.1016/S0896-6273(00)80101-4 (GABA receptors).
#   Eyal G. et al. (2018), Human cortical pyramidal neurons: from spines to spikes via models,
#     Front. Cell. Neurosci. 12, doi:10.3389/fncel.2018.00181 (dendritic AMPA/NMDA).
#   Duarte R., Morrison A. (2017), Leveraging heterogeneity for neural computation with fading memory in
#     layer 2/3 cortical microcircuits, bioRxiv, doi:10.1101/230821 (somatic AMPA, mouse dendritic AMPA/NMDA; the Tripod
#     paper cites it as Duarte and Morrison 2019).
## Tripod
MilesGabaSoma = _predefined(Receptor; E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38, target = :gaba)

DuarteGluSoma = _predefined(Receptor; E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73, target = :glu)

EyalGluDend = Glutamatergic(
    AMPA = _predefined(Receptor; E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73, target = :glu),
    NMDA = _predefined(Receptor;   # ReceptorVoltage (nmda = 1)
        E_rev = 0.0,
        τr = 8,
        τd = 35.0,
        g0 = 1.31,
        nmda = 1.0f0,
        target = :glu,
    ),
)
MilesGabaDend = GABAergic(
    _predefined(Receptor; E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27, target = :gaba),
    _predefined(Receptor; E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006, target = :gaba), # τd = 100.0
)

TripodSomaReceptors = Receptors(DuarteGluSoma, MilesGabaSoma)
TripodDendReceptors = Receptors(EyalGluDend, MilesGabaDend)

# Mouse dendritic glutamatergic receptors of the published Tripod code (`DuarteGluDend` in synapses.jl, Duarte and Morrison):
# AMPA E 0, τr 0.25, τd 2, g0 0.73; NMDA E 0, τr 0.99, τd 100, g0 0.159.
"""
    DuarteGluDend :: Glutamatergic

Dendritic AMPA and NMDA receptors of the mouse Tripod (`M_*` configurations of the published code, `DuarteGluDend` in
`model/synapses.jl`, after Duarte and Morrison, bioRxiv 2017, doi:10.1101/230821):

| Receptor | E_rev (mV) | τr (ms) | τd (ms) | g0 (nS) | NMDA block |
|:---------|-----------:|--------:|--------:|--------:|:----------:|
| AMPA | 0 | 0.25 | 2   | 0.73  | no  |
| NMDA | 0 | 0.99 | 100 | 0.159 | yes |
"""
DuarteGluDend = Glutamatergic(
    AMPA = _predefined(Receptor; E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73, target = :glu),
    NMDA = _predefined(Receptor; E_rev = 0.0, τr = 0.99, τd = 100.0, g0 = 0.159, nmda = 1.0f0, target = :glu),
)
TripodDendReceptorsMouse = Receptors(DuarteGluDend, MilesGabaDend)

# Receptors of the published Tripod code (8698e1a) that differ from the library ones: GABAa reversal -75 mV on soma and
# dendrite (the library uses -70 mV), somatic GABAa τr 0.3 ms (0.1), somatic AMPA τr 0.25 ms (0.26).
PaperGluSoma = _predefined(Receptor; E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73, target = :glu)
PaperGabaSoma = _predefined(Receptor; E_rev = -75.0, τr = 0.3, τd = 15.0, g0 = 0.38, target = :gaba)
PaperGabaDend = GABAergic(
    _predefined(Receptor; E_rev = -75.0, τr = 4.8, τd = 29.0, g0 = 0.27, target = :gaba),
    _predefined(Receptor; E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006, target = :gaba),
)

"""
    TripodSomaSynapse :: ReceptorSynapse

Default somatic synapse of `Tripod` and `BallAndStick`: a `ReceptorSynapse` with two receptors,
`glu_receptors = [1]`, `gaba_receptors = [2]`, `NMDA = EyalNMDA`.

| Receptor | E_rev (mV) | τr (ms) | τd (ms) | g0 (nS) |
|:---------|-----------:|--------:|--------:|--------:|
| AMPA (`DuarteGluSoma`) | 0 | 0.26 | 2.0 | 0.73 |
| GABAa (`MilesGabaSoma`) | -70 | 0.1 | 15.0 | 0.38 |

The GABA parameters are attributed to Miles et al. (1996), Neuron 16(4):815-823; the AMPA
parameters to Duarte and Morrison (bioRxiv 2017, doi:10.1101/230821; the Tripod paper cites it as Duarte and
Morrison 2019). The published Tripod code uses AMPA τr 0.25 ms and GABAa E_rev -75 mV, τr 0.3 ms
(`TripodSomaSynapsePaper`).
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
doi:10.3389/fncel.2018.00181; GABA parameters to Miles et al. (1996), Neuron 16(4):815-823.
"""
TripodDendSynapse = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = TripodDendReceptors,
    NMDA = EyalNMDA,
)

"""
    TripodDendSynapseMouse :: ReceptorSynapse

Dendritic synapse of the mouse Tripod: `TripodDendSynapse` with the glutamatergic receptors of `DuarteGluDend` (AMPA τr 0.25,
τd 2, g0 0.73; NMDA τr 0.99, τd 100, g0 0.159) and the GABAergic receptors of the human set. The NMDA block is `EyalNMDA`
(`b = 3.36`, `k = -0.077`), which is what the published Tripod code uses for the mouse as well; the paper prints
`b = 3.57`, `k = -0.062` for the mouse (`MouseNMDA`; pass `NMDA = MouseNMDA` to use them). Use it with `mouse_dend`:
```julia
Tripod(param = TripodParameter(ds = distal_proximal, physiology = mouse_dend), dend_syn = TripodDendSynapseMouse)
```
The default `dend_syn` of `Tripod` is unchanged (`TripodDendSynapse`); `TripodPaper(physiology = :mouse)` selects both.
"""
TripodDendSynapseMouse = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = TripodDendReceptorsMouse,
    NMDA = EyalNMDA,
)

"""
    TripodSomaSynapsePaper :: ReceptorSynapse

Somatic synapse of the published Tripod code (`aquaresima/tripod_neuron` at 8698e1a): AMPA (E 0, τr 0.25, τd 2, g0 0.73) and
GABAa (E -75, τr 0.3, τd 15, g0 0.38), `decay = 32` (the `exp32` of that code). `glu_receptors = [1]`, `gaba_receptors = [2]`.
"""
TripodSomaSynapsePaper = ReceptorSynapse(
    glu_receptors = [1],
    gaba_receptors = [2],
    syn = Receptors(PaperGluSoma, PaperGabaSoma),
    NMDA = EyalNMDA32,
    decay = 32,
)

"""
    TripodDendSynapsePaper :: ReceptorSynapse

Human dendritic synapse of the published Tripod code (8698e1a): the receptors of `TripodDendSynapse` with GABAa E_rev -75 mV
(`PaperGabaDend`), the Mg block of that code (`EyalNMDA32`, exponential `(1 + x/32)^32`) and `decay = 32`.
"""
TripodDendSynapsePaper = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = Receptors(EyalGluDend, PaperGabaDend),
    NMDA = EyalNMDA32,
    decay = 32,
)

"""
    TripodDendSynapsePaperMouse :: ReceptorSynapse

Mouse dendritic synapse of the published Tripod code (8698e1a, `mouse_synapses`): `DuarteGluDend` glutamatergic receptors,
GABAa E_rev -75 mV, block `EyalNMDA32`, `decay = 32`.
"""
TripodDendSynapsePaperMouse = ReceptorSynapse(
    glu_receptors = [1, 2],
    gaba_receptors = [3, 4],
    syn = Receptors(DuarteGluDend, PaperGabaDend),
    NMDA = EyalNMDA32,
    decay = 32,
)

## Soma parameters

SomaGlu = Glutamatergic(
    _predefined(Receptor; E_rev = 0.0, τr = 1ms, τd = 6.0ms, g0 = 0.7, target = :glu),
    _predefined(Receptor;   # ReceptorVoltage (nmda = 1)
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
    _predefined(Receptor; E_rev = -70.0, τr = 0.5, τd = 10.0, g0 = 2.0, target = :gaba),
    _predefined(Receptor; E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006, target = :gaba),
)

# Somatic NMDA Mg block: the NMDAVoltageDependency defaults (b = 3.36, k = -0.077).
# (An earlier definition with b = 3.57, k = -0.062 was always overridden by this line and has
# been removed in 1.8.3; the effective value is unchanged.)
"""
    SomaNMDA :: NMDAVoltageDependency{Float32}

Somatic NMDA magnesium block, equal to the `NMDAVoltageDependency` defaults
(`b = 3.36` mM, `k = -0.077` 1/mV, `mg = 1` mM).
"""
SomaNMDA = _predefined(NMDAVoltageDependency)
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

export SomaNMDA,
    SomaSynapse,
    TripodSomaSynapse,
    TripodDendSynapse,
    TripodDendSynapseMouse,
    TripodSomaSynapsePaper,
    TripodDendSynapsePaper,
    TripodDendSynapsePaperMouse,
    DuarteGluDend,
    EyalNMDA,
    EyalNMDA32,
    MouseNMDA,
    SomaReceptors
