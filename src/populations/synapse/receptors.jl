# synapse.jl

@doc raw"""
    Receptor(; name = "Receptor", E_rev = 0.0, τr = -1, τd = -1, g0 = 0, nmda = 0, target = :none)

Parameters of one synaptic receptor with double-exponential kinetics, used by
`ReceptorSynapse` and `MultiReceptorSynapse` (and by the dendritic models through them).
`ReceptorVoltage` is an alias of `Receptor` (used for NMDA receptors, `nmda = 1`).

The derived fields are computed from `τr`, `τd` and `g0` by the keyword constructor and
should not be passed by hand:
```math
g_{syn} = g_0 \cdot \mathrm{norm\_synapse}(\tau_r, \tau_d), \qquad
\alpha = \frac{\tau_d - \tau_r}{\tau_d \tau_r}, \qquad
\tau_r^{-} = 1/\tau_r, \quad \tau_d^{-} = 1/\tau_d
```
so that a unit weight produces a conductance with peak ``g_0`` (see `ReceptorSynapse`).

# Fields
- `name::String = "Receptor"`: label.
- `E_rev::T = 0.0`: reversal potential (mV).
- `τr::T = -1.0`: rise time constant (ms); non-positive values mean "unset".
- `τd::T = -1.0`: decay time constant (ms); non-positive values mean "unset".
- `g0::T = 0.0`: peak conductance per unit weight (nS).
- `gsyn::T`: `g0 * norm_synapse(τr, τd)` if `g0 > 0`, else `0` (nS).
- `α::T`: `α_synapse(τr, τd) = (τd - τr) / (τd τr)` (1/ms).
- `τr⁻::T`: `1/τr` if positive, else `0` (1/ms).
- `τd⁻::T`: `1/τd` if positive, else `0` (1/ms).
- `nmda::T = 0.0`: if non-zero the receptor current is multiplied by the NMDA magnesium block.
- `target::Symbol = :none`: input group of the receptor, used by `MultiReceptorSynapse`
  (e.g. `:glu`, `:gaba`, `:AMPA`); ignored by `ReceptorSynapse`.

`T` defaults to `Float32`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
AMPA = SNN.Receptor(E_rev = 0mV, τr = 0.26ms, τd = 2ms, g0 = 0.73nS, target = :glu)
NMDA = SNN.ReceptorVoltage(E_rev = 0mV, τr = 8ms, τd = 35ms, g0 = 1.31nS, nmda = 1, target = :glu)
AMPA.gsyn
```
"""
Receptor

abstract type AbstractReceptor end

@snn_kw struct Receptor{T = Float32} <: AbstractReceptor
    name::String = "Receptor"
    E_rev::T = 0.0
    τr::T = -1.0f0
    τd::T = -1.0f0
    g0::T = 0.0f0
    gsyn::T = g0 > 0 ? g0 * norm_synapse(τr, τd) : 0.0f0
    α::T = α_synapse(τr, τd)
    τr⁻::T = 1 / τr > 0 ? 1 / τr : 0.0f0
    τd⁻::T = 1 / τd > 0 ? 1 / τd : 0.0f0
    nmda::T = 0.0f0
    target::Symbol = :none
end

"""
    ReceptorArray = Vector{Receptor{Float32}}

Vector of receptors, the type of the `syn` field of `ReceptorSynapse` and `MultiReceptorSynapse`.
"""
ReceptorArray = Vector{Receptor{Float32}}

"""
    ReceptorVoltage = Receptor

Alias of `Receptor`, used to mark voltage-dependent (NMDA) receptors, which are created with
`nmda = 1`. It is not a distinct type.
"""
ReceptorVoltage = Receptor



"""
    Receptors(; AMPA = Receptor(), NMDA = ReceptorVoltage(), GABAa = Receptor(), GABAb = Receptor())
    Receptors(AMPA, NMDA, GABAa, GABAb)
    Receptors(glu::Glutamatergic, gaba::GABAergic)
    Receptors(args...)

Build a `ReceptorArray` (`Vector{Receptor{Float32}}`). `Receptors` is a function, not a type.
The four-receptor forms return `[AMPA, NMDA, GABAa, GABAb]`, so that in a `ReceptorSynapse` the
glutamatergic receptors have indices `[1, 2]` and the GABAergic ones `[3, 4]` (the default
`glu_receptors`/`gaba_receptors`). The variadic form collects any number of receptors in the
given order. The keyword defaults are the empty `Receptor()` (zero conductance), so all
receptors are normally passed explicitly.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
recs = SNN.Receptors(SNN.Receptor(E_rev = 0mV, τr = 1ms, τd = 6ms, g0 = 0.7nS, target = :glu),
                     SNN.Receptor(E_rev = -70mV, τr = 0.5ms, τd = 10ms, g0 = 2nS, target = :gaba))
length(recs)  # 2
```
"""
Receptors

function Receptors(;
    AMPA::Receptor{T} = Receptor(),
    NMDA::ReceptorVoltage{T} = ReceptorVoltage(),
    GABAa::Receptor{T} = Receptor(),
    GABAb::Receptor{T} = Receptor(),
) where {T<:Float32}
    return ReceptorArray([AMPA, NMDA, GABAa, GABAb])
end

function Receptors(
    AMPA::Receptor{T},
    NMDA::ReceptorVoltage{T},
    GABAa::Receptor{T},
    GABAb::Receptor{T},
) where {T<:Float32}
    return ReceptorArray([AMPA, NMDA, GABAa, GABAb])
end

function Receptors(args...)
    return ReceptorArray(collect(args))
end


"""
    infer_receptors(receptors::ReceptorArray)::NamedTuple

Infer receptor types and their indices from a ReceptorArray.

This function processes an array of receptors and returns a named tuple where each field
corresponds to a receptor type (specified by the `target` field of each receptor). The value
of each field is a vector of indices that reference the receptors of that type in the input array.

# Arguments
- `receptors::ReceptorArray`: An array of receptor objects to be processed

# Returns
- `NamedTuple`: A named tuple where each field name is a receptor type (Symbol) and the
  corresponding value is a vector of indices (Int) of receptors of that type in the input array

# Errors
- Logs an error (`@error`, it does not throw) if a receptor has `target == :none`; that
  receptor is then grouped under `:none`.

# Example

```julia
using SpikingNeuralNetworks
receptors = SNN.ReceptorArray([
    SNN.Receptor(target = :glu),
    SNN.Receptor(target = :gaba),
    SNN.Receptor(target = :glu),
])
result = SNNModels.infer_receptors(receptors)
# result.glu == [1, 3], result.gaba == [2] (the order of the fields is not guaranteed)
```
"""
function infer_receptors(receptors::ReceptorArray)::NamedTuple
    rec_name = Symbol[]
    rec_id = Int[]
    for (i, receptor) in enumerate(receptors)
        if receptor.target === :none
            @error "Receptor target not defined in MultiReceptorSynapse"
        end
        push!(rec_name, receptor.target)
        push!(rec_id, i)
    end
    recs = Dict{Symbol,Vector{Int}}()
    for (name, id) in zip(rec_name, rec_id)
        if haskey(recs, name)
            push!(recs[name], id)
        else
            recs[name] = [id]
        end
    end
    return (; recs...)
end



"""
    Glutamatergic(; AMPA = Receptor(), NMDA = ReceptorVoltage())
    Glutamatergic(AMPA, NMDA)

Pair of glutamatergic receptors (AMPA and NMDA), used with `GABAergic` to build a
four-receptor array via `Receptors(glu, gaba)`.

# Fields
- `AMPA::Receptor = Receptor()`: AMPA receptor.
- `NMDA::Receptor = ReceptorVoltage()`: NMDA receptor (set `nmda = 1` to enable the magnesium block).
"""
Glutamatergic

@kwdef struct Glutamatergic
    AMPA::Receptor = Receptor()
    NMDA::ReceptorVoltage = ReceptorVoltage()
end

"""
    GABAergic(; GABAa = Receptor(), GABAb = Receptor())
    GABAergic(GABAa, GABAb)

Pair of GABAergic receptors (GABAa and GABAb), used with `Glutamatergic` in `Receptors(glu, gaba)`.

# Fields
- `GABAa::Receptor = Receptor()`: fast GABAa receptor.
- `GABAb::Receptor = Receptor()`: slow GABAb receptor.
"""
GABAergic

@kwdef struct GABAergic
    GABAa::Receptor = Receptor()
    GABAb::Receptor = Receptor()
end

"""
    Receptors(glu::Glutamatergic, gaba::GABAergic) -> ReceptorArray

Return `[glu.AMPA, glu.NMDA, gaba.GABAa, gaba.GABAb]`.
"""
function Receptors(glu::Glutamatergic, gaba::GABAergic)
    return Receptors(glu.AMPA, glu.NMDA, gaba.GABAa, gaba.GABAb)
end

export Receptor,
    Receptors,
    ReceptorVoltage,
    GABAergic,
    Glutamatergic,
    ReceptorArray,
    NMDAVoltageDependency

"""
    norm_synapse(receptor::Receptor)

Normalisation factor of `receptor`, `norm_synapse(receptor.τr, receptor.τd)`.
"""
function norm_synapse(receptor::Receptor)
    norm_synapse(receptor.τr, receptor.τd)
end

@doc raw"""
    norm_synapse(τr, τd)

Inverse of the peak of the difference of exponentials ``e^{-t/\tau_d} - e^{-t/\tau_r}``:
```math
t_p = \frac{\tau_r \tau_d}{\tau_d - \tau_r}\ln\frac{\tau_d}{\tau_r}, \qquad
\mathrm{norm\_synapse} = \left(e^{-t_p/\tau_d} - e^{-t_p/\tau_r}\right)^{-1}
```
Used to set `gsyn = g0 * norm_synapse(τr, τd)` in `Receptor`, so that `g0` is the peak
conductance per unit weight. Requires `τd != τr`, both positive.

# Example
```julia
using SpikingNeuralNetworks
SNN.norm_synapse(0.26, 2.0)   # ≈ 1.559
```
"""
function norm_synapse(τr, τd)
    t_p = τr * τd / (τd - τr) * log(τd / τr)
    return 1 / (-exp(-t_p / τr) + exp(-t_p / τd))
end

"""
    α_synapse(τr, τd)

Return `(τd - τr) / (τd * τr)` (= `1/τr - 1/τd`), the increment of the rise variable per unit
weight in the receptor kinetics (see `ReceptorSynapse`).
"""
function α_synapse(τr, τd)
    return (τd - τr) / (τd * τr)
end

const Mg_mM = 1.0f0
const nmda_b = 3.36f0   # voltage dependence of nmda channels
const nmda_k = -0.077f0     # Eyal 2018

@doc raw"""
    NMDAVoltageDependency(; b = 3.36, k = -0.077, mg = 1.0)

Parameters of the voltage-dependent magnesium block of NMDA receptors. The block factor
multiplying the current of every receptor with `nmda != 0` is
```math
B(V) = \frac{1}{1 + \frac{[\mathrm{Mg}]}{b}\, e^{k V}}
```
(see `nmda_gating`), the Jahr and Stevens form; the default values are those given in the
code comments for Eyal et al. (2018).

# Fields
- `b::T = 3.36`: (mM).
- `k::T = -0.077`: voltage sensitivity (1/mV).
- `mg::T = 1.0`: extracellular magnesium concentration (mM).

`T` defaults to `Float32`. Predefined instances: `SomaNMDA` (defaults) and `EyalNMDA`
(identical values).

# References
Jahr C. E., Stevens C. F. (1990), J. Neurosci. (functional form of the block).
Eyal G. et al. (2018), Human cortical pyramidal neurons: from spines to spikes via models,
Front. Cell. Neurosci. 12, doi:10.3389/fncel.2018.00181 (cited in the code as "Eyal 2018").

# Example
```julia
using SpikingNeuralNetworks
SNN.nmda_gating(-70.0f0, SNN.NMDAVoltageDependency())
```
"""
NMDAVoltageDependency

@snn_kw struct NMDAVoltageDependency{T = Float32}
    b::T = nmda_b
    k::T = nmda_k
    mg::T = Mg_mM
end

@doc raw"""
    nmda_gating(v, NMDA::NMDAVoltageDependency)

Magnesium block factor ``B(v) = 1 / (1 + (mg/b)\, e^{k v})`` at membrane potential `v` (mV).
"""
function nmda_gating(v, NMDA::NMDAVoltageDependency)
    @unpack b, k, mg = NMDA
    return 1 / (1.0f0 + (mg / b) * exp256(k * v))
end

export norm_synapse,
    EyalNMDA,
    Receptor,
    Receptors,
    ReceptorVoltage,
    GABAergic,
    Glutamatergic,
    ReceptorArray,
    NMDAVoltageDependency,
    nmda_gating
