# synapse.jl

"""
Receptor struct represents a synaptic receptor with parameters for reversal potential, rise time, decay time, and conductance.

# Fields
- `E_rev::T`: Reversal potential (default: 0.0)
- `τr::T`: Rise time constant (default: -1.0)
- `τd::T`: Decay time constant (default: -1.0)
- `g0::T`: Maximum conductance (default: 0.0)
- `gsyn::T`: Synaptic conductance (default: calculated based on `g0`, `τr`, and `τd`)
- `α::T`: Alpha factor for the differential equation (default: calculated based on `τr` and `τd`)
- `τr⁻::T`: Inverse of rise time constant (default: calculated based on `τr`)
- `τd⁻::T`: Inverse of decay time constant (default: calculated based on `τd`)
- `nmda::T`: NMDA factor (default: 0.0)
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
end

SynapseArray = Vector{Receptor{Float32}}
ReceptorVoltage = Receptor



"""
Synapse struct represents a synaptic connection with different types of receptors.

# Fields
- `AMPA::T`: AMPA receptor
- `NMDA::T`: NMDA receptor (with voltage dependency)
- `GABAa::T`: GABAa receptor
- `GABAb::T`: GABAb receptor
"""
Synapse

function Synapse(;AMPA::Receptor{T} = Receptor(),
                  NMDA::ReceptorVoltage{T} = ReceptorVoltage(),
                  GABAa::Receptor{T} = Receptor(),
                  GABAb::Receptor{T} = Receptor()) where {T <: Float32}
    return SynapseArray([AMPA, NMDA, GABAa, GABAb])
end

function Synapse(AMPA::Receptor{T},
                  NMDA::ReceptorVoltage{T},
                  GABAa::Receptor{T},
                  GABAb::Receptor{T}) where {T <: Float32}
    return SynapseArray([AMPA, NMDA, GABAa, GABAb])
end

"""
Glutamatergic struct represents a group of glutamatergic receptors.

# Fields
- `AMPA::T`: AMPA receptor
- `NMDA::T`: NMDA receptor
"""
Glutamatergic

@kwdef struct Glutamatergic
    AMPA::Receptor = Receptor()
    NMDA::ReceptorVoltage = ReceptorVoltage()
end

"""
GABAergic struct represents a group of GABAergic receptors.

# Fields
- `GABAa::T`: GABAa receptor
- `GABAb::T`: GABAb receptor
"""
GABAergic

@kwdef struct GABAergic
    GABAa::Receptor = Receptor()
    GABAb::Receptor = Receptor()
end

"""
Construct a Synapse from Glutamatergic and GABAergic receptors.

# Arguments
- `glu::Glutamatergic`: Glutamatergic receptors
- `gaba::GABAergic`: GABAergic receptors

# Returns
- `Synapse`: A Synapse object
"""
function Synapse(glu::Glutamatergic, gaba::GABAergic)
    return Synapse(glu.AMPA, glu.NMDA, gaba.GABAa, gaba.GABAb)
end

export Receptor, Synapse, ReceptorVoltage, GABAergic, Glutamatergic, SynapseArray, NMDAVoltageDependency

"""
Calculate the normalization factor for a synapse.

# Arguments
- `synapse::Receptor`: The receptor for which to calculate the normalization factor

# Returns
- `Float32`: The normalization factor
"""
function norm_synapse(synapse::Receptor)
    norm_synapse(synapse.τr, synapse.τd)
end

"""
Calculate the normalization factor for a synapse given rise and decay time constants.

# Arguments
- `τr`: Rise time constant
- `τd`: Decay time constant

# Returns
- `Float32`: The normalization factor
"""
function norm_synapse(τr, τd)
    p = [1, τr, τd]
    t_p = p[2] * p[3] / (p[3] - p[2]) * log(p[3] / p[2])
    return 1 / (-exp(-t_p / p[2]) + exp(-t_p / p[3]))
end

"""
Calculate the alpha factor for a synapse given rise and decay time constants.

# Arguments
- `τr`: Rise time constant
- `τd`: Decay time constant

# Returns
- `Float32`: The alpha factor
"""
function α_synapse(τr, τd)
    return (τd - τr) / (τd * τr)
end

Mg_mM = 1.0f0
nmda_b = 3.36   # voltage dependence of nmda channels
nmda_k = -0.077     # Eyal 2018

"""
NMDAVoltageDependency struct represents the voltage dependence of NMDA receptors.

# Fields
- `b::T`: Voltage dependence factor (default: 3.36)
- `k::T`: Voltage dependence factor (default: -0.077)
- `mg::T`: Magnesium concentration (default: 1.0)
"""
NMDAVoltageDependency
@snn_kw struct NMDAVoltageDependency{T<:Float32}
    b::T = nmda_b
    k::T = nmda_k
    mg::T = Mg_mM
end

function nmda_gating(v, NMDA::NMDAVoltageDependency)
    @unpack b, k, mg = NMDA
    return 1/ (1.0f0 + (mg / b) * exp256(k * v))
end

export norm_synapse,
    EyalNMDA,
    Receptor,
    Synapse,
    ReceptorVoltage,
    GABAergic,
    Glutamatergic,
    SynapseArray,
    synapsearray,
    NMDAVoltageDependency,
    nmda_gating
