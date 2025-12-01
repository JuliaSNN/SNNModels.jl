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
    target::Symbol = :none
end

ReceptorArray = Vector{Receptor{Float32}}
ReceptorVoltage = Receptor



"""
Receptors struct represents a synaptic connection with different types of receptors.

# Fields
- `AMPA::T`: AMPA receptor
- `NMDA::T`: NMDA receptor (with voltage dependency)
- `GABAa::T`: GABAa receptor
- `GABAb::T`: GABAb receptor
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

# Throws
- Error: If any receptor in the input array has a `target` field set to `:none`

# Example

```julia
receptors = ReceptorArray([
    Receptor(target=:glu),
    Receptor(target=:gaba),
    Receptor(target=:glu),
])  
result = infer_receptors(receptors)
# result will be (; glu = [1, 3], gaba = [2])
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
Construct a Receptors from Glutamatergic and GABAergic receptors.

# Arguments
- `glu::Glutamatergic`: Glutamatergic receptors
- `gaba::GABAergic`: GABAergic receptors

# Returns
- `Receptors`: A Receptors object
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
Calculate the normalization factor for a receptor.

# Arguments
- `receptor::Receptor`: The receptor for which to calculate the normalization factor

# Returns
- `Float32`: The normalization factor
"""
function norm_synapse(receptor::Receptor)
    norm_synapse(receptor.τr, receptor.τd)
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
    t_p = τr * τd / (τd - τr) * log(τd / τr)
    return 1 / (-exp(-t_p / τr) + exp(-t_p / τd))
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

@snn_kw struct NMDAVoltageDependency{T = Float32}
    b::T = nmda_b
    k::T = nmda_k
    mg::T = Mg_mM
end

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
    synapsearray,
    NMDAVoltageDependency,
    nmda_gating
