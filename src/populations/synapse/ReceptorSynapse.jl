abstract type AbstractReceptorParameter <: AbstractSynapseParameter end

"""
    ReceptorSynapse{FT,VIT,ST,NMDAT,VFT} <: AbstractReceptorParameter

A synaptic parameter type that models receptor-based synaptic dynamics with NMDA voltage dependence.

# Fields
- `NMDA::NMDAT`: Parameters for NMDA voltage dependence (default: `NMDAVoltageDependency()`)
- `glu_receptors::VIT`: Indices of glutamate receptors (default: `[1, 2]`)
- `gaba_receptors::VIT`: Indices of GABA receptors (default: `[3, 4]`)
- `syn::ST`: Array of receptor parameters (default: `SomaReceptors`)

# Type Parameters
- `FT`: Floating point type (default: `Float32`)
- `VIT`: Vector of integers type (default: `Vector{Int}`)
- `ST`: Receptor array type (default: `ReceptorArray`)
- `NMDAT`: NMDA voltage dependency type (default: `NMDAVoltageDependency{Float32}`)
- `VFT`: Vector of floating point type (default: `Vector{Float32}`)

This type implements receptor-based synaptic dynamics with NMDA voltage dependence, where synaptic currents are calculated based on receptor activation and voltage-dependent NMDA modulation.
"""
ReceptorSynapse

@snn_kw struct ReceptorSynapse{
        VIT = Vector{Int},
        ST = Vector{Receptor{Float32}},
        NMDAT = NMDAVoltageDependency{Float32},
    } <: AbstractReceptorParameter
    ## Synapses
    NMDA::NMDAT = NMDAVoltageDependency()
    glu_receptors::VIT = [1, 2]
    gaba_receptors::VIT = [3, 4]
    syn::ST=SomaReceptors
end

ReceptorSynapseType = ReceptorSynapse{
        Vector{Int},
        Vector{Receptor{Float32}},
        NMDAVoltageDependency{Float32},
    }

@snn_kw struct ReceptorSynapseVars{MFT = Matrix{Float32}}  <: AbstractSynapseVariable
    N::Int = 100
    g::MFT = zeros(Float32, N, 4)
    h::MFT = zeros(Float32, N, 4)
end

ReceptorSynapse(syn::ReceptorArray, NMDA::NMDAVoltageDependency{Float32}; kwargs...) = ReceptorSynapse(; kwargs..., syn=syn, NMDA=NMDA)

function synaptic_variables(
    synapse::ReceptorSynapse,
    N::Int,
) 
    num_receptors = length(synapse.syn)
    return ReceptorSynapseVars(;
        N = N,
        g = zeros(Float32, N, num_receptors),
        h = zeros(Float32, N, num_receptors),
    )
end

@inline function update_synapses!(
    p::P,
    synapse::ReceptorSynapse,
    glu::Vector{Float32},
    gaba::Vector{Float32},
    synvars::ReceptorSynapseVars,
    dt::Float32,
) where {P<:AbstractPopulation}
    @unpack glu_receptors, gaba_receptors = synapse
    @unpack g, h = synvars
    @unpack N = p
    @inbounds for n in glu_receptors
        @unpack τr⁻, τd⁻, α = synapse.syn[n]
        @turbo for i ∈ 1:N
            h[i, n] += glu[i] * α
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end
    @simd for n in gaba_receptors
        @unpack τr⁻, τd⁻, α = synapse.syn[n]
        @turbo for i ∈ 1:N
            h[i, n] += gaba[i] * α
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end

    fill!(glu, 0.0f0)
    fill!(gaba, 0.0f0)
end

function update_synapses!(
    p::P,
    synapse::T,
    synvars::ReceptorSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractReceptorParameter}
    @unpack N, glu, gaba = p
    @unpack g, h = synvars
    update_synapses!(p, synapse, glu, gaba, synvars, dt)
    fill!(glu, 0.0f0)
    fill!(gaba, 0.0f0)
end

@inline function synaptic_current!(
    syn::ReceptorSynapse,
    v::Float32,
    synvars::ReceptorSynapseVars,
    is::Vector{Float32},
    comp::Int,
    neuron::Int,
)
    @unpack mg, b, k = syn.NMDA
    @unpack g = synvars
    is[comp] = 0.0f0
    @inbounds @fastmath begin
        @simd for n in eachindex(syn.syn)
            @unpack gsyn, E_rev, nmda = syn.syn[n]
            is[comp] +=
                gsyn *
                g[neuron, n] *
                (v - E_rev) *
                (nmda==0.0f0 ? 1.0f0 : 1/(1.0f0 + (mg / b) * exp256(k * v)))
        end
    end
    is[comp] = clamp(is[comp], -1500, 1500)
end

@inline function synaptic_current!(
    p::T,
    synapse::P,
    synvars::ReceptorSynapseVars,
) where {T<:AbstractGeneralizedIF,P<:AbstractReceptorParameter}
    @unpack N, v, syn_curr = p
    @unpack g, h = synvars
    @unpack syn, NMDA = synapse
    @unpack mg, b, k = NMDA
    fill!(syn_curr, 0.0f0)
    @inbounds @fastmath for n in eachindex(syn)
        @unpack gsyn, E_rev, nmda = syn[n]
        for neuron ∈ 1:N
            syn_curr[neuron] +=
                gsyn *
                g[neuron, n] *
                (v[neuron] - E_rev) *
                (nmda==0.0f0 ? 1.0f0 : 1/(1.0f0 + (mg / b) * exp256(k * v[neuron])))
        end
    end
    return
end

export ReceptorSynapse