abstract type AbstractReceptorParameter <: AbstractSynapseParameter end
abstract type AbstractReceptorVariable <: AbstractSynapseVariable end

"""
    ReceptorSynapse{FT,VIT,ST,NMDAT,VFT} <: AbstractReceptorParameter

A synaptic parameter type that models receptor-based synaptic dynamics with NMDA voltage dependence.

# Fields
- `NMDA::NMDAT`: Parameters for NMDA voltage dependence (default: `NMDAVoltageDependency()`)
- `glu_receptors::VIT`: Indices of glutamate receptors (default: `[1, 2]`)
- `gaba_receptors::VIT`: Indices of GABA receptors (default: `[3, 4]`)
- `syn::ST`: Array of receptor parameters (default: `SomaReceptors`)

# Type Parameters
- `VIT`: Vector of integers type (default: `Vector{Int}`)
- `ST`: Receptor array type (default: `ReceptorArray`)
- `NMDAT`: NMDA voltage dependency type (default: `NMDAVoltageDependency{Float32}`)

This type implements conductance-based synaptic dynamics with AMPA, NMDA, GABAa, and GABAb receptors. Synaptic currents are calculated based on receptor activation and voltage-dependent NMDA modulation.
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

ReceptorSynapse(syn::ReceptorArray, NMDA::NMDAVoltageDependency{Float32}; kwargs...) = ReceptorSynapse(; kwargs..., syn = syn, NMDA = NMDA)

function synaptic_receptors(synapse:: ReceptorSynapse, N::Int) 
    return (glu=zeros(Float32, N), gaba=zeros(Float32, N))
end


"""
    ReceptorSynapse{FT,VIT,ST,NMDAT,VFT} <: AbstractReceptorParameter

A synaptic parameter type that models receptor-based synaptic dynamics with NMDA voltage dependence.

# Fields
- `NMDA::NMDAT`: Parameters for NMDA voltage dependence (default: `NMDAVoltageDependency()`)
- `glu_receptors::VIT`: Indices of glutamate receptors (default: `[1, 2]`)
- `gaba_receptors::VIT`: Indices of GABA receptors (default: `[3, 4]`)
- `syn::ST`: Array of receptor parameters (default: `SomaReceptors`)

# Type Parameters
- `VIT`: Vector of integers type (default: `Vector{Int}`)
- `ST`: Receptor array type (default: `ReceptorArray`)
- `NMDAT`: NMDA voltage dependency type (default: `NMDAVoltageDependency{Float32}`)

This type implements conductance-based synaptic dynamics with AMPA, NMDA, GABAa, and GABAb receptors. Synaptic currents are calculated based on receptor activation and voltage-dependent NMDA modulation.
"""
MultiReceptorSynapse

@snn_kw struct MultiReceptorSynapse{
    ST = Vector{Receptor{Float32}},
    NMDAT = NMDAVoltageDependency{Float32},
    REC <: NamedTuple
} <: AbstractReceptorParameter
    ## Synapses
    NMDA::NMDAT = NMDAVoltageDependency()
    syn::ST=SomaReceptors
    receptors::REC = infer_receptors(syn)
end

MultiReceptorSynapse(syn::ReceptorArray, NMDA::NMDAVoltageDependency{Float32}, receptors::NamedTuple) = ReceptorSynapse(; syn = syn, NMDA = NMDA, receptors = receptors)

function synaptic_receptors(synapse::MultiReceptorSynapse, N::Int)

    receptors = Dict{Symbol, Vector{Float32}}()
    for rec in keys(synapse.receptors)
        receptors[rec] = zeros(Float32, N)
    end
    return (; receptors...) 
end

"""
    ReceptorSynapseVars{MFT} <: AbstractReceptorVariable
A synaptic variable type that stores the state variables for receptor-based synaptic dynamics.
# Fields
- `N::Int`: Number of synapses
- `g::MFT`: Matrix of conductances for each receptor type
- `h::MFT`: Matrix of auxiliary variables for each receptor type
"""
ReceptorSynapseVars
@snn_kw struct ReceptorSynapseVars{MFT = Matrix{Float32}} <: AbstractReceptorVariable
    N::Int = 100
    g::MFT = zeros(Float32, N, 4)
    h::MFT = zeros(Float32, N, 4)
end


function synaptic_variables(synapse::T, N::Int) where {T<:AbstractReceptorParameter}
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
    receptors::RECT,
    synvars::ReceptorSynapseVars,
    dt::Float32,
) where {P<:AbstractPopulation, RECT<:NamedTuple}
    @unpack glu_receptors, gaba_receptors = synapse
    @unpack N, g, h = synvars
    @unpack glu, gaba = receptors
    for name in keys(receptors)
        fill!(getfield(receptors, name), 0.0f0)
    end
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

@inline function update_synapses!(
    p::P,
    synapse::MultiReceptorSynapse,
    receptors::RECT,
    synvars::ReceptorSynapseVars,
    dt::Float32,
) where {P<:AbstractPopulation, RECT<:NamedTuple}
    for name in keys(synapse.receptors)
        ids = synapse.receptors[name]
        target::Vector{Float32} = getfield(receptors, name)
        @inbounds for n in ids
            update_receptor!(synvars, synapse.syn[n], target, n,  dt)
        end
    end
    for name in keys(receptors)
        fill!(getfield(receptors, name), 0.0f0)
    end
end

@inline function update_receptor!(
    synvars::T,
    receptor::Receptor{Float32},
    target::Vector{Float32},
    n ::Int,
    dt::Float32,     
) where {T<:AbstractReceptorVariable}
    @unpack N, g, h = synvars
    @unpack τr⁻, τd⁻, α = receptor
    for i ∈ 1:N
        h[i, n] += target[i] * α
        g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
        h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
    end
end





@inline function synaptic_current!(
    p::T,
    synapse::P,
    synvars::S,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {
    T<:AbstractGeneralizedIF,
    P<:AbstractReceptorParameter,
    S<:AbstractReceptorVariable,
    VT1<:AbstractVector,
    VT2<:AbstractVector,
}
    @unpack N = p
    @unpack g, h = synvars
    @unpack syn, NMDA = synapse
    @unpack mg, b, k = NMDA
    fill!(syncurr, 0.0f0)
    # @inbounds @fastmath 
    for n in eachindex(syn)
        @unpack gsyn, E_rev, nmda = syn[n]
        for neuron ∈ 1:N
            syncurr[neuron] +=
                gsyn *
                g[neuron, n] *
                (v[neuron] - E_rev) *
                (nmda==0.0f0 ? 1.0f0 : 1/(1.0f0 + (mg / b) * exp256(k * v[neuron])))
        end
    end
    # @show syncurr
end

export ReceptorSynapse, MultiReceptorSynapse
