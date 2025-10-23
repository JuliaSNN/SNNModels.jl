abstract type AbstractReceptorParameter <: AbstractSynapseParameter end
abstract type AbstractReceptorVariable <: AbstractSynapseVariable end

"""
    MultiReceptorSynapseVars{MFT} <: AbstractReceptorVariable
A synaptic variable type that stores the state variables for receptor-based synaptic dynamics.
# Fields
- `N::Int`: Number of synapses
- `g::MFT`: Matrix of conductances for each receptor type
- `h::MFT`: Matrix of auxiliary variables for each receptor type
"""
MultiReceptorSynapseVars
@snn_kw struct MultiReceptorSynapseVars{MFT = Matrix{Float32}} <: AbstractReceptorVariable
    N::Int = 100
    g::MFT = zeros(Float32, N, 4)
    h::MFT = zeros(Float32, N, 4)
end

MultiReceptorSynapse(syn::ReceptorArray, NMDA::NMDAVoltageDependency{Float32}; kwargs...) =
    MultiReceptorSynapse(; kwargs..., syn = syn, NMDA = NMDA)

function synaptic_variables(synapse::MultiReceptorSynapse, N::Int)
    num_receptors = length(synapse.syn)
    return MultiReceptorSynapseVars(;
        N = N,
        g = zeros(Float32, N, num_receptors),
        h = zeros(Float32, N, num_receptors),
    )
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

export MultiReceptorSynapse
