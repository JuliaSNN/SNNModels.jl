
## Synaptic updates 
abstract type AbstractSynapseParameter end
abstract type AbstractSinExpParameter <: AbstractSynapseParameter end
abstract type AbstractDoubleExpParameter <: AbstractSynapseParameter end
abstract type AbstractReceptorParameter <: AbstractSynapseParameter end
abstract type AbstractCurrentParameter <: AbstractSynapseParameter end
abstract type AbstractDeltaParameter <: AbstractSynapseParameter end


## Receptor Receptors updates
# NMDA::NMDAVoltageDependency = NMDAVoltageDependency(
#     b = 3.36,  # NMDA voltage dependency parameter
#     k = -0.077,  # NMDA voltage dependency parameter
#     mg = 1.0f0,  # NMDA voltage dependency parameter
# ),
@snn_kw struct ReceptorSynapse{
        FT = Float32,
        VIT = Vector{Int},
        ST = ReceptorArray,
        NMDAT = NMDAVoltageDependency{Float32},
        VFT = Vector{Float32},
    } <: AbstractReceptorParameter
    ## Synapses
    NMDA::NMDAT = NMDAVoltageDependency()
    glu_receptors::VIT = [1, 2]
    gaba_receptors::VIT = [3, 4]
    syn::ST=SomaReceptors
end

@inline function update_synapses!(
    p::P,
    synapse::ReceptorSynapse,
    glu::Vector{Float32},
    gaba::Vector{Float32},
    g::Matrix{Float32},
    h::Matrix{Float32},
    dt::Float32,
) where {P<:AbstractPopulation}
    @unpack glu_receptors, gaba_receptors = synapse
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
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractReceptorParameter}
    @unpack N, g, h, glu, gaba, hi, he = p
    update_synapses!(p, synapse, glu, gaba, g, h, dt)
    fill!(hi, 0.0f0)
    fill!(he, 0.0f0)
end


@inline function synaptic_current!(
    syn::ReceptorSynapse,
    v::Float32,
    g,
    is::Vector{Float32},
    comp::Int,
    neuron::Int,
)
    @unpack mg, b, k = syn.NMDA
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
) where {T<:AbstractGeneralizedIF,P<:AbstractReceptorParameter}
    @unpack N, g, h, g, v, syn_curr = p
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

## Double Exponential Receptors updates
@snn_kw struct DoubleExpSynapse{FT = Float32} <: AbstractDoubleExpParameter
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
end

function update_synapses!(
    p::P,
    synapse::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpParameter}
    @unpack N, ge, gi, he, hi = p
    @unpack τde, τre, τdi, τri = synapse
    @inbounds for i ∈ 1:N
        ge[i] += dt * (-ge[i] / τde + he[i])
        he[i] += dt * (-he[i] / τre)
        gi[i] += dt * (-gi[i] / τdi + hi[i])
        hi[i] += dt * (-hi[i] / τri)
    end
end

@inline function synaptic_current!(
    p::P,
    param::T,
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpParameter}
    @unpack gsyn_e, gsyn_i, E_e, E_i = param
    @unpack N, v, ge, gi, syn_curr = p
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

## Single Exponential Receptors updates
@snn_kw struct SingleExpSynapse{FT = Float32} <: AbstractSinExpParameter
    ## Synapses
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 0.5ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
end
function update_synapses!(
    p::P,
    synapse::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    @unpack N, ge, gi, he, hi = p
    @unpack τe, τi = synapse
    @fastmath @inbounds for i ∈ 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    @unpack gsyn_e, gsyn_i, E_e, E_i = synapse
    @unpack N, v, ge, gi, syn_curr = p
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

## Delta Receptors updates
@snn_kw struct DeltaSynapse{FT = Float32} <: AbstractDeltaParameter
end

@inline function update_synapses!(
    p::P,
    synapse::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter}
    @unpack N, ge, gi = p
    # @inbounds for i = 1:N
    # end
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter}
    @unpack N, v, ge, gi, syn_curr = p
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = -(ge[i] - gi[i])
        ge[i] = 0.0f0
        gi[i] = 0.0f0
    end
end


## Current Receptors updates

@snn_kw struct CurrentSynapse{FT = Float32} <: AbstractCurrentParameter
    τe::FT = 6ms # Rise time for excitatory synapses
    τi::FT = 2ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
end

@inline function update_synapses!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractCurrentParameter}
    @unpack N, ge, gi = p
    @unpack τe, τi = param
    @fastmath @inbounds for i ∈ 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
end

@inline function synaptic_current!(
    p::P,
    param::T,
) where {P<:AbstractGeneralizedIF,T<:AbstractCurrentParameter}
    @unpack E_e, E_i = param
    @unpack N, v, ge, gi, syn_curr = p
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = -(ge[i] - gi[i])
    end
end


## Synaptic currents


# if nmda > 0.0f0
#     @simd for neuron ∈ 1:N
#         syn_curr[i] +=
#             gsyn * g[i, r] * (v[i] - E_rev) / (1.0f0 + (mg / b) * exp256(k * (v[i])))
#     end
# else
#     @simd for i ∈ 1:N
#         syn_curr[i] += gsyn * g[i, r] * (v[i] - E_rev)
#     end

export ReceptorSynapse, DoubleExpSynapse, SingleExpSynapse, CurrentSynapse, DeltaSynapse