
## Synaptic updates 
abstract type AbstractSinExpParameter <: AbstractGeneralizedIFParameter end
abstract type AbstractDoubleExpParameter <: AbstractGeneralizedIFParameter end
abstract type AbstractReceptorParameter <: AbstractGeneralizedIFParameter end
abstract type AbstractCurrentParameter <: AbstractGeneralizedIFParameter end
abstract type AbstractDeltaParameter <: AbstractGeneralizedIFParameter end

## Receptor Synapse updates

function update_synapses!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractReceptorParameter}
    @unpack N, g, h, glu, gaba, hi, he = p
    @unpack glu_receptors, gaba_receptors, syn = param

    @inbounds for n in glu_receptors
        @unpack τr⁻, τd⁻, α = syn[n]
        @turbo for i ∈ 1:N
            h[i, n] += (he[i] + glu[i]) * α
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end

    @inbounds for n in gaba_receptors
        @unpack τr⁻, τd⁻, α = syn[n]
        @turbo for i ∈ 1:N
            h[i, n] += (hi[i] + gaba[i]) * α
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end

    fill!(gaba, 0.0f0)
    fill!(glu, 0.0f0)
    fill!(hi, 0.0f0)
    fill!(he, 0.0f0)
end

@inline function synaptic_current!(
    p::T,
    param::P,
) where {T<:AbstractGeneralizedIF,P<:AbstractReceptorParameter}
    @unpack N, g, h, g, v, syn_curr = p
    @unpack syn, NMDA = param
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

## Double Exponential Synapse updates

function update_synapses!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpParameter}
    @unpack N, ge, gi, he, hi = p
    @unpack τde, τre, τdi, τri = param
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

## Single Exponential Synapse updates

function update_synapses!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    @unpack N, ge, gi, he, hi = p
    @unpack τe, τi = param
    @fastmath @inbounds for i ∈ 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
end

@inline function synaptic_current!(
    p::P,
    param::T,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    @unpack gsyn_e, gsyn_i, E_e, E_i = param
    @unpack N, v, ge, gi, syn_curr = p
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

## Delta Synapse updates

@inline function update_synapses!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter}
    @unpack N, ge, gi = p
    # @inbounds for i = 1:N
    # end
end

@inline function synaptic_current!(
    p::P,
    param::T,
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter}
    @unpack N, v, ge, gi, syn_curr = p
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = -(ge[i] - gi[i])
        ge[i] = 0.0f0
        gi[i] = 0.0f0
    end
end


## Current Synapse updates
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
