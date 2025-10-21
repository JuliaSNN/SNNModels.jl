
## Synaptic updates 
abstract type AbstractSynapseParameter <: AbstractComponent end
abstract type AbstractSinExpParameter <: AbstractSynapseParameter end
abstract type AbstractDoubleExpParameter <: AbstractSynapseParameter end
abstract type AbstractReceptorParameter <: AbstractSynapseParameter end
abstract type AbstractCurrentParameter <: AbstractSynapseParameter end
abstract type AbstractDeltaParameter <: AbstractSynapseParameter end

abstract type AbstractSynapseVariable <: AbstractComponent end



get_synapse_symbol(post::AbstractPopulation, sym::Symbol) = sym

get_synapse_symbol(post::T, sym::Symbol) where {T<:AbstractGeneralizedIF} =
    get_synapse_symbol(post.synapse, sym) # for dendrites, we assume the target is always :d

function get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractDoubleExpParameter}
    sym == :glu && return :glu
    sym == :gaba && return :gaba
    sym == :he && return :glu
    sym == :hi && return :gaba
    sym == :ge && return :glu
    sym == :gi && return :gaba
    error("Synapse symbol $sym not found in DoubleExpSynapse")
end

function synaptic_variables(
    synapse::ReceptorSynapse,
    N::Int,
)
    return ReceptorSynapseVars(N = N)
end


function get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractSinExpParameter}
    sym == :glu && return :glu
    sym == :gaba && return :gaba
    sym == :ge && return :glu
    sym == :gi && return :gaba
    error("Synapse symbol $sym not found in SingleExpSynapse")
end

function synaptic_variables(
    synapse::SingleExpSynapse,
    N::Int,
)
    return SingleExpSynapseVars(N = N)
end

function get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractDeltaParameter}
    sym == :glu && return :glu
    sym == :gaba && return :gaba
    sym == :ge && return :glu
    sym == :gi && return :gaba
    error("Synapse symbol $sym not found in CurrentSynapse")
end
function synaptic_variables(
    synapse::DeltaSynapse,
    N::Int,
)
    return DeltaSynapseVars(N = N)
end


function get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractCurrentParameter}
    return sym
end

function synaptic_variables(
    synapse::CurrentSynapse,
    N::Int,
)
    return CurrentSynapseVars(N = N)
end


function get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractReceptorParameter}
    return sym
end

function synaptic_variables(
    synapse::ReceptorSynapse,
    N::Int,
)
    return ReceptorSynapseVars(N = N)
end


## Receptor Receptors updates
# NMDA::NMDAVoltageDependency = NMDAVoltageDependency(
#     b = 3.36,  # NMDA voltage dependency parameter
#     k = -0.077,  # NMDA voltage dependency parameter
#     mg = 1.0f0,  # NMDA voltage dependency parameter
# ),
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

@snn_kw ReceptorSynapseVars{VFT = Vector{Float32}, MFT = Matrix{Float32}}  <: AbstractSynapseVariable
begin
    N::Int = 100
    g::MFT = zeros(Float32, N, 4)
    h::MFT = zeros(Float32, N, 4)
end

ReceptorSynapse(syn::ReceptorArray, NMDA::NMDAVoltageDependency{Float32}; kwargs...) = ReceptorSynapse(; kwargs..., syn=syn, NMDA=NMDA)

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

"""
    DoubleExpSynapse{FT} <: AbstractDoubleExpParameter

A synaptic parameter type that models double exponential synaptic dynamics.

# Fields
- `τre::FT`: Rise time constant for excitatory synapses (default: 1ms)
- `τde::FT`: Decay time constant for excitatory synapses (default: 6ms)
- `τri::FT`: Rise time constant for inhibitory synapses (default: 0.5ms)
- `τdi::FT`: Decay time constant for inhibitory synapses (default: 2ms)
- `E_i::FT`: Reversal potential for inhibitory synapses (default: -75mV)
- `E_e::FT`: Reversal potential for excitatory synapses (default: 0mV)
- `gsyn_e::FT`: Synaptic conductance for excitatory synapses (default: 1.0f0)
- `gsyn_i::FT`: Synaptic conductance for inhibitory synapses (default: 1.0f0)

# Type Parameters
- `FT`: Floating point type (default: `Float32`)

This type implements double exponential synaptic dynamics, where synaptic currents are calculated using separate rise and decay time constants for both excitatory and inhibitory synapses.
"""
DoubleExpSynapse
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

@snn_kw DoubleExpSynapseVars{VFT = Vector{Float32}}  <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
    he::VFT = zeros(Float32, N)
    hi::VFT = zeros(Float32, N)
begin


function update_synapses!(
    p::P,
    synapse::T,
    synvars::DoubleExpSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpParameter}
    @unpack N, ge, gi, he, hi = synvars
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
    synvars::DoubleExpSynapseVars
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpParameter}
    @unpack gsyn_e, gsyn_i, E_e, E_i = param
    @unpack N, v, syn_curr = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

"""
    SingleExpSynapse{FT} <: AbstractSinExpParameter

A synaptic parameter type that models single exponential synaptic dynamics.

# Fields
- `τe::FT`: Decay time constant for excitatory synapses (default: 6ms)
- `τi::FT`: Rise time constant for inhibitory synapses (default: 0.5ms)
- `E_i::FT`: Reversal potential for inhibitory synapses (default: -75mV)
- `E_e::FT`: Reversal potential for excitatory synapses (default: 0mV)
- `gsyn_e::FT`: Synaptic conductance for excitatory synapses (default: 1.0f0)
- `gsyn_i::FT`: Synaptic conductance for inhibitory synapses (default: 1.0f0)

# Type Parameters
- `FT`: Floating point type (default: `Float32`)

This type implements single exponential synaptic dynamics, where synaptic currents are calculated using separate time constants for both excitatory and inhibitory synapses.
"""
SingleExpSynapse
@snn_kw struct SingleExpSynapse{FT = Float32} <: AbstractSinExpParameter
    ## Synapses
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 0.5ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
end

@snn_kw SingleExpSynapseVars{VFT = Vector{Float32}}  <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)
begin


function update_synapses!(
    p::P,
    synapse::T,
    synvars::SingleExpSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    @unpack N, ge, gi = p
    @unpack τe, τi = synapse
    @fastmath @inbounds for i ∈ 1:N
        ge[i] += dt * (-ge[i] / τe)
        gi[i] += dt * (-gi[i] / τi)
    end
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::SingleExpSynapseVars,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    @unpack gsyn_e, gsyn_i, E_e, E_i = synapse
    @unpack N, v, ge, gi, syn_curr = p
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

## Delta Receptors updates
"""
    DeltaSynapse{FT} <: AbstractDeltaParameter

A synaptic parameter type that models delta (instantaneous) synaptic dynamics.

# Fields
None - this type implements instantaneous synaptic dynamics where synaptic inputs are applied directly without any time constants.

# Type Parameters
- `FT`: Floating point type (default: `Float32`)

This type implements delta synaptic dynamics, where synaptic inputs are applied instantaneously without any time delays or decay. The synaptic current is calculated as the difference between excitatory and inhibitory inputs.
"""
struct DeltaSynapse <: AbstractDeltaParameter end

@snn_kw DeltaSynapseVars{VFT = Vector{Float32}}  <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(Float32, N)
    gi::VFT = zeros(Float32, N)

@inline function update_synapses!(
    p::P,
    synapse::T,
    synvars::DeltaSynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter}
end

@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::DeltaSynapseVars,
) where {P<:AbstractGeneralizedIF,T<:AbstractDeltaParameter}
    @unpack N, v, syn_curr = p
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ 1:N
        syn_curr[i] = -(ge[i] - gi[i])
        ge[i] = 0.0f0
        gi[i] = 0.0f0
    end
end


## Current Receptors updates

"""
    CurrentSynapse{FT} <: AbstractCurrentParameter

A synaptic parameter type that models current-based synaptic dynamics.

# Fields
- `τe::FT`: Decay time constant for excitatory synapses (default: 6ms)
- `τi::FT`: Decay time constant for inhibitory synapses (default: 2ms)
- `E_i::FT`: Reversal potential for inhibitory synapses (default: -75mV)
- `E_e::FT`: Reversal potential for excitatory synapses (default: 0mV)

# Type Parameters
- `FT`: Floating point type (default: `Float32`)

This type implements current-based synaptic dynamics, where synaptic currents are calculated using separate time constants for both excitatory and inhibitory synapses.
"""
CurrentSynapse
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


export ReceptorSynapse, DoubleExpSynapse, SingleExpSynapse, CurrentSynapse, DeltaSynapse, AbstractSynapseParameter