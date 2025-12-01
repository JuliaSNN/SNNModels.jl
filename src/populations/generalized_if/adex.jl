"""
    AdExParameter{FT} <: AbstractGeneralizedIFParameter

The AdEx model extends the leaky integrate-and-fire model with exponential spiking dynamics and spike-triggered adaptation.
This implementation follows the parameterization from Brette and Gerstner (2005).

# Fields
- `C::FT`: Membrane capacitance (default: 281 pF)
- `gl::FT`: Leak conductance (default: 40 nS)
- `Vt::FT`: Membrane potential threshold (default: -50 mV)
- `Vr::FT`: Reset potential (default: -70.6 mV)
- `El::FT`: Resting membrane potential (default: -70.6 mV)
- `τm::FT`: Membrane time constant (default: C/gl)
- `R::FT`: Resistance (default: nS/gl)
- `ΔT::FT`: Slope factor (default: 2 mV)
- `τw::FT`: Adaptation time constant (default: 144 ms)
- `a::FT`: Subthreshold adaptation parameter (default: 4 nS)
- `b::FT`: Spike-triggered adaptation parameter (default: 80.5 pA)

"""
AdExParameter
@snn_kw mutable struct AdExParameter{FT = Float32} <: AbstractGeneralizedIFParameter
    C::FT = 281pF        #(pF)
    gl::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = -70.6mV # Resting membrane potential 
    τm::FT = C / gl # Membrane time constant
    R::FT = nS / gl # Resistance
    ΔT::FT = 2mV # Slope factor
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
end



"""
    AdEx{VFT, MFT,  GIFT, SYNT} <: AbstractGeneralizedIF

The AdEx model implements the adaptive exponential integrate-and-fire neuron model with support for any synaptic model.

# Fields
## Population Info
- `name::String`: Name of the neuron model (default: "AdEx")
- `id::String`: Unique identifier for the neuron population (default: random 12-character string)
- `records::Dict`: Dictionary for storing simulation records (initialized empty)

## Model Parameters
- `param::GIFT`: Parameters for the AdEx model (default: `AdExParameter()`)
- `synapse::SYNT`: Synaptic parameters (default: `DoubleExpSynapse()`)
- `spike::PST`: Post-spike parameters (default: `PostSpike()`)
- `N::Int32`: Number of neurons in the population (default: 100)

## Model variable
- `v::VFT`: Membrane potential (initialized randomly between `Vr` and `Vt`)
- `w::VFT`: Adaptation current (initialized to zeros)
- `fire::VBT`: Spike flags (initialized to false)
- `θ::VFT`: Membrane potential thresholds (initialized to `Vt`)
- `tabs::VIT`: Absolute refractory period counters (initialized to ones)
- `I::VFT`: External current (initialized to zeros)
- `syn_curr::VFT`: Total synaptic current (initialized to zeros)

## Synapses
- `synvars::SYNV`: Synaptic variables for the synapse model
- `receptors<:NamedTuple`: Synaptic receptors triggered by spike events

"""
AdEx

@snn_kw struct AdEx{
    IT = Int32,
    VFT = Vector{Float32},
    PST<:PostSpike,
    SYNT<:AbstractSynapseParameter,
    SYNV<:AbstractSynapseVariable,
    AdExt<:AdExParameter,
    RECT<:NamedTuple,
} <: AbstractGeneralizedIF

    name::String = "AdEx"
    id::String = randstring(12)

    param::AdExt = AdExParameter()
    synapse::SYNT = DoubleExpSynapse()
    spike::PST = PostSpike()

    N::IT = 100 # Number of neurons
    v::VFT = param.Vr .+ rand(Float32, N) .* (param.Vt - param.Vr)
    w::VFT = zeros(Float32, N) # Adaptation current

    fire::VBT = zeros(Bool, N) # Store spikes
    θ::VFT = ones(Float32, N) .* param.Vt # Array with membrane potential thresholds
    tabs::VIT = ones(Int, N) # Membrane time constant
    I::VFT = zeros(Float32, N) # Current

    # Two receptors synaptic conductance
    syn_curr::VFT = zeros(Float32, N)
    synvars::SYNV = synaptic_variables(synapse, N) # Synaptic variables for receptor model
    receptors::RECT = synaptic_receptors(synapse, N)

    records::Dict = Dict()
end


function synaptic_target(targets::Dict, post::T, sym::Symbol, target) where {T<:AdEx}
    syn = get_synapse_symbol(post.synapse, sym)
    sym = Symbol(syn)
    g = getfield(post.receptors, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end


function Population(
    param::AdExParameter;
    synapse::AbstractSynapseParameter,
    N,
    spike = PostSpike(),
    kwargs...,
)
    return AdEx(; N, param, synapse, spike, SYNT = typeof(synapse), kwargs...)
end


"""
	[Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function update_neuron!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AdEx,T<:AdExParameter{Float32}}
    @unpack N, v, w, fire, θ, I, tabs, syn_curr = p
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b = param
    @unpack At, τA, τabs = p.spike

    @inbounds for i ∈ 1:N
        # Reset membrane potential after spike
        v[i] = ifelse(fire[i], Vr, v[i])

        # Absolute refractory period
        fire[i] = false
        tabs[i] -= 1
        tabs[i] > 0 && continue

        # Adaptation current 
        w[i] += dt * (a * (v[i] - El) - w[i]) / τw
        # Membrane potential
        v[i] +=
            dt * (
                -(v[i] - El)  # leakage
                + (ΔT < 0.0f0 ? 0.0f0 : ΔT * exp((v[i] - θ[i]) / ΔT)) # exponential term
                - R * syn_curr[i] # excitatory synapses
                - R * w[i] # adaptation
                + R * I[i] # external current
            ) / (τm[i])

        θ[i] += dt * (Vt - θ[i]) / τA
        fire[i] = v[i] >= 0mV#$param.AP_membrane
        v[i] = ifelse(fire[i], 20.0f0, v[i]) # Set membrane potential to spike potential
        w[i] = ifelse(fire[i], w[i] + b, w[i])
        θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
end


function update_neuron!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AdEx,T<:AdExParameter{Vector{Float32}}}
    @unpack N, v, w, fire, θ, I, tabs, syn_curr = p
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b = param
    @unpack At, τA, τabs = p.spike

    @inbounds for i ∈ 1:N
        v[i] = ifelse(fire[i], Vr[i], v[i])
        # Absolute refractory period
        fire[i] = false
        tabs[i] -= 1
        tabs[i] > 0 && continue

        w[i] += dt * (a[i] * (v[i] - El[i]) - w[i]) / τw[i]
        v[i] +=
            dt * (
                -(v[i] - El[i])  # leakage
                +
                (ΔT[i] < 0.0f0 ? 0.0f0 : ΔT[i] * exp((v[i] - θ[i]) / ΔT[i])) # exponential term
                -
                R[i] * syn_curr[i] # excitatory synapses
                - R[i] * w[i] # adaptation
                + R[i] * I[i] # external current
            ) / (τm[i])

        θ[i] += dt * (Vt[i] - θ[i]) / τA
        fire[i] = v[i] >= 0mV#$param.AP_membrane
        v[i] = ifelse(fire[i], 20.0f0, v[i]) # Set membrane potential to spike potential
        w[i] = ifelse(fire[i], w[i] + b[i], w[i])
        θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end
end


export AdEx, AdExParameter
