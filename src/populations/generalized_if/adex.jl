# abstract type AbstractAdEx <: AbstractGeneralizedIF end

@snn_kw mutable struct AdExParameter{FT = Float32} <: AbstractDoubleExpParameter
    C::FT = 281pF        #(pF)
    gL::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = C / gL # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = -70.6mV # Resting membrane potential 
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    Vspike::FT = 20mV # Spike potential
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
    τabs::FT = 1ms # Absolute refractory period

    ## Synapses
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale
end

@snn_kw struct AdExSinExpParameter{FT = Float32} <: AbstractSinExpParameter
    C::FT = 281pF        #(pF)
    gL::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = C / gL # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = -70.6mV # Resting membrane potential 
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    Vspike::FT = 20mV # Spike potential
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
    τabs::FT = 1ms # Absolute refractory period

    ## Synapses
    τe::FT = 6ms # Decay time for excitatory synapses
    τi::FT = 0.5ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses 
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale
end

@snn_kw struct AdExReceptorParameter{
    FT = Float32,
    VIT = Vector{Int},
    ST = SynapseArray,
    NMDAT = NMDAVoltageDependency{Float32},
    VFT = Vector{Float32},
} <: AbstractReceptorParameter
    C::FT = 281pF        #(pF)
    gL::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = C / gL # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -70.6mV # Reset potential
    El::FT = -70.6mV # Resting membrane potential 
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    Vspike::FT = 20mV # Spike potential
    τw::FT = 144ms # Adaptation time constant (Spike-triggered adaptation time scale)
    a::FT = 4nS # Subthreshold adaptation parameter
    b::FT = 80.5pA # Spike-triggered adaptation parameter (amount by which the voltage is increased at each threshold crossing)
    τabs::FT = 1ms # Absolute refractory period

    ## Dynamic spike threshold
    At::FT = 10mV # Post spike threshold increase
    τt::FT = 30ms # Adaptive threshold time scale

    ## Synapses
    NMDA::NMDAT = SomaNMDA
    glu_receptors::VIT = [1, 2]
    gaba_receptors::VIT = [3, 4]
    syn::ST = SomaSynapse
end


## Generalized integrate and fire
@snn_kw struct AdEx{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    VIT = Vector{Int},
    VBT = Vector{Bool},
    GIFParam<:AbstractGeneralizedIFParameter,
} <: AbstractGeneralizedIF
    name::String = "Generalized IF"
    id::String = randstring(12)

    param::GIFParam = AdExParameter()
    N::Int32 = 100 # Number of neurons
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w::VFT = zeros(N) # Adaptation current
    ξ_het::VFT = ones(N) # Membrane time constant

    fire::VBT = zeros(Bool, N) # Store spikes
    θ::VFT = ones(N) * param.Vt # Array with membrane potential thresholds
    tabs::VIT = ones(N) # Membrane time constant
    I::VFT = zeros(N) # Current

    # Two receptors synaptic conductance
    syn_curr::VFT = zeros(N)
    ge::VFT = hasproperty(param, :syn) ? zeros(0) : zeros(N) # Time-dependent conductance
    gi::VFT = hasproperty(param, :syn) ? zeros(0) : zeros(N) # Time-dependent conductance
    he::VFT = zeros(N)
    hi::VFT = zeros(N)

    # Glu/Gaba conductance
    g::MFT = hasproperty(param, :syn) ? zeros(N, 4) : zeros(0, 0)
    h::MFT = hasproperty(param, :syn) ? zeros(N, 4) : zeros(0, 0)
    glu::VFT = zeros(N)
    gaba::VFT = zeros(N)

    records::Dict = Dict()
end


function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target::Nothing = nothing,
) where {T<:AbstractGeneralizedIF}
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end


"""
	[Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""

function update_neuron!(
    p::P,
    param::T,
    dt::Float32,
) where {P<:AdEx,T<:AbstractGeneralizedIFParameter}
    @unpack N, v, w, fire, θ, I, ξ_het, tabs, syn_curr = p
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b, At, τt, τabs = param

    @inbounds for i ∈ 1:N
        # Reset membrane potential after spike
        v[i] = ifelse(fire[i], Vr, v[i])

        # Absolute refractory period
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end

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
            ) / (τm * ξ_het[i])
        # Double exponential
        θ[i] += dt * (Vt - θ[i]) / τt

        # Spike
        fire[i] = v[i] >= param.Vspike
        # fire[i] = v[i] > θ[i] + 5.0f0
        v[i] = ifelse(fire[i], 20.0f0, v[i]) # Set membrane potential to spike potential

        # Spike-triggered adaptation
        w[i] = ifelse(fire[i], w[i] + b, w[i])
        θ[i] = ifelse(fire[i], θ[i] + At, θ[i])

        # Absolute refractory period
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
        # increase adaptation current
    end
end

export AdEx, AdExParameter, AdExSinExpParameter, AdExReceptorParameter
