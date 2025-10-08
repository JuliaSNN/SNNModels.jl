# abstract type AbstractAdEx <: AbstractGeneralizedIF end

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

## Generalized integrate and fire
@snn_kw struct AdEx{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    VIT = Vector{Int},
    VBT = Vector{Bool},
    GIFT<:AbstractGeneralizedIFParameter,
    SYNT<:AbstractSynapseParameter
} <: AbstractGeneralizedIF
name::String = "AdEx"
    id::String = randstring(12)

    param::GIFT  = AdExParameter()
    synapse::SYNT = DoubleExpSynapse()
    spike::PostSpike = PostSpike()

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
    ge::VFT = synapse isa AbstractReceptorParameter ? zeros(0) : zeros(N) # Time-dependent conductance
    gi::VFT = synapse isa AbstractReceptorParameter ? zeros(0) : zeros(N) # Time-dependent conductance
    he::VFT = zeros(N)
    hi::VFT = zeros(N)

    # Glu/Gaba conductance
    g::MFT = synapse isa AbstractReceptorParameter ? zeros(N, 4) : zeros(0, 0)
    h::MFT = synapse isa AbstractReceptorParameter ? zeros(N, 4) : zeros(0, 0)
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


function Population(param::AdExParameter, synapse::AbstractSynapseParameter; N, kwargs...)
    return AdEx(;N, param, synapse, kwargs...)
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
    @unpack τm, Vt, Vr, El, R, ΔT, τw, a, b =param
    @unpack At, τA, τabs = p.spike

    # @inbounds 
    for i ∈ 1:N
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
        θ[i] += dt * (Vt - θ[i]) / τA

        # Spike
        fire[i] = v[i] >= 0mV#$param.AP_membrane
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

export AdEx, AdExParameter
