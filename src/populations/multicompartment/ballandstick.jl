# BallAndStick
"""
    BallAndStick{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    VDT = Dendrite{Vector{Float32}},
    SYND <: AbstractSynapseParameter,
    SYNS <: AbstractSynapseParameter,
    SYNDV <: AbstractSynapseVariable,
    SYNSV <: AbstractSynapseVariable,
    SOMAT <: AbstractGeneralizedIFParameter,
    PST <: AbstractSpikeParameter,
    IT = Int32,
    } <: AbstractDendriteIF

A ball-and-stick neuron model with dendritic and somatic compartments.
The model incorporates adaptive exponential integrate-and-fire dynamics
with synaptic inputs to both the soma and dendrites. The soma includes adaptation currents and dynamic thresholds for spike generation.
The dendrite is modeled with separate passive compartments, and the soma integrates input currents from both dendrites. The current flows between the soma and the dendrite is governed by axial conductances, defined in the dendrite parameters. The dendrite parameters are computed based on passive membrane properties and geometrical properties, defined in the `DendNeuronParameter`.
The model accepts any synaptic model for both soma and dendrites


# Fields

## Population Info
- `name::String`: Name of the neuron model ("BallAndStick" by default)
- `id::String`: Unique identifier for the neuron model
- `N::IT`: Number of neurons in the population (default: 100)
- `records::Dict`: Dictionary for storing simulation records

## Model Parameters
- `param::DendNeuronParameter`: Parameters specific to the dendrite-neuron model
- `adex::SOMAT`: Adaptive Exponential Integrate-and-Fire (AdEx) parameters
- `dend_syn::SYND`: Dendritic synapse parameters (default: TripodDendSynapse)
- `soma_syn::SYNS`: Somatic synapse parameters (default: TripodSomaSynapse)
- `spike::PST`: Post-spike parameters (default: PostSpike)
- `d::VDT`: Dendrite properties

## Model Variables
- `v_s::VFT`: Somatic membrane potential
- `w_s::VFT`: Somatic adaptation current
- `v_d::VFT`: Dendritic membrane potential
- `synvars_s::SYNSV`: Somatic synaptic variables
- `synvars_d::SYNDV`: Dendritic synaptic variables
- `Is::VFT`: Somatic external input current
- `Id::VFT`: Dendritic external input current
- `gaba_d::VFT`: GABA receptor conductance in dendrite
- `glu_d::VFT`: Glutamate receptor conductance in dendrite
- `gaba_s::VFT`: GABA receptor conductance in soma
- `glu_s::VFT`: Glutamate receptor conductance in soma
- `fire::VBT`: Boolean array indicating which neurons fired
- `tabs::VFT`: Absolute refractory period counters
- `θ::VFT`: Dynamic threshold for spike initiation

## Temporary Variables for Integration
- `Δv::MFT`: Temporary variable for voltage changes during integration
- `Δv_temp::MFT`: Temporary variable for voltage changes during integration
- `is::MFT`: Synaptic input currents
- `ic::VFT`: Axial current between compartments

This model implements a ball-and-stick neuron with separate somatic and dendritic compartments,
using adaptive exponential integrate-and-fire dynamics with synaptic inputs to both compartments.
The model supports Heun integration for numerical stability.
"""
BallAndStick
@snn_kw struct BallAndStick{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    VDT = Dendrite{Vector{Float32}},
    SYND<:AbstractSynapseParameter,
    SYNS<:AbstractSynapseParameter,
    SYNDV<:AbstractSynapseVariable,
    SYNSV<:AbstractSynapseVariable,
    SOMAT<:AbstractGeneralizedIFParameter,
    PST<:AbstractSpikeParameter,
    RECT<:NamedTuple,
    IT = Int32,
} <: AbstractDendriteIF     ## These are compulsory parameters

    name::String = "BallAndStick"
    id::String = randstring(12)
    N::IT = 100
    param::DendNeuronParameter = BallAndStickParameter()
    adex::SOMAT = AdExParameter()
    dend_syn::SYND = TripodDendSynapse
    soma_syn::SYNS = TripodSomaSynapse
    spike::PST = PostSpike()

    # Membrane potential and adaptation
    d::VDT = create_dendrite(N, param.ds[1])
    v_s::VFT = rand_value(N, adex.Vt, adex.Vr)
    w_s::VFT = zeros(N)
    v_d::VFT = rand_value(N, adex.Vt, adex.Vr)

    # Synapses
    synvars_s::SYNSV = synaptic_variables(soma_syn, N)
    synvars_d::SYNDV = synaptic_variables(dend_syn, N)

    ## Ext input
    Is::VFT = zeros(N)
    Id::VFT = zeros(N)

    # Receptors properties
    receptors_d::RECT = synaptic_receptors(dend_syn, N) #! target
    receptors_s::RECT = synaptic_receptors(soma_syn, N) #! target

    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    tabs::VFT = zeros(Int, N)
    θ::VFT = ones(N) * adex.Vt
    records::Dict = Dict()

    ## Temporary variables for integration
    Δv::MFT = zeros(N, 3)
    Δv_temp::MFT = zeros(N, 3)
    is::MFT = zeros(N, 2)
    ic::VFT = zeros(1)
end

function integrate!(p::BallAndStick, param::DendNeuronParameter, dt::Float32)
    @unpack N, v_s, w_s, v_d = p
    @unpack fire, θ, tabs = p
    @unpack Δv, Δv_temp, is = p

    @unpack synvars_s, synvars_d, d = p
    @unpack receptors_d, receptors_s = p

    @unpack spike, adex, soma_syn, dend_syn = p
    @unpack AP_membrane, up, τabs, At, τA = spike
    @unpack El, Vr, Vt, τw, a, b = adex

    update_synapses!(p, soma_syn, receptors_s, synvars_s, dt)
    update_synapses!(p, dend_syn, receptors_d, synvars_d, dt)

    ## Heun integration
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)
    update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    update_neuron!(p, param, Δv, dt)

    @inbounds for i ∈ 1:N
        tabs[i] -= 1
        θ[i] += dt * (Vt - θ[i]) / τA
        if tabs[i] > τabs / dt # backpropagation period
            v_s[i] = AP_membrane
            v_d[i] += dt * (v_s[i] - v_d[i]) * d.gax[i] / d.C[i]
        elseif tabs[i] > 0 # absolute refractory period
            v_s[i] = Vr
            v_d[i] += dt * (v_s[i] - v_d[i]) * d.gax[i] / d.C[i]
        elseif tabs[i] <= 0
            fire[i] = v_s[i] .+ Δv[i, 1] * dt >= -10mV
            Δv[i, 1] = ifelse(fire[i], AP_membrane - v_s[i], Δv[i, 1])
            v_s[i] = ifelse(fire[i], AP_membrane, v_s[i])
            w_s[i] = ifelse(fire[i], w_s[i] + b, w_s[i])
            θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
            tabs[i] = ifelse(fire[i], round(Int, (up + τabs) / dt), tabs[i])
            fire[i] && continue
            v_s[i] += 0.5 * dt * (Δv_temp[i, 1] + Δv[i, 1])
            v_d[i] += 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2])
            w_s[i] += 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3])
        end
    end
end


@inline function update_neuron!(
    p::BallAndStick,
    param::DendNeuronParameter,
    Δv::Matrix{Float32},
    dt::Float32,
)
    @unpack v_d, v_s, w_s, θ, tabs, fire = p
    @unpack d = p
    @unpack is, ic = p
    @unpack adex, spike, soma_syn, dend_syn = p
    @unpack AP_membrane, up, τabs, At, τA = spike
    @unpack C, gl, El, ΔT, Vt, Vr, a, b, τw = adex
    @unpack synvars_s, synvars_d = p


    @views synaptic_current!(p, soma_syn, synvars_s, v_s[:], is[:, 1])
    @views synaptic_current!(p, dend_syn, synvars_d, v_d[:], is[:, 2])
    clamp!(is, -1000, 1000)

    @fastmath @inbounds for i ∈ 1:p.N
        ic[1] = -((v_d[i] + Δv[i, 2] * dt) - (v_s[i] + Δv[i, 1] * dt)) * d.gax[i]


        Δv[i, 1] =
            1/C * (
                + gl * (-(v_s[i] + Δv[i, 1] * dt) + El) +
                ΔT * exp256(1 / ΔT * (v_s[i] + Δv[i, 1] * dt - θ[i])) - w_s[i]  # adaptation
                - is[i, 1]   # synapses
                - ic[1] # axial currents
                # + I[i]  # external current
            )

        Δv[i, 2] = ((-(v_d[i] + Δv[i, 2] * dt) + El) * d.gm[i] - is[i, 2] + ic[1]) / d.C[i]
        Δv[i, 3] = (a * ((v_s[i] + Δv[i, 1]) - El) - (w_s[i] + Δv[i, 3])) / τw
    end
end


export BallAndStick
