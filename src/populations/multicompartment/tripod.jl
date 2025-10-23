"""
    Tripod{VFT = Vector{Float32},
            MFT = Matrix{Float32},
            VDT = Dendrite{Vector{Float32}},
            SYNS <: AbstractSynapseParameter,
            SYND <: AbstractSynapseParameter,
            SYNSV <: AbstractSynapseVariable,
            SYNDV <: AbstractSynapseVariable,
            SOMAT <: AbstractGeneralizedIFParameter,
            PST <: AbstractSpikeParameter,
            IT = Int32,
} <: AbstractDendriteIF

A struct representing a tripod neuron model with two dendrites and a soma.
The model incorporates adaptive exponential integrate-and-fire dynamics
with synaptic inputs to both the soma and dendrites. The soma includes adaptation currents and dynamic thresholds for spike generation.
The dendrites are modeled with separate passive compartments, and the soma integrates input currents from both dendrites. The current flows between the soma and dendrites are governed by axial conductances, defined in the dendrite parameters. The dendrites are computed based on passive membrane properties and geometrical properties, defined in the `DendNeuronParameter`.
The model accepts any synaptic model for both soma and dendrites

The model leverages Heun integration for improved numerical stability.


# Fields
## Population Info
- `id::String`: Unique identifier for the neuron (default: random 12-character string).
- `name::String`: Name of the neuron (default: "Tripod").
- `N::IT`: Number of neurons in the population (default: 100).
- `records::Dict`: Dictionary for storing simulation records (initialized empty).

## Model Parameters
- `param::DendNeuronParameter`: Parameters for the dendrite neuron (default: `TripodParameter()`).
- `adex::SOMAT`: Adaptive Exponential Integrate-and-Fire parameters (default: `AdExParameter()`).
- `soma_syn::SYNS`: Synaptic parameters for the soma (default: `TripodSomaSynapse`).
- `dend_syn::SYND`: Synaptic parameters for the dendrites (default: `TripodDendSynapse`).
- `spike::PST`: Spike parameters (default: `PostSpike()`).
- `d1::VDT`: First dendrite parameters (created using `create_dendrite`).
- `d2::VDT`: Second dendrite parameters (created using `create_dendrite`).

## Model Variables
- `v_s::VFT`: Soma membrane potential (initialized randomly between `Vt` and `Vr`).
- `w_s::VFT`: Adaptation current for the soma (initialized to zeros).
- `v_d1::VFT`: First dendrite membrane potential (initialized randomly between `Vt` and `Vr`).
- `v_d2::VFT`: Second dendrite membrane potential (initialized randomly between `Vt` and `Vr`).
- `I::VFT`: External current input to the soma (initialized to zeros).
- `I_d::VFT`: External current input to the dendrites (initialized to zeros).
- `synvars_s::SYNSV`: Synaptic variables for the soma (initialized using `synaptic_variables`).
- `synvars_d1::SYNDV`: Synaptic variables for the first dendrite (initialized using `synaptic_variables`).
- `synvars_d2::SYNDV`: Synaptic variables for the second dendrite (initialized using `synaptic_variables`).
- `glu_d1::VFT`: Glutamate synaptic input to the first dendrite (initialized to zeros).
- `gaba_d1::VFT`: GABA synaptic input to the first dendrite (initialized to zeros).
- `glu_d2::VFT`: Glutamate synaptic input to the second dendrite (initialized to zeros).
- `gaba_d2::VFT`: GABA synaptic input to the second dendrite (initialized to zeros).
- `glu_s::VFT`: Glutamate synaptic input to the soma (initialized to zeros).
- `gaba_s::VFT`: GABA synaptic input to the soma (initialized to zeros).
- `fire::VBT`: Boolean array indicating which neurons have spiked (initialized to false).
- `tabs::VFT`: Absolute refractory period counters (initialized to zeros).
- `θ::VFT`: Dynamic threshold for spike generation (initialized to `Vt`).
- `records::Dict`: Dictionary for storing simulation records (initialized empty).

## Temporary Variables for Integration
- `Δv::MFT`: Temporary matrix for voltage changes during integration (initialized to zeros).
- `Δv_temp::MFT`: Temporary matrix for voltage changes during integration (initialized to zeros).
- `is::MFT`: Matrix for synaptic currents (initialized to zeros).
- `ic::VFT`: Vector for axial currents between soma and dendrites (initialized to zeros).

# Type Parameters
- `VFT`: Type for vector fields (default: `Vector{Float32}`).
- `MFT`: Type for matrix fields (default: `Matrix{Float32}`).
- `VDT`: Type for dendrite parameters (default: `Dendrite{Vector{Float32}}`).
- `SYNS`: Type for soma synaptic parameters (default: `AbstractSynapseParameter`).
- `SYND`: Type for dendrite synaptic parameters (default: `AbstractSynapseParameter`).
- `SYNSV`: Type for soma synaptic variables (default: `AbstractSynapseVariable`).
- `SYNDV`: Type for dendrite synaptic variables (default: `AbstractSynapseVariable`).
- `SOMAT`: Type for soma parameters (default: `AbstractGeneralizedIFParameter`).
- `PST`: Type for spike parameters (default: `AbstractSpikeParameter`).
- `IT`: Type for integer fields (default: `Int32`).

# Functions
- `synaptic_target`: Helper function to set synaptic targets for the neuron.
"""
Tripod
@snn_kw struct Tripod{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    VDT = Dendrite{Vector{Float32}},
    SYNS<:AbstractSynapseParameter,
    SYND<:AbstractSynapseParameter,
    SYNSV<:AbstractSynapseVariable,
    SYNDV<:AbstractSynapseVariable,
    SOMAT<:AbstractGeneralizedIFParameter,
    PST<:AbstractSpikeParameter,
    RECT<:NamedTuple,
    IT = Int32,
} <: AbstractDendriteIF
    id::String = randstring(12)
    name::String = "Tripod"
    ## These are compulsory parameters
    N::IT = 100
    param::DendNeuronParameter = TripodParameter()
    adex::SOMAT = AdExParameter()
    soma_syn::SYNS = TripodSomaSynapse
    dend_syn::SYND = TripodDendSynapse
    spike::PST = PostSpike()
    d1::VDT = create_dendrite(N, param.ds[1])
    d2::VDT = create_dendrite(N, param.ds[2])

    # Membrane potential and adaptation
    v_s::VFT = rand_value(N, adex.Vt, adex.Vr)
    w_s::VFT = zeros(N)
    v_d1::VFT = rand_value(N, adex.Vt, adex.Vr)
    v_d2::VFT = rand_value(N, adex.Vt, adex.Vr)
    I::VFT = zeros(N)
    I_d::VFT = zeros(N)

    # Synapses dendrites
    synvars_s::SYNSV = synaptic_variables(soma_syn, N)
    synvars_d1::SYNDV = synaptic_variables(dend_syn, N)
    synvars_d2::SYNDV = synaptic_variables(dend_syn, N)

    receptors_s::RECT = (glu = zeros(Float32, N), gaba = zeros(Float32, N)) #! target
    receptors_d1::RECT = (glu = zeros(Float32, N), gaba = zeros(Float32, N)) #! target
    receptors_d2::RECT = (glu = zeros(Float32, N), gaba = zeros(Float32, N)) #! target

    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    tabs::VFT = zeros(Int, N)
    θ::VFT = ones(N) * adex.Vt
    records::Dict = Dict()

    ## Temporary variables for integration
    Δv::MFT = zeros(N, 4)
    Δv_temp::MFT = zeros(N, 4)
    is::MFT = zeros(N, 3)
    ic::VFT = zeros(2)
end



# function Population(param <:T; N::Int=100, name::String="TripodNeuron", kwargs...) where T<:TripodParameter
#     return Tripod(;
#         N = N,
#         param = param,
#         name = name,
#         kwargs...,
#     )

# end

function integrate!(p::Tripod, param::DendNeuronParameter, dt::Float32)
    @unpack N, v_s, w_s, v_d1, v_d2 = p
    @unpack fire, θ, tabs = p
    @unpack Δv, Δv_temp, is = p

    @unpack synvars_s, synvars_d1, synvars_d2, d1, d2, I_d = p
    @unpack receptors_d1, receptors_d2, receptors_s = p

    @unpack spike, adex, soma_syn, dend_syn = p
    @unpack AP_membrane, up, τabs, At, τA = spike
    @unpack El, Vr, Vt, τw, a, b = adex

    # Update all synaptic conductance
    update_synapses!(p, soma_syn, receptors_s, synvars_s, dt)
    update_synapses!(p, dend_syn, receptors_d1, synvars_d1, dt)
    update_synapses!(p, dend_syn, receptors_d2, synvars_d2, dt)

    ## Heun integration
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)

    update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    update_neuron!(p, param, Δv, dt)
    # @show Δv.+Δv_temp

    @inbounds for i ∈ 1:N
        tabs[i] -= 1
        θ[i] += dt * (Vt - θ[i]) / τA
        if tabs[i] > τabs / dt # backpropagation period
            v_s[i] = AP_membrane
            v_d1[i] += dt * (v_s[i] - v_d1[i]) * d1.gax[i] / d1.C[i]
            v_d2[i] += dt * (v_s[i] - v_d2[i]) * d2.gax[i] / d2.C[i]
        elseif tabs[i] > 0 # absolute refractory period
            v_s[i] = Vr
            v_d1[i] += dt * (v_s[i] - v_d1[i]) * d1.gax[i] / d1.C[i]
            v_d2[i] += dt * (v_s[i] - v_d2[i]) * d2.gax[i] / d2.C[i]
        elseif tabs[i] <= 0
            fire[i] = v_s[i] .+ Δv[i, 1] * dt >= -10mV
            Δv[i, 1] = ifelse(fire[i], AP_membrane - v_s[i], Δv[i, 1])
            v_s[i] = ifelse(fire[i], AP_membrane, v_s[i])
            w_s[i] = ifelse(fire[i], w_s[i] + b, w_s[i])
            θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
            tabs[i] = ifelse(fire[i], round(Int, (up + τabs) / dt), tabs[i])
            fire[i] && continue
            v_s[i] += 0.5 * dt * (Δv_temp[i, 1] + Δv[i, 1])
            v_d1[i] += 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2])
            v_d2[i] += 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3])
            w_s[i] += 0.5 * dt * (Δv_temp[i, 4] + Δv[i, 4])
        end
    end
end

@inline function update_neuron!(
    p::Tripod,
    param::DendNeuronParameter,
    Δv::Matrix{Float32},
    dt::Float32,
)
    @unpack v_d1, v_d2, v_s, I_d, I, w_s, θ, tabs, fire = p
    @unpack d1, d2 = p
    @unpack is, ic = p
    @unpack adex, spike, soma_syn, dend_syn = p
    @unpack AP_membrane, up, τabs, At, τA = spike
    @unpack C, gl, El, ΔT, Vt, Vr, a, b, τw = adex
    @unpack synvars_s, synvars_d1, synvars_d2 = p


    @views synaptic_current!(p, soma_syn, synvars_s, v_s[:], is[:, 1])
    @views synaptic_current!(p, dend_syn, synvars_d1, v_d1[:], is[:, 2])
    @views synaptic_current!(p, dend_syn, synvars_d2, v_d2[:], is[:, 3])
    clamp!(is, -1500, 1500)

    @fastmath @inbounds for i ∈ 1:p.N
        ic[1] = -((v_d1[i] + Δv[i, 2] * dt) - (v_s[i] + Δv[i, 1] * dt)) * d1.gax[i]
        ic[2] = -((v_d2[i] + Δv[i, 3] * dt) - (v_s[i] + Δv[i, 1] * dt)) * d2.gax[i]
        Δv[i, 2] =
            ((-(v_d1[i] + Δv[i, 2] * dt) + El) * d1.gm[i] - is[2] + ic[1] + I_d[i]) /
            d1.C[i]
        Δv[i, 3] =
            ((-(v_d2[i] + Δv[i, 3] * dt) + El) * d2.gm[i] - is[3] + ic[2] + I_d[i]) /
            d2.C[i]
        Δv[i, 1] =
            1/C * (
                + gl * (-(v_s[i] + Δv[i, 1] * dt) + El) +
                ΔT * exp256(1 / ΔT * (v_s[i] + Δv[i, 1] * dt - θ[i])) - w_s[i]  # adaptation
                - is[i, 1]   # synapses
                - sum(ic) # axial currents
                + I[i]  # external current
            )
        # Δv[i, 4] = (a * (v_s[i]- El) - (w_s[i])) / τw
        Δv[i, 4] = (a * ((v_s[i] + Δv[i, 1]) - El) - (w_s[i] + Δv[i, 4])) / τw
    end
end


export Tripod
