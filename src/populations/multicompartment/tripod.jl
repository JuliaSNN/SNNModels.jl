

"""
    Tripod

A neuron model with a soma and two dendrites, implementing the Adaptive Exponential Integrate-and-Fire (AdEx) dynamics.
The model includes synaptic conductances for both soma and dendrites, and supports NMDA voltage-dependent synapses.

# Fields
- `id::String`: Unique identifier for the neuron (default: random 12-character string)
- `name::String`: Name of the neuron (default: "Tripod")
- `N::IT`: Number of neurons in the population (default: 100)
- `param::DendNeuronParameter`: Parameters for the neuron model (default: `TripodParameter`)
- `d1::VDT`: First dendrite structure
- `d2::VDT`: Second dendrite structure
- `v_s::VFT`: Soma membrane potential (initialized randomly between Vt and Vr)
- `w_s::VFT`: Adaptation current for soma (initialized to zeros)
- `v_d1::VFT`: First dendrite membrane potential (initialized randomly between Vt and Vr)
- `v_d2::VFT`: Second dendrite membrane potential (initialized randomly between Vt and Vr)
- `I::VFT`: External current to soma (initialized to zeros)
- `I_d::VFT`: External current to dendrites (initialized to zeros)
- `g_d1::MFT`: Conductance for first dendrite synapses (initialized to zeros)
- `g_d2::MFT`: Conductance for second dendrite synapses (initialized to zeros)
- `h_d1::MFT`: Synaptic gating variable for first dendrite (initialized to zeros)
- `h_d2::MFT`: Synaptic gating variable for second dendrite (initialized to zeros)
- `g_s::MFT`: Conductance for soma synapses (initialized to zeros)
- `h_s::MFT`: Synaptic gating variable for soma (initialized to zeros)
- `glu_d1::VFT`: Glutamate synaptic input to first dendrite (initialized to zeros)
- `gaba_d1::VFT`: GABA synaptic input to first dendrite (initialized to zeros)
- `glu_d2::VFT`: Glutamate synaptic input to second dendrite (initialized to zeros)
- `gaba_d2::VFT`: GABA synaptic input to second dendrite (initialized to zeros)
- `glu_s::VFT`: Glutamate synaptic input to soma (initialized to zeros)
- `gaba_s::VFT`: GABA synaptic input to soma (initialized to zeros)
- `fire::VBT`: Boolean vector indicating which neurons fired (initialized to false)
- `after_spike::VFT`: Counter for refractory period (initialized to zeros)
- `θ::VFT`: Threshold potential (initialized to Vt)
- `records::Dict`: Dictionary for storing simulation data (initialized empty)
- `Δv::VFT`: Temporary voltage change vector (initialized to zeros)
- `Δv_temp::VFT`: Temporary voltage change vector (initialized to zeros)
- `cs::VFT`: Axial current vector (initialized to zeros)
- `is::VFT`: Synaptic current vector (initialized to zeros)
"""
Tripod
@snn_kw struct Tripod{
    VFT = Vector{Float32},
    MFT = Matrix{Float32},
    VDT = Dendrite{Vector{Float32}},
    SYNV<: AbstractSynapseVariable,
    IT = Int32,
} <: AbstractDendriteIF
    id::String = randstring(12)
    name::String = "Tripod"
    ## These are compulsory parameters
    N::IT = 100
    param::DendNeuronParameter = TripodParameter
    d1::VDT = create_dendrite(N, param.ds[1])
    d2::VDT = create_dendrite(N, param.ds[2])

    # Membrane potential and adaptation
    v_s::VFT = rand_value(N, param.adex.Vt, param.adex.Vr)
    w_s::VFT = zeros(N)
    v_d1::VFT = rand_value(N, param.adex.Vt, param.adex.Vr)
    v_d2::VFT = rand_value(N, param.adex.Vt, param.adex.Vr)
    I::VFT = zeros(N)
    I_d::VFT = zeros(N)

    # Synapses dendrites
    synvars_s::SYNV = synaptic_variables(param.soma_syn, N)
    synvars_d1::SYNV = synaptic_variables(param.dend_syn, N)
    synvars_d2::SYNV = synaptic_variables(param.dend_syn, N)

    glu_d1::VFT = zeros(N) #! target
    gaba_d1::VFT = zeros(N) #! target
    glu_d2::VFT = zeros(N) #! target
    gaba_d2::VFT = zeros(N) #! target
    glu_s::VFT = zeros(N) #! target
    gaba_s::VFT = zeros(N) #! target

    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    tabs::VFT = zeros(Int, N)
    θ::VFT = ones(N) * param.adex.Vt
    records::Dict = Dict()

    ## Temporary variables for integration
    Δv::MFT = zeros(N, 4)
    Δv_temp::MFT = zeros(N, 4)
    is::MFT = zeros(N, 3)
    ic::VFT = zeros(2)
end

function synaptic_target(targets::Dict, post::Tripod, sym::Symbol, target::Symbol)
    syn = get_synapse_symbol(post.param.soma_syn, sym)
    sym = Symbol("$(syn)_$target")
    v = Symbol("v_$target")
    g = getfield(post,sym )
    hasfield(typeof(post), v) && (v_post = getfield(post, v))

    push!(targets, :sym => sym)
    push!(targets, :g => post.id)

    return g, v_post
end


function integrate!(p::Tripod, param::DendNeuronParameter, dt::Float32)
    @unpack N, v_s, w_s, v_d1, v_d2 = p
    @unpack fire, θ, tabs = p
    @unpack Δv, Δv_temp, is = p

    @unpack synvars_s, synvars_d1, synvars_d2 = p
    @unpack glu_d1, glu_d2, glu_s, gaba_d1, gaba_d2, gaba_s = p

    @unpack spike, adex,  soma_syn, dend_syn = param
    @unpack AP_membrane, up, τabs, At, τA  = spike 
    @unpack El, Vr, Vt, τw, a, b = adex

    # Update all synaptic conductance
    update_synapses!(p, soma_syn, glu_s, gaba_s, synvars_s, dt)
    update_synapses!(p, dend_syn, glu_d1, gaba_d1, synvars_d1, dt)
    update_synapses!(p, dend_syn, glu_d2, gaba_d2, synvars_d2, dt)

    ## Heun integration
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)

    update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv

    # @views synaptic_current!(p, soma_syn,  synvars_s, v_s + Δv[:,1] * dt, is[:,1])
    # @views synaptic_current!(p, dend_syn, synvars_d1, v_d1 + Δv[:,2] * dt, is[:,2])
    # @views synaptic_current!(p, dend_syn, synvars_d2, v_d2 + Δv[:,3] * dt, is[:,3])
    # clamp!(is, -1500, 1500)
    update_neuron!(p, param, Δv, dt)
    # @show Δv.+Δv_temp

    @show Δv
    @inbounds for i ∈ 1:N
        v_s[i]  += 0.5 * dt * (Δv_temp[i, 1] + Δv[i, 1])
        v_d1[i] += 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2])
        v_d2[i] += 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3])
        w_s[i]  += 0.5 * dt * (Δv_temp[i, 4] + Δv[i, 4])

        @show v_s, v_d1, v_d2
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
    @unpack adex, spike, soma_syn, dend_syn = param
    @unpack AP_membrane, up, τabs, At, τA  = spike 
    @unpack C, gl, El, ΔT, Vt, Vr, a, b, τw = adex
    @unpack synvars_s, synvars_d1, synvars_d2 = p

    @fastmath @inbounds for i ∈ 1:p.N
        fire[i] = v_s[i] >= -10mV
        v_s[i]  = ifelse(fire[i], AP_membrane, v_s[i]) 
        w_s[i]  = ifelse(fire[i], w_s[i] + b, w_s[i])
        θ[i]    = ifelse(fire[i], θ[i] + At, θ[i])
        tabs[i] = ifelse(fire[i], round(Int, (up + τabs) / dt), tabs[i])
    end

    @views synaptic_current!(p, soma_syn,  synvars_s, v_s[:], is[:,1])
    @views synaptic_current!(p, dend_syn, synvars_d1, v_d1[:], is[:,2])
    @views synaptic_current!(p, dend_syn, synvars_d2, v_d2[:], is[:,3])
    clamp!(is, -1000, 1000)

    @fastmath @inbounds for i ∈ 1:p.N

        tabs[i] -= 1
        θ[i]    += dt * (Vt - θ[i]) / τA
        if tabs[i] > (τabs + up) / dt # backpropagation period
            continue
        elseif tabs[i] > 0 # absolute refractory period
            v_s[i] = Vr
            continue
        else
            ic[1] = -((v_d1[i] + Δv[i,2] * dt) - (v_s[i] + Δv[i,1] * dt)) * d1.gax[i]
            ic[2] = -((v_d2[i] + Δv[i,3] * dt) - (v_s[i] + Δv[i,1] * dt)) * d2.gax[i]
            Δv[i, 2] = ((-(v_d1[i] + Δv[i,2] * dt) + El) * d1.gm[i] - is[2] + ic[1] + I_d[i]) / d1.C[i]
            Δv[i, 3] = ((-(v_d2[i] + Δv[i,3] * dt) + El) * d2.gm[i] - is[3] + ic[2] + I_d[i]) / d2.C[i]
            @show "Integration"
            Δv[i,1] =
                1/C * (
                    + gl * (-(v_s[i] + Δv[i, 1] * dt) + El) 
                    + ΔT * exp64(1 / ΔT * (v_s[i] + Δv[i, 1] * dt - θ[i])) - w_s[i]  # adaptation
                    - is[i, 1]   # synapses
                    - sum(ic)*1 # axial currents
                    + I[i]  # external current
                )
            Δv[i, 4] = (a * (v_s[i] +Δv[i, 1] - El) - (w_s[i]+Δv[i, 4])) / τw
            @show "Firing: ", fire[i]
        end
    end
end


export Tripod
