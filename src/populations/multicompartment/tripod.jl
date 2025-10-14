

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
    MFT = Matrix{Float32},
    VFT = Vector{Float32},
    VDT = Dendrite{Vector{Float32}},
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
    g_d1::MFT = zeros(N, 4)
    g_d2::MFT = zeros(N, 4)
    h_d1::MFT = zeros(N, 4)
    h_d2::MFT = zeros(N, 4)
    # Synapses soma
    g_s::MFT = zeros(N, 4)
    h_s::MFT = zeros(N, 4)

    glu_d1::VFT = zeros(N) #! target
    gaba_d1::VFT = zeros(N) #! target
    glu_d2::VFT = zeros(N) #! target
    gaba_d2::VFT = zeros(N) #! target
    glu_s::VFT = zeros(N) #! target
    gaba_s::VFT = zeros(N) #! target

    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    after_spike::VFT = zeros(Int, N)
    θ::VFT = ones(N) * param.adex.Vt
    records::Dict = Dict()
    Δv::VFT = zeros(3)
    Δv_temp::VFT = zeros(3)
    cs::VFT = zeros(2)
    is::VFT = zeros(3)
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
    @fastmath @inbounds begin
        @unpack N, v_s, w_s, v_d1, v_d2 = p
        @unpack fire, θ, after_spike, Δv, Δv_temp = p
        @unpack spike, adex = param
        @unpack El, Vr, Vt, τw, a, b = adex
        @unpack AP_membrane, up, τabs, At, τA  = spike 
        @unpack d1, d2 = p
        @unpack soma_syn, dend_syn = param
        @unpack N, g_d1, g_d2, h_d1, h_d2, g_s, h_s = p
        @unpack glu_d1, glu_d2, glu_s, gaba_d1, gaba_d2, gaba_s = p

        # Update all synaptic conductance
        update_synapses!(p, soma_syn, glu_s, gaba_s, g_s, h_s, dt)
        update_synapses!(p, dend_syn, glu_d1, gaba_d1, g_d1, h_d1, dt)
        update_synapses!(p, dend_syn, glu_d2, gaba_d2, g_d2, h_d2, dt)
        for i ∈ 1:N
            # implementation of the absolute refractory period with backpropagation (up) and after spike (τabs)
            if after_spike[i] > (τabs + up - up) / dt # backpropagation
                v_s[i] = AP_membrane
                ## backpropagation effect
                c1 = (v_s[i] - v_d1[i]) * d1.gax[i]
                c2 = (v_s[i] - v_d2[i]) * d2.gax[i]
                ## apply currents
                v_d1[i] += dt * c1 / d1.C[i]
                v_d2[i] += dt * c2 / d2.C[i]
            elseif after_spike[i] > 0 # absolute refractory period
                v_s[i] = Vr
                c1 = (Vr - v_d1[i]) * d1.gax[i]
                c2 = (Vr - v_d2[i]) * d2.gax[i]
                # ## apply currents
                v_d1[i] += dt * c1 / d1.C[i]
                v_d2[i] += dt * c2 / d2.C[i]
            else
                ## Heun integration
                for _i ∈ 1:3
                    Δv_temp[_i] = 0.0f0
                    Δv[_i] = 0.0f0
                end
                update_neuron!(p, Δv, i, param, 0.0f0)
                for _i ∈ 1:3
                    Δv_temp[_i] = Δv[_i]
                end
                update_neuron!(p, Δv, i, param, dt)
                v_s[i] += 0.5 * dt * (Δv_temp[1] + Δv[1])
                v_d1[i] += 0.5 * dt * (Δv_temp[2] + Δv[2])
                v_d2[i] += 0.5 * dt * (Δv_temp[3] + Δv[3])
                w_s[i] += dt * (a * (v_s[i] - El) - w_s[i]) / τw
            end
        end
        # reset firing
        @unpack At, τA = param.spike
        fire .= false
        @inbounds for i ∈ 1:N
            θ[i] -= dt * (θ[i] - Vt) / τA
            after_spike[i] -= 1
            if after_spike[i] < 0
                ## spike ?
                if v_s[i] > θ[i] + 10.0f0
                    fire[i] = true
                    θ[i] += At
                    v_s[i] = AP_membrane
                    w_s[i] += b ##  *τw
                    after_spike[i] = (up + τabs) / dt
                end
            end
        end
    end
    # return
end

@inline function update_neuron!(
    p::Tripod,
    Δv::Vector{Float32},
    i::Int64,
    param::DendNeuronParameter,
    dt::Float32,
)

    @fastmath @inbounds begin
        @unpack v_d1, v_d2, v_s, I_d, I, w_s, θ = p
        @unpack g_d1, g_d2, g_s = p
        @unpack d1, d2 = p
        @unpack is, cs = p
        @unpack soma_syn, dend_syn = param
        @unpack C, gl, El, ΔT = param.adex


        #compute axial currents
        cs[1] = -((v_d1[i] + Δv[2] * dt) - (v_s[i] + Δv[1] * dt)) * d1.gax[i]
        cs[2] = -((v_d2[i] + Δv[3] * dt) - (v_s[i] + Δv[1] * dt)) * d2.gax[i]

        synaptic_current!(soma_syn, v_s[i] + Δv[1] * dt, g_s, is, 1, i)
        synaptic_current!(dend_syn, v_d1[i] + Δv[2] * dt, g_d1, is, 2, i)
        synaptic_current!(dend_syn, v_d2[i] + Δv[3] * dt, g_d2, is, 3, i)

        # update membrane potential
        Δv[1] =
            1/C * (
                + gl * (-v_s[i] + Δv[1] * dt + El) +
                ΔT * exp64(1 / ΔT * (v_s[i] + Δv[1] * dt - θ[i])) - w_s[i]  # adaptation
                - is[1]   # synapses
                - sum(cs) # axial currents
                + I[i]  # external current
            )

        Δv[2] =
            ((-(v_d1[i] + Δv[2] * dt) + El) * d1.gm[i] - is[2] + cs[1] + I_d[i]) / d1.C[i]
        Δv[3] =
            ((-(v_d2[i] + Δv[3] * dt) + El) * d2.gm[i] - is[3] + cs[2] + I_d[i]) / d2.C[i]
    end
end


export Tripod
