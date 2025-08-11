"""
This is a struct representing a spiking neural network model that include two dendrites and a soma based on the adaptive exponential integrate-and-fire model (AdEx)


# Fields 
- `t::VIT` : tracker of simulation index [0] 
- `param::AdExSoma` : Parameters for the AdEx model.
- `N::Int32` : The number of neurons in the network.
- `d::VDT`: Dendritic compartment parameters.

- `v_s::VFT` : Somatic membrane potential.
- `w_s::VFT` : Adaptation variables for each soma.
- `v_d::VFT`: Dendritic membrane potential for dendrite.
- `g_s::MFT` , `g_d::MFT` : Conductance of somatic and dendritic synapses.
- `h_s::MFT`, `h_d::MFT`  : Synaptic gating variables.

- `fire::VBT` : Boolean array indicating which neurons have fired.
- `after_spike::VFT` : Post-spike timing.
- `postspike::PST` : Model for post-spike behavior.
- `θ::VFT` : Individual neuron firing thresholds.
- `records::Dict` : A dictionary to store simulation results.
- `Δv::VFT` , `Δv_temp::VFT` : Variables to hold temporary voltage changes.
- `cs::VFT` , `is::VFT` : Temporary variables for currents.
"""
BallAndStick
@snn_kw struct BallAndStick{
    MFT = Matrix{Float32},
    VIT = Vector{Int32},
    VST = Vector{Symbol},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    VDT = Dendrite{Vector{Float32}},
    IT = Int32,
    FT = Float32,
    ST = SynapseArray,
} <: AbstractDendriteIF     ## These are compulsory parameters
    name::String = "BallAndStick"
    id::String = randstring(12)
    N::IT = 100
    param::DendNeuronParameter = BallAndStickParameter()
    d::VDT = create_dendrite(N, param.ds[1])

    # Membrane potential and adaptation
    v_s::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    w_s::VFT = zeros(N)
    v_d::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    # Synapses
    g_d::MFT = zeros(N, 4)
    h_d::MFT = zeros(N, 4)
    # Synapses soma
    g_s::MFT = zeros(N, 4)
    h_s::MFT = zeros(N, 4)

    ## Ext input
    Is::VFT = zeros(N)
    Id::VFT = zeros(N)

    # Receptors properties
    glu_receptors::VST = [:AMPA, :NMDA]
    gaba_receptors::VST = [:GABAa, :GABAb]
    all_receptors::VST = vcat(glu_receptors..., gaba_receptors)
    gaba_d::VFT = zeros(N) #! target
    glu_d::VFT = zeros(N) #! target
    gaba_s::VFT = zeros(N) #! target
    glu_s::VFT = zeros(N) #! target

    records::Dict = Dict()

    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    after_spike::VFT = zeros(Int, N)
    θ::VFT = ones(N) * param.Vt
    Δv::VFT = zeros(3)
    Δv_temp::VFT = zeros(3)
    cs::VFT = zeros(2)
    is::VFT = zeros(3)
end

function synaptic_target(targets::Dict, post::BallAndStick, sym::Symbol, target)
    _sym = Symbol("$(sym)_$target")
    _v = Symbol("v_$target")
    g = getfield(post, _sym)
    v_post = getfield(post, _v)
    push!(targets, :sym => _sym)
    return g, v_post
end


function integrate!(p::BallAndStick, param::DendNeuronParameter, dt::Float32)
    @unpack N, v_s, w_s, v_d = p
    @unpack fire, θ, after_spike,  Δv, Δv_temp = p
    @unpack Er, up, τabs, BAP, AP_membrane, Vr, Vt, τw, a, b, postspike = param
    @unpack d = p
    @unpack NMDA, soma_syn, dend_syn = param
    @unpack N, g_s, g_d, h_s, h_d = p
    @unpack glu_d, glu_s, gaba_d, gaba_s = p

    update_synapses!(p, soma_syn, glu_s, gaba_s, g_s, h_s, dt)
    update_synapses!(p, dend_syn, glu_d, gaba_d, g_d, h_d, dt)
    # update the neurons
    @inbounds for i ∈ 1:N
        if after_spike[i] > τabs
            v_s[i] = BAP
            ## backpropagation effect
            c1 = (BAP - v_d[i]) * d.gax[i] / 100
            ## apply currents
            v_d[i] += dt * c1 / d.C[i]
        elseif after_spike[i] > 0
            v_s[i] = Vr
            c1 = (Vr - v_d[i]) * d.gax[i] / 100
            # # apply currents
            v_d[i] += dt * c1 / d.C[i]
        else
            ## Heun integration
            for _i ∈ 1:2
                Δv_temp[_i] = 0.0f0
                Δv[_i] = 0.0f0
            end
            update_ballandstick!(p, Δv, i, param,soma_syn, dend_syn,  0.f0)
            for _i ∈ 1:2
                Δv_temp[_i] = Δv[_i]
            end
            update_ballandstick!(p, Δv, i, param,  soma_syn, dend_syn, dt)
            @fastmath v_s[i] += 0.5 * dt * (Δv_temp[1] + Δv[1])
            @fastmath v_d[i] += 0.5 * dt * (Δv_temp[2] + Δv[2])
            @fastmath w_s[i] += dt * (param.a * (v_s[i] - param.Er) - w_s[i]) / param.τw
        end
    end

    # reset firing
    fire .= false
    @inbounds for i ∈ 1:N
        θ[i] -= dt * (θ[i] - Vt) / postspike.τA
        after_spike[i] -= 1
        if after_spike[i] < 0
            ## spike ?
            if v_s[i] > θ[i] + 10.0f0
                fire[i] = true
                θ[i] += postspike.A
                v_s[i] = AP_membrane
                w_s[i] += b ##  *τw
                after_spike[i] = (up + τabs) / dt
            end
        end
    end
    return
end


function update_ballandstick!(
    p::BallAndStick,
    Δv::Vector{Float32},
    i::Int64,
    param::DendNeuronParameter,
    soma_syn::Synapse,
    dend_syn::Synapse,
    dt::Float32,
)

    @fastmath @inbounds begin
        @unpack v_d, v_s, w_s, g_s, g_d, θ, d, Is, Id = p
        @unpack is, cs = p

        #compute axial currents
        cs[1] = -((v_d[i] + Δv[2] * dt) - (v_s[i] + Δv[1] * dt)) * d.gax[i]

        fill!(is, 0.0f0)
        synaptic_current!(p, param, soma_syn, v_s[i] + Δv[1] * dt, g_s, is, 1, i)
        synaptic_current!(p, param, dend_syn, v_d[i] + Δv[2] * dt, g_d, is, 2, i)

        ## update synaptic currents soma
        @turbo for _i ∈ 1:2
            is[_i] = clamp(is[_i], -1500, 1500)
        end

        # update membrane potential
        @unpack C, gl, Er, ΔT = param
        Δv[1] =
            (
                gl * (
                    (-v_s[i] + Δv[1] * dt + Er) +
                    ΔT * exp64(1 / ΔT * (v_s[i] + Δv[1] * dt - θ[i]))
                ) - w_s[i] - is[1] - cs[1] + Is[i]
            ) / C
        Δv[2] = ((-(v_d[i] + Δv[2] * dt) + Er) * d.gm[i] - is[2] + cs[1] + Id[i]) / d.C[i]
    end

end

export BallAndStick
