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
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    VDT = Dendrite{Vector{Float32}},
    IT = Int32,
    FT = Float32,
    ST = SynapseArray,
    NMDAT = NMDAVoltageDependency,
} <: AbstractDendriteIF     ## These are compulsory parameters
    name::String = "BallAndStick"
    id::String = randstring(12)
    N::IT = 100
    param::DendNeuronParameter = BallAndStickParameter()
    d::VDT = create_dendrite(N, param.ds[1])
    soma_syn::ST = synapsearray(param.soma_syn)
    dend_syn::ST = synapsearray(param.dend_syn)
    NMDA::NMDAT = param.NMDA

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
    glu_receptors::VIT = [1, 2]
    gaba_receptors::VIT = [3, 4]
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
    @unpack d, NMDA, soma_syn, dend_syn = p


    update_synapses!(p, dt)
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
            update_ballandstick!(p, Δv, i, param, 0.f0)
            for _i ∈ 1:2
                Δv_temp[_i] = Δv[_i]
            end
            update_ballandstick!(p, Δv, i, param, dt)
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

function update_synapses!(
    p::BallAndStick,
    dt::Float32,
)
    @unpack N, g_s, g_d, h_s, h_d = p
    @unpack glu_d, glu_s, gaba_d, gaba_s, glu_receptors, gaba_receptors, = p
    @unpack soma_syn, dend_syn = p

    @inbounds for n in glu_receptors
        @unpack τr⁻, τd⁻, α = dend_syn[n]
        @turbo for i ∈ 1:N
            h_d[i, n] += glu_d[i] * α
            g_d[i, n] = exp64(-dt * τd⁻) * (g_d[i, n] + dt * h_d[i, n])
            h_d[i, n] = exp64(-dt * τr⁻) * (h_d[i, n])
        end
        @unpack τr⁻, τd⁻, α = soma_syn[n]
        @turbo for i ∈ 1:N
            h_s[i, n] += glu_s[i] * α
            g_s[i, n] = exp64(-dt * τd⁻) * (g_s[i, n] + dt * h_s[i, n])
            h_s[i, n] = exp64(-dt * τr⁻) * (h_s[i, n])
        end
    end

    @inbounds for n in gaba_receptors
        @unpack τr⁻, τd⁻, α = dend_syn[n]
        @turbo for i ∈ 1:N
            h_d[i, n] += gaba_d[i] * α
            g_d[i, n] = exp64(-dt * τd⁻) * (g_d[i, n] + dt * h_d[i, n])
            h_d[i, n] = exp64(-dt * τr⁻) * (h_d[i, n])
        end
        @unpack τr⁻, τd⁻, α = soma_syn[n]
        @turbo for i ∈ 1:N
            h_s[i, n] += gaba_s[i] * α
            g_s[i, n] = exp64(-dt * τd⁻) * (g_s[i, n] + dt * h_s[i, n])
            h_s[i, n] = exp64(-dt * τr⁻) * (h_s[i, n])
        end
    end

    fill!(glu_s, 0.0f0)
    fill!(glu_d, 0.0f0)
    fill!(gaba_s, 0.0f0)
    fill!(gaba_d, 0.0f0)

end


function update_ballandstick!(
    p::BallAndStick,
    Δv::Vector{Float32},
    i::Int64,
    param::DendNeuronParameter,
    dt::Float32,
)

    @fastmath @inbounds begin
        @unpack v_d, v_s, w_s, g_s, g_d, θ, d, Is, Id = p
        @unpack is, cs = p
        @unpack soma_syn, dend_syn, NMDA = p
        @unpack mg, b, k = NMDA

        #compute axial currents
        cs[1] = -((v_d[i] + Δv[2] * dt) - (v_s[i] + Δv[1] * dt)) * d.gax[i]

        for _i ∈ 1:2
            is[_i] = 0.0f0
        end
        ## update synaptic currents soma
        for r in eachindex(soma_syn)
            @unpack gsyn, E_rev, nmda = soma_syn[1]
            if nmda > 0.0f0
                is[1] +=
                    gsyn * g_s[i, r] * (v_s[i] + Δv[1] * dt - E_rev) /
                    (1.0f0 + (mg / b) * exp256(k * (v_s[i] + Δv[1] * dt)))
            else
                is[1] += gsyn * g_s[i, r] * (v_s[i] + Δv[1] * dt - E_rev)
            end
        end
        ## update synaptic currents dendrites
        for r in eachindex(dend_syn)
            @unpack gsyn, E_rev, nmda = dend_syn[r]
            if nmda > 0.0f0
                is[2] +=
                    gsyn * g_d[i, r] * (v_d[i] + Δv[2] * dt - E_rev) /
                    (1.0f0 + (mg / b) * exp256(k * (v_d[i] + Δv[2] * dt)))
            else
                is[2] += gsyn * g_d[i, r] * (v_d[i] + Δv[2] * dt - E_rev)
            end
        end
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
