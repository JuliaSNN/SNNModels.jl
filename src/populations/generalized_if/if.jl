@snn_kw struct IFParameter{FT = Float32} <: AbstractDoubleExpParameter
    C::FT = 281pF        #(pF)
    gL::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = 20ms
    Vt::FT = -50mV # Membrane threshold potential
    Vr::FT = -60mV # Membrane reset potential
    El::FT = -70mV    # Membrane leak potential
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 2ms # Absolute refractory period
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
    a::FT = 0.0 # Subthreshold adaptation parameter
    b::FT = 0.0 #80.5pA # 'sra' current increment
    τw::FT = 0.0 #144ms # adaptation time constant (~Ca-activated K current inactivation)
end


@snn_kw struct IFCurrentParameter{FT = Float32} <: AbstractCurrentParameter
    C::FT = 281pF        #(pF)
    gL::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = 20ms # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -60mV # Reset potential
    El::FT = -70mV # Resting membrane potential
    R::FT = nS / gL # 40nS Membrane conductance
    ΔT::FT = 2mV # Slope factor
    τabs::FT = 2ms # Absolute refractory period
    #synapses
    τe::FT = 6ms # Rise time for excitatory synapses
    τi::FT = 2ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
end

@snn_kw struct IFCurrentDeltaParameter{FT = Float32} <: AbstractDeltaParameter
    C::FT = 281pF        #(pF)
    gL::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = 20ms # Membrane time constant
    Vt::FT = -50mV # Membrane potential threshold
    Vr::FT = -60mV # Reset potential
    El::FT = -70mV # Resting membrane potential
    R::FT = nS / gL # 40nS Membrane conductance
    ΔT::FT = 2mV # Slope factor
    τabs::FT = 2ms # Absolute refractory period
    #synapses
end


@snn_kw struct IFSinExpParameter{FT = Float32} <: AbstractSinExpParameter
    C::FT = 281pF        #(pF)
    gL::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = 20ms
    Vt::FT = -50mV
    Vr::FT = -60mV
    El::FT = Vr
    R::FT = nS / gL # Resistance
    ΔT::FT = 2mV # Slope factor
    τe::FT = 6ms # Rise time for excitatory synapses
    τi::FT = 2ms # Rise time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential
    E_e::FT = 0mV # Reversal potential
    τabs::FT = 2ms # Absolute refractory period
    gsyn_e::FT = 1.0 # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0 # Synaptic conductance for inhibitory synapses
end


@snn_kw mutable struct IF{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    GIFT<:AbstractGeneralizedIFParameter,
} <: AbstractGeneralizedIF
    id::String = randstring(12)
    name::String = "IF"
    param::GIFT = IFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    he::VFT = zeros(N)
    hi::VFT = zeros(N)
    tabs::VFT = zeros(N)
    w::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    syn_curr::VFT = zeros(N) # Synaptic current
    records::Dict = Dict()
    Δv::VFT = zeros(Float32, N)
    Δv_temp::VFT = zeros(Float32, N)
end


"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function update_neuron!(
    p::IF,
    param::T,
    dt::Float32,
) where {T<:AbstractGeneralizedIFParameter}
    @unpack N, v, w, I, tabs, fire, syn_curr = p
    @unpack τm, El, R, Vt, Vr, τabs = param

    @inbounds for i = 1:N
        # Idle time
        if tabs[i] > 0
            fire[i] = false
            tabs[i] -= 1
            continue
        end
        # Membrane potential
        v[i] += dt/τm * (-(v[i] - El) + R*(-w[i] + I[i]) - R*syn_curr[i])

        # Spike and absolute refractory period
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        tabs[i] = ifelse(fire[i], round(Int, τabs / dt), tabs[i])
    end


    # Adaptation current
    if (hasfield(typeof(param), :τw) && param.τw > 0.0f0)
        @unpack a, b, τw = param
        @inbounds for i = 1:N
            w[i] = ifelse(fire[i], w[i] + param.b, w[i])
            (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
        end
    end
end



export IF, IFParameter, IFSinExpParameter, IFCurrentParameter, IFCurrentDeltaParameter


# function Heun_update_neuron!(p::IF, param::T, dt::Float32) where {T<:AbstractIFParameter}
#     function _update_neuron!(
#         Δv::Vector{Float32},
#         p::IF,
#         param::T,
#         dt::Float32,
#     ) where {T<:AbstractIFParameter}
#         @unpack N, v, ge, gi, w, I, tabs, fire = p
#         @unpack τm, Vr, El, R, E_i, E_e, τabs, gsyn_e, gsyn_i = param
#         @inbounds for i = 1:N
#             if tabs[i] > 0
#                 v[i] = Vr
#                 fire[i] = false
#                 tabs[i] -= 1
#                 continue
#             end
#             Δv[i] =
#                 (
#                     -(v[i] + Δv[i] * dt - El) / R +# leakage
#                     -ge[i] * (v[i] + Δv[i] * dt - E_e) * gsyn_e +
#                     -gi[i] * (v[i] + Δv[i] * dt - E_i) * gsyn_i +
#                     -w[i] # adaptation
#                     +
#                     I[i] #synaptic term
#                 ) * R / τm
#         end
#     end
#     @unpack Δv_temp, Δv = p
#     _update_neuron!(Δv, p, param, dt)
#     @turbo for i = 1:p.N
#         Δv_temp[i] = Δv[i]
#     end
#     _update_neuron!(Δv, p, param, dt)
#     @turbo for i = 1:p.N
#         p.v[i] += 0.5f0 * (Δv_temp[i] + Δv[i]) * dt
#     end
#     !(hasfield(typeof(param), :τw) && param.τw > 0.0f0) && (return)
#     @unpack a, b, τw = param
#     @inbounds for i = 1:N
#         (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
#     end
# end
