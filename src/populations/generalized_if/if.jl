@snn_kw struct IFParameter{FT = Float32} <: AbstractGeneralizedIFParameter
    C::FT = 281pF        #(pF)
    gl::FT = 40nS         #(nS) leak conductance #BretteGerstner2005 says 30 nS
    τm::FT = 20ms
    Vt::FT = -50mV # Membrane threshold potential
    Vr::FT = -60mV # Membrane reset potential
    El::FT = -70mV    # Membrane leak potential
    R::FT = nS / gl # Resistance
    ΔT::FT = 2mV # Slope factor
    a::FT = 0.0 # Subthreshold adaptation parameter
    b::FT = 0.0 #80.5pA # 'sra' current increment
    τw::FT = 0.0 #144ms # adaptation time constant (~Ca-activated K current inactivation)
end


@snn_kw mutable struct IF{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    GIFT<:AbstractGeneralizedIFParameter,
    SYNT<:AbstractSynapseParameter
} <: AbstractGeneralizedIF

    param::GIFT = IFParameter()
    synapse::SYNT = DoubleExpSynapse()
    spike::PostSpike = PostSpike()

    id::String = randstring(12)
    name::String = "IF"
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

function Population(param::IFParameter; synapse::AbstractSynapseParameter, spike::PostSpike, N, kwargs...)
    return IF(;N, param, synapse, spike, kwargs...)
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
    @unpack τm, El, R, Vt, Vr= param
    @unpack τabs = p.spike

    # @inbounds 
    for i = 1:N
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
        # @inbounds 
        for i = 1:N
            w[i] = ifelse(fire[i], w[i] + param.b, w[i])
            (w[i] += dt * (a * (v[i] - El) - w[i]) / τw)
        end
    end
end



export IF, IFParameter


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
