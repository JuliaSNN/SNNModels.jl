"""
    IFParameter{FT<:AbstractFloat} <: AbstractGeneralizedIFParameter

A parameter struct for the Leaky Integrate-and-Fire neuron model.

# Fields
- `C::FT`: Membrane capacitance (default: 281 pF)
- `gl::FT`: Leak conductance (default: 40 nS)
- `τm::FT`: Membrane time constant (default: 20 ms)
- `Vt::FT`: Membrane threshold potential (default: -50 mV)
- `Vr::FT`: Membrane reset potential (default: -60 mV)
- `El::FT`: Membrane leak potential (default: -70 mV)
- `R::FT`: Membrane resistance (calculated as 1/gl)
- `ΔT::FT`: Slope factor for exponential spiking (default: 2 mV)
- `a::FT`: Subthreshold adaptation parameter (default: 0.0)
- `b::FT`: Spike-triggered adaptation current increment (default: 0.0)
- `τw::FT`: Adaptation time constant (default: 0.0)

This struct implements the Integrate-and-Fire neuron model with optional adaptation currents.
The parameters are based on the standard Izhikevich model but adapted for generalized integrate-and-fire dynamics.
"""
IFParameter

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

"""
    IF{VFT, VBT, GIFT, SYNT} <: AbstractGeneralizedIF

A mutable struct representing a population of Leaky Integrate-and-Fire (LIF) neurons with optional adaptation currents.


# Fields
- `param::IFParameter`: Neuron parameters (default: `IFParameter()`)
- `synapse<:AbstractSynapseParameter`: Synapse parameters (default: `DoubleExpSynapse()`)
- `spike::PostSpike`: Post-spike behavior parameters (default: `PostSpike()`)

- `id::String`: Unique identifier for the population (default: random 12-character string)
- `name::String`: Name of the population (default: "IF")
- `N::Int32`: Number of neurons in the population (default: 100)
- `v::VFT`: Membrane potentials (initialized between `Vr` and `Vt`)
- `ge::VFT`: Excitatory synaptic conductances (default: zeros)
- `gi::VFT`: Inhibitory synaptic conductances (default: zeros)
- `he::VFT`: Excitatory synaptic currents (default: zeros)
- `hi::VFT`: Inhibitory synaptic currents (default: zeros)
- `tabs::VFT`: Absolute refractory periods (default: zeros)
- `w::VFT`: Adaptation currents (default: zeros)
- `fire::VBT`: Spike flags (default: zeros)
- `I::VFT`: External currents (default: zeros)
- `syn_curr::VFT`: Total synaptic currents (default: zeros)
- `records::Dict`: Dictionary for storing simulation records (default: empty)
- `Δv::VFT`: Temporary storage for Heun integration (default: zeros)
- `Δv_temp::VFT`: Temporary storage for Heun integration (default: zeros)

# Type Parameters
- `VFT`: Type of vector for floating-point values (default: `Vector{Float32}`)
- `VBT`: Type of vector for boolean values (default: `Vector{Bool}`)

This struct represents a population of neurons following the generalized Integrate-and-Fire model with optional adaptation currents.
It includes all necessary state variables for simulating the neuron dynamics and synaptic interactions.
"""
IF

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

function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target::Nothing = nothing,
) where {T<:IF}
    syn = get_synapse_symbol(post.synapse, sym)
    sym = Symbol(syn)
    g = getfield(post, sym)
    v_post = getfield(post, :v)
    push!(targets, :sym => sym)
    return g, v_post
end


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
