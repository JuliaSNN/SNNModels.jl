"""
    DendNeuronParameter{FT, IT, DT, ST, NMDAT, PST, PT}

A parameter struct for the Tripod neuron model, implementing an Adaptive Exponential Integrate-and-Fire (AdEx) model with dendritic compartments.

# Fields
- `C::FT`: Membrane capacitance (default: 281pF)
- `gl::FT`: Leak conductance (default: 40nS)
- `R::FT`: Total membrane resistance (default: nS/gl * GΩ)
- `τm::FT`: Membrane time constant (default: C/gl)
- `El::FT`: Resting potential (default: -70.6mV)
- `Vr::FT`: Reset potential (default: -55.6mV)
- `Vt::FT`: Rheobase threshold (default: -50.4mV)
- `ΔT::FT`: Slope factor (default: 2mV)
- `τw::FT`: Adaptation current time constant (default: 144ms)
- `a::FT`: Subthreshold adaptation conductance (default: 4nS)
- `b::FT`: Spike-triggered adaptation increment (default: 80.5pA)
- `AP_membrane::FT`: After-potential membrane parameter (default: 10.0f0mV)
- `BAP::FT`: Backpropagating action potential parameter (default: 1.0f0mV)
- `up::FT`: Spike upstroke duration (default: 1ms)
- `τabs::FT`: Absolute refractory period (default: 2ms)
- `postspike::PST`: Post-spike dynamics (default: PostSpike(A=10, τA=30ms))
- `ds::DT`: Dendritic segment lengths (default: [200um, (200um, 400um)])
- `physiology::PT`: Dendritic physiology (default: human_dend)
- `soma_syn::ST`: Soma synapse type (default: TripodSomaSynapse)
- `dend_syn::ST`: Dendritic synapse type (default: TripodDendSynapse)
- `NMDA::NMDAT`: NMDA voltage dependency parameters (default: NMDAVoltageDependency(mg=Mg_mM, b=nmda_b, k=nmda_k))

# Type Parameters
- `FT`: Floating-point type for membrane parameters (default: Float32)
- `IT`: Integer type for time-related parameters (default: Int64)
- `DT`: Type for dendritic segment lengths (default: Vector{DendLength})
- `ST`: Receptors type (default: Receptors)
- `NMDAT`: NMDA voltage dependency type (default: NMDAVoltageDependency{Float32})
- `PST`: Post-spike dynamics type (default: PostSpike{Float32})
- `PT`: Physiology type (default: Physiology)

# Examples
```jldoctest
# julia> TripodParameter = DendNeuronParameter(ds = [200um, (200um, 400um)])
# DendNeuronParameter{Float32, Int64, Vector{DendLength}, Receptors, NMDAVoltageDependency{Float32}, PostSpike{Float32}, Physiology}(281.0, 40.0, 25.0, 7.03125, -70.6, -55.6, -50.4, 2.0, 144.0, 4.0, 80.5, 10.0, 1.0, 1, 2, PostSpike{Float32}(10, 30.0), [200.0, (200.0, 400.0)], human_dend, TripodSomaSynapse, TripodDendSynapse, NMDAVoltageDependency{Float32}(0.001, 0.062, 3.57))

# julia> BallAndStickParameter = DendNeuronParameter(ds = [(150um, 400um)])
# DendNeuronParameter{Float32, Int64, Vector{DendLength}, Receptors, NMDAVoltageDependency{Float32}, PostSpike{Float32}, Physiology}(281.0, 40.0, 25.0, 7.03125, -70.6, -55.6, -50.4, 2.0, 144.0, 4.0, 80.5, 10.0, 1.0, 1, 2, PostSpike{Float32}(10, 30.0), [(150.0, 400.0)], human_dend, TripodSomaSynapse, TripodDendSynapse, NMDAVoltageDependency{Float32}(0.001, 0.062, 3.57))
```
"""
DendNeuronParameter

DendLength = Union{Float32,Tuple}
abstract type AbstractDendriticTree end
struct TripodNeuron <: AbstractDendriticTree end
struct BallAndStickNeuron <: AbstractDendriticTree end
struct Multipod <: AbstractDendriticTree end

    # RT=ReceptorSynapseType,
    # AdExT = AdExParameter{Float32}

[(:s=>:d1), (:s=>:d2)] |> typeof

@snn_kw struct DendNeuronParameter{
    DT=Vector{DendLength},
    GT=Vector{Pair{Symbol,Symbol}},
    PT = Physiology{Float32},
    NT <: AbstractDendriticTree
} <: AbstractGeneralizedIFParameter

    ## Dend parameters
    ds::DT = [(200um, 400um) , (200um, 400um)] ## Dendritic segment lengths
    physiology::PT = human_dend
    geometry::GT = [(:s=>:d1), (:s=>:d2)]  ## Geometry between soma and dendrites
    type::NT = TripodNeuron()
end

function TripodParameter(;
        ds = [(200um, 400um), (200um, 400um)],
        type = TripodNeuron(),
        physiology = human_dend,
        geometry = [(:s=>:d1), (:s=>:d2)],
        )
    return DendNeuronParameter(
        ds = ds,
        type = type,
        physiology = physiology,
        geometry = geometry,
    )
end

function BallAndStickParameter(;
        ds = [(150um, 400um)],
        type = BallAndStickNeuron(),
        physiology = human_dend,
        geometry = [(:s=>:d)],
        )
    return DendNeuronParameter(
        ds = ds,
        type = type,
        physiology = physiology,
        geometry = geometry,
    )
end

function Population(param::T; kwargs...) where {T<:DendNeuronParameter}
    if param.type isa TripodNeuron
        return Tripod(;param, kwargs...)
    elseif param.type isa BallAndStickNeuron
        return BallAndStick(;param, kwargs...)
    else
        error("Dendritic segments must be either 1 (BallAndStick) or 2 (Tripod).")
    end
end

export BallAndStickParameter, TripodParameter, DendNeuronParameter

# function MulticompartmentNeuron(;
#     N::Int = 100,
#     name::String = "TripodExc",
#     ds = [(150um, 400um), (150um, 400um)],  # dendritic lengths

#     # After spike timescales and membrane
#     adex = (
#         C = 281pF,  # membrane capacitance
#         gl = 40nS,  # leak conductance
#         τm = 281pF / 40nS,  # membrane time constant
#         Vt = -50.4mV,  # threshold potential
#         Vr = -55.6mV,  # resting potential
#         El = -70.6mV,  # reset potential
#         R = (1 / 40)GΩ,  # membrane resistance

#         ΔT = 2mV,  # slope factor
#         τw = 144ms,  # adaptation time constant
#         a = 4nS,  # subthreshold adaptation conductance
#         b = 80.5pA,  # spike-triggered adaptation current
#         τabs = 2ms,  # absolute refractory period
#     ),


#     dend_syn::ReceptorSynapse = TripodDendSynapse, # defines glutamaterbic and gabaergic receptors in the dendrites
#     soma_syn::ReceptorSynapse = TripodSomaSynapse,  # connect EyalGluDend to MilesGabaDend
#     spike=PostSpike(),
# )
#     param = DendNeuronParameter(;
#         adex,
#         spike,
#         ds, 
#         soma_syn,
#         dend_syn,
#     )

#     if length(dend) == 2
#         return Tripod(
#             N = N,
#             param = param,
#             d1 = create_dendrite(N, dend[1]),
#             d2 = create_dendrite(N, dend[2]),
#             name = name,
#         )
#     end
#     if length(dend) == 1
#         return BallAndStick(
#             N = N,
#             param = param,
#             d = create_dendrite(N, dend[1]),
#             name = name,
#         )
#     end
# end


    # AdEx model
    # adex::AdExT = AdExParameter(
    #         C = 281pF,
    #         gl  = 40nS,
    #         R = 0.025GΩ,
    #         τm  = 7ms,
    #         El  = -70.6mV,     
    #         Vr  = -55.6mV,
    #         Vt  = -50.4mV,
    #         ΔT  = 2mV,
    #         τw  = 144ms,
    #         a = 4nS,
    #         b = 80.5pA,
    # )

    # # After spike timescales and membrane
    # spike::PST = PostSpike(At = 10, τA = 30ms)


# @inline function update_synapses!(
#     p::P,
#     param::DendNeuronParameter,
#     syn::ReceptorArray,
#     glu::Vector{Vector{Float32}},
#     gaba::Vector{Vector{Float32}},
#     g::Array{Float32,3},
#     h::Array{Float32,3},
#     d::Int,
#     dt::Float32,
# ) where {P<:AbstractPopulation}
#     @unpack glu_receptors, gaba_receptors = param
#     @unpack N = p
#     @inbounds for n in glu_receptors
#         @unpack τr⁻, τd⁻, α = syn[n]
#         @turbo for i ∈ 1:N
#             h[d, i, n] += glu[d][i] * α
#             g[d, i, n] = exp64(-dt * τd⁻) * (g[d, i, n] + dt * h[i, n])
#             h[d, i, n] = exp64(-dt * τr⁻) * (h[d, i, n])
#         end
#     end
#     @simd for n in gaba_receptors
#         @unpack τr⁻, τd⁻, α = syn[n]
#         @turbo for i ∈ 1:N
#             h[d, i, n] += gaba[d][i] * α
#             g[d, i, n] = exp64(-dt * τd⁻) * (g[d, i, n] + dt * h[d, i, n])
#             h[d, i, n] = exp64(-dt * τr⁻) * (h[d, i, n])
#         end
#     end

#     fill!(glu[d], 0.0f0)
#     fill!(gaba[d], 0.0f0)
# end

