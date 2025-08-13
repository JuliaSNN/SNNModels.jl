"""
    DendNeuronParameter{FT, IT, DT, ST, NMDAT, PST, PT}

A parameter struct for the Tripod neuron model, implementing an Adaptive Exponential Integrate-and-Fire (AdEx) model with dendritic compartments.

# Fields
- `C::FT`: Membrane capacitance (default: 281pF)
- `gl::FT`: Leak conductance (default: 40nS)
- `R::FT`: Total membrane resistance (default: nS/gl * GΩ)
- `τm::FT`: Membrane time constant (default: C/gl)
- `Er::FT`: Resting potential (default: -70.6mV)
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
- `ST`: Synapse type (default: Synapse)
- `NMDAT`: NMDA voltage dependency type (default: NMDAVoltageDependency{Float32})
- `PST`: Post-spike dynamics type (default: PostSpike{Float32})
- `PT`: Physiology type (default: Physiology)

# Examples
```jldoctest
julia> TripodParameter = DendNeuronParameter(ds = [200um, (200um, 400um)])
DendNeuronParameter{Float32, Int64, Vector{DendLength}, Synapse, NMDAVoltageDependency{Float32}, PostSpike{Float32}, Physiology}(281.0, 40.0, 25.0, 7.03125, -70.6, -55.6, -50.4, 2.0, 144.0, 4.0, 80.5, 10.0, 1.0, 1, 2, PostSpike{Float32}(10, 30.0), [200.0, (200.0, 400.0)], human_dend, TripodSomaSynapse, TripodDendSynapse, NMDAVoltageDependency{Float32}(0.001, 0.062, 3.57))

julia> BallAndStickParameter = DendNeuronParameter(ds = [(150um, 400um)])
DendNeuronParameter{Float32, Int64, Vector{DendLength}, Synapse, NMDAVoltageDependency{Float32}, PostSpike{Float32}, Physiology}(281.0, 40.0, 25.0, 7.03125, -70.6, -55.6, -50.4, 2.0, 144.0, 4.0, 80.5, 10.0, 1.0, 1, 2, PostSpike{Float32}(10, 30.0), [(150.0, 400.0)], human_dend, TripodSomaSynapse, TripodDendSynapse, NMDAVoltageDependency{Float32}(0.001, 0.062, 3.57))
```
"""
DendNeuronParameter

DendLength = Union{Float32, Tuple}

@snn_kw struct DendNeuronParameter{FT = Float32, 
                                    IT = Int64, 
                                    VIT = Vector{Int64}, 
                                    DT=Vector{DendLength},
                                    ST = SynapseArray,
                                    NMDAT = NMDAVoltageDependency{Float32},
                                    PST = PostSpike{Float32},
                                    PT = Physiology} <: AbstractAdExParameter
    #Membrane parameters
    C::FT = 281pF           # (pF) membrane timescale
    gl::FT = 40nS                # (nS) gl is the leaking conductance,opposite of Rm
    R::FT = nS / gl * GΩ               # (GΩ) total membrane resistance
    τm::FT = C / gl                # (ms) C / gl
    Er::FT = -70.6mV          # (mV) resting potential

    # AdEx model
    Vr::FT = -55.6mV     # (mV) Reset potential of membrane
    Vt::FT = -50.4mV          # (mv) Rheobase threshold
    ΔT::FT = 2mV            # (mV) Threshold sharpness

    # Adaptation parameters
    τw::FT = 144ms          #ms adaptation current relaxing time
    a::FT = 4nS            #nS adaptation current to membrane
    b::FT = 80.5pA         #pA adaptation current increase due to spike

    # After spike timescales and membrane
    AP_membrane::FT = 10.0f0mV
    BAP::FT = 1.0f0mV
    up::FT = 1ms
    τabs::FT = 2ms
    postspike::PST = PostSpike(A = 10, τA = 30ms)

    ## Dend parameters
    ds::DT # = [200um , (200um, 400um)] ## Dendritic segment lengths
    physiology::PT = human_dend

    ## Synapse parameters
    glu_receptors::VIT = [1, 2]
    gaba_receptors::VIT = [3, 4]
    soma_syn::ST = TripodSomaSynapse 
    dend_syn::ST = TripodDendSynapse 
    NMDA::NMDAT = NMDAVoltageDependency(mg = Mg_mM, b = nmda_b, k = nmda_k)
end

TripodParameter = DendNeuronParameter(ds = [200um, (200um, 400um)])
BallAndStickParameter = DendNeuronParameter(ds = [(150um, 400um)])

function MulticompartmentNeuron(;
    N::Int = 100,
    name::String = "TripodExc",
    dend = [(150um, 400um), (150um, 400um)],  # dendritic lengths
    NMDA::NMDAVoltageDependency = NMDAVoltageDependency(
        b = 3.36,  # NMDA voltage dependency parameter
        k = -0.077,  # NMDA voltage dependency parameter
        mg = 1.0f0,  # NMDA voltage dependency parameter
    ),
        # After spike timescales and membrane
    adex_param = (
        C = 281pF,  # membrane capacitance
        gl = 40nS,  # leak conductance
        R = (1 / 40)GΩ,  # membrane resistance
        τm = 281pF / 40nS,  # membrane time constant
        Er = -70.6mV,  # reset potential
        Vr = -55.6mV,  # resting potential
        Vt = -50.4mV,  # threshold potential
        ΔT = 2mV,  # slope factor
        τw = 144ms,  # adaptation time constant
        a = 4nS,  # subthreshold adaptation conductance
        b = 80.5pA,  # spike-triggered adaptation current
        AP_membrane = 2.0f0mV,  # action potential membrane potential
        BAP = 1.0f0mV,  # burst afterpotential
        up = 1ms,  # refractory period
        τabs = 2ms,  # absolute refractory period
    ),
    dend_syn::SynapseArray = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
    soma_syn::SynapseArray =  Synapse(DuarteGluSoma, MilesGabaSoma),  # connect EyalGluDend to MilesGabaDend
    postspike = PostSpike(A = 10.0, τA = 30.0), # post-spike adaptation
)
    param = DendNeuronParameter(;adex_param..., 
        ds = dend, 
        soma_syn = soma_syn, 
        dend_syn = dend_syn, 
        NMDA = NMDA,
    )
    if length(dend) == 2
        return Tripod(N = N, param = param, d1 = create_dendrite(N, dend[1]), d2 = create_dendrite(N, dend[2]), name= name)
    end
    if length(dend) == 1
        return BallAndStick(N = N, param = param, d = create_dendrite(N, dend[1]), name= name)
    end
end

export DendNeuronParameter, TripodParameter, BallAndStickParameter, MulticompartmentNeuron

@inline function update_synapses!(
    p::P,
    param::DendNeuronParameter,
    syn::SynapseArray,
    glu::Vector{Float32},
    gaba::Vector{Float32},
    g::Matrix{Float32},
    h::Matrix{Float32},
    dt::Float32,
) where {P <: AbstractPopulation}
    @unpack glu_receptors, gaba_receptors = param
    @unpack N = p
    @inbounds for n in glu_receptors
        @unpack τr⁻, τd⁻, α = syn[n]
        @turbo for i ∈ 1:N
            h[i, n] += glu[i] * α
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end
    @simd for n in gaba_receptors
        @unpack τr⁻, τd⁻, α = syn[n]
        @turbo for i ∈ 1:N
            h[i, n] += gaba[i] * α
            g[i, n] = exp64(-dt * τd⁻) * (g[i, n] + dt * h[i, n])
            h[i, n] = exp64(-dt * τr⁻) * (h[i, n])
        end
    end

    fill!(glu, 0.0f0)
    fill!(gaba, 0.0f0)
end

@inline function update_synapses!(
    p::P,
    param::DendNeuronParameter,
    syn::SynapseArray,
    glu::Vector{Vector{Float32}},
    gaba::Vector{Vector{Float32}},
    g:: Array{Float32,3},
    h::Array{Float32,3},
    d::Int,
    dt::Float32,
) where {P <: AbstractPopulation}
    @unpack glu_receptors, gaba_receptors = param
    @unpack N = p
    @inbounds for n in glu_receptors
        @unpack τr⁻, τd⁻, α = syn[n]
        @turbo for i ∈ 1:N
            h[d, i, n] += glu[d][i] * α
            g[d, i, n] = exp64(-dt * τd⁻) * (g[d, i, n] + dt * h[i, n])
            h[d, i, n] = exp64(-dt * τr⁻) * (h[d, i, n])
        end
    end
    @simd for n in gaba_receptors
        @unpack τr⁻, τd⁻, α = syn[n]
        @turbo for i ∈ 1:N
            h[d, i, n] += gaba[d][i] * α
            g[d, i, n] = exp64(-dt * τd⁻) * (g[d, i, n] + dt * h[d, i, n])
            h[d, i, n] = exp64(-dt * τr⁻) * (h[d, i, n])
        end
    end

    fill!(glu[d], 0.0f0)
    fill!(gaba[d], 0.0f0)
end


@inline function synaptic_current!(
            param::DendNeuronParameter, 
            syn::SynapseArray,
            v::Float32, 
            g,            
            is::Vector{Float32}, 
            comp::Int, 
            neuron::Int)
    @unpack glu_receptors, gaba_receptors = param
    @unpack mg, b, k = param.NMDA
    is[comp] = 0.f0
    @inbounds @fastmath begin
        @simd for n in glu_receptors
            @unpack gsyn, E_rev, nmda = syn[n]
            is[comp] += gsyn * g[neuron, n] * (v - E_rev) * (nmda==0.f0 ? 1.f0 : 1/(1.0f0 + (mg / b) * exp256(k * v)))
        end
        @simd for n in gaba_receptors
            @unpack gsyn, E_rev, nmda, name = syn[n]
            is[comp] += gsyn * g[neuron, n] * (v - E_rev)        
        end
    end
    is[comp] = clamp(is[comp], -1500, 1500)
end
