abstract type AbstractDoubleExpParameter <: AbstractSynapseParameter end

@doc raw"""
    DoubleExpSynapse(; τre = 1ms, τde = 6ms, τri = 0.5ms, τdi = 2ms,
                       E_i = -75mV, E_e = 0mV, gsyn_e = 1, gsyn_i = 1)

Conductance-based synapse with double-exponential (rise and decay) kinetics, one excitatory
and one inhibitory conductance per neuron. This is the default synapse of `IF` and `AdEx`.

# Equations
Each spike increments the auxiliary variable ``h``, which drives the conductance ``g``:
```math
\frac{dh_e}{dt} = -\frac{h_e}{\tau_{re}} + \sum_k w_k\,\delta(t - t_k), \qquad
\frac{dg_e}{dt} = -\frac{g_e}{\tau_{de}} + h_e
```
(same for ``h_i, g_i`` with ``\tau_{ri}, \tau_{di}``), and
```math
I_{syn} = g_{syn,e}\, g_e\,(V - E_e) + g_{syn,i}\, g_i\,(V - E_i).
```
The kernel is not normalised: a unit jump of ``h`` gives a conductance time course
``\frac{\tau_r \tau_d}{\tau_d - \tau_r}\left(e^{-t/\tau_d} - e^{-t/\tau_r}\right)``.

# Integration
Forward Euler: the input is added to `he`/`hi`; then `ge += dt (-ge/τde + he)` (using the
updated `he`) and `he += -dt he / τre` (same for the inhibitory pair). The receptor buffers
are then zeroed.

# Fields
- `τre::FT = 1ms`: rise time constant, excitatory (ms).
- `τde::FT = 6ms`: decay time constant, excitatory (ms).
- `τri::FT = 0.5ms`: rise time constant, inhibitory (ms).
- `τdi::FT = 2ms`: decay time constant, inhibitory (ms).
- `E_i::FT = -75mV`: inhibitory reversal potential (mV).
- `E_e::FT = 0mV`: excitatory reversal potential (mV).
- `gsyn_e::FT = 1.0`: scaling of the excitatory conductance (dimensionless).
- `gsyn_i::FT = 1.0`: scaling of the inhibitory conductance (dimensionless).

`FT` defaults to `Float32`. State variables: `DoubleExpSynapseVars`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
syn = SNN.DoubleExpSynapse(τre = 1ms, τde = 6ms, τri = 0.5ms, τdi = 2ms)
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS), synapse = syn)
```
"""
DoubleExpSynapse

@snn_kw struct DoubleExpSynapse{FT = Float32} <: AbstractDoubleExpParameter
    τre::FT = 1ms # Rise time for excitatory synapses
    τde::FT = 6ms # Decay time for excitatory synapses
    τri::FT = 0.5ms # Rise time for inhibitory synapses
    τdi::FT = 2ms # Decay time for inhibitory synapses
    E_i::FT = -75mV # Reversal potential excitatory synapses
    E_e::FT = 0mV #Reversal potential excitatory synapses
    gsyn_e::FT = 1.0f0 #norm_synapse(τre, τde) # Synaptic conductance for excitatory synapses
    gsyn_i::FT = 1.0f0 #norm_synapse(τri, τdi) # Synaptic conductance for inhibitory synapses
end

"""
    DoubleExpSynapseVars{VFT} <: AbstractSynapseVariable

State variables of `DoubleExpSynapse`, created by `synaptic_variables(::DoubleExpSynapse, N)`.

# Fields
- `N::Int = 100`: number of neurons.
- `ge::VFT`: excitatory conductance (nS).
- `gi::VFT`: inhibitory conductance (nS).
- `he::VFT`: excitatory rise (auxiliary) variable (nS/ms).
- `hi::VFT`: inhibitory rise (auxiliary) variable (nS/ms).
"""
DoubleExpSynapseVars
@snn_kw struct DoubleExpSynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    ge::VFT = zeros(eltype(VFT), N)
    gi::VFT = zeros(eltype(VFT), N)
    he::VFT = zeros(eltype(VFT), N)
    hi::VFT = zeros(eltype(VFT), N)
end

function synaptic_variables(synapse::DoubleExpSynapse{FT}, N::Int) where {FT}
    return DoubleExpSynapseVars(; N = N, VFT = Vector{FT})
end

function update_synapses!(
    p::P,
    synapse::T,
    receptors::RECT,
    synvars::DoubleExpSynapseVars,
    dt::AbstractFloat,
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpParameter,RECT<:NamedTuple}
    update_synapses!(p, synapse, receptors, synvars, dt, 1:Int(synvars.N))
end

# Chunk method: the neurons `rng` only; clears the spike buffers of the same neurons.
function update_synapses!(
    p::P,
    synapse::T,
    receptors::RECT,
    synvars::DoubleExpSynapseVars,
    dt::AbstractFloat,
    rng::UnitRange{Int},
) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpParameter,RECT<:NamedTuple}
    @unpack ge, gi, he, hi = synvars
    @unpack τde, τre, τdi, τri = synapse
    @unpack gaba, glu = receptors
    z = zero(eltype(glu))
    @inbounds @simd for i ∈ rng
        he[i] += glu[i]
        hi[i] += gaba[i]
        ge[i] += dt * (-ge[i] / τde + he[i])
        he[i] += dt * (-he[i] / τre)
        gi[i] += dt * (-gi[i] / τdi + hi[i])
        hi[i] += dt * (-hi[i] / τri)
        glu[i] = z
        gaba[i] = z
    end
end


@inline function synaptic_current!(
    p::T,
    synapse::DoubleExpSynapse,
    synvars::DoubleExpSynapseVars,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
    rng::UnitRange{Int} = 1:Int(p.N),
) where {T<:AbstractPopulation,VT1<:AbstractVector,VT2<:AbstractVector}
    @unpack gsyn_e, gsyn_i, E_e, E_i = synapse
    @unpack ge, gi = synvars
    @inbounds @simd for i ∈ rng
        syncurr[i] = ge[i] * (v[i] - E_e) * gsyn_e + gi[i] * (v[i] - E_i) * gsyn_i
    end
end

_chunk_synapse(::AbstractDoubleExpParameter) = true

export DoubleExpSynapse
