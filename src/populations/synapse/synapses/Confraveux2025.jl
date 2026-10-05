abstract type AbstractConfavreux2025 <: AbstractSynapseParameter end

@doc raw"""
    Confavreux2025Synapse(; τAMPA = 5ms, τNMDA = 100ms, τGABA = 10ms,
                            E_i = -80mV, E_e = 0mV, α = 0.23)

Conductance-based synapse with an AMPA, a slow NMDA-like and a GABA conductance, in which the
NMDA conductance is a low-pass filtered copy of the AMPA conductance and excitation is a
fixed mixture of the two. There is no magnesium block.

# Equations
```math
\frac{dg_{AMPA}}{dt} = -\frac{g_{AMPA}}{\tau_{AMPA}} + x_{glu}(t), \qquad
\frac{dg_{GABA}}{dt} = -\frac{g_{GABA}}{\tau_{GABA}} + x_{gaba}(t), \qquad
\tau_{NMDA}\frac{dg_{NMDA}}{dt} = g_{AMPA} - g_{NMDA}
```
```math
I_{syn} = \left(\alpha\, g_{AMPA} + (1-\alpha)\, g_{NMDA}\right)(V - E_e) + g_{GABA}\,(V - E_i)
```
where ``x_{glu}``, ``x_{gaba}`` are the contents of the receptor buffers (sum of the weights
of the spikes received in the step).

# Integration
Forward Euler, in this order: `gAMPA += dt (-gAMPA/τAMPA + glu)`,
`gGABA += dt (-gGABA/τGABA + gaba)`, `gNMDA += dt (gAMPA - gNMDA)/τNMDA` (with the updated
`gAMPA`). Note that the input enters multiplied by `dt`: a spike of weight `w` increments
`gAMPA` by `w dt`, unlike the other synapse models, where the increment is `w`.

# Fields
- `τAMPA::FT = 5ms`: decay time constant of the AMPA conductance (ms).
- `τNMDA::FT = 100ms`: time constant of the NMDA low-pass filter (ms).
- `τGABA::FT = 10ms`: decay time constant of the GABA conductance (ms).
- `E_i::FT = -80mV`: inhibitory reversal potential (mV).
- `E_e::FT = 0mV`: excitatory reversal potential (mV).
- `α::FT = 0.23`: fraction of the excitatory conductance carried by AMPA (the NMDA share is
  ``1 - \alpha``).

`FT` defaults to `Float32`. State variables: `Confavreux2025SynapseVars`.

# References
The name refers to Confavreux et al. (2025); the full reference is not given in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS),
           synapse = SNN.Confavreux2025Synapse())
```
"""
Confavreux2025Synapse

@snn_kw struct Confavreux2025Synapse{FT = Float32} <: AbstractConfavreux2025
    τAMPA::FT = 5ms # Rise time for excitatory synapses
    τNMDA::FT = 100ms # Decay time for excitatory synapses
    τGABA::FT = 10ms # Rise time for inhibitory synapses
    E_i::FT = -80mV # Reversal potential excitatory synapses
    E_e::FT = 0mV #Reversal potential excitatory synapses
    α::FT = 0.23f0 # NMDA voltage dependence parameter
end

"""
    Confavreux2025SynapseVars{VFT} <: AbstractSynapseVariable

State variables of `Confavreux2025Synapse`.

# Fields
- `N::Int = 100`: number of neurons.
- `gAMPA::VFT`: AMPA conductance (nS).
- `gNMDA::VFT`: NMDA conductance (nS).
- `gGABA::VFT`: GABA conductance (nS).
"""
Confavreux2025SynapseVars

@snn_kw struct Confavreux2025SynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
    N::Int = 100
    gAMPA::VFT = zeros(Float32, N)
    gNMDA::VFT = zeros(Float32, N)
    gGABA::VFT = zeros(Float32, N)
end

function synaptic_variables(synapse::Confavreux2025Synapse, N::Int)
    return Confavreux2025SynapseVars(;
        N = N,
    )
end

function update_synapses!(
    p::P,
    synapse::T,
    receptors::RECT,
    synvars::Confavreux2025SynapseVars,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractConfavreux2025,RECT<:NamedTuple}
    @unpack N, gAMPA, gNMDA, gGABA = synvars
    @unpack τAMPA, τNMDA, τGABA = synapse
    @unpack gaba, glu = receptors
    @inbounds @simd for i ∈ 1:N
        gAMPA[i] += dt * (-gAMPA[i] / τAMPA + glu[i])
        gGABA[i] += dt * (-gGABA[i] / τGABA + gaba[i])
        gNMDA[i] += dt * (gAMPA[i] - gNMDA[i])/ τNMDA
    end
    fill!(glu, 0.0f0)
    fill!(gaba, 0.0f0)
end


@inline function synaptic_current!(
    p::T,
    synapse::Confavreux2025Synapse,
    synvars::Confavreux2025SynapseVars,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {T<:AbstractPopulation,VT1<:AbstractVector,VT2<:AbstractVector}
    @unpack gAMPA, gNMDA, gGABA = synvars
    @unpack E_e, E_i, α = synapse
    @unpack N = p
    @inbounds @simd for i ∈ 1:N
        syncurr[i] = (α * gAMPA[i] + (1 - α)*gNMDA[i]) * (v[i] - E_e) + gGABA[i] * (v[i] - E_i)
    end
end

export Confavreux2025Synapse