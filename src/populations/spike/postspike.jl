"""
    PostSpike

A structure defining the parameters of a post-synaptic spike event.

# Fields
- `A::FT`: Amplitude of the Post-Synaptic Potential (PSP).
- `τA::FT`: Time constant of the PSP.

The type `FT` represents Float32.
"""
PostSpike

@snn_kw struct PostSpike{FT = Float32} <: AbstractSpikeParameter
    At::FT = 0mV
    τA::FT = 10ms
    AP_membrane::FT = 10.0f0mV
    τabs::FT = 1ms # Absolute refractory period
    up::FT = 1ms
end

export PostSpike
