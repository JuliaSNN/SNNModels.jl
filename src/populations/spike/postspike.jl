"""
    PostSpike{FT = Float32}(; At = 0mV, τA = 30ms, AP_membrane = 20mV, τabs = 2ms, up = 1ms)

Spike-related parameters shared by the generalized integrate-and-fire populations
(`IF`, `AdEx`, `Tripod`, `BallAndStick`). Passed as the `spike` field of the population.

Which fields are read depends on the model:
- `IF`: only `τabs`.
- `AdEx`: `τabs`, `At`, `τA` (adaptive threshold). The spike peak is hard-coded to 20 mV in
  `AdEx`, `AP_membrane` is not used there.
- `Tripod` / `BallAndStick`: `τabs`, `up`, `At`, `τA`, `AP_membrane`.

The defaults changed in 2.1.0 (results change for every population that relies on them): `τabs` 1 -> 2 ms (all models),
`τA` 10 -> 30 ms (`AdEx`, `Tripod`, `BallAndStick` with `At != 0`), `AP_membrane` 10 -> 20 mV (`Tripod`, `BallAndStick`).
`At = 0` and `up = 1` are unchanged.

# Fields
- `At::FT = 0mV`: increment of the adaptive spike threshold ``θ`` at each spike (mV).
- `τA::FT = 30ms`: time constant with which ``θ`` relaxes back to `Vt` (ms).
- `AP_membrane::FT = 20mV`: somatic membrane potential imposed during the action potential
  (multicompartment models only) (mV).
- `τabs::FT = 2ms`: absolute refractory period (ms). It is converted to an integer number of
  steps `round(Int, τabs / dt)` when a spike occurs.
- `up::FT = 1ms`: duration of the action potential (multicompartment models only); these models
  set the refractory counter to `round(Int, (up + τabs) / dt)` (ms).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.AdEx(N = 10, spike = SNN.PostSpike(τabs = 2ms, At = 5mV, τA = 30ms))
```
"""
PostSpike

@snn_kw struct PostSpike{FT = Float32} <: AbstractSpikeParameter
    At::FT = 0mV
    τA::FT = 30ms
    AP_membrane::FT = 20.0f0mV
    τabs::FT = 2ms # Absolute refractory period
    up::FT = 1ms
end

convert_precision(x::PostSpike, ::Type{T}) where {T<:AbstractFloat} = _convert_float_struct(x, T)

export PostSpike
