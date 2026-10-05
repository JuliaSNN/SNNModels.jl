const metre = 1e2 |> Float32
const meter = metre |> Float32
const cm = metre / 1e2 |> Float32
const mm = metre / 1e3 |> Float32
const um = metre / 1e6 |> Float32
const nm = metre / 1e9 |> Float32
const cm2 = cm * cm |> Float32
const m2 = metre * metre |> Float32
const um2 = um * um |> Float32
const nm2 = nm * nm |> Float32
const second = 1e3 |> Float32
const s = second |> Float32
const ms = second / 1e3 |> Float32
const Hz = 1 / second |> Float32
const kHz = Hz * 1e3 |> Float32
const voltage = 1e3 |> Float32
const mV = voltage / 1e3 |> Float32
const ampere = 1e12 |> Float32
const mA = ampere / 1e3 |> Float32
const uA = ampere / 1e6 |> Float32
const μA = ampere / 1e6 |> Float32
const nA = ampere / 1e9 |> Float32
const pA = ampere / 1e12 |> Float32
const farad = 1e12 |> Float32
const mF = farad / 1e3 |> Float32
const uF = farad / 1e6 |> Float32
const μF = farad / 1e6 |> Float32
const nF = farad / 1e9 |> Float32
const pF = farad / 1e12 |> Float32
const ufarad = uF |> Float32
const siemens = 1e9 |> Float32
const mS = siemens / 1e3 |> Float32
const msiemens = mS |> Float32
const nS = siemens / 1e9 |> Float32
const nsiemens = nS |> Float32
const Ω = 1 / siemens |> Float32
const MΩ = Ω * 1e6 |> Float32
const GΩ = Ω * 1e9 |> Float32
const M = 1e6 |> Float32
const mM = M / 1e3 |> Float32
const uM = M*1e-6 |> Float32
const nM = M*1e-9 |> Float32

second / Ω ≈ farad
dt = 0.125ms

@assert second / Ω ≈ farad
@assert Ω * siemens ≈ 1
@assert Ω * ampere ≈ voltage
@assert ampere * second / voltage == farad

"""
    @load_units

Define the SNNModels unit constants as local variables in the calling scope.

SNNModels represents physical quantities as plain `Float32` numbers in a fixed unit system; the
constants convert a number to that system (e.g. `10ms == 10f0`, `1s == 1000f0`,
`10Hz == 0.01f0`). The base units are

| Quantity      | Base unit | Value 1 equals |
|:------------- |:--------- |:-------------- |
| length        | `cm`      | 1 cm (`metre = 100`, `um = 1e-4`) |
| time          | `ms`      | 1 ms (`s = second = 1000`) |
| frequency     | `kHz`     | 1 kHz (`Hz = 1e-3`) |
| voltage       | `mV`      | 1 mV |
| current       | `pA`      | 1 pA |
| capacitance   | `pF`      | 1 pF |
| conductance   | `nS`      | 1 nS |
| resistance    | `GΩ`      | 1 GΩ (`MΩ = 1e-3`) |
| concentration | `uM`      | 1 μM (`M = 1e6`, `mM = 1e3`) |

Constants defined by the macro: `metre, meter, cm, mm, um, nm, cm2, m2, um2, nm2, second, s, ms,
Hz, kHz, voltage, mV, ampere, mA, uA, μA, nA, pA, farad, uF, μF, nF, pF, ufarad, siemens, mS,
msiemens, nS, nsiemens, Ω, MΩ, GΩ, M, mM, uM, nM`. (`mF` exists in the module but is not
defined by the macro.)

The default integration step of `sim!`/`train!` is `0.125ms`.

# Example
```julia
using SpikingNeuralNetworks
@load_units
duration = 2s      # 2000.0f0
rate = 10Hz        # 0.01f0 (per ms)
```
"""
macro load_units()
    exs = map((
        :metre,
        :Hz,
        :kHz,
        :meter,
        :cm,
        :mm,
        :um,
        :nm,
        :cm2,
        :m2,
        :um2,
        :nm2,
        :second,
        :s,
        :ms,
        :Hz,
        :voltage,
        :mV,
        :ampere,
        :mA,
        :uA,
        :μA,
        :nA,
        :pA,
        :farad,
        :Ω,
        :uF,
        :μF,
        :nF,
        :pF,
        :ufarad,
        :siemens,
        :mS,
        :msiemens,
        :nS,
        :nsiemens,
        :Ω,
        :MΩ,
        :GΩ,
        :M,
        :mM,
        :uM,
        :nM,
    )) do s
        :($s = getfield($@__MODULE__, $(QuoteNode(s))))
    end
    ex = Expr(:block, exs...)
    esc(ex)
end

export @load_units
