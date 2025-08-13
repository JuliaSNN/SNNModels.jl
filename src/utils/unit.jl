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
    Load all the units defined in the module into the current scope.
    This macro generates a block of expressions that assign the unit constants
        
    The base units in the module are:
    - cm : centimeters
    - ms : milliseconds
    - kHz : kilohertz
    - mV : millivolts
    - pA : picoamperes
    - pF : picofarads
    - nS : nanosiemens
    - GΩ : gigaohms
    - uM : micromolar

    The derived units in the module are obtained as multiple or division of the base units. 

    The standard integration time step is 0.125ms, which is used in the simulation.
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
