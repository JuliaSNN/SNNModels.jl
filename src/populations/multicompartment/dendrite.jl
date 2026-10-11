## Set physiology
"""
    Physiology(Ri, Rd, Cd)
    Physiology(; Ri, Rd, Cd)

Passive cable properties of a dendrite, per unit length or area, in the library unit system.

# Fields
- `Ri::T`: axial (intracellular) resistivity, e.g. `200Ω*cm` (stored in GΩ cm).
- `Rd::T`: specific membrane resistance, e.g. `38907Ω*cm^2` (stored in GΩ cm²).
- `Cd::T`: specific membrane capacitance, e.g. `0.5μF/cm^2` (stored in pF/cm²).

Predefined: `human_dend` (`Ri = 200 Ω cm`, `Rd = 38907 Ω cm²`, `Cd = 0.5 μF/cm²`) and
`mouse_dend` (`Ri = 200 Ω cm`, `Rd = 1700 Ω cm²`, `Cd = 1 μF/cm²`). References for these
values are not given in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
phys = SNN.Physiology(Ri = 150Ω*cm, Rd = 20000Ω*cm^2, Cd = 1μF/cm^2)
```
"""
@kwdef struct Physiology{T}
    Ri::T ## in Ω*cm
    Rd::T ## in Ω*cm^2
    Cd::T ## in pF/cm^2
end

"""
    human_dend :: Physiology{Float32}

Human cortical dendrite: `Ri = 200 Ω cm`, `Rd = 38907 Ω cm²`, `Cd = 0.5 μF/cm²`.
Default `physiology` of `TripodParameter`, `BallAndStickParameter` and `create_dendrite`.
"""
human_dend = Physiology(200 * Ω * cm, 38907 * Ω * cm^2, 0.5μF / cm^2|>Float32)
"""
    mouse_dend :: Physiology{Float32}

Mouse cortical dendrite: `Ri = 200 Ω cm`, `Rd = 1700 Ω cm²`, `Cd = 1 μF/cm²`.
"""
mouse_dend = Physiology(200 * Ω * cm, 1700Ω * cm^2, 1μF / cm^2 |> Float32)

export human_dend, mouse_dend

@doc raw"""
    G_axial(; Ri, d, l)

Axial conductance (nS) of a cylinder of length `l` and diameter `d` (cm, e.g. `200um`) with
axial resistivity `Ri` (GΩ cm, e.g. `200Ω*cm`):
``G_{ax} = \frac{\pi d^2}{4 R_i l}``.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
SNN.G_axial(Ri = 200Ω*cm, d = 4um, l = 200um)   # ≈ 31.4 nS
```
"""
function G_axial(; Ri = Ri, d = d, l = l)
    ((π * d * d) / (Ri * l * 4))
end

@doc raw"""
    G_mem(; Rd, d, l)

Membrane (leak) conductance (nS) of the lateral surface of a cylinder of length `l` and
diameter `d` (cm) with specific membrane resistance `Rd` (GΩ cm²):
``G_m = \frac{\pi d l}{R_d}``.
"""
function G_mem(; Rd = Rd, d = d, l = l)
    ((l * d * π) / Rd)
end

@doc raw"""
    C_mem(; Cd, d, l)

Membrane capacitance (pF) of the lateral surface of a cylinder of length `l` and diameter `d`
(cm) with specific capacitance `Cd` (pF/cm²): ``C = C_d\,\pi d l``.
"""
function C_mem(; Cd = Cd, d = d, l = l)
    (Cd * π * d * l)
end


"""
    Dendrite(; N = 100, El, l, d, C, gax, gm)

Passive properties of one dendritic compartment for each of the `N` neurons of a dendritic
population (`Tripod` has two, `d1` and `d2`; `BallAndStick` has one, `d`). Normally built with
`create_dendrite(N, l)`.

# Fields (all of type `VFT` (`Vector{Float32}` by default, `Vector{Float64}` in a `Float64` population) and length `N`, default `zeros(N)`)
- `N::Int32 = 100`: number of neurons.
- `El::VFT`: leak reversal (resting) potential of the dendrite (mV), used by the
  `Tripod`/`BallAndStick` dendritic equations. `create_dendrite` sets it from its `El` keyword
  (default `-70.6mV`); `Tripod` and `BallAndStick` pass the somatic `adex.El` by default.
- `l::VFT`: compartment length (cm); `-1` marks a disconnected compartment (`l <= 0`).
- `d::VFT`: compartment diameter (cm).
- `C::VFT`: membrane capacitance (pF), `C_mem`.
- `gax::VFT`: axial conductance to the soma (nS), `G_axial`.
- `gm::VFT`: membrane leak conductance (nS), `G_mem`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
d = SNN.create_dendrite(10, (150um, 300um))   # 10 dendrites, length uniform in 150-300 μm
d.gax[1], d.C[1]
```
"""
Dendrite

@snn_kw struct Dendrite{VFT = Vector{Float32}}
    N::Int32 = 100
    El::VFT = zeros(N)             # (mV) resting potential
    l::VFT = zeros(N) # μm distance from next compartment
    d::VFT = zeros(N) # μm dendrite diameter
    C::VFT = zeros(N)
    gax::VFT = zeros(N)# (nS) axial conductance
    gm::VFT = zeros(N)
end

"""
    create_dendrite(N::Int, l; El = -70.6mV, d = 4um, physiology = human_dend, FT = Float32) -> Dendrite
    create_dendrite(l; d = 4um, physiology = human_dend) -> NamedTuple
    create_dendrite(; l, kwargs...)

Compute the passive parameters of a cylindrical dendrite of length `l` and diameter `d`
from the cable properties `physiology` (`Physiology`).

`l` is either a length (cm, e.g. `200um`) or a tuple `(lmin, lmax)`, in which case the length
is drawn uniformly from `lmin:1um:lmax` (independently for each of the `N` neurons in the
first form). Lengths above `500um` raise an error. For `l > 0` it returns
`(gm = G_mem(Rd, d, l), gax = G_axial(Ri, d, l), C = C_mem(Cd, d, l), l, d)`; for `l <= 0` it
returns a disconnected compartment `(gm = 1, gax = 0, C = 1, l = -1, d)`.
The `N` form returns a `Dendrite{Vector{FT}}` with these values per neuron and leak reversal `El`.

Precision: with `FT = Float32` the geometry is computed with the `Float32` unit constants, as before 1.10 (bit-identical). With
another `FT` (`Float64`) it is computed in `Float64` from the decimal values that the arguments denote: a length or diameter
given in the `Float32` unit system (`200um` is `0.0199999996f0`, `4um` is `0.000399999990f0`) is read as the shortest decimal
that rounds to it (`0.02`, `0.0004`), a range `(lmin, lmax)` is sampled on the decimal 1 μm grid (the same draw as in
`Float32`, so a `Float64` population built from the same seed has the same lengths), and the predefined `human_dend` and
`mouse_dend` are taken with their decimal cable constants (`Ri = 2e-7`, `Rd = 3.8907e-5` or `1.7e-5`, `Cd = 5e5` or `1e6`
in the library units); any other `Physiology` is read field by field as the shortest decimal.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
SNN.create_dendrite(200um)                          # one dendrite, human physiology
SNN.create_dendrite(5, (150um, 400um); physiology = SNN.mouse_dend)
```
"""
function create_dendrite(N::Int, l; El::Real = -70.6mV, FT::Type{<:AbstractFloat} = Float32, kwargs...)
    dendrites = Dendrite(N = N, VFT = Vector{FT})
    for i = 1:N
        dendrite = FT === Float32 ? create_dendrite(l; kwargs...) : _create_dendrite_decimal(l; kwargs...)
        dendrites.El[i] = El
        dendrites.l[i] = dendrite.l
        dendrites.d[i] = dendrite.d
        dendrites.C[i] = dendrite.C
        dendrites.gax[i] = dendrite.gax
        dendrites.gm[i] = dendrite.gm
    end
    return dendrites
end

create_dendrite(; l, kwargs...) = create_dendrite(l; kwargs...)

function create_dendrite(l; d::Real = 4um, physiology = human_dend)
    if isa(l, Tuple)
        l = rand(l[1]:1um:l[2])
    else
        l = l
    end
    l > 500um && error("Dendrite length must be less than 500um")
    @unpack Ri, Rd, Cd = physiology
    if l <= 0
        return (gm = 1.0f0, gax = 0.0f0, C = 1.0f0, l = -1, d = d)
    else
        return (
            gm = G_mem(Rd = Rd, d = d, l = l),
            gax = G_axial(Ri = Ri, d = d, l = l),
            C = C_mem(Cd = Cd, d = d, l = l),
            l = l,
            d = d,
        )
    end
end

# The decimal value that a Float32 quantity of the unit system denotes: the shortest decimal that rounds to it
# (`_decimal(200um) === 0.02`); other reals are converted.
_decimal(x::Float32) = parse(Float64, string(x))
_decimal(x::Real) = Float64(x)
# cable constants in Float64: the decimal values of the predefined physiologies (the Float32 `human_dend`, `mouse_dend` are
# computed with the Float32 unit constants: `200 * Ω * cm` is not 2e-7 in Float32), any other one field by field
function _decimal_physiology(p::Physiology)
    p === human_dend && return Physiology(2e-7, 3.8907e-5, 5e5)     # 200 Ω cm, 38907 Ω cm², 0.5 μF/cm²
    p === mouse_dend && return Physiology(2e-7, 1.7e-5, 1e6)        # 200 Ω cm, 1700 Ω cm², 1 μF/cm²
    return Physiology(_decimal(p.Ri), _decimal(p.Rd), _decimal(p.Cd))
end

# `create_dendrite(l; d, physiology)` in Float64 (the `N` form with `FT !== Float32`, see its docstring): decimal lengths and
# diameter, the same random draw of the length as the Float32 method (an index of a range of the same length).
function _create_dendrite_decimal(l; d::Real = 4um, physiology = human_dend)
    if isa(l, Tuple)
        l = rand(_decimal(l[1]):1e-4:_decimal(l[2]))
    else
        l = _decimal(l)
    end
    l > 500um && error("Dendrite length must be less than 500um")
    d = _decimal(d)
    @unpack Ri, Rd, Cd = _decimal_physiology(physiology)
    if l <= 0
        return (gm = 1.0, gax = 0.0, C = 1.0, l = -1, d = d)
    end
    return (gm = G_mem(Rd = Rd, d = d, l = l), gax = G_axial(Ri = Ri, d = d, l = l), C = C_mem(Cd = Cd, d = d, l = l), l = l, d = d)
end

# Predefined dendritic length ranges, usable as `ds` in `TripodParameter`/`BallAndStickParameter`
# (each tuple is a (min, max) range sampled by `create_dendrite`).
"""
    proximal_distal = [(150um, 400um), (150um, 400um)]

Two dendrites with lengths drawn in 150-400 μm (Tripod `ds`).
"""
proximal_distal = [(150um, 400um), (150um, 400um)]
"""
    proximal_proximal = [(150um, 300um), (150um, 300um)]

Two dendrites with lengths drawn in 150-300 μm (Tripod `ds`).
"""
proximal_proximal = [(150um, 300um), (150um, 300um)]
"""
    proximal = [(150um, 300um)]

One dendrite with length drawn in 150-300 μm (BallAndStick `ds`).
"""
proximal = [(150um, 300um)]
"""
    all_lengths = [(150um, 400um)]

One dendrite with length drawn in 150-400 μm (BallAndStick `ds`).
"""
all_lengths = [(150um, 400um)]

export create_dendrite,
    Dendrite,
    Physiology,
    proximal_distal,
    proximal_proximal,
    proximal,
    all_lengths

export G_axial, G_mem, C_mem
