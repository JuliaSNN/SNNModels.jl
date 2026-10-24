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
`mouse_dend` (`Ri = 200 Ω cm`, `Rd = 1700 Ω cm²`, `Cd = 1 μF/cm²`). Sources, as cited in the code of the
published Tripod model (`aquaresima/tripod_neuron`, `model/parameters.jl`): human, Eyal G. et al. (2016),
Unique membrane properties and enhanced signal processing in human neocortical neurons, eLife 5:e16553,
doi:10.7554/eLife.16553; mouse, Dasika V. K., White J. A., Colburn H. S. (2007), Simple models show the general
advantages of dendrites in coincidence detection, J. Neurophysiol. 97(5):3449-3459, https://doi.org/10/dc8tr4
(DOI as given in that code).

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

Human cortical dendrite: `Ri = 200 Ω cm`, `Rd = 38907 Ω cm²`, `Cd = 0.5 μF/cm²` (Eyal et al. 2016, eLife
5:e16553). Default `physiology` of `TripodParameter`, `BallAndStickParameter` and `create_dendrite`.
"""
human_dend = Physiology(200 * Ω * cm, 38907 * Ω * cm^2, 0.5μF / cm^2|>Float32)
"""
    mouse_dend :: Physiology{Float32}

Mouse cortical dendrite: `Ri = 200 Ω cm`, `Rd = 1700 Ω cm²`, `Cd = 1 μF/cm²` (Dasika et al. 2007, J. Neurophysiol.
97(5):3449-3459).
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

# Fields (vectors of type `VFT` and length `N`, default `zeros(N)`)
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
first form). Lengths above `1000um` raise an error: this is a guard against a length given without its
unit (`400` is 400 cm), not a physical limit; up to SNNModels 2.0 the bound was `500um`, which excluded the
550 um dendrites of the Tripod paper (Fig. 3). For `l > 0` it returns
`(gm = G_mem(Rd, d, l), gax = G_axial(Ri, d, l), C = C_mem(Cd, d, l), l, d)`; for `l <= 0` it
returns a disconnected compartment `(gm = 1, gax = 0, C = 1, l = -1, d)`.
The `N` form returns a `Dendrite{Vector{FT}}` with these values per neuron and leak reversal `El`.

Precision: with `FT = Float32` the geometry is computed with the `Float32` unit constants. With `FT = Float64` it is
computed from the decimal values the arguments denote (`5um`, stored as `0.00049999997f0`, is read as `0.0005`; the cable
constants likewise in their units). A range is sampled with the same RNG draws as in `Float32`, so the same seed gives the
same lengths in both precisions.

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

# Sanity bound on a dendrite length: catches a length written without its unit (`400` is 400 cm). The bound was 500um before 2.1.0.
const _MAX_DENDRITE_LENGTH = 1000um
_check_length(l) = l > _MAX_DENDRITE_LENGTH &&
    error("Dendrite length must be at most 1000um (got $(l) cm); write lengths with their unit, e.g. 400um")

function create_dendrite(l; d::Real = 4um, physiology = human_dend)
    if isa(l, Tuple)
        l = rand(l[1]:1um:l[2])
    else
        l = l
    end
    _check_length(l)
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

# The decimal value a Float32 quantity denotes. `_decimal(x, u)`: `x / u` (`u` a power of ten, e.g. `um`) snapped to a decimal
# with at most 6 fractional digits within Float32 rounding, times `u` (`5um` -> 0.0005). Otherwise, and for `_decimal(x)`,
# the shortest decimal that rounds to `x`.
_decimal(x::Float32) = parse(Float64, string(x))
_decimal(x::Real) = Float64(x)
function _decimal(x::Float32, u::Float32)
    ud = _decimal(u)
    p = round(Int, log10(ud))
    (ud === _exp10(1.0, p) && isfinite(x)) || return _decimal(x)
    n = Float64(x) / Float64(u)
    for m = 0:6
        k = round(n * 10.0^m)
        abs(n * 10.0^m - k) <= 4 * eps(Float32) * abs(n * 10.0^m) && return _exp10(k, p - m)
    end
    return _decimal(x)
end
_decimal(x::Real, u::Float32) = Float64(x)
_exp10(k::Float64, e::Int) = e >= 0 ? k * 10.0^e : k / 10.0^(-e)    # exact powers of ten up to 1e22: one rounding
# cable constants in Float64, each read in its unit (`Ri` in Ω cm, `Rd` in Ω cm², `Cd` in μF/cm²)
_decimal_physiology(p::Physiology) =
    Physiology(_decimal(p.Ri, Float32(Ω * cm)), _decimal(p.Rd, Float32(Ω * cm^2)), _decimal(p.Cd, Float32(μF / cm^2)))

# `create_dendrite(l; d, physiology)` in Float64: a range is sampled from the same Float32 range as the Float32 method (same
# RNG draws), then lengths and diameter are read as decimals of `um`.
function _create_dendrite_decimal(l; d::Real = 4um, physiology = human_dend)
    l = isa(l, Tuple) ? rand(l[1]:1um:l[2]) : l
    _check_length(l)    # checked on the given value, as in Float32
    l = _decimal(l, um)
    d = _decimal(d, um)
    @unpack Ri, Rd, Cd = _decimal_physiology(physiology)
    if l <= 0
        return (gm = 1.0, gax = 0.0, C = 1.0, l = -1, d = d)
    end
    return (gm = G_mem(Rd = Rd, d = d, l = l), gax = G_axial(Ri = Ri, d = d, l = l), C = C_mem(Cd = Cd, d = d, l = l), l = l, d = d)
end

# Predefined dendritic lengths, usable as `ds` in `TripodParameter`/`BallAndStickParameter`.
# A tuple is a (min, max) range sampled by `create_dendrite`; a length is used as given.
"""
    distal_distal = [400um, 400um]

Tripod `ds` of the paper configuration "distal-distal": two dendrites of 400 um (`d1`, `d2`). Quaresima et al. (2023),
J. Physiol.; `parameters.jl` of `aquaresima/tripod_neuron` (`H_distal_distal`).
"""
distal_distal = [400um, 400um]
"""
    proximal_proximal = [150um, 150um]

Tripod `ds` of the paper configuration "proximal-proximal": two dendrites of 150 um (`d1`, `d2`). Before SNNModels 2.1
`proximal_proximal` was the random preset now named `random_proximal_proximal`.
"""
proximal_proximal = [150um, 150um]
"""
    proximal_distal = [150um, 400um]

Tripod `ds` with `d1` proximal (150 um) and `d2` distal (400 um). Before SNNModels 2.1 `proximal_distal` was the random
preset now named `random_proximal_distal`. The published Tripod code names its default configuration
`distal_proximal` (`d1` 400 um, `d2` 150 um, `H_distal_proximal`), which is `distal_proximal` here.
"""
proximal_distal = [150um, 400um]
"""
    distal_proximal = [400um, 150um]

Tripod `ds` with `d1` distal (400 um) and `d2` proximal (150 um): the default configuration of the published Tripod code
(`H_distal_proximal`, `M_distal_proximal`), whose protocols stimulate `d1`.
"""
distal_proximal = [400um, 150um]
"""
    soma_only = [0um, 0um]

Tripod `ds` with both dendrites disconnected (`create_dendrite` returns `gax = 0` for `l <= 0`): a point-neuron control.
The original `soma_only` option of the published code also routes the dendritic inputs to the soma and sets the dendritic
potentials equal to the somatic one; here the dendrites are inert and the inputs have to be connected to the soma (`:s`).
"""
soma_only = [0um, 0um]
"""
    random_proximal_distal = [(150um, 400um), (150um, 400um)]

Two dendrites with lengths drawn in 150-400 μm (Tripod `ds`). Named `proximal_distal` up to SNNModels 2.0.
"""
random_proximal_distal = [(150um, 400um), (150um, 400um)]
"""
    random_proximal_proximal = [(150um, 300um), (150um, 300um)]

Two dendrites with lengths drawn in 150-300 μm (Tripod `ds`). Named `proximal_proximal` up to SNNModels 2.0.
"""
random_proximal_proximal = [(150um, 300um), (150um, 300um)]
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
    distal_distal,
    proximal_proximal,
    proximal_distal,
    distal_proximal,
    soma_only,
    random_proximal_distal,
    random_proximal_proximal,
    proximal,
    all_lengths

export G_axial, G_mem, C_mem
