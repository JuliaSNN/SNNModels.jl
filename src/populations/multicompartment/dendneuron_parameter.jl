
# One entry of `ds`: a length (cm), a `(min, max)` range, or `length => diameter` (per-dendrite diameter, used by `Multipod`).
DendLength = Union{Float32,Tuple,Pair}
abstract type AbstractDendriticTree end
struct TripodNeuron <: AbstractDendriticTree end
struct BallAndStickNeuron <: AbstractDendriticTree end
struct MultipodNeuron <: AbstractDendriticTree end

# Largest number of dendrites of a `Multipod` (one concrete type per number, generated at load time).
const MULTIPOD_MAX_DENDRITES = 8

"""
    DendNeuronParameter(; ds = [(200um, 400um), (200um, 400um)], physiology = human_dend, d = 4um, El_dend = nothing,
                          geometry = [(:s=>:d1), (:s=>:d2)], type = <from length(ds)>)

Parameter struct of the multicompartment dendritic populations (`Tripod`, `BallAndStick`, `Multipod`). It
carries the morphology only; the somatic parameters are in the population field `adex`
(an `AdExParameter`) and the synapses in `soma_syn`/`dend_syn`.

`Population(param::DendNeuronParameter; kwargs...)` builds a `Tripod` when `type` is
`TripodNeuron()`, a `BallAndStick` when it is `BallAndStickNeuron()` and a `Multipod` when it is `MultipodNeuron()`.

# Fields
- `ds::DT = [(200um, 400um), (200um, 400um)]`: one entry per dendrite, either a length (cm, any `Real`, stored as `Float32`)
  or a `(min, max)` range from which `create_dendrite` draws the length of each neuron. `Multipod` also accepts
  `length => diameter` (the length a number or a range), the diameter of that dendrite replacing `d`.
- `physiology::PT = human_dend`: cable properties (`Physiology`: `Ri`, `Rd`, `Cd`), used by `Tripod` and `BallAndStick` to build their dendrites.
- `d::Float32 = 4um`: dendrite diameter (cm), common to all dendrites of the neuron. It enters the cable formulas of
  `create_dendrite`: `g_ax ∝ d²`, `g_m ∝ d` and `C ∝ d` at fixed length. The default is the 4 um of the published model.
- `El_dend::Union{Nothing,Float32} = nothing`: leak reversal potential of the dendrites (mV). `nothing` (default): the
  dendrites use the somatic `adex.El`. A number: `create_dendrite(...; El = El_dend)`, so the dendritic leak stays fixed
  whatever the somatic `El` (the original Tripod fixes it at -70.6 mV). It is read once, when the `Tripod`/`BallAndStick`
  is constructed (`Dendrite.El`); with `nothing`, changing `adex.El` afterwards does not move the dendritic leak either
  (it keeps the value `adex.El` had at construction), see `Tripod`.
- `geometry::GT`: connectivity of the compartments, `[(:s=>:d1), (:s=>:d2)]` for one or two dendrites and the star
  `[(:s=>:d1), ..., (:s=>:dN)]` for more. Stored for bookkeeping; the populations hard-code the soma-dendrite coupling
  (all dendrites attached to the soma).
- `type::NT`: `BallAndStickNeuron()` if `length(ds) == 1`, `TripodNeuron()` if `length(ds) == 2`, `MultipodNeuron()` if
  `3 <= length(ds) <= $(MULTIPOD_MAX_DENDRITES)` (`MultipodParameter` sets `MultipodNeuron()` for any number of dendrites);
  other lengths raise an error.

Type parameters: `ET = Union{Nothing,Float32}`, `DT = Vector{DendLength}` (`DendLength = Union{Float32,Tuple}`),
`GT = Vector{Pair{Symbol,Symbol}}`, `PT = Physiology{Float32}`, `NT<:AbstractDendriticTree`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
param = SNN.DendNeuronParameter(ds = [(150um, 300um), (300um, 400um)], physiology = SNN.mouse_dend)
T = SNN.Population(param; N = 10)   # a Tripod
```
"""
DendNeuronParameter
@snn_kw struct DendNeuronParameter{
    ET=Union{Nothing,Float32},
    DT=Vector{DendLength},
    GT=Vector{Pair{Symbol,Symbol}},
    PT = Physiology{Float32},
    NT<:AbstractDendriticTree,
} <: AbstractGeneralizedIFParameter

    ## Dend parameters
    ds::DT = [(200um, 400um), (200um, 400um)] ## Dendritic segment lengths
    physiology::PT = human_dend
    d::Float32 = 4um  ## Dendrite diameter (cm)
    El_dend::ET = nothing  ## Dendritic leak reversal (mV); nothing: somatic adex.El
    geometry::GT = length(ds) <= 2 ? [(:s=>:d1), (:s=>:d2)] : _star_geometry(length(ds))  ## Geometry between soma and dendrites
    type::NT = begin
        if length(ds) == 1
            BallAndStickNeuron()
        elseif length(ds) == 2
            TripodNeuron()
        elseif 3 <= length(ds) <= MULTIPOD_MAX_DENDRITES
            MultipodNeuron()
        else
            error(
                "Dendritic segments must be 1 (BallAndStick), 2 (Tripod) or 3 to $(MULTIPOD_MAX_DENDRITES) (Multipod); got $(length(ds)).",
            )
        end
    end
end

# `ds` accepts any `Real` length (Float64, Int, ...) besides `(min, max)` tuples; scalar lengths are stored as `Float32`, and so are
# both sides of a `length => diameter` pair (its length may be a range).
_dend_length(l::Tuple) = l
_dend_length(l::Pair) = _dend_length(l.first) => Float32(l.second)
_dend_length(l) = Float32(l)
snn_kw_finalize(::Type{<:DendNeuronParameter}, f::NamedTuple) = (; f..., ds = DendLength[_dend_length(l) for l in f.ds])

# Star topology of `n` dendrites: every dendrite attached to the soma.
_star_geometry(n::Integer) = Pair{Symbol,Symbol}[(:s => Symbol(:d, k)) for k in 1:n]

# Dendritic leak reversal used at construction: `param.El_dend`, or the somatic `adex.El` when it is `nothing`.
_dend_El(param::DendNeuronParameter, adex) = isnothing(param.El_dend) ? adex.El : param.El_dend

"""
    TripodParameter(; ds = [(200um, 400um), (200um, 400um)], physiology = human_dend, d = 4um,
                      El_dend = nothing, geometry = [(:s=>:d1), (:s=>:d2)]) -> DendNeuronParameter

Morphology of a `Tripod` neuron: soma plus two dendrites with lengths `ds[1]`, `ds[2]` and diameter `d`.
`El_dend` is the dendritic leak reversal (mV, `nothing`: the somatic `adex.El`).
`ds` must have two entries; the fixed paper configurations `distal_distal`, `proximal_proximal`, `proximal_distal`,
`distal_proximal` and the random presets `random_proximal_distal`, `random_proximal_proximal` can be given as `ds`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
T = SNN.Tripod(N = 10, param = SNN.TripodParameter(ds = SNN.distal_proximal, d = 3um))
```
"""
function TripodParameter(;
    ds = [(200um, 400um), (200um, 400um)],
    physiology = human_dend,
    d = 4um,
    El_dend = nothing,
    geometry = [(:s=>:d1), (:s=>:d2)],
)
    return DendNeuronParameter(ds = ds, physiology = physiology, d = d, El_dend = El_dend, geometry = geometry)
end

"""
    BallAndStickParameter(; ds = [(150um, 400um)], physiology = human_dend, d = 4um,
                            El_dend = nothing, geometry = [(:s=>:d)]) -> DendNeuronParameter

Morphology of a `BallAndStick` neuron: soma plus one dendrite of length `ds[1]` and diameter `d`.
`El_dend` is the dendritic leak reversal (mV, `nothing`: the somatic `adex.El`).
`ds` must have one entry.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
B = SNN.BallAndStick(N = 10, param = SNN.BallAndStickParameter(ds = [300um]))
```
"""
function BallAndStickParameter(;
    ds = [(150um, 400um)],
    physiology = human_dend,
    d = 4um,
    El_dend = nothing,
    geometry = [(:s=>:d)],
)
    return DendNeuronParameter(ds = ds, physiology = physiology, d = d, El_dend = El_dend, geometry = geometry)
end

"""
    Population(param::DendNeuronParameter; kwargs...)

Build a `Tripod` (two dendrites), a `BallAndStick` (one dendrite) or a `Multipod` population according to
`param.type`; `kwargs` are passed to the population constructor.
"""
function Population(param::T; kwargs...) where {T<:DendNeuronParameter}
    if param.type isa TripodNeuron
        return Tripod(; param, kwargs...)
    elseif param.type isa BallAndStickNeuron
        return BallAndStick(; param, kwargs...)
    elseif param.type isa MultipodNeuron
        return Multipod(; param, kwargs...)
    else
        error("Dendritic segments must be 1 (BallAndStick), 2 (Tripod) or 1 to $(MULTIPOD_MAX_DENDRITES) (Multipod).")
    end
end

"""
    synaptic_target(targets::Dict, post::AbstractDendriteIF, sym::Symbol, target::Symbol)

Target the receptor buffer `post.receptors_<target>.<sym>` of compartment `target` (`:s`,
`:d1`, `:d2` for `Tripod`; `:s`, `:d` for `BallAndStick`; `:s`, `:d1`, ..., `:dN` for `Multipod`); `sym` is mapped with
`get_synapse_symbol` (`:ge`/`:he` -> `:glu`, `:gi`/`:hi` -> `:gaba`). Returns the buffer and
`post.v_<target>`. Records `:sym => "<sym>_<target>"` in `targets`. An unknown `target`
raises an `ArgumentError`.
"""
function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target::Symbol,
) where {T<:AbstractDendriteIF}
    receps = Symbol("receptors_$target")
    v = Symbol("v_$target")
    if !(hasfield(typeof(post), receps) && hasfield(typeof(post), v))
        comps = [Symbol(string(f)[3:end]) for f in fieldnames(typeof(post)) if startswith(string(f), "v_")]
        throw(ArgumentError("unknown compartment `$target` for $(nameof(typeof(post))); valid targets: $(comps)"))
    end
    sym = get_synapse_symbol(post.soma_syn, sym)
    g = getfield(getfield(post, receps), sym)
    v_post = getfield(post, v)
    push!(targets, :sym => "$(sym)_$target")
    push!(targets, :g => post.id)
    return g, v_post
end


export BallAndStickParameter, TripodParameter, DendNeuronParameter
