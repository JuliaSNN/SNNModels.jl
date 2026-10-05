
DendLength = Union{Float32,Tuple}
abstract type AbstractDendriticTree end
struct TripodNeuron <: AbstractDendriticTree end
struct BallAndStickNeuron <: AbstractDendriticTree end
struct Multipod <: AbstractDendriticTree end

"""
    DendNeuronParameter(; ds = [(200um, 400um), (200um, 400um)], physiology = human_dend,
                          geometry = [(:s=>:d1), (:s=>:d2)], type = <from length(ds)>)

Parameter struct of the multicompartment dendritic populations (`Tripod`, `BallAndStick`). It
carries the morphology only; the somatic parameters are in the population field `adex`
(an `AdExParameter`) and the synapses in `soma_syn`/`dend_syn`.

`Population(param::DendNeuronParameter; kwargs...)` builds a `Tripod` when `type` is
`TripodNeuron()` and a `BallAndStick` when it is `BallAndStickNeuron()`.

# Fields
- `ds::DT = [(200um, 400um), (200um, 400um)]`: one entry per dendrite, either a length (cm)
  or a `(min, max)` range from which `create_dendrite` draws the length of each neuron.
- `physiology::PT = human_dend`: cable properties (`Physiology`).
- `geometry::GT = [(:s=>:d1), (:s=>:d2)]`: connectivity of the compartments. Stored for
  bookkeeping; `Tripod` and `BallAndStick` hard-code soma-dendrite coupling.
- `type::NT`: `BallAndStickNeuron()` if `length(ds) == 1`, `TripodNeuron()` if
  `length(ds) == 2`; other lengths raise an error.

Type parameters: `DT = Vector{DendLength}` (`DendLength = Union{Float32,Tuple}`),
`GT = Vector{Pair{Symbol,Symbol}}`, `PT = Physiology{Float32}`, `NT<:AbstractDendriticTree`.

Note that `create_dendrite` is called with its default diameter `d = 4um`; the diameter cannot
be set through `DendNeuronParameter`.

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
    DT=Vector{DendLength},
    GT=Vector{Pair{Symbol,Symbol}},
    PT = Physiology{Float32},
    NT<:AbstractDendriticTree,
} <: AbstractGeneralizedIFParameter

    ## Dend parameters
    ds::DT = [(200um, 400um), (200um, 400um)] ## Dendritic segment lengths
    physiology::PT = human_dend
    geometry::GT = [(:s=>:d1), (:s=>:d2)]  ## Geometry between soma and dendrites
    type::NT = begin
        if length(ds) == 1
            BallAndStickNeuron()
        elseif length(ds) == 2
            TripodNeuron()
        else
            error(
                "MulticompartmentNeuron not implemented yet. Dendritic segments must be either 1 (BallAndStick) or 2 (Tripod).",
            )
        end
    end
end

"""
    TripodParameter(; ds = [(200um, 400um), (200um, 400um)], physiology = human_dend,
                      geometry = [(:s=>:d1), (:s=>:d2)]) -> DendNeuronParameter

Morphology of a `Tripod` neuron: soma plus two dendrites with lengths `ds[1]`, `ds[2]`.
`ds` must have two entries.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
T = SNN.Tripod(N = 10, param = SNN.TripodParameter(ds = SNN.proximal_distal))
```
"""
function TripodParameter(;
    ds = [(200um, 400um), (200um, 400um)],
    physiology = human_dend,
    geometry = [(:s=>:d1), (:s=>:d2)],
)
    return DendNeuronParameter(ds = ds, physiology = physiology, geometry = geometry)
end

"""
    BallAndStickParameter(; ds = [(150um, 400um)], physiology = human_dend,
                            geometry = [(:s=>:d)]) -> DendNeuronParameter

Morphology of a `BallAndStick` neuron: soma plus one dendrite of length `ds[1]`.
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
    geometry = [(:s=>:d)],
)
    return DendNeuronParameter(ds = ds, physiology = physiology, geometry = geometry)
end

"""
    Population(param::DendNeuronParameter; kwargs...)

Build a `Tripod` (two dendrites) or a `BallAndStick` (one dendrite) population according to
`param.type`; `kwargs` are passed to the population constructor.
"""
function Population(param::T; kwargs...) where {T<:DendNeuronParameter}
    if param.type isa TripodNeuron
        return Tripod(; param, kwargs...)
    elseif param.type isa BallAndStickNeuron
        return BallAndStick(; param, kwargs...)
    else
        error("Dendritic segments must be either 1 (BallAndStick) or 2 (Tripod).")
    end
end

"""
    synaptic_target(targets::Dict, post::AbstractDendriteIF, sym::Symbol, target::Symbol)

Target the receptor buffer `post.receptors_<target>.<sym>` of compartment `target` (`:s`,
`:d1`, `:d2` for `Tripod`; `:s`, `:d` for `BallAndStick`); `sym` is mapped with
`get_synapse_symbol` (`:ge`/`:he` -> `:glu`, `:gi`/`:hi` -> `:gaba`). Returns the buffer and
`post.v_<target>`. Records `:sym => "<sym>_<target>"` in `targets`.
"""
function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target::Symbol,
) where {T<:AbstractDendriteIF}
    receps = Symbol("receptors_$target")
    v = Symbol("v_$target")
    sym = get_synapse_symbol(post.soma_syn, sym)
    g = getfield(getfield(post, receps), sym)
    hasfield(typeof(post), v) && (v_post = getfield(post, v))
    push!(targets, :sym => "$(sym)_$target")
    push!(targets, :g => post.id)
    return g, v_post
end


export BallAndStickParameter, TripodParameter, DendNeuronParameter
