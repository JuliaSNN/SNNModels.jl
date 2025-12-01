
DendLength = Union{Float32,Tuple}
abstract type AbstractDendriticTree end
struct TripodNeuron <: AbstractDendriticTree end
struct BallAndStickNeuron <: AbstractDendriticTree end
struct Multipod <: AbstractDendriticTree end

"""
    DendNeuronParameter{DT,GT,PT,NT} <: AbstractGeneralizedIFParameter
A parameter struct for multicompartment dendritic neuron models.
# Fields
- `ds::DT`: Vector of tuples defining the lengths of dendritic segments (default:
    [(200um, 400um), (200um, 400um)] for TripodNeuron)
- `physiology::PT`: Physiology parameters for the dendritic model (default: `human_dend`)
- `geometry::GT`: Vector of pairs defining the geometry between soma and dendrites
    (default: [(:s=>:d1), (:s=>:d2)] for TripodNeuron)
- `type::NT`: Type of dendritic tree (default: `TripodNeuron`)
# Type Parameters
- `DT`: Type of dendritic lengths (default: `Vector{DendLength}`)
- `GT`: Type of geometry (default: `Vector{Pair{Symbol,Symbol}}`)
- `PT`: Physiology type (default: `Physiology{Float32}`)
- `NT`: Dendritic tree type (default: `AbstractDendriticTree`) 
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

function TripodParameter(;
    ds = [(200um, 400um), (200um, 400um)],
    physiology = human_dend,
    geometry = [(:s=>:d1), (:s=>:d2)],
)
    return DendNeuronParameter(ds = ds, physiology = physiology, geometry = geometry)
end

function BallAndStickParameter(;
    ds = [(150um, 400um)],
    physiology = human_dend,
    geometry = [(:s=>:d)],
)
    return DendNeuronParameter(ds = ds, physiology = physiology, geometry = geometry)
end

function Population(param::T; kwargs...) where {T<:DendNeuronParameter}
    if param.type isa TripodNeuron
        return Tripod(; param, kwargs...)
    elseif param.type isa BallAndStickNeuron
        return BallAndStick(; param, kwargs...)
    else
        error("Dendritic segments must be either 1 (BallAndStick) or 2 (Tripod).")
    end
end

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
