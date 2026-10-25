"""
    AbstractPopulationParameter <: AbstractParameter

Supertype of the parameter types of populations (the `param` field of an `AbstractPopulation`).

The simulation loop dispatches on the pair `(population, population.param)`:
`integrate!(p, p.param, dt)` under `sim!` and `train!`, and in addition
`update_traces!(p, p.param, dt, T)` and `plasticity!(p, p.param, dt, T)` under `train!`.
For subtypes of `AbstractPopulationParameter` the fallbacks of these three functions do nothing,
so a new model only needs to implement `integrate!`.
"""
abstract type AbstractPopulationParameter <: AbstractParameter end

"""
    integrate!(p::AbstractPopulation, param::AbstractPopulationParameter, dt)

Advance the state of population `p` by one time step `dt` (ms). Every population model
implements a method of `integrate!` for its own parameter type; it is called once per step by
`sim!` and `train!`, after the stimuli and before the connections. `dt` is converted to the precision of the
population (`float_type(p)`). The fallback does nothing for a `Float32` step and throws an `ArgumentError` for
another precision (a model without a `Float64` method).

See the documentation of each population type for the equations and the integration scheme.
"""
integrate!(p::AbstractPopulation, param::AbstractPopulationParameter, dt::AbstractFloat) =
    dt isa Float32 ? nothing : throw(ArgumentError(
        "population $(nameof(typeof(p))) has no integrate! method for steps of type $(typeof(dt)) (its model is not implemented in $(typeof(dt)); " *
        "populations with Float64 state: $(_FLOAT64_POPULATIONS))"))

# populations that have a Float64 model (`VFT = Vector{Float64}`); the error messages list them
const _FLOAT64_POPULATIONS = "Identity, IF, AdEx, ExtendedIF, HH, IZ, MorrisLecar, Rate, WilsonCowan, Tripod, BallAndStick, " *
                             "Poisson, VariablePoisson, InhomogeneousPoisson"

# Copy of a parametric parameter struct with its floating-point fields and type parameters converted to `T` (the methods of
# `convert_precision` for synapse models and spike parameters); `x` itself when it is already `T` or has no float type parameter.
function _convert_float_struct(x::S, ::Type{T}) where {S,T<:AbstractFloat}
    ps = S.parameters
    (isempty(ps) || !any(_is_float_param, ps) || _float_type(S) === T) && return x
    newps = map(p -> _convert_param(p, T), ps)
    vals = map(f -> _convert_value(getfield(x, f), T), fieldnames(S))
    return Base.typename(S).wrapper{newps...}(vals...)
end

# float type of the state of a population type `T`: the element type of its field `field`, Float32 when not parameterised
_state_float(::Type{T}, field::Symbol) where {T} = (V = fieldtype(T, field); V isa DataType && V <: AbstractArray ? eltype(V) : Float32)
# One precision per population (`snn_kw_finalize`): each parameter object (`what => obj`) must have the float type `FT` of the
# state; objects without a floating-point type parameter (`DeltaSynapse()`, `RateParameter()`) pass.
function _check_precision(name, FT, objs::Pair...)
    for (what, obj) in objs
        T = _float_type(typeof(obj))
        (T === nothing || T === FT) || throw(ArgumentError(
            "$name mixes precisions: the state is $FT but `$what` is a $(typeof(obj)) ($T). " *
            "Build all with the same floating-point type (`FT = $FT`, or `convert_precision(x, $FT)`); the defaults follow `VFT`, " *
            "so `$name(N = ..., VFT = Vector{$FT})` alone gives a $FT population (see `float_type`)."))
    end
    return nothing
end
plasticity!(
    p::AbstractPopulation,
    param::AbstractPopulationParameter,
    dt::AbstractFloat,
    T::Time,
) = nothing

update_traces!(
    p::AbstractPopulation,
    param::AbstractPopulationParameter,
    dt::AbstractFloat,
    T::Time,
) = nothing

## Spikes
"""
    AbstractSpikeParameter

Supertype of the spike parameters stored in the `spike` field of generalized integrate-and-fire
populations. The only concrete subtype is `PostSpike`.
"""
abstract type AbstractSpikeParameter end
include("spike/postspike.jl")

## Spike index lists, shared by populations and connections
include("spike_index.jl")

## Neurons
include("poisson.jl")
include("inhomogeneous_poisson.jl")
include("iz.jl")
include("hh.jl")
include("morrislecar.jl")
include("rate.jl")
include("identity.jl")

## gIF and synapses
"""
    AbstractGeneralizedIFParameter <: AbstractPopulationParameter

Supertype of the neuron parameters of generalized integrate-and-fire models
(`IFParameter`, `AdExParameter`, `ExtendedIFParameter`, `HetRecParameter`, the multicompartment
parameters). `make_heterogeneous` accepts any subtype.
"""
abstract type AbstractGeneralizedIFParameter <: AbstractPopulationParameter end

"""
    AbstractGeneralizedIF <: AbstractPopulation

Supertype of generalized integrate-and-fire populations. A concrete subtype combines three
parameter objects:
- `param`: the neuron model (an `AbstractGeneralizedIFParameter`, e.g. `IFParameter`),
- `synapse`: the synapse model (an `AbstractSynapseParameter`, e.g. `DoubleExpSynapse`),
- `spike`: the spike parameters (`PostSpike`),

and holds the state vectors `v`, `fire`, `tabs`, `I`, `syn_curr`, the synaptic state `synvars` and
the `receptors` NamedTuple into which presynaptic spikes are written.

The generic step (`integrate!` in `generalized_if/gif.jl`) is
1. `update_synapses!(p, p.synapse, p.receptors, p.synvars, dt)`: add the spikes accumulated in
   `receptors` to the synaptic variables and integrate them;
2. `synaptic_current!(p, p.synapse, p.synvars)`: compute `syn_curr` from the synaptic variables
   and the membrane potential `v`;
3. `update_neuron!(p, p.param, dt)`: integrate the membrane equation, detect spikes, reset.

`ExtendedIF` and `HetRec` define their own `integrate!` instead.
"""
abstract type AbstractGeneralizedIF <: AbstractPopulation end

## Synapses
include("synapse/receptors.jl")
include("synapse/synapses.jl")
include("synapse/receptor_types.jl")
include("synapse/synaptic_targets.jl")

include("generalized_if/gif.jl")
include("generalized_if/if.jl")
include("generalized_if/adex.jl")
include("generalized_if/membrane.jl")
include("generalized_if/if_extended.jl")

## Heterogeneous recurrent
include("hetrec.jl")

## Rate / mean-field
include("wilsoncowan.jl")

## Multicompartment
"""
    AbstractDendriteIF <: AbstractGeneralizedIF

Supertype of the multicompartment populations (an adaptive exponential soma coupled to
passive dendritic compartments): `Tripod` and `BallAndStick`. Connections to these
populations select the compartment with the `target` argument of `synaptic_target`
(e.g. `:s` for the soma, `:d` or `:d1`/`:d2` for the dendrites).
"""
abstract type AbstractDendriteIF <: AbstractGeneralizedIF end
include("multicompartment/dendrite.jl")
include("multicompartment/dendneuron_parameter.jl")
include("multicompartment/tripod.jl")
include("multicompartment/tripod_paper.jl")
include("multicompartment/ballandstick.jl")

## Spike lists owned by the integrate-and-fire family (see `spike_index`)
# A population type not listed here falls back to a new list per connection (correct, slower).
const _OWNED_LIST_POPULATIONS = Union{IF,AdEx,ExtendedIF,HetRec,Tripod,BallAndStick,HH,IZ,MorrisLecar}

spike_index(p::_OWNED_LIST_POPULATIONS) = getfield(p, :spikes)

function _publish_spikes!(p::_OWNED_LIST_POPULATIONS)
    s = getfield(p, :spikes)
    s.active = true
    s.valid = false
    return nothing
end

function _retire_spikes!(p::_OWNED_LIST_POPULATIONS)
    s = getfield(p, :spikes)
    s.active = false
    s.valid = false
    return nothing
end

## Population constructor
"""
    Population(param; kwargs...)
    Population(; param, kwargs...)

Construct the population type that corresponds to the parameter object `param`.
Methods exist for `IFParameter` (returns `IF`; requires `synapse`, `spike` and `N`),
`AdExParameter` (returns `AdEx`; requires `synapse` and `N`, `spike = PostSpike()`),
`PoissonParameter` subtypes (returns `Poisson`), `InhomogeneousPoissonParam`
(returns `InhomogeneousPoisson`), `IdentityParam` (returns `Identity`) and `HetRecParameter`
(returns `HetRec`); the multicompartment models add their own methods. Remaining keyword
arguments are forwarded to the population constructor.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Population(SNN.AdExParameter(); synapse = SNN.DoubleExpSynapse(), N = 50, name = "E")
P = SNN.Population(SNN.PoissonParameter(10Hz); N = 50)
```
"""
Population(; param, kwargs...) = Population(param; kwargs...)

## Heterogeneous populations
"""
    make_heterogeneous(param::AbstractGeneralizedIFParameter, N::Int; kwargs...)

Return a copy of `param` in which every field is a `Vector{Float32}` of length `N`.
For each field name passed as keyword argument, the values are drawn with
`rand(kwargs[field], N)` (the argument can be a `Distribution` or any collection accepted by
`rand`); the other fields are filled with the value they have in `param`. The result is built
by calling the constructor of the same parameter type with `FT = Vector{Float32}`.

Only models whose `update_neuron!` indexes the parameters per neuron accept the result: in
SNNModels 1.8.4 this is `AdEx` (method for `AdExParameter{Vector{Float32}}`). `IF` does not
support vector-valued parameters.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
param = SNN.make_heterogeneous(SNN.AdExParameter(), 50; El = SNN.SNNModels.Uniform(-72mV, -68mV))
E = SNN.AdEx(N = 50, param = param)
SNN.sim!([E]; duration = 100ms)
```
"""
function make_heterogeneous(
    param::T,
    N::Int;
    kwargs...,
) where {T<:AbstractGeneralizedIFParameter}
    # ξ_het = ones(Float32, N)
    _type = typeof(param)
    het_dict = Dict{Symbol,Vector{Float32}}()
    for fields in fieldnames(_type)
        if haskey(kwargs, fields)
            het_dict[fields] = rand(kwargs[fields], N)
        else
            het_dict[fields] = fill(getfield(param, fields), N)
        end
    end
    # Sampled membrane parameters follow the pair rule (see `membrane_update`): the others are
    # recomputed per neuron instead of being copied from `param`.
    if param isa MembraneParameter
        sampled = Tuple(k for k in keys(kwargs) if k in MEMBRANE_FIELDS)
        if !isempty(sampled)
            current = (; (k => het_dict[k] for k in MEMBRANE_FIELDS)...)
            new = NamedTuple{sampled}(Tuple(het_dict[k] for k in sampled))
            for (k, v) in pairs(membrane_update(current, new))
                het_dict[k] = v
            end
        end
    end
    return getfield(SNNModels, nameof(_type))(; het_dict..., FT = Vector{Float32})
end


export AbstractDendriteIF,
    AbstractGeneralizedIF,
    AbstractGeneralizedIFParameter,
    Population,
    integrate!,
    plasticity!,
    make_heterogeneous
