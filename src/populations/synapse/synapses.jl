@doc raw"""
    AbstractSynapseParameter <: AbstractComponent

Supertype of the synapse models used by point-neuron populations (`IF`, `AdEx`, `ExtendedIF`,
...) and by the compartments of multicompartment populations (`Tripod`, `BallAndStick`).

A synapse model is a parameter struct that is stored in the population (field `synapse`, or
`soma_syn`/`dend_syn` for dendritic models). It defines how the spikes delivered by the
connections are turned into a synaptic current. Each population owns three objects created
from it:

- `receptors::NamedTuple` (built by `synaptic_receptors`): one `Vector{Float32}` of length `N`
  per receptor group, by default `(glu = ..., gaba = ...)`. Connections (`SpikingSynapse`)
  add the weight of every presynaptic spike to the entry of the postsynaptic neuron in
  `forward!`.
- `synvars::AbstractSynapseVariable` (built by `synaptic_variables`): the state variables
  (conductances, auxiliary rise variables) of the synapse model.
- the current vector `syn_curr` (or the columns of `is` in dendritic models), written by
  `synaptic_current!`.

At every `integrate!` step the population calls, in order,
`update_synapses!` (move the content of `receptors` into the synaptic state, integrate the
state over `dt`, then zero the `receptors` buffers) and `synaptic_current!` (compute the
current from the state and the membrane potential). The neuron equations use the sign
convention ``C\,dV/dt = \ldots - I_{syn}``, so a positive `syn_curr` hyperpolarises.

# Interface to implement for a new synapse model
- `synaptic_variables(synapse, N)` returning an `AbstractSynapseVariable`;
- `synaptic_receptors(synapse, N)` (optional, default `(glu = zeros(Float32, N), gaba = zeros(Float32, N))`);
- `update_synapses!(p, synapse, receptors, synvars, dt)`;
- `synaptic_current!(p, synapse, synvars, v, syn_curr)`.

# Available models
`DeltaSynapse`, `CurrentSynapse`, `SingleExpSynapse`, `DoubleExpSynapse`,
`DoubleExpCurrentSynapse`, `ReceptorSynapse`, `MultiReceptorSynapse`, `Confavreux2025Synapse`.
"""
abstract type AbstractSynapseParameter <: AbstractComponent end

"""
    AbstractSynapseVariable <: AbstractComponent

Supertype of the state containers of the synapse models (one container per population or
per compartment, created by `synaptic_variables(synapse, N)`).

# Available subtypes
- `DeltaSynapseVars` (for `DeltaSynapse`): `ge`, `gi`
- `CurrentSynapseVars` (for `CurrentSynapse`): `ge`, `gi`
- `SingleExpSynapseVars` (for `SingleExpSynapse`): `ge`, `gi`
- `DoubleExpSynapseVars` (for `DoubleExpSynapse`): `ge`, `gi`, `he`, `hi`
- `DoubleExpCurrentSynapseVars` (for `DoubleExpCurrentSynapse`): `ge`, `gi`, `he`, `hi`
- `ReceptorSynapseVars` (for `ReceptorSynapse` and `MultiReceptorSynapse`, via the abstract
  `AbstractReceptorVariable`): matrices `g`, `h` of size `N x n_receptors`
- `Confavreux2025SynapseVars` (for `Confavreux2025Synapse`): `gAMPA`, `gNMDA`, `gGABA`

All state variables are `Float32`.
"""
abstract type AbstractSynapseVariable <: AbstractComponent end

include("synapses/CurrentSynapse.jl")
include("synapses/DeltaSynapse.jl")
include("synapses/DoubleExpSynapse.jl")
include("synapses/SingleExpSynapse.jl")
include("synapses/ReceptorSynapse.jl")
# include("synapses/MultiReceptorSynapse.jl")
include("synapses/DoubleExpCurrentSynapse.jl")
include("synapses/Confraveux2025.jl")

"""
    get_synapse_symbol(synapse::AbstractSynapseParameter, sym::Symbol) -> Symbol

Map the receptor symbol given to a connection (e.g. `SpikingSynapse(pre, post, :ge)`) to the
name of the receptor buffer of the postsynaptic population. The aliases `:ge` and `:he` map to
`:glu`, and `:gi` and `:hi` map to `:gaba`; any other symbol (e.g. `:glu`, `:gaba`, or a receptor
group name of a `MultiReceptorSynapse` such as `:AMPA`) is returned unchanged.
"""
function get_synapse_symbol(synapse::T, sym::Symbol) where {T<:AbstractSynapseParameter}
    sym == :glu && return :glu
    sym == :gaba && return :gaba
    sym == :he && return :glu
    sym == :hi && return :gaba
    sym == :ge && return :glu
    sym == :gi && return :gaba
    return sym
    # error("Synapse symbol $sym not found in DoubleExpSynapse")
end

"""
    synaptic_variables(synapse::AbstractSynapseParameter, N::Int) -> AbstractSynapseVariable

Allocate the state variables (all zero, `Float32`) of the synapse model `synapse` for `N`
neurons or compartments. Every synapse model implements a method; the fallback throws an error.
"""
function synaptic_variables(synapse::AbstractSynapseParameter, N::Int)
    error("synaptic_variables not implemented for synapse type $(typeof(synapse))")
end

"""
    synaptic_receptors(synapse::AbstractSynapseParameter, N::Int) -> NamedTuple

Allocate the spike-input buffers of a population: a `NamedTuple` of `Vector{Float32}` of length
`N`, one per receptor group. The default is `(glu = zeros(Float32, N), gaba = zeros(Float32, N))`;
`MultiReceptorSynapse` creates one buffer per distinct receptor `target`. Connections add their
weights into these buffers and `update_synapses!` empties them at every step.
"""
function synaptic_receptors(synapse::AbstractSynapseParameter, N::Int)
    return (glu = zeros(Float32, N), gaba = zeros(Float32, N))
    # error("synaptic_receptors not implemented for synapse type $(typeof(synapse))")
end

"""
    update_synapses!(p, synapse, receptors::NamedTuple, synvars::AbstractSynapseVariable, dt::Float32)

Advance the synaptic state `synvars` of population `p` by one time step `dt`: add the input
accumulated since the previous step in the `receptors` buffers (sum of the weights of the
presynaptic spikes, written by the connections), integrate the synaptic kinetics, and reset
the buffers to zero. Each synapse model provides a method; see the docstring of the model for
the equations and the integration scheme.

The method with signature `(p, synapse, glu, gaba, synvars, dt)` defined in this file is a
legacy fallback that only throws an error.
"""
function update_synapses!(
    p::P,
    synapse::T,
    glu::Vector{Float32},
    gaba::Vector{Float32},
    synvars::AbstractSynapseVariable,
    dt::Float32,
) where {P<:AbstractGeneralizedIF,T<:AbstractSinExpParameter}
    error("update_synapses! not implemented for synapse type $(typeof(synapse))")
end

@doc raw"""
    synaptic_current!(p, synapse, synvars, v::AbstractVector, syn_curr::AbstractVector)
    synaptic_current!(p, synapse, synvars)

Write into `syn_curr` the synaptic current (pA) of every neuron of `p`, computed from the
synaptic state `synvars` and the membrane potential `v` (mV). Conductance-based models
return ``\sum_r g_r (V - E_r)``, current-based models return ``-(g_e - g_i)``; the neuron
equations subtract `syn_curr`. The three-argument form, used by generalized IF populations,
reads `v` and `syn_curr` from the population fields `p.v` and `p.syn_curr`.

The fallback method defined in this file (for `AbstractSinExpParameter`) only throws an error.
"""
@inline function synaptic_current!(
    p::P,
    synapse::T,
    synvars::AbstractSynapseVariable,
    v::VT1, # membrane potential
    syncurr::VT2, # synaptic current
) where {
    P<:AbstractGeneralizedIF,
    T<:AbstractSinExpParameter,
    VT1<:AbstractVector,
    VT2<:AbstractVector,
}
    error("synaptic_current! not implemented for synapse type $(typeof(synapse))")
end

# NOTE: `get_synapse_symbols` and `MultiRecetorSynapse` are exported below but not defined
# (the defined names are `get_synapse_symbol` and `MultiReceptorSynapse`).
export synaptic_current!,
    update_synapses!,
    synaptic_variables,
    synaptic_target,
    get_synapse_symbols,
    CurrentSynapse,
    DeltaSynapse,
    DoubleExpSynapse,
    SingleExpSynapse,
    MultiRecetorSynapse,
    ReceptorSynapse,
    AbstractSynapseParameter,
    AbstractSynapseVariable
