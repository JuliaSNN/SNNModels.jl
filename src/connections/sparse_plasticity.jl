#=
Plasticity of sparse synapses (AbstractSparseSynapse, e.g. SpikingSynapse)
==========================================================================

A sparse synapse carries two independent rules, each made of a parameter object and a
variable (state) object:

    LTPParam :: LTPParameter   with  LTPVars :: PlasticityVariables   (long-term, acts on W)
    STPParam :: STPParameter   with  STPVars :: PlasticityVariables   (short-term, acts on ρ)

The state is created by `plasticityvariables(param, Npre, Npost)`. A rule is executed only
if `any(vars.active)`. Both rules are called only from `train!` (never from `sim!`):
per step and per connection, `update_traces!` before `forward!`, `plasticity!` after it.
=#

abstract type LTPVariables <: PlasticityVariables end

# vector type of the state of a rule: the precision of its parameters (see `float_type`, `convert_precision`)
_vft(param) = Vector{float_type(param)}

"""
    LTPParameter <: PlasticityParameter

Abstract supertype of the long-term plasticity rules (passed to `SpikingSynapse` with the
keyword `LTPParam`): the pair/triplet STDP rules (`STDPParameter` subtypes), the inhibitory
rules (`iSTDPParameter` subtypes), the voltage-based rules `vSTDPParameter` (event-driven, the default) and `vSTDPDenseParameter` (clock-driven reference), and `NoLTP` (no plasticity, default).
"""
abstract type LTPParameter <: PlasticityParameter end
abstract type STPVariables <: PlasticityVariables end
"""
    STPParameter <: PlasticityParameter

Abstract supertype of the short-term plasticity rules (passed to `SpikingSynapse` with the
keyword `STPParam`): the Markram/Tsodyks rules (`MarkramSTPParameter`,
`MarkramSTPParameterTimestep`, `MarkramSTPParameterHet`) and `NoSTP` (default). STP rules
modulate the per-synapse efficacy `ρ`, so that a presynaptic spike adds `W[s] * ρ[s]` to the
target conductance.
"""
abstract type STPParameter <: PlasticityParameter end

"""
    NoLTP(; active = [false])

Null long-term plasticity rule, the default `LTPParam` of `SpikingSynapse`. Its variables are
`NoVariables()`, whose `active` flag is `false`, so `train!` skips the LTP update.
`NoSTDP` is a (non-constant) global holding an instance `NoLTP()`.
"""
NoLTP

@snn_kw struct NoLTP <: LTPParameter
    active::VBT = [false]
end
"""
    NoSTP(; active = [false])

Null short-term plasticity rule, the default `STPParam` of `SpikingSynapse`. The efficacy
`ρ` then stays at its initial value 1 for every synapse.
"""
NoSTP

@snn_kw struct NoSTP <: STPParameter
    active::VBT = [false]
end
"""
    NoVariables(; active = [false])

Empty plasticity state returned by `plasticityvariables(::NoLTP, ...)` and
`plasticityvariables(::NoSTP, ...)`. `active = [false]` disables the rule.
"""
NoVariables

@snn_kw struct NoVariables <: PlasticityVariables
    active::VBT = [false]
end

"""
    LTP()

Empty marker type, subtype of `LTPParameter`. No `plasticityvariables` or `plasticity!`
method is defined for it, so it cannot be used as a rule; use a concrete rule or `NoLTP()`.
"""
struct LTP <: LTPParameter end
"""
    STP()

Empty marker type, subtype of `STPParameter`. No `plasticityvariables` or `plasticity!`
method is defined for it, so it cannot be used as a rule; use a concrete rule or `NoSTP()`.
"""
struct STP <: STPParameter end

"""
    plasticityvariables(param::PlasticityParameter, Npre, Npost)

Create the state (variables) of the plasticity rule `param` for a connection with `Npre`
presynaptic and `Npost` postsynaptic neurons. Called by the `SpikingSynapse` constructor and
by `change_plasticity!`. Returns `NoVariables()` for `NoLTP`/`NoSTP`, `STDPVariables` for the
pair STDP rules, `STDPTripletVariables` for `STDPTriplet`, `STDPStructuredVariables` for
`STDPSymmetric`/`STDPAntiSymmetric`, `iSTDPVariables` for the iSTDP rules, `vSTDPVariables`
for `vSTDPParameter`, `vSTDPDenseVariables` for `vSTDPDenseParameter` and `MarkramSTPVariables` for the Markram STP rules. The element type of the state is that of the (converted) rule, `float_type(param)`.
"""
plasticityvariables(param::NoLTP, Npre, Npost) = NoVariables()
plasticityvariables(param::NoSTP, Npre, Npost) = NoVariables()

"""
    plasticity!(c::AbstractSparseSynapse, param::AbstractSpikingSynapseParameter, dt::AbstractFloat, T::Time)

Apply one step of the plasticity rules of the sparse synapse `c`: first the short-term rule
`plasticity!(c, c.STPParam, c.STPVars, dt, T)` if `any(c.STPVars.active)`, then the long-term
rule `plasticity!(c, c.LTPParam, c.LTPVars, dt, T)` if `any(c.LTPVars.active)`.

It is called by `train!` after `forward!` of the connection, at every step; `sim!` never
calls it. The rule-specific methods `plasticity!(c, param::PlasticityParameter,
variables::PlasticityVariables, dt, T)` update `c.W` (LTP) or `c.ρ` (STP) in place; see the
docstring of each rule for its equations.
"""
function plasticity!(
    c::PT,
    param::ST,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse, ST<:AbstractSpikingSynapseParameter}
    any(c.STPVars.active) && plasticity!(c, c.STPParam, c.STPVars, dt, T)
    any(c.LTPVars.active) && plasticity!(c, c.LTPParam, c.LTPVars, dt, T)
    return nothing
end

"""
    update_traces!(c::AbstractSparseSynapse, param::AbstractSpikingSynapseParameter, dt::AbstractFloat, T::Time)

Pre-propagation part of the plasticity rules of `c`, called by `train!` at every step
*before* `forward!` (never by `sim!`): `update_traces!(c, c.STPParam, c.STPVars, dt, T)` if
the STP rule is active, then the same for the LTP rule. The fallback method for a generic
rule does nothing; `MarkramSTPParameter` (event-driven) and `MarkramSTPParameterHet` use it to
compute the efficacy `ρ` of the synapses of the neurons that spike in this step, so that the
spike is transmitted with the updated efficacy.
"""
function update_traces!(
    c::PT,
    param::ST,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse, ST<:AbstractSpikingSynapseParameter}
    any(c.STPVars.active) && update_traces!(c, c.STPParam, c.STPVars, dt, T)
    any(c.LTPVars.active) && update_traces!(c, c.LTPParam, c.LTPVars, dt, T)
    return nothing
end

function update_traces!(
    c::PT,
    param::PlasticityParameter,
    variables::PlasticityVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse} end

"""
    set_plasticity!(c::AbstractSparseSynapse, param::LTPParameter, state::Bool)
    set_plasticity!(c::AbstractSparseSynapse, param::STPParameter, state::Bool)

Switch the long-term (if `param` is an `LTPParameter`) or short-term (`STPParameter`) rule of
`c` on (`state = true`) or off, by setting the `active` flag of `c.LTPVars` or `c.STPVars`.
Only the type of `param` is used to choose the rule. For the variant that acts on any
connection see `set_plasticity!(synapse::AbstractConnection, bool)`.
"""
function set_plasticity!(c::AbstractSparseSynapse, param::LTPParameter, state::Bool)
    c.LTPVars.active .= state
end

function set_plasticity!(c::AbstractSparseSynapse, param::STPParameter, state::Bool)
    c.STPVars.active .= state
end

"""
    set_STP!(c::AbstractConnection, state::Bool)

Activate (`true`) or deactivate (`false`) the short-term plasticity rule of a sparse synapse
by setting `c.STPVars.active`. A no-op for connections that are not `AbstractSparseSynapse`.
Deactivation freezes `ρ` at its current value. Has an effect only under `train!`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 10, param = SNN.PoissonParameter(10Hz))
I = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.5, μ = 1.0), STPParam = SNN.MarkramSTPParameter())
SNN.set_STP!(syn, false)
```
"""
function set_STP!(c::AbstractSparseSynapse, state::Bool)
    c.STPVars.active .= state
end

"""
    set_LTP!(c::AbstractConnection, state::Bool)

Activate (`true`) or deactivate (`false`) the long-term plasticity rule of a sparse synapse
by setting `c.LTPVars.active`. A no-op for connections that are not `AbstractSparseSynapse`.
Has an effect only under `train!` (`sim!` never runs plasticity).
"""
function set_LTP!(c::AbstractSparseSynapse, state::Bool)
    c.LTPVars.active .= state
end

set_LTP!(c::AbstractConnection, state) = nothing
set_STP!(c::AbstractConnection, state) = nothing

function plasticity!(
    c::AbstractSparseSynapse,
    param::PT,
    variables::NoVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:PlasticityParameter} end


#=
Reading the state of a plasticity rule
--------------------------------------
Some rules store a state variable, or the synaptic weights, in a form that is not their value:

  - lazily decayed traces (`STDPVariables.tpre`, `tpost`; `STDPTripletVariables.r1`, `r2`, `o1`, `o2`):
    value = stored x `scale`;
  - event-updated short-term variables (`MarkramSTPVariables.u`, `x` of the event-driven rules): updated
    only when the presynaptic neuron spikes, value at time t = exact relaxation from `last_spike`;
  - deferred weight updates (`vSTDPVariables` of the event-driven vSTDP): `syn.W` holds the weights as of the
    last materialisation of each synapse; the current weights add the pending potentiation.

One mechanism covers all of them. A variables type declares the fields that are not values in
`lazy_variables(vars)`, which may include the name `:W` of the synaptic weight vector, and defines how to read them:
`current_value(vars, param, key, t)` for a field of the variables, `current_weights(c, vars)` for `W`.
`plasticity_value(obj, group, key, T)` (a field of the variables) and `plasticity_value(syn, :W, T)` (the
weights) are the accessors; `trace_values(syn, key)` is the same without a time, and `sync_traces!(syn)`
makes the stored fields hold their values (also for `W`, through `sync_weights!(c, vars)`). The recorder
(`monitor!(syn, keys, :LTPVars)`, `:STPVars`, and `monitor!(syn, :W)`) reads through the accessor, so records never
contain stored values. A field declared lazy without an accessor method throws an `ArgumentError`, at
`monitor!` time for a recording. Direct field access (`syn.LTPVars.tpre`, `syn.STPVars.u`, `syn.W`) returns the stored
values.
=#

"""
    lazy_variables(vars::PlasticityVariables) -> Tuple{Vararg{Symbol}}

Names of the fields of `vars` that are stored in a form other than their value (scaled or updated only at
events), and `:W` if the rule defers the update of the synaptic weights. Empty by default. A variables type that
returns names here must have a `current_value(vars, param, key, t)` method (see `current_value`) that gives their
values, and, for `:W`, methods `current_weights(c, vars)` and `sync_weights!(c, vars)` (see
`plasticity_value`): the generic methods throw an `ArgumentError` for them, so a record or a read of such a field never
contains stored values without notice.
"""
lazy_variables(::PlasticityVariables) = ()

"""
    plasticity_value(obj, group::Symbol, key::Symbol, T)
    plasticity_value(syn::AbstractSparseSynapse, :W, T = 0)

Current value of the plasticity variable `key` of `getfield(obj, group)` (`group = :LTPVars` or `:STPVars`
of a sparse synapse) at time `T` (a `Time` or a number, ms), without changing the state. For the lazily
decayed STDP traces this is the stored trace times its `scale` (same as [`trace_values`](@ref)); for `u` and
`x` of the event-driven Markram rules the exact relaxation from the last spike of each presynaptic neuron to
`T`; for every other field the field itself. The rule of the group (`obj.LTPParam`, `obj.STPParam`) is
passed to [`current_value`](@ref), the extension point.

`plasticity_value(syn, :W)` is the synaptic weight vector as of the end of the last step: `syn.W` itself for every
rule, except for rules that defer the weight update (the event-driven `vSTDPParameter`, which lists `:W` in
[`lazy_variables`](@ref)): a scratch vector (overwritten by the next call, no allocation after the first) with
the pending potentiation added and clamped, state unchanged. Read weights through it (or call
[`sync_traces!`](@ref) first) whenever a deferred rule may be active: plots, analyses, saving.

This is the accessor used by `monitor!`; direct field access (`syn.LTPVars.tpre`, `syn.STPVars.u`, `syn.W`) returns
the stored values, which are not the values for lazily stored variables.

The value read between steps is the one after the last step that ran. Under `sim!` (which does not run the
plasticity rules) the event-updated short-term variables keep relaxing in this accessor with the clock, as
they will at the next spike. A time given as a number is converted to the precision of `obj` (`float_type`).
"""
plasticity_value(obj, group::Symbol, key::Symbol, T) =
    current_value(getfield(obj, group), _group_rule(obj, group), key, _time_ms(obj, T))

function plasticity_value(c::AbstractSparseSynapse, key::Symbol, T = 0)
    key === :W || throw(ArgumentError("plasticity_value(syn, key, T): only :W is read here; for a field of the rule use plasticity_value(syn, :LTPVars | :STPVars, key, T)"))
    return current_weights(c)
end

"""
    current_value(vars, param, key::Symbol, t)

Extension point of [`plasticity_value`](@ref): the value at time `t` (ms) of the field `key` of the
variables `vars` of the rule `param`. The default returns the field; a variables type with lazily stored
fields declares them in [`lazy_variables`](@ref) and defines a method (see `STDPVariables`,
`STDPTripletVariables`, `MarkramSTPVariables`). For a declared lazy field without a method the default
throws an `ArgumentError`, so that a record never contains stored values without notice.
"""
function current_value(vars::PlasticityVariables, param, key::Symbol, t::Real)
    key in lazy_variables(vars) && throw(ArgumentError(
        "plasticity variable :$key of $(nameof(typeof(vars))) is stored lazily and has no current_value method"))
    return getfield(vars, key)
end
current_value(vars, param, key::Symbol, t::Real) = getfield(vars, key)   # not a plasticity rule: as it is

"""
    current_weights(c::AbstractSparseSynapse)
    current_weights(c, vars::PlasticityVariables)

Thin wrapper of `plasticity_value(c, :W)` and its extension point (not exported). The two-argument form is
the one a rule that defers the weight update (`:W` in `lazy_variables(vars)`) defines; without a method such a rule
throws an `ArgumentError` here, as for any lazy field. A synapse without plasticity state returns `c.W`.
"""
current_weights(c::AbstractSparseSynapse) = hasfield(typeof(c), :LTPVars) ? current_weights(c, c.LTPVars) : c.W
function current_weights(c, vars::PlasticityVariables)
    :W in lazy_variables(vars) && throw(ArgumentError(
        "the weights W of this synapse are updated lazily by $(nameof(typeof(vars))) and it has no current_weights method"))
    return c.W
end

"""
    sync_weights!(c::AbstractSparseSynapse)
    sync_weights!(c, vars::PlasticityVariables)

Materialise the deferred weight updates of `c` in place (`c.W` becomes `plasticity_value(c, :W)`), a no-op for
the other rules; returns `c`. Called by `sync_traces!(c)`, which is the entry point; thin wrapper (not exported)
and extension point of the rules that list `:W` in `lazy_variables` (without a method they throw an `ArgumentError`).
"""
sync_weights!(c::AbstractConnection) = c
sync_weights!(c::AbstractSparseSynapse) = (hasfield(typeof(c), :LTPVars) && sync_weights!(c, c.LTPVars); c)
function sync_weights!(c, vars::PlasticityVariables)
    :W in lazy_variables(vars) && throw(ArgumentError(
        "the weights W of this synapse are updated lazily by $(nameof(typeof(vars))) and it has no sync_weights! method"))
    return nothing
end

_group_rule(c::AbstractSparseSynapse, group::Symbol) =
    group === :LTPVars ? c.LTPParam : group === :STPVars ? c.STPParam : nothing
_group_rule(obj, group::Symbol) = nothing
_time_ms(obj, T::Time) = get_time(T)
_time_ms(obj, t::Real) = convert(float_type(obj), t)

## STP
include("sparse_plasticity/STP.jl")

## STDP
"""
    STDPParameter <: LTPParameter

Abstract supertype of the spike-timing-dependent rules: `STDPGerstner`, `STDPConfavreux2025`,
`STDPMexicanHat`, `STDPWeightDependent`, `STDPTriplet`, the structured rules
(`STDPSymmetric`, `STDPAntiSymmetric`) and the inhibitory rules (`iSTDPParameter`). The
default state of an `STDPParameter` is `STDPVariables`; subtypes with other state define
their own `plasticityvariables` method.
"""
abstract type STDPParameter <: LTPParameter end
NoSTDP = NoLTP()
"""
    NoSTDP

Non-constant global holding `NoLTP()`, kept for backward compatibility. Use `NoLTP()`.
"""
NoSTDP
include("sparse_plasticity/vSTDP.jl")
include("sparse_plasticity/iSTDP.jl")
# include("sparse_plasticity/longshortSP.jl")
include("sparse_plasticity/STDP_kernels.jl")
include("sparse_plasticity/STDP_traces.jl")
include("sparse_plasticity/STDP_weight_dependent.jl")
include("sparse_plasticity/STDP_triplet.jl")
include("sparse_plasticity/STDP_structured.jl")

"""
    change_plasticity!(syn; LTP = nothing, STP = nothing)

Replace the long-term rule (`LTP`, an `LTPParameter`) and/or the short-term rule (`STP`, an
`STPParameter`) of the sparse synapse `syn` by a rule of the same type, re-creating the
corresponding state with `plasticityvariables` (traces and STP variables are reset). Arguments
left to `nothing` are not changed. Same behaviour as `update_plasticity!(c::SpikingSynapse; LTP, STP)`.
The rule types are type parameters of `SpikingSynapse`: a rule of another type is an
`ArgumentError`, use `with_plasticity` to get a synapse with the new rule.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 10, param = SNN.PoissonParameter(10Hz))
I = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.5, μ = 1.0), LTPParam = SNN.STDPGerstner())
SNN.change_plasticity!(syn; LTP = SNN.STDPGerstner(A_pre = 2e-4))
```
"""
function change_plasticity!(syn; LTP = nothing, STP = nothing)
    update_plasticity!(syn; LTP, STP)
end

export SpikingSynapse,
    PlasticityParameter,
    SpikingSynapseParameter,
    NoSTDP,
    plasticityvariables,
    plasticity!,
    change_plasticity!,
    set_plasticity!,
    set_STP!,
    set_LTP!,
    update_traces!,
    NoVariables


export LTP, STP, NoLTP, NoSTP
