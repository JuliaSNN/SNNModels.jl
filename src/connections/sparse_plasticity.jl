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
rules (`iSTDPParameter` subtypes), the voltage-based rules `vSTDPParameter` (event-driven) and
`vSTDPDenseParameter` (clock-driven), and `NoLTP` (no plasticity, default).
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
for `vSTDPParameter`, `vSTDPDenseVariables` for `vSTDPDenseParameter` and `MarkramSTPVariables`
for the Markram STP rules. The element type of the state is `float_type(param)`.
"""
plasticityvariables(param::NoLTP, Npre, Npost) = NoVariables()
plasticityvariables(param::NoSTP, Npre, Npost) = NoVariables()

"""
    needs_transpose(rule::PlasticityParameter)

Whether the rule reads the transposed index arrays `J`/`index` of a `SpikingSynapse` (the inputs of a postsynaptic
neuron). `true` by default (also for user-defined rules); `false` for `NoLTP`, `NoSTP`, the Markram STP rules and vSTDP.
"""
needs_transpose(::PlasticityParameter) = true
needs_transpose(::NoLTP) = false
needs_transpose(::NoSTP) = false

# whether a sparse synapse has a rule at all (a type-level constant for `SpikingSynapse`)
_has_rules(c) = true

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
    _has_rules(c) || return nothing
    _ensure_rule_arrays!(c)   # serial, before the rule kernels fork
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
    _has_rules(c) || return nothing
    _ensure_rule_arrays!(c)
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
Some rules store a field in a form that is not its value:
  - lazily decayed traces (`STDPVariables.tpre`, `tpost`, the triplet traces): value = stored x `scale`;
  - event-updated `u`, `x` of the event-driven Markram rules: value = relaxation from `last_spike`;
  - deferred weight updates of the event-driven vSTDP: value = `syn.W` + pending potentiation.
A variables type lists such fields (and `:W`) in `lazy_variables(vars)` and defines `current_value` (and
`current_weights`/`sync_weights!` for `:W`). `plasticity_value` and `trace_values` read values, `sync_traces!`
writes them into the stored fields; the recorder reads through `plasticity_value`. Direct field access returns
the stored form.
=#

"""
    lazy_variables(vars::PlasticityVariables) -> Tuple{Vararg{Symbol}}

Names of the fields of `vars` stored in a form other than their value, and `:W` if the rule defers weight updates.
Empty by default. Each name needs a `current_value(vars, param, key, t)` method (for `:W`: `current_weights(c, vars)`
and `sync_weights!(c, vars)`); the generic methods throw an `ArgumentError` for a listed name.
"""
lazy_variables(::PlasticityVariables) = ()

"""
    plasticity_value(obj, group::Symbol, key::Symbol, T)
    plasticity_value(syn::AbstractSparseSynapse, :W, T = 0)

Value of the plasticity variable `key` of `getfield(obj, group)` (`group = :LTPVars` or `:STPVars`) at time `T`
(a `Time` or a number in ms), without changing the state. Lazily stored fields (see [`lazy_variables`](@ref)) are
converted through [`current_value`](@ref); any other field is returned as it is. Used by `monitor!`.

`plasticity_value(syn, :W)` is the weight vector after the last step: `syn.W`, or, for a rule that defers weight
updates (`vSTDPParameter`), a scratch vector with the pending updates applied, overwritten by the next call. Read
weights through it, or call [`sync_traces!`](@ref) first, whenever such a rule may be active.
"""
plasticity_value(obj, group::Symbol, key::Symbol, T) =
    current_value(getfield(obj, group), _group_rule(obj, group), key, _time_ms(obj, T))

function plasticity_value(c::AbstractSparseSynapse, key::Symbol, T = 0)
    key === :W || throw(ArgumentError("plasticity_value(syn, key, T): only :W is read here; for a field of the rule use plasticity_value(syn, :LTPVars | :STPVars, key, T)"))
    return current_weights(c)
end

"""
    current_value(vars, param, key::Symbol, t)

Extension point of [`plasticity_value`](@ref): value at time `t` (ms) of the field `key` of the variables `vars`
of the rule `param`. The default returns the field, and throws an `ArgumentError` for a field listed in
[`lazy_variables`](@ref).
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

Implementation of `plasticity_value(c, :W)` (not exported). A rule listing `:W` in `lazy_variables` defines the
two-argument method; the default returns `c.W` and throws for such a rule.
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

Write the deferred weight updates into `c.W` (a no-op for the other rules); returns `c`. Not exported; called by
`sync_traces!(c)`. A rule listing `:W` in `lazy_variables` defines the two-argument method.
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
include("sparse_plasticity/STDP_kernels.jl")
include("sparse_plasticity/STDP_traces.jl")
include("sparse_plasticity/STDP_weight_dependent.jl")
include("sparse_plasticity/STDP_triplet.jl")
include("sparse_plasticity/STDP_structured.jl")

"""
    change_plasticity!(syn; LTP = nothing, STP = nothing)

Replace the long-term rule (`LTP`, an `LTPParameter`) and/or the short-term rule (`STP`, an
`STPParameter`) of the sparse synapse `syn` by a rule of the same type, re-creating the
corresponding state (traces and STP variables are reset). Alias of `update_plasticity!`: a rule of
another type is an `ArgumentError`, use `with_plasticity` instead.
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
