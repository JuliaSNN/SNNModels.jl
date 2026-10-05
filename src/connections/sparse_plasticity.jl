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

"""
    LTPParameter <: PlasticityParameter

Abstract supertype of the long-term plasticity rules (passed to `SpikingSynapse` with the
keyword `LTPParam`): the pair/triplet STDP rules (`STDPParameter` subtypes), the inhibitory
rules (`iSTDPParameter` subtypes), `vSTDPParameter`, and `NoLTP` (no plasticity, default).
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
for `vSTDPParameter` and `MarkramSTPVariables` for the Markram STP rules.
"""
plasticityvariables(param::NoLTP, Npre, Npost) = NoVariables()
plasticityvariables(param::NoSTP, Npre, Npost) = NoVariables()

"""
    plasticity!(c::AbstractSparseSynapse, param::AbstractSpikingSynapseParameter, dt::Float32, T::Time)

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
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse, ST<:AbstractSpikingSynapseParameter}
    any(c.STPVars.active) && plasticity!(c, c.STPParam, c.STPVars, dt, T)
    any(c.LTPVars.active) && plasticity!(c, c.LTPParam, c.LTPVars, dt, T)
end

"""
    update_traces!(c::AbstractSparseSynapse, param::AbstractSpikingSynapseParameter, dt::Float32, T::Time)

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
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse, ST<:AbstractSpikingSynapseParameter}
    any(c.STPVars.active) && update_traces!(c, c.STPParam, c.STPVars, dt, T)
    any(c.LTPVars.active) && update_traces!(c, c.LTPParam, c.LTPVars, dt, T)
end

function update_traces!(
    c::PT,
    param::PlasticityParameter,
    variables::PlasticityVariables,
    dt::Float32,
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
    dt::Float32,
    T::Time,
) where {PT<:PlasticityParameter} end


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
`STPParameter`) of the mutable sparse synapse `syn`, re-creating the corresponding state
with `plasticityvariables` (traces and STP variables are reset). Arguments left to `nothing`
are not changed. Same behaviour as `update_plasticity!(c::SpikingSynapse; LTP, STP)`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 10, param = SNN.PoissonParameter(10Hz))
I = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.5, μ = 1.0))
SNN.change_plasticity!(syn; LTP = SNN.STDPGerstner())
```
"""
function change_plasticity!(syn; LTP = nothing, STP = nothing)
    @unpack fireI, fireJ = syn
    Npre, Npost = length(fireJ), length(fireI)
    if !isnothing(LTP)
        syn.LTPParam = LTP
        syn.LTPVars = plasticityvariables(LTP, Npre, Npost)
    end
    if !isnothing(STP)
        syn.STPParam = STP
        syn.STPVars = plasticityvariables(STP, Npre, Npost)
    end
end

export SpikingSynapse,
    PlasticityParameter,
    SpikingSynapseParameter,
    no_STDPParameter,
    NoSTDP,
    no_PlasticityVariables,
    plasticityvariables,
    plasticity!,
    change_plasticity!, set_plasticity!,
    set_STP!, set_LTP!,
     update_traces!,
     NoVariables


export LTP, STP, NoLTP, NoSTP
