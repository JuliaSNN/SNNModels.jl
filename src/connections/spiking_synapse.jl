
@doc raw"""
    SpikingSynapseParameter()

Parameter of a `SpikingSynapse` without transmission delays (no fields). Selects the
`forward!` method that adds ``W_{ij}\,\rho_{ij}`` to the target in the same step in which
the presynaptic neuron fires.
"""
struct SpikingSynapseParameter <: AbstractSpikingSynapseParameter end

@doc raw"""
    SpikingSynapseDelayParameter(; delaytime)

Parameter of a `SpikingSynapse` with per-synapse transmission delays (not exported; built by
the `SpikingSynapse` constructor when `delay_dist` is given). The delayed increments are kept in
a ring buffer indexed by arrival step: at the first `forward!` (and whenever `dt` or the delays
change) every delay is rounded up to a whole number of steps, `ceil(delay / dt)`, the first step
whose time is not before the arrival time (so 0.8 ms at `dt = 0.125` ms arrives 7 steps later).
A delay stored once (common to all synapses) is delivered through one ring column per step; without LTP and STP the spike lists
are delayed instead (`spike_ring`, `W` read at arrival, the buffer has 0 rows); switching path carries the spikes in flight over (ring to buffer column, buffer drained for `d` steps)
(`delay_kernels.jl`).

# Fields
- `delaytime::Vector{FT}`: delay of each synapse (ms), in the CSC order of `W`; a single value when all synapses have the
  same delay (e.g. `delay_dist = Dirac(d)`), which then applies to every synapse.
- `delaysteps::Vector{Int32}`: delay of each synapse (or the common delay) in steps of the `dt` of the last `forward!`.
- `buffer::Matrix{FT}`: `Npost x (max(delaysteps) + 1)` increments ``W_{ij}\,\rho_{ij}``
  (evaluated at emission time) per target and arrival step, a ring indexed from `pos`.
- `dt::Vector{FT}`: the step `delaysteps` was computed for (`NaN` before the first `forward!`).
- `pos::Vector{Int}`: ring slot (0-based) of the current step.
- `spike_ring::Vector{Vector{Int32}}`: without plasticity and with a delay of `d` steps stored once, the spike lists of the last `d + 1` steps (delayed instead of the increments).
- `drain::Vector{Int}`: steps left to deliver increments still in the buffer after a switch from the buffer path to the spike-list path.
"""
SpikingSynapseDelayParameter

const SpikeRing = Vector{Vector{Int32}}   # spike lists of the delayed constant-delay path (delay_kernels.jl)
@snn_kw mutable struct SpikingSynapseDelayParameter{VFT = Vector{Float32}, VIT = Vector{Int32}, MFT = Matrix{Float32}, VPT = Vector{Int}} <: AbstractSpikingSynapseParameter
    delaytime::VFT
    delaysteps::VIT = Int32[]
    buffer::MFT = zeros(eltype(delaytime), 0, 1)
    dt::VFT = [eltype(delaytime)(NaN)]
    pos::VPT = [0]
    spike_ring::SpikeRing = Vector{Int32}[]
    drain::VPT = [0]
end

# Round the delays to steps of `dt` (up) and size the ring buffer; a no-op while `dt` and the delays are unchanged.
function _delay_steps!(param::SpikingSynapseDelayParameter, dt::AbstractFloat, Npost::Int)
    @unpack delaytime, delaysteps = param
    (param.dt[1] == dt && length(delaysteps) == length(delaytime)) && return nothing
    resize!(delaysteps, length(delaytime))
    for s in eachindex(delaytime)
        # 1f-4 steps of tolerance for the rounding error of the quotient (0.75 ms / 0.125 ms is 6 steps)
        delaysteps[s] = max(0, ceil(Int32, delaytime[s] / dt - oftype(dt, 1e-4)))
    end
    nslots = Int(isempty(delaysteps) ? 0 : maximum(delaysteps)) + 1
    size(param.buffer, 2) == nslots || (param.buffer = zeros(eltype(param.buffer), 0, nslots))   # rows: _delay_buffer!
    param.pos[1] = clamp(param.pos[1], 0, nslots - 1)
    param.dt[1] = dt
    return nothing
end

@snn_kw mutable struct SpikingSynapse{
                VIT = Vector{Int32},
                VFT = Vector{Float32},
                SYNP <: AbstractSpikingSynapseParameter,
                LTPP <: LTPParameter,
                STPP <: STPParameter,
                LTPV <: PlasticityVariables,
                STPV <: PlasticityVariables
                } <: AbstractSpikingSynapse
    id::String = randstring(12)
    name::String = "SpikingSynapse"
    param::SYNP = SpikingSynapseParameter()
    LTPParam::LTPP = NoLTP()
    STPParam::STPP = NoSTP()
    LTPVars::LTPV = NoVariables()
    STPVars::STPV = NoVariables()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    ρ::VFT  # short-term plasticity
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    v_post::VFT
    g::VFT  # rise conductance
    targets::Dict = Dict()
    records::Dict = Dict()
    pre_spikes::SpikeIndex = SpikeIndex(length(fireJ)) # indices of the presynaptic spikes of the step
    post_spikes::SpikeIndex = SpikeIndex(length(fireI)) # indices of the postsynaptic spikes of the step
    chunkptr::ChunkPtr = ChunkPtr() # sub-row pointers for the threaded forward!/plasticity (built on first use)
end

@doc raw"""
    SpikingSynapse(pre, post, sym, comp = nothing; conn, delay_dist = nothing, dt = 0.125f0,
                   LTPParam = NoLTP(), STPParam = NoSTP(), name = "SpikingSynapse")

Sparse synapse that propagates the spikes of `pre` to the target variable `sym` (and
compartment `comp`) of `post`.

# Transmission
At every step (`forward!`, called by both `sim!` and `train!`), for every presynaptic neuron
``j`` with `pre.fire[j] == true` and every stored synapse ``(i, j)``:
```math
g_i \leftarrow g_i + W_{ij}\,\rho_{ij}
```
where ``g`` is the postsynaptic variable resolved by `synaptic_target(targets, post, sym, comp)`
(for generalized IF models the generic mapping of `get_synapse_symbol` sends `:ge`/`:he` to
the receptor array `:glu` and `:gi`/`:hi` to `:gaba`; other symbols are used as given), ``W_{ij}`` is the weight and ``\rho_{ij}`` the
short-term efficacy (1 without STP). The units of ``W`` are the units of the target variable
(e.g. nS for conductances, pA for current synapses); ``W`` is added once per spike, the
postsynaptic model then integrates the target.

With `delay_dist`, every synapse has a fixed delay ``d_{ij}`` (ms) and the increment is put in
a queue of the postsynaptic neuron with arrival time ``t + d_{ij}``; at every step all queued
increments with arrival time ``\le t`` are added to ``g_i``.

# Arguments
- `pre`, `post`: populations (`AbstractPopulation`).
- `sym::Symbol`: target conductance/current of `post` (e.g. `:ge`, `:gi`, `:he`, `:hi`, `:g`,
  or a receptor name); `comp`: compartment for multicompartment models (e.g. `:d1`, `:s`).
- `conn`: either a `NamedTuple` of `sparse_matrix` options (`p` or `ρ`, `μ`, `σ`, `dist`, `rule`,
  `γ`, `kmin`) or an explicit `Npost x Npre` matrix (dense or sparse, any element type).
- `delay_dist`: optional `Distribution`; one delay (ms) per synapse is drawn from it (stored once if all
  drawn delays are equal, e.g. `Dirac(d)`).
- `dt`: unused (kept for backward compatibility).
- `LTPParam`: long-term plasticity rule (`STDPGerstner`, `STDPTriplet`, `STDPWeightDependent`,
  `iSTDPRate`, `vSTDPParameter`, ...). Applied only when the network is run with `train!`;
  `sim!` propagates spikes but never updates the weights.
- `STPParam`: short-term plasticity rule acting on the efficacy `ρ` (e.g. `MarkramSTPParameter`),
  also applied only by `train!`.
- `name`: name used in `print_model` and records.

`LTPParam` and `STPParam` are keyword-argument (and field) names, not types: the abstract
rule types are `LTPParameter` and `STPParameter`.

# Fields
- `rowptr`, `colptr`, `I`, `J`, `index`, `W`: double sparse storage, as returned by `dsparse`
  (CSC column pointers `colptr`, postsynaptic index `I` and presynaptic index `J` of every
  synapse, row pointers `rowptr` of the transposed matrix and the map `index` from row-major
  to CSC position); `W::Vector{FT}` holds the weights in CSC order (`FT` = precision of the synapse).
  `J` and `index` are empty unless something reads them: a plasticity rule that walks the inputs of a
  neuron (`needs_transpose`), `presynaptic`, `presynaptic_idxs`, `SynapseNormalization`, `AggregateScaling`;
  call `SNNModels.ensure_transpose!(c)` before reading them in user code (since 1.10). `with_plasticity`/`without_plasticity`
  hand `J`, `index` and `ρ` over only to a synapse whose rules need them.
- `ρ::Vector{FT}`: short-term efficacy per synapse (initialised to 1), stored only with a short-term rule
  (`STPParam`); otherwise empty (ρ = 1). Read it with `SNNModels.efficacy(c)`; call
  `SNNModels.materialize_efficacy!(c)` before writing to it (since 1.10).
- `fireI`, `fireJ`: references to `post.fire` and `pre.fire`.
- `g`, `v_post`: references to the target variable and to the membrane potential of the
  target compartment.
- `param`: `SpikingSynapseParameter` or `SpikingSynapseDelayParameter`.
- `LTPParam`, `STPParam`, `LTPVars`, `STPVars`: plasticity rules and their state.
- `targets::Dict`: ids of `pre`/`post`, target symbol, connection type.
- `records::Dict`: recorded variables.
- `chunkptr::ChunkPtr`: cache of the sub-row pointers of the threaded `forward!`/plasticity (see Threading).
- `pre_spikes`, `post_spikes`: `SpikeIndex` scratch lists of the neurons that fired in the current step,
  built from `fireJ`/`fireI` once per step and shared by `forward!` and the event-driven rules. If `pre` (or
  `post`) keeps its own list up to date (`Identity` driven by `SpikeTimeStimulus`), the synapse shares it and never
  scans the flags.

# Threading
With `neurons_per_thread > 0` (see `set_neurons_per_thread!`) and more than one Julia thread, `forward!` (with and without delays), the pair, triplet,
weight-dependent, structured and inhibitory event-driven STDP rules and the event-driven STP efficacy update are partitioned by postsynaptic range, with the
same chunks as the postsynaptic population: chunk `c` adds to `g[post chunk c]` (and the delay buffer rows of that range) and updates the weights of the
synapses that end there, using the sub-row pointers `chunkptr` (built on first use, 4 (nchunks + 1) bytes per presynaptic neuron, invalidated by
`update_sparse_matrix!`). Every target receives its additions in the order of the serial loop, so the results are bit-identical for any chunking and any
number of threads. Synapses with less work than `SNNModels.SYN_MIN_OPS[]` operations per step run on the calling thread. The voltage-based vSTDP and
`iSTDPPotential` plasticity stays serial.

# Notes
- Precision. The synapse has the floating-point type of the variable it writes into, i.e. of the target
  population (`float_type(post)`: `Float32` by default, `Float64` for a population built with
  `VFT = Vector{Float64}`). All synaptic data (`W`, `ρ`, delays, delay queues, the state of the plasticity
  rules) have that type whatever the element type of `conn` or of `delay_dist`; conversion happens in the
  constructor. The rules `LTPParam`, `STPParam` are converted to it too (`convert_precision`; build a rule with
  `FT = Float64` to have `Float64` values: a `Float32` rule keeps its `Float32` values). The presynaptic
  population may have any precision. If the population mixes precisions (its target variable and membrane
  potential differ), the constructor throws an `ArgumentError`.
- Weights of a rule that defers their update (the event-driven `vSTDPParameter`) are not stored in `W`
  between materialisations: read them with `plasticity_value(syn, :W)` (see `sync_traces!`).
- If `pre == post`, autapses are removed structurally (no stored zero-weight self synapses
  that plasticity could grow). With `rule = :Fixed` this lowers the in-degree by one for the
  neurons that had drawn themselves.
- `conn` as a `NamedTuple` is built by `sparse_matrix`, which since SNNModels 1.8.2 uses a
  different random stream than before: seeded networks do not reproduce earlier realisations
  (same statistics).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 400); I = SNN.IF(N = 100)
EI = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 3.0))
IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 5.0), LTPParam = SNN.iSTDPRate(r = 5Hz))
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 2.0), delay_dist = SNN.SNNModels.Uniform(1ms, 3ms))
model = SNN.compose(; E, I, EI, IE, EE)
SNN.sim!(model; duration = 100ms)
```
"""
SpikingSynapse

function input_N(pre::AbstractPopulation)
    return pre.N
end

function output_N(post::AbstractPopulation)
    return post.N
end

function SpikingSynapse(
    pre::AbstractPopulation,
    post::AbstractPopulation,
    sym::Symbol,
    comp::Union{Symbol,Nothing} = nothing;
    conn::Connectivity,
    delay_dist::Union{Distribution,Nothing} = nothing,
    dt::AbstractFloat = 0.125f0,
    LTPParam::LTPParameter = NoLTP(),
    STPParam::STPParameter = NoSTP(),
    name::String = "SpikingSynapse",
)

    # get the presynaptic and postsynaptic firing
    fireI, fireJ = post.fire, pre.fire

    # get the conductance and membrane potential of the target compartment if multicompartment model
    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:SpikingSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, sym, comp)

    # the precision of the synapse is that of the variable it writes into (the target population): weights,
    # efficacy, delays, plasticity state and rules follow it
    FT = _synapse_float_type(post, g, v_post)

    # set the synaptic weight matrix
    w = sparse_matrix(output_N(pre), input_N(post), conn, FT)
    # remove autapses if pre == post (structural removal, the matrix stays sparse)
    (pre == post) && remove_autapses!(w)
    # sparse storage in Int32 (`W` aliases the values of `w`); the transposed arrays `J`, `index` only if a rule reads them
    colptr, I, W = _csc_int32(w)
    rowptr = _rowptr_int32(I, size(w, 1))
    J, index = Int32[], Int32[]

    # set the paramter for the synaptic plasticity
    LTPParam = convert_precision(LTPParam, FT)
    STPParam = convert_precision(STPParam, FT)
    LTPVars = plasticityvariables(LTPParam, pre.N, post.N)
    STPVars = plasticityvariables(STPParam, pre.N, post.N)

    # short-term efficacy: stored only with a short-term rule (empty = 1 for every synapse, see `efficacy`)
    ρ = STPParam isa NoSTP ? FT[] : ones(FT, length(W))

    # Network targets

    if isnothing(delay_dist)
        param = SpikingSynapseParameter()
    else
        # delays have the precision of the synapse whatever the element type of `delay_dist`
        delaytime = FT.(rand(delay_dist, length(W)))
        # a delay common to all synapses (e.g. `Dirac`) is stored once (same draw, so the random stream is unchanged)
        # (a new vector: `resize!` would keep the memory of the full one)
        length(delaytime) > 1 && allequal(delaytime) && (delaytime = delaytime[1:1])
        param = SpikingSynapseDelayParameter(; delaytime, VFT = Vector{FT}, MFT = Matrix{FT})
    end
    c = SpikingSynapse(;
            VFT = Vector{FT},
            ρ = ρ,
            param = param,
            g = g,
            targets = targets,
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
            LTPVars,
            STPVars,
            LTPParam,
            STPParam,
            name,
            pre_spikes = spike_index(pre),
            post_spikes = spike_index(post),
        )
    needs_transpose(c) && ensure_transpose!(c)
    return c
end

"""
    has_transpose(c::SpikingSynapse)
    ensure_transpose!(c::SpikingSynapse)

Whether the transposed index arrays `c.J`, `c.index` are present; build them in place if not (see
`ensure_transpose!(::AbstractConnection)`). Serial; called only outside the threaded kernels.
"""
has_transpose(c::SpikingSynapse) = length(c.index) == length(c.W) && length(c.J) == length(c.W)

# whether a rule of `c` reads `J`/`index` (a constant per synapse type)
needs_transpose(c::SpikingSynapse) = needs_transpose(c.LTPParam) || needs_transpose(c.STPParam)
needs_transpose(c::AbstractConnection) = true
_has_rules(c::SpikingSynapse) = !(c.LTPParam isa NoLTP && c.STPParam isa NoSTP)

# step guard of the rules (serial, before any threaded kernel): a synapse loaded from a file, built with the keyword
# constructor or whose arrays were emptied gets `J`/`index` before a rule reads them, and `ρ` before a short-term rule
# writes it (`train!` calls `update_traces!` first, so this runs before the Markram kernels). `NoSTP` is a type parameter:
# the ρ branch compiles away for a synapse without a short-term rule.
_ensure_rule_arrays!(c) = c
function _ensure_rule_arrays!(c::SpikingSynapse)
    needs_transpose(c) && !has_transpose(c) && ensure_transpose!(c)
    c.STPParam isa NoSTP || length(c.ρ) == length(c.W) || materialize_efficacy!(c)
    return c
end

function ensure_transpose!(c::SpikingSynapse)
    has_transpose(c) && return c
    n = length(c.W)
    _column_index!(resize!(c.J, n), c.colptr)
    _transpose_index!(resize!(c.index, n), c.colptr, c.I, c.rowptr)
    return c
end

"""
    efficacy(c::SpikingSynapse)
    materialize_efficacy!(c::SpikingSynapse)

The short-term efficacy `ρ` is stored only for a synapse with a short-term rule (`STPParam`); otherwise `c.ρ` is empty and
ρ = 1 for every synapse. `efficacy(c)` returns the efficacy of every synapse: `c.ρ` itself when it is stored, else a new
vector of ones (writing to it has no effect). `materialize_efficacy!(c)` stores it (ones) if it is not stored and returns
`c.ρ`: after it, writes to `c.ρ` are used by `forward!` (as in SNNModels up to 1.9).
"""
efficacy(c::SpikingSynapse) = length(c.ρ) == length(c.W) ? c.ρ : ones(eltype(c.W), length(c.W))

function materialize_efficacy!(c::SpikingSynapse)
    length(c.ρ) == length(c.W) && return c.ρ
    fill!(resize!(c.ρ, length(c.W)), one(eltype(c.ρ)))
    return c.ρ
end

"""
    update_plasticity!(c::SpikingSynapse; LTP = nothing, STP = nothing)

Replace the long-term (`LTP`) and/or short-term (`STP`) plasticity rule of `c` by a rule of the
same type (e.g. other parameters) and re-create the corresponding state with `plasticityvariables`.
Arguments left to `nothing` are not changed. A rule of another type is an `ArgumentError`: the
rule types are type parameters of `SpikingSynapse`, use [`with_plasticity`](@ref) instead.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 50)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 2.0), LTPParam = SNN.STDPGerstner())
SNN.SNNModels.update_plasticity!(EE; LTP = SNN.STDPGerstner(A_pre = 2e-4))
```
"""
function update_plasticity!(c::SpikingSynapse; LTP = nothing, STP = nothing)
    Npre, Npost = length(c.fireJ), length(c.fireI)
    FT = float_type(c)
    LTP = isnothing(LTP) ? nothing : convert_precision(LTP, FT)
    STP = isnothing(STP) ? nothing : convert_precision(STP, FT)
    isnothing(LTP) || sync_weights!(c)   # the state of the old rule is dropped: materialise its deferred weight updates
    if !isnothing(LTP)
        vars = plasticityvariables(LTP, Npre, Npost)
        _same_plasticity_types(c.LTPParam, LTP, c.LTPVars, vars) || throw(_plasticity_type_error("LTP"))
        c.LTPParam = LTP
        c.LTPVars = vars
    end
    if !isnothing(STP)
        vars = plasticityvariables(STP, Npre, Npost)
        _same_plasticity_types(c.STPParam, STP, c.STPVars, vars) || throw(_plasticity_type_error("STP"))
        c.STPParam = STP
        c.STPVars = vars
    end
    _ensure_rule_arrays!(c)
end

# The rule and its state are type parameters of SpikingSynapse: only a rule of the same type can be set in place.
_same_plasticity_types(p, q, v, w) = typeof(p) === typeof(q) && typeof(v) === typeof(w)
_plasticity_type_error(kind) = ArgumentError(
    "the $kind rule of a SpikingSynapse is a type parameter: the new rule has a different type, so it cannot be set in place. " *
    "Use `with_plasticity` to build a synapse with the new rule and replace the old one in the model.")

"""
    with_plasticity(c::SpikingSynapse; LTP = nothing, STP = nothing)
    without_plasticity(c::SpikingSynapse)

New `SpikingSynapse` with the connectivity, weights and delays of `c` and the long-term rule `LTP` and/or the short-term
rule `STP`. A rule left to `nothing` is kept, with its state; `NoLTP()` / `NoSTP()` switch that part off;
`without_plasticity(c)` switches both off. The rule types are type parameters of `SpikingSynapse`, so this is how a rule of
another type is installed (`update_plasticity!` and `change_plasticity!` only accept a rule of the same type as the current
one); the caller replaces `c` by the result in its model. `c` itself is not modified.

# Switching on
A rule of any type, converted to the precision of the synapse, with fresh state (`plasticityvariables`). The arrays the new
rules need are created in the new synapse: the transposed index `J`/`index` (+8 B/syn) for a rule that `needs_transpose`
(pair, triplet, weight-dependent, structured and inhibitory STDP), the efficacy `ρ` (+4 B/syn, ones) for a short-term rule.
They are shared with `c` when `c` already has them; the vectors of `c` are never resized or refilled.

# Switching off
The deferred weight updates of the old rule are written back first (`sync_weights!`, event-driven vSTDP); the new synapse
holds the frozen weights `W`, no `J`/`index`, no `ρ` and `NoVariables` state: it has the type, the memory (about 9 B/syn with
a delay common to all synapses) and the per-step work of a synapse built without plasticity, and gives the same run. With
`STP = NoSTP()` the efficacy is 1 again, also when the old short-term rule had moved it (to freeze `ρ` at its current
values instead, keep the rule and call `set_STP!(c, false)`). The final values of the old rule stay readable on `c`, e.g.
`plasticity_value(c, :LTPVars, :tpre, T)`.

# What is shared
`W`, the connectivity (`I`, `colptr`, `rowptr`), `g`, `v_post`, `fireI`, `fireJ`, `param` (delays and delay queue),
`records`, `id`, `name`, the spike lists and the chunk cache are the same objects in `c` and in the result, as is a kept
rule with its state: simulating one changes the state seen by the other, so use only one of the two in a model (or
`deepcopy(c)` first). `c` stays a valid synapse with its own rules. The arrays the result does not take over are freed when
nothing references `c` any more; a `SynapseNormalization` or `AggregateScaling` built on `c` keeps using `c` and its
`J`/`index`, one built on the result builds them there.

# Records
`:W`, `:ρ` (a recorded `ρ` is kept, as ones after a switch-off) and the population records continue across the switch.
The record of a rule variable continues if the new rule has that variable (e.g. `:tpre` from `STDPGerstner` to
`STDPWeightDependent`); otherwise it stops at the switch and stays readable (`getvariable`, `record`); since `records`
is shared, `c` stops sampling it too. A stopped record is not reopened by a later switch (`clear_records!` first).

# Replacing the synapse in a model
`model = compose(; pop = model.pop, syn = merge(model.syn, (; EE = EE2)), stim = model.stim, time = model.time)`, or
assign into a `Vector{AbstractConnection}` (a concretely typed vector cannot hold a synapse with other rule types).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100); E.I .= 300.0
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5))
EE = SNN.with_plasticity(EE; LTP = SNN.STDPGerstner())    # J/index built (+8 B/syn)
model = SNN.compose(; E, EE)
SNN.train!(; model, duration = 100ms)
EE = SNN.without_plasticity(model.syn.EE)                  # W frozen, back to a never-plastic synapse
model = SNN.compose(; pop = model.pop, syn = (; EE), time = model.time)
SNN.sim!(; model, duration = 100ms)
```
"""
function with_plasticity(c::SpikingSynapse; LTP = nothing, STP = nothing)
    Npre, Npost = length(c.fireJ), length(c.fireI)
    FT = float_type(c)
    LTP = isnothing(LTP) ? nothing : convert_precision(LTP, FT)
    STP = isnothing(STP) ? nothing : convert_precision(STP, FT)
    isnothing(LTP) || sync_weights!(c)   # the state of the old rule is replaced: materialise its deferred weight updates
    fields = Dict{Symbol,Any}(f => getfield(c, f) for f in fieldnames(SpikingSynapse))
    fields[:VFT] = typeof(c.W)
    fields[:VIT] = typeof(c.I)
    if !isnothing(LTP)
        fields[:LTPParam] = LTP
        fields[:LTPVars] = plasticityvariables(LTP, Npre, Npost)
    end
    if !isnothing(STP)
        fields[:STPParam] = STP
        fields[:STPVars] = plasticityvariables(STP, Npre, Npost)
    end
    newLTP = isnothing(LTP) ? c.LTPParam : LTP
    newSTP = isnothing(STP) ? c.STPParam : STP
    # the lazy arrays go only to a synapse whose rules need them; the vectors of `c` are never resized or refilled
    needT = needs_transpose(newLTP) || needs_transpose(newSTP)
    if needT && has_transpose(c)
        fields[:J], fields[:index] = c.J, c.index          # shared, already built
    else
        fields[:J], fields[:index] = typeof(c.J)(), typeof(c.index)()   # fresh: filled below if needed, else released
    end
    if isnothing(STP)
        fields[:ρ] = c.ρ                                   # rule kept: its efficacy too, whatever its state
    elseif !(STP isa NoSTP)
        fields[:ρ] = length(c.ρ) == length(c.W) ? c.ρ : ones(eltype(c.W), length(c.W))
    else
        # switched off: efficacy 1 (empty), or stored ones if `ρ` is being recorded (the record continues with its value)
        fields[:ρ] = _records_key(c, :ρ) ? ones(eltype(c.W), length(c.W)) : typeof(c.ρ)()
    end
    new = SpikingSynapse(; fields...)
    needT && ensure_transpose!(new)                        # no-op when shared; fills the fresh vectors otherwise
    isnothing(LTP) || _close_rule_records!(new, :LTPVars)
    isnothing(STP) || _close_rule_records!(new, :STPVars)
    return new
end

# whether `key` is being sampled by the recorder of `c`
_records_key(c, key::Symbol) = haskey(c.records, :data) && key in c.records[:data]

# Records of the variables of a replaced rule (`group` = :LTPVars or :STPVars): a key whose field the new state has goes on
# being sampled; any other key is closed (removed from `records[:data]`): its samples, times and `var_map` entry stay
# readable, no further sample is taken. `records` is shared with the source synapse, which stops sampling it too.
function _close_rule_records!(new::SpikingSynapse, group::Symbol)
    records = new.records
    (haskey(records, :data) && haskey(records, :meta)) || return new
    var_map = get(records[:meta], :var_map, nothing)
    isnothing(var_map) && return new
    vars = getfield(new, group)
    filter!(records[:data]) do k
        m = get(var_map, k, nothing)
        isnothing(m) || m[1] !== group || hasproperty(vars, m[2])
    end
    return new
end

"""
    without_plasticity(c::SpikingSynapse)

`with_plasticity(c; LTP = NoLTP(), STP = NoSTP())`: a synapse with the weights of `c` frozen and no rule, with the type, memory
and per-step work of a synapse built without plasticity. See [`with_plasticity`](@ref).
"""
without_plasticity(c::SpikingSynapse) = with_plasticity(c; LTP = NoLTP(), STP = NoSTP())


@doc raw"""
    forward!(c::SpikingSynapse, param::SpikingSynapseParameter, dt, T)

For each presynaptic neuron ``j`` that fired, add ``W_{ij}\rho_{ij}`` to `c.g[i]` for all its
targets ``i`` (no delay).
"""
function forward!(c::SpikingSynapse, param::SpikingSynapseParameter, dt::AbstractFloat, T::Time)
    @unpack colptr, I, W, g = c
    spikes = presynaptic_spikes(c)
    Npost = length(c.fireI)
    ptr = _syn_chunk_ptr(c.chunkptr, Npost, colptr, I, spikes)
    # function barrier: the kernels are specialised on the efficacy (a vector, or `nothing` for ρ = 1) before any fork
    _forward_nodelay!(g, W, _efficacy_or_nothing(c), I, colptr, ptr, spikes, Npost)
end

function _forward_nodelay!(g, W, ρ, I, colptr, ptr, spikes, Npost)
    if ptr === nothing
        @inbounds for j ∈ spikes # loop on presynaptic neurons that fired
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += _wρ_fast(W, ρ, s)
            end
        end
    else
        # partition by postsynaptic range: chunk `ch` handles the entries of every spiking column whose post lies in its range
        foreach_chunk_c(Npost) do ch, rng
            _forward_chunk!(g, W, ρ, I, ptr, ch, spikes)
        end
    end
    return nothing
end

@inline function _forward_chunk!(g, W, ρ, I, ptr, ch, spikes)
    @inbounds for j ∈ spikes
        @inbounds @fastmath @simd for s ∈ ptr[ch, j]:(ptr[ch+1, j]-1)
            g[I[s]] += _wρ_fast(W, ρ, s)
        end
    end
end

# increment of synapse `s`: W ρ, or W when the efficacy is not stored (ρ = 1; W * 1 == W exactly, same additions).
# `_wρ_fast` keeps the @fastmath product of the undelayed kernels (same instructions as the inlined `W[s] * ρ[s]` before).
@inline _wρ(W, ρ::AbstractVector, s) = @inbounds W[s] * ρ[s]
@inline _wρ(W, ::Nothing, s) = @inbounds W[s]
@inline _wρ_fast(W, ρ::AbstractVector, s) = @inbounds @fastmath W[s] * ρ[s]
@inline _wρ_fast(W, ::Nothing, s) = @inbounds W[s]
# delay (steps) of synapse `s`: one value for all synapses, or one per synapse
@inline _dstep(d::Integer, s) = d
@inline _dstep(d::AbstractVector, s) = @inbounds d[s]
_efficacy_or_nothing(c) = isempty(c.ρ) ? nothing : c.ρ

_chunkptr(c::SpikingSynapse) = c.chunkptr
presynaptic_spikes(c::SpikingSynapse) = spike_list!(c.pre_spikes, c.fireJ)
postsynaptic_spikes(c::SpikingSynapse) = spike_list!(c.post_spikes, c.fireI)
prepare_spikes!(c::SpikingSynapse) = (_arm!(c.pre_spikes); _arm!(c.post_spikes))
release_spikes!(c::SpikingSynapse) = (_disarm!(c.pre_spikes); _disarm!(c.post_spikes))



@doc raw"""
    forward!(c::SpikingSynapse, param::SpikingSynapseDelayParameter, dt, T)

Delayed transmission. For each presynaptic spike, add the increment ``W_{ij}\rho_{ij}`` to the
ring buffer of target ``i`` at the arrival step (`delaysteps[s]` steps after the current one,
delays rounded up to whole steps, see `SpikingSynapseDelayParameter`); then add to `c.g[i]` the
increments that arrive in the current step and clear them.
With a delay of `d` steps stored once the arrival column is computed once per step; without LTP and STP the spike list emitted
`d` steps ago is scattered into `c.g` instead (`W` read at arrival).
"""
function forward!(c::SpikingSynapse, param::SpikingSynapseDelayParameter, dt::AbstractFloat, T::Time)
    @unpack colptr, I, W, g = c
    Npost = length(c.fireI)
    _delay_steps!(param, dt, Npost)
    _use_spike_ring(c) && return _forward_spike_ring!(c, param, Npost)   # constant delay, no plasticity (delay_kernels.jl)
    _delay_buffer!(c, param, Npost)
    @unpack delaysteps, buffer = param
    nslots = size(buffer, 2)
    pos = param.pos[1]
    spikes = presynaptic_spikes(c)
    # the arrival pass over the Npost targets counts as Npost / 4 operations
    ptr = _syn_chunk_ptr(c.chunkptr, Npost, colptr, I, spikes, Npost >> 2)
    # function barrier on the efficacy (vector or `nothing`) and on the delays (one per synapse, or one for all: delay_kernels.jl)
    d = length(delaysteps) == 1 ? delaysteps[1] : delaysteps
    _forward_delay!(g, W, _efficacy_or_nothing(c), I, colptr, ptr, spikes, Npost, d, buffer, pos, nslots)
    param.pos[1] = pos + 1 == nslots ? 0 : pos + 1
end

function _forward_delay!(g, W, ρ, I, colptr, ptr, spikes, Npost, delaysteps, buffer, pos, nslots)
    if ptr === nothing
        @inbounds for j ∈ spikes # loop on presynaptic neurons that fired
            for s ∈ colptr[j]:(colptr[j+1]-1)
                slot = pos + _dstep(delaysteps, s)
                slot >= nslots && (slot -= nslots)
                buffer[I[s], slot+1] += _wρ(W, ρ, s)
            end
        end
        # arrivals of this step
        @inbounds @simd for i ∈ axes(buffer, 1)
            g[i] += buffer[i, pos+1]
            buffer[i, pos+1] = 0
        end
    else
        # chunk `ch` owns the rows `rng` of the ring buffer (its targets): it adds the increments of its sub-rows and then the arrivals of its targets
        foreach_chunk_c(Npost) do ch, rng
            _delay_chunk!(g, W, ρ, I, ptr, ch, rng, spikes, delaysteps, buffer, pos, nslots)
        end
    end
    return nothing
end

@inline function _delay_chunk!(g, W, ρ, I, ptr, ch, rng, spikes, delaysteps, buffer, pos, nslots)
    @inbounds for j ∈ spikes
        for s ∈ ptr[ch, j]:(ptr[ch+1, j]-1)
            slot = pos + _dstep(delaysteps, s)
            slot >= nslots && (slot -= nslots)
            buffer[I[s], slot+1] += _wρ(W, ρ, s)
        end
    end
    @inbounds @simd for i ∈ rng
        g[i] += buffer[i, pos+1]
        buffer[i, pos+1] = 0
    end
end


export SpikingSynapse, update_plasticity!, with_plasticity, without_plasticity


# precision of a synapse onto `post`: the element type of the variable it writes into; it must be that of the membrane
# potential vector it reads (one precision per target population)
function _synapse_float_type(post, g, v_post)
    FT = eltype(g)
    FT <: AbstractFloat || throw(ArgumentError("the target variable of $(nameof(typeof(post))) has element type $FT, not a floating-point type"))
    (isempty(v_post) || eltype(v_post) === FT) || throw(ArgumentError(
        "the target population $(nameof(typeof(post))) mixes precisions: the target variable is $FT but the membrane potential is $(eltype(v_post)). " *
        "Build the population with a single floating-point type (see `float_type`)."))
    return FT
end
