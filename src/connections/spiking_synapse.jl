
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
the `SpikingSynapse` constructor when `delay_dist` is given). Each delay is rounded up to whole
steps, `ceil(delay / dt)`, at the first `forward!` and whenever `dt` changes; the increments
``W_{ij}\,\rho_{ij}`` (evaluated at emission) wait in a ring buffer indexed by arrival step. A delay
common to all synapses without LTP and STP delays the spike lists instead (`delay_kernels.jl`).

# Fields
- `delaytime::Vector{FT}`: delay of each synapse (ms) in the CSC order of `W`, or one value common to all synapses.
- `delaysteps::Vector{Int32}`: the same in steps of the `dt` of the last `forward!`.
- `buffer::Matrix{FT}`: `Npost x (max(delaysteps) + 1)` ring of pending increments, indexed from `pos`.
- `dt::Vector{FT}`: the step `delaysteps` was computed for (`NaN` before the first `forward!`).
- `pos::Vector{Int}`: ring slot (0-based) of the current step.
- `spike_ring::Vector{Vector{Int32}}`: spike lists of the last `d + 1` steps (spike-list path only).
- `drain::Vector{Int}`: steps left to deliver the buffer after a switch to the spike-list path.
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
        # 1e-4 steps of tolerance for the rounding error of the quotient
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

With `delay_dist`, every synapse has a fixed delay ``d_{ij}`` (ms), rounded up to whole steps; the
increment computed at emission is added to ``g_i`` at the arrival step (see `SpikingSynapseDelayParameter`).

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
- `rowptr`, `colptr`, `I`, `J`, `index`, `W`: double sparse storage as returned by `dsparse`
  (CSC column pointers `colptr`, postsynaptic index `I` and presynaptic index `J` of every
  synapse, row pointers `rowptr` of the transposed matrix and the map `index` from row-major
  to CSC position); `W::Vector{FT}` holds the weights in CSC order.
  `J` and `index` are built only when something reads them (a rule that `needs_transpose`, `presynaptic`,
  `presynaptic_idxs`, `SynapseNormalization`, `AggregateScaling`); otherwise empty. Call
  `SNNModels.ensure_transpose!(c)` before reading them in user code.
- `ρ::Vector{FT}`: short-term efficacy per synapse, stored only with a short-term rule; otherwise empty (ρ = 1).
  Read it with `SNNModels.efficacy(c)`; call `SNNModels.materialize_efficacy!(c)` before writing to it.
- `fireI`, `fireJ`: references to `post.fire` and `pre.fire`.
- `g`, `v_post`: references to the target variable and to the membrane potential of the
  target compartment.
- `param`: `SpikingSynapseParameter` or `SpikingSynapseDelayParameter`.
- `LTPParam`, `STPParam`, `LTPVars`, `STPVars`: plasticity rules and their state.
- `targets::Dict`: ids of `pre`/`post`, target symbol, connection type.
- `records::Dict`: recorded variables.
- `pre_spikes`, `post_spikes`: `SpikeIndex` lists of the neurons of `pre`/`post` that fired in the current step,
  shared by `forward!` and the event-driven rules (the population's own list when it keeps one).
- `chunkptr::ChunkPtr`: cached sub-row pointers of the threaded kernels.

# Threading
With `neurons_per_thread > 0` (see `set_neurons_per_thread!`) and more than one thread, `forward!` and the
event-driven STDP, iSTDP and STP updates are partitioned by postsynaptic range, with the chunks of the
postsynaptic population. Each target receives its additions in serial order, so results are bit-identical
for any number of threads. Synapses with fewer than `SNNModels.SYN_MIN_OPS[]` operations per step run serially;
vSTDP and `iSTDPPotential` are always serial.

# Notes
- Precision: the synapse has the floating-point type `FT` of the target population (`float_type(post)`). `W`, `ρ`,
  delays and the plasticity rules and state are converted to it (`convert_precision`; a `Float32` rule keeps
  `Float32` values). A target population whose target variable and membrane potential differ in precision is an
  `ArgumentError`.
- Weights of the event-driven `vSTDPParameter` are updated lazily: read them with `plasticity_value(syn, :W)`.
- If `pre == post`, autapses are removed structurally (no stored zero-weight self synapses
  that plasticity could grow). With `rule = :Fixed` this lowers the in-degree by one for the
  neurons that had drawn themselves.

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

    # the precision of the synapse is that of the variable it writes into
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
        # a delay common to all synapses is stored once (a new vector: `resize!` would keep the memory)
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

Whether the transposed index arrays `c.J`, `c.index` are present; build them in place if not.
Not thread-safe: call outside the threaded kernels.
"""
has_transpose(c::SpikingSynapse) = length(c.index) == length(c.W) && length(c.J) == length(c.W)

# whether a rule of `c` reads `J`/`index` (a constant per synapse type)
needs_transpose(c::SpikingSynapse) = needs_transpose(c.LTPParam) || needs_transpose(c.STPParam)
needs_transpose(c::AbstractConnection) = true
_has_rules(c::SpikingSynapse) = !(c.LTPParam isa NoLTP && c.STPParam isa NoSTP)

# Serial step guard of the rules: build `J`/`index` and `ρ` if a rule needs them and they are missing (e.g. a synapse
# loaded from a file or built with the keyword constructor). Must run before any threaded kernel.
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

`c.ρ` is stored only with a short-term rule; otherwise it is empty and ρ = 1. `efficacy(c)` returns `c.ρ` when
stored, else a new vector of ones (writes to it have no effect). `materialize_efficacy!(c)` stores ones if needed and
returns `c.ρ`, whose values `forward!` then uses.
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
rule `STP`, converted to the precision of `c`, with fresh state. A rule left to `nothing` is kept with its state;
`NoLTP()` / `NoSTP()` switch that part off; `without_plasticity(c)` switches both off. This is how a rule of another type
is installed (`update_plasticity!` and `change_plasticity!` accept only the current type); the caller replaces `c` by the
result in its model. `c` is not modified.

Switching on builds the arrays the new rules need (`J`/`index` for a rule that `needs_transpose`, `ρ` for a short-term
rule), reusing those of `c` when present. Switching off first writes back deferred weight updates (`sync_weights!`); the
result has the type, memory and per-step work of a synapse built without plasticity. With `STP = NoSTP()` the efficacy
is 1 again (to freeze `ρ` instead, keep the rule and call `set_STP!(c, false)`).

# What is shared
`W`, the connectivity, `g`, `v_post`, `fireI`, `fireJ`, `param` (delays and delay queue), `records`, `id`, `name`, the spike
lists, the chunk cache and any kept rule with its state are the same objects in `c` and in the result: use only one of the
two in a model, or `deepcopy(c)` first. A `SynapseNormalization` or `AggregateScaling` built on `c` keeps using `c`.

# Records
`:W`, `:ρ` and the population records continue across the switch. A rule variable continues if the new rule has it
(e.g. `:tpre` from `STDPGerstner` to `STDPWeightDependent`); otherwise its record stops (for `c` too, `records` is
shared) and stays readable. A stopped record is not reopened by a later switch.

# Replacing the synapse in a model
`model = compose(; pop = model.pop, syn = merge(model.syn, (; EE = EE2)), stim = model.stim, time = model.time)`, or
assign into a `Vector{AbstractConnection}`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100); E.I .= 300.0
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.5))
EE = SNN.with_plasticity(EE; LTP = SNN.STDPGerstner())
model = SNN.compose(; E, EE)
SNN.train!(; model, duration = 100ms)
EE = SNN.without_plasticity(model.syn.EE)                  # W frozen
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

# Stop sampling the recorded variables of `group` (:LTPVars or :STPVars) that the new rule state lacks; their samples stay readable.
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

# increment of synapse `s`: W ρ, or W when the efficacy is not stored (ρ = 1). `_wρ_fast` is the @fastmath variant of the
# undelayed kernels; the delayed kernels avoid @fastmath (see delay_kernels.jl).
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

Delayed transmission: add each increment ``W_{ij}\rho_{ij}`` to the ring buffer of target ``i`` at its arrival step,
then add the arrivals of the current step to `c.g` (see `SpikingSynapseDelayParameter`).
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
