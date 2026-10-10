
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

# Fields
- `delaytime::Vector{FT}`: delay of each synapse (ms), in the CSC order of `W`.
- `delaysteps::Vector{Int32}`: delay of each synapse in steps of the `dt` of the last `forward!`.
- `buffer::Matrix{FT}`: `Npost x (max(delaysteps) + 1)` increments ``W_{ij}\,\rho_{ij}``
  (evaluated at emission time) per target and arrival step, a ring indexed from `pos`.
- `dt::Vector{FT}`: the step `delaysteps` was computed for (`NaN` before the first `forward!`).
- `pos::Vector{Int}`: ring slot (0-based) of the current step.
"""
SpikingSynapseDelayParameter

@snn_kw mutable struct SpikingSynapseDelayParameter{VFT = Vector{Float32}, VIT = Vector{Int32}, MFT = Matrix{Float32}, VPT = Vector{Int}} <: AbstractSpikingSynapseParameter
    delaytime::VFT
    delaysteps::VIT = Int32[]
    buffer::MFT = zeros(eltype(delaytime), 0, 1)
    dt::VFT = [eltype(delaytime)(NaN)]
    pos::VPT = [0]
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
    size(param.buffer) == (Npost, nslots) || (param.buffer = zeros(eltype(param.buffer), Npost, nslots))
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
- `delay_dist`: optional `Distribution`; one delay (ms) per synapse is drawn from it.
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
- `rowptr`, `colptr`, `I`, `J`, `index`, `W`: double sparse storage returned by `dsparse`
  (CSC column pointers `colptr`, postsynaptic index `I` and presynaptic index `J` of every
  synapse, row pointers `rowptr` of the transposed matrix and the map `index` from row-major
  to CSC position); `W::Vector{FT}` holds the weights in CSC order (`FT` = precision of the synapse).
- `ρ::Vector{FT}`: short-term efficacy per synapse (initialised to 1).
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
number of threads. Synapses with less work than `SNNModels.SYN_MIN_OPS[]` operations per step run on the calling thread. The voltage-based vSTDP, CaRule and
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
    # get the sparse representation of the synaptic weight matrix
    rowptr, colptr, I, J, index, W = dsparse(w)

    # set the paramter for the synaptic plasticity
    LTPParam = convert_precision(LTPParam, FT)
    STPParam = convert_precision(STPParam, FT)
    LTPVars = plasticityvariables(LTPParam, pre.N, post.N)
    STPVars = plasticityvariables(STPParam, pre.N, post.N)

    # short term plasticity
    ρ = ones(FT, length(W))

    # Network targets

    if isnothing(delay_dist)
        param = SpikingSynapseParameter()
    else
        # delays have the precision of the synapse whatever the element type of `delay_dist`
        delaytime = FT.(rand(delay_dist, length(W)))
        param = SpikingSynapseDelayParameter(; delaytime, VFT = Vector{FT}, MFT = Matrix{FT})
    end
    return SpikingSynapse(;
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
end

# The rule and its state are type parameters of SpikingSynapse: only a rule of the same type can be set in place.
_same_plasticity_types(p, q, v, w) = typeof(p) === typeof(q) && typeof(v) === typeof(w)
_plasticity_type_error(kind) = ArgumentError(
    "the $kind rule of a SpikingSynapse is a type parameter: the new rule has a different type, so it cannot be set in place. " *
    "Use `with_plasticity` to build a synapse with the new rule and replace the old one in the model.")

"""
    with_plasticity(c::SpikingSynapse; LTP = nothing, STP = nothing)

New `SpikingSynapse` that shares all arrays (`W`, `ρ`, `g`, `fireI`, `fireJ`, the connectivity, ...)
with `c` but has the rule `LTP` and/or `STP`, with fresh state (`plasticityvariables`). Rules left to
`nothing` are kept with their state. Since the rule types are type parameters of `SpikingSynapse`,
this is how a rule of another type is installed; the caller replaces `c` by the result in its model.
`update_plasticity!` and `change_plasticity!` only accept a rule of the same type as the current one.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 50)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 2.0))
EE = SNN.SNNModels.with_plasticity(EE; LTP = SNN.STDPGerstner())
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
    return SpikingSynapse(; fields...)
end


@doc raw"""
    forward!(c::SpikingSynapse, param::SpikingSynapseParameter, dt, T)

For each presynaptic neuron ``j`` that fired, add ``W_{ij}\rho_{ij}`` to `c.g[i]` for all its
targets ``i`` (no delay).
"""
function forward!(c::SpikingSynapse, param::SpikingSynapseParameter, dt::AbstractFloat, T::Time)
    @unpack colptr, I, W, g, ρ = c
    spikes = presynaptic_spikes(c)
    Npost = length(c.fireI)
    ptr = _syn_chunk_ptr(c.chunkptr, Npost, colptr, I, spikes)
    if ptr === nothing
        @inbounds for j ∈ spikes # loop on presynaptic neurons that fired
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s] * ρ[s]
            end
        end
    else
        # partition by postsynaptic range: chunk `ch` handles the entries of every spiking column whose post lies in its range
        foreach_chunk_c(Npost) do ch, rng
            _forward_chunk!(g, W, ρ, I, ptr, ch, spikes)
        end
    end
end

@inline function _forward_chunk!(g, W, ρ, I, ptr, ch, spikes)
    @inbounds for j ∈ spikes
        @inbounds @fastmath @simd for s ∈ ptr[ch, j]:(ptr[ch+1, j]-1)
            g[I[s]] += W[s] * ρ[s]
        end
    end
end

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
"""
function forward!(c::SpikingSynapse, param::SpikingSynapseDelayParameter, dt::AbstractFloat, T::Time)
    @unpack colptr, I, W, g, ρ = c
    Npost = length(c.fireI)
    _delay_steps!(param, dt, Npost)
    @unpack delaysteps, buffer = param
    nslots = size(buffer, 2)
    pos = param.pos[1]
    spikes = presynaptic_spikes(c)
    # the arrival pass over the Npost targets counts as Npost / 4 operations
    ptr = _syn_chunk_ptr(c.chunkptr, Npost, colptr, I, spikes, Npost >> 2)
    if ptr === nothing
        @inbounds for j ∈ spikes # loop on presynaptic neurons that fired
            for s ∈ colptr[j]:(colptr[j+1]-1)
                slot = pos + delaysteps[s]
                slot >= nslots && (slot -= nslots)
                buffer[I[s], slot+1] += W[s] * ρ[s]
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
    param.pos[1] = pos + 1 == nslots ? 0 : pos + 1
end

@inline function _delay_chunk!(g, W, ρ, I, ptr, ch, rng, spikes, delaysteps, buffer, pos, nslots)
    @inbounds for j ∈ spikes
        for s ∈ ptr[ch, j]:(ptr[ch+1, j]-1)
            slot = pos + delaysteps[s]
            slot >= nslots && (slot -= nslots)
            buffer[I[s], slot+1] += W[s] * ρ[s]
        end
    end
    @inbounds @simd for i ∈ rng
        g[i] += buffer[i, pos+1]
        buffer[i, pos+1] = 0
    end
end


export SpikingSynapse, update_plasticity!, with_plasticity


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
