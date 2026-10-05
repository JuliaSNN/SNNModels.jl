
@doc raw"""
    SpikingSynapseParameter()

Parameter of a `SpikingSynapse` without transmission delays (no fields). Selects the
`forward!` method that adds ``W_{ij}\,\rho_{ij}`` to the target in the same step in which
the presynaptic neuron fires.
"""
struct SpikingSynapseParameter <: AbstractSpikingSynapseParameter end

@doc raw"""
    SpikingSynapseDelayParameter(; delaytime, spike_time, spike_w)

Parameter of a `SpikingSynapse` with per-synapse transmission delays (not exported; built by
the `SpikingSynapse` constructor when `delay_dist` is given).

# Fields
- `delaytime::Vector{Float32}`: delay of each synapse (ms), in the CSC order of `W`.
- `spike_time::Vector{Vector{Float32}}`: per postsynaptic neuron, sorted arrival times of the
  queued spikes (ms).
- `spike_w::Vector{Vector{Float32}}`: per postsynaptic neuron, the increments
  ``W_{ij}\,\rho_{ij}`` (evaluated at emission time) matching `spike_time`.
"""
SpikingSynapseDelayParameter

@snn_kw struct SpikingSynapseDelayParameter{VVFT =Vector{Vector{Float32}}, VFT = Vector{Float32}} <: AbstractSpikingSynapseParameter
    delaytime::VFT
    spike_time::VVFT
    spike_w::VVFT
end

@snn_kw mutable struct SpikingSynapse{
                VIT = Vector{Int32},
                VFT = Vector{Float32},
                SYNP <: AbstractSpikingSynapseParameter
                } <: AbstractSpikingSynapse
    id::String = randstring(12)
    name::String = "SpikingSynapse"
    param::SYNP = SpikingSynapseParameter()
    LTPParam::LTPParameter = NoLTP()
    STPParam::STPParameter = NoSTP()
    LTPVars::PlasticityVariables = NoPlasticityVariables()
    STPVars::PlasticityVariables = NoPlasticityVariables()
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
  to CSC position); `W::Vector{Float32}` holds the weights in CSC order.
- `ρ::Vector{Float32}`: short-term efficacy per synapse (initialised to 1).
- `fireI`, `fireJ`: references to `post.fire` and `pre.fire`.
- `g`, `v_post`: references to the target variable and to the membrane potential of the
  target compartment.
- `param`: `SpikingSynapseParameter` or `SpikingSynapseDelayParameter`.
- `LTPParam`, `STPParam`, `LTPVars`, `STPVars`: plasticity rules and their state.
- `targets::Dict`: ids of `pre`/`post`, target symbol, connection type.
- `records::Dict`: recorded variables.

# Notes
- All synaptic data (`W`, `ρ`, delays, delay queues) are `Float32` whatever the element
  type of `conn` or of `delay_dist`; conversion happens in the constructor.
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
    dt::Float32 = 0.125f0,
    LTPParam::LTPParameter = NoLTP(),
    STPParam::STPParameter = NoSTP(),
    name::String = "SpikingSynapse",
)

    # set the synaptic weight matrix
    w = sparse_matrix(output_N(pre), input_N(post), conn)
    # remove autapses if pre == post (structural removal, the matrix stays sparse)
    (pre == post) && remove_autapses!(w)
    # get the sparse representation of the synaptic weight matrix
    rowptr, colptr, I, J, index, W = dsparse(w)
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

    # set the paramter for the synaptic plasticity
    LTPVars = plasticityvariables(LTPParam, pre.N, post.N)
    STPVars = plasticityvariables(STPParam, pre.N, post.N)

    # short term plasticity
    ρ = ones(Float32, length(W))

    # Network targets

    if isnothing(delay_dist)
        param = SpikingSynapseParameter()
    else
        # delays are Float32 whatever the element type of `delay_dist`
        delaytime = Float32.(rand(delay_dist, length(W)))
        spike_time = [Float32[] for _ in 1:length(fireI)]
        spike_w = [Float32[] for _ in 1:length(fireI)]
        param = SpikingSynapseDelayParameter(;
            delaytime,
            spike_time,
            spike_w,
        )
    end
    return SpikingSynapse(;
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
        )   
end

"""
    update_plasticity!(c::SpikingSynapse; LTP = nothing, STP = nothing)

Replace the long-term (`LTP`) and/or short-term (`STP`) plasticity rule of `c` and
re-create the corresponding state with `plasticityvariables`. Arguments left to `nothing`
are not changed.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 50)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 2.0))
SNN.SNNModels.update_plasticity!(EE; LTP = SNN.STDPGerstner())
```
"""
function update_plasticity!(c::SpikingSynapse; LTP = nothing, STP = nothing)
    if !isnothing(LTP)
        c.LTPParam = LTP
        c.LTPVars = plasticityvariables(c.LTPParam, length(c.fireJ), length(c.fireI))
    end
    if !isnothing(STP)
        c.STPParam = STP
        c.STPVars = plasticityvariables(c.STPParam, length(c.fireJ), length(c.fireI))
    end
end


@doc raw"""
    forward!(c::SpikingSynapse, param::SpikingSynapseParameter, dt, T)

For each presynaptic neuron ``j`` that fired, add ``W_{ij}\rho_{ij}`` to `c.g[i]` for all its
targets ``i`` (no delay).
"""
function forward!(c::SpikingSynapse, param::SpikingSynapseParameter, dt::Float32, T::Time)
    @unpack colptr, I, W, fireJ, g, ρ = c
    @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s] * ρ[s]
            end
        end
    end
end



@doc raw"""
    forward!(c::SpikingSynapse, param::SpikingSynapseDelayParameter, dt, T)

Delayed transmission. For each presynaptic spike, insert the increment ``W_{ij}\rho_{ij}``
with arrival time ``t + d_{ij}`` into the sorted queue of target ``i``; then add to `c.g[i]`
every queued increment whose arrival time is ``\le t`` and remove it from the queue.
"""
function forward!(c::SpikingSynapse, param::SpikingSynapseDelayParameter, dt::Float32, T::Time)
    @unpack colptr, I, W, fireJ, fireI, g, ρ = c
    @unpack delaytime, spike_time, spike_w = param

    for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                i = I[s]
                times = spike_time[i]
                weights = spike_w[i]
                spike = get_time(T) + delaytime[s]
                first_spike_id = findlast(.<(spike), times)
                first_spike_id = first_spike_id === nothing ? 0 : first_spike_id
                insert!(times, first_spike_id+1, spike)
                insert!(weights, first_spike_id+1, W[s] * ρ[s])
            end
        end
    end
    @fastmath @inbounds @simd for i ∈ eachindex(fireI)
        if !isempty(spike_time[i])
            times = spike_time[i]
            weights = spike_w[i]
            while !isempty(times) && times[1] <= get_time(T)
                g[i] += weights[1]
                popfirst!(times)
                popfirst!(weights)
            end
        end
    end
end


export SpikingSynapse, SpikingSynapseDelay, update_plasticity!
