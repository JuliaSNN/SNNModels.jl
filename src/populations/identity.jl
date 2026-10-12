"""
    IdentityParam <: AbstractPopulationParameter

Parameter type of `Identity` (no fields). `Population(IdentityParam(); kwargs...)` returns an
`Identity`.
"""
struct IdentityParam <: AbstractPopulationParameter end

"""
    Identity(; N = 100, name = "identity", kwargs...)

Relay population: each neuron emits a spike in every step in which it received positive input,
so that spikes arriving through a connection are relayed to downstream connections. Because
populations are integrated before connections are forwarded, input written in step ``t`` is
turned into a spike in step ``t + 1`` (one-step delay). Any target symbol used by a connection is mapped to the input field `g`.

# Integration
For each neuron, per step: `h += g`; `spikecount = g` if `g > 0` (else 0); `fire = g > 0`;
then `g = 0`.

# Sparse update
If every component that writes `g` (counted in `nwriters`) is a stimulus that logs the neurons it
writes (`SpikeTimeStimulus`, the Poisson stimuli; see `attach_driver!`), `integrate!` visits only those neurons and the
neurons that fired in the previous step, with the same result as the dense update, and feeds the
spike list `spikes` that its connections share. Any other writer, or more writes than neurons in a
step, makes the update dense. Do not write `g`, `fire` or `h` of such a population by hand; if a
connection was redirected to another target vector after construction, call [`release_input!`](@ref).

# Fields
- `name::String = "identity"`, `id::String = randstring(12)`, `param::IdentityParam`.
- `N::Int32 = 100`.
- `g::VFT`: input received in the current step (summed synaptic weights), reset
  each step. `VFT = Vector{Float64}` gives a `Float64` population (see `float_type`).
- `spikecount::VFT`: input of the last step for neurons that fired (0 otherwise).
- `h::VFT`: cumulative input since construction.
- `fire::Vector{Bool}`; `records::Dict`.
- `spikes::SpikeIndex`, `inbox::TouchList`, `nwriters`, `ndrivers`, `tracked`: state of the sparse update.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
P = SNN.Poisson(N = 10, param = SNN.PoissonParameter(50Hz))
Id = SNN.Identity(N = 10)
s = SNN.SpikingSynapse(P, Id, :g; conn = (μ = 1, p = 1.0))
model = SNN.compose(; P, Id, s)
SNN.monitor!(Id, [:fire])
SNN.sim!(model, 100ms)
```
"""
Identity
@snn_kw mutable struct Identity{VFT = Vector{Float32},IT = Int32} <: AbstractPopulation
    name::String = "identity"
    id::String = randstring(12)
    param::IdentityParam = IdentityParam()
    N::IT = 100
    g::VFT = zeros(N)
    spikecount::VFT = zeros(N)
    h::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    records::Dict = Dict()
    spikes::SpikeIndex = SpikeIndex(N)  # neurons with fire[i] (ascending) while tracked
    inbox::TouchList = TouchList(N)     # neurons written by list-aware stimuli since the last integrate!
    nwriters::Int = 0                   # components writing g (synaptic_target calls)
    ndrivers::Int = 0                   # of which list-aware stimuli (attach_driver!)
    tracked::Bool = true                # `spikes` lists exactly the true entries of `fire`
end

# the sparse update is valid while every writer of g logs its writes in `inbox`
@inline _list_driven(p::Identity) = p.ndrivers > 0 && p.ndrivers == p.nwriters

function integrate!(p::Identity, param::IdentityParam, dt_::AbstractFloat)
    dt = _step_dt(p, dt_)   # in the precision of the population
    driven = _list_driven(p)
    if driven && p.tracked && p.inbox.n <= length(p.inbox.idx)
        _integrate_sparse!(p)
    else
        _integrate_dense!(p)
        p.inbox.n = 0
        # a dense step leaves `fire` in a known state: re-derive the list if the sparse path can resume
        if driven
            p.spikes.n = _flags_to_list!(p.spikes.idx, p.fire)
        end
        p.tracked = driven
    end
    p.spikes.fed = driven
    return nothing
end

function _integrate_dense!(p::Identity)
    @unpack g, h, fire, spikecount = p
    n = length(g)
    z = zero(eltype(spikecount))
    (length(h) == n && length(fire) == n && length(spikecount) == n) ||
        throw(DimensionMismatch("Identity state vectors must have the same length"))
    # branchless: the neuron fires on any positive input, spikecount is that input
    foreach_chunk(n) do rng
        @inbounds @simd for i in rng
            gi = g[i]
            pos = gi > 0
            h[i] += gi
            fire[i] = pos
            spikecount[i] = ifelse(pos, gi, z)
            g[i] = 0
        end
    end
end

# The dense update restricted to the logged neurons (`inbox`) and those that fired last step. A neuron
# logged twice is processed once: its input is zeroed by the first visit.
function _integrate_sparse!(p::Identity)
    @unpack g, h, fire, spikecount, spikes, inbox = p
    idx = spikes.idx
    @inbounds for q in 1:spikes.n
        j = idx[q]
        fire[j] = false
        spikecount[j] = zero(eltype(spikecount))
    end
    n = 0
    @inbounds for q in 1:inbox.n
        i = inbox.idx[q]
        gi = g[i]
        gi == 0 && continue
        h[i] += gi
        g[i] = 0
        if gi > 0
            fire[i] = true
            spikecount[i] = gi
            n += 1
            idx[n] = i
        end
    end
    spikes.n = n
    inbox.n = 0
    # the list is ascending, as the one built by scanning `fire`
    @inbounds for q in 2:n
        if idx[q] < idx[q-1]
            sort!(view(idx, 1:n))
            break
        end
    end
    return nothing
end

attach_driver!(p::Identity) = (p.ndrivers += 1; p.inbox)

"""
    release_input!(p::Identity)

Declare that one component built with `p` as target no longer writes to `p.g` (e.g. the `g` of a
connection was replaced after construction), removing it from the writer count of the sparse update.
Calling it for a component that still writes to `p.g` makes `p` miss that input.
"""
release_input!(p::Identity) = (p.nwriters = max(p.nwriters - 1, 0); p)
spike_index(p::Identity) = p.spikes

function Population(::IdentityParam; kwargs...)
    return Identity(; kwargs...)
end

function synaptic_target(targets::Dict, post::Identity, sym::Symbol, target)
    g = getfield(post, :g)
    v_post = zeros(eltype(g), length(g))
    post.nwriters += 1   # every writer of g is counted, see attach_driver!
    push!(targets, :sym => :g)
    return g, v_post
end



export Identity, IdentityParam, release_input!
