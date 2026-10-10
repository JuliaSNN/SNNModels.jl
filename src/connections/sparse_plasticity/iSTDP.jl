"""
    iSTDPParameter <: STDPParameter

Abstract supertype of the inhibitory STDP rules of Vogels et al. (2011): `iSTDPRate`,
`iSTDPPotential` and `iSTDPTime`. All of them use `iSTDPVariables` and
Euler-integrated (not exact-decay) traces. See the notes on `iSTDPRate`.
"""
abstract type iSTDPParameter <: STDPParameter end

@doc raw"""
    iSTDPRate(; η = 0.01pA, r = 3Hz, τy = 50ms, Wmax = 243pF, Wmin = 0.01pF)

Inhibitory STDP with a target postsynaptic rate (Vogels, T. P., Sprekeler, H.,
Zenke, F., Clopath, C., & Gerstner, W. (2011). Inhibitory plasticity balances excitation
and inhibition in sensory pathways and memory networks. Science, 334(6062), 1569-1573.
https://doi.org/10.1126/science.1211095).

# Equations
Traces ``x_j`` (presynaptic) and ``y_i`` (postsynaptic) jump by 1 at a spike and decay with
time constant ``τ_y``. With ``α = 2 r τ_y``:
```math
Δw_{ij} = η\,(y_i - α) \;\text{at a presynaptic spike of } j, \qquad
Δw_{ij} = η\, x_j \;\text{at a postsynaptic spike of } i,
```
then ``w_{ij}`` is clamped to `[Wmin, Wmax]`. For uncorrelated pre- and postsynaptic Poisson
trains with rates ``r_j`` and ``r_i`` the mean drift is ``2 η τ_y r_j (r_i - r)``: inhibition
grows onto neurons that fire above the target rate `r` and decreases onto the others.

Fields (weights in the units of `W`, e.g. pF):
- `η = 0.01pA`: learning rate.
- `r = 3Hz`: target postsynaptic rate.
- `τy = 50ms`: time constant of the pre- and postsynaptic traces.
- `Wmax = 243pF`, `Wmin = 0.01pF`: weight bounds.

Algorithm (per step, `tpre`, `tpost` are Euler-integrated traces that jump by 1 at a spike;
`I[s]` is the postsynaptic and `J[s]` the presynaptic index of synapse `s`):
- presynaptic spike of `j`, for every outgoing synapse `s` (post `i = I[s]`):
  `W[s] += η * (tpost[i] - 2 r τy)`. Depression is the constant term `-2 η r τy`.
- postsynaptic spike of `i`, for every incoming synapse `s` (pre `j = J[s]`):
  `W[s] += η * tpre[j]`.
- every weight touched is clamped to `[Wmin, Wmax]`.

Unlike the event-driven pair rules (`STDPGerstner` etc.), traces are Euler-integrated (per-step factor
`1 - dt/τy`) and a spike is added to its trace before the postsynaptic pass of the same step reads `tpre`, so
a pre and a post spike in the same step do interact. The Euler decay is applied lazily: one scalar per trace
type is multiplied per step, no pass over the neurons (see `iSTDPVariables`: the stored traces are not the
values), so the step cost grows with the number of spikes only. The weights agree with the dense Euler
implementation to Float32 rounding.

Plasticity is applied only when the network is run with `train!`; `sim!` never calls it.

!!! warning "Bug fixed in SNNModels 1.8.2"
    In SNNModels 1.5.0 - 1.8.1 (and SpikingNeuralNetworks.jl from v1.0.0, commit 680a30c) the
    potentiation applied at a postsynaptic spike used a `@turbo` loop that reassigned its loop
    variable. LoopVectorization ignored the reassignment, so the update hit the synapses stored
    at CSC positions `rowptr[i]:rowptr[i+1]-1` (synapses onto unrelated postsynaptic neurons)
    instead of the incoming synapses of the spiking neuron `i`. The depression term was correct.
    Simulations that used `iSTDPRate` with those versions must be rerun.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 80)
I = SNN.Poisson(N = 20, param = SNN.PoissonParameter(20Hz))
IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 5.0), LTPParam = SNN.iSTDPRate(r = 5Hz))
SNN.train!(model = SNN.compose(; E, I, IE), duration = 500ms)   # plasticity needs train!
```
"""
iSTDPRate

@snn_kw struct iSTDPRate{FT = Float32} <: iSTDPParameter
    η::FT = 0.01pA
    r::FT = 3Hz
    τy::FT = 50ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end

@doc """
    iSTDPTime(; η = 0.01pA, τy = 50ms, Wmax = 243pF, Wmin = 0.01pF)

Parameter container (`η`, `τy`, `Wmax`, `Wmin`) of a time-based variant of the Vogels et al.
(2011) rule. The current code defines no `plasticity!` method for it, so it cannot be used as
`LTPParam` (the update was removed in commit e4ce94f, 2025-12-24; before that it contained the
same post-spike `@turbo` bug described in `iSTDPRate`). Use `iSTDPRate` or `iSTDPPotential`.
"""
iSTDPTime

@snn_kw struct iSTDPTime{FT = Float32} <: iSTDPParameter
    η::FT = 0.01pA
    τy::FT = 50ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end

@doc raw"""
    iSTDPPotential(; η = 0.001pA, v0 = -50mV, τy = 200ms, Wmax = 243pF, Wmin = 0.01pF)

Inhibitory STDP in which the postsynaptic trace follows the membrane potential instead of the
spike train (a variant of the Vogels et al. 2011 rule, with `tpost` low-pass filtering `v_post`).
The rule increases inhibition onto neurons whose filtered potential is above `v0` and
decreases it otherwise. Reference for the variant: not given in the code.

# Equations
```math
τ_y \frac{dx_j}{dt} = -x_j + τ_y \sum_k δ(t - t_j^k), \qquad
τ_y \frac{dy_i}{dt} = -(y_i - V_i),
```
```math
Δw_{ij} = η\,(y_i - v_0) \;\text{at a presynaptic spike of } j, \qquad
Δw_{ij} = η\, x_j \;\text{at a postsynaptic spike of } i,
```
with ``V_i`` the postsynaptic potential (`v_post`) and clamping to `[Wmin, Wmax]`.
At the first `plasticity!` call the trace `tpost` is set to the postsynaptic potential.
(Up to SNNModels 1.8.4 it started at 0 mV, so for the first few `τy` it was above `v0` whatever
the potential, which potentiated inhibition at the start of every simulation.)

Fields: `η = 0.001pA` (learning rate), `v0 = -50mV` (reference potential),
`τy = 200ms` (trace time constant), `Wmax = 243pF`, `Wmin = 0.01pF`.

Algorithm (per step; `tpre` is an Euler-integrated spike trace, `tpost` an Euler-integrated
trace of `v_post`):
- presynaptic spike of `j`, every outgoing synapse `s` (post `i = I[s]`):
  `W[s] += η * (tpost[i] - v0)`.
- postsynaptic spike of `i`, every incoming synapse `s` (pre `j = J[s]`):
  `W[s] += η * tpre[j]`.
- every weight touched is clamped to `[Wmin, Wmax]`.

Plasticity is applied only under `train!`. This rule was not affected by the `iSTDPRate` bug
fixed in SNNModels 1.8.2 (its postsynaptic loop never used `@turbo`); it now uses the same plain
`@simd` loops.
"""
iSTDPPotential

@snn_kw mutable struct iSTDPPotential{FT = Float32} <: iSTDPParameter
    η::FT = 0.001pA
    v0::FT = -50mV
    τy::FT = 200ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end


@doc """
    iSTDPVariables

State of the inhibitory STDP rules: `tpre` (length `Npre`) and `tpost` (length `Npost`)
traces, `last_spike` (unused by the update), the `active` flag and `initialized` (used by
`iSTDPPotential`: while false, the first `plasticity!` call sets `tpost` to `v_post`). Record the traces with
`LTPVars` as the variable set, e.g. `monitor!(syn, [:tpost], :LTPVars)`.

!!! warning "iSTDPRate: the stored traces are not the trace values"
    The traces of `iSTDPRate` are decayed lazily (Euler factor `1 - dt/τy` per step on the scalars `scale`,
    no pass over the neurons): the value of `tpre[k]` is `tpre[k] * scale[1]`, of `tpost[k]`
    `tpost[k] * scale[2]`. Read them with `trace_values(syn, :tpre)`, `sync_traces!(syn)` first, or the
    recorder (which stores values). `iSTDPPotential` keeps its dense traces (`scale` stays 1): `tpost` follows
    the membrane potential, which drives it at every step.
"""
iSTDPVariables

@snn_kw struct iSTDPVariables{VFT = Vector{Float32},IT = Int,VDT = Vector{Float64}} <: PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    tpost::VFT = zeros(Npost) # postsynaptic trace (spike trace, or filtered v_post for iSTDPPotential); iSTDPRate: STORED, value = tpost * scale[2]
    tpre::VFT = zeros(Npre) # presynaptic spike trace; iSTDPRate: STORED, value = tpre * scale[1]
    last_spike::VFT = zeros(Npost) # last spike time for each postsynaptic neuron
    active::VBT = [true]
    initialized::VBT = [false] # iSTDPPotential: tpost set to v_post at the first call
    scale::VDT = [1.0, 1.0]    # iSTDPRate: factors of the stored (tpre, tpost); stays 1 for iSTDPPotential
    decay::VDT = fill(NaN, 3)  # iSTDPRate: cached dt, τy and the Euler factor 1 - dt/τy
end

function plasticityvariables(param::T, Npre, Npost) where {T<:iSTDPParameter}
    return iSTDPVariables(Npre = Npre, Npost = Npost, VFT = _vft(param))
end

function sync_traces!(v::iSTDPVariables)
    for (x, k) in ((v.tpre, 1), (v.tpost, 2))
        v.scale[k] == 1.0 || (_rescale!(x, eltype(x)(v.scale[k])); v.scale[k] = 1.0)
    end
    return v
end

function trace_values(v::iSTDPVariables, key::Symbol)
    x = getfield(v, key)
    k = key === :tpre ? 1 : key === :tpost ? 2 : 0
    return (k == 0 || v.scale[k] == 1.0) ? x : x .* eltype(x)(v.scale[k])
end

lazy_variables(::iSTDPVariables) = (:tpre, :tpost)

# iSTDPRate stores its traces scaled; iSTDPPotential (dense: tpost is driven by v_post) as they are
current_value(v::iSTDPVariables, param::iSTDPRate, key::Symbol, t::Real) = trace_values(v, key)
current_value(v::iSTDPVariables, param, key::Symbol, t::Real) = getfield(v, key)

"""
    plasticity!(c::AbstractSparseSynapse, param::iSTDPRate, variables::iSTDPVariables, dt::AbstractFloat, T::Time)

One step of the inhibitory STDP of Vogels et al. (2011) with target rate `param.r`; modifies
`c.W` in place. Called by `train!` (never by `sim!`). See `iSTDPRate` for the update equations.

# Arguments
- `c`: the synapse; uses `rowptr`, `colptr`, `index`, `I`, `J`, `W`, `fireI` (post spikes), `fireJ` (pre spikes).
- `param::iSTDPRate`: `η`, `r`, `τy`, `Wmin`, `Wmax`.
- `variables::iSTDPVariables`: traces `tpre`, `tpost`, updated in place.
- `dt::AbstractFloat`: time step (ms); `T::Time`: simulation clock (unused).

# Algorithm
- The traces are decayed by the Euler factor `1 - dt/τy` (lazily, on a scalar); a spike adds 1.
- Presynaptic spike (reads `tpost` of the previous step): walk the outgoing synapses (`colptr`) and add `η (tpost[i] - 2 r τy)`.
- Postsynaptic spike `i` (reads `tpre` of this step, spikes included): walk the incoming synapses `k = rowptr[i]:rowptr[i+1]-1`, map to the
  CSC position `s = index[k]` and add `η tpre[J[s]]`. Weights are clamped to `[Wmin, Wmax]`.
"""
function plasticity!(
    c::AbstractSparseSynapse,
    param::iSTDPRate,
    plasticity::iSTDPVariables,
    dt::AbstractFloat,
    T::Time,
)
    @unpack rowptr, colptr, index, I, J, W = c
    @unpack η, r, τy, Wmax, Wmin = param
    @unpack tpre, tpost, scale, decay = plasticity
    # a new dt or τy: bring the stored traces to their values before the factor changes
    _decay_stale(decay, dt, τy) && sync_traces!(plasticity)
    (f,) = _euler_factors!(decay, dt, τy)
    spikes_pre, spikes_post = presynaptic_spikes(c), postsynaptic_spikes(c)
    α = 2 * r * τy
    # 1. presynaptic spikes read the postsynaptic trace of the previous step (value = stored * scale)
    spost = _read_scale(scale, 2, tpost)
    _pre_spike_pass!((w, i, j) -> η * (tpost[i] * spost - α), W, colptr, I, spikes_pre, Wmin, Wmax)
    # 2. Euler decay (one scalar per trace type), then +1 for this step's spikes
    _advance_scale!(scale, 1, tpre, f)
    _advance_scale!(scale, 2, tpost, f)
    @inbounds for j in spikes_pre
        tpre[j] += _write_inc(scale, 1, tpre)
    end
    @inbounds for i in spikes_post
        tpost[i] += _write_inc(scale, 2, tpost)
    end
    # 3. postsynaptic spikes read the presynaptic trace of this step (decayed, with this step's spikes added)
    spre = _read_scale(scale, 1, tpre)
    _post_spike_pass!((w, i, j) -> η * (tpre[j] * spre), W, rowptr, index, J, spikes_post, Wmin, Wmax)
    return nothing
end

"""
    plasticity!(c::AbstractSparseSynapse, param::iSTDPPotential, variables::iSTDPVariables, dt::AbstractFloat, T::Time)

One step of the membrane-potential variant of the inhibitory STDP of Vogels et al. (2011);
modifies `c.W` in place. Called by `train!` (never by `sim!`). See `iSTDPPotential` for the
equations.

# Algorithm
- Presynaptic pass, for every `j`: Euler decay `tpre[j] += -dt * tpre[j] / τy`; if `j` fired,
  `tpre[j] += 1` and every outgoing synapse gets `η (tpost[i] - v0)` (depression if the
  filtered potential is below `v0`), clamped to `[Wmin, Wmax]`.
- At the first call, `tpost .= v_post`.
- Postsynaptic pass, for every `i`: Euler step `tpost[i] += dt * (v_post[i] - tpost[i]) / τy`;
  if `i` fired, every incoming synapse gets `η tpre[j]`, clamped to `[Wmin, Wmax]`.
- `T` is unused.
"""
function plasticity!(
    c::AbstractSparseSynapse,
    param::iSTDPPotential,
    plasticity::iSTDPVariables,
    dt::AbstractFloat,
    T::Time,
)
    @unpack rowptr, colptr, index, I, J, W, v_post, fireI, fireJ, g = c
    @unpack η, v0, τy, Wmax, Wmin = param
    @unpack tpre, tpost, initialized = plasticity
    if !initialized[1]
        tpost .= v_post
        initialized[1] = true
    end

    # if pre-synaptic inhibitory neuron fires
    @fastmath @inbounds for j in eachindex(fireJ) # presynaptic indices j
        tpre[j] += dt * (-tpre[j]) / τy
        if fireJ[j] # presynaptic neuron
            tpre[j] += 1
            @simd for s = colptr[j]:(colptr[j+1]-1)
                W[s] = clamp(W[s] + η * (tpost[I[s]] - v0), Wmin, Wmax)
            end
        end
    end

    # if post-synaptic excitatory neuron fires
    @fastmath @inbounds for i in eachindex(fireI) # postsynaptic indices i
        # trace of the membrane potential
        tpost[i] += dt * -(tpost[i] - v_post[i]) / τy
        if fireI[i] # postsynaptic neuron
            @simd for k = rowptr[i]:(rowptr[i+1]-1)
                s = index[k]
                W[s] = clamp(W[s] + η * tpre[J[s]], Wmin, Wmax)
            end
        end
    end
end

export iSTDPRate,
    iSTDPTime, iSTDPPotential, iSTDPVariables, plasticityvariables, plasticity!
