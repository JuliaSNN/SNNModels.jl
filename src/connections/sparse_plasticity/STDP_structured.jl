# Define the struct to hold synapse parameters for both Exponential and Mexican Hat STDP
# STDP Parameters Structure
abstract type STDPStructuredParameter <: STDPParameter end

"""
    STDPStructuredParameter <: STDPParameter

Abstract supertype of `STDPSymmetric` and `STDPAntiSymmetric`; their state is
`STDPStructuredVariables`.
"""
STDPStructuredParameter

@doc raw"""
    STDPSymmetric(; A_x = 3e-2, A_y = 3e-2, τ_x = 50ms, τ_y = 500ms, αpre = 0pF, αpost = 0pF,
                    Wmax = 30pF, Wmin = 0pF)

Symmetric STDP rule with a difference-of-exponentials kernel, as used for structured
inhibitory plasticity by Festa, Cusseddu and Gjorgjieva (2024).

# Equations
For one pair with ``Δt = t_{post} - t_{pre}``:
```math
Δw(Δt) = \frac{A_x}{2τ_x} e^{-|Δt|/τ_x} - \frac{A_y}{2τ_y} e^{-|Δt|/τ_y},
```
whose integral over ``Δt`` is ``A_x - A_y`` (zero with the defaults). With ``τ_x < τ_y``
the kernel potentiates near-coincident spikes and depresses spikes farther apart. In
addition every presynaptic spike adds `αpre` and every postsynaptic spike `αpost`.

Implementation: each neuron has two traces (time constants ``τ_x`` and ``τ_y``) that jump
by 1 at a spike:
- presynaptic spike of j, each target i:
  ``w_{ij} \mathrel{+}= α_{pre} + \frac{A_x}{2τ_x} o_{x,i} - \frac{A_y}{2τ_y} o_{y,i}``
- postsynaptic spike of i, each source j:
  ``w_{ij} \mathrel{+}= α_{post} + \frac{A_x}{2τ_x} r_{x,j} - \frac{A_y}{2τ_y} r_{y,j}``

# Integration
`plasticity!` (only under `train!`): (1) pre-spike pass, (2) post-spike pass, both reading
the traces before this step's spikes are added; (3) forward-Euler decay of all traces (factor
`1 - dt/τ`), then +1 for the neurons that fired (the traces are stored lazily, one scale per trace type and
no pass over the neurons: the stored vectors are not the values, read them with `trace_values`); (4) clamp to `[Wmin, Wmax]` of the touched weights (all
weights once at the first step).

# Fields
- `A_x::FT = 3e-2`: amplitude of the narrow (potentiating) kernel.
- `A_y::FT = 3e-2`: amplitude of the wide (depressing) kernel.
- `τ_x::FT = 50ms`, `τ_y::FT = 500ms`: kernel time constants.
- `αpre::FT = 0pF`, `αpost::FT = 0pF`: constant change at each pre / post spike.
- `Wmax::FT = 30pF`, `Wmin::FT = 0pF`: weight bounds.

# References
Festa, D., Cusseddu, C. & Gjorgjieva, J., "Structured stabilization in recurrent neural
circuits through inhibitory synaptic plasticity" (2024), as cited in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 1.0),
                         LTPParam = SNN.STDPSymmetric(A_x = 1e-2, A_y = 1e-2))
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```
"""
STDPSymmetric
@snn_kw struct STDPSymmetric{FT = Float32} <: STDPStructuredParameter
    A_x::FT = 3e-2    # LTP learning rate (inhibitory synapses)
    A_y::FT = 3e-2    # LTD learning rate (inhibitory synapses)
    τ_x::FT = 50ms       # Time constant for pre-synaptic spike trace
    τ_y::FT = 500ms      # Time constant for post-synaptic spike trace
    αpre::FT = 0.0pF
    αpost::FT = 0.0pF
    Wmax::FT = 30.0pF   # Max weight
    Wmin::FT = 0.0pF    # Min weight (negative for inhibition)
end

@doc raw"""
    STDPAntiSymmetric(; A_y = 3e-2, A_x = 3e-2, τ_x = 50ms, τ_y = 50ms, αpre = 0pF, αpost = 0pF,
                        Wmax = 30pF, Wmin = 0pF)

Antisymmetric (Hebbian, pair-based) STDP rule with normalised exponential lobes, as used by
Festa, Cusseddu and Gjorgjieva (2024).

# Equations
For one pair with ``Δt = t_{post} - t_{pre}``:
```math
Δw(Δt) = \begin{cases} \dfrac{A_x}{τ_x} e^{-Δt/τ_x} & Δt > 0 \\[1ex]
                      -\dfrac{A_y}{τ_y} e^{Δt/τ_y} & Δt < 0 \end{cases}
```
with integral ``A_x - A_y`` (zero with the defaults); in addition every presynaptic spike adds
`αpre` and every postsynaptic spike `αpost`.

Implementation: a presynaptic trace ``r_{x,j}`` (``τ_x``) and a postsynaptic trace
``o_{y,i}`` (``τ_y``), both jumping by 1 at a spike:
- presynaptic spike of j, each target i: ``w_{ij} \mathrel{+}= α_{pre} - \frac{A_y}{τ_y} o_{y,i}``
- postsynaptic spike of i, each source j: ``w_{ij} \mathrel{+}= α_{post} + \frac{A_x}{τ_x} r_{x,j}``

# Integration
As for [`STDPSymmetric`](@ref): pre pass, post pass, Euler decay and +1 of the traces,
clamp of the touched weights. Only under `train!`.

# Fields
- `A_y::FT = 3e-2`: depression amplitude (post-before-pre).
- `A_x::FT = 3e-2`: potentiation amplitude (pre-before-post).
- `τ_x::FT = 50ms`, `τ_y::FT = 50ms`: time constants of the pre and post traces.
- `αpre::FT = 0pF`, `αpost::FT = 0pF`: constant change at each pre / post spike.
- `Wmax::FT = 30pF`, `Wmin::FT = 0pF`: weight bounds.

# References
Festa, D., Cusseddu, C. & Gjorgjieva, J., "Structured stabilization in recurrent neural
circuits through inhibitory synaptic plasticity" (2024), as cited in the code.
"""
STDPAntiSymmetric

@snn_kw struct STDPAntiSymmetric{FT = Float32} <: STDPStructuredParameter
    A_y::FT = 3e-2     # LTD learning rate (inhibitory synapses)
    A_x::FT = 3e-2    # LTP learning rate (inhibitory synapses)
    τ_x::FT = 50ms       # Time constant for pre-synaptic spike trace
    τ_y::FT = 50ms      # Time constant for post-synaptic spike trace
    αpre::FT = 0.0pF
    αpost::FT = 0.0pF
    Wmax::FT = 30.0pF   # Max weight
    Wmin::FT = 0.0pF    # Min weight (negative for inhibition)
end


"""
    STDPStructuredVariables(; Npre, Npost, to_x = zeros(Npost), to_y = zeros(Npost),
                            tr_x = zeros(Npre), tr_y = zeros(Npre), initialized = [false],
                            active = [true])

Traces of `STDPSymmetric` / `STDPAntiSymmetric`: `to_x`, `to_y` are postsynaptic traces
(length `Npost`, time constants `τ_x`, `τ_y`), `tr_x`, `tr_y` presynaptic traces (length
`Npre`). `STDPAntiSymmetric` uses only `tr_x` and `to_y`. `initialized` marks the one-off
clamp of all weights; `active` enables the rule.

!!! warning "The stored traces are not the trace values"
    The Euler decay is lazy: the value of `to_x` is `to_x * scale[1]`, of `to_y` `to_y * scale[2]`,
    of `tr_x` `tr_x * scale[3]`, of `tr_y` `tr_y * scale[4]`; `decay` caches `dt`, `τ_x`, `τ_y` and their
    factors `1 - dt/τ`. Read values with `trace_values(syn, :to_x)`, `sync_traces!(syn)` first, or the recorder.
"""
STDPStructuredVariables

@snn_kw struct STDPStructuredVariables{VFT = Vector{Float32},IT = Int,VDT = Vector{Float64}} <:
               LTPVariables
    Npost::IT                      # Number of post-synaptic neurons
    Npre::IT                       # Number of pre-synaptic neurons
    to_x::VFT = zeros(Npost)        # STORED post-synaptic spike trace, τ_x (value = to_x * scale[1])
    to_y::VFT = zeros(Npost)        # STORED post-synaptic spike trace, τ_y (times scale[2])
    tr_x::VFT = zeros(Npre)         # STORED pre-synaptic spike trace, τ_x (times scale[3])
    tr_y::VFT = zeros(Npre)         # STORED pre-synaptic spike trace, τ_y (times scale[4])
    initialized::VBT = [false]      # one-off clamp of all weights done
    active::VBT = [true]
    scale::VDT = ones(4)            # factors of the stored traces (to_x, to_y, tr_x, tr_y)
    decay::VDT = fill(NaN, 5)       # cached dt, τ_x, τ_y and their Euler factors 1 - dt/τ
end

function plasticityvariables(param::T, Npre, Npost) where {T<:STDPStructuredParameter}
    return STDPStructuredVariables(Npre = Npre, Npost = Npost, VFT = _vft(param))
end

const _STRUCTURED_SCALE = (to_x = 1, to_y = 2, tr_x = 3, tr_y = 4)

function sync_traces!(v::STDPStructuredVariables)
    for key in keys(_STRUCTURED_SCALE)
        k = _STRUCTURED_SCALE[key]
        v.scale[k] == 1.0 || (_rescale!(getfield(v, key), eltype(getfield(v, key))(v.scale[k])); v.scale[k] = 1.0)
    end
    return v
end

function trace_values(v::STDPStructuredVariables, key::Symbol)
    x = getfield(v, key)
    k = get(_STRUCTURED_SCALE, key, 0)
    return (k == 0 || v.scale[k] == 1.0) ? x : x .* eltype(x)(v.scale[k])
end

lazy_variables(::STDPStructuredVariables) = (:to_x, :to_y, :tr_x, :tr_y)

current_value(v::STDPStructuredVariables, param, key::Symbol, t::Real) = trace_values(v, key)

# Euler traces (x <- x (1 - dt/τ), then +1 for the spikes of the step, AFTER the weight passes: the passes read the
# traces of the previous step, spikes included). Stored lazily, with the Euler factor `1 - dt/τ` as the per-step
# factor of the scale. Order inside one trace type: decay first (`_advance_scale!`), then the increment
# `1/scale` (value `x f + 1`), unlike the exponential pair rules.
@inline function _structured_prepare!(plasticity::STDPStructuredVariables, dt::AbstractFloat, τ_x, τ_y)
    # a new dt or τ: bring the stored traces to their values before the factors change
    _decay_stale(plasticity.decay, dt, τ_x, τ_y) && sync_traces!(plasticity)
    return _euler_factors!(plasticity.decay, dt, τ_x, τ_y)
end

# `traces`: tuple of (stored trace, index in `scale`, spike list, one-step factor)
@inline function _structured_step!(c, plasticity::STDPStructuredVariables, f_pre, f_post, traces::Tuple, Wmin, Wmax)
    @unpack rowptr, colptr, I, J, index, W = c
    @unpack initialized, scale = plasticity
    spikes_pre, spikes_post = presynaptic_spikes(c), postsynaptic_spikes(c)
    _initial_clamp!(W, initialized, Wmin, Wmax)
    # weight passes: traces of the previous step (value = stored * scale), no clamp between the passes
    _pre_spike_add!(f_pre, W, colptr, I, spikes_pre)
    _post_spike_add!(f_post, W, rowptr, index, J, spikes_post)
    # decay, then +1 for this step's spikes
    for (x, k, spikes, f) in traces
        _advance_scale!(scale, k, x, f)
        a = _write_inc(scale, k, x)
        @inbounds for n in spikes
            x[n] += a
        end
    end
    _clamp_touched!(W, colptr, rowptr, index, spikes_pre, spikes_post, Wmin, Wmax)
    return nothing
end

# SymmetricSTDP and AntiSymmetricSTDP
function plasticity!(
    c::PT,
    param::STDPAntiSymmetric,
    plasticity::STDPStructuredVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack tr_x, to_y, scale = plasticity
    @unpack A_x, A_y, τ_x, τ_y, Wmax, Wmin, αpre, αpost = param
    fx, fy = _structured_prepare!(plasticity, dt, τ_x, τ_y)
    s_toy, s_trx = _read_scale(scale, 2, to_y), _read_scale(scale, 3, tr_x)
    f_pre = (w, i, j) -> αpre - A_y / τ_y * (to_y[i] * s_toy)       # pre spike
    f_post = (w, i, j) -> αpost + A_x / τ_x * (tr_x[j] * s_trx)     # post spike
    sp, sq = presynaptic_spikes(c), postsynaptic_spikes(c)
    _structured_step!(c, plasticity, f_pre, f_post, ((tr_x, 3, sp, fx), (to_y, 2, sq, fy)), Wmin, Wmax)
end

function plasticity!(
    c::PT,
    param::STDPSymmetric,
    plasticity::STDPStructuredVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack to_x, tr_x, to_y, tr_y, scale = plasticity
    @unpack A_x, A_y, τ_x, τ_y, Wmax, Wmin, αpre, αpost = param
    fx, fy = _structured_prepare!(plasticity, dt, τ_x, τ_y)
    s_tox, s_toy, s_trx, s_try = _read_scale(scale, 1, to_x), _read_scale(scale, 2, to_y), _read_scale(scale, 3, tr_x), _read_scale(scale, 4, tr_y)
    f_pre = (w, i, j) -> αpre + (A_x / 2τ_x * (to_x[i] * s_tox) - A_y / 2τ_y * (to_y[i] * s_toy))      # pre spike
    f_post = (w, i, j) -> αpost + (A_x / 2τ_x * (tr_x[j] * s_trx) - A_y / 2τ_y * (tr_y[j] * s_try))    # post spike
    sp, sq = presynaptic_spikes(c), postsynaptic_spikes(c)
    _structured_step!(c, plasticity, f_pre, f_post,
                      ((tr_x, 3, sp, fx), (tr_y, 4, sp, fy), (to_x, 1, sq, fx), (to_y, 2, sq, fy)), Wmin, Wmax)
end

export STDPSymmetric, STDPAntiSymmetric
