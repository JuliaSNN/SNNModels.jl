@doc """
    STDPGerstner(; A_post = -1e-4, A_pre = 1e-4, τpre = 20ms, τpost = 20ms, Wmax = 30pF, Wmin = 0pF)

Additive pair-based STDP with all-to-all spike interaction, event-driven
(Gerstner, W., Kempter, R., van Hemmen, J. L., & Wagner, H. (1996). A neuronal
learning rule for sub-millisecond temporal coding. Nature, 383(6595), 76–78.
https://doi.org/10.1038/383076a0; implementation as Auryn `STDPConnection`).

Traces ``x_{pre}`` (time constant `τpre`) and ``x_{post}`` (`τpost`) jump by 1 at
each spike and decay exponentially. Weight updates, with the traces read before
this step's spikes are added:

- presynaptic spike of j, for each target i:  ``w_{ij} \\mathrel{+}= A_{post}\\, x_{post,i}``
- postsynaptic spike of i, for each source j: ``w_{ij} \\mathrel{+}= A_{pre}\\, x_{pre,j}``

followed by clamping to `[Wmin, Wmax]`. For a single pair with
``Δt = t_{post} - t_{pre}``: ``Δw = A_{pre} e^{-Δt/τ_{pre}}`` if ``Δt > 0`` and
``Δw = A_{post} e^{Δt/τ_{post}}`` if ``Δt < 0`` (0 if both spikes fall in the same step).

Sign convention: the amplitudes are signed and used as given. `A_pre > 0` gives
potentiation for pre-before-post, `A_post < 0` gives depression for
post-before-pre (classical Hebbian STDP, the default). Any sign combination is
allowed (e.g. anti-Hebbian with `A_pre < 0 < A_post`).

Fields (type `FT`, `Float32` by default; time in ms, weights in the units of `W`, e.g. pF):
- `A_post = -1e-4`: weight change per unit `x_post` at a presynaptic spike (post-before-pre); `< 0` is LTD.
- `A_pre = 1e-4`: weight change per unit `x_pre` at a postsynaptic spike (pre-before-post); `> 0` is LTP.
- `τpre = 20ms`, `τpost = 20ms`: trace time constants.
- `Wmax = 30pF`, `Wmin = 0pF`: weight bounds (use `Inf`/`-Inf` to disable).
Algorithm: event-driven, Auryn ordering (see `STDP_kernels.jl`). Per step: (1) pre spikes walk
their outgoing synapses, (2) post spikes walk their incoming synapses, (3) the traces of the
neurons that fired are incremented by 1, (4) the traces decay by `exp(-dt/τ)`, lazily (see
`STDPVariables`: the stored `tpre`, `tpost` are not the trace values).
Same-step pre and post spikes do not interact; earlier history does. Only touched weights
are clamped (all weights once at the first step).

Plasticity runs only under `train!`; `sim!` leaves the weights untouched.

!!! warning "Behaviour change in SNNModels 1.8.2"
    Up to SNNModels 1.8 the traces were incremented by `A_pre`/`A_post` and multiplied by
    them again, so the effective amplitudes were ``A^2`` and the sign was lost (a negative
    `A_post` potentiated). Now the amplitude is applied once and the default `A_post` is
    `-1e-4` (LTD). Parameter sets tuned against the old version must be rescaled: an old
    `A = 5e-2` corresponds to a new amplitude of `2.5e-3`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.STDPGerstner(A_pre = 1e-2, A_post = -1.05e-2, τpre = 16.8ms, τpost = 33.7ms,
                        Wmin = 0, Wmax = 50)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 10.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 1s)   # not sim!
```
"""
STDPGerstner

@snn_kw struct STDPGerstner{FT = Float32} <: STDPParameter
    A_post::FT = -10e-5pA / mV        # amplitude at a pre spike (post-before-pre); < 0: LTD
    A_pre::FT = 10e-5pA / (mV * mV)   # amplitude at a post spike (pre-before-post); > 0: LTP
    τpre::FT = 20ms                   # Time constant for pre-synaptic spike trace
    τpost::FT = 20ms                  # Time constant for post-synaptic spike trace
    Wmax::FT = 30.0pF                 # Max weight
    Wmin::FT = 0.0pF                  # Min weight (negative for inhibition)
end

@doc """
    STDPConfavreux2025(; η = 0.01, α = 0, β = 0, κ = 1, γ = 1, τpre = 20ms, τpost = 20ms,
                         Wmin = 0pF, Wmax = 30pF)

Pair-based STDP with rate terms, event-driven (same traces and ordering as
`STDPGerstner`, traces increment by 1 and are read before this step's spikes are added):

- presynaptic spike of j, each target i:  ``w_{ij} \\mathrel{+}= η (κ\\, x_{post,i} + α)``
- postsynaptic spike of i, each source j: ``w_{ij} \\mathrel{+}= η (γ\\, x_{pre,j} + β)``

then clamping to `[Wmin, Wmax]`. For one pair with ``Δt = t_{post} - t_{pre}``, the trace
terms give ``Δw = η γ e^{-Δt/τ_{pre}}`` for ``Δt > 0`` and ``Δw = η κ e^{Δt/τ_{post}}`` for
``Δt < 0``; every presynaptic spike adds ``η α`` and every postsynaptic spike ``η β``.

Fields: `η = 0.01` (learning rate), `α = 0`, `β = 0` (rate terms applied at every
presynaptic / postsynaptic spike respectively), `κ = 1` (weight of the post-before-pre term,
read at a presynaptic spike), `γ = 1` (weight of the pre-before-post term, read at a
postsynaptic spike), `τpre = τpost = 20ms`, `Wmin = 0pF`, `Wmax = 30pF`.
Unlike `STDPGerstner` the sign of the trace terms is carried by `κ`, `γ`, `α`, `β`, not by a
separate amplitude, and `η` multiplies both. With the defaults both pairings potentiate.
Applied only under `train!`. State: `STDPVariables`.

Reference: named "Confavreux et al. 2025" in the code; the full reference is not given in the
code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.STDPConfavreux2025(η = 1e-3, κ = -1.0, γ = 1.0, α = -0.1)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 1.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```
"""
STDPConfavreux2025

@snn_kw struct STDPConfavreux2025{FT = Float32} <: STDPParameter
    η::FT = 0.01
    α::FT = 0 ## baseline rate dependency post
    β::FT = 0 ## baseline rate dependency pre
    κ::FT = 1.0f0 # stdp pre->post
    γ::FT = 1.0f0 # stdp post->pre
    τpre::FT = 20ms            
    τpost::FT = 20ms      
    Wmin::FT = 0.0pF
    Wmax::FT = 30.0pF
end

@doc """
    STDPMexicanHat(; A = 0.1, τ = 20ms, Wmax = 30pF, Wmin = 0pF)

STDP with a Mexican-hat kernel:
``Δw = A\\, (1 - z)\\, e^{-z/\\sqrt{2}}`` with ``z = \\left[\\ln(x_{pre}/x_{post})\\right]^2``, where
`x_pre` and `x_post` are the pre- and postsynaptic traces (same time constant `τ`). For a
single pair, ``\\ln(x_{pre}/x_{post}) = -Δt/τ`` up to the Euler discretisation of the traces, so
``z = (Δt/τ)^2`` and the kernel is the symmetric "Mexican hat" ``(1 - z) e^{-z/\\sqrt{2}}``
in ``Δt``: potentiation for ``|Δt| < τ``, depression beyond. The update is applied at every
pre- and every postsynaptic spike of a synapse whose traces are both non-zero (a result
`NaN` is replaced by 0).

Fields: `A = 10e-2pA / mV` (= 0.1, amplitude), `τ = 20ms` (trace time constant),
`Wmax = 30pF`, `Wmin = 0pF`. State: `STDPVariables`. Reference not given in the code.

Pre spikes walk their outgoing synapses and post spikes their incoming ones (event-driven);
only touched weights are clamped. The traces are Euler-integrated (per-step factor `1 - dt/τ`)
and stored lazily like those of `STDPGerstner`; the spikes of the step are added before the
weight update, so same-step pre and post spikes interact. Applied only under `train!`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 1.0),
                         LTPParam = SNN.STDPMexicanHat(A = 1e-3))
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```
"""
STDPMexicanHat

@snn_kw struct STDPMexicanHat{FT = Float32} <: STDPParameter
    A::FT = 10e-2pA / mV    # LTD learning rate (inhibitory synapses)
    τ::FT = 20ms                    # Time constant for pre-synaptic spike trace
    Wmax::FT = 30.0pF                # Max weight
    Wmin::FT = 0.0pF               # Min weight (negative for inhibition)
end

## Common variables for STDP rules
@doc """
    STDPVariables(; Npre, Npost, tpre = zeros(Npre), tpost = zeros(Npost),
                  last_pre = zeros(Npre), last_post = zeros(Npost), initialized = [false],
                  active = [true], scale = [1.0, 1.0], decay = fill(NaN, 5))

Per-neuron state of the pair-based STDP rules.

!!! warning "The stored traces are not the trace values"
    The traces decay lazily (see `STDP_kernels.jl`): the trace of neuron `k` is `tpre[k] * scale[1]`
    (`tpost[k] * scale[2]`). Read values with [`trace_values`](@ref), or call [`sync_traces!`](@ref)
    first; `monitor!`, `getvariable` and `plasticity_value` already return values.

- `tpre`, `tpost`: stored presynaptic / postsynaptic traces.
- `last_pre`, `last_post`: time of the last spike (informational, not used in updates).
- `initialized`: set after the one-off clamp of all weights on the first step.
- `active`: the rule runs only if `any(active)` (see `set_LTP!`).
- `scale`: factors of the stored `tpre`, `tpost` (`Float64`, in `[1e-6, 1]`).
- `decay`: cache of `dt`, `τpre`, `τpost` and their one-step decay factors.

Created by `plasticityvariables(::STDPParameter, Npre, Npost)`.
"""
STDPVariables

@snn_kw struct STDPVariables{VFT = Vector{Float32},IT = Int,VDT = Vector{Float64}} <: LTPVariables
    Npost::IT                                   # Number of post-synaptic neurons
    Npre::IT                                    # Number of pre-synaptic neurons
    tpre::VFT = zeros(Float32, Npre)            # STORED pre-synaptic spike trace (value = tpre * scale[1])
    tpost::VFT = zeros(Float32, Npost)          # STORED post-synaptic spike trace (value = tpost * scale[2])
    last_pre::VFT = zeros(Float32, Npre)        # Last pre-synaptic spike time
    last_post::VFT = zeros(Float32, Npost)      # Last post-synaptic spike time
    initialized::VBT = [false]                  # one-off clamp of all weights done
    active::VBT = [true]
    scale::VDT = [1.0, 1.0]                     # factors of the stored traces (pre, post)
    decay::VDT = fill(NaN, 5)       # cached dt, τpre, τpost and their decay factors (_decay_factors!)
end

# Function to initialize plasticity variables
function plasticityvariables(param::T, Npre, Npost) where {T<:STDPParameter}
    return STDPVariables(Npre = Npre, Npost = Npost, VFT = _vft(param))
end

"""
    sync_traces!(vars)
    sync_traces!(c::AbstractSparseSynapse)

Make the lazily stored traces of the long-term rule hold their values (multiply by `scale`, reset it
to 1), so that `syn.LTPVars.tpre` etc. can be read directly. `sync_traces!(syn)` also writes deferred
weight updates into `syn.W` (`vSTDPParameter`). A no-op for variables without lazy traces. Returns its
argument.

The simulation does not need it. A sync rounds the traces once, so a run that syncs is not bit-identical
to one that does not; [`trace_values`](@ref) reads without changing the state. The event-driven Markram
`u`, `x` are not synced: read them with [`plasticity_value`](@ref).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 10, param = SNN.PoissonParameter(20Hz))
post = SNN.IF(N = 5)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 1.0, μ = 1.0), LTPParam = SNN.STDPGerstner())
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 100ms)
SNN.sync_traces!(syn)
syn.LTPVars.tpre   # presynaptic traces at the end of the run
```
"""
sync_traces!(v::PlasticityVariables) = v
sync_traces!(c::AbstractSparseSynapse) = (sync_weights!(c); sync_traces!(c.LTPVars); c)
function sync_traces!(v::STDPVariables)
    for (x, k) in ((v.tpre, 1), (v.tpost, 2))
        v.scale[k] == 1.0 || (_rescale!(x, eltype(x)(v.scale[k])); v.scale[k] = 1.0)
    end
    return v
end

"""
    trace_values(vars, key::Symbol)
    trace_values(c::AbstractSparseSynapse, key::Symbol)   # key = :W: the synaptic weights

Values of the trace `key` of the long-term rule (`key = :W`: the weights, as `plasticity_value(c, :W)`). For a
lazily decaying trace a new vector, state unchanged; any other field is returned as it is.
"""
trace_values(v, key::Symbol) = getfield(v, key)
trace_values(c::AbstractSparseSynapse, key::Symbol) = key === :W ? current_weights(c) : trace_values(c.LTPVars, key)
function trace_values(v::STDPVariables, key::Symbol)
    x = getfield(v, key)
    k = key === :tpre ? 1 : key === :tpost ? 2 : 0
    (k == 0 || v.scale[k] == 1.0) && return x
    return x .* eltype(x)(v.scale[k])
end

lazy_variables(::STDPVariables) = (:tpre, :tpost)

current_value(v::STDPVariables, param, key::Symbol, t::Real) = trace_values(v, key)

##

# Event-driven pair STDP step shared by the trace rules (ordering in STDP_kernels.jl): pre pass with `f_pre(w, x_post)`,
# post pass with `f_post(w, x_pre)`, then trace increment and decay.
@inline function _pair_stdp!(c, variables::STDPVariables, f_pre::F1, f_post::F2,
                             τpre, τpost, Wmin, Wmax, dt::AbstractFloat, T::Time) where {F1,F2}
    @unpack rowptr, colptr, I, J, index, W = c
    @unpack tpre, tpost, last_pre, last_post, initialized, scale, decay = variables
    t = get_time(T)
    # a new dt or τ: bring the stored traces to their values before the factors change
    _decay_stale(decay, dt, τpre, τpost) && sync_traces!(variables)
    fpre, fpost = _decay_factors!(decay, dt, τpre, τpost)
    spikes_pre, spikes_post = presynaptic_spikes(c), postsynaptic_spikes(c)
    _initial_clamp!(W, initialized, Wmin, Wmax)
    # value = stored * scale; the value at the start of this step is the one read below
    spre, spost = _read_scale(scale, 1, tpre), _read_scale(scale, 2, tpost)
    # 1-2. weight updates, traces read before this step's spikes are added
    _stdp_passes!(c, (w, i, j) -> f_pre(w, tpost[i] * spost), (w, i, j) -> f_post(w, tpre[j] * spre), spikes_pre, spikes_post, Wmin, Wmax)
    # 3. trace increment for the spikes of this step
    isempty(spikes_pre) || _spike_increment!(tpre, last_pre, spikes_pre, t, _write_inc(scale, 1, tpre))
    isempty(spikes_post) || _spike_increment!(tpost, last_post, spikes_post, t, _write_inc(scale, 2, tpost))
    # 4. exact decay over one step
    _advance_scale!(scale, 1, tpre, fpre)
    _advance_scale!(scale, 2, tpost, fpost)
    return nothing
end

# Event-driven additive pair STDP (Gerstner 1996 / Auryn STDPConnection)
function plasticity!(
    c::PT,
    param::STDPGerstner,
    variables::STDPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack A_pre, A_post, τpre, τpost, Wmax, Wmin = param
    # pre spike of j onto post i: post-before-pre term, A_post * x_post[i]
    f_pre = (w, xpost) -> A_post * xpost
    # post spike of i from pre j: pre-before-post term, A_pre * x_pre[j]
    f_post = (w, xpre) -> A_pre * xpre
    _pair_stdp!(c, variables, f_pre, f_post, τpre, τpost, Wmin, Wmax, dt, T)
end

# Event-driven pair STDP with rate terms (Confavreux et al. 2025)
function plasticity!(
    c::PT,
    param::STDPConfavreux2025,
    variables::STDPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack η, α, β, κ, γ, τpre, τpost, Wmin, Wmax = param
    f_pre = (w, xpost) -> η * (κ * xpost + α)   # pre spike: pre-post term
    f_post = (w, xpre) -> η * (γ * xpre + β)    # post spike: post-pre term
    _pair_stdp!(c, variables, f_pre, f_post, τpre, τpost, Wmin, Wmax, dt, T)
end

function MexicanHat(x::AbstractFloat)
    r = (1 - x) * exp(-x / sqrt(oftype(x, 2)))
    return isnan(r) ? zero(r) : r
end

# STDPMexicanHat: Euler decay and spike increment BEFORE the weight update; touched synapses clamped after both passes.
function plasticity!(
    c::PT,
    param::STDPMexicanHat,
    variables::STDPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W = c
    @unpack tpre, tpost, last_pre, last_post, initialized, scale, decay = variables
    @unpack A, τ, Wmax, Wmin = param
    t = get_time(T)
    _decay_stale(decay, dt, τ, τ) && sync_traces!(variables)
    fpre, fpost = _euler_factors!(decay, dt, τ, τ)
    spikes_pre, spikes_post = presynaptic_spikes(c), postsynaptic_spikes(c)
    _initial_clamp!(W, initialized, Wmin, Wmax)
    # traces: Euler decay, then +1 for this step's spikes
    _advance_scale!(scale, 1, tpre, fpre)
    _advance_scale!(scale, 2, tpost, fpost)
    isempty(spikes_pre) || _spike_increment!(tpre, last_pre, spikes_pre, t, _write_inc(scale, 1, tpre))
    isempty(spikes_post) || _spike_increment!(tpost, last_post, spikes_post, t, _write_inc(scale, 2, tpost))
    spre, spost = _read_scale(scale, 1, tpre), _read_scale(scale, 2, tpost)
    f = (w, i, j) -> begin
        xpre, xpost = tpre[j] * spre, tpost[i] * spost
        abs(xpost * xpre) > 0 ? A * MexicanHat(log(xpre / xpost)^2) : zero(xpre)
    end
    _add_passes!(c, f, f, spikes_pre, spikes_post, Wmin, Wmax)
    return nothing
end

# Export the relevant functions and structs
export STDPVariables, plasticityvariables, plasticity!, STDPMexicanHat, STDPGerstner, STDPConfavreux2025,
    sync_traces!, trace_values, plasticity_value
