@doc """
    STDPGerstner{FT = Float32}

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

Fields (all `Float32`; time in ms, weights in the units of `W`, e.g. pF):
- `A_post = -1e-4`: weight change per unit `x_post` at a presynaptic spike (post-before-pre); `< 0` is LTD.
- `A_pre = 1e-4`: weight change per unit `x_pre` at a postsynaptic spike (pre-before-post); `> 0` is LTP.
- `τpre = 20ms`, `τpost = 20ms`: trace time constants.
- `Wmax = 30pF`, `Wmin = 0pF`: weight bounds (use `Inf`/`-Inf` to disable).

Algorithm: event-driven, Auryn ordering (see `STDP_kernels.jl`). Per step: (1) pre spikes walk
their outgoing synapses, (2) post spikes walk their incoming synapses, (3) the traces of the
neurons that fired are incremented by 1, (4) all traces are multiplied by `exp(-dt/τ)`.
Same-step pre and post spikes do not interact; earlier history does. Only touched weights
are clamped (all weights once at the first step). Serial, no threading. Matches Brian2
(relative weight difference 2e-6) and Auryn (2e-7) and the analytic kernel above.

Plasticity runs only under `train!`; `sim!` leaves the weights untouched.

!!! warning "Behaviour change in SNNModels 1.9"
    Up to SNNModels 1.8 the traces were incremented by `A_pre`/`A_post` and multiplied by
    them again, so the effective amplitudes were ``A^2`` and the sign was lost (a negative
    `A_post` potentiated). Now the amplitude is applied once and the default `A_post` is
    `-1e-4` (LTD). Parameter sets tuned against the old version must be rescaled: an old
    `A = 5e-2` corresponds to a new amplitude of `2.5e-3`.

# Example
```julia
rule = SNN.STDPGerstner(A_pre = 1e-2, A_post = -1.05e-2, τpre = 16.8ms, τpost = 33.7ms,
                        Wmin = 0, Wmax = 50)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 10.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 10s)   # not sim!
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
    STDPConfavreux2025{FT = Float32}

Pair-based STDP with rate terms, event-driven (same traces and ordering as
`STDPGerstner`, traces increment by 1):

- presynaptic spike:  ``w \\mathrel{+}= η (κ\\, x_{post} + α)``
- postsynaptic spike: ``w \\mathrel{+}= η (γ\\, x_{pre} + β)``

then clamping to `[Wmin, Wmax]`.

Fields: `η = 0.01` (learning rate), `α = 0`, `β = 0` (rate terms applied at every
presynaptic / postsynaptic spike respectively), `κ = 1`, `γ = 1` (weights of the
pre-post and post-pre trace terms), `τpre = τpost = 20ms`, `Wmin = 0pF`, `Wmax = 30pF`.
Unlike `STDPGerstner` the sign of the trace terms is carried by `κ`, `γ`, `α`, `β`, not by a
separate amplitude, and `η` multiplies both. Applied only under `train!`.
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
    STDPMexicanHat{FT = Float32}

STDP with a Mexican-hat kernel whose integral is zero:
``Δw = A\\, (1 - x)\\, e^{-x/\\sqrt{2}}`` with ``x = \\log(x_{pre}/x_{post})^2``, where
`x_pre` and `x_post` are the pre- and postsynaptic traces (time constant `τ`).

Fields: `A = 1e-1` (amplitude), `τ = 20ms`, `Wmax = 30pF`, `Wmin = 0pF`.

Pre spikes walk their outgoing synapses and post spikes their incoming ones (event-driven);
only touched weights are clamped. The traces keep their Euler integration, with the spike
added before the weight update, and the rule is otherwise unchanged by the event-driven
rewrite. Applied only under `train!`.
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
    STDPVariables

Per-neuron state of the pair-based STDP rules.

- `tpre`, `tpost`: presynaptic / postsynaptic traces. For the event-driven rules
  (`STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent`) they hold the current
  trace value, decayed by `exp(-dt/τ)` every step; for `STDPMexicanHat` they are the
  Euler-integrated traces of that rule.
- `last_pre`, `last_post`: time of the last spike (informational, not used in updates).
- `initialized`: set after the one-off clamp of all weights on the first step.
"""
STDPVariables

@snn_kw struct STDPVariables{VFT = Vector{Float32},IT = Int} <: LTPVariables
    Npost::IT                                   # Number of post-synaptic neurons
    Npre::IT                                    # Number of pre-synaptic neurons
    tpre::VFT = zeros(Float32, Npre)            # Pre-synaptic spike trace
    tpost::VFT = zeros(Float32, Npost)          # Post-synaptic spike trace
    last_pre::VFT = zeros(Float32, Npre)        # Last pre-synaptic spike time
    last_post::VFT = zeros(Float32, Npost)      # Last post-synaptic spike time
    initialized::VBT = [false]                  # one-off clamp of all weights done
    active::VBT = [true]
end

# Function to initialize plasticity variables
function plasticityvariables(param::T, Npre, Npost) where {T<:STDPParameter}
    return STDPVariables(Npre = Npre, Npost = Npost)
end

##

"""
    _pair_stdp!(c, variables, f_pre, f_post, τpre, τpost, Wmin, Wmax, dt, T)

Event-driven pair STDP step shared by the trace rules (see STDP_kernels.jl for the
ordering convention): pre-spike pass with `f_pre`, post-spike pass with `f_post`,
then trace increment by 1 and exact decay.
"""
@inline function _pair_stdp!(c, variables::STDPVariables, f_pre::F1, f_post::F2,
                             τpre, τpost, Wmin, Wmax, dt::Float32, T::Time) where {F1,F2}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI = c
    @unpack tpre, tpost, last_pre, last_post, initialized = variables
    t = get_time(T)
    _initial_clamp!(W, initialized, Wmin, Wmax)
    # 1-2. weight updates, traces read before this step's spikes are added
    _pre_spike_pass!(f_pre, W, colptr, I, fireJ, Wmin, Wmax)
    _post_spike_pass!(f_post, W, rowptr, index, J, fireI, Wmin, Wmax)
    # 3. trace increment for the spikes of this step
    _spike_increment!(tpre, last_pre, fireJ, t)
    _spike_increment!(tpost, last_post, fireI, t)
    # 4. exact decay over one step
    _decay!(tpre, Float32(exp(-dt / τpre)))
    _decay!(tpost, Float32(exp(-dt / τpost)))
    return nothing
end

# Event-driven additive pair STDP (Gerstner 1996 / Auryn STDPConnection)
function plasticity!(
    c::PT,
    param::STDPGerstner,
    variables::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack A_pre, A_post, τpre, τpost, Wmax, Wmin = param
    @unpack tpre, tpost = variables
    # pre spike of j onto post i: post-before-pre term, A_post * x_post[i]
    f_pre = (w, i, j) -> A_post * tpost[i]
    # post spike of i from pre j: pre-before-post term, A_pre * x_pre[j]
    f_post = (w, i, j) -> A_pre * tpre[j]
    _pair_stdp!(c, variables, f_pre, f_post, τpre, τpost, Wmin, Wmax, dt, T)
end

# Event-driven pair STDP with rate terms (Confavreux et al. 2025)
function plasticity!(
    c::PT,
    param::STDPConfavreux2025,
    variables::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack η, α, β, κ, γ, τpre, τpost, Wmin, Wmax = param
    @unpack tpre, tpost = variables
    f_pre = (w, i, j) -> η * (κ * tpost[i] + α)   # pre spike: pre-post term
    f_post = (w, i, j) -> η * (γ * tpre[j] + β)   # post spike: post-pre term
    _pair_stdp!(c, variables, f_pre, f_post, τpre, τpost, Wmin, Wmax, dt, T)
end

function MexicanHat(x::Float32)
    r = (1 - x) * exp(-x / sqrt(2.0f0))
    return isnan(r) ? 0.0f0 : r
end

# STDPMexicanHat keeps its Euler-integrated traces (incremented before the weight
# update, as in its original definition); only the weight passes are event-driven:
# pre spikes walk their outgoing synapses, post spikes their incoming synapses, and
# only touched synapses are clamped, after both passes.
function plasticity!(
    c::PT,
    param::STDPMexicanHat,
    plasticity::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI = c
    @unpack tpre, tpost, initialized = plasticity
    @unpack A, τ, Wmax, Wmin = param

    _initial_clamp!(W, initialized, Wmin, Wmax)
    @inbounds @fastmath begin
        # traces: Euler decay, then +1 for this step's spikes
        @simd for i in eachindex(fireI)
            tpost[i] += dt * (-tpost[i]) / τ
            tpost[i] += fireI[i]
        end
        @simd for j in eachindex(fireJ)
            tpre[j] += dt * (-tpre[j]) / τ
            tpre[j] += fireJ[j]
        end
        # pre spikes: outgoing synapses
        for j in eachindex(fireJ)
            fireJ[j] || continue
            for s = colptr[j]:(colptr[j+1]-1)
                i = I[s]
                if abs(tpost[i] * tpre[j]) > 0.0f0
                    W[s] += A * MexicanHat((log(tpre[j] / tpost[i]))^2)
                end
            end
        end
        # post spikes: incoming synapses
        for i in eachindex(fireI)
            fireI[i] || continue
            for st = rowptr[i]:(rowptr[i+1]-1)
                s = index[st]
                j = J[s]
                if abs(tpost[i] * tpre[j]) > 0.0f0
                    W[s] += A * MexicanHat(log(tpre[j] / tpost[i])^2)
                end
            end
        end
    end
    _clamp_touched!(W, colptr, rowptr, index, fireJ, fireI, Wmin, Wmax)
end

# Export the relevant functions and structs
export STDPVariables, plasticityvariables, plasticity!, STDPMexicanHat, STDPGerstner, STDPConfavreux2025
