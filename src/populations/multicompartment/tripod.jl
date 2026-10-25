"""
    AbstractTripodScheme

Integration scheme of `Tripod`, the `scheme` field: `HeunScheme()` (default, Heun method on the full state) or
`PaperScheme()` (the split two-stage scheme and spike rules of the published Tripod code, see `TripodPaper`).
"""
abstract type AbstractTripodScheme end

"""
    HeunScheme()

Default integration scheme of `Tripod`: Heun (explicit trapezoidal) method on the whole state, see the Integration section
of `Tripod`.
"""
struct HeunScheme <: AbstractTripodScheme end

"""
    PaperScheme(; AP_spike = 21mV)

Integration scheme that reproduces the published Tripod code (`aquaresima/tripod_neuron`, commit 8698e1a,
`simulator/simulation.jl`) step by step; selected by `TripodPaper`. It differs from `HeunScheme` in the following (all of them
checked against that code, see `validation/tripod_audit` of the JuliaSNN publication repository):
- order of operations in a step: external currents added as an Euler kick `dt I / C` to the potentials; synapses advanced;
  then, per neuron, the spike test `V_s >= Vspike` on the potential at the start of the step (not on a prediction);
- free integration: the axial currents are computed once from the potentials at the start of the step and kept fixed in
  both stages; each dendrite takes a two-stage Heun step with them; `w_s` takes a forward-Euler step with the old `V_s`;
  then the soma takes a two-stage Heun step with the new `w_s` and the fixed axial currents; the exponential term uses
  `exp32o`; the somatic synaptic current is not clamped (the dendritic ones are, at ±1500 pA);
- spike: `V_s` is set to `AP_spike`, `w_s += b`, the dendrites are not integrated in the spike step;
- backpropagation and refractory period: the dendrites are integrated in full against the somatic potential stored from the
  previous step (so the clamp acts with one step of delay, as in the original), then `V_s` is set to `spike.AP_membrane`
  (backpropagation) or `adex.Vr` (refractory);
- `Vspike` is the threshold of the spike test (the original uses the exponential-term threshold, `adex.Vt`).
With `spike = PostSpike(up = 1ms, τabs = 2.1ms)` and `dt = 0.1 ms` the spike step, 9 backpropagation steps and 21 refractory steps
of the original are obtained. The scheme is tuned to `dt = 0.1 ms`; other steps give a warning (once).

# Field
- `AP_spike::Float32 = 21mV`: somatic potential on the spike step (the following backpropagation steps use `spike.AP_membrane`,
  20 mV in `TripodPaper`).
"""
Base.@kwdef struct PaperScheme <: AbstractTripodScheme
    AP_spike::Float32 = 21.0f0
end

@doc raw"""
    Tripod(; N = 100, param = TripodParameter(), adex = AdExParameter(),
             soma_syn = TripodSomaSynapse, dend_syn = TripodDendSynapse, spike = PostSpike(), ...)

Population of three-compartment neurons: an AdEx-like soma coupled to two passive dendrites
(`d1`, `d2`) through axial conductances. The dendritic geometry and cable properties are
given by `param` (a `DendNeuronParameter`), the somatic parameters by `adex`, the somatic and
dendritic synapses by `soma_syn` and `dend_syn` (any `AbstractSynapseParameter` with a
five-argument `synaptic_current!`, i.e. not `DeltaSynapse`).

Connections select the compartment with the fourth argument of `SpikingSynapse`:
`SpikingSynapse(pre, T, :glu, :d1; conn)`, with targets `:s`, `:d1`, `:d2`.

# Equations
Soma (AdEx-like, parameters from `adex`, an `AdExParameter`), for each dendrite ``k``:
```math
\begin{aligned}
C \frac{dV_s}{dt} &= g_L (E_L - V_s) + g_L \Delta_T\, e^{(V_s - \theta)/\Delta_T} - w_s
    - I_{syn,s} - \sum_k g_{ax,k}\,(V_s - V_{d,k}) + I \\
C_{d,k} \frac{dV_{d,k}}{dt} &= g_{m,k} (E_{L,k} - V_{d,k}) - I_{syn,d,k}
    + g_{ax,k}\,(V_s - V_{d,k}) + I_d \\
\tau_w \frac{dw_s}{dt} &= a (V_s - E_L) - w_s \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```
``I_{syn,s}`` and ``I_{syn,d,k}`` are computed by `synaptic_current!` of `soma_syn` and
`dend_syn` with the compartment potentials of the Heun stage being evaluated, and are clamped to
``\pm 1500`` pA (`Tripod`) or ``\pm 1000`` pA (`BallAndStick`). ``C_{d,k}``, ``g_{m,k}``,
``g_{ax,k}`` and the dendritic leak reversal ``E_{L,k}`` come from the `Dendrite` structs (see
`create_dendrite`); by default ``E_{L,k}`` is the somatic ``E_L`` (`adex.El`),
unless `param.El_dend` is set (a number, mV; it then stays fixed whatever `adex.El`). ``E_{L,k}`` is read at construction: changing `adex.El` of an existing
population moves the somatic leak and the adaptation coupling, not the dendritic leak (`d1.El`, `d2.El`; set those directly to change it). The exponential term is multiplied by ``g_L``, as in
the AdEx model (Brette and Gerstner 2005) and in the code of the published Tripod model (`equations.jl` of
`aquaresima/tripod_neuron`: `gl * (-v + Er + ΔT * exp(...))`). Eq. 1 of the bioRxiv preprint of that paper (Quaresima et al.
2023, J. Physiol.) prints the exponential term without ``g_L``, which is not the equation the code integrates.

Spike: when the predicted somatic potential ``V_s + dt\,\dot V_s`` reaches `Vspike`
(population field, default ``0`` mV, was ``-10`` mV before 2.1.0; `adex.Vt` is only the resting value of ``\theta``), the
neuron fires: ``V_s \leftarrow`` `AP_membrane`,
``w_s \leftarrow w_s + b``, ``\theta \leftarrow \theta + A_t``, and the refractory counter is set
to `n_up + n_abs` steps, with `n_up = max(1, round(up/dt))` and `n_abs = max(1, round(τabs/dt))`.
The soma is clamped at `AP_membrane` until the counter reaches `n_abs` (back-propagation
period; the spike step itself counts as one), then at `adex.Vr` for `n_abs` steps. In both periods
``w_s`` is not integrated, and the dendrites are integrated in full (leak, all synaptic currents including NMDA, axial
current, external current) with the same Heun scheme as in free integration, the soma being held at its clamp value in both
stages; the dendrites do not move on the spike step itself. (Up to SNNModels 2.0 the dendrites only relaxed axially towards
the clamped soma in these periods, without leak and synaptic currents, so that a somatic spike could end an NMDA plateau; see
the changelog of 2.1.0.) ``\theta`` relaxes to `adex.Vt` at every step (forward Euler). The published Tripod code
(`TripodNeuron.jl`) detects spikes differently: it fires when ``V_s \ge V_T`` on the potential at the start of the step, before
integrating, so that `Vspike = adex.Vt` does not reproduce it (this model fires about one step earlier). The rule, the
refractory sequence and the numerics of the published code are available as `scheme = PaperScheme()` (see `TripodPaper`).

# Integration
Heun (explicit trapezoidal) method on ``x = (V_s, V_{d,k}, w_s)``. Synaptic conductances are
advanced first, once per step, by `update_synapses!`. Then ``k_1 = f(x_n)`` is evaluated at the
current state (stored in `Δv_temp`) and ``k_2 = f(x_n + dt\,k_1)`` at the Euler-predicted state
(stored in `Δv`); every term of ``f``, including the synaptic currents, the axial currents and
the adaptation current in the somatic equation, is evaluated at the same stage state. The state
is advanced by ``x_{n+1} = x_n + \frac{dt}{2}(k_1 + k_2)``. The scheme is second-order accurate
between spikes. During the backpropagation and refractory periods the soma is set to its clamp value before the first stage
and does not move in the Euler prediction of the second, and the dendrites follow the same Heun update.
The synaptic currents are evaluated at the stage potentials (the Euler-predicted potentials at the second stage), and the
conductances are advanced with the `exp64` approximation of the exponential decay (see `ReceptorSynapse`).

# Fields
## Population info
- `id::String = randstring(12)`, `name::String = "Tripod"`, `N::IT = 100`, `records::Dict`.

## Parameters
- `param::DendNeuronParameter = TripodParameter()`: morphology (`ds`, `physiology`, diameter `d`).
- `adex::SOMAT = AdExParameter()`: somatic parameters; uses `C`, `gl`, `El`, `ΔT`, `Vt`
  (resting value of the threshold ``\theta``), `Vr`, `a`, `b`, `τw`.
- `soma_syn::SYNS = TripodSomaSynapse`, `dend_syn::SYND = TripodDendSynapse`: synapse models.
- `spike::PST = PostSpike()`: `AP_membrane`, `up`, `τabs`, `At`, `τA`.
- `d1::VDT`, `d2::VDT`: `Dendrite`s built with `create_dendrite(N, param.ds[k]; El = _dend_El(param, adex), FT = eltype(VFT), physiology = param.physiology, d = param.d)`, where `_dend_El(param, adex)` is `param.El_dend` if not `nothing`, else `adex.El`.
- `Vspike::Float32 = 0mV`: spike detection threshold on the predicted somatic potential (`Float32` in both precisions);
  with `PaperScheme` the threshold of the test on the potential at the start of the step.
- `scheme::AbstractTripodScheme = HeunScheme()`: integration scheme, `HeunScheme()` (this page) or `PaperScheme()` (the published
  code, see `TripodPaper`).

## State variables
- `v_s`, `v_d1`, `v_d2::VFT`: compartment potentials (mV), initialised uniformly in `[Vr, Vt]`.
  `VFT = Vector{Float64}` gives a `Float64` population (see `float_type`): `adex`, `spike`, the synapse models and the
  dendrites follow, with the decimal default values; `param` (morphology) and `Vspike` stay `Float32`.
- `w_s::VFT`: somatic adaptation current (pA), zeros.
- `I::VFT`: external current into the soma (pA); `I_d::VFT`: external current injected in
  each of the two dendrites (pA).
- `fire::VBT`: spike flags; `tabs::VFT`: refractory counters (steps); `θ::VFT`: dynamic
  threshold of the exponential term (mV), initialised to `Vt`.
- `spikes::SpikeIndex`: neurons that fired in the step (see `spike_index`).

## Synapses
- `synvars_s`, `synvars_d1`, `synvars_d2`: synaptic state variables (`synaptic_variables`).
- `receptors_s`, `receptors_d1`, `receptors_d2::NamedTuple`: input buffers (`synaptic_receptors`).

## Work arrays
- `Δv`, `Δv_temp::MFT` (`N x 4`): derivatives of `(v_s, v_d1, v_d2, w_s)` at the two Heun stages.
- `is::MFT` (`N x 3`): synaptic currents of soma, d1, d2; `ic::VFT` (length 2): unused, kept for compatibility;
  `v_pred::MFT` (`N x 3`): stage potentials of soma, d1, d2 (mV).

# References
Quaresima A. et al. (2023), "The Tripod neuron: a minimal structural reduction of the
dendritic tree", J. Physiol. (preprint bioRxiv 10.1101/2022.09.05.506197). Receptor parameters: see `TripodSomaSynapse`
and `TripodDendSynapse`; the configuration of the published code: `TripodPaper`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
T = SNN.Tripod(N = 10)
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
syn = SNN.SpikingSynapse(E, T, :glu, :d1; conn = (p = 0.2, μ = 2.0))
model = SNN.compose(; E, T, syn)
SNN.monitor!(T, [:v_s, :v_d1, :fire])
SNN.sim!(; model, duration = 200ms)
```
"""
Tripod
@snn_kw struct Tripod{
    VFT = Vector{Float32},
    MFT = Matrix{eltype(VFT)},
    VDT = Dendrite{VFT},
    SYNS<:AbstractSynapseParameter,
    SYND<:AbstractSynapseParameter,
    SYNSV<:AbstractSynapseVariable,
    SYNDV<:AbstractSynapseVariable,
    SOMAT<:AbstractGeneralizedIFParameter,
    PST<:AbstractSpikeParameter,
    RECTS<:NamedTuple,
    RECTD<:NamedTuple,
    SCH<:AbstractTripodScheme,
    IT = Int32,
} <: AbstractDendriteIF
    id::String = randstring(12)
    name::String = "Tripod"
    ## These are compulsory parameters
    N::IT = 100
    param::DendNeuronParameter = TripodParameter()
    adex::SOMAT = AdExParameter(FT = eltype(VFT))
    soma_syn::SYNS = convert_precision(TripodSomaSynapse, eltype(VFT))
    dend_syn::SYND = convert_precision(TripodDendSynapse, eltype(VFT))
    spike::PST = PostSpike(FT = eltype(VFT))
    Vspike::Float32 = 0mV
    scheme::SCH = HeunScheme()
    d1::VDT = create_dendrite(N, param.ds[1]; El = _dend_El(param, adex), FT = eltype(VFT), physiology = param.physiology, d = param.d)
    d2::VDT = create_dendrite(N, param.ds[2]; El = _dend_El(param, adex), FT = eltype(VFT), physiology = param.physiology, d = param.d)

    # Membrane potential and adaptation
    v_s::VFT = rand_value(N, adex.Vt, adex.Vr)
    w_s::VFT = zeros(N)
    v_d1::VFT = rand_value(N, adex.Vt, adex.Vr)
    v_d2::VFT = rand_value(N, adex.Vt, adex.Vr)
    I::VFT = zeros(N)
    I_d::VFT = zeros(N)

    # Synapses dendrites
    synvars_s::SYNSV = synaptic_variables(soma_syn, N, eltype(VFT))
    synvars_d1::SYNDV = synaptic_variables(dend_syn, N, eltype(VFT))
    synvars_d2::SYNDV = synaptic_variables(dend_syn, N, eltype(VFT))

    receptors_s::RECTS = synaptic_receptors(soma_syn, N, eltype(VFT))
    receptors_d1::RECTD = synaptic_receptors(dend_syn, N, eltype(VFT))
    receptors_d2::RECTD = synaptic_receptors(dend_syn, N, eltype(VFT))

    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    spikes::SpikeIndex = owned_spike_index(N)
    tabs::VFT = zeros(Int, N)
    θ::VFT = ones(N) * adex.Vt
    records::Dict = Dict()

    ## Temporary variables for integration
    Δv::MFT = zeros(N, 4)
    Δv_temp::MFT = zeros(N, 4)
    is::MFT = zeros(N, 3)
    ic::VFT = zeros(2)
    v_pred::MFT = zeros(N, 3)
end

"""
    integrate!(p::Tripod, param::DendNeuronParameter, dt::AbstractFloat)

Advance a `Tripod` population by `dt`: update the synapses of the three compartments, then
integrate the membrane equations with the Heun method and handle spikes and refractoriness
(see `Tripod`).
"""
function integrate!(p::Tripod, param::DendNeuronParameter, dt_::AbstractFloat)
    dt = _step_dt(p, dt_)   # in the precision of the population
    N = Int(p.N)
    @unpack spike, adex, soma_syn, dend_syn = p
    @unpack τabs, up = spike
    # refractory counters in steps; each period lasts at least one step
    n_abs = max(1, round(Int, τabs / dt))
    n_ref = max(1, round(Int, up / dt)) + n_abs
    _check_state_lengths(N, p.v_s, p.w_s, p.v_d1, p.v_d2, p.fire, p.θ, p.tabs, p.d1.gax, p.d1.C, p.d2.gax, p.d2.C)
    (size(p.Δv, 1) >= N && size(p.Δv, 2) >= 4 && size(p.Δv_temp, 1) >= N && size(p.Δv_temp, 2) >= 4) ||
        throw(DimensionMismatch("Δv and Δv_temp must have N rows and 4 columns"))
    if _chunk_synapse(soma_syn) && _chunk_synapse(dend_syn)
        # all stages are per-neuron with no shared scratch: one fork/join per population and step
        foreach_chunk_c(N) do c, rng
            _tripod_chunk!(p, param, dt, rng, n_abs, n_ref)
        end
    elseif !(p.scheme isa HeunScheme)
        throw(ArgumentError("Tripod with $(nameof(typeof(p.scheme))) needs receptor synapse models with chunk methods (ReceptorSynapse)"))
    else
        # synapse models without chunk methods: the stages one after the other over the whole population
        _tripod_serial!(p, param, dt, n_abs, n_ref)
    end
end

# one chunk: synapses, the two Heun stages and the state update of the neurons `rng`
_tripod_chunk!(p::Tripod, param::DendNeuronParameter, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int) =
    _tripod_chunk!(p.scheme, p, param, dt, rng, n_abs, n_ref)

function _tripod_chunk!(::HeunScheme, p::Tripod, param::DendNeuronParameter, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int)
    @unpack Δv, Δv_temp, fire = p
    @unpack soma_syn, dend_syn = p
    update_synapses!(p, soma_syn, p.receptors_s, p.synvars_s, dt, rng)
    update_synapses!(p, dend_syn, p.receptors_d1, p.synvars_d1, dt, rng)
    update_synapses!(p, dend_syn, p.receptors_d2, p.synvars_d2, dt, rng)
    @inbounds for k in 1:4
        @simd ivdep for i in rng
            Δv[i, k] = zero(eltype(Δv))
            Δv_temp[i, k] = zero(eltype(Δv_temp))
        end
    end
    @inbounds for i in rng
        fire[i] = false
    end
    _tripod_clamp_soma!(p, rng, n_abs)
    update_neuron!(p, param, Δv, dt, rng)
    @inbounds for k in 1:4
        @simd ivdep for i in rng
            Δv_temp[i, k] = Δv[i, k]
        end
    end
    _tripod_hold_soma!(p, rng, n_abs)
    update_neuron!(p, param, Δv, dt, rng)
    _tripod_state!(p, dt, rng, n_abs, n_ref)
end

# Neurons in the backpropagation or refractory period (counter after this step `tabs - 1 > 0`): the soma is held at its clamp value
# for the whole step, so that both Heun stages of the dendrites see it (the dendritic equations are integrated in full).
function _tripod_clamp_soma!(p::Tripod, rng::UnitRange{Int}, n_abs::Int)
    @unpack v_s, tabs, spike, adex = p
    AP, Vr = spike.AP_membrane, adex.Vr
    nabs_f = convert(eltype(tabs), n_abs)
    @inbounds @simd ivdep for i ∈ rng
        t1 = tabs[i] - 1
        v_s[i] = ifelse(t1 > nabs_f, AP, ifelse(t1 > 0, Vr, v_s[i]))
    end
end

# Between the two Heun stages: a clamped soma does not move in the Euler prediction.
function _tripod_hold_soma!(p::Tripod, rng::UnitRange{Int}, n_abs::Int)
    @unpack Δv, tabs = p
    @inbounds @simd ivdep for i ∈ rng
        Δv[i, 1] = ifelse(tabs[i] - 1 > 0, zero(eltype(Δv)), Δv[i, 1])
    end
end

# the same stages over the whole population, for synapse models that have no chunk methods
function _tripod_serial!(p::Tripod, param::DendNeuronParameter, dt::AbstractFloat, n_abs::Int, n_ref::Int)
    @unpack Δv, Δv_temp, fire = p
    @unpack soma_syn, dend_syn = p
    update_synapses!(p, soma_syn, p.receptors_s, p.synvars_s, dt)
    update_synapses!(p, dend_syn, p.receptors_d1, p.synvars_d1, dt)
    update_synapses!(p, dend_syn, p.receptors_d2, p.synvars_d2, dt)
    fill!(Δv, zero(eltype(Δv)))
    fill!(Δv_temp, zero(eltype(Δv_temp)))
    fill!(fire, false)
    rng = 1:Int(p.N)
    _tripod_clamp_soma!(p, rng, n_abs)
    update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    _tripod_hold_soma!(p, rng, n_abs)
    update_neuron!(p, param, Δv, dt)
    _tripod_state!(p, dt, rng, n_abs, n_ref)
end

# state update after the Heun stages (spike detection, refractoriness, clamping), neurons `rng`
function _tripod_state!(p::Tripod, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int)
    @unpack v_s, w_s, v_d1, v_d2, fire, θ, tabs, Δv, Δv_temp, d1, d2 = p
    @unpack spike, adex, Vspike = p
    @unpack AP_membrane, At, τA = spike
    @unpack Vr, Vt, b = adex
    nabs_f = convert(eltype(tabs), n_abs)    # the counters are floats: compare and store without mixing types
    FT = eltype(v_s)
    nref_f = convert(eltype(tabs), n_ref)
    # Branchless: every case (backpropagation, refractory, free integration, spike) is evaluated and the applicable one kept.
    # `ivdep`: for a chunk range not starting at 1, LLVM cannot prove that the store to `Δv[i, 1]` does not alias the
    # loads of the other columns and would not vectorise; the entries are distinct.
    @inbounds @simd ivdep for i ∈ rng
        t1 = tabs[i] - 1
        th1 = θ[i] + dt * (Vt - θ[i]) / τA
        bp = t1 > nabs_f                 # backpropagation period
        rf = !bp & (t1 > 0)            # absolute refractory period
        act = t1 <= 0                   # free integration (or spike)
        vs0 = v_s[i]
        vd1 = v_d1[i]
        vd2 = v_d2[i]
        ws0 = w_s[i]
        dv1 = Δv[i, 1]

        # bp or rf: the soma is clamped (the Heun stages already integrated the dendrites against the clamped soma)
        vsc = ifelse(bp, AP_membrane, Vr)

        # act: spike or Heun update
        spk = act & (vs0 + dv1 * dt >= Vspike)
        keep = act & !spk
        vs_h = convert(FT, vs0 + 0.5 * dt * (Δv_temp[i, 1] + dv1))   # the Float64 sum stored in the state precision
        vd1_h = convert(FT, vd1 + 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2]))
        vd2_h = convert(FT, vd2 + 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3]))
        ws_h = convert(FT, ws0 + 0.5 * dt * (Δv_temp[i, 4] + Δv[i, 4]))

        clamped = bp | rf
        fire[i] = spk
        Δv[i, 1] = ifelse(spk, AP_membrane - vs0, dv1)
        v_s[i] = ifelse(clamped, vsc, ifelse(spk, AP_membrane, ifelse(keep, vs_h, vs0)))
        # dendrites: Heun update, also while the soma is clamped; they hold on the spike step
        v_d1[i] = ifelse(spk, vd1, vd1_h)
        v_d2[i] = ifelse(spk, vd2, vd2_h)
        w_s[i] = ifelse(spk, ws0 + b, ifelse(keep, ws_h, ws0))
        θ[i] = ifelse(spk, th1 + At, th1)
        tabs[i] = ifelse(spk, nref_f, t1)
    end
end

@inline function update_neuron!(
    p::Tripod,
    param::DendNeuronParameter,
    Δv::Matrix{FT},
    dt::FT,
) where {FT<:AbstractFloat}
    update_neuron!(p, param, Δv, dt, 1:Int(p.N))
end

# One Heun stage for the neurons `rng`.
@inline function update_neuron!(
    p::Tripod,
    param::DendNeuronParameter,
    Δv::Matrix{FT},
    dt::FT,
    rng::UnitRange{Int},
) where {FT<:AbstractFloat}
    @unpack v_d1, v_d2, v_s, I_d, I, w_s, θ = p
    @unpack d1, d2 = p
    @unpack is, v_pred = p
    @unpack adex, soma_syn, dend_syn = p
    @unpack C, gl, El, ΔT, a, τw = adex
    @unpack synvars_s, synvars_d1, synvars_d2 = p

    # Heun stage: evaluate the derivatives at the predicted state x + dt * Δv
    # (Δv = 0 at the first stage, i.e. at the current state).
    N = p.N
    d1gax, d1gm, d1El, d1C = d1.gax, d1.gm, d1.El, d1.C
    d2gax, d2gm, d2El, d2C = d2.gax, d2.gm, d2.El, d2.C
    _check_state_lengths(N, v_s, v_d1, v_d2, w_s, θ, I, I_d, d1gax, d1gm, d1El, d1C, d2gax, d2gm, d2El, d2C)
    (size(Δv, 1) >= N && size(Δv, 2) >= 4 && size(is, 1) >= N && size(is, 2) >= 3 && size(v_pred, 1) >= N && size(v_pred, 2) >= 3) ||
        throw(DimensionMismatch("Δv, is and v_pred must have N rows (4, 3 and 3 columns)"))
    last(rng) <= N || throw(BoundsError(v_s, last(rng)))

    @inbounds @simd ivdep for i ∈ rng
        v_pred[i, 1] = v_s[i] + Δv[i, 1] * dt
        v_pred[i, 2] = v_d1[i] + Δv[i, 2] * dt
        v_pred[i, 3] = v_d2[i] + Δv[i, 3] * dt
    end

    @views _synaptic_current!(p, soma_syn, synvars_s, v_pred[:, 1], is[:, 1], rng)
    @views _synaptic_current!(p, dend_syn, synvars_d1, v_pred[:, 2], is[:, 2], rng)
    @views _synaptic_current!(p, dend_syn, synvars_d2, v_pred[:, 3], is[:, 3], rng)
    @inbounds for k in 1:3
        @simd ivdep for i in rng
            is[i, k] = clamp(is[i, k], FT(-1500), FT(1500))
        end
    end

    # The `muladd` order reproduces the earlier `@fastmath` loop on x86-64 with FMA. No `@fastmath`, so LLVM keeps this order
    # in the vector body and the scalar remainder: a neuron's result does not depend on its lane or chunk. `ivdep`: as in `_tripod_state!`.
    @inbounds @simd ivdep for i ∈ rng
        vs = v_pred[i, 1]
        vd1 = v_pred[i, 2]
        vd2 = v_pred[i, 3]
        ws = muladd(Δv[i, 4], dt, w_s[i])
        ic1 = (vs - vd1) * d1gax[i] # axial current soma -> d1
        ic2 = (vs - vd2) * d2gax[i] # axial current soma -> d2
        # dendrites: leak - synapses + axial + external
        Δv[i, 2] = ((muladd(d1El[i] - vd1, d1gm[i], -is[i, 2]) + ic1) + I_d[i]) / d1C[i]
        Δv[i, 3] = ((muladd(d2El[i] - vd2, d2gm[i], -is[i, 3]) + ic2) + I_d[i]) / d2C[i]
        # soma: gl * (El - vs + ΔT * exp((vs - θ) / ΔT)) + I - (adaptation + synapses + axial currents)
        e = exp256((vs - θ[i]) / ΔT)
        Δv[i, 1] = muladd(gl, muladd(ΔT, e, El - vs), I[i] - (((ws + is[i, 1]) + ic1) + ic2)) / C
        Δv[i, 4] = muladd(a, vs - El, -ws) / τw
    end
end

# one precision per population (see `IF`); `param` holds the morphology only (precision-agnostic)
function snn_kw_finalize(::Type{T}, f::NamedTuple) where {T<:Tripod}
    _check_precision("Tripod", _state_float(T, :v_s), "adex" => f.adex, "soma_syn" => f.soma_syn, "dend_syn" => f.dend_syn,
                     "spike" => f.spike)
    f.scheme isa PaperScheme && !(f.soma_syn isa ReceptorSynapse && f.dend_syn isa ReceptorSynapse) &&
        throw(ArgumentError("Tripod with PaperScheme needs `ReceptorSynapse` models for `soma_syn` and `dend_syn`"))
    return f
end

export Tripod
