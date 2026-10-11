# BallAndStick
@doc raw"""
    BallAndStick(; N = 100, param = BallAndStickParameter(), adex = AdExParameter(),
                   soma_syn = TripodSomaSynapse, dend_syn = TripodDendSynapse, spike = PostSpike(), ...)

Population of two-compartment neurons: an AdEx-like soma coupled to one passive dendrite
(`d`) through an axial conductance. Same equations and integration scheme as `Tripod`, with a
single dendrite. Connections select the compartment with the fourth argument of
`SpikingSynapse`: targets `:s` and `:d`.

# Equations
Soma (AdEx-like, parameters from `adex`, an `AdExParameter`), with one dendrite (``k = 1``):
```math
\begin{aligned}
C \frac{dV_s}{dt} &= g_L (E_L - V_s) + g_L \Delta_T\, e^{(V_s - \theta)/\Delta_T} - w_s
    - I_{syn,s} - \sum_k g_{ax,k}\,(V_s - V_{d,k}) + I_s \\
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
`create_dendrite`); by default ``E_{L,k}`` is the somatic ``E_L`` (`adex.El`). The exponential term is multiplied by ``g_L``, as in
the AdEx model (Brette and Gerstner 2005) and in the published Tripod model (Quaresima et al.
2023, Eq. 1, where ``g_L`` multiplies both the leak and the exponential term; the published code,
TripodNeuron.jl, computes `gl * (-v + Er + ΔT * exp((v - θ) / ΔT))`).

!!! note "Changed after SNNModels 1.8.4"
    Up to SNNModels 1.8.4 the exponential term was ``\Delta_T e^{(V_s-\theta)/\Delta_T}`` without
    ``g_L`` (a mV-valued term in a pA-valued equation), so the spike-initiation current was
    ``g_L / 1\,\mathrm{nS}`` times (40 times with the default `gl = 40nS`) smaller than in the
    published model.

Spike: when the predicted somatic potential ``V_s + dt\,\dot V_s`` reaches `Vspike`
(population field, default ``-10`` mV; `adex.Vt` is only the resting value of ``\theta``), the
neuron fires: ``V_s \leftarrow`` `AP_membrane`,
``w_s \leftarrow w_s + b``, ``\theta \leftarrow \theta + A_t``, and the refractory counter is set
to `n_up + n_abs` steps, with `n_up = max(1, round(up/dt))` and `n_abs = max(1, round(τabs/dt))`.
The soma is clamped at `AP_membrane` until the counter reaches `n_abs` (back-propagation
period; the spike step itself counts as one), then at `adex.Vr` for `n_abs` steps; in both periods the
dendrites only relax towards the soma through the axial term (forward Euler) and `w_s` is not
integrated. ``\theta`` relaxes to `adex.Vt` at every step (forward Euler). The published Tripod
code (TripodNeuron.jl) detects spikes at ``V_s \ge V_T``; `Vspike = adex.Vt` reproduces that
rule.

# Integration
Heun (explicit trapezoidal) method on ``x = (V_s, V_{d,k}, w_s)``. Synaptic conductances are
advanced first, once per step, by `update_synapses!`. Then ``k_1 = f(x_n)`` is evaluated at the
current state (stored in `Δv_temp`) and ``k_2 = f(x_n + dt\,k_1)`` at the Euler-predicted state
(stored in `Δv`); every term of ``f``, including the synaptic currents, the axial currents and
the adaptation current in the somatic equation, is evaluated at the same stage state. The state
is advanced by ``x_{n+1} = x_n + \frac{dt}{2}(k_1 + k_2)``. The scheme is second-order accurate
between spikes.

!!! note "Changed after SNNModels 1.8.4"
    Up to SNNModels 1.8.4 the refractory counter was `round((up + τabs)/dt)` and the
    back-propagation test `tabs > τabs/dt`: when `up + τabs` rounded to a value below
    `τabs/dt + 1` (e.g. `up = τabs = 0.1ms` at `dt = 0.125ms`) the reset to `Vr` was skipped and
    the neuron fired every second step (about 1 kHz). With `up` and `τabs` multiples of `dt`
    the result is unchanged.
    Up to SNNModels 1.8.4 the predicted adaptation state was `w_s + Δv` and `v_s + Δv`
    (without the factor `dt`), the first-stage adaptation derivative read the already updated
    somatic derivative, and the synaptic currents and `w_s` in the somatic equation were not
    evaluated at the predicted state. Results therefore depended on `dt`.

``I_s`` and ``I_d`` are the external currents `Is` and `Id` (pA).

!!! note "Changed after SNNModels 1.8.4"
    `Is` and `Id` were ignored up to SNNModels 1.8.4; the dendritic leak reversal was the
    somatic `adex.El` instead of `d.El` (same value by default).

# Fields
## Population info
- `name::String = "BallAndStick"`, `id::String = randstring(12)`, `N::IT = 100`, `records::Dict`.

## Parameters
- `param::DendNeuronParameter = BallAndStickParameter()`: morphology.
- `adex::SOMAT = AdExParameter()`: somatic parameters (see `Tripod`).
- `dend_syn::SYND = TripodDendSynapse`, `soma_syn::SYNS = TripodSomaSynapse`: synapse models.
- `spike::PST = PostSpike()`: spike shape and refractoriness.
- `d::VDT`: `Dendrite` built with `create_dendrite(N, param.ds[1]; El = adex.El, FT = eltype(VFT))`.
- `Vspike::Float32 = -10mV`: spike detection threshold on the predicted somatic potential (`Float32` in both precisions;
  compared after promotion).

## State variables
- `v_s`, `v_d::VFT`: somatic and dendritic potentials (mV), initialised uniformly in `[Vr, Vt]`.
  `VFT = Vector{Float32}` by default; `VFT = Vector{Float64}` gives a `Float64` population: `adex`, `spike`, the synapse models
  (`convert_precision` of the defaults), the dendrites, the synaptic variables, receptor buffers and work arrays follow (see
  `float_type`); `param` (morphology) and `Vspike` stay `Float32`.
- `w_s::VFT`: somatic adaptation current (pA).
- `Is`, `Id::VFT`: external currents into the soma and the dendrite (pA), zeros.
- `fire::VBT`, `tabs::VFT`, `θ::VFT`: spike flags, refractory counters (steps), dynamic threshold (mV).

## Synapses
- `synvars_s`, `synvars_d`: synaptic state variables; `receptors_s`, `receptors_d::NamedTuple`: input buffers.

## Work arrays
- `Δv`, `Δv_temp::MFT` (`N x 3`): derivatives of `(v_s, v_d, w_s)`; `is::MFT` (`N x 2`);
  `ic::VFT` (length 1): not written since 1.10, kept for compatibility; `v_pred::MFT` (`N x 2`): stage potentials of soma and dendrite.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
B = SNN.BallAndStick(N = 10)
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
syn = SNN.SpikingSynapse(E, B, :glu, :d; conn = (p = 0.2, μ = 2.0))
model = SNN.compose(; E, B, syn)
SNN.sim!(; model, duration = 200ms)
```
"""
BallAndStick
@snn_kw struct BallAndStick{
    VFT = Vector{Float32},
    MFT = Matrix{eltype(VFT)},
    VDT = Dendrite{VFT},
    SYND<:AbstractSynapseParameter,
    SYNS<:AbstractSynapseParameter,
    SYNDV<:AbstractSynapseVariable,
    SYNSV<:AbstractSynapseVariable,
    SOMAT<:AbstractGeneralizedIFParameter,
    PST<:AbstractSpikeParameter,
    RECT<:NamedTuple,
    IT = Int32,
} <: AbstractDendriteIF     ## These are compulsory parameters

    name::String = "BallAndStick"
    id::String = randstring(12)
    N::IT = 100
    param::DendNeuronParameter = BallAndStickParameter()
    adex::SOMAT = AdExParameter(FT = eltype(VFT))
    dend_syn::SYND = convert_precision(TripodDendSynapse, eltype(VFT))
    soma_syn::SYNS = convert_precision(TripodSomaSynapse, eltype(VFT))
    spike::PST = PostSpike(FT = eltype(VFT))
    Vspike::Float32 = -10mV

    # Membrane potential and adaptation
    d::VDT = create_dendrite(N, param.ds[1]; El = adex.El, FT = eltype(VFT))
    v_s::VFT = rand_value(N, adex.Vt, adex.Vr)
    w_s::VFT = zeros(N)
    v_d::VFT = rand_value(N, adex.Vt, adex.Vr)

    # Synapses
    synvars_s::SYNSV = synaptic_variables(soma_syn, N, eltype(VFT))
    synvars_d::SYNDV = synaptic_variables(dend_syn, N, eltype(VFT))

    ## Ext input
    Is::VFT = zeros(N)
    Id::VFT = zeros(N)

    # Receptors properties
    receptors_d::RECT = synaptic_receptors(dend_syn, N, eltype(VFT)) #! target
    receptors_s::RECT = synaptic_receptors(soma_syn, N, eltype(VFT)) #! target

    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
    tabs::VFT = zeros(Int, N)
    θ::VFT = ones(N) * adex.Vt
    records::Dict = Dict()

    ## Temporary variables for integration
    Δv::MFT = zeros(N, 3)
    Δv_temp::MFT = zeros(N, 3)
    is::MFT = zeros(N, 2)
    ic::VFT = zeros(1)
    v_pred::MFT = zeros(N, 2)
end

"""
    integrate!(p::BallAndStick, param::DendNeuronParameter, dt::AbstractFloat)

Advance a `BallAndStick` population by `dt` (synapses, Heun step of soma and dendrite, spikes
and refractoriness; see `BallAndStick`).
"""
function integrate!(p::BallAndStick, param::DendNeuronParameter, dt::AbstractFloat)
    N = Int(p.N)
    @unpack spike, soma_syn, dend_syn = p
    @unpack τabs, up = spike
    # refractory counters in steps; each period lasts at least one step
    n_abs = max(1, round(Int, τabs / dt))
    n_ref = max(1, round(Int, up / dt)) + n_abs
    _check_state_lengths(N, p.v_s, p.w_s, p.v_d, p.fire, p.θ, p.tabs, p.d.gax, p.d.C)
    (size(p.Δv, 1) >= N && size(p.Δv, 2) >= 3 && size(p.Δv_temp, 1) >= N && size(p.Δv_temp, 2) >= 3) ||
        throw(DimensionMismatch("Δv and Δv_temp must have N rows and 3 columns"))
    if _chunk_synapse(soma_syn) && _chunk_synapse(dend_syn)
        # one fork/join per population and step, no shared scratch (see `Tripod`)
        foreach_chunk_c(N) do c, rng
            _bas_chunk!(p, param, dt, rng, n_abs, n_ref)
        end
    else
        update_synapses!(p, soma_syn, p.receptors_s, p.synvars_s, dt)
        update_synapses!(p, dend_syn, p.receptors_d, p.synvars_d, dt)
        fill!(p.Δv, zero(eltype(p.Δv)))
        fill!(p.Δv_temp, zero(eltype(p.Δv_temp)))
        fill!(p.fire, false)
        update_neuron!(p, param, p.Δv, dt)
        p.Δv_temp .= p.Δv
        update_neuron!(p, param, p.Δv, dt)
        _bas_state!(p, dt, 1:N, n_abs, n_ref)
    end
end

function _bas_chunk!(p::BallAndStick, param::DendNeuronParameter, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int)
    @unpack Δv, Δv_temp, fire = p
    @unpack soma_syn, dend_syn = p
    update_synapses!(p, soma_syn, p.receptors_s, p.synvars_s, dt, rng)
    update_synapses!(p, dend_syn, p.receptors_d, p.synvars_d, dt, rng)
    @inbounds for k in 1:3
        @simd ivdep for i in rng
            Δv[i, k] = zero(eltype(Δv))
            Δv_temp[i, k] = zero(eltype(Δv_temp))
        end
    end
    @inbounds for i in rng
        fire[i] = false
    end
    update_neuron!(p, param, Δv, dt, rng)
    @inbounds for k in 1:3
        @simd ivdep for i in rng
            Δv_temp[i, k] = Δv[i, k]
        end
    end
    update_neuron!(p, param, Δv, dt, rng)
    _bas_state!(p, dt, rng, n_abs, n_ref)
end

function _bas_state!(p::BallAndStick, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int)
    @unpack v_s, w_s, v_d, fire, θ, tabs, Δv, Δv_temp, d = p
    @unpack spike, adex, Vspike = p
    @unpack AP_membrane, At, τA = spike
    @unpack Vr, Vt, b = adex
    gax, Cd = d.gax, d.C
    nabs_f = convert(eltype(tabs), n_abs)    # the counters are floats (type of the state): compare and store without mixing types
    FT = eltype(v_s)
    nref_f = convert(eltype(tabs), n_ref)
    # Branchless (`ifelse`) and vectorised, bit-identical to the branching loop of SNNModels <= 1.9 (backpropagation period,
    # absolute refractory period, free integration or spike): all the cases are evaluated and the one that applies is kept.
    # `ivdep`: with a chunk range that does not start at 1 LLVM cannot exclude that the store to `Δv[i, 1]` overlaps the loads of the other columns
    # of `Δv` and does not vectorise (5x slower); they never overlap (distinct matrix entries).
    @inbounds @simd ivdep for i ∈ rng
        t1 = tabs[i] - 1
        th1 = θ[i] + dt * (Vt - θ[i]) / τA
        bp = t1 > nabs_f                # backpropagation period
        rf = !bp & (t1 > 0)            # absolute refractory period
        act = t1 <= 0                   # free integration (or spike)
        vs0 = v_s[i]
        vd0 = v_d[i]
        ws0 = w_s[i]
        dv1 = Δv[i, 1]

        # bp or rf: the soma is clamped, the dendrite relaxes towards it
        vsc = ifelse(bp, AP_membrane, Vr)
        vdc = vd0 + dt * (vsc - vd0) * gax[i] / Cd[i]

        # act: spike or Heun update
        spk = act & (vs0 + dv1 * dt >= Vspike)
        keep = act & !spk
        vs_h = convert(FT, vs0 + 0.5 * dt * (Δv_temp[i, 1] + dv1))   # the old loop stored the Float64 sum into a Float32 array
        vd_h = convert(FT, vd0 + 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2]))   # the old loop stored the Float64 sum into a Float32 array
        ws_h = convert(FT, ws0 + 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3]))   # the old loop stored the Float64 sum into a Float32 array

        clamped = bp | rf
        fire[i] = spk
        Δv[i, 1] = ifelse(spk, AP_membrane - vs0, dv1)
        v_s[i] = ifelse(clamped, vsc, ifelse(spk, AP_membrane, ifelse(keep, vs_h, vs0)))
        v_d[i] = ifelse(clamped, vdc, ifelse(keep, vd_h, vd0))
        w_s[i] = ifelse(spk, ws0 + b, ifelse(keep, ws_h, ws0))
        θ[i] = ifelse(spk, th1 + At, th1)
        tabs[i] = ifelse(spk, nref_f, t1)
    end
end


@inline function update_neuron!(
    p::BallAndStick,
    param::DendNeuronParameter,
    Δv::Matrix{FT},
    dt::FT,
) where {FT<:AbstractFloat}
    update_neuron!(p, param, Δv, dt, 1:Int(p.N))
end

# One Heun stage for the neurons `rng`
@inline function update_neuron!(
    p::BallAndStick,
    param::DendNeuronParameter,
    Δv::Matrix{FT},
    dt::FT,
    rng::UnitRange{Int},
) where {FT<:AbstractFloat}
    @unpack v_d, v_s, w_s, θ, Is, Id = p
    @unpack d = p
    @unpack is, v_pred = p
    @unpack adex, soma_syn, dend_syn = p
    @unpack C, gl, El, ΔT, a, τw = adex
    @unpack synvars_s, synvars_d = p

    # Heun stage: evaluate the derivatives at the predicted state x + dt * Δv
    # (Δv = 0 at the first stage, i.e. at the current state).
    N = p.N
    dgax, dgm, dEl, dC = d.gax, d.gm, d.El, d.C
    _check_state_lengths(N, v_s, v_d, w_s, θ, Is, Id, dgax, dgm, dEl, dC)
    (size(Δv, 1) >= N && size(Δv, 2) >= 3 && size(is, 1) >= N && size(is, 2) >= 2 && size(v_pred, 1) >= N && size(v_pred, 2) >= 2) ||
        throw(DimensionMismatch("Δv, is and v_pred must have N rows (3, 2 and 2 columns)"))
    last(rng) <= N || throw(BoundsError(v_s, last(rng)))

    @inbounds @simd ivdep for i ∈ rng
        v_pred[i, 1] = v_s[i] + Δv[i, 1] * dt
        v_pred[i, 2] = v_d[i] + Δv[i, 2] * dt
    end

    @views _synaptic_current!(p, soma_syn, synvars_s, v_pred[:, 1], is[:, 1], rng)
    @views _synaptic_current!(p, dend_syn, synvars_d, v_pred[:, 2], is[:, 2], rng)
    @inbounds for k in 1:2
        @simd ivdep for i in rng
            is[i, k] = clamp(is[i, k], FT(-1000), FT(1000))
        end
    end

    # Axial current in a register, no `@fastmath`, the order of the compiled `@fastmath` loop of SNNModels <= 1.9 written with
    # `muladd`: vectorised, lane-independent and bit-identical to the previous releases on x86-64 with FMA; see `Tripod`.
    @inbounds @simd ivdep for i ∈ rng
        vs = v_pred[i, 1]
        vd = v_pred[i, 2]
        ws = muladd(Δv[i, 3], dt, w_s[i])
        icd = (vs - vd) * dgax[i] # axial current soma -> d
        # soma: gl * (El - vs + ΔT * exp((vs - θ) / ΔT)) + Is - (axial current + adaptation + synapses)
        e = exp256((vs - θ[i]) / ΔT)
        Δv[i, 1] = muladd(gl, muladd(ΔT, e, El - vs), Is[i] - (icd + (ws + is[i, 1]))) / C
        # dendrite: leak - synapses + axial + external
        Δv[i, 2] = ((muladd(dEl[i] - vd, dgm[i], -is[i, 2]) + icd) + Id[i]) / dC[i]
        Δv[i, 3] = muladd(a, vs - El, -ws) / τw
    end
end

# one precision per population (see `IF`); `param` holds the morphology only (precision-agnostic)
function snn_kw_finalize(::Type{T}, f::NamedTuple) where {T<:BallAndStick}
    _check_precision("BallAndStick", _state_float(T, :v_s), "adex" => f.adex, "soma_syn" => f.soma_syn, "dend_syn" => f.dend_syn,
                     "spike" => f.spike)
    return f
end

export BallAndStick
