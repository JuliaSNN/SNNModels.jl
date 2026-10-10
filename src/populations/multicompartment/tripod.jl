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

# Fields
## Population info
- `id::String = randstring(12)`, `name::String = "Tripod"`, `N::IT = 100`, `records::Dict`.

## Parameters
- `param::DendNeuronParameter = TripodParameter()`: morphology (`ds`, `physiology`).
- `adex::SOMAT = AdExParameter()`: somatic parameters; uses `C`, `gl`, `El`, `ΔT`, `Vt`
  (resting value of the threshold ``\theta``), `Vr`, `a`, `b`, `τw`.
- `soma_syn::SYNS = TripodSomaSynapse`, `dend_syn::SYND = TripodDendSynapse`: synapse models.
- `spike::PST = PostSpike()`: `AP_membrane`, `up`, `τabs`, `At`, `τA`.
- `d1::VDT`, `d2::VDT`: `Dendrite`s built with `create_dendrite(N, param.ds[k]; El = adex.El)`.
- `Vspike::Float32 = -10mV`: spike detection threshold on the predicted somatic potential.

## State variables
- `v_s`, `v_d1`, `v_d2::VFT`: compartment potentials (mV), initialised uniformly in `[Vr, Vt]`.
- `w_s::VFT`: somatic adaptation current (pA), zeros.
- `I::VFT`: external current into the soma (pA); `I_d::VFT`: external current injected in
  each of the two dendrites (pA).
- `fire::VBT`: spike flags; `tabs::VFT`: refractory counters (steps); `θ::VFT`: dynamic
  threshold of the exponential term (mV), initialised to `Vt`.

## Synapses
- `synvars_s`, `synvars_d1`, `synvars_d2`: synaptic state variables (`synaptic_variables`).
- `receptors_s`, `receptors_d1`, `receptors_d2::NamedTuple`: input buffers (`synaptic_receptors`).

## Work arrays
- `Δv`, `Δv_temp::MFT` (`N x 4`): derivatives of `(v_s, v_d1, v_d2, w_s)` at the two Heun stages.
- `is::MFT` (`N x 3`): synaptic currents of soma, d1, d2; `ic::VFT` (length 2): axial currents
  soma -> dendrite (pA) of the last evaluated stage; `v_pred::MFT` (`N x 3`): stage potentials
  of soma, d1, d2 (mV).

# References
Quaresima A. et al. (2023), "The Tripod neuron: a minimal structural reduction of the
dendritic tree", J. Physiol. (canonical source of the model; the reference is not given in
the code). Receptor parameters: see `TripodSomaSynapse` and `TripodDendSynapse`.

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
    MFT = Matrix{Float32},
    VDT = Dendrite{Vector{Float32}},
    SYNS<:AbstractSynapseParameter,
    SYND<:AbstractSynapseParameter,
    SYNSV<:AbstractSynapseVariable,
    SYNDV<:AbstractSynapseVariable,
    SOMAT<:AbstractGeneralizedIFParameter,
    PST<:AbstractSpikeParameter,
    RECTS<:NamedTuple,
    RECTD<:NamedTuple,
    IT = Int32,
} <: AbstractDendriteIF
    id::String = randstring(12)
    name::String = "Tripod"
    ## These are compulsory parameters
    N::IT = 100
    param::DendNeuronParameter = TripodParameter()
    adex::SOMAT = AdExParameter()
    soma_syn::SYNS = TripodSomaSynapse
    dend_syn::SYND = TripodDendSynapse
    spike::PST = PostSpike()
    Vspike::Float32 = -10mV
    d1::VDT = create_dendrite(N, param.ds[1]; El = adex.El)
    d2::VDT = create_dendrite(N, param.ds[2]; El = adex.El)

    # Membrane potential and adaptation
    v_s::VFT = rand_value(N, adex.Vt, adex.Vr)
    w_s::VFT = zeros(N)
    v_d1::VFT = rand_value(N, adex.Vt, adex.Vr)
    v_d2::VFT = rand_value(N, adex.Vt, adex.Vr)
    I::VFT = zeros(N)
    I_d::VFT = zeros(N)

    # Synapses dendrites
    synvars_s::SYNSV = synaptic_variables(soma_syn, N)
    synvars_d1::SYNDV = synaptic_variables(dend_syn, N)
    synvars_d2::SYNDV = synaptic_variables(dend_syn, N)

    receptors_s::RECTS = synaptic_receptors(soma_syn, N)
    receptors_d1::RECTD = synaptic_receptors(dend_syn, N)
    receptors_d2::RECTD = synaptic_receptors(dend_syn, N)

    # Spike model and threshold
    fire::VBT = zeros(Bool, N)
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
    integrate!(p::Tripod, param::DendNeuronParameter, dt::Float32)

Advance a `Tripod` population by `dt`: update the synapses of the three compartments, then
integrate the membrane equations with the Heun method and handle spikes and refractoriness
(see `Tripod`).
"""
function integrate!(p::Tripod, param::DendNeuronParameter, dt::Float32)
    @unpack N, v_s, w_s, v_d1, v_d2 = p
    @unpack fire, θ, tabs = p
    @unpack Δv, Δv_temp, is = p

    @unpack synvars_s, synvars_d1, synvars_d2, d1, d2, I_d = p
    @unpack receptors_d1, receptors_d2, receptors_s = p

    @unpack spike, adex, soma_syn, dend_syn, Vspike = p
    @unpack AP_membrane, up, τabs, At, τA = spike
    @unpack Vr, Vt, b = adex
    # refractory counters in steps; each period lasts at least one step
    n_abs = max(1, round(Int, τabs / dt))
    n_ref = max(1, round(Int, up / dt)) + n_abs

    # Update all synaptic conductance
    update_synapses!(p, soma_syn, receptors_s, synvars_s, dt)
    update_synapses!(p, dend_syn, receptors_d1, synvars_d1, dt)
    update_synapses!(p, dend_syn, receptors_d2, synvars_d2, dt)

    ## Heun integration
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)

    update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    update_neuron!(p, param, Δv, dt)
    # @show Δv.+Δv_temp

    _check_state_lengths(N, v_s, w_s, v_d1, v_d2, fire, θ, tabs, d1.gax, d1.C, d2.gax, d2.C)
    (size(Δv, 1) >= N && size(Δv, 2) >= 4 && size(Δv_temp, 1) >= N && size(Δv_temp, 2) >= 4) ||
        throw(DimensionMismatch("Δv and Δv_temp must have N rows and 4 columns"))
    gax1, C1, gax2, C2 = d1.gax, d1.C, d2.gax, d2.C
    nabs_f = convert(eltype(tabs), n_abs)    # the counters are `Float32`: compare and store without mixing types
    nref_f = convert(eltype(tabs), n_ref)
    # Branchless (`ifelse`) and vectorised, bit-identical to the branching loop of SNNModels <= 1.9 (backpropagation period,
    # absolute refractory period, free integration or spike): all the cases are evaluated and the one that applies is kept.
    @inbounds @simd for i ∈ 1:N
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

        # bp or rf: the soma is clamped, the dendrites relax towards it
        vsc = ifelse(bp, AP_membrane, Vr)
        vd1c = vd1 + dt * (vsc - vd1) * gax1[i] / C1[i]
        vd2c = vd2 + dt * (vsc - vd2) * gax2[i] / C2[i]

        # act: spike or Heun update
        spk = act & (vs0 + dv1 * dt >= Vspike)
        keep = act & !spk
        vs_h = convert(Float32, vs0 + 0.5 * dt * (Δv_temp[i, 1] + dv1))   # the old loop stored the Float64 sum into a Float32 array
        vd1_h = convert(Float32, vd1 + 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2]))   # the old loop stored the Float64 sum into a Float32 array
        vd2_h = convert(Float32, vd2 + 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3]))   # the old loop stored the Float64 sum into a Float32 array
        ws_h = convert(Float32, ws0 + 0.5 * dt * (Δv_temp[i, 4] + Δv[i, 4]))   # the old loop stored the Float64 sum into a Float32 array

        clamped = bp | rf
        fire[i] = spk
        Δv[i, 1] = ifelse(spk, AP_membrane - vs0, dv1)
        v_s[i] = ifelse(clamped, vsc, ifelse(spk, AP_membrane, ifelse(keep, vs_h, vs0)))
        v_d1[i] = ifelse(clamped, vd1c, ifelse(keep, vd1_h, vd1))
        v_d2[i] = ifelse(clamped, vd2c, ifelse(keep, vd2_h, vd2))
        w_s[i] = ifelse(spk, ws0 + b, ifelse(keep, ws_h, ws0))
        θ[i] = ifelse(spk, th1 + At, th1)
        tabs[i] = ifelse(spk, nref_f, t1)
    end
end

@inline function update_neuron!(
    p::Tripod,
    param::DendNeuronParameter,
    Δv::Matrix{Float32},
    dt::Float32,
)
    @unpack v_d1, v_d2, v_s, I_d, I, w_s, θ = p
    @unpack d1, d2 = p
    @unpack is, ic, v_pred = p
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

    @inbounds @simd for i ∈ 1:N
        v_pred[i, 1] = v_s[i] + Δv[i, 1] * dt
        v_pred[i, 2] = v_d1[i] + Δv[i, 2] * dt
        v_pred[i, 3] = v_d2[i] + Δv[i, 3] * dt
    end

    @views synaptic_current!(p, soma_syn, synvars_s, v_pred[:, 1], is[:, 1])
    @views synaptic_current!(p, dend_syn, synvars_d1, v_pred[:, 2], is[:, 2])
    @views synaptic_current!(p, dend_syn, synvars_d2, v_pred[:, 3], is[:, 3])
    clamp!(is, -1500, 1500)

    # This `@fastmath` loop is kept as it was (scalar, `ic` stored and reloaded). Its floating-point result depends on how LLVM reassociates the
    # fast-math expression: with the axial currents in registers (which lets it vectorise, about 5x faster) the derivatives
    # differ from this loop in the last bit (1 ulp in `Δv`), so the vectorised variant is not used (see the changelog).
    @fastmath @inbounds for i ∈ 1:N
        vs = v_pred[i, 1]
        vd1 = v_pred[i, 2]
        vd2 = v_pred[i, 3]
        ws = w_s[i] + Δv[i, 4] * dt
        ic[1] = (vs - vd1) * d1gax[i] # axial current soma -> d1
        ic[2] = (vs - vd2) * d2gax[i] # axial current soma -> d2
        Δv[i, 2] = ((d1El[i] - vd1) * d1gm[i] - is[i, 2] + ic[1] + I_d[i]) / d1C[i]
        Δv[i, 3] = ((d2El[i] - vd2) * d2gm[i] - is[i, 3] + ic[2] + I_d[i]) / d2C[i]
        Δv[i, 1] =
            (
                gl * (El - vs) +
                gl * ΔT * exp256((vs - θ[i]) / ΔT) - ws  # adaptation
                - is[i, 1]   # synapses
                - ic[1] - ic[2] # axial currents
                + I[i]  # external current
            ) / C
        Δv[i, 4] = (a * (vs - El) - ws) / τw
    end
end


export Tripod
