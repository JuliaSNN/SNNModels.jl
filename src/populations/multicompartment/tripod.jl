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
C \frac{dV_s}{dt} &= g_L (E_L - V_s) + \Delta_T\, e^{(V_s - \theta)/\Delta_T} - w_s
    - I_{syn,s} - \sum_k g_{ax,k}\,(V_s - V_{d,k}) + I \\
C_{d,k} \frac{dV_{d,k}}{dt} &= g_{m,k} (E_L - V_{d,k}) - I_{syn,d,k}
    + g_{ax,k}\,(V_s - V_{d,k}) + I_d \\
\tau_w \frac{dw_s}{dt} &= a (V_s - E_L) - w_s \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```
``I_{syn,s}`` and ``I_{syn,d,k}`` are computed by `synaptic_current!` of `soma_syn` and
`dend_syn` with the compartment potentials at the beginning of the step, and are clamped to
``\pm 1500`` pA (`Tripod`) or ``\pm 1000`` pA (`BallAndStick`). ``C_{d,k}``, ``g_{m,k}``,
``g_{ax,k}`` come from the `Dendrite` structs (see `create_dendrite`). The dendritic leak
reversal is the somatic ``E_L`` (`adex.El`). Note that, as implemented, the exponential term is
not multiplied by ``g_L`` (standard AdEx uses ``g_L \Delta_T e^{(V-\theta)/\Delta_T}``).

Spike: when the predicted somatic potential ``V_s + dt\,\dot V_s`` reaches ``-10`` mV
(hard-coded, not `adex.Vt`), the neuron fires: ``V_s \leftarrow`` `AP_membrane`,
``w_s \leftarrow w_s + b``, ``\theta \leftarrow \theta + A_t``, and the refractory counter is set
to `round((up + τabs)/dt)` steps. During the first `up` the soma is clamped at `AP_membrane`
(back-propagation period), during the following `τabs` at `adex.Vr`; in both periods the
dendrites only relax towards the soma through the axial term (forward Euler) and `w_s` is not
integrated. ``\theta`` relaxes to `adex.Vt` at every step (forward Euler).

# Integration
Heun (explicit trapezoidal) method on ``(V_s, V_{d,k}, w_s)``: the derivatives are evaluated
at the current state (`Δv_temp`) and at the Euler-predicted state (`Δv`), and the state is
advanced by ``\frac{dt}{2}(\Delta v_{temp} + \Delta v)``. In the predicted adaptation
derivative the code uses `v_s + Δv` and `w_s + Δv` (without the factor `dt`) where the
voltage equations use `v + Δv dt`. Synaptic conductances are advanced first, once per step, by
`update_synapses!`.

# Fields
## Population info
- `id::String = randstring(12)`, `name::String = "Tripod"`, `N::IT = 100`, `records::Dict`.

## Parameters
- `param::DendNeuronParameter = TripodParameter()`: morphology (`ds`, `physiology`).
- `adex::SOMAT = AdExParameter()`: somatic parameters; uses `C`, `gl`, `El`, `ΔT`, `Vt`
  (resting value of the threshold ``\theta``), `Vr`, `a`, `b`, `τw`.
- `soma_syn::SYNS = TripodSomaSynapse`, `dend_syn::SYND = TripodDendSynapse`: synapse models.
- `spike::PST = PostSpike()`: `AP_membrane`, `up`, `τabs`, `At`, `τA`.
- `d1::VDT`, `d2::VDT`: `Dendrite`s built with `create_dendrite(N, param.ds[1])` and `param.ds[2]`.

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
- `is::MFT` (`N x 3`): synaptic currents of soma, d1, d2; `ic::VFT` (length 2): axial currents.

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
    d1::VDT = create_dendrite(N, param.ds[1])
    d2::VDT = create_dendrite(N, param.ds[2])

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

    @unpack spike, adex, soma_syn, dend_syn = p
    @unpack AP_membrane, up, τabs, At, τA = spike
    @unpack Vr, Vt, b = adex

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

    @inbounds for i ∈ 1:N
        tabs[i] -= 1
        θ[i] += dt * (Vt - θ[i]) / τA
        if tabs[i] > τabs / dt # backpropagation period
            v_s[i] = AP_membrane
            v_d1[i] += dt * (v_s[i] - v_d1[i]) * d1.gax[i] / d1.C[i]
            v_d2[i] += dt * (v_s[i] - v_d2[i]) * d2.gax[i] / d2.C[i]
        elseif tabs[i] > 0 # absolute refractory period
            v_s[i] = Vr
            v_d1[i] += dt * (v_s[i] - v_d1[i]) * d1.gax[i] / d1.C[i]
            v_d2[i] += dt * (v_s[i] - v_d2[i]) * d2.gax[i] / d2.C[i]
        elseif tabs[i] <= 0
            fire[i] = v_s[i] .+ Δv[i, 1] * dt >= -10mV
            Δv[i, 1] = ifelse(fire[i], AP_membrane - v_s[i], Δv[i, 1])
            v_s[i] = ifelse(fire[i], AP_membrane, v_s[i])
            w_s[i] = ifelse(fire[i], w_s[i] + b, w_s[i])
            θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
            tabs[i] = ifelse(fire[i], round(Int, (up + τabs) / dt), tabs[i])
            fire[i] && continue
            v_s[i] += 0.5 * dt * (Δv_temp[i, 1] + Δv[i, 1])
            v_d1[i] += 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2])
            v_d2[i] += 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3])
            w_s[i] += 0.5 * dt * (Δv_temp[i, 4] + Δv[i, 4])
        end
    end
end

@inline function update_neuron!(
    p::Tripod,
    param::DendNeuronParameter,
    Δv::Matrix{Float32},
    dt::Float32,
)
    @unpack v_d1, v_d2, v_s, I_d, I, w_s, θ, tabs, fire = p
    @unpack d1, d2 = p
    @unpack is, ic = p
    @unpack adex, spike, soma_syn, dend_syn = p
    @unpack AP_membrane, up, τabs, At, τA = spike
    @unpack C, gl, El, ΔT, Vt, Vr, a, b, τw = adex
    @unpack synvars_s, synvars_d1, synvars_d2 = p


    @views synaptic_current!(p, soma_syn, synvars_s, v_s[:], is[:, 1])
    @views synaptic_current!(p, dend_syn, synvars_d1, v_d1[:], is[:, 2])
    @views synaptic_current!(p, dend_syn, synvars_d2, v_d2[:], is[:, 3])
    clamp!(is, -1500, 1500)

    @fastmath @inbounds for i ∈ 1:p.N
        ic[1] = -((v_d1[i] + Δv[i, 2] * dt) - (v_s[i] + Δv[i, 1] * dt)) * d1.gax[i]
        ic[2] = -((v_d2[i] + Δv[i, 3] * dt) - (v_s[i] + Δv[i, 1] * dt)) * d2.gax[i]
        Δv[i, 2] =
            ((-(v_d1[i] + Δv[i, 2] * dt) + El) * d1.gm[i] - is[i, 2] + ic[1] + I_d[i]) /
            d1.C[i]
        Δv[i, 3] =
            ((-(v_d2[i] + Δv[i, 3] * dt) + El) * d2.gm[i] - is[i, 3] + ic[2] + I_d[i]) /
            d2.C[i]
        Δv[i, 1] =
            1/C * (
                + gl * (-(v_s[i] + Δv[i, 1] * dt) + El) +
                ΔT * exp256(1 / ΔT * (v_s[i] + Δv[i, 1] * dt - θ[i])) - w_s[i]  # adaptation
                - is[i, 1]   # synapses
                - sum(ic) # axial currents
                + I[i]  # external current
            )
        # Δv[i, 4] = (a * (v_s[i]- El) - (w_s[i])) / τw
        Δv[i, 4] = (a * ((v_s[i] + Δv[i, 1]) - El) - (w_s[i] + Δv[i, 4])) / τw
    end
end


export Tripod
