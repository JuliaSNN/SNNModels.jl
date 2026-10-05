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
C \frac{dV_s}{dt} &= g_L (E_L - V_s) + \Delta_T\, e^{(V_s - \theta)/\Delta_T} - w_s
    - I_{syn,s} - \sum_k g_{ax,k}\,(V_s - V_{d,k}) \\
C_{d,k} \frac{dV_{d,k}}{dt} &= g_{m,k} (E_L - V_{d,k}) - I_{syn,d,k}
    + g_{ax,k}\,(V_s - V_{d,k}) \\
\tau_w \frac{dw_s}{dt} &= a (V_s - E_L) - w_s \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```
``I_{syn,s}`` and ``I_{syn,d,k}`` are computed by `synaptic_current!` of `soma_syn` and
`dend_syn` with the compartment potentials of the Heun stage being evaluated, and are clamped to
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
Heun (explicit trapezoidal) method on ``x = (V_s, V_{d,k}, w_s)``. Synaptic conductances are
advanced first, once per step, by `update_synapses!`. Then ``k_1 = f(x_n)`` is evaluated at the
current state (stored in `Δv_temp`) and ``k_2 = f(x_n + dt\,k_1)`` at the Euler-predicted state
(stored in `Δv`); every term of ``f``, including the synaptic currents, the axial currents and
the adaptation current in the somatic equation, is evaluated at the same stage state. The state
is advanced by ``x_{n+1} = x_n + \frac{dt}{2}(k_1 + k_2)``. The scheme is second-order accurate
between spikes.

!!! note "Changed after SNNModels 1.8.4"
    Up to SNNModels 1.8.4 the predicted adaptation state was `w_s + Δv` and `v_s + Δv`
    (without the factor `dt`), the first-stage adaptation derivative read the already updated
    somatic derivative, and the synaptic currents and `w_s` in the somatic equation were not
    evaluated at the predicted state. Results therefore depended on `dt`.

The fields `Is` and `Id` (external currents) exist but are not used by the equations in the
current implementation.

# Fields
## Population info
- `name::String = "BallAndStick"`, `id::String = randstring(12)`, `N::IT = 100`, `records::Dict`.

## Parameters
- `param::DendNeuronParameter = BallAndStickParameter()`: morphology.
- `adex::SOMAT = AdExParameter()`: somatic parameters (see `Tripod`).
- `dend_syn::SYND = TripodDendSynapse`, `soma_syn::SYNS = TripodSomaSynapse`: synapse models.
- `spike::PST = PostSpike()`: spike shape and refractoriness.
- `d::VDT`: `Dendrite` built with `create_dendrite(N, param.ds[1])`.

## State variables
- `v_s`, `v_d::VFT`: somatic and dendritic potentials (mV), initialised uniformly in `[Vr, Vt]`.
- `w_s::VFT`: somatic adaptation current (pA).
- `Is`, `Id::VFT`: external currents (pA), currently unused.
- `fire::VBT`, `tabs::VFT`, `θ::VFT`: spike flags, refractory counters (steps), dynamic threshold (mV).

## Synapses
- `synvars_s`, `synvars_d`: synaptic state variables; `receptors_s`, `receptors_d::NamedTuple`: input buffers.

## Work arrays
- `Δv`, `Δv_temp::MFT` (`N x 3`): derivatives of `(v_s, v_d, w_s)`; `is::MFT` (`N x 2`);
  `ic::VFT` (length 1); `v_pred::MFT` (`N x 2`): stage potentials of soma and dendrite.

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
    MFT = Matrix{Float32},
    VDT = Dendrite{Vector{Float32}},
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
    adex::SOMAT = AdExParameter()
    dend_syn::SYND = TripodDendSynapse
    soma_syn::SYNS = TripodSomaSynapse
    spike::PST = PostSpike()

    # Membrane potential and adaptation
    d::VDT = create_dendrite(N, param.ds[1])
    v_s::VFT = rand_value(N, adex.Vt, adex.Vr)
    w_s::VFT = zeros(N)
    v_d::VFT = rand_value(N, adex.Vt, adex.Vr)

    # Synapses
    synvars_s::SYNSV = synaptic_variables(soma_syn, N)
    synvars_d::SYNDV = synaptic_variables(dend_syn, N)

    ## Ext input
    Is::VFT = zeros(N)
    Id::VFT = zeros(N)

    # Receptors properties
    receptors_d::RECT = synaptic_receptors(dend_syn, N) #! target
    receptors_s::RECT = synaptic_receptors(soma_syn, N) #! target

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
    integrate!(p::BallAndStick, param::DendNeuronParameter, dt::Float32)

Advance a `BallAndStick` population by `dt` (synapses, Heun step of soma and dendrite, spikes
and refractoriness; see `BallAndStick`).
"""
function integrate!(p::BallAndStick, param::DendNeuronParameter, dt::Float32)
    @unpack N, v_s, w_s, v_d = p
    @unpack fire, θ, tabs = p
    @unpack Δv, Δv_temp, is = p

    @unpack synvars_s, synvars_d, d = p
    @unpack receptors_d, receptors_s = p

    @unpack spike, adex, soma_syn, dend_syn = p
    @unpack AP_membrane, up, τabs, At, τA = spike
    @unpack El, Vr, Vt, τw, a, b = adex

    update_synapses!(p, soma_syn, receptors_s, synvars_s, dt)
    update_synapses!(p, dend_syn, receptors_d, synvars_d, dt)

    ## Heun integration
    fill!(Δv, 0.0f0)
    fill!(Δv_temp, 0.0f0)
    fill!(fire, false)
    update_neuron!(p, param, Δv, dt)
    Δv_temp .= Δv
    update_neuron!(p, param, Δv, dt)

    @inbounds for i ∈ 1:N
        tabs[i] -= 1
        θ[i] += dt * (Vt - θ[i]) / τA
        if tabs[i] > τabs / dt # backpropagation period
            v_s[i] = AP_membrane
            v_d[i] += dt * (v_s[i] - v_d[i]) * d.gax[i] / d.C[i]
        elseif tabs[i] > 0 # absolute refractory period
            v_s[i] = Vr
            v_d[i] += dt * (v_s[i] - v_d[i]) * d.gax[i] / d.C[i]
        elseif tabs[i] <= 0
            fire[i] = v_s[i] .+ Δv[i, 1] * dt >= -10mV
            Δv[i, 1] = ifelse(fire[i], AP_membrane - v_s[i], Δv[i, 1])
            v_s[i] = ifelse(fire[i], AP_membrane, v_s[i])
            w_s[i] = ifelse(fire[i], w_s[i] + b, w_s[i])
            θ[i] = ifelse(fire[i], θ[i] + At, θ[i])
            tabs[i] = ifelse(fire[i], round(Int, (up + τabs) / dt), tabs[i])
            fire[i] && continue
            v_s[i] += 0.5 * dt * (Δv_temp[i, 1] + Δv[i, 1])
            v_d[i] += 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2])
            w_s[i] += 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3])
        end
    end
end


@inline function update_neuron!(
    p::BallAndStick,
    param::DendNeuronParameter,
    Δv::Matrix{Float32},
    dt::Float32,
)
    @unpack v_d, v_s, w_s, θ = p
    @unpack d = p
    @unpack is, ic, v_pred = p
    @unpack adex, soma_syn, dend_syn = p
    @unpack C, gl, El, ΔT, a, τw = adex
    @unpack synvars_s, synvars_d = p

    # Heun stage: evaluate the derivatives at the predicted state x + dt * Δv
    # (Δv = 0 at the first stage, i.e. at the current state).
    @inbounds for i ∈ 1:p.N
        v_pred[i, 1] = v_s[i] + Δv[i, 1] * dt
        v_pred[i, 2] = v_d[i] + Δv[i, 2] * dt
    end

    @views synaptic_current!(p, soma_syn, synvars_s, v_pred[:, 1], is[:, 1])
    @views synaptic_current!(p, dend_syn, synvars_d, v_pred[:, 2], is[:, 2])
    clamp!(is, -1000, 1000)

    @fastmath @inbounds for i ∈ 1:p.N
        vs = v_pred[i, 1]
        vd = v_pred[i, 2]
        ws = w_s[i] + Δv[i, 3] * dt
        ic[1] = (vs - vd) * d.gax[i] # axial current soma -> d
        Δv[i, 1] =
            (
                gl * (El - vs) +
                ΔT * exp256((vs - θ[i]) / ΔT) - ws  # adaptation
                - is[i, 1]   # synapses
                - ic[1] # axial current
                # + I[i]  # external current
            ) / C
        Δv[i, 2] = ((El - vd) * d.gm[i] - is[i, 2] + ic[1]) / d.C[i]
        Δv[i, 3] = (a * (vs - El) - ws) / τw
    end
end


export BallAndStick
