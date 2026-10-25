# BallAndStick
@doc raw"""
    BallAndStick(; N = 100, param = BallAndStickParameter(), adex = AdExParameter(),
                   soma_syn = TripodSomaSynapse, dend_syn = TripodDendSynapse, spike = PostSpike(), ...)

Population of two-compartment neurons: an AdEx-like soma coupled to one passive dendrite
(`d`) through an axial conductance. Same equations, spike rule and Heun integration as `Tripod`
with a single dendrite (during the backpropagation and refractory periods the soma is clamped and the dendrite is integrated
in full, with leak and synaptic currents; before SNNModels 2.1 it only relaxed towards the soma); the synaptic currents are
clamped to ``\pm 1000`` pA and the external currents are `Is` (soma) and `Id` (dendrite). The dendrite diameter is
`param.d`. Connections select the compartment with the fourth
argument of `SpikingSynapse`: targets `:s` and `:d`.

# Fields
## Population info
- `name::String = "BallAndStick"`, `id::String = randstring(12)`, `N::IT = 100`, `records::Dict`.

## Parameters
- `param::DendNeuronParameter = BallAndStickParameter()`: morphology.
- `adex::SOMAT = AdExParameter()`: somatic parameters (see `Tripod`).
- `dend_syn::SYND = TripodDendSynapse`, `soma_syn::SYNS = TripodSomaSynapse`: synapse models.
- `spike::PST = PostSpike()`: spike shape and refractoriness.
- `d::VDT`: `Dendrite` built with `create_dendrite(N, param.ds[1]; El = _dend_El(param, adex), FT = eltype(VFT), physiology = param.physiology, d = param.d)`, where `_dend_El(param, adex)` is `param.El_dend` if not `nothing`, else `adex.El` (read at construction).
- `Vspike::Float32 = -10mV`: spike detection threshold on the predicted somatic potential.

## State variables
- `v_s`, `v_d::VFT`: somatic and dendritic potentials (mV), initialised uniformly in `[Vr, Vt]`.
  `VFT = Vector{Float64}` gives a `Float64` population, as for `Tripod`.
- `w_s::VFT`: somatic adaptation current (pA).
- `Is`, `Id::VFT`: external currents into the soma and the dendrite (pA), zeros.
- `fire::VBT`, `tabs::VFT`, `θ::VFT`: spike flags, refractory counters (steps), dynamic threshold (mV).
- `spikes::SpikeIndex`: neurons that fired in the step (see `spike_index`).

## Synapses
- `synvars_s`, `synvars_d`: synaptic state variables; `receptors_s`, `receptors_d::NamedTuple`: input buffers.

## Work arrays
- `Δv`, `Δv_temp::MFT` (`N x 3`): derivatives of `(v_s, v_d, w_s)`; `is::MFT` (`N x 2`);
  `ic::VFT` (length 1): unused, kept for compatibility; `v_pred::MFT` (`N x 2`): stage potentials of soma and dendrite.

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
    d::VDT = create_dendrite(N, param.ds[1]; El = _dend_El(param, adex), FT = eltype(VFT), physiology = param.physiology, d = param.d)
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
    spikes::SpikeIndex = owned_spike_index(N)
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
function integrate!(p::BallAndStick, param::DendNeuronParameter, dt_::AbstractFloat)
    dt = _step_dt(p, dt_)   # in the precision of the population
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
        _bas_clamp_soma!(p, 1:N, n_abs)
        update_neuron!(p, param, p.Δv, dt)
        p.Δv_temp .= p.Δv
        _bas_hold_soma!(p, 1:N, n_abs)
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
    _bas_clamp_soma!(p, rng, n_abs)
    update_neuron!(p, param, Δv, dt, rng)
    @inbounds for k in 1:3
        @simd ivdep for i in rng
            Δv_temp[i, k] = Δv[i, k]
        end
    end
    _bas_hold_soma!(p, rng, n_abs)
    update_neuron!(p, param, Δv, dt, rng)
    _bas_state!(p, dt, rng, n_abs, n_ref)
end

# Neurons in the backpropagation or refractory period: the soma is held at its clamp value for the whole step, so that both Heun
# stages of the dendrite see it (the dendritic equation is integrated in full); see `Tripod`.
function _bas_clamp_soma!(p::BallAndStick, rng::UnitRange{Int}, n_abs::Int)
    @unpack v_s, tabs, spike, adex = p
    AP, Vr = spike.AP_membrane, adex.Vr
    nabs_f = convert(eltype(tabs), n_abs)
    @inbounds @simd ivdep for i ∈ rng
        t1 = tabs[i] - 1
        v_s[i] = ifelse(t1 > nabs_f, AP, ifelse(t1 > 0, Vr, v_s[i]))
    end
end

# Between the two Heun stages: a clamped soma does not move in the Euler prediction.
function _bas_hold_soma!(p::BallAndStick, rng::UnitRange{Int}, n_abs::Int)
    @unpack Δv, tabs = p
    @inbounds @simd ivdep for i ∈ rng
        Δv[i, 1] = ifelse(tabs[i] - 1 > 0, zero(eltype(Δv)), Δv[i, 1])
    end
end

function _bas_state!(p::BallAndStick, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int)
    @unpack v_s, w_s, v_d, fire, θ, tabs, Δv, Δv_temp, d = p
    @unpack spike, adex, Vspike = p
    @unpack AP_membrane, At, τA = spike
    @unpack Vr, Vt, b = adex
    nabs_f = convert(eltype(tabs), n_abs)    # the counters are floats: compare and store without mixing types
    FT = eltype(v_s)
    nref_f = convert(eltype(tabs), n_ref)
    # branchless and `ivdep`, as in `_tripod_state!`
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

        # bp or rf: the soma is clamped (the Heun stages already integrated the dendrite against the clamped soma)
        vsc = ifelse(bp, AP_membrane, Vr)

        # act: spike or Heun update
        spk = act & (vs0 + dv1 * dt >= Vspike)
        keep = act & !spk
        vs_h = convert(FT, vs0 + 0.5 * dt * (Δv_temp[i, 1] + dv1))
        vd_h = convert(FT, vd0 + 0.5 * dt * (Δv_temp[i, 2] + Δv[i, 2]))
        ws_h = convert(FT, ws0 + 0.5 * dt * (Δv_temp[i, 3] + Δv[i, 3]))

        clamped = bp | rf
        fire[i] = spk
        Δv[i, 1] = ifelse(spk, AP_membrane - vs0, dv1)
        v_s[i] = ifelse(clamped, vsc, ifelse(spk, AP_membrane, ifelse(keep, vs_h, vs0)))
        v_d[i] = ifelse(spk, vd0, vd_h)    # Heun update, also while the soma is clamped; it holds on the spike step
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

    # `muladd` order and no `@fastmath`: see the `Tripod` stage
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
