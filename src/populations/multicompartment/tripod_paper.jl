# `PaperScheme` of `Tripod` (the integration of the published code) and the `TripodPaper` configuration.
# Original: https://github.com/aquaresima/tripod_neuron at commit 8698e1a (paper release), src/TripodNeuron.jl/src/
#   simulator/simulation.jl (`run_tripod`), simulator/equations.jl, model/{parameters,soma,dendrites,synapses}.jl.

# ---------------------------------------------------------------------------------------------------------------------------------
# Scalar receptor current of one compartment of one neuron: sum over receptors of gsyn g (v - E) B(v), as `_synaptic_current_rng!`
# ---------------------------------------------------------------------------------------------------------------------------------
@inline function _receptor_current(synapse::ReceptorSynapse, synvars::ReceptorSynapseVars, v, i::Int)
    @unpack g = synvars
    @unpack syn, NMDA = synapse
    I = zero(v)
    @inbounds for n in eachindex(syn)
        r = syn[n]
        cur = r.gsyn * g[i, n] * (v - r.E_rev)
        if r.nmda != zero(r.nmda)
            cur *= _nmda_gating(NMDA.approx, v, NMDA.b, NMDA.k, NMDA.mg)
        end
        I += cur
    end
    return I
end

# dendritic derivative with the axial current `c` (into the dendrite) fixed; synaptic current clamped to ±1500 pA
@inline function _paper_fd(v, c, gm, El, Cd, synapse, synvars, i)
    isyn = clamp(_receptor_current(synapse, synvars, v, i), oftype(v, -1500), oftype(v, 1500))
    return (-(v - El) * gm - isyn + c) / Cd
end

# two-stage Heun step of a dendrite (`ΔvDend2` of the original)
@inline function _paper_dend(vd, c, gm, El, Cd, synapse, synvars, i, dt)
    k1 = _paper_fd(vd, c, gm, El, Cd, synapse, synvars, i)
    k2 = _paper_fd(vd + k1 * dt, c, gm, El, Cd, synapse, synvars, i)
    return convert(typeof(vd), vd + dt * (0.5 * (k1 + k2)))
end

# two-stage Heun step of the soma with `w` and the axial current fixed (`ΔvSoma2` of the original)
@inline function _paper_soma(vs, wn, th, axial, gl, El, ΔT, Cinv, synapse, synvars, i, dt)
    f(v) = (gl * (-v + El + ΔT * exp32o(inv(ΔT) * (v - th))) - wn - _receptor_current(synapse, synvars, v, i) - axial) * Cinv
    k1 = f(vs)
    k2 = f(vs + k1 * dt)
    return convert(typeof(vs), vs + dt * (0.5 * (k1 + k2)))
end

const _PAPER_DT = 0.1
function _paper_check_dt(dt)
    abs(dt - _PAPER_DT) > 1e-4 &&
        @warn "Tripod with PaperScheme (TripodPaper) is tuned to dt = 0.1 ms (refractory period and clamp lengths, numerics of the " *
              "published code); got dt = $dt ms. Use sim!/train! with dt = 0.1ms." maxlog = 1
    return nothing
end

function _tripod_chunk!(::PaperScheme, p::Tripod, param::DendNeuronParameter, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int)
    _paper_check_dt(dt)
    @unpack fire = p
    update_synapses!(p, p.soma_syn, p.receptors_s, p.synvars_s, dt, rng)
    update_synapses!(p, p.dend_syn, p.receptors_d1, p.synvars_d1, dt, rng)
    update_synapses!(p, p.dend_syn, p.receptors_d2, p.synvars_d2, dt, rng)
    @inbounds for i in rng
        fire[i] = false
        p.Δv[i, 1] = zero(eltype(p.Δv))
    end
    _paper_step!(p, dt, rng, n_abs, n_ref)
end

# One step of the original loop for the neurons `rng`; see `PaperScheme` for the list of its rules.
function _paper_step!(p::Tripod, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int)
    @unpack v_s, w_s, v_d1, v_d2, fire, θ, tabs, I, I_d, d1, d2, adex, spike, Vspike, scheme = p
    @unpack soma_syn, dend_syn, synvars_s, synvars_d1, synvars_d2 = p
    @unpack C, gl, El, ΔT, a, τw, Vr, Vt, b = adex
    @unpack AP_membrane, At, τA = spike
    FT = eltype(v_s)
    AP_spike = convert(FT, scheme.AP_spike)
    nabs_f = convert(eltype(tabs), n_abs)
    nref_f = convert(eltype(tabs), n_ref)
    Cinv = inv(C)
    gax1, gm1, El1, C1 = d1.gax, d1.gm, d1.El, d1.C
    gax2, gm2, El2, C2 = d2.gax, d2.gm, d2.El, d2.C
    @inbounds for i in rng
        vs = v_s[i]
        vd1 = v_d1[i]
        vd2 = v_d2[i]
        w = w_s[i]
        # external currents: Euler kick before the step
        vs = convert(FT, vs + (dt * Cinv) * I[i])
        vd1 = convert(FT, vd1 + (dt / C1[i]) * I_d[i])
        vd2 = convert(FT, vd2 + (dt / C2[i]) * I_d[i])

        t1 = tabs[i] - 1
        th1 = θ[i] + dt * (Vt - θ[i]) / τA
        # axial currents into the dendrites from the potentials at the start of the step (kept fixed over the step)
        c1 = (vs - vd1) * gax1[i]
        c2 = (vs - vd2) * gax2[i]
        spk = false
        if t1 <= 0
            if vs >= Vspike
                spk = true
                vs = AP_spike
                w += b
                # the dendrites do not move in the spike step
            else
                vd1n = _paper_dend(vd1, c1, gm1[i], El1[i], C1[i], dend_syn, synvars_d1, i, dt)
                vd2n = _paper_dend(vd2, c2, gm2[i], El2[i], C2[i], dend_syn, synvars_d2, i, dt)
                wn = convert(FT, w + dt * (muladd(a, vs - El, -w) / τw))
                vs = _paper_soma(vs, wn, θ[i], c1 + c2, gl, El, ΔT, Cinv, soma_syn, synvars_s, i, dt)
                vd1, vd2, w = vd1n, vd2n, wn
            end
        else
            # backpropagation or refractory period: dendrites in full, then the soma is clamped
            vd1 = _paper_dend(vd1, c1, gm1[i], El1[i], C1[i], dend_syn, synvars_d1, i, dt)
            vd2 = _paper_dend(vd2, c2, gm2[i], El2[i], C2[i], dend_syn, synvars_d2, i, dt)
            vs = t1 > nabs_f ? AP_membrane : Vr
        end
        fire[i] = spk
        v_s[i] = vs
        v_d1[i] = vd1
        v_d2[i] = vd2
        w_s[i] = w
        θ[i] = spk ? th1 + At : th1
        tabs[i] = spk ? nref_f : t1
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------
# TripodPaper
# ---------------------------------------------------------------------------------------------------------------------------------
const _PAPER_CONFIGS = (
    distal_distal = distal_distal,
    proximal_proximal = proximal_proximal,
    proximal_distal = proximal_distal,
    distal_proximal = distal_proximal,
    soma_only = soma_only,
)

@doc raw"""
    TripodPaper(; N = 100, physiology = :human, config = :distal_proximal, d = 4um, VFT = Vector{Float32}, kwargs...) -> Tripod

`Tripod` population configured as the model of the published code of Quaresima et al. (2023), "The Tripod neuron: a minimal
structural reduction of the dendritic tree", J. Physiol. (preprint bioRxiv 10.1101/2022.09.05.506197), repository
`aquaresima/tripod_neuron` at commit 8698e1a (the paper release; `src/TripodNeuron.jl/src/model/*.jl` and
`simulator/simulation.jl`). The parameters follow that code, not the preprint where the two differ (audit in
`validation/tripod_audit` of the JuliaSNN publication repository). Use `dt = 0.1ms` in `sim!`/`train!`: the refractory
lengths and the numerics are those of the original at that step; other steps give a warning (once).

# Keywords
- `physiology::Symbol = :human`: `:human` (`human_dend`, `TripodDendSynapsePaper`) or `:mouse` (`mouse_dend`,
  `TripodDendSynapsePaperMouse`: dendritic AMPA τr 0.25 / NMDA τr 0.99, τd 100, g0 0.159 ms/nS, `DuarteGluDend`).
- `config::Symbol = :distal_proximal`: dendrite lengths of `d1` and `d2`: `:distal_distal` (400, 400 um), `:proximal_proximal`
  (150, 150), `:distal_proximal` (400, 150; the default of the original code, its protocols stimulate `d1`),
  `:proximal_distal` (150, 400) or `:soma_only` (both dendrites disconnected; a point-neuron control: the original also
  routes the dendritic inputs to the soma, here connect the inputs to `:s`).
- `d = 4um`: dendrite diameter. `VFT = Vector{Float32}`: state type (the original is `Float32`; `Vector{Float64}` runs the
  same scheme in double precision).
- `kwargs...`: any other `Tripod` field (`name`, `adex`, `spike`, `soma_syn`, `dend_syn`, `v_s`, ...) replaces the value below.

# Parameters (source: original code, 8698e1a)
| Item | Value | Original |
|:-----|:------|:---------|
| soma | `C` 281 pF, `gl` 40 nS, `El` = `Vr` = -70.6 mV, `ΔT` 2 mV, `Vt` -50.4 mV, `a` 4 nS, `b` 80.5 pA, `τw` 144 ms | `AdExParams` (`soma.jl`) |
| spike test | `V_s >= Vspike = -50.4 mV` on the potential at the start of the step, before integrating | `simulation.jl:165` (`v_soma >= AdEx.θ`) |
| spike shape | `V_s` = 21 mV on the spike step (`PaperScheme(AP_spike)`), 20 mV during backpropagation (`spike.AP_membrane`), `up = 1 ms` | `PostSpike` (`AP_membrane` 21, `BAP` 20), `AdEx.up` |
| refractory | `τabs = 2.1 ms`: with `dt = 0.1 ms` the spike step, 9 backpropagation and 21 refractory steps (`AdEx.idle` 2 ms, counter 30, 21 steps at `u_r`) | `simulation.jl:165-190` |
| adaptive threshold | none (`At = 0`) | `soma_th = AdEx.θ` constant |
| geometry | cylinders of diameter `d` = 4 um, `Ri` 200 Ω cm; human `Rd` 38907 Ω cm², `Cd` 0.5 μF/cm²; mouse `Rd` 1700, `Cd` 1; lengths by `config`; `g_m = π d l / Rd`, `g_ax = π d² / (4 Ri l)`, `C = Cd π d l` | `dendrites.jl`, `parameters.jl` |
| initial state | `V_s = V_d1 = V_d2 = El`, `w = 0`, no synaptic activity | `run_tripod` |
| soma receptors | AMPA E 0, τr 0.25, τd 2, g0 0.73; GABAa E -75, τr 0.3, τd 15, g0 0.38 (`TripodSomaSynapsePaper`) | `DuarteGluSoma`, `MilesGabaSoma` |
| dendritic receptors | AMPA E 0, τr 0.26, τd 2, g0 0.73; NMDA τr 8, τd 35, g0 1.31; GABAa E -75, τr 4.8, τd 29, g0 0.27; GABAb E -90, τr 30, τd 400, g0 0.006 (`TripodDendSynapsePaper`); mouse glutamatergic: `DuarteGluDend` | `EyalGluDend`, `MilesGabaDend`, `DuarteGluDend` |
| NMDA block | `b` 3.36, `k` -0.077/mV, `Mg` 1 mM, exponential `(1 + x/32)^32` (`EyalNMDA32`): block 0.0225 at -70 mV, 0.1508 at -40 mV. The mouse also uses these values (the preprint prints 3.57 / -0.062: `MouseNMDA`) | `equations.jl:17-20` |
| synaptic decay | `(1 + x/32)^32` (`ReceptorSynapse(decay = 32)`) | `double_exp` |
| currents | dendritic synaptic current clamped to ±1500 pA, somatic not clamped; external currents added as a kick `dt I / C` | `_Δv_dend`, `simulation.jl:82-84` |
| integration | `PaperScheme`: split two-stage scheme with frozen axial currents, see `PaperScheme` | `ΔvDend2`, `ΔvSoma2` |

The original multiplies the dendritic derivative during the backpropagation by `postspike.BAP_gax`; its default is 1 and it is
set to 2 only in a revision-figure script (`FigRevision/backpropagation.jl`), so it is not implemented here.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
T = SNN.TripodPaper(N = 1, physiology = :human, config = :distal_proximal)
T.receptors_d1.glu[1] = 100f0                 # one event of weight 100 on d1 (paper Fig. 8)
for _ in 1:3000
    SNN.integrate!(T, T.param, 0.1f0)         # dt = 0.1 ms
end
```
"""
function TripodPaper(;
    N::Integer = 100,
    physiology::Symbol = :human,
    config::Symbol = :distal_proximal,
    d = 4um,
    VFT::Type{<:AbstractVector} = Vector{Float32},
    kwargs...,
)
    FT = eltype(VFT)
    physiology in (:human, :mouse) || throw(ArgumentError("physiology must be :human or :mouse, got :$physiology"))
    haskey(_PAPER_CONFIGS, config) ||
        throw(ArgumentError("config must be one of $(keys(_PAPER_CONFIGS)), got :$config"))
    param = TripodParameter(
        ds = copy(_PAPER_CONFIGS[config]),
        physiology = physiology === :human ? human_dend : mouse_dend,
        d = d,
    )
    adex = AdExParameter(FT = FT, C = 281pF, gl = 40nS, El = -70.6mV, Vr = -70.6mV, Vt = -50.4mV, ΔT = 2mV, a = 4nS, b = 80.5pA, τw = 144ms)
    spike = PostSpike(FT = FT, At = 0mV, τA = 30ms, AP_membrane = 20mV, τabs = 2.1ms, up = 1ms)
    soma_syn = convert_precision(TripodSomaSynapsePaper, FT)
    dend_syn = convert_precision(physiology === :human ? TripodDendSynapsePaper : TripodDendSynapsePaperMouse, FT)
    opts = (
        N = N,
        VFT = VFT,
        name = "TripodPaper",
        param = param,
        adex = adex,
        spike = spike,
        soma_syn = soma_syn,
        dend_syn = dend_syn,
        Vspike = -50.4mV,
        scheme = PaperScheme(),
        v_s = fill(adex.El, N),
        v_d1 = fill(adex.El, N),
        v_d2 = fill(adex.El, N),
        w_s = zeros(FT, N),
    )
    return Tripod(; merge(opts, NamedTuple(kwargs))...)
end

export TripodPaper, PaperScheme, HeunScheme, AbstractTripodScheme
