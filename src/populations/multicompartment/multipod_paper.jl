# `PaperScheme` of `Multipod` and the `MultipodPaper` configuration.
#
# Purpose: the split two-stage scheme and the spike rules of the published Tripod code (see `PaperScheme`, `tripod_paper.jl`)
#          for any number of dendrites, and the preset with the parameters of that code.
# Inputs:  a `Multipod{D}` with `scheme = PaperScheme()` and `ReceptorSynapse` models.
# Outputs: `_multipod_chunk!(::PaperScheme, ...)`, `MultipodPaper`.
# The step is `_paper_step!(::Tripod)` with the two dendrites replaced by a tuple of `D` dendrites; the axial currents enter the
# somatic step as the left-to-right sum `((c1 + c2) + c3) + ...`, which is `c1 + c2` of `Tripod` for D = 2.

# ---------------------------------------------------------------------------------------------------------------------------------
# PaperScheme step
# ---------------------------------------------------------------------------------------------------------------------------------

function _multipod_chunk!(::PaperScheme, p::Multipod, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int)
    _paper_check_dt(dt)
    @unpack fire = p

    # advance the conductances of every compartment (discrete update of the published code when `exact = false`)
    update_synapses!(p, p.soma_syn, p.receptors_s, p.synvars_s, dt, rng)
    for (receptors, synvars) in zip(_receptors_dends(p), _synvars_dends(p))
        update_synapses!(p, p.dend_syn, receptors, synvars, dt, rng)
    end

    @inbounds for i in rng
        fire[i] = false
        p.Δv[i, 1] = zero(eltype(p.Δv))
    end
    _multipod_paper_step!(p, dt, rng, n_abs, n_ref)
end

# One step of the original loop for the neurons `rng`; see `PaperScheme` for the list of its rules.
function _multipod_paper_step!(p::Multipod{D}, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int) where {D}
    @unpack v_s, w_s, fire, θ, tabs, I, I_d, adex, spike, Vspike, scheme = p
    @unpack soma_syn, dend_syn, synvars_s = p
    @unpack C, gl, El, ΔT, a, τw, Vr, Vt, b = adex
    @unpack AP_membrane, At, τA = spike
    v_ds = _v_dends(p)
    svs = _synvars_dends(p)
    dends = _dendrites(p)
    FT = eltype(v_s)
    AP_spike = convert(FT, scheme.AP_spike)
    nabs_f = convert(eltype(tabs), n_abs)
    nref_f = convert(eltype(tabs), n_ref)
    Cinv = inv(C)

    @inbounds for i in rng
        # external currents: Euler kick before the step
        vs = convert(FT, v_s[i] + (dt * Cinv) * I[i])
        vd = ntuple(k -> convert(FT, v_ds[k][i] + (dt / dends[k].C[i]) * I_d[i]), Val(D))
        w = w_s[i]

        t1 = tabs[i] - 1
        th1 = θ[i] + dt * (Vt - θ[i]) / τA

        # axial currents into the dendrites from the potentials at the start of the step (kept fixed over the step)
        c = ntuple(k -> (vs - vd[k]) * dends[k].gax[i], Val(D))

        # two-stage step of every dendrite with its frozen axial current
        step_dends() = ntuple(k -> _paper_dend(vd[k], c[k], dends[k].gm[i], dends[k].El[i], dends[k].C[i], dend_syn, svs[k], i, dt), Val(D))

        spk = false
        if t1 <= 0
            if vs >= Vspike
                spk = true
                vs = AP_spike
                w += b
                # the dendrites do not move in the spike step
            else
                vdn = step_dends()
                wn = convert(FT, w + dt * (muladd(a, vs - El, -w) / τw))
                vs = _paper_soma(vs, wn, θ[i], _sum_left(first(c), Base.tail(c)), gl, El, ΔT, Cinv, soma_syn, synvars_s, i, dt)
                vd, w = vdn, wn
            end
        else
            # backpropagation or refractory period: dendrites in full, then the soma is clamped
            vd = step_dends()
            vs = t1 > nabs_f ? AP_membrane : Vr
        end

        fire[i] = spk
        v_s[i] = vs
        ntuple(k -> (v_ds[k][i] = vd[k]; nothing), Val(D))
        w_s[i] = w
        θ[i] = spk ? th1 + At : th1
        tabs[i] = spk ? nref_f : t1
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------
# MultipodPaper
# ---------------------------------------------------------------------------------------------------------------------------------

@doc raw"""
    MultipodPaper(; N = 100, lengths = [400um, 150um], physiology = :human, d = 4um, VFT = Vector{Float32}, kwargs...) -> Multipod

`Multipod` population configured as the model of the published Tripod code (the parameters, receptors, spike rules and
`PaperScheme` of `TripodPaper`), with one dendrite per entry of `lengths` (fixed lengths, star topology). With two lengths it
is `TripodPaper` with these lengths (bit for bit; `lengths = [400um, 150um]`, the default, is `TripodPaper()`, configuration
`:distal_proximal`). Use `dt = 0.1ms` in `sim!`/`train!` (other steps give a warning, once).

# Keywords
- `lengths`: dendrite lengths (cm, e.g. `[400um, 250um, 150um]`), `1 <= length(lengths) <= 8`; `0um` gives a disconnected
  dendrite (as `soma_only` of `TripodPaper`).
- `physiology::Symbol = :human`: `:human` (`human_dend`, `TripodDendSynapsePaper`) or `:mouse` (`mouse_dend`,
  `TripodDendSynapsePaperMouse`).
- `d = 4um`: dendrite diameter, a number or one diameter per dendrite.
- `VFT = Vector{Float32}`: state type (`Vector{Float64}` runs the same scheme in double precision).
- `kwargs...`: any other `Multipod` field (`name`, `adex`, `spike`, `soma_syn`, `dend_syn`, `v_s`, ...) replaces the preset value.

The somatic parameters, spike shape, refractory period, receptors and NMDA block are those listed in `TripodPaper`; the
dendritic leak reversal is fixed at -70.6 mV (`El_dend`), and all compartments start at -70.6 mV with `w = 0`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
M = SNN.MultipodPaper(N = 1, lengths = [400um, 400um, 150um, 150um])
M.receptors_d1.glu[1] = 100f0                 # one event of weight 100 on d1
for _ in 1:3000
    SNN.integrate!(M, M.param, 0.1f0)         # dt = 0.1 ms
end
```
"""
function MultipodPaper(;
    N::Integer = 100,
    lengths::AbstractVector = [400um, 150um],
    physiology::Symbol = :human,
    d = 4um,
    VFT::Type{<:AbstractVector} = Vector{Float32},
    kwargs...,
)
    FT = eltype(VFT)
    physiology in (:human, :mouse) || throw(ArgumentError("physiology must be :human or :mouse, got :$physiology"))
    all(l -> l isa Real, lengths) || throw(ArgumentError("MultipodPaper takes fixed lengths (numbers), got $(lengths)"))

    # morphology, soma, spike and receptors: the values of `TripodPaper`
    param = MultipodParameter(
        ds = collect(lengths),
        physiology = physiology === :human ? human_dend : mouse_dend,
        d = d,
        El_dend = -70.6mV,
    )
    adex = AdExParameter(FT = FT, C = 281pF, gl = 40nS, El = -70.6mV, Vr = -70.6mV, Vt = -50.4mV, ΔT = 2mV, a = 4nS, b = 80.5pA, τw = 144ms)
    spike = PostSpike(FT = FT, At = 0mV, τA = 30ms, AP_membrane = 20mV, τabs = 2.1ms, up = 1ms)
    soma_syn = convert_precision(TripodSomaSynapsePaper, FT)
    dend_syn = convert_precision(physiology === :human ? TripodDendSynapsePaper : TripodDendSynapsePaperMouse, FT)

    # initial state of the original: every compartment at El, no adaptation
    D = length(lengths)
    v_dends = NamedTuple{_dend_fieldnames(:v_d, D)}(ntuple(_ -> fill(adex.El, N), D))
    opts = (
        N = N,
        VFT = VFT,
        name = "MultipodPaper",
        param = param,
        adex = adex,
        spike = spike,
        soma_syn = soma_syn,
        dend_syn = dend_syn,
        Vspike = -50.4mV,
        scheme = PaperScheme(),
        v_s = fill(adex.El, N),
        v_dends...,
        w_s = zeros(FT, N),
    )
    return Multipod(; merge(opts, NamedTuple(kwargs))...)
end

export MultipodPaper
