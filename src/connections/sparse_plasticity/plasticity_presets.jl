#=
Named parameter sets (presets) of the long-term plasticity rules
================================================================

Purpose: the parameter sets of the Tripod plasticity studies, expressed with the existing rule types.
  - voltage-based STDP (Clopath et al. 2010), `vSTDPParameter` / `vSTDPDenseParameter`:
        lkd_vstdp, clopath_vstdp_visualcortex, bono_vstdp
  - triplet STDP (Pfister & Gerstner 2006), `STDPTriplet`:  pfister_visualcortex
  - nonlinear temporally asymmetric Hebbian plasticity (Gütig et al. 2003), `STDPWeightDependent`:  gutig_nltah

Inputs: keyword overrides of any field of the returned rule.
Outputs: a rule object, to be passed as `LTPParam` to a `SpikingSynapse`.
No new update equations are defined here.
=#

# ── Voltage-based STDP ────────────────────────────────────────────────────────────────────────────

# Parameter sets of the Tripod code: (a⁻, a⁺, θ⁻, θ⁺, τu, τv, τx, ε, Wmin, Wmax), units pA/mV, pA/mV², mV, ms, pF.
const _LKD_VSTDP = (a_ltd = 8.0e-4, a_ltp = 1.4e-3, θ_ltd = -70.0, θ_ltp = -49.0, τu = 10.0, τv = 7.0, τx = 15.0,
                    ε = 0.0, Wmin = 0.78, Wmax = 11.4)
const _CLOPATH_VC_VSTDP = (a_ltd = 14.0e-5 / 2.5, a_ltp = 8.0e-5 / 2.5, θ_ltd = -70.6, θ_ltp = 25.3, τu = 10.0,
                           τv = 7.0, τx = 15.0, ε = 1.0, Wmin = 0.5, Wmax = 1.5)
const _BONO_VSTDP = (a_ltd = 5.0e-4 / 2.5, a_ltp = 1.5e-3 / 2.5, θ_ltd = -69.0, θ_ltp = -15.0, τu = 35.0, τv = 35.0,
                     τx = 15.0, ε = 1.0, Wmin = 0.5, Wmax = 1.5)

"""
    _vstdp_preset(set; convention, dt, dense, kwargs...)

Build `vSTDPParameter` (or `vSTDPDenseParameter` with `dense = true`) from a parameter set of the Tripod code.
`convention = :tripod` converts the amplitudes so that the library rule computes the update of the Tripod code
at time step `dt` (see `lkd_vstdp`); `convention = :verbatim` copies them. Keyword arguments override fields.
"""
function _vstdp_preset(set::NamedTuple; convention::Symbol = :tripod, dt::Real = 0.1ms, dense::Bool = false, kwargs...)
    # amplitudes in the convention of the library rule
    if convention === :tripod
        dt > 0 || throw(ArgumentError("the :tripod convention needs dt > 0"))
        A_LTD = dt * set.a_ltd            # the Tripod code multiplies the per-spike LTD by dt
        A_LTP = set.a_ltp * (set.τx - dt) # its trace jumps by 1 and decays in the step of the spike
    elseif convention === :verbatim
        A_LTD, A_LTP = set.a_ltd, set.a_ltp
    else
        throw(ArgumentError("convention must be :tripod or :verbatim, got :$convention"))
    end

    fields = (; A_LTD, A_LTP, θ_LTD = set.θ_ltd, θ_LTP = set.θ_ltp, τu = set.τu, τv = set.τv, τx = set.τx,
              Wmin = set.Wmin, Wmax = set.Wmax, delay_uv = set.ε)
    Rule = dense ? vSTDPDenseParameter : vSTDPParameter
    return Rule(; merge(fields, NamedTuple(kwargs))...)
end

const _VSTDP_PRESET_CONVENTION = raw"""
# Convention
The values are those of the Tripod plasticity code (Quaresima et al., `STDP_params.jl`). That code integrates
the rule of [`vSTDPParameter`](@ref) with two other scalings, both converted by `convention = :tripod`
(default) at the time step `dt` (default 0.1 ms, the step of the Tripod code):
- the presynaptic trace jumps by 1 and decays in the step of the spike, ``x^{T} = (τ_x - dt)\, x``, so
  `A_LTP = a⁺ (τx - dt)`;
- the depression at a presynaptic spike is ``dt\, a^-\, [u - θ_{LTD}]_+``, so `A_LTD = dt a⁻`.
With these values and the same `dt` the library rule computes the update of the Tripod code. With
`convention = :verbatim` the amplitudes are copied (`A_LTD = a⁻`, `A_LTP = a⁺`), the convention of
Clopath et al. (2010) and Litwin-Kumar & Doiron (2014): per-spike depression without `dt`, trace jump
``1/τ_x``. In both conventions `delay_uv` is the delay ``ε`` of the parameter set.

`dense = true` returns [`vSTDPDenseParameter`](@ref); keyword arguments override any field (e.g. `Wmax`, `FT`).
"""

@doc """
    lkd_vstdp(; convention = :tripod, dt = 0.1ms, dense = false, kwargs...)

Voltage-based STDP (Clopath et al. 2010) with the amplitudes, thresholds and time constants of the
excitatory synapses of Litwin-Kumar & Doiron (2014) and the weight bounds of the Tripod code:
`a⁻ = 8e-4` pA/mV, `a⁺ = 1.4e-3` pA/mV², `θ_LTD = -70` mV, `θ_LTP = -49` mV, `τu = 10` ms, `τv = 7` ms,
`τx = 15` ms, no delay (`delay_uv = 0`), `Wmin = 0.78` pF, `Wmax = 11.4` pF. Returns a [`vSTDPParameter`](@ref).

The defaults of `vSTDPParameter` share `θ`, `τv`, `τx` and the verbatim amplitudes but have `τu = 20` ms and
the bounds 0.1-30 pF.

$(_VSTDP_PRESET_CONVENTION)
# References
Clopath, C., Büsing, L., Vasilaki, E. & Gerstner, W. (2010). Connectivity reflects coding: a model of
voltage-based STDP with homeostasis. Nature Neuroscience 13, 344-352.

Litwin-Kumar, A. & Doiron, B. (2014). Formation and maintenance of neuronal assemblies through synaptic
plasticity. Nature Communications 5, 5319.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.AdEx(N = 10, param = SNN.AdExParameter())
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 2.0), LTPParam = SNN.lkd_vstdp())
```
"""
lkd_vstdp(; kwargs...) = _vstdp_preset(_LKD_VSTDP; kwargs...)

@doc """
    clopath_vstdp_visualcortex(; convention = :tripod, dt = 0.1ms, dense = false, kwargs...)

Voltage-based STDP (Clopath et al. 2010) with the visual-cortex parameter set of the Tripod code:
`a⁻ = 14e-5/2.5` pA/mV, `a⁺ = 8e-5/2.5` pA/mV², `θ_LTD = -70.6` mV, `θ_LTP = +25.3` mV, `τu = 10` ms,
`τv = 7` ms, `τx = 15` ms, `delay_uv = 1` ms, `Wmin = 0.5`, `Wmax = 1.5`. Returns a [`vSTDPParameter`](@ref).

The visual-cortex fit of Clopath et al. (2010) has `A_LTD = 14e-5`, `A_LTP = 8e-5` and `θ_LTP = -45.3` mV;
the Tripod code divides both amplitudes by 2.5 and sets `θ_LTP = +25.3` mV (potentiation only while the
potential is above +25.3 mV). Pass `θ_LTP = -45.3` (and the amplitudes) to use the published values.

$(_VSTDP_PRESET_CONVENTION)
# References
Clopath, C., Büsing, L., Vasilaki, E. & Gerstner, W. (2010). Connectivity reflects coding: a model of
voltage-based STDP with homeostasis. Nature Neuroscience 13, 344-352.
"""
clopath_vstdp_visualcortex(; kwargs...) = _vstdp_preset(_CLOPATH_VC_VSTDP; kwargs...)

@doc """
    bono_vstdp(; convention = :tripod, dt = 0.1ms, dense = false, kwargs...)

Voltage-based STDP (Clopath et al. 2010) for dendritic synapses, the parameter set of the Tripod
dendrite simulations after Bono & Clopath (2017): `a⁻ = 5e-4/2.5` pA/mV, `a⁺ = 1.5e-3/2.5` pA/mV²,
`θ_LTD = -69` mV, `θ_LTP = -15` mV, `τu = τv = 35` ms, `τx = 15` ms, `delay_uv = 1` ms, `Wmin = 0.5`,
`Wmax = 1.5`. Returns a [`vSTDPParameter`](@ref).

Use it on a synapse onto a dendritic compartment: `SpikingSynapse(pre, tripod, :glu, :d1; LTPParam =
bono_vstdp())` filters and thresholds the potential `tripod.v_d1` of that dendrite, so depression and
potentiation follow the local dendritic potential and not the somatic one.
A `SynapseNormalization([syn]; param = MultiplicativeNorm(τ = 20ms))` per dendritic synapse keeps the
summed weight of each dendrite constant, as in the Tripod code.

$(_VSTDP_PRESET_CONVENTION)
# References
Bono, J. & Clopath, C. (2017). Modeling somatic and dendritic spike mediated plasticity at the single
neuron and network level. Nature Communications 8, 706.

Clopath, C., Büsing, L., Vasilaki, E. & Gerstner, W. (2010). Connectivity reflects coding: a model of
voltage-based STDP with homeostasis. Nature Neuroscience 13, 344-352.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
T = SNN.Tripod(N = 1)
E = SNN.Poisson(N = 20, param = SNN.PoissonParameter(10Hz))
syn = SNN.SpikingSynapse(E, T, :glu, :d1; conn = (p = 1.0, μ = 1.0), LTPParam = SNN.bono_vstdp())
norm = SNN.SynapseNormalization([syn]; param = SNN.MultiplicativeNorm(τ = 20ms))
SNN.train!(model = SNN.compose(; T, E, syn, norm), duration = 500ms)
SNN.plasticity_value(syn, :W)
```
"""
bono_vstdp(; kwargs...) = _vstdp_preset(_BONO_VSTDP; kwargs...)

# ── Triplet STDP ──────────────────────────────────────────────────────────────────────────────────

@doc raw"""
    pfister_visualcortex(alltoall::Bool = true, full::Bool = true; kwargs...)

Triplet STDP ([`STDPTriplet`](@ref)) with the visual-cortex parameter sets of Pfister & Gerstner (2006),
as in the Tripod code: all-to-all (`alltoall = true`) or nearest-spike (`false`) interaction, full
(`full = true`, four amplitudes) or minimal (`false`, ``A_2^+ = A_3^- = 0``) model. ``τ_+ = 16.8`` ms and
``τ_- = 33.7`` ms in all four:

| interaction, model | ``A_2^+`` | ``A_3^+`` | ``A_2^-`` | ``A_3^-`` | ``τ_x`` (ms) | ``τ_y`` (ms) |
|:--|--:|--:|--:|--:|--:|--:|
| all-to-all, full | 5e-10 | 6.2e-3 | 7e-3 | 2.3e-4 | 101 | 125 |
| all-to-all, minimal | 0 | 6.5e-3 | 7.1e-3 | 0 | (101) | 125 |
| nearest-spike, full | 8.8e-11 | 5.3e-2 | 6.6e-3 | 3.1e-3 | 714 | 40 |
| nearest-spike, minimal | 0 | 5.2e-2 | 8e-3 | 0 | (714) | 40 |

In the minimal models ``τ_x`` is unused (``A_3^- = 0``); it is set to the value of the full model of the
same interaction. The amplitudes are absolute weight changes per pair or triplet, in the units of the
weights (`W`): the default bounds of `STDPTriplet` are `Wmin = 0`, `Wmax = 30` pF. Keyword arguments
override any field (e.g. `Wmax`, `FT`).

Equations (``ε`` = before the increments of the step): at a presynaptic spike
``w \leftarrow w - o_1(t-ε)\,(A_2^- + A_3^- r_2(t-ε))``, at a postsynaptic spike
``w \leftarrow w + r_1(t-ε)\,(A_2^+ + A_3^+ o_2(t-ε))``; then the traces of the spiking neurons are
increased by 1 (all-to-all) or set to 1 (nearest-spike).

# References
Pfister, J.-P. & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity.
Journal of Neuroscience 26, 9673-9682 (Table 4, visual cortex data of Sjöström et al. 2001).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.pfister_visualcortex(false, true; Wmax = 10.0)   # nearest-spike, full model
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 5.0), LTPParam = rule)
```
"""
function pfister_visualcortex(alltoall::Bool = true, full::Bool = true; kwargs...)
    common = (τ_plus = 16.8ms, τ_minus = 33.7ms, nearest = !alltoall)
    if alltoall
        set = full ? (A2_plus = 5.0e-10, A3_plus = 6.2e-3, A2_minus = 7.0e-3, A3_minus = 2.3e-4, τ_x = 101ms, τ_y = 125ms) :
                     (A2_plus = 0.0, A3_plus = 6.5e-3, A2_minus = 7.1e-3, A3_minus = 0.0, τ_x = 101ms, τ_y = 125ms)
    else
        set = full ? (A2_plus = 8.8e-11, A3_plus = 5.3e-2, A2_minus = 6.6e-3, A3_minus = 3.1e-3, τ_x = 714ms, τ_y = 40ms) :
                     (A2_plus = 0.0, A3_plus = 5.2e-2, A2_minus = 8.0e-3, A3_minus = 0.0, τ_x = 714ms, τ_y = 40ms)
    end
    return STDPTriplet(; merge(common, set, NamedTuple(kwargs))...)
end

# ── Gütig et al. 2003 ─────────────────────────────────────────────────────────────────────────────

@doc raw"""
    gutig_nltah(; τ, λ, μ, α = 1, Wmin = 0, Wmax = 1, kwargs...)

Nonlinear temporally asymmetric Hebbian plasticity (NLTAH) of Gütig et al. (2003), as the weight-dependent
pair rule [`STDPWeightDependent`](@ref) with `η = λ`, `α = α`, `μ_plus = μ_minus = μ`, `τpre = τpost = τ`.

With the normalised weight ``\hat w = (w - W_{min})/(W_{max} - W_{min})`` and every pre/post spike pair
(all-to-all), ``Δt = t_{post} - t_{pre}``:
```math
Δ\hat w = λ\,(1 - \hat w)^μ\, e^{-Δt/τ} \quad (Δt > 0), \qquad
Δ\hat w = -λ\,α\,\hat w^μ\, e^{Δt/τ} \quad (Δt < 0),
```
the weight being read before the update. `μ = 0` is additive STDP, `μ = 1` multiplicative STDP.
With `Wmin = 0`, `Wmax = 1` (the defaults) ``\hat w = w``, the formulation of the paper. A pre and a post
spike in the same time step do not interact (see `STDPWeightDependent`).

`τ`, `λ` and `μ` are required; keyword arguments override any field of `STDPWeightDependent`.

# References
Gütig, R., Aharonov, R., Rotter, S. & Sompolinsky, H. (2003). Learning input correlations through
nonlinear temporally asymmetric Hebbian plasticity. Journal of Neuroscience 23, 3697-3714.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
rule = SNN.gutig_nltah(τ = 20ms, λ = 0.01, μ = 0.4, α = 1.05)
```
"""
function gutig_nltah(; τ::Real, λ::Real, μ::Real, α::Real = 1.0, Wmin::Real = 0.0, Wmax::Real = 1.0, kwargs...)
    fields = (η = λ, α = α, μ_plus = μ, μ_minus = μ, τpre = τ, τpost = τ, Wmin = Wmin, Wmax = Wmax)
    return STDPWeightDependent(; merge(fields, NamedTuple(kwargs))...)
end

export lkd_vstdp, clopath_vstdp_visualcortex, bono_vstdp, pfister_visualcortex, gutig_nltah
