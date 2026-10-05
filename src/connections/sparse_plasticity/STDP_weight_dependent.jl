@doc """
    STDPWeightDependent(; η = 1e-3, α = 1, μ_plus = 1, μ_minus = 1, τpre = 20ms, τpost = 20ms,
                          Wmax = 30pF, Wmin = 0pF)

Weight-dependent (soft-bound) pair STDP, all-to-all interaction, event-driven
(Gütig, R., Aharonov, R., Rotter, S., & Sompolinsky, H. (2003). Learning input
correlations through nonlinear temporally asymmetric Hebbian plasticity.
J. Neurosci., 23(9), 3697–3714; Morrison, Diesmann & Gerstner (2008), Biol. Cybern.
98, 459–478; implementation as Auryn `STDPwdConnection`, Zenke & Gerstner 2014).

Traces ``x_{pre}`` (`τpre`) and ``x_{post}`` (`τpost`) jump by 1 at each spike and
decay exponentially; they are read before this step's spikes are added. With
``\\tilde w = w - W_{min}`` and ``\\tilde W = W_{max} - W_{min}``:

- presynaptic spike (LTD):
  ``w \\mathrel{-}= η\\, α\\, \\tilde W^{1-μ_-}\\, \\tilde w^{μ_-}\\, x_{post}``
- postsynaptic spike (LTP):
  ``w \\mathrel{+}= η\\, \\tilde W^{1-μ_+}\\, (W_{max} - w)^{μ_+}\\, x_{pre}``

followed by clamping to `[Wmin, Wmax]`. `η` is a relative learning rate (the
factors ``\\tilde W^{1-μ}`` make the update scale with the weight range):
`μ = 0` is additive STDP with hard bounds and amplitudes ``η \\tilde W`` (LTP) and
``-η α \\tilde W`` (LTD); `μ = 1` is multiplicative STDP, LTD ``= -η α (w - W_{min}) x_{post}``
and LTP ``= η (W_{max} - w) x_{pre}``. With `Wmin = 0` this is
exactly Auryn's `STDPwdConnection` (`learning_rate = η`, `param_alpha = α`,
`param_mu_plus`, `param_mu_minus`).

Fields: `η = 1e-3` (relative learning rate), `α = 1` (LTD/LTP asymmetry), `μ_plus = 1`,
`μ_minus = 1` (weight-dependence exponents), `τpre = τpost = 20ms`, `Wmax = 30pF`,
`Wmin = 0pF`. State: `STDPVariables`. Event-driven with the ordering of `STDP_kernels.jl`,
serial, applied only under `train!`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.STDPWeightDependent(η = 1e-3, α = 1.05, μ_plus = 0.0, μ_minus = 1.0)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 10.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```
"""
STDPWeightDependent

@snn_kw struct STDPWeightDependent{FT = Float32} <: STDPParameter
    η::FT = 1e-3            # learning rate
    α::FT = 1.0             # LTD / LTP asymmetry
    μ_plus::FT = 1.0        # LTP weight-dependence exponent
    μ_minus::FT = 1.0       # LTD weight-dependence exponent
    τpre::FT = 20ms         # Time constant for pre-synaptic spike trace
    τpost::FT = 20ms        # Time constant for post-synaptic spike trace
    Wmax::FT = 30.0pF       # Max weight
    Wmin::FT = 0.0pF        # Min weight
end

# Uses STDPVariables (via plasticityvariables(::STDPParameter, ...)).
function plasticity!(
    c::PT,
    param::STDPWeightDependent,
    variables::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack η, α, μ_plus, μ_minus, τpre, τpost, Wmax, Wmin = param
    @unpack tpre, tpost = variables
    W̃ = Wmax - Wmin
    fudge_dep = η * α * W̃^(1 - μ_minus)
    fudge_pot = η * W̃^(1 - μ_plus)
    # pre spike: LTD ∝ (w - Wmin)^μ_minus * x_post[i]
    f_pre = (w, i, j) -> -fudge_dep * max(w - Wmin, 0.0f0)^μ_minus * tpost[i]
    # post spike: LTP ∝ (Wmax - w)^μ_plus * x_pre[j]
    f_post = (w, i, j) -> fudge_pot * max(Wmax - w, 0.0f0)^μ_plus * tpre[j]
    _pair_stdp!(c, variables, f_pre, f_post, τpre, τpost, Wmin, Wmax, dt, T)
end

export STDPWeightDependent
