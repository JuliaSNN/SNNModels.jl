
@doc raw"""
    InhomogeneousPoissonParam{FT = Float32}(; β = 0, τ = 50ms, r0 = 1kHz, rate_timescale = 400ms)

Parameters of `InhomogeneousPoisson`.
- `β::FT = 0.0`: gain of the filtered noise on the rate.
- `τ::FT = 50ms`: time constant of the noise low-pass filter (ms).
- `r0::FT = 1kHz`: target rate of each neuron.
- `rate_timescale::FT = 400ms`: time constant of the feedback that keeps the mean rate at `r0`.
"""
InhomogeneousPoissonParam

@snn_kw struct InhomogeneousPoissonParam{FT = Float32} <: AbstractPopulationParameter
    β::FT = 0.0
    τ::FT = 50.0ms
    r0::FT = 1kHz
    rate_timescale::FT = 400ms
end

@doc raw"""
    InhomogeneousPoisson(; N = 100, param::InhomogeneousPoissonParam, name = "InhomogeneousPoisson")
    Population(param::InhomogeneousPoissonParam; N, kwargs...)

Population of independent inhomogeneous Poisson generators, each with its own slowly
fluctuating rate. `param` has no default.

# Equations
For neuron ``i`` (``\xi_{i,t}`` uniform in ``[-0.5, 0.5]``):
```math
\begin{aligned}
\eta_i &\leftarrow \eta_i\,(1 - dt/\tau) + \xi_{i,t}\, dt/\tau \\
\nu_i &= \left[\tfrac{r_0}{2}\, F(\beta \eta_i) + \rho_i\right]_+, \qquad
         F(x) = \begin{cases} x & x > 0 \\ 1 & x \le 0 \end{cases} \\
\rho_i &\leftarrow \rho_i + (r_0 - \nu_i)\, dt / \tau_{rate} \\
P(\text{spike}) &= 1 - e^{-\nu_i dt}
\end{aligned}
```
with ``\tau_{rate}`` = `rate_timescale`; ``\rho_i`` (field `r`) is initialised to ``r_0``.
The noise uses `randcache_β` (one uniform number per neuron and step). The spikes are sampled sparsely: every neuron
fires with probability ``1 - e^{-\nu_i dt}`` independently, but the candidates are generated with the largest probability of the
step by geometric skipping and thinned (see `Poisson`), so the spike draws cost one random number per spike instead of one per
neuron (above a per-step probability of 3 % every neuron is tested with one vectorised uniform draw, as in SNNModels 1.9). The sequence of
random numbers differs from SNNModels 1.9 (global RNG, `Random.seed!`).

# Fields
- `id`, `param`, `name = "InhomogeneousPoisson"`, `N::Int32 = 100`, `fire::Vector{Bool}`.
- `r::VFT`: rate offsets ``\rho_i``; `noise::VFT`: ``\eta_i``. `VFT = Vector{Float32}` by default; `VFT = Vector{Float64}` with
  `param = InhomogeneousPoissonParam(FT = Float64)` gives a `Float64` population (see `float_type`).
- `pspike::VFT`: spike probability ``1 - e^{-\nu_i dt}`` of each neuron in the last step.
- `spikes::SpikeIndex`: the ascending list of the neurons that fired in the last step (fed by `integrate!`).
- `randcache_β`; `records::Dict`; `targets::Dict` (unused).

Reference not given in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
P = SNN.Population(SNN.InhomogeneousPoissonParam(r0 = 10Hz, β = 5); N = 100)
SNN.monitor!(P, [:fire])
SNN.sim!([P]; duration = 500ms)
```
"""
InhomogeneousPoisson

@snn_kw struct InhomogeneousPoisson{VFT = Vector{Float32},IT = Int32} <: AbstractPopulation
    id::String = randstring(12)
    param::InhomogeneousPoissonParam
    name::String = "InhomogeneousPoisson"
    ##
    N::IT=100
    fire::VBT = falses(N)
    r::VFT= ones(eltype(VFT), N) * param.r0
    noise::VFT = zeros(eltype(VFT), N)
    pspike::VFT = zeros(eltype(VFT), N)  # spike probability of each neuron in the last step
    spikes::SpikeIndex = fed_spike_index(N)  # ascending list of the neurons with fire[j], fed by `integrate!`
    randcache_β::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
end

function Population(p::InhomogeneousPoissonParam; kwargs...)
    return InhomogeneousPoisson(param = p; kwargs...)
end

function integrate!(p::InhomogeneousPoisson, param::InhomogeneousPoissonParam, dt::AbstractFloat)
    @unpack N, randcache_β, fire, spikes, pspike = p
    ## Inhomogeneous Poisson process
    @unpack r0, β, τ, rate_timescale = param
    @unpack noise, r = p
    _check_state_lengths(N, randcache_β, fire, pspike, noise, r)

    rand!(randcache_β)
    FT = eltype(pspike)
    cc = one(FT) - dt / τ
    pmax = zero(FT)
    # rate of every neuron, vectorised: noise and rate offset as in SNNModels 1.9, and the spike probability of the step
    @inbounds @simd for i = 1:N
        re = randcache_β[i] - FT(0.5)
        ni = (noise[i] - re) * cc + re
        noise[i] = ni
        x = ni * β
        E = r0 / 2 * ifelse(x > zero(FT), x, one(FT)) + r[i]
        Erate = ifelse(E > zero(FT), E, zero(FT))
        r[i] += (r0 - Erate) / rate_timescale * dt
        pi_ = one(FT) - @inline(exp(-Erate * dt))
        pspike[i] = pi_
        pmax = max(pmax, pi_)
    end
    # sparse sampling of the spikes (see `Poisson`); the uniform cache is free again after the noise update
    if pmax >= _POP_DENSE_SPIKE_P
        _pop_dense!(j -> @inbounds(pspike[j]), fire, spikes, randcache_β, Int(N))
    else
        _pop_sparse_start!(fire, spikes)
        _pop_thinned!(j -> Float64(@inbounds(pspike[j])), fire, spikes, Int(N), Float64(pmax))
    end
    return nothing
end

spike_index(p::InhomogeneousPoisson) = p.spikes

# one precision per population (see `IF`)
function snn_kw_finalize(::Type{T}, f::NamedTuple) where {T<:InhomogeneousPoisson}
    _check_precision("InhomogeneousPoisson", _state_float(T, :r), "param" => f.param)
    return f
end

export InhomogeneousPoisson, InhomogeneousPoissonParam, integrate!
