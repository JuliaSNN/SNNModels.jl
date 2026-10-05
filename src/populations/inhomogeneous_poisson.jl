
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
The noise uses `randcache_β`, the spike draw uses `rand(Float32)` from the global RNG.

# Fields
- `id`, `param`, `name = "InhomogeneousPoisson"`, `N::Int32 = 100`, `fire::BitVector`.
- `r::Vector{Float32}`: rate offsets ``\rho_i``; `noise::Vector{Float32}`: ``\eta_i``.
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
    r::VFT= ones(Float32, N) * param.r0
    noise::VFT = zeros(Float32, N)
    # sparse connectivity
    randcache_β::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
end

function Population(p::InhomogeneousPoissonParam; kwargs...)
    return InhomogeneousPoisson(param = p; kwargs...)
end

function integrate!(p::InhomogeneousPoisson, param::InhomogeneousPoissonParam, dt::Float32)
    @unpack N, randcache_β, fire = p
    ## Inhomogeneous Poisson process
    @unpack r0, β, τ, rate_timescale = param
    @unpack noise, r = p
    R(x::Float32, v0::Float32 = 0.0f0) = x > 0.0f0 ? x : v0

    re::Float32 = 0.0f0
    cc::Float32 = 0.0f0
    Erate::Float32 = 0.0f0
    rand!(randcache_β)
    fire .= false
    @inbounds @fastmath for i = 1:N
        re = randcache_β[i] - 0.5f0
        cc = 1.0f0 - dt / τ
        noise[i] = (noise[i] - re) * cc + re
        Erate = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
        r[i] += (r0 - Erate) / rate_timescale * dt
        @assert Erate >= 0
        p_spike = 1f0 - exp(-Erate * dt)
        fire[i] = rand(Float32) < p_spike
    end
end

export InhomogeneousPoisson, InhomogeneousPoissonParam, integrate!
