"""
    PoissonParameter <: AbstractPopulationParameter
    PoissonParameter(rate::Real)                   -> PoissonHomoParameter
    PoissonParameter(rate::AbstractVector{<:AbstractFloat}) -> PoissonHetParameter

Abstract parameter type of the `Poisson` population, and convenience constructor: a scalar rate
gives a homogeneous population, a vector of rates (one per neuron) a heterogeneous one.
Rates are in the library unit system (use `Hz`, e.g. `10Hz`).
"""
abstract type PoissonParameter <: AbstractPopulationParameter end

"""
    PoissonHomoParameter{FT = Float32}(; rate = 1Hz)

Same firing rate `rate` (Hz units) for all neurons of a `Poisson` population.
"""
PoissonHomoParameter

@snn_kw struct PoissonHomoParameter{FT = Float32} <: PoissonParameter
    rate::FT = 1Hz
end

"""
    PoissonHetParameter{FT = Float32}(; rate = 1Hz)

Per-neuron firing rates for a `Poisson` population; `rate` must be a vector of length `N`
(construct it with `PoissonParameter(rates::Vector)`, which sets `FT = Vector{Float32}`).
"""
PoissonHetParameter

@snn_kw struct PoissonHetParameter{FT = Float32} <: PoissonParameter
    rate::FT = 1Hz
end

function PoissonParameter(rate::FT) where FT <: Real
    return PoissonHomoParameter(Float32(rate))
end

function PoissonParameter(rate::AbstractVector{FT}) where FT <: AbstractFloat
    return PoissonHetParameter(Float32.(rate))
end

@snn_kw mutable struct Poisson{VFT = Vector{Float32}, IT = Int32, PP <: PoissonParameter} <: AbstractPopulation
    id::String = randstring(12)
    name::String = "Poisson"
    param::PP = PoissonHomoParameter()
    N::IT = 100
    randcache::VFT = rand(N)
    fire::VBT = zeros(Bool, N)
    records::Dict = Dict()
end

function Population(p::PP; kwargs...) where PP <: PoissonParameter
    return Poisson(param = p; kwargs...)
end

@doc raw"""
    Poisson(; N = 100, param = PoissonHomoParameter(), name = "Poisson", kwargs...)
    Population(param::PoissonParameter; N, kwargs...)

Population of independent Poisson spike generators. It has no input: connections cannot target
it, it is used as presynaptic population.

# Equations
In each step of length ``dt`` neuron ``i`` fires with probability
```math
P(\text{spike}) = \nu_i\, dt ,
```
the Bernoulli approximation of a Poisson process of rate ``\nu_i`` (accurate when
``\nu_i dt \ll 1``). ``\nu_i`` is `param.rate` (homogeneous) or `param.rate[i]`
(heterogeneous).

# Integration
`rand!(randcache)`, then `fire[i] = randcache[i] < rate * dt`.

# Fields
- `id::String = randstring(12)`, `name::String = "Poisson"`.
- `param::PoissonParameter = PoissonHomoParameter()` (1 Hz).
- `N::Int32 = 100`; `randcache::Vector{Float32}`; `fire::Vector{Bool}`; `records::Dict`.

# References
[Poisson model of spike generation (D. Heeger, handout)](https://www.cns.nyu.edu/~david/handouts/poisson.pdf), linked in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
P  = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
Ph = SNN.Population(SNN.PoissonParameter(collect(range(1Hz, 50Hz, length = 100))); N = 100)
SNN.monitor!([P, Ph], [:fire])
SNN.sim!([P, Ph]; duration = 1s)
SNN.firing_rate(P; interval = 0:10ms:1s)
```
"""
Poisson

function integrate!(p::Poisson, param::PoissonHomoParameter , dt::Float32) 
    @unpack N, randcache, fire = p
    @unpack rate = param
    rand!(randcache)
    @inbounds for i = 1:N
        fire[i] = randcache[i] < rate * dt
    end
end

function integrate!(p::Poisson, param::PoissonHetParameter , dt::Float32) 
    @unpack N, randcache, fire = p
    @unpack rate = param
    rand!(randcache)
    @inbounds for i = 1:N
        fire[i] = randcache[i] < rate[i] * dt
    end
end

@doc raw"""
    VariablePoissonParameter(; β = 0, τ = 50ms, r0 = 1kHz)

Parameters of `VariablePoisson`.
- `β::Float32 = 0.0`: gain of the filtered noise on the rate.
- `τ::Float32 = 50ms`: time constant of the noise low-pass filter (ms).
- `r0::Float32 = 1kHz`: target total rate of the population.
"""
VariablePoissonParameter

@snn_kw struct VariablePoissonParameter <: AbstractPopulationParameter
    β::Float32 = 0.0
    τ::Float32 = 50.0ms
    r0::Float32 = 1kHz
end


@doc raw"""
    VariablePoisson(; N = 100, param::VariablePoissonParameter, name = "VariablePoisson")

Population of `N` spike generators sharing one time-varying total rate. `param` has no
default and must be given. Not constructible through `Population`.

# Equations
One scalar noise ``\eta`` (``\xi_t`` uniform in ``[-0.5, 0.5]``), rate offset ``\rho`` and
total rate ``\nu``:
```math
\begin{aligned}
\eta &\leftarrow \eta\,(1 - dt/\tau) + \xi_t\, dt/\tau \\
\nu &= \left[\tfrac{r_0}{2}\, F(\beta \eta) + \rho\right]_+, \qquad
      F(x) = \begin{cases} x & x > 0 \\ 1 & x \le 0 \end{cases} \\
\rho &\leftarrow \rho + (r_0 - \nu)\, dt / 400\,\text{ms}
\end{aligned}
```
and each neuron fires with probability ``\nu\, dt / N``, so ``\nu`` is the rate of the whole
population. With ``\beta = 0`` the slow feedback on ``\rho`` (initialised to ``r_0``) drives
``\nu`` to ``r_0``. Note the discontinuity of ``F`` at 0 (values of ``\beta\eta`` in ``(0, 1)``
reduce the rate below the ``\beta\eta \le 0`` value); this is how the code is written.

# Fields
- `id`, `param`, `name = "VariablePoisson"`, `N::Int32 = 100`, `fire`.
- `noise::Vector{Float32}` (length 1): ``\eta``; `r::Vector{Float32}` (length 1): ``\rho``.
- `randcache`; `records::Dict`.

Reference not given in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
P = SNN.VariablePoisson(N = 100, param = SNN.VariablePoissonParameter(r0 = 2kHz))
SNN.monitor!(P, [:fire])
SNN.sim!([P]; duration = 500ms)
```
"""
VariablePoisson

@snn_kw struct VariablePoisson{VFT = Vector{Float32},VBT = Vector{Bool},IT = Int32} <:
               AbstractPopulation
    id::String = randstring(12)
    param::VariablePoissonParameter
    name::String = "VariablePoisson"
    N::IT = 100
    fire::VBT = zeros(Bool, N)
    noise::VFT = zeros(Float32, 1)
    r::VFT = ones(Float32, 1) * param.r0
    ##
    randcache::VFT = rand(N) # random cache
    records::Dict = Dict()
end


function integrate!(p::VariablePoisson, param::VariablePoissonParameter, dt::Float32)
    @unpack N, randcache, fire = p

    ## Inhomogeneous Poisson process
    @unpack r0, β, τ = param
    @unpack noise, r = p
    # Irate::Float32 = r0 * kIE
    R(x::Float32, v0::Float32 = 0.0f0) = x > 0.0f0 ? x : v0

    # Excitatory spike
    re::Float32 = 0.0f0
    cc::Float32 = 0.0f0
    Erate::Float32 = 0.0f0
    rand!(randcache)
    re = rand() - 0.5f0
    cc = 1.0f0 - dt / τ
    i = 1
    noise[i] = (noise[i] - re) * cc + re
    Erate = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
    r[i] += (r0 - Erate) / 400ms * dt
    @assert Erate >= 0
    # @inbounds @fastmath 
    for j = 1:N # loop on presynaptic neurons
        if randcache[j] < Erate / N * dt
            fire[j] = true
        else
            fire[j] = false
        end
    end
end


export Poisson, PoissonHetParameter, PoissonHomoParameter,
    PoissonParameter, integrate!, VariablePoisson, VariablePoissonParameter, stimulate!
