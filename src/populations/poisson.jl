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
    randcache::VFT = rand(N)   # uniform numbers of the dense regime (probabilities above 3 % per step)
    fire::VBT = zeros(Bool, N)
    spikes::SpikeIndex = fed_spike_index(N)  # ascending list of the neurons with fire[j], fed by `integrate!`
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
The spikes are sampled sparsely, without one random number per neuron and step: the index gap to the next
firing neuron of the Bernoulli(`rate * dt`) sequence is drawn from the geometric distribution (homogeneous rate), or
candidates are generated with the largest probability of the population and thinned (heterogeneous rates), so that a step
costs one random draw per spike (see `PoissonStimulusLayer`). Probabilities and independence are those above; the random
number stream differs from SNNModels 1.9 (and so do seeded results). When the largest per-step probability exceeds 3 % (rates
above 300 Hz at `dt = 0.1 ms`) skipping is slower than testing every neuron, and the step uses one vectorised uniform draw per neuron
(`rand!(randcache)`, then `fire[i] = randcache[i] < rate * dt`) as SNNModels 1.9 did. `fire` holds the spikes of the last step and
`spikes` the ascending list of the neurons that fired, which connections read instead of scanning `fire`.

# Fields
- `id::String = randstring(12)`, `name::String = "Poisson"`.
- `param::PoissonParameter = PoissonHomoParameter()` (1 Hz).
- `N::Int32 = 100`; `randcache::Vector{Float32}` (used only in the dense regime); `fire::Vector{Bool}`; `spikes::SpikeIndex` (list of the
  neurons that fired, fed by `integrate!`); `records::Dict`.

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

# ---- sparse sampling of the spikes of a population (shared by Poisson, InhomogeneousPoisson and VariablePoisson)

# Above this per-step probability testing every neuron (one vectorised uniform draw each) is cheaper than geometric skipping
# (about 0.5 ns per neuron against 15 ns per spike at one core; measured crossover 3 %).
const _POP_DENSE_SPIKE_P = 0.03

# Neuron `j` fires: flag it and append it to the (ascending) list.
@inline function _pop_fire!(fire::AbstractVector{Bool}, spikes::SpikeIndex, j::Int)
    @inbounds begin
        fire[j] = true
        n = spikes.n + 1
        spikes.n = n
        spikes.idx[n] = j
    end
    return nothing
end

# Clear the flags set in the previous step (where they were set, not with a pass over `N`).
@inline function _pop_clear!(fire::AbstractVector{Bool}, spikes::SpikeIndex)
    @inbounds for q in 1:spikes.n
        fire[spikes.idx[q]] = false
    end
    spikes.n = 0
    return nothing
end

# Dense regime: every neuron is tested against its probability `pj(j)` (`Float32`) with one uniform number of `cache` (freshly drawn), as in
# SNNModels 1.9. Only `fire` is written: the list is not built and not offered (`fed = false`), so the connections scan `fire` themselves
# (`spike_list!`), as they did before the sparse sampling.
function _pop_dense!(pj::F, fire, spikes::SpikeIndex, cache::AbstractVector{Float32}, N::Int) where {F}
    (length(cache) >= N && length(fire) >= N) ||
        throw(DimensionMismatch("random cache or flags shorter than N = $N"))
    rand!(cache)
    @inbounds @simd for j in 1:N
        fire[j] = cache[j] < pj(j)
    end
    spikes.fed = false
    spikes.valid = false
    spikes.n = 0
    return nothing
end

# Entering (or staying in) the sparse regime: the list is fed again. After a dense step the flags are not described by the list,
# so all of them are cleared; otherwise only those set in the previous step are.
@inline function _pop_sparse_start!(fire::AbstractVector{Bool}, spikes::SpikeIndex)
    if spikes.fed
        _pop_clear!(fire, spikes)
    else
        fill!(fire, false)
        spikes.n = 0
        spikes.fed = true
        spikes.valid = false
    end
    return nothing
end

# Bernoulli(`pr`) trial for each of `N` neurons by geometric skipping: the gap to the next spike is `floor(log(1 - U) / log(1 - pr))`.
# The previous flags must have been cleared.
function _pop_bernoulli!(fire, spikes::SpikeIndex, N::Int, pr::Float64)
    pr > 0 || return nothing
    lq = log1p(-pr)
    j = 0
    @inbounds while true
        x = log1p(-rand()) / lq
        j + 1.0 + x >= N + 1.0 && break
        j += 1 + floor(Int, x)
        _pop_fire!(fire, spikes, j)
    end
    return nothing
end

# Neuron `j` fires with probability `pj(j) <= pmax`, independently: candidates with probability `pmax` (geometric skipping), each kept with
# probability `pj(j) / pmax`, so every neuron fires with probability exactly `pj(j)`. The previous flags must have been cleared.
function _pop_thinned!(pj::F, fire, spikes::SpikeIndex, N::Int, pmax::Float64) where {F}
    pmax > 0 || return nothing
    lq = log1p(-pmax)
    j = 0
    @inbounds while true
        x = log1p(-rand()) / lq
        j + 1.0 + x >= N + 1.0 && break
        j += 1 + floor(Int, x)
        q = pj(j)
        (q >= pmax || rand() * pmax < q) && _pop_fire!(fire, spikes, j)
    end
    return nothing
end

function integrate!(p::Poisson, param::PoissonHomoParameter , dt::Float32)
    @unpack N, fire, spikes, randcache = p
    pr = Float64(param.rate * dt)
    if pr >= _POP_DENSE_SPIKE_P
        p32 = Float32(param.rate * dt)
        _pop_dense!(j -> p32, fire, spikes, randcache, Int(N))
    else
        _pop_sparse_start!(fire, spikes)
        _pop_bernoulli!(fire, spikes, Int(N), pr)
    end
    return nothing
end

function integrate!(p::Poisson, param::PoissonHetParameter , dt::Float32)
    @unpack N, fire, spikes, randcache = p
    @unpack rate = param
    length(rate) >= N || throw(DimensionMismatch("the rate vector has $(length(rate)) entries for N = $N neurons"))
    n = Int(N)
    if !spikes.fed
        # dense regime of the last step: stay dense and read the largest probability in the pass that tests the neurons, so the
        # high-rate step costs no more than SNNModels 1.9; when the rates dropped, resume the sparse sampling next step
        length(randcache) >= n && length(fire) >= n || throw(DimensionMismatch("random cache or flags shorter than N = $N"))
        rand!(randcache)
        pm = 0.0f0
        @inbounds @simd for j in 1:n
            q = rate[j] * dt
            fire[j] = randcache[j] < q
            pm = @fastmath max(pm, q)
        end
        spikes.valid = false
        spikes.n = 0
        if Float64(pm) < _POP_DENSE_SPIKE_P
            spikes.n = _flags_to_list!(spikes.idx, fire)
            spikes.fed = true
        end
        return nothing
    end
    rmax = eltype(rate)(-Inf)
    @inbounds @fastmath for j in 1:n
        rmax = max(rmax, rate[j])
    end
    pmax = Float64(rmax * dt)
    if pmax >= _POP_DENSE_SPIKE_P
        _pop_dense!(j -> @inbounds(rate[j] * dt), fire, spikes, randcache, n)
    else
        _pop_clear!(fire, spikes)
        _pop_thinned!(j -> Float64(@inbounds(rate[j]) * dt), fire, spikes, n, pmax)
    end
    return nothing
end

spike_index(p::Poisson) = p.spikes

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
- `noise::Vector{Float32}` (length 1): ``\eta``; `r::Vector{Float32}` (length 1): ``\rho``; `Erate::Vector{Float32}` (length 1): the total rate ``\nu`` of the last step.
- `spikes::SpikeIndex`: the ascending list of the neurons that fired in the last step (fed by `integrate!`);
  `randcache` (dense regime only); `records::Dict`. The spikes are sampled sparsely (see `Poisson`).

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
    Erate::VFT = zeros(Float32, 1)  # total rate of the last step
    spikes::SpikeIndex = fed_spike_index(N)  # ascending list of the neurons with fire[j], fed by `integrate!`
    ##
    randcache::VFT = rand(N) # random cache of the dense regime
    records::Dict = Dict()
end


function integrate!(p::VariablePoisson, param::VariablePoissonParameter, dt::Float32)
    @unpack N, fire, spikes, randcache = p

    ## Inhomogeneous Poisson process
    @unpack r0, β, τ = param
    @unpack noise, r, Erate = p
    R(x::Float32, v0::Float32 = 0.0f0) = x > 0.0f0 ? x : v0

    re::Float32 = rand() - 0.5f0
    cc::Float32 = 1.0f0 - dt / τ
    i = 1
    noise[i] = (noise[i] - re) * cc + re
    E::Float32 = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
    r[i] += (r0 - E) / 400ms * dt
    @assert E >= 0
    Erate[1] = E
    # every neuron fires with probability E / N * dt (sparse sampling, see `Poisson`)
    pr = Float64(E / N * dt)
    if pr >= _POP_DENSE_SPIKE_P
        p32 = Float32(E / N * dt)
        _pop_dense!(j -> p32, fire, spikes, randcache, Int(N))
    else
        _pop_sparse_start!(fire, spikes)
        _pop_bernoulli!(fire, spikes, Int(N), pr)
    end
    return nothing
end

spike_index(p::VariablePoisson) = p.spikes


export Poisson, PoissonHetParameter, PoissonHomoParameter,
    PoissonParameter, integrate!, VariablePoisson, VariablePoissonParameter, stimulate!
