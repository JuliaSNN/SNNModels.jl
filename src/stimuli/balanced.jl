@doc raw"""
    BalancedParameter(; kIE = 1.0, β = 0.0, τ = 50ms, r0 = 1kHz, w = 1.0, wIE = 1.0, same_input = false)

Parameter of a [`BalancedStimulus`](@ref): Poisson excitatory input with a slowly
fluctuating rate and Poisson inhibitory input with a fixed rate, delivered to every neuron of
the target population.

# Equations
Per step (``\Delta t`` = `dt`), for each neuron ``n``:
```math
g^{I}_n \leftarrow g^{I}_n + w\, w_{IE}\, k^{I}_n, \qquad k^{I}_n \sim \mathrm{Poisson}(k_{IE}\, r_0\, \Delta t)
```
The excitatory rate ``r^{E}`` is driven by a noise variable ``\eta`` and an adaptive
offset ``r`` (``u`` uniform on ``[-1/2, 1/2]``, ``[x]_+ = \max(x, 0)``, and ``R(x)`` = ``x``
if ``x > 0`` else 1):
```math
\eta \leftarrow (\eta - u)(1 - \Delta t/\tau) + u, \qquad
r^{E} = \Big[\tfrac{r_0}{2}\, R(\beta\,\eta) + r\Big]_+, \qquad
r \leftarrow r + \frac{r_0 - r^{E}}{400\,\mathrm{ms}}\,\Delta t
```
and the excitatory target receives ``w\,k^{E}`` with ``k^{E} \sim \mathrm{Poisson}(r^{E}\,\Delta t)``
(see the warnings in `BalancedStimulus` on how the excitatory draws are applied in 1.8.4).

# Fields
- `kIE::Float32 = 1.0`: ratio of the inhibitory rate to `r0`.
- `β::Float32 = 0.0`: amplitude of the rate fluctuations (dimensionless).
- `τ::Float32 = 50ms`: correlation time of the rate noise.
- `r0::Float32 = 1kHz`: baseline rate (library units: `1kHz` = 1 per ms).
- `w::Float32 = 1.0`: increment per input spike (excitatory and inhibitory).
- `wIE::Float32 = 1.0`: extra factor on the inhibitory increment.
- `same_input::Bool = false`: use a single excitatory rate process (index 1) instead of
  one per neuron.

Reference not given in the code.
"""
BalancedParameter

@snn_kw struct BalancedParameter{FT = Float32} <: AbstractStimulusParameter
    kIE::FT = 1.0
    β::FT = 0.0
    τ::FT = 50.0ms
    r0::FT = 1kHz
    w::FT = 1.0
    wIE::FT = 1.0
    same_input::Bool = false
end

"""
    BalancedStimulus(post::AbstractPopulation, sym_e::Symbol, sym_i::Symbol, target = nothing;
                     param::BalancedParameter, name = "Balanced")
    Stimulus(param::BalancedParameter, post, sym, target = nothing; kwargs...)

Balanced excitatory and inhibitory Poisson drive to all `post.N` neurons of `post`, with
rates defined by [`BalancedParameter`](@ref). `sym_e` and `sym_i` select the excitatory and
inhibitory target variables (`target` is the compartment for multicompartment models).

!!! warning "Known defects in SNNModels 1.8.4"
    - With `same_input = false` (the default) `stimulate!` throws
      `UndefVarError: randcache not defined`, so the default configuration cannot be
      simulated.
    - With `same_input = true` all excitatory draws are added to neuron 1 only.
    - In the per-neuron branch the excitatory target of neuron `i` receives `N` Poisson
      draws per step instead of one.
    - The generic `Stimulus(param::BalancedParameter, post, sym)` passes `sym` as both the
      excitatory and the inhibitory target, so inhibition is added to the same variable.
    - Passing a number as `param` fails (it refers to the undefined `BSParam`).

# Fields
- `id::String`; `param::BalancedParameter`; `name::String = "Balanced"`
- `N::Int32`: number of target neurons (`post.N`).
- `ge::Vector{Float32}`, `gi::Vector{Float32}`: excitatory and inhibitory targets of `post`
  (shared).
- `fire::Vector{Bool} = zeros(Bool, 0)`: unused.
- `r::Vector{Float32}`: adaptive rate offset per neuron (initialised to `r0`).
- `noise::Vector{Float32}`: rate noise per neuron (initialised to 0).
- `randcache_β::Vector{Float32}`: uniform random numbers for the noise.
- `records::Dict`, `targets::Dict`
"""
BalancedStimulus

@snn_kw struct BalancedStimulus{VFT = Vector{Float32},IT = Int32} <: AbstractStimulus
    id::String = randstring(12)
    param::BalancedParameter
    name::String = "Balanced"
    ##
    N::IT
    ge::VFT # target conductance for exc
    gi::VFT # target conductance for inh
    fire::VBT = zeros(Bool, 0)
    r::VFT
    noise::VFT
    # sparse connectivity
    randcache_β::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
end


function BalancedStimulus(
    post::T,
    sym_e::Symbol,
    sym_i::Symbol,
    target = nothing;
    param::Union{BalancedParameter,R},
    name::String = "Balanced",
) where {T<:AbstractPopulation,R<:Real}

    N = post.N
    targets = Dict(:pre => :BalancedStim, :post => post.id)
    ge, _ = synaptic_target(targets, post, sym_e, target)
    gi, _ = synaptic_target(targets, post, sym_i, target)

    if typeof(param) <: Real
        r = param
        param = BSParam(rate = (x, y) -> r, r * param.kIE)
    end

    r = ones(Float32, post.N) * param.r0
    noise = zeros(Float32, post.N)

    return BalancedStimulus(;
        param = param,
        N,
        targets,
        r,
        noise = noise,
        ge = ge,
        gi = gi,
        name = name,
    )
end


"""
    Stimulus(param::BalancedParameter, post::AbstractPopulation, sym::Symbol, target = nothing; kwargs...)

Build a [`BalancedStimulus`](@ref) with `sym` as both the excitatory and the inhibitory target.
"""
function Stimulus(
    param::BalancedParameter,
    post::T,
    sym::Symbol,
    target = nothing;
    kwargs...,
) where {T<:AbstractPopulation}
    return BalancedStimulus(post, sym, sym, target; param, kwargs...)
end


"""
    stimulate!(p::BalancedStimulus, param::BalancedParameter, time::Time, dt::Float32)

One step of the balanced input (equations in [`BalancedParameter`](@ref)). See the warnings in
[`BalancedStimulus`](@ref): the default `same_input = false` branch throws in SNNModels 1.8.4.
"""
function stimulate!(p::BalancedStimulus, param::BalancedParameter, time::Time, dt::Float32)
    @unpack N, randcache_β, ge, gi = p

    ## Inhomogeneous Poisson process
    @unpack r0, β, τ, w, kIE, wIE, same_input = param
    @unpack noise, r = p
    R(x::Float32, v0::Float32 = 0.0f0) = x > 0.0f0 ? x : v0

    # Inhibitory spike
    my_rate = Distributions.Poisson{Float32}(r0 * kIE * dt)
    @fastmath @simd for n = 1:N
        gi[n] += w * rand(my_rate) * wIE
    end

    # Excitatory spike
    re::Float32 = 0.0f0
    cc::Float32 = 0.0f0
    Erate::Float32 = 0.0f0
    rand!(randcache_β)
    if same_input
        i = 1
        re = randcache_β[i] - 0.5f0
        cc = 1.0f0 - dt / τ
        noise[i] = (noise[i] - re) * cc + re
        Erate = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
        r[i] += (r0 - Erate) / 400ms * dt
        @assert Erate >= 0

        my_rate = Distributions.Poisson{Float32}(Erate * dt)
        @fastmath @simd for n = 1:N
            ge[i] += w * rand(my_rate)
        end
    else
        @inbounds @fastmath for i = 1:N
            re = randcache_β[i] - 0.5f0
            cc = 1.0f0 - dt / τ
            noise[i] = (noise[i] - re) * cc + re
            Erate = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
            r[i] += (r0 - Erate) / 400ms * dt
            @assert Erate >= 0
            rand!(randcache)
            my_rate = Distributions.Poisson{Float32}(Erate * dt)
            @fastmath @simd for n = 1:N
                ge[i] += w * rand(my_rate)
            end
        end
    end

end

export BalancedStimulus, stimulate!, BSParam, BalancedParameter
