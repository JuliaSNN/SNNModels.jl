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
and the excitatory target of neuron ``n`` receives ``w\,k^{E}_n`` with
``k^{E}_n \sim \mathrm{Poisson}(r^{E}_n\,\Delta t)`` (one draw per neuron and step). With
`same_input = true` a single rate process (``\eta``, ``r``, ``r^E`` shared by all neurons) is used
and every neuron draws its own Poisson count at that rate. Note the rule ``R(x)`` (1 for
``x \le 0``), transcribed from the code.

# Fields
- `kIE::Float32 = 1.0`: ratio of the inhibitory rate to `r0`.
- `β::Float32 = 0.0`: amplitude of the rate fluctuations (dimensionless).
- `τ::Float32 = 50ms`: correlation time of the rate noise.
- `r0::Float32 = 1kHz`: baseline rate (library units: `1kHz` = 1 per ms).
- `w::Float32 = 1.0`: increment per input spike (excitatory and inhibitory).
- `wIE::Float32 = 1.0`: extra factor on the inhibitory increment.
- `same_input::Bool = false`: use a single excitatory rate process shared by all neurons
  instead of one per neuron.

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

`param` may also be a number, the baseline rate `r0` (library units, e.g. `2kHz`), used with
the other defaults of `BalancedParameter`.

!!! note "Changed after SNNModels 1.8.4"
    Up to 1.8.4: the default `same_input = false` threw `UndefVarError: randcache`;
    `same_input = true` added all excitatory draws to neuron 1; the per-neuron branch added `N`
    Poisson draws per step to each neuron (rate multiplied by `N`); `Stimulus(param, post, sym)`
    used `sym` as both the excitatory and the inhibitory target; a numeric `param` referred to
    the undefined `BSParam`.

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
        param = BalancedParameter(r0 = Float32(param))
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


const _BALANCED_INH = Dict(:ge => :gi, :glu => :gaba, :he => :hi, :AMPA => :GABAa)

"""
    Stimulus(param::BalancedParameter, post::AbstractPopulation, sym_e::Symbol, sym_i::Symbol, target = nothing; kwargs...)
    Stimulus(param::BalancedParameter, post::AbstractPopulation, sym::Symbol, target = nothing; kwargs...)

Build a [`BalancedStimulus`](@ref). In the second form `sym` is the excitatory target and the
inhibitory one is its counterpart (`:ge` -> `:gi`, `:glu` -> `:gaba`, `:he` -> `:hi`,
`:AMPA` -> `:GABAa`); other symbols raise an `ArgumentError`.
"""
function Stimulus(
    param::BalancedParameter,
    post::T,
    sym_e::Symbol,
    sym_i::Symbol,
    target = nothing;
    kwargs...,
) where {T<:AbstractPopulation}
    return BalancedStimulus(post, sym_e, sym_i, target; param, kwargs...)
end

function Stimulus(
    param::BalancedParameter,
    post::T,
    sym::Symbol,
    target = nothing;
    kwargs...,
) where {T<:AbstractPopulation}
    haskey(_BALANCED_INH, sym) || throw(ArgumentError("no inhibitory counterpart known for :$sym; use Stimulus(param, post, sym_e, sym_i)"))
    return BalancedStimulus(post, sym, _BALANCED_INH[sym], target; param, kwargs...)
end


"""
    stimulate!(p::BalancedStimulus, param::BalancedParameter, time::Time, dt::Float32)

One step of the balanced input (equations in [`BalancedParameter`](@ref)).
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

    # Excitatory spikes: one Poisson draw per neuron and step
    cc = 1.0f0 - dt / τ
    rand!(randcache_β)
    if same_input
        # one rate process shared by all neurons (state stored at index 1)
        re = randcache_β[1] - 0.5f0
        noise[1] = (noise[1] - re) * cc + re
        Erate = R(r0 / 2 * R(noise[1] * β, 1.0f0) + r[1], 0.0f0)
        r[1] += (r0 - Erate) / 400ms * dt
        my_rate = Distributions.Poisson{Float32}(Erate * dt)
        @inbounds for i = 1:N
            ge[i] += w * rand(my_rate)
        end
    else
        @inbounds for i = 1:N
            re = randcache_β[i] - 0.5f0
            noise[i] = (noise[i] - re) * cc + re
            Erate = R(r0 / 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
            r[i] += (r0 - Erate) / 400ms * dt
            ge[i] += w * rand(Distributions.Poisson{Float32}(Erate * dt))
        end
    end
end

export BalancedStimulus, stimulate!, BalancedParameter
