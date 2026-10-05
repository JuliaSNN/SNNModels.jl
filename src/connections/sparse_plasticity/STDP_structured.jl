# Define the struct to hold synapse parameters for both Exponential and Mexican Hat STDP
# STDP Parameters Structure
abstract type STDPStructuredParameter <: STDPParameter end

"""
    STDPStructuredParameter <: STDPParameter

Abstract supertype of `STDPSymmetric` and `STDPAntiSymmetric`; their state is
`STDPStructuredVariables`.
"""
STDPStructuredParameter

@doc raw"""
    STDPSymmetric(; A_x = 3e-2, A_y = 3e-2, τ_x = 50ms, τ_y = 500ms, αpre = 0pF, αpost = 0pF,
                    Wmax = 30pF, Wmin = 0pF)

Symmetric STDP rule with a difference-of-exponentials kernel, as used for structured
inhibitory plasticity by Festa, Cusseddu and Gjorgjieva (2024).

# Equations
For one pair with ``Δt = t_{post} - t_{pre}``:
```math
Δw(Δt) = \frac{A_x}{2τ_x} e^{-|Δt|/τ_x} - \frac{A_y}{2τ_y} e^{-|Δt|/τ_y},
```
whose integral over ``Δt`` is ``A_x - A_y`` (zero with the defaults). With ``τ_x < τ_y``
the kernel potentiates near-coincident spikes and depresses spikes farther apart. In
addition every presynaptic spike adds `αpre` and every postsynaptic spike `αpost`.

Implementation: each neuron has two traces (time constants ``τ_x`` and ``τ_y``) that jump
by 1 at a spike:
- presynaptic spike of j, each target i:
  ``w_{ij} \mathrel{+}= α_{pre} + \frac{A_x}{2τ_x} o_{x,i} - \frac{A_y}{2τ_y} o_{y,i}``
- postsynaptic spike of i, each source j:
  ``w_{ij} \mathrel{+}= α_{post} + \frac{A_x}{2τ_x} r_{x,j} - \frac{A_y}{2τ_y} r_{y,j}``

# Integration
`plasticity!` (only under `train!`): (1) pre-spike pass, (2) post-spike pass, both reading
the traces before this step's spikes are added; (3) forward-Euler decay of all traces, then
+1 for the neurons that fired; (4) clamp to `[Wmin, Wmax]` of the touched weights (all
weights once at the first step).

# Fields
- `A_x::FT = 3e-2`: amplitude of the narrow (potentiating) kernel.
- `A_y::FT = 3e-2`: amplitude of the wide (depressing) kernel.
- `τ_x::FT = 50ms`, `τ_y::FT = 500ms`: kernel time constants.
- `αpre::FT = 0pF`, `αpost::FT = 0pF`: constant change at each pre / post spike.
- `Wmax::FT = 30pF`, `Wmin::FT = 0pF`: weight bounds.

# References
Festa, D., Cusseddu, C. & Gjorgjieva, J., "Structured stabilization in recurrent neural
circuits through inhibitory synaptic plasticity" (2024), as cited in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 1.0),
                         LTPParam = SNN.STDPSymmetric(A_x = 1e-2, A_y = 1e-2))
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```
"""
STDPSymmetric
@snn_kw struct STDPSymmetric{FT = Float32} <: STDPStructuredParameter
    A_x::FT = 3e-2    # LTP learning rate (inhibitory synapses)
    A_y::FT = 3e-2    # LTD learning rate (inhibitory synapses)
    τ_x::FT = 50ms       # Time constant for pre-synaptic spike trace
    τ_y::FT = 500ms      # Time constant for post-synaptic spike trace
    αpre::FT = 0.0pF
    αpost::FT = 0.0pF
    Wmax::FT = 30.0pF   # Max weight
    Wmin::FT = 0.0pF    # Min weight (negative for inhibition)
end

@doc raw"""
    STDPAntiSymmetric(; A_y = 3e-2, A_x = 3e-2, τ_x = 50ms, τ_y = 50ms, αpre = 0pF, αpost = 0pF,
                        Wmax = 30pF, Wmin = 0pF)

Antisymmetric (Hebbian, pair-based) STDP rule with normalised exponential lobes, as used by
Festa, Cusseddu and Gjorgjieva (2024).

# Equations
For one pair with ``Δt = t_{post} - t_{pre}``:
```math
Δw(Δt) = \begin{cases} \dfrac{A_x}{τ_x} e^{-Δt/τ_x} & Δt > 0 \\[1ex]
                      -\dfrac{A_y}{τ_y} e^{Δt/τ_y} & Δt < 0 \end{cases}
```
with integral ``A_x - A_y`` (zero with the defaults); in addition every presynaptic spike adds
`αpre` and every postsynaptic spike `αpost`.

Implementation: a presynaptic trace ``r_{x,j}`` (``τ_x``) and a postsynaptic trace
``o_{y,i}`` (``τ_y``), both jumping by 1 at a spike:
- presynaptic spike of j, each target i: ``w_{ij} \mathrel{+}= α_{pre} - \frac{A_y}{τ_y} o_{y,i}``
- postsynaptic spike of i, each source j: ``w_{ij} \mathrel{+}= α_{post} + \frac{A_x}{τ_x} r_{x,j}``

# Integration
As for [`STDPSymmetric`](@ref): pre pass, post pass, Euler decay and +1 of the traces,
clamp of the touched weights. Only under `train!`.

# Fields
- `A_y::FT = 3e-2`: depression amplitude (post-before-pre).
- `A_x::FT = 3e-2`: potentiation amplitude (pre-before-post).
- `τ_x::FT = 50ms`, `τ_y::FT = 50ms`: time constants of the pre and post traces.
- `αpre::FT = 0pF`, `αpost::FT = 0pF`: constant change at each pre / post spike.
- `Wmax::FT = 30pF`, `Wmin::FT = 0pF`: weight bounds.

# References
Festa, D., Cusseddu, C. & Gjorgjieva, J., "Structured stabilization in recurrent neural
circuits through inhibitory synaptic plasticity" (2024), as cited in the code.
"""
STDPAntiSymmetric

@snn_kw struct STDPAntiSymmetric{FT = Float32} <: STDPStructuredParameter
    A_y::FT = 3e-2     # LTD learning rate (inhibitory synapses)
    A_x::FT = 3e-2    # LTP learning rate (inhibitory synapses)
    τ_x::FT = 50ms       # Time constant for pre-synaptic spike trace
    τ_y::FT = 50ms      # Time constant for post-synaptic spike trace
    αpre::FT = 0.0pF
    αpost::FT = 0.0pF
    Wmax::FT = 30.0pF   # Max weight
    Wmin::FT = 0.0pF    # Min weight (negative for inhibition)
end


"""
    STDPStructuredVariables(; Npre, Npost, to_x = zeros(Npost), to_y = zeros(Npost),
                            tr_x = zeros(Npre), tr_y = zeros(Npre), initialized = [false],
                            active = [true])

Traces of `STDPSymmetric` / `STDPAntiSymmetric`: `to_x`, `to_y` are postsynaptic traces
(length `Npost`, time constants `τ_x`, `τ_y`), `tr_x`, `tr_y` presynaptic traces (length
`Npre`). `STDPAntiSymmetric` uses only `tr_x` and `to_y`. `initialized` marks the one-off
clamp of all weights; `active` enables the rule.
"""
STDPStructuredVariables

@snn_kw struct STDPStructuredVariables{VFT = Vector{Float32},IT = Int} <:
               PlasticityVariables
    Npost::IT                      # Number of post-synaptic neurons
    Npre::IT                       # Number of pre-synaptic neurons
    to_x::VFT = zeros(Npost)        # post-synaptic spike trace, τ_x
    to_y::VFT = zeros(Npost)        # post-synaptic spike trace, τ_y
    tr_x::VFT = zeros(Npre)         # pre-synaptic spike trace, τ_x
    tr_y::VFT = zeros(Npre)         # pre-synaptic spike trace, τ_y
    initialized::VBT = [false]      # one-off clamp of all weights done
    active::VBT = [true]
end

function plasticityvariables(param::T, Npre, Npost) where {T<:STDPStructuredParameter}
    return STDPStructuredVariables(Npre = Npre, Npost = Npost)
end


# SymmetricSTDP and AntiSymmetricSTDP
function plasticity!(
    c::PT,
    param::STDPAntiSymmetric,
    plasticity::STDPStructuredVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack tr_x, to_y = plasticity
    @unpack A_x, A_y, τ_x, τ_y, Wmax, Wmin, αpre, αpost = param
    # Update weights based on pre-post spike timing
    @inbounds @fastmath begin

        for j = 1:(length(colptr)-1) # loop over post-synaptic neurons
            if fireJ[j]
                @turbo for s = colptr[j]:(colptr[j+1]-1)
                    i = I[s]
                    # @info "Pre synaptic firing, time: $(get_time(T)) trace: $(to_y[i])"
                    W[s] += αpre - A_y / τ_y * to_y[i]  # pre spike
                end
            end
        end

        for i = 1:(length(rowptr)-1) # loop over post-synaptic neurons
            if fireI[i]
                @turbo for st = rowptr[i]:(rowptr[i+1]-1)
                    j = J[index[st]]
                    s = index[st]
                    # @info "Post synaptic firing, time: $(get_time(T)) trace: $(tr_x[j])"
                    W[s] += αpost + A_x / τ_x * tr_x[j]  # post spike
                end
            end
        end

        # Update traces based on pre synpatic firing
        @turbo for i in eachindex(fireI)
            to_y[i] += dt * (-to_y[i]) / τ_y
        end
        @simd for i in findall(fireI)
            to_y[i] += 1
        end

        # Update traces based on pre synpatic firing
        @turbo for j in eachindex(fireJ)
            tr_x[j] += dt * (-tr_x[j]) / τ_x
        end
        @simd for j in findall(fireJ)
            tr_x[j] += 1
        end

    end
    # Clamp the weights touched in this step (all weights once, on the first step)
    _initial_clamp!(W, plasticity.initialized, Wmin, Wmax)
    _clamp_touched!(W, colptr, rowptr, index, fireJ, fireI, Wmin, Wmax)
end

function plasticity!(
    c::PT,
    param::STDPSymmetric,
    plasticity::STDPStructuredVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack to_x, tr_x, to_y, tr_y = plasticity
    @unpack A_x, A_y, τ_x, τ_y, Wmax, Wmin, αpre, αpost = param


    # Update weights based on pre-post spike timing
    @inbounds @fastmath begin
        # for i in 1:length(rowptr)-1 # loop over post-synaptic neurons
        for j = 1:(length(colptr)-1) # loop over post-synaptic neurons
            if fireJ[j]
                @turbo for s = colptr[j]:(colptr[j+1]-1)
                    i = I[s]
                    W[s] += αpre + (A_x / 2τ_x * to_x[i] - A_y / 2τ_y * to_y[i])  # pre spike
                end
            end
        end

        # Update weights based on pre-post spike timing
        # for j in 1:length(colptr)-1 # loop over pre-synaptic neurons
        for i = 1:(length(rowptr)-1) # loop over post-synaptic neurons
            if fireI[i]
                @turbo for st = rowptr[i]:(rowptr[i+1]-1)
                    j = J[index[st]]
                    s = index[st]
                    W[s] += αpost + (A_x / 2τ_x * tr_x[j] - A_y / 2τ_y * tr_y[j])  # post spike
                end
            end
        end
        @turbo for i in eachindex(fireI)
            to_x[i] += dt * (-to_x[i]) / τ_x
            to_y[i] += dt * (-to_y[i]) / τ_y
        end
        @simd for i in findall(fireI)
            to_x[i] += 1
            to_y[i] += 1
        end

        @turbo for j in eachindex(fireJ)
            tr_x[j] += dt * (-tr_x[j]) / τ_x
            tr_y[j] += dt * (-tr_y[j]) / τ_y
        end

        @simd for j in findall(fireJ)
            tr_x[j] += 1
            tr_y[j] += 1
        end

    end
    # Clamp the weights touched in this step (all weights once, on the first step)
    _initial_clamp!(W, plasticity.initialized, Wmin, Wmax)
    _clamp_touched!(W, colptr, rowptr, index, fireJ, fireI, Wmin, Wmax)
end
# Function to implement STDP update rule

export STDPSymmetric, STDPAntiSymmetric
