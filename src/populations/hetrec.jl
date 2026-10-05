@doc raw"""
    HetRecParameter(; Nd = 2, overlap = 0.5, τd = Uniform(10, 100), rate = Uniform(0, 1),
                     τabs = 5ms, steepness = 1, τm = 20ms, τrate = 100ms)

Parameters of the `HetRec` population: stochastic spiking units whose input is filtered by
`Nd` dendritic compartments per neuron with heterogeneous time constants. The docstring of the
original code describes it as a heterogeneous-timescale, non-recurrent layer.

# Fields
- `Nd::Int = 2`: number of dendritic compartments per neuron (the population has `N * Nd`
  dendrites).
- `overlap::Float32 = 0.5`: probability that the soma of neuron ``i`` also reads dendrite
  ``j`` of another neuron ``k \neq i``; `0` = each soma reads only its own `Nd` dendrites,
  `1` = every soma reads all dendrites.
- `τd::Distribution = Uniform(10.0f0, 100.0f0)`: distribution of the dendritic time constants
  (ms), one sample per dendrite.
- `rate::Distribution = Uniform(0.0f0, 1.0f0)`: distribution of the maximal firing rate ``r_i``
  of each neuron, in spikes per ms (1 = 1 kHz).
- `τabs::Float32 = 5ms`: absolute refractory period (ms).
- `steepness::Float32 = 1.0`: slope of the sigmoid firing nonlinearity (1/mV).
- `τm::Float32 = 20ms`: somatic integration time constant (ms).
- `τrate::Float32 = 100ms`: time constant of the adaptive firing baseline `trace` (ms).

The number of neurons is not a field: it is the `N` keyword of `Population`.
Reference not given in the code.
"""
HetRecParameter

@snn_kw struct HetRecParameter  <: AbstractGeneralizedIFParameter
    Nd::Int = 2 ## number of dendritic compartments per neuron
    overlap::Float32 = 0.5 ## overlap of dendritic inputs across neurons (0: non-overlapping, 1: fully overlapping)
    τd::Distribution = Uniform(10.0f0, 100.0f0) ## distribution of dendritic time constants
    rate::Distribution = Uniform(0.0f0, 1.0f0) ## distribution of firing rates
    τabs::Float32 = 5ms ## absolute refractory period
    steepness::Float32 = 1.0f0 ## steepness of the firing rate nonlinearity
    τm::Float32 = 20ms ## membrane time constant
    τrate ::Float32 = 100ms ## time constant for firing rate adaptation
end


@doc raw"""
    Population(param::HetRecParameter; N = 100)
    HetRec

Population of `N` stochastic spiking neurons, each with `param.Nd` leaky dendritic compartments
with heterogeneous time constants. Create it with `Population(HetRecParameter(...); N)`, which
samples `τd` and `r`, builds the dendrite-to-soma mapping and the sparse matrix of the mapping.

Input connections target the dendrites: the presynaptic spikes are written into
`receptors.glu` / `receptors.gaba` (use the target symbols `:glu`, `:gaba`; size `N * Nd`) and
filtered by a `CurrentSynapse` (``τ_e = 6`` ms, ``τ_i = 2`` ms).

# Equations
For dendrite ``d`` and neuron ``i`` (``g_E``, ``g_I``: exponentially decaying current-based
synaptic variables of the `CurrentSynapse`):
```math
\begin{aligned}
\tau_d\, \frac{dv_d}{dt} &= -v_d + g_E - g_I \\
\tau_m\, \frac{dv_s^i}{dt} &= \sum_{d \in \mathcal{D}_i} \left(W_{id}\, v_d - v_s^i\right) \\
\tau_{rate}\, \frac{da_i}{dt} &= -a_i + [\text{not refractory}]\,(v_s^i - a_i) + \tau_{rate} \textstyle\sum_f \delta(t - t_i^f) \\
P(\text{spike of } i \text{ in } [t, t + dt]) &= r_i\, \sigma\!\left(k\, (v_s^i - a_i)\right) dt
\end{aligned}
```
with ``\sigma(x) = 1 / (1 + e^{-x})``, ``k`` = `steepness`, ``\mathcal{D}_i`` the dendrites read
by neuron ``i`` (its own `Nd` dendrites plus those of other neurons selected with probability
`overlap`, ``W_{id} = 1``) and ``a_i`` the adaptive baseline `trace`.

# Integration
Per step: synapse update; dendrites `v_d += dt * (-v_d - is) / τd` with `is = -(g_E - g_I)`;
for each neuron, one forward-Euler step of the somatic equation,
`v_s += dt / τm * Σ_d (W v_d - v_s)` (with ``k_i`` connected dendrites the soma relaxes to their
mean with time constant ``τ_m / k_i``); `tabs -= 1`, `fire = false`,
`trace += dt * (-trace / τrate)`; if not refractory, `trace += dt * (v_s - trace) / τrate` and
the neuron fires with the probability above, which sets `tabs = round(Int, τabs / dt)` and
`trace += 1`.

!!! note "Changed after SNNModels 1.8.4"
    Up to 1.8.4 the soma was updated once per connected dendrite in sequence (same equation,
    but the result depended on the order of the dendrites) and the adaptive baseline was
    updated with `trace += (v_s - trace) / τrate`, without `dt`, so its time constant was
    `τrate * dt` (in steps) and results depended on `dt`.

# Fields
- `id`, `name = "HetRec"`, `N::Int32 = 100`, `param::HetRecParameter`.
- `v_d` (length `N * Nd`), `v_s` (length `N`): dendritic and somatic potentials.
- `M::Matrix{Float32}`: dense mapping (unused after construction, zeros by default).
- `r`: maximal rates (1/ms); `fire`; `tabs`; `trace`: adaptive baseline; `randcache`.
- `τd`: dendritic time constants (ms).
- `rowptr`, `colptr`, `I`, `J`, `index`, `W`: CSC representation of the dendrite-to-soma
  mapping (columns = neurons, rows = dendrites).
- `is`: dendritic synaptic current; `synapse = CurrentSynapse()`, `synvars`, `receptors`
  (size `N * Nd`); `records::Dict`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
H = SNN.Population(SNN.HetRecParameter(Nd = 3, overlap = 0.2); N = 20)
P = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
s = SNN.SpikingSynapse(P, H, :glu; conn = (μ = 1, p = 0.2))
model = SNN.compose(; P, H, s)
SNN.monitor!(H, [:fire, :v_s])
SNN.sim!(model, 200ms)
```
"""
HetRec

@snn_kw struct HetRec{
    VFT = Vector{Float32},
    VIT = Vector{Int32},
    MFT = Matrix{Float32},
    IT = Int32,
    SYN<:AbstractSynapseParameter,
    SYNV<:AbstractSynapseVariable,
    RECT<:NamedTuple,
    VBT = Vector{Bool},
} <: AbstractGeneralizedIF
    id::String = randstring(12)
    name::String = "HetRec"
    ## These are compulsory parameters
    N::IT = 100
    param::HetRecParameter = HetRecParameter()

    # Membrane potential and adaptation
    v_d::VFT = zeros(N*param.Nd)
    v_s::VFT = zeros(N)
    M::MFT = zeros(N, N*param.Nd)
    r::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    tabs::VIT = zeros(N)
    trace::VFT = zeros(N)
    randcache::VFT = rand(N) # random cache for stochastic firing

    ## Timescales
    τd::VFT = zeros(N*param.Nd)
 
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight

    is::VFT = zeros(N*param.Nd)
    synapse::SYN = CurrentSynapse()
    synvars::SYNV = synaptic_variables(CurrentSynapse(), N*param.Nd)
    receptors::RECT = synaptic_receptors(CurrentSynapse(), N*param.Nd)
    records::Dict= Dict()
end

function Population(param::HetRecParameter; N=100, kwargs...)
    @unpack Nd, overlap, rate, τd = param
    # Initialize the time constants τd based on the specified distribution
    τd_values = rand(τd, N * Nd)
    rs_values = rand(rate, N)

    M = zeros(Float32, N, N * Nd)
    for i in 1:N
        for j in 1:Nd
            M[i, (j-1)*N + i] = 1
            for k in 1:N
                if k != i
                    M[i, (j-1)*N + k] = rand() < overlap 
                end
            end
        end
    end

    w = sparse_matrix(N, N*Nd, M')
    # this matrix is inversed pre-post because the connections are unique in the post-pre direction
    rowptr, colptr, I, J, index, W = dsparse(w)

    return HetRec(;
        N = N,
        param = param,
        v_d = zeros(Float32, N * Nd),
        is = zeros(Float32, N * Nd),
        v_s = zeros(Float32, N),
        r = rs_values,
        τd = τd_values,
        @symdict(rowptr, colptr, I, J, index, W)...,
    )
end

function input_N(pre::HetRec)
    return pre.N * pre.param.Nd
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target::Nothing,
) where {T<:HetRec}
    v = post.v_d
    g = getfield(post.receptors, sym)
    push!(targets, :sym => "v_d")
    push!(targets, :g => post.id)
    return g, v
end

function integrate!(p::HetRec, param::HetRecParameter, dt::Float32)
    @unpack N, v_s, v_d, τd, synapse, receptors, synvars, is, tabs, trace, fire, r, randcache = p
    @unpack Nd, overlap, steepness, τm, τabs, τrate = param
    @unpack colptr, I, J, W = p

    update_synapses!(p, synapse, receptors, synvars, dt)
    synaptic_current!(p, synapse, synvars, v_s, is)
    # @inbounds 
    @. v_d += dt * (-v_d - is) / τd
    rand!(randcache)
    @inbounds for i in 1:N
        # τm dv_s/dt = Σ_d (W_id v_d - v_s), evaluated at the state of the beginning of the step
        drive = 0.0f0
        @simd for s in colptr[i]:(colptr[i+1]-1)
            drive += W[s] * v_d[I[s]] - v_s[i]
        end
        v_s[i] += drive * dt / τm
        tabs[i] -= 1
        fire[i] = false
        trace[i] += dt * (-trace[i]/τrate)
        tabs[i] > 0 && continue
        trace[i] += dt * (v_s[i]-trace[i])/τrate
        if randcache[i] < r[i] * (1 / (1 + exp(-steepness * (v_s[i] - trace[i]))))  * dt
            fire[i] = true
            tabs[i] = round(Int, τabs/dt)
            trace[i] += 1.0f0
        end
    end
end


export HetRec, HetRecParameter