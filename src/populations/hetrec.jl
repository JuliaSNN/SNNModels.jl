"""
    HetRecParameter <: AbstractGeneralizedIFParameter

Parameters for the **HetRec layer** (*heterogeneous timescale, non-recurrent* layer).

The HetRec layer is a population of **non-recurrent neurons** whose input is integrated in
multiple dendritic compartments with **heterogeneous dendritic integration timescales**.
Each neuron has `Nd` dendritic compartments, and each compartment is assigned a time constant
sampled from `τd`. Dendritic states are mixed into the soma by a sparse mapping controlled by
the dendritic tree **overlap** constraint.

## Dendritic tree overlap
The parameter `overlap` constrains how much dendritic input is shared across neurons:

- `overlap = 0`: **no overlap**; dendrites are not shared across neurons (each soma reads only its
  "own" dendrites). This corresponds to the default conceptual behavior of the HetRec layer.
- `0 < overlap < 1`: partial overlap; each soma reads a random subset of other neurons' dendrites.
- `overlap = 1`: full overlap; dendritic compartments are maximally shared across neurons.

## Fields
- `Nd::Int`: Number of dendritic compartments per neuron (default: `2`)
- `N::Int`: Number of neurons in the population (default: `100`)
- `overlap::Float32`: Dendritic overlap level across neurons (`0` non-overlapping, `1` fully overlapping; default: `0.5`)
- `τd::Distribution`: Distribution of dendritic time constants (sampled for each compartment; default: `Uniform(10.0f0, 100.0f0)`)
- `rate::Distribution`: Distribution of baseline firing-rate parameters (sampled per neuron; default: `Uniform(0.0f0, 1.0f0)`)
- `τabs::Float32`: Absolute refractory period (default: `5ms`)
- `steepness::Float32`: Steepness of the soma firing nonlinearity (default: `1.0f0`)
- `τm::Float32`: Soma integration / filtering time constant (default: `20ms`)
- `τrate::Float32`: Time constant of the firing-rate trace / adaptation variable (default: `100ms`)
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
        @simd for s in colptr[i]:(colptr[i+1]-1)
            v_s[i] += (W[s] * v_d[I[s]] - v_s[i]) * dt / τm
        end
        tabs[i] -= 1
        fire[i] = false
        trace[i] += dt * (-trace[i]/τrate)
        tabs[i] > 0 && continue
        trace[i] += (v_s[i]-trace[i])/τrate
        if randcache[i] < r[i] * (1 / (1 + exp(-steepness * (v_s[i] - trace[i]))))  * dt
            fire[i] = true
            tabs[i] = round(Int, τabs/dt)
            trace[i] += 1.0f0
        end
    end
end


export HetRec, HetRecParameter