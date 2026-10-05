"""
    connect!(c, j, i, μ = randn(Float32))

Set the weight of the synapse from presynaptic neuron `j` to postsynaptic neuron `i` of the
sparse connection `c` to `μ`, creating the synapse if it does not exist, and rebuild the
sparse storage with `update_sparse_matrix!(c, W)`.

The per-synapse arrays are kept consistent (see `update_sparse_matrix!`): existing synapses keep
their short-term efficacy `ρ` and delay, a new synapse gets `ρ = 1` and the mean delay.
(Up to SNNModels 1.8.4 `ρ` and the delays kept their old length, so `forward!` read out of
bounds after a synapse was created.)
"""
function connect!(c, j, i, μ = randn(Float32))
    W = matrix(c)
    W[i, j] = μ
    update_sparse_matrix!(c, W)
    return nothing
end

"""
    matrix(c::AbstractConnection)
    matrix(c::AbstractConnection, sym::Symbol)

Return the weights of the sparse connection `c` as a `SparseMatrixCSC` of size
`N_post x N_pre` (rows: postsynaptic, columns: presynaptic). With `sym`, the matrix holds
the per-synapse field `sym` of `c` instead of `W` (e.g. `:ρ`).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 50); I = SNN.IF(N = 20)
EI = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 2.0))
W = SNN.matrix(EI)          # 20 x 50
size(W)
```
"""
function matrix(c::C) where {C<:AbstractConnection}
    return sparse(c.I, c.J, c.W, length(c.rowptr) - 1, length(c.colptr) - 1)
end


function matrix(c::C, sym::Symbol) where {C<:AbstractConnection}
    return sparse(c.I, c.J, getfield(c, sym), length(c.rowptr) - 1, length(c.colptr) - 1)
end

"""
    matrix_record(c::AbstractConnection, sym::Symbol, time::Number)
    matrix_record(c::AbstractConnection, sym::Symbol, time::AbstractVector)

Rebuild the `N_post x N_pre` sparse matrix of the recorded per-synapse variable `sym`
(e.g. `:W`, recorded with `monitor!(c, [:W])`) at time `time` (ms), interpolating the
record (`record(c, sym, range = true)`). For a vector of times, the matrices are
concatenated along the third dimension. `time` must lie within the recorded range
(asserted). The sparsity pattern is the current one of `c`.
"""
function matrix_record(c::C, sym::Symbol, time::Number) where {C<:AbstractConnection}
    W, r = record(c, sym, range = true)
    @assert time <= r[end] && time >= r[1] "Time $time not in recorded range $(r[1]):$(r[end])"
    return matrix(c, W, time)
end

function matrix_record(c::C, sym::Symbol, time::AbstractVector) where {C<:AbstractConnection}
    W, r = record(c, sym, range = true)
    @assert all(time .<= r[end] .&& time .>= r[1]) "Time $time not in recorded range $(r[1]):$(r[end])"
    return [matrix(c, W, t) for t in time] |> x -> cat(x..., dims = 3)
end


"""
    matrix(c::AbstractConnection, W, time::Number)
    matrix(c::AbstractConnection, W, time::AbstractVector)

Sparse `N_post x N_pre` matrix of the interpolated record `W` (callable as
`W(synapses, time)`) at `time`; for a vector of times, a 3-dimensional array. Used by
`matrix_record`.
"""
function matrix(c::C, W::AbstractArray, time::Number) where {C<:AbstractConnection}
    return sparse(c.I, c.J, W(axes(W, 1), time), length(c.rowptr) - 1, length(c.colptr) - 1)
end

function matrix(c::C, W::AbstractArray, time::AbstractVector) where {C<:AbstractConnection}
    return [
        sparse(c.I, c.J, W(axes(W, 1), t), length(c.rowptr) - 1, length(c.colptr) - 1) for
        t in time
    ] |> x -> cat(x..., dims = 3)
end

"""
    update_weights!(c::AbstractConnection, j, i, w)
    update_weights!(c::AbstractConnection, js::Vector, is::Vector, w::Real)

Set to `w` the weight of the existing synapse from presynaptic `j` to postsynaptic `i`
(first form), or of every existing synapse from any `j in js` to any `i in is` (second
form). Synapses that do not exist are not created (see `connect!`).
"""
function update_weights!(c::C, j, i, w) where {C<:AbstractConnection}
    @unpack colptr, I, W = c
    for s = colptr[j]:(colptr[j+1]-1)
        if I[s] == i
            W[s] = w
            break
        end
    end
end

function update_weights!(
    c::C,
    js::Vector,
    is::Vector,
    w::Real,
) where {C<:AbstractConnection}
    @unpack colptr, I, W = c
    for j in js
        for s = colptr[j]:(colptr[j+1]-1)
            if I[s] ∈ is
                W[s] = w
            end
        end
    end
end


##

"""
    presynaptic_idxs(c::AbstractConnection, i::Int)

Range of positions, in the row-major (transposed) ordering, of the synapses onto
postsynaptic neuron `i`: `c.rowptr[i]:(c.rowptr[i+1]-1)`. These are not indices into
`c.W`: the corresponding synapses are `c.index[presynaptic_idxs(c, i)]` (CSC positions),
e.g. `c.W[c.index[presynaptic_idxs(c, i)]]` are the input weights of `i`.
"""
function presynaptic_idxs(c::C, i::Int) where {C<:AbstractConnection}
    @unpack rowptr, index, J, W = c
    rowptr[i]:(rowptr[i+1]-1)
end

"""
    presynaptic(c::AbstractConnection)
    presynaptic(c::AbstractConnection, i::Int)
    presynaptic(c::AbstractConnection, is::AbstractVector)

Presynaptic neuron indices of the connection `c`: for every postsynaptic neuron (first
form, a vector of vectors), for neuron `i` (second form) or for each neuron in `is` (third
form).

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 50); I = SNN.IF(N = 20)
EI = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 2.0))
SNN.SNNModels.presynaptic(EI, 1)     # inputs of I neuron 1
SNN.SNNModels.postsynaptic(EI, 1)    # targets of E neuron 1
```
"""
function presynaptic(c::C) where {C<:AbstractConnection}
    @unpack rowptr, index, J, W = c
    [J[index[rowptr[i]:(rowptr[i+1]-1)]] for i = 1:(length(rowptr)-1)]
end

function presynaptic(c::C, i::Int) where {C<:AbstractConnection}
    @unpack rowptr, index, J, W = c
    J[index[rowptr[i]:(rowptr[i+1]-1)]]
end

function presynaptic(c::C, is::AbstractVector) where {C<:AbstractConnection}
    @unpack rowptr, index, J, W = c
    presyn = Vector{Vector{Int}}()
    for i in is
        push!(presyn, J[index[rowptr[i]:(rowptr[i+1]-1)]])
    end
    return presyn
end

##

"""
    postsynaptic_idxs(c::AbstractConnection, j::Int)

Range of CSC positions of the synapses of presynaptic neuron `j`:
`c.colptr[j]:(c.colptr[j+1]-1)`. These index `c.W`, `c.I` and the other per-synapse arrays
directly.
"""
function postsynaptic_idxs(c::C, j::Int) where {C<:AbstractConnection}
    @unpack colptr, I, index = c
    colptr[j]:(colptr[j+1]-1)
end

"""
    postsynaptic(c::AbstractConnection)
    postsynaptic(c::AbstractConnection, j::Int)
    postsynaptic(c::AbstractConnection, js::AbstractVector)

Postsynaptic neuron indices of the connection `c`: for every presynaptic neuron (first
form), for neuron `j`, or for each neuron in `js`.
"""
function postsynaptic(c::C) where {C<:AbstractConnection}
    @unpack colptr, I, index = c
    [I[colptr[j]:(colptr[j+1]-1)] for j = 1:(length(colptr)-1)]
end

function postsynaptic(c::C, j::Int) where {C<:AbstractConnection}
    @unpack colptr, I, index = c
    I[colptr[j]:(colptr[j+1]-1)]
end

function postsynaptic(c::C, js::AbstractVector) where {C<:AbstractConnection}
    @unpack colptr, I, index = c
    postsyn = Vector{Vector{Int}}()
    for j in js
        push!(postsyn, I[colptr[j]:(colptr[j+1]-1)])
    end
    return postsyn
end


"""
    indices(c::AbstractConnection, js::AbstractVector, is::AbstractVector)

CSC positions (indices into `c.W`) of all existing synapses from presynaptic neurons in
`js` to postsynaptic neurons in `is`.
"""
function indices(c::C, js::AbstractVector, is::AbstractVector) where {C<:AbstractConnection}
    @unpack colptr, I, W = c
    indices = Int[]
    for j in js
        for s = colptr[j]:(colptr[j+1]-1)
            if I[s] ∈ is
                push!(indices, s)
            end
        end
    end
    return indices
end

"""
    set_plasticity!(synapse::AbstractConnection, bool::Bool)
    has_plasticity(synapse::AbstractConnection)

Switch / read the plasticity of a connection. For a sparse synapse with plasticity variables
(`SpikingSynapse`) `set_plasticity!` sets the `active` flag of both `LTPVars` and `STPVars`
(as `set_LTP!` and `set_STP!`), and `has_plasticity` returns whether a long-term or
short-term rule is present and active. For other connections the `active` field of
`param` is used if it exists; otherwise `set_plasticity!` is a no-op and `has_plasticity`
returns `false`. (Up to SNNModels 1.8.4 both raised a `FieldError` for `SpikingSynapse`.)
"""
function set_plasticity!(synapse::AbstractConnection, bool::Bool)
    if hasfield(typeof(synapse), :LTPVars)
        synapse.LTPVars.active .= bool
        synapse.STPVars.active .= bool
    elseif hasfield(typeof(synapse.param), :active)
        synapse.param.active[1] = bool
    end
    return nothing
end
"""
    has_plasticity(synapse::AbstractConnection)

Whether the connection has an active plasticity rule (see `set_plasticity!`).
"""
function has_plasticity(synapse::AbstractConnection)
    if hasfield(typeof(synapse), :LTPVars)
        ltp = !(synapse.LTPParam isa NoLTP) && Bool(synapse.LTPVars.active[1])
        stp = !(synapse.STPParam isa NoSTP) && Bool(synapse.STPVars.active[1])
        return ltp || stp
    elseif hasfield(typeof(synapse.param), :active)
        return Bool(synapse.param.active[1])
    end
    return false
end
# """function dsparse

using SpecialFunctions, Roots

# function gamma_for_mean(μ::Float64, kmin::Int=1; γ_max::Float64=5.0)
#     # Define the function to find the root of
#     f(γ) = zeta(γ - 1, kmin) / zeta(γ, kmin) - μ

#     # Find γ in the range (2, γ_max] where the mean is finite
#     if μ == Inf
#         return 2.0  # Mean is infinite for γ ≤ 2
#     else
#         result = find_zero(f, 3.0001)
#         return result
#     end
# end

"""
    sparse_matrix(Npre, Npost; p | ρ, μ = 1, σ = 0, dist = :Normal, rule = :Fixed, γ, kmin)

Random connectivity matrix of size `Npost x Npre` (rows: postsynaptic, columns:
presynaptic) as a `SparseMatrixCSC{Float32,Int}`, built directly in CSC form in
O(nnz + Npre + Npost) time and memory: no dense `Npost x Npre` array is ever allocated.

Exactly one of `p` and `ρ` (connection probability / density, in [0, 1]) must be given.
Unknown keywords are ignored with a warning (up to SNNModels 1.8.4 they were ignored silently,
so a typo such as `sigma` went unnoticed); `w` is not used (pass a matrix as `conn` instead).

# Connectivity rules (`rule`)
- `:Fixed` / `:FixedIn` (default): every postsynaptic neuron receives exactly
  `K = Npre - round(Int, (1 - ρ) * Npre)` inputs, sampled without replacement.
- `:FixedOut`: every presynaptic neuron projects to exactly
  `K = Npost - round(Int, (1 - ρ) * Npost)` targets, sampled without replacement.
- `:Bernoulli`: every pair is connected independently with probability `ρ`. Sampled by
  geometric skipping over the column-major index space (O(nnz) random draws).
- `:PowerLaw`: the out-degree of each presynaptic neuron is
  `min(round(Int, rand(Pareto(γ, kmin))), Npost - 1)`; targets sampled without replacement.

# Weights
One draw per connection from `dist(|μ|, σ)`, where `dist` is a two-parameter `Distributions`
type or its `Symbol` (default `:Normal`; for `:LogNormal`, `μ` and `σ` are
the parameters of the logarithm), with Float32 parameters. Draws `<= 0` are not synapses and
are removed, so with `σ > 0` the realised degrees can be lower than `K`; a negative `μ` flips
the sign of all weights afterwards (with a warning). This is the same treatment as the
previous dense generator. Weights are in the units of the target variable (e.g. nS).

Autapses are not removed here; `SpikingSynapse` removes them when `pre == post`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
w = SNN.SNNModels.sparse_matrix(100, 40; p = 0.1, μ = 2.0, σ = 0.5, rule = :Bernoulli)
size(w)   # (40, 100): Npost x Npre
```

# Reproducibility
The random stream differs from the dense generator used before SNNModels 1.8.2, so seeded
networks do not reproduce the old realisations; the statistics (degree distributions,
weight moments, sparsity) are the same. The old generator is still available as the
non-exported `SNNModels.sparse_matrix_dense_legacy`.
"""
function sparse_matrix(
    Npre,
    Npost;
    w = nothing,
    dist = :Normal,
    μ = 1,
    σ = 0,
    ρ = nothing,
    p = nothing,
    rule = :Fixed,
    γ = -1,
    kmin = -1,
    kwargs...,
)
    isempty(kwargs) || @warn "sparse_matrix: unknown connectivity keyword(s) $(collect(keys(kwargs))) ignored"
    isnothing(w) || @warn "sparse_matrix: the keyword `w` is not used; pass a matrix as `conn` instead"
    @assert (isnothing(p) || isnothing(ρ)) && !(isnothing(p) && isnothing(ρ)) "Specify either p or ρ"
    ρ = isnothing(ρ) ? p : ρ
    @assert ρ >= 0 && ρ <= 1 "ρ must be in [0, 1]"
    @debug "Constructing sparse matrix with $rule rule, $dist distribution, μ=$μ, σ=$σ, ρ=$ρ"
    syn_sign = μ ≈ 0 ? 1 : sign(μ)
    if syn_sign == -1
        @warn "You are using negative synaptic weights "
        μ = abs(μ)
    end
    Npre, Npost = Int(Npre), Int(Npost)

    # 1. Sparsity pattern (column pointers and sorted row indices).
    if rule == :FixedOut
        K = Npost - round(Int, (1 - ρ) * Npost)
        colptr, rowval = _fixed_out_pattern(Npost, Npre, j -> K)
    elseif rule == :FixedIn || rule == :Fixed
        K = Npre - round(Int, (1 - ρ) * Npre)
        colptr, rowval = _fixed_in_pattern(Npost, Npre, K)
    elseif rule == :Bernoulli
        colptr, rowval = _bernoulli_pattern(Npost, Npre, Float64(ρ))
    elseif rule == :PowerLaw
        if Npre > 0
            @assert γ > 0 "For PowerLaw connection rule, γ must be defined and positive"
            @assert kmin > 0 "For PowerLaw connection rule, kmin must be defined and positive"
        end
        degrees = [min(round(Int, rand(Distributions.Pareto(γ, kmin))), Npost - 1) for _ = 1:Npre]
        colptr, rowval = _fixed_out_pattern(Npost, Npre, j -> degrees[j])
    else
        throw(ArgumentError("Unknown connection mode: $rule; use :Fixed or :Bernoulli"))
    end

    # 2. One Float32 weight per connection; non-positive draws are not synapses.
    my_dist = dist isa Symbol ? getfield(Distributions, dist) : dist
    nzval = Vector{Float32}(undef, length(rowval))
    rand!(my_dist(Float32(μ), Float32(σ)), nzval)
    w = SparseMatrixCSC{Float32,Int}(Npost, Npre, colptr, rowval, nzval)
    _filter_csc!(w, (i, j, v) -> v > 0)
    syn_sign == -1 && (nonzeros(w) .*= -1)
    return w
end

# Bernoulli(p) pattern by geometric skipping. Entries are visited in column-major order of
# the flattened index space 1:Npost*Npre; the gap to the next connected entry is the
# number of failures before a success, Geometric(p) = floor(log(U) / log(1 - p)) with
# U ~ Uniform(0, 1]. Cost O(nnz), rows come out sorted within each column.
function _bernoulli_pattern(Npost::Int, Npre::Int, p::Float64)
    L = Npost * Npre
    colptr = zeros(Int, Npre + 1)
    rowval = Int[]
    (p <= 0 || L == 0) && (colptr .= 1; return colptr, rowval)
    sizehint!(rowval, ceil(Int, L * p + 5 * sqrt(L * p) + 16))
    logq = log1p(-p)                   # -Inf for p == 1: every gap is 0
    pos = 0                            # last connected linear index (0 = none yet)
    @inbounds while true
        gap = floor(log(1.0 - rand()) / logq)
        gap >= L - pos && break        # compared in Float64: no Int overflow on huge gaps
        pos += Int(gap) + 1
        col = (pos - 1) ÷ Npost + 1
        push!(rowval, pos - (col - 1) * Npost)
        colptr[col+1] += 1             # column counts, turned into pointers below
    end
    colptr[1] = 1
    cumsum!(colptr, colptr)
    return colptr, rowval
end

# Fixed in-degree K: each postsynaptic row i draws K distinct presynaptic columns. The
# (row, column) pairs are scattered into CSC by a counting sort on the column; rows are
# visited in increasing order, so they come out sorted within each column.
function _fixed_in_pattern(Npost::Int, Npre::Int, K::Int)
    K = clamp(K, 0, Npre)
    cols = Vector{Int}(undef, Npost * K)
    perm = collect(1:Npre)
    for i = 1:Npost
        _sample_distinct!(view(cols, ((i-1)*K+1):(i*K)), perm)
    end
    colptr = zeros(Int, Npre + 1)
    @inbounds for j in cols
        colptr[j+1] += 1
    end
    colptr[1] = 1
    cumsum!(colptr, colptr)
    next = colptr[1:Npre]              # write cursor per column
    rowval = Vector{Int}(undef, length(cols))
    @inbounds for i = 1:Npost, s = ((i-1)*K+1):(i*K)
        j = cols[s]
        rowval[next[j]] = i
        next[j] += 1
    end
    return colptr, rowval
end

# Fixed out-degree degree(j) per presynaptic column j: distinct rows sampled without
# replacement, sorted, written column by column.
function _fixed_out_pattern(Npost::Int, Npre::Int, degree::F) where {F}
    colptr = Vector{Int}(undef, Npre + 1)
    colptr[1] = 1
    for j = 1:Npre
        colptr[j+1] = colptr[j] + clamp(degree(j), 0, Npost)
    end
    rowval = Vector{Int}(undef, colptr[end] - 1)
    perm = collect(1:Npost)
    for j = 1:Npre
        r = view(rowval, colptr[j]:(colptr[j+1]-1))
        _sample_distinct!(r, perm)
        sort!(r)
    end
    return colptr, rowval
end

# Fill `out` with length(out) distinct elements of `perm`, uniformly at random, by a
# partial Fisher-Yates shuffle: O(length(out)) work, no allocation. `perm` is any
# permutation of the population and stays one, so it is reused across calls.
function _sample_distinct!(out::AbstractVector{Int}, perm::Vector{Int})
    n = length(perm)
    @inbounds for t in eachindex(out)
        k = t - first(eachindex(out)) + 1
        r = rand(k:n)
        perm[k], perm[r] = perm[r], perm[k]
        out[t] = perm[k]
    end
    return out
end

# Keep only the stored entries (i, j, v) of a CSC matrix for which keep(i, j, v) is true,
# compacting in place (structural removal: no explicit zeros are left, nothing densified).
function _filter_csc!(A::SparseMatrixCSC, keep::F) where {F}
    colptr, rowval, nzval = A.colptr, A.rowval, A.nzval
    k = 1
    @inbounds for j = 1:size(A, 2)
        start, stop = colptr[j], colptr[j+1] - 1   # read before colptr[j] is overwritten
        colptr[j] = k
        for s = start:stop
            if keep(rowval[s], j, nzval[s])
                rowval[k] = rowval[s]
                nzval[k] = nzval[s]
                k += 1
            end
        end
    end
    colptr[end] = k
    resize!(rowval, k - 1)
    resize!(nzval, k - 1)
    return A
end

"""
    remove_autapses!(w::SparseMatrixCSC)

Structurally remove the diagonal (self-connections) of a square connectivity matrix,
without densifying it.
"""
remove_autapses!(w::SparseMatrixCSC) = _filter_csc!(w, (i, j, v) -> i != j)

# Previous generator: draws a dense Npost x Npre Float64 matrix and zeroes entries.
# O(Npre * Npost) memory and time (80 GB at 1e5 neurons). Not exported; kept only so
# that the tests can compare the statistics of `sparse_matrix` against it.
function sparse_matrix_dense_legacy(
    Npre,
    Npost;
    w = nothing,
    dist = :Normal,
    μ = 1,
    σ = 0,
    ρ = nothing,
    p = nothing,
    rule = :Fixed,
    γ = -1,
    kmin = -1,
    kwargs...,
)
    isempty(kwargs) || @warn "sparse_matrix: unknown connectivity keyword(s) $(collect(keys(kwargs))) ignored"
    isnothing(w) || @warn "sparse_matrix: the keyword `w` is not used; pass a matrix as `conn` instead"
    @assert (isnothing(p) || isnothing(ρ)) && !(isnothing(p) && isnothing(ρ)) "Specify either p or ρ"
    ρ = isnothing(ρ) ? p : ρ
    @assert ρ >= 0 && ρ <= 1 "ρ must be in [0, 1]"
    @debug "Constructing sparse matrix with $rule rule, $dist distribution, μ=$μ, σ=$σ, ρ=$ρ"
    syn_sign = μ ≈ 0 ? 1 : sign(μ)
    if syn_sign == -1
        @warn "You are using negative synaptic weights "
        μ = abs(μ)
    end

    my_dist = dist isa Symbol ? getfield(Distributions, dist) : dist
    w = rand(my_dist(μ, σ), Npost, Npre) # Construct a random dense matrix with dimensions post.N x pre.N
    if rule == :FixedOut
        # Set to zero a fraction (1-ρ)*Npost of the weights in each column
        for pre = 1:Npre
            targets =
                ρ > 0 ? sample(1:Npost, round(Int, (1-ρ)*Npost); replace = false) : 1:Npost
            w[targets, pre] .= 0
        end
    elseif rule == :FixedIn || rule == :Fixed
        for post = 1:Npost
            pres = ρ > 0 ? sample(1:Npre, round(Int, (1-ρ)*Npre); replace = false) : 1:Npre
            w[post, pres] .= 0
        end
    elseif rule == :Bernoulli
        # Set to zero each weight with probability (1-ρ)
        w[[n for n in eachindex(w[:]) if rand() < 1-ρ]] .= 0
    elseif rule == :PowerLaw
        for pre = 1:Npre
            @assert γ > 0 "For PowerLaw connection rule, γ must be defined and positive"
            @assert kmin > 0 "For PowerLaw connection rule, kmin must be defined and positive"
            n = round(Int, rand(Distributions.Pareto(γ, kmin)))
            n = minimum((n, Npost-1))
            targets = sample(1:Npost, Npost-n; replace = false)
            w[targets, pre] .= 0
        end
        # do nothing
    else
        throw(ArgumentError("Unknown connection mode: $rule; use :Fixed or :Bernoulli"))
    end
    w[w .<= 0] .= 0 # no negative weights
    w = sparse(w)
    @assert size(w) == (Npost, Npre) "The size of the synaptic weight is not correct: $(size(w)) != ($Npost, $Npre)"
    # Synaptic data are always Float32. The random draw above is left in the
    # element type implied by (μ, σ) so that seeded networks are bit-identical to
    # earlier versions; the conversion happens here, at the constructor boundary.
    return _float32_sparse(w .* syn_sign)
end

# Convert any sparse/dense connectivity matrix to SparseMatrixCSC{Float32}.
_float32_sparse(w::SparseMatrixCSC) = SparseMatrixCSC{Float32,Int}(w)
_float32_sparse(w::AbstractMatrix) = SparseMatrixCSC{Float32,Int}(sparse(Float32.(w)))


"""
    sparse_matrix(Npre, Npost, conn::NamedTuple)
    sparse_matrix(Npre, Npost, conn::AbstractMatrix)

Connectivity of a connection constructor: a `NamedTuple` is splatted into the keyword form
of `sparse_matrix`; a matrix (dense or sparse, any element type, size `Npost x Npre`,
asserted) is converted to `SparseMatrixCSC{Float32,Int}` as is (stored zeros of a sparse
input are kept, explicit zeros of a dense input are dropped).
"""
sparse_matrix(Npre, Npost, conn::NamedTuple) = sparse_matrix(Npre, Npost; conn...)

function sparse_matrix(Npre, Npost, conn::AbstractMatrix)
    w = conn
    @assert size(w) == (Npost, Npre) "The size of the synaptic weight is not correct: $(size(w)) != ($Npost, $Npre)"
    return _float32_sparse(w)
end


"""
    update_sparse_matrix!(c::AbstractConnection, W::SparseMatrixCSC)
    update_sparse_matrix!(c::AbstractConnection)

Rebuild the double sparse storage of `c` (`rowptr`, `colptr`, `I`, `J`, `index`, `W`). The
first form replaces it with the matrix `W` (same size, asserted; the number of synapses may
change). The second form rebuilds it from the current `c.I`, `c.J`, `c.W` (used after
`synaptic_turnover!` has changed postsynaptic indices), keeping the size of the matrix
(`length(c.rowptr) - 1` postsynaptic, `length(c.colptr) - 1` presynaptic neurons).

The per-synapse arrays are resized and reordered with the synapses: the short-term efficacy
`ρ` and, for delayed synapses, the delays `param.delaytime`. In the first form a synapse
`(i, j)` that already existed keeps its values and a new synapse gets `ρ = 1` and the mean
delay; in the second form every stored synapse keeps its values (its CSC position may change).
The plasticity variables are per neuron and are not affected.

!!! note "Changed after SNNModels 1.8.4"
    Up to 1.8.4 `ρ` and the delays were neither resized nor reordered (out-of-bounds reads in
    `forward!` after `connect!` created a synapse, and a mismatch between `W` and `ρ` after a
    turnover), and the second form inferred the matrix size from the largest stored indices
    (the matrix shrank if the last neurons had no synapse).
"""
function update_sparse_matrix!(c::S, W::SparseMatrixCSC) where {S<:AbstractConnection}
    old_I, old_J = copy(c.I), copy(c.J)
    rowptr, colptr, I, J, index, W = dsparse(W)
    @assert length(rowptr) == length(c.rowptr) "Rowptr length mismatch"
    @assert length(colptr) == length(c.colptr) "Colptr length mismatch"

    # position of every new synapse in the old storage (0 = new synapse)
    old_pos = Dict{Tuple{Int,Int},Int}()
    for s in eachindex(old_I)
        old_pos[(Int(old_I[s]), Int(old_J[s]))] = s
    end
    perm = [get(old_pos, (Int(I[s]), Int(J[s])), 0) for s in eachindex(I)]

    resize!(c.I, length(I))
    resize!(c.J, length(I))
    resize!(c.W, length(I))
    resize!(c.index, length(I))

    @inbounds @simd for i in eachindex(I)
        c.I[i] = I[i]
        c.J[i] = J[i]
        c.W[i] = W[i]
        c.index[i] = index[i]
    end
    c.colptr = colptr
    c.rowptr = rowptr
    _remap_synapse_state!(c, perm)
    return nothing
end

function update_sparse_matrix!(c::S) where {S<:AbstractConnection}
    Npost, Npre = length(c.rowptr) - 1, length(c.colptr) - 1
    # CSC order: by presynaptic neuron, then postsynaptic neuron
    perm = sortperm(collect(zip(c.J, c.I)))
    A = sparse(c.I[perm], c.J[perm], c.W[perm], Npost, Npre)
    @assert nnz(A) == length(c.W) "duplicate synapses after rewiring"
    rowptr, colptr, I, J, index, W = dsparse(A)

    @inbounds @simd for i in eachindex(I)
        c.I[i] = I[i]
        c.J[i] = J[i]
        c.W[i] = W[i]
        c.index[i] = index[i]
    end
    c.colptr = colptr
    c.rowptr = rowptr
    _remap_synapse_state!(c, perm)
    return nothing
end

# Reorder / resize the per-synapse arrays of `c` after a rebuild of its sparse storage:
# `perm[s]` is the old position of the synapse now stored at `s` (0 for a new synapse).
function _remap_synapse_state!(c, perm::AbstractVector{<:Integer})
    if hasfield(typeof(c), :ρ)
        old = copy(c.ρ)
        resize!(c.ρ, length(perm))
        for s in eachindex(perm)
            c.ρ[s] = perm[s] > 0 ? old[perm[s]] : 1.0f0
        end
    end
    if hasfield(typeof(c), :param) && hasfield(typeof(c.param), :delaytime)
        old = copy(c.param.delaytime)
        d0 = isempty(old) ? 0.0f0 : Float32(sum(old) / length(old))
        resize!(c.param.delaytime, length(perm))
        for s in eachindex(perm)
            c.param.delaytime[s] = perm[s] > 0 ? old[perm[s]] : d0
        end
    end
    return c
end


"""
    dsparse(A::SparseMatrixCSC) -> (rowptr, colptr, I, J, index, V)

Double sparse representation of the `N_post x N_pre` matrix `A`, used by all sparse
connections:
- `colptr`, `I` (row indices = postsynaptic neurons), `V` (values): the CSC arrays of `A`,
  ordered by presynaptic neuron;
- `J`: the column (presynaptic neuron) of every stored entry;
- `rowptr`: column pointers of `sparse(A')`, i.e. row pointers of `A`;
- `index`: map from row-major position to CSC position, so that
  `V[index[rowptr[i]:(rowptr[i+1]-1)]]` are the entries of row `i`.

The returned arrays alias the internal arrays of `A` (`colptr`, `I`, `V`).
"""
function dsparse(A)
    # them in a special data structure leads to savings in space and execution time, compared to dense arrays.
    At = sparse(A') # Transposes the input sparse matrix A and stores it as At.
    colptr = A.colptr # Retrieves the column pointer array from matrix A
    rowptr = At.colptr # Retrieves the column pointer array from the transposed matrix At
    I = rowvals(A) # Retrieves the row indices of non-zero elements from matrix A
    V = nonzeros(A) # Retrieves the values of non-zero elements from matrix A
    J = zero(I) # Initializes an array J of the same size as I filled with zeros.
    index = zeros(Int, size(I)) # Initializes an array index of the same size as I filled with zeros.


    for j = 1:(length(colptr)-1) # Starts a loop iterating through the columns of the matrix.
        J[colptr[j]:(colptr[j+1]-1)] .= j # Assigns column indices to J for each element in the column range.
    end
    coldown = zeros(eltype(index), length(colptr) - 1) # Initializes an array coldown with a specific type and size.
    for i = 1:(length(rowptr)-1) # Iterates through the rows of the transposed matrix At.
        for st = rowptr[i]:(rowptr[i+1]-1) # Iterates through the range of elements in the current row.
            j = At.rowval[st] # Retrieves the column index from the transposed matrix At.
            index[st] = colptr[j] + coldown[j] # Computes an index for the index array.
            coldown[j] += 1 # Updates coldown for indexing.
        end
    end
    # Test.@test At.nzval == A.nzval[index]
    rowptr, colptr, I, J, index, V # Returns the modified rowptr, colptr, I, J, index, and V arrays.
end

export dsparse,
    matrix,
    matrix_record,
    extract_items,
    sparse_matrix,
    indices,
    update_weights!,
    presynaptic,
    postsynaptic,
    connect!,
    set_plasticity!,
    has_plasticity,
    update_sparse_matrix!,
    presynaptic_idxs,
    postsynaptic_idxs,
    synaptic_turnover!
