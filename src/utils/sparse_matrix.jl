function connect!(c, j, i, μ = randn(Float32))
    W = matrix(c)
    W[i, j] = μ
    update_sparse_matrix!(c, W)
    return nothing
end

function matrix(c::C) where {C<:AbstractConnection}
    return sparse(c.I, c.J, c.W, length(c.rowptr) - 1, length(c.colptr) - 1)
end


function matrix(c::C, sym::Symbol) where {C<:AbstractConnection}
    return sparse(c.I, c.J, getfield(c, sym), length(c.rowptr) - 1, length(c.colptr) - 1)
end

function matrix(c::C, sym::Symbol, time::Number) where {C<:AbstractConnection}
    W, r = record(c, sym, range = true)
    @assert time <= r[end] && time >= r[1] "Time $time not in recorded range $(r[1]):$(r[end])"
    return matrix(c, W, time)
end

function matrix(c::C, sym::Symbol, time::AbstractVector) where {C<:AbstractConnection}
    W, r = record(c, sym, range = true)
    @assert all(time .<= r[end] .&& time .>= r[1]) "Time $time not in recorded range $(r[1]):$(r[end])"
    return [matrix(c, W, t) for t in time] |> x -> cat(x..., dims = 3)
end


function matrix(c::C, W::AbstractArray, time::Number) where {C<:AbstractConnection}
    return sparse(c.I, c.J, W[:, time], length(c.rowptr) - 1, length(c.colptr) - 1)
end

function matrix(c::C, W::AbstractArray, time::AbstractVector) where {C<:AbstractConnection}
    return [
        sparse(c.I, c.J, W[:, t], length(c.rowptr) - 1, length(c.colptr) - 1) for t in time
    ] |> x -> cat(x..., dims = 3)
end

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

function presynaptic_idxs(c::C, i::Int) where {C<:AbstractConnection}
    @unpack rowptr, index, J, W = c
    rowptr[i]:(rowptr[i+1]-1)
end

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

function postsynaptic_idxs(c::C, j::Int) where {C<:AbstractConnection}
    @unpack colptr, I, index = c
    colptr[j]:(colptr[j+1]-1)
end

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

function set_plasticity!(synapse::AbstractConnection, bool::Bool)
    synapse.param.active[1] = bool
end
function has_plasticity(synapse::AbstractConnection)
    synapse.param.active[1] |> Bool
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
    @assert (isnothing(p) || isnothing(ρ)) && !(isnothing(p) && isnothing(ρ)) "Specify either p or ρ"
    ρ = isnothing(ρ) ? p : ρ
    @assert ρ >= 0 && ρ <= 1 "ρ must be in [0, 1]"
    @debug "Constructing sparse matrix with $rule rule, $dist distribution, μ=$μ, σ=$σ, ρ=$ρ"
    syn_sign = μ ≈ 0 ? 1 : sign(μ)
    if syn_sign == -1
        @warn "You are using negative synaptic weights "
        μ = abs(μ)
    end

    my_dist = getfield(Distributions, dist)
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
    return w .* syn_sign
end


sparse_matrix(Npre, Npost, conn::NamedTuple) = sparse_matrix(Npre, Npost; conn...)

function sparse_matrix(Npre, Npost, conn::AbstractMatrix)
    w = conn
    @assert size(w) == (Npost, Npre) "The size of the synaptic weight is not correct: $(size(w)) != ($Npost, $Npre)"
    return sparse(w)
end


function update_sparse_matrix!(c::S, W::SparseMatrixCSC) where {S<:AbstractConnection}
    rowptr, colptr, I, J, index, W = dsparse(W)
    @assert length(rowptr) == length(c.rowptr) "Rowptr length mismatch"
    @assert length(colptr) == length(c.colptr) "Colptr length mismatch"

    resize!(c.I, length(I))
    resize!(c.J, length(I))
    resize!(c.W, length(I))
    resize!(c.index, length(I))

    @assert length(c.I) ==
            length(c.J) ==
            length(c.index) ==
            length(c.W) ==
            length(I) ==
            length(J) ==
            length(index) ==
            length(W) "Length mismatch"

    @inbounds @simd for i in eachindex(I)
        c.I[i] = I[i]
        c.J[i] = J[i]
        c.W[i] = W[i]
        c.index[i] = index[i]
    end
    c.colptr = colptr
    c.rowptr = rowptr
    return nothing
end

function update_sparse_matrix!(c::S) where {S<:AbstractConnection}
    rowptr, colptr, I, J, index, W = sparse(c.I, c.J, c.W) |> dsparse

    @inbounds @simd for i in eachindex(I)
        c.I[i] = I[i]
        c.J[i] = J[i]
        c.W[i] = W[i]
        c.index[i] = index[i]
    end
    c.colptr = colptr
    c.rowptr = rowptr
    return nothing
end



"""
    synaptic_turnover!(C::SpikingSynapse; p_rewire=0.05, p_pre = x->rand(), p_new = x->rand(), μ = 3.0)

Perform synaptic turnover on a spiking synapse connection matrix.

# Arguments
- `C::SpikingSynapse`: The spiking synapse connection to modify
- `p_rewire::Float64=0.05`: Probability threshold for rewiring existing connections
- `p_pre::Function=x->rand()`: Function that returns probability for each presynaptic connection `s` to be rewired 
- `p_new::Function=x->rand()`: Function that returns probability for selecting new postsynaptic neurons
- `μ::Float64=3.0`: Weight value for new connections

# Description
This function implements synaptic turnover by:
1. Generating thresholds for selecting connections to rewire. 
2. Identifying plausible new connections for each presynaptic neuron
3. Selecting connections to rewire based on the probability thresholds
4. Replacing the selected connections with new ones
5. Updating the sparse matrix structure

The function modifies the connection matrix in-place and updates its sparse matrix representation.
"""
function synaptic_turnover!(
    C::S;
    p_rewire = x->rand() .< 0.05,
    p_new = x->rand(),
    μ = 3.0,
) where {S<:AbstractConnection}
    pre_tt = rand(Uniform(0, 1), length(C.fireJ))
    post_tt = rand(Uniform(0, 1), length(C.fireI))
    all_post = Set(1:length(C.fireI))
    new_connections = map(postsynaptic(C)) do pre
        plausible = setdiff(all_post, pre) |> collect
        plausible[sortperm(p_new.(plausible))]
    end

    rep_connections = Int[]
    rep_neurons = Int[]
    pre_tt .* post_tt
    @unpack rowptr, colptr, I, J, index, W, fireJ = C
    for j in eachindex(fireJ)
        for s in postsynaptic_idxs(C, j)
            !p_rewire(s) && continue
            push!(rep_connections, s)
            push!(rep_neurons, pop!(new_connections[j]))
        end
    end
    for (s, new_post) in zip(rep_connections, rep_neurons)
        C.I[s] = new_post
        C.W[s] = μ
    end
    update_sparse_matrix!(C)
end




function dsparse(A)
    # them in a special data structure leads to savings in space and execution time, compared to dense arrays.
    At = sparse(A') # Transposes the input sparse matrix A and stores it as At.
    colptr = A.colptr # Retrieves the column pointer array from matrix A
    rowptr = At.colptr # Retrieves the column pointer array from the transposed matrix At
    I = rowvals(A) # Retrieves the row indices of non-zero elements from matrix A
    V = nonzeros(A) # Retrieves the values of non-zero elements from matrix A
    J = zero(I) # Initializes an array J of the same size as I filled with zeros.
    index = zeros(Int, size(I)) # Initializes an array index of the same size as I filled with zeros.


    # FIXME: Breaks when A is empty
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
