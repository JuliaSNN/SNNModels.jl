using Random
using LinearAlgebra
using Parameters
using SpecialFunctions

"""
    place_populations(Npop, grid_size)

Place the neurons of each population uniformly at random in a box of size `grid_size`.

# Arguments
- `Npop`: `NamedTuple` (or `Dict`) of population sizes; only entries of type `Int64` are used,
  the others are skipped.
- `grid_size`: vector with the box size in each dimension (any number of dimensions).

# Returns
- `NamedTuple` with, for each population, a `Vector` of `N` points, each point a
  `Vector{Float32}` with coordinates in `[0, grid_size[d])`.

# Example
```julia
using SpikingNeuralNetworks
points = SNN.place_populations((E = 80, I = 20), [1.0f0, 1.0f0])
length(points.E)   # 80
```
"""
function place_populations(Npop, grid_size)
    Pops = Dict{Symbol,Vector}()
    for k in keys(Npop)
        !(typeof(Npop[k]) == Int64) && continue
        points = [rand(Float32, length(grid_size)) .* grid_size for _ = 1:Npop[k]]
        Pops[k] = points
    end
    return Pops |> dict2ntuple
end

@doc raw"""
    periodic_distance(x1::Float32, x2::Float32, grid_size::Float32)
    periodic_distance(point1::Vector{Float32}, point2::Vector{Float32}, grid_size::Real)
    periodic_distance(point1::Vector{Float32}, point2::Vector{Float32}, grid_size::Vector)

Distance on a torus (periodic boundary conditions).

- Scalars: ``d = \min(|x_1 - x_2|,\ L - |x_1 - x_2|)``.
- Points with a scalar `grid_size` ``L`` (same size in every dimension): Euclidean periodic
  distance ``\sqrt{\sum_n d_n^2}``.
- Points with a vector `grid_size` ``(L_1, L_2, \dots)``: Euclidean periodic distance
  ``\sqrt{\sum_n d_n^2}`` with ``d_n`` computed with ``L_n``. (Up to SNNModels 1.8.4 this method
  computed ``\sqrt{(\sum_n d_n)^2} = \sum_n d_n``, the L1 distance.)

# Example
```julia
using SpikingNeuralNetworks
SNN.periodic_distance([0.05f0, 0.0f0], [0.95f0, 0.0f0], 1.0f0)   # 0.1
```
"""
function periodic_distance(point1::Float32, point2::Float32, grid_size::Float32)
    abs(min(abs(point1 - point2), grid_size - abs(point1 - point2)))
end
function periodic_distance(
    point1::Vector{Float32},
    point2::Vector{Float32},
    grid_size::Vector{T},
) where {T<:Real}
    return sqrt(
        sum(
            map(eachindex(point1)) do n
                min(abs(point1[n] - point2[n]), grid_size[n] - abs(point1[n] - point2[n]))^2
            end,
        ),
    )
end
function periodic_distance(
    point1::Vector{Float32},
    point2::Vector{Float32},
    grid_size::T,
) where {T<:Real}
    return sqrt(
        sum([
            periodic_distance(point1[n], point2[n], grid_size)^2 for n in eachindex(point1)
        ]),
    )
end

"""
    neurons_within_circle(points, center, distance, grid_size)

Boolean mask of the neurons whose `periodic_distance` from `center` is `<= distance`.

# Arguments
- `points::Vector{Vector{Float32}}`: The coordinates of the neurons.
- `center::Vector{Float32}`: The coordinates of the center point.
- `distance`: The maximum distance from the center point.
- `grid_size`: The size of the grid (scalar or vector, see `periodic_distance`).

# Returns
- `Vector{Bool}` with one entry per point (use `findall` to get indices).
"""
function neurons_within_circle(points, center, distance, grid_size)
    map(x->periodic_distance(x, center, grid_size) <= distance, points)
end

function neurons_within(func::Function, kwargs...) end

"""
    neurons_outside_area(points, center, distance, grid_size)

Find the indices of neurons outside a specified area around a center point.

# Arguments
- `points::Vector{Vector{Float32}}`: The coordinates of the neurons.
- `center::Vector{Float32}`: The coordinates of the center point.
- `distance`: The minimum distance from the center point.
- `grid_size`: The size of the grid (scalar or vector, see `periodic_distance`).

# Returns
- `Vector{Int}`: indices of the neurons with `periodic_distance > distance`.
"""
function neurons_outside_area(points, center, distance, grid_size)
    return [
        i for
        i = 1:length(points) if periodic_distance(points[i], center, grid_size) > distance
    ]
end


@doc raw"""
    gaussian_weight(pre::Vector{Float32}, post::Vector{Float32} = [0, 0]; σx, σy, grid_size::Vector{Float32})

Gaussian profile ``\exp(-(d_x/\sigma_x)^2 - (d_y/\sigma_y)^2)`` of the periodic distances
``d_x, d_y`` between two 2D points (computed with `exp64`; note there is no factor 1/2 in the
exponent).
# Arguments
- `pre::Vector{Float32}`: The coordinates of the pre-synaptic neuron.
- `post::Vector{Float32}`: The coordinates of the post-synaptic neuron.
- `σx::Float32`: The standard deviation of the Gaussian in the x-direction.
- `σy::Float32`: The standard deviation of the Gaussian in the y-direction.
- `grid_size::Vector{Float32}`: The size of the grid.

# Returns
- `weight::Float32`: The Gaussian weight between the pre- and post-synaptic neurons.
"""
function gaussian_weight(
    pre::Vector{Float32},
    post::Vector{Float32} = [0.0f0, 0.0f0];
    σx::Float32,
    σy::Float32,
    grid_size::Vector{Float32},
)
    begin
        x = periodic_distance(post[1], pre[1], grid_size[1])
        y = periodic_distance(post[2], pre[2], grid_size[2])
        return exp64(-(x/σx)^2 - (y/σy)^2)
    end
end


@doc raw"""
    compute_connections(pre::Symbol, post::Symbol, points; conn::NamedTuple, spatial::NamedTuple, dist::Sampleable)

Draw a distance-dependent connectivity from the population `pre` to `post` on a 2D torus.

`points` is the output of `place_populations`; `points.pre` and `points.post` are used.
Connection weights are drawn from the distribution `dist`. Periodic distances are computed
with `periodic_distance`. `spatial.grid_size` must have length 2. Two rules are available,
selected by `spatial.type`:

- `:critical_distance` (fields `dc`, `ϵ`, `grid_size`, `p_long`): with
  ``A = L_x L_y``, ``p_l = `` `spatial.p_long[pre]`, ``\gamma_s = A / (\pi d_c^2)``,
  ``\gamma_l = A / (A - \pi d_c^2)``, a pair closer than ``d_c`` is connected with probability
  ``p_{short} = (1 - p_l)\,\gamma_s\,\epsilon\,p`` and a farther pair with
  ``p_{long} = p_l\,\gamma_l\,\epsilon\,p``, where ``p`` = `conn.p`. Autapses are excluded
  when `pre == post`. The returned `P` is all zeros.
- `:gaussian` (fields `σs`, `grid_size`, `ϵ`): ``(\sigma_x, \sigma_y)`` = `spatial.σs[pre]`;
  the pair ``(i, j)`` is connected with probability ``P_{ij} = \gamma\,\epsilon\,p\,g_{ij}``,
  with ``g_{ij}`` = `gaussian_weight(pre_j, post_i)` and ``\gamma`` the inverse of the mean of
  the Gaussian profile over a 200x200 grid, so that the mean probability is ``\epsilon p``.
  Pairs with `i == j` are excluded when `pre == post` (up to SNNModels 1.8.4 also when
  `pre != post`).

Any other `spatial.type` returns `nothing`.

# Returns
- `L::BitMatrix` (`N_post x N_pre`): connection mask.
- `W::Matrix{Float32}`: weights (`rand(dist)` where `L` is true, 0 elsewhere).
- `P::Matrix{Float32}`: connection probabilities (`:gaussian` only).

# Example
```julia
using SpikingNeuralNetworks, Distributions
points = SNN.place_populations((E = 100,), [1.0f0, 1.0f0])
spatial = (type = :gaussian, σs = (E = (0.1f0, 0.1f0),), grid_size = [1.0f0, 1.0f0], ϵ = 1.0f0)
L, W, P = SNN.compute_connections(:E, :E, points; conn = (p = 0.1,), spatial, dist = Normal(1, 0.1))
```
"""
function compute_connections(pre::Symbol, post::Symbol, points; conn::NamedTuple, spatial::NamedTuple, dist::Sampleable)
    @unpack grid_size = spatial
    @assert length(grid_size) == 2 "grid_size must be a vector of length 2"
    pre_points = getfield(points, pre)
    post_points = getfield(points, post)
    N_pre = length(getfield(points, pre))
    N_post = length(getfield(points, post))
    L = falses(N_post, N_pre)
    W = zeros(Float32, N_post, N_pre)
    P = zeros(Float32, N_post, N_pre)

    if spatial.type == :critical_distance
        @unpack dc, ϵ, grid_size, p_long = spatial
        pl = getfield(spatial.p_long, pre)
        area = grid_size[1]*grid_size[2]
        γs = area / (π * dc^2)
        γl = area / (area - π * dc^2)
        p_short = (1 - pl) * γs * ϵ * conn.p
        p_long = (pl) * γl * ϵ * conn.p

        @inbounds for j = 1:N_pre
            for i = 1:N_post
                pre == post && i == j && continue
                distance = periodic_distance(post_points[i], pre_points[j], grid_size)
                if distance < dc
                    if rand() <= p_short
                        L[i, j] = true
                        W[i, j] = rand(dist)
                    end
                else
                    if rand() <= p_long
                        L[i, j] = true
                        W[i, j] = rand(dist)
                    end
                end
            end
        end
        return L, W, P
    end
    if spatial.type == :gaussian

        @unpack σs, grid_size, ϵ = spatial
        X, Y = grid_size
        xs = range(-X/2, stop = X/2, length = 200) |> collect |> z->Float32.(z)
        ys = range(-Y/2, stop = Y/2, length = 200) |> collect |> z->Float32.(z)
        σx, σy = Float32.(getfield(σs, pre))
        γ = 1/mean([gaussian_weight([_x, _y]; σx, σy, grid_size) for _x in xs for _y in ys])
        p = Float32(conn.p * ϵ * γ)
        randcache = rand(N_post, N_pre)
        for j = 1:N_pre
            for i = 1:N_post
                if pre == post && i == j # no autapses (only within one population)
                    P[i, j] = 0.0f0
                    L[i, j] = false
                    W[i, j] = 0.0f0
                    continue
                end
                p0 = gaussian_weight(
                    pre_points[j],
                    post_points[i];
                    σx = σx,
                    σy = σy,
                    grid_size,
                )
                P[i, j] = p0 * p
                pre == post && i == j && continue
                p == 0 && continue
                randcache[i, j] > p && continue
                if randcache[i, j] <= p * p0
                    L[i, j] = true
                    # W[i, j] = conn.μ
                    W[i, j] = rand(dist)
                end
            end
        end
        # @info "$pre => $post average conn weight: $(mean(W))"
        # @info "$pre => $post average conn probability: $(mean(P))"
        return L, W, P
    end
end

@doc raw"""
    linear_network(N; σ_w = 0.38, w_max = 2.0, kwargs...)

Weight matrix of a ring network: `N` neurons at angles ``\theta_i = 2\pi i / N`` with
```math
W_{ij} = w_0 + (w_{max} - w_0)\,\exp\!\left(-\frac{d(\theta_i, \theta_j)^2}{2\sigma_w^2}
ight),
\qquad d = \min(|\theta_i - \theta_j|,\ 2\pi - |\theta_i - \theta_j|),
```
where the baseline
``w_0 = w_{max}\,\sigma_w\,(\mathrm{erf}(\pi/(\sqrt{2}\sigma_w)) - \sqrt{2\pi}) /
(\sigma_w\,\mathrm{erf}(\pi/(\sqrt{2}\sigma_w)) - \sqrt{2\pi})`` is computed by the function.
The diagonal is set to zero. `kwargs` are ignored.

Reference not given in the code.

# Returns
- `W::Matrix{Float64}` (`N x N`).
"""
function linear_network(N; σ_w = 0.38, w_max = 2.0, kwargs...)
    # Function to calculate wθ^sE
    function wθ_sE(θ_j, θ_i, w_0, w_, σ_w)
        return w_0 +
               (w_ - w_0) * exp(-(min(abs(θ_j - θ_i), 2π - abs(θ_j - θ_i)))^2 / (2 * σ_w^2))
    end

    # Function to calculate w_0
    function w_0(w, σ_w)
        return w * σ_w * (erf(π / (sqrt(2) * σ_w)) - sqrt(2π)) /
               (σ_w * erf(π / (sqrt(2) * σ_w)) - sqrt(2π))
    end

    w_norm = w_0(w_max, σ_w)

    neuron_position = [i * 2π / N for i = 1:N]
    W = zeros(N, N)
    for i = 1:N
        for j = 1:N
            W[i, j] = wθ_sE(neuron_position[i], neuron_position[j], w_norm, w_max, σ_w)
            W[j, j] = 0.0f0
        end
    end
    return W
end

"""
    spatial_activity(points, activity; T, L = nothing, N = nothing, grid_size = (x = [0, 0.1], y = [0, 0.1]))

Average `activity` over the cells of a regular 2D grid and over time windows.

# Arguments
- `points`: tuple `(xs, ys)` of the neuron coordinates.
- `activity::Matrix`: `N_neurons x N_timepoints`.
- `T`: time windows. A number `T` gives the windows `(1+(t-1)T):(tT)` for
  `t = 1:(N_timepoints ÷ T)`; a vector gives the windows explicitly (each element a range of
  column indices). (Up to SNNModels 1.8.4 the last column of each window was excluded.)
- `L` or `N` (exactly one must be given): cell side (number, or `(x = Lx, y = Ly)`), or number of
  cells per dimension (number, or `(x = Nx, y = Ny)`).
- `grid_size = (x = [x0, x1], y = [y0, y1])`: extent of the grid.

# Returns
- `spatial_avg::Array{Any,3}`: `nx x ny x n_windows`, mean activity of the neurons in each cell
  and window (0 for empty cells).
- `x_range`, `y_range`: ranges spanning `grid_size.x` and `grid_size.y` with
  `max(nx, 2)` and `max(ny, 2)` points.

# Example
```julia
using SpikingNeuralNetworks
xs = [0.05, 0.15, 0.25, 0.35]
ys = [0.05, 0.15, 0.25, 0.35]
activity = rand(4, 100)          # 4 neurons, 100 time points
avg, xr, yr = SNN.spatial_activity((xs, ys), activity; T = 10, L = 0.1,
                                   grid_size = (x = [0, 0.4], y = [0, 0.4]))
size(avg)   # (4, 4, 10)
```
"""
function spatial_activity(points, activity; T, L=nothing, N=nothing, grid_size = (x = [0, 0.1], y = [0, 0.1]))
    @assert isnothing(L) && !isnothing(N) || !isnothing(L) && isnothing(N) "Either L or N must be provided, but not both."

    xs, ys = points
    _, num_values = size(activity)

    time_indices = Vector{}()
    if isa(T, Number)
        for t = 1:(num_values÷T)
            push!(time_indices, (1+(t-1)*T):(t*T))
        end
    elseif isa(T, AbstractVector)
        for t in eachindex(T)
            push!(time_indices, T[t])
        end
    else
        error("T is: $(typeof(T)). It must be a number or a vector of time_indices")
    end

    if isnothing(L)
        Nx, Ny = isa(N, Number) ? (N, N) : (N.x, N.y)
        Lx = (grid_size.x[2] - grid_size.x[1]) / Nx
        Ly = (grid_size.y[2] - grid_size.y[1]) / Ny
    else
        Lx, Ly = isa(L, Number) ? (L, L) : (L.x, L.y)
    end

    # Define the grid size
    @unpack x, y = grid_size
    x_range = ceil(Int, diff(collect(x))[1] / Lx)
    y_range = ceil(Int, diff(collect(y))[1] / Ly)

    spatial_avg = Array{Any,3}(undef, x_range, y_range, length(time_indices))
    spatial_avg[:].=0
    for t in eachindex(time_indices)
        interval = time_indices[t]
        for j = 1:x_range
            for k = 1:y_range
                # Find points within the current grid cell
                indices_x = findall(_x -> ((j-1)*Lx <= _x-x[1] < j*Lx), xs) |> Set
                indices_y = findall(_y -> ((k-1)*Ly <= _y-y[1] < k*Ly), ys) |> Set
                indices = intersect(indices_x, indices_y) |> collect
                isempty(indices) && continue
                spatial_avg[j, k, t] = mean(activity[indices, interval])
            end
        end
    end
    x_range = range(x[1], stop = x[end], length = maximum([x_range, 2]))
    y_range = range(y[1], stop = y[end], length = maximum([y_range, 2]))
    return spatial_avg, x_range, y_range
end

export place_populations,
    periodic_distance,
    compute_connections,
    neurons_within_circle,
    neurons_outside_area,
    linear_network,
    spatial_activity
