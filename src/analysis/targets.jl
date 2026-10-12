using DSP
using Statistics
using Distributions

"""
    asynchronous_state(model, interval = nothing, pop = :Exc) -> (cv, ff, si)

Summary statistics of the asynchronous-irregular state of the population `model.pop.<pop>`.

- `cv`: mean over neurons of the ISI coefficient of variation `std(ISI) / (mean(ISI) + 1e-6)`
  (NaN set to 0), with the spikes restricted to `interval`.
- `ff`: `var / mean` of all the spike counts of the binned activity matrix (all neurons and bins
  pooled).
- `si`: mean of the `N x N` covariance matrix of the binned spike counts of the neurons
  (`cov(bins, dims = 2)`), used as a synchrony index.

`interval` (default `0s:0.5s:get_time(model)`) is an `AbstractRange`; its step is the bin width.

# Example
```julia
using SpikingNeuralNetworks
@load_units
Exc = SNN.Poisson(N = 20, param = SNN.PoissonParameter(10Hz))
SNN.monitor!(Exc, [:fire])
model = SNN.compose(; Exc, silent = true)
sim!(model, 2s)
cv, ff, si = SNN.asynchronous_state(model, 0s:100ms:2s, :Exc)
```
"""
function asynchronous_state(model, interval = nothing, pop = :Exc)
    population = getfield(model.pop, pop)
    interval = interval === nothing ? (0s:0.5s:get_time(model)) : interval
    bins, _ = bin_spiketimes(population; interval, do_sparse = false)
    # Calculate the Coefficient of Variation (CV) of ISIs
    isis = isi(population; interval)
    cv = std.(isis) ./ (mean.(isis) .+ 1e-6)  # Adding a small value to avoid division by zero
    cv[isnan.(cv)] .= 0.0  # Replace NaN values with 0.0
    cv = mean(cv)

    # Calculate the Fano Factor (FF)
    ff = var(bins) / mean(bins)  # Fano Factor

    ## Calculate the Synchrony Index (SI)
    si = mean(cov(bins, dims = 2))

    return cv, ff, si
end

"""
    is_attractor_state(pop::AbstractPopulation, interval::AbstractVector; ratio = 0.3, σ = 10.0f0, false_value = missing)

Test whether the time-averaged firing-rate profile over the neurons of `pop` (ordered by index,
e.g. a ring network) is a single bump.

The per-neuron mean rate over `interval` (from `firing_rate(pop; interval)`) is smoothed with
`gaussian_kernel_estimate(rates, σ; boundary = :continuous)` (periodic). The profile is
unimodal if `is_unimodal(kde, ratio)` holds for the profile or for the profile shifted by half
its length.

# Returns
- `(width, kde)`: if unimodal, `width` is the number of neurons above half of the peak divided
  by `σ`;
- `(false_value, kde)` otherwise.
"""
function is_attractor_state(
    pop::T,
    interval::AbstractVector;
    ratio::Real = 0.3,
    σ::Real = 10.0f0,
    false_value = missing,
) where {T<:AbstractPopulation}
    # Calculate the firing rate over the last N seconds

    rates, r = firing_rate(pop; interval, interpolate = true)
    ave_rate = mean(rates, dims = 2)[:, 1]
    kde = gaussian_kernel_estimate(ave_rate, σ, boundary = :continuous)
    # Check if the firing rate distribution is unimodal
    if (is_unimodal(kde, ratio) || is_unimodal(circshift(kde, length(kde) ÷ 2), ratio))
        # get the half width in σ
        peak, center = findmax(kde)
        return length(findall(x -> x > peak/2, kde))/σ, kde
    else
        return false_value, kde
    end
end


# Find bimodal value
# Here I use a simple algorithm that is described in :
# Journal of the Royal Statistical Society. Series B (Methodological)
# Using Kernel Density Estimates to Investigate Multimodality
# https://www.jstor.org/stable/2985156

# It consists in  using Normal kernels to approximate the data and then leverages a theorem on decreasing monotonicity of the number of maxima as function of the window span.

# Kernel Density Estimation
function KDE(t::Real, h::Real, ys)
    ndf(x, h) = exp(-x^2 / h)
    1 / length(ys) * 1 / h * sum(ndf.(ys .- t, h))
end

# Distribution
function globalKDE(h::Real, ys; xs::AbstractVector, distance::Function )
    kde = zeros(Float64, length(xs))
    @fastmath @inbounds for n = eachindex(xs)
            kde[n] = KDE(xs[n], h, ys)
    end
    return kde
end

"""
    get_maxima(data)

Indices `x` of the strict interior local maxima of `data` (`data[x] > data[x-1]` and
`data[x] > data[x+1]`).
"""
function get_maxima(data)
    arg_maxima = []
    for x = 2:(length(data)-1)
        (data[x] > data[x-1]) && (data[x] > data[x+1]) && (push!(arg_maxima, x))
    end
    return arg_maxima
end

"""
    is_unimodal(kernel, ratio)

`true` if at most one local maximum of `kernel` (see `get_maxima`) exceeds `ratio` times the
largest local maximum, i.e. maxima below `ratio` of the main peak are treated as spurious.
Throws if `kernel` has no interior local maximum.
"""
function is_unimodal(kernel, ratio)
    maxima = get_maxima(kernel)
    z = maximum(kernel[maxima])
    real = []
    for n in maxima
        m = kernel[n]
        if (abs(m / z) > ratio)
            push!(real, m)
        end
    end
    if length(real) > 1
        return false
    else
        return true
    end
end


# Spikes of a sorted train inside [t0, t1], as a view (no allocation).
_restrict(st::AbstractVector{Float32}, t0::Float32, t1::Float32) =
    @view st[searchsortedfirst(st, t0):searchsortedlast(st, t1)]

# Fraction of [t0, t1] covered by the union of the windows [s-Δt, s+Δt], clipped to [t0, t1], of the
# spikes s of a sorted train inside [t0, t1] (Cutts & Eglen 2014). Non-mutating, no allocation.
function _tile_fraction(spiketrain::AbstractVector{Float32}, Δt::Float32, t0::Float32, t1::Float32)
    st = _restrict(spiketrain, t0, t1)
    isempty(st) && return 0f0
    covered = 0f0
    lo, hi = max(st[1] - Δt, t0), min(st[1] + Δt, t1)
    @inbounds for n in 2:length(st)
        a = max(st[n] - Δt, t0)
        if a > hi   # disjoint window: close the current segment
            covered += hi - lo
            lo = a
        end
        hi = min(st[n] + Δt, t1)
    end
    return (covered + hi - lo) / (t1 - t0)
end

# Fraction of spikes in A with a coincident spike in B within Δt.
# B must be pre-sorted; O(Nₐ log N_B) via binary search, zero allocations.
function _coincident_fraction(A::AbstractVector{Float32}, B::AbstractVector{Float32}, Δt::Float32)
    isempty(A) && return 0f0
    nB    = length(B)
    count = 0
    @inbounds for t in A
        lo = searchsortedfirst(B, t - Δt)
        if lo <= nB && B[lo] <= t + Δt
            count += 1
        end
    end
    return Float32(count) / length(A)
end

# One term of the STTC; 1 when P = T = 1 (0/0, Elephant convention).
_sttc_term(P::Float32, T::Float32) = P * T == 1f0 ? 1f0 : (P - T) / (1f0 - P * T)

# A and B sorted and restricted to the interval, TA and TB their tile fractions.
function _sttc_pair(A::AbstractVector{Float32}, B::AbstractVector{Float32}, TA::Float32, TB::Float32, Δt::Float32)
    (isempty(A) || isempty(B)) && return 0f0
    PA = _coincident_fraction(A, B, Δt)
    PB = _coincident_fraction(B, A, Δt)
    return 0.5f0 * (_sttc_term(PA, TB) + _sttc_term(PB, TA))
end

_sorted(st::Vector{Float32}) = issorted(st) ? st : sort(st)

@doc raw"""
    STTC(spiketrainA::Vector{Float32}, spiketrainB::Vector{Float32}, Δt::Float32, interval::AbstractVector)

Spike Time Tiling Coefficient between two spike trains,
```math
\mathrm{STTC} = \frac{1}{2}\left(\frac{P_A - T_B}{1 - P_A T_B} + \frac{P_B - T_A}{1 - P_B T_A}\right),
```
computed on the spikes inside `[t0, t1] = [interval[1], interval[end]]`. ``P_A`` is the fraction
of spikes of A with a spike of B within ``\pm\Delta t``; ``T_A`` is the length of the union of the
windows ``[s - \Delta t, s + \Delta t]`` around the spikes of A, clipped to `[t0, t1]`, divided by
`t1 - t0`. A term with ``P T = 1`` (0/0) is 1, as in Elephant; the result is 0 if either train has
no spike in the interval. The inputs are not modified.

The coefficient saturates when ``\Delta t`` times the firing rate is about 1 or more (``T \to 1``):
choose ``\Delta t`` much shorter than the inter-spike interval.

# References
Cutts, C. S., & Eglen, S. J. (2014). Detecting pairwise correlations in spike trains: an
objective comparison of methods and application to the study of retinal waves. Journal of
Neuroscience, 34(43), 14288–14303.

# Example
```julia
using SpikingNeuralNetworks
a = Float32[10, 50, 90]
b = Float32[11, 52, 300]
SNN.STTC(a, b, 5.0f0, 0:1:400)
```
"""
function STTC(spiketrainA::Vector{Float32}, spiketrainB::Vector{Float32}, Δt::Float32, interval::AbstractVector)
    t0, t1 = Float32(interval[1]), Float32(interval[end])
    A = _restrict(_sorted(spiketrainA), t0, t1)
    B = _restrict(_sorted(spiketrainB), t0, t1)
    _sttc_pair(A, B, _tile_fraction(A, Δt, t0, t1), _tile_fraction(B, Δt, t0, t1), Δt)
end

"""
    tile_interval(spiketrainA::Vector{Float32}, Δt::Float32, interval::StepRangeLen{Float32})

Fraction ``T_A`` of `interval` covered by the windows ``±Δt`` around the spikes of
`spiketrainA` (the tiling term of `STTC`). Sorts `spiketrainA` in place.
"""
function tile_interval(spiketrainA::Vector{Float32}, Δt::Float32, interval::StepRangeLen{Float32})
    sort!(spiketrainA)
    _tile_fraction(spiketrainA, Δt, Float32(interval[1]), Float32(interval[end]))
end

"""
    STTC(spiketrains::Vector{Vector{Float32}}, Δt, interval = nothing) -> Matrix{Float32}
    STTC(pop::AbstractPopulation; ΔT, interval)
    STTC(pop::AbstractPopulation, ΔT::Real, interval::AbstractVector)

Symmetric matrix of the pairwise `STTC` values of a set of spike trains (diagonal = 1, pairs with
a train without spikes in the interval = 0), computed with threads over rows.

# Arguments
- `spiketrains`: A vector of vectors containing the spike times of each neuron.
- `Δt`: The coincidence window (ms).
- `interval`: recording interval; if `nothing`, it spans from the first spike minus `Δt` to the
  last spike plus `Δt`.

The population methods use `spiketimes(pop)`.
"""
function STTC(spiketrains::Vector{Vector{Float32}}, Δt, interval = nothing)
    n  = length(spiketrains)
    Δt = Float32(Δt)
    if isnothing(interval)
        ss     = reduce(vcat, spiketrains)
        isempty(ss) && return zeros(Float32, n, n)
        istart = minimum(ss) - Δt
        iend   = maximum(ss) + Δt
    else
        istart = Float32(interval[1])
        iend   = Float32(interval[end])
    end

    sorted = [_restrict(_sorted(st), istart, iend) for st in spiketrains]
    T = [_tile_fraction(st, Δt, istart, iend) for st in sorted]

    sttc_matrix = zeros(Float32, n, n)
    for i in 1:n; sttc_matrix[i, i] = 1f0; end

    Threads.@threads for i in 1:n
        @inbounds for j in (i+1):n
            v = _sttc_pair(sorted[i], sorted[j], T[i], T[j], Δt)
            sttc_matrix[i, j] = v
            sttc_matrix[j, i] = v
        end
    end
    return sttc_matrix
end

function STTC(pop::T; ΔT, interval) where {T<:AbstractPopulation}
    STTC(spiketimes(pop), ΔT, interval)
end

STTC(pop::T, ΔT::R, interval::V) where {T<:AbstractPopulation,R<:Real,V<:AbstractVector} =
    STTC(pop; ΔT, interval)




export is_unimodal,
    get_maxima, gaussian_kernel_estimate, gaussian_kernel, asynchronous_state

# #Trash spurious values (below 30% of the true maximum)
# function count_maxima(kernel, ratio)
#     maxima = get_maxima(kernel)
#     z = maximum(kernel[maxima])
#     real_maxima = []
#     for n in maxima
#         m = kernel[n]
#         if (abs(m / z) > ratio)
#             push!(real_maxima, m)
#         end
#     end
#     return length(real_maxima)
# end

# # Return the critical window (hence the bimodal factor)
# function critical_window(data; ratio = 0.3, max_b = 50, v_range = collect(-90:-35))
#     for h = 1:max_b
#         kernel = globalKDE(h, data, v_range = v_range)
#         bimodal = false
#         try
#             bimodal = isbimodal(kernel, ratio)
#         catch
#             bimodal = false
#             @error "Bimodal failed"
#         end
#         if !bimodal
#             return h
#         end
#     end
#     return max_b
# end

# # Return the critical window (hence the bimodal factor)
# function all_windows(data, ratio = 0.3; max_b = 50)
#     counter = zeros(max_b)
#     for h = 1:max_b
#         kernel = globalKDE(h, data)
#         counter[h] = count_maxima(kernel, ratio)
#     end
#     return counter
# end

"""
    gaussian_kernel(σ::Real, ll::Int)

Gaussian kernel `exp(-t^2 / (2σ^2))`, normalised to unit sum, sampled at `ll` equally spaced
points `t` between `-(ll ÷ 2)` and `ll ÷ 2` (unit spacing when `ll` is odd).
# Arguments
- `σ`: Standard deviation of the Gaussian kernel.
- `length`: Length of the kernel.
# Returns
- `kernel`: A vector representing the Gaussian kernel.
"""
function gaussian_kernel(σ::Real, ll::Int)
    t = range(-(ll ÷ 2), stop = ll ÷ 2, length = ll)
    kernel = exp.(-(t .^ 2) / (2 * σ^2))
    return kernel ./ sum(kernel)  # Normalize the kernel
end

"""
    gaussian_kernel_estimate(support_vector::Vector, σ::Real; boundary = :continuous)

Smooth `support_vector` with `gaussian_kernel(σ, length(support_vector))` by convolution.

# Arguments
- `support_vector`: The input vector.
- `σ`: Standard deviation of the Gaussian kernel, in samples.
- `boundary`: `:continuous` extends the vector periodically (wrap-around) before convolving;
  `:closed` pads it with zeros. Any other value throws an error.

# Returns
- The smoothed vector, cut from the full convolution of the extended vector: same length as
  the input for odd lengths, one element shorter for even lengths.
"""
function gaussian_kernel_estimate(support_vector::Vector, σ::Real; boundary = :continuous)

    # Apply the kernel using convolution
    # estimated_vector = conv(support_vector, kernel)
    kernel = gaussian_kernel(σ, length(support_vector))

    # Handle closed boundary conditions
    # Extend the support vector to handle boundaries
    ll = length(support_vector) ÷ 2
    if boundary == :continuous
        extension_left = support_vector[(end-ll):end]
        extension_right = support_vector[1:ll]
    elseif boundary == :closed
        extension_left = zeros(size(support_vector[(end-ll):end]))
        extension_right = zeros(size(support_vector[1:ll]))
    else
        error("Invalid boundary condition. Use :continuous or :closed.")
    end
    extended_vector = vcat(extension_left, support_vector, extension_right)

    # Apply the kernel to the extended vector
    extended_estimated_vector = conv(extended_vector, kernel)

    return extended_estimated_vector[2(1+ll):(end-2ll)]
end



export gaussian_kernel_estimate,
    gaussian_kernel, asynchronous_state, is_attractor_state, STTC, tile_interval



# # Example support vector
# support_vector = [2.0, 2.0, 4.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 4.0, 5.0, 4.0, 3.0, 2.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 8.0, 10,]

# # Standard deviation of the Gaussian kernel
# σ = 1.0

# # Length of the kernel

# # Apply the Gaussian kernel estimate
# estimated_vector = gaussian_kernel_estimate(support_vector, 2.0, boundary=:continuous)
# rotated_array = circshift(estimated_vector, 10)
