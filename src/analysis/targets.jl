using DSP
using Statistics
using Distributions

"""
    inter_spike_interval(spiketimes::Vector{Float32})

Calculate the inter-spike intervals (ISIs) for a given set of spike times.

# Arguments
- `spiketimes`: A vector of spike times for a single neuron.

# Returns
- `isis`: A vector of inter-spike intervals.
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
    is_attractor_state(spiketimes::Spiketimes, interval::AbstractVector, N::Int)

Check if the network is in an attractor state by verifying that the average firing rate over the last N seconds of the simulation is a unimodal distribution.

# Arguments
- `spiketimes`: A vector of vectors containing the spike times of each neuron.
- `interval`: The time interval over which to compute the firing rate.
- `N`: The number of seconds over which to check the unimodality of the firing rate distribution.

# Returns
- `is_attractor`: A boolean indicating whether the network is in an attractor state.
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

# # Kernel Density Estimation
# function KDE(t::Real, h::Real, ys)
#     ndf(x, h) = exp(-x^2 / h)
#     1 / length(data) * 1 / h * sum(ndf.(ys .- t, h))
# end

# # Distribution
# function globalKDE(h::Real, ys; xs::AbstractVector, distance::Function )
#     kde = zeros(Float64, length(xs))
#     @fastmath @inbounds for n = eachindex(xs)
#             kde[n] = KDE(xs[n], h, ys)
#     end
#     return kde
# end

#Get its maxima
function get_maxima(data)
    arg_maxima = []
    for x = 2:(length(data)-1)
        (data[x] > data[x-1]) && (data[x] > data[x+1]) && (push!(arg_maxima, x))
    end
    return arg_maxima
end

#Trash spurious values (below 30% of the true maximum)
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


"""
    STTC(spiketrain1::Vector{Float32}, spiketrain2::Vector{Float32}, Δt::Float32)
Calculate the Spike Time Tiling Coefficient (STTC) between two spike trains.
# Arguments
- `spiketrain1`: A vector of spike times for the first neuron.
- `spiketrain2`: A vector of spike times for the second neuron.
- `Δt`: The time window for considering spikes as coincident.
# Returns
- `sttc_value`: The calculated STTC value.
"""
##
function STTC(spiketrainA, spiketrainB, Δt, interval)
    # Implementation of the Spike Time Tiling Coefficient (STTC)
    TA = tile_interval(spiketrainA, Δt, interval)
    TB = tile_interval(spiketrainB, Δt, interval)
    PA = sum([any(abs.(spiketrainB .- t) .<= Δt) for t in spiketrainA]) / length(spiketrainA)
    PB = sum([any(abs.(spiketrainA .- t) .<= Δt) for t in spiketrainB]) / length(spiketrainB)
    sttc_value = 0.5 * ( (PA - TB) / (1 - PA*TB) + (PB - TA) / (1 - PB*TA) )
    return sttc_value
end

function tile_interval(spiketrainA, Δt, interval)
    width = Δt
    sort!(spiketrainA)
    for n in eachindex(spiketrainA)
        n == 1 && continue
        spiketrainA[n] < interval[1] && continue
        spiketrainA[n] > interval[end] && continue

        if spiketrainA[n] - spiketrainA[n-1] < 2Δt
            width += spiketrainA[n] - spiketrainA[n-1]
        else
            width += 2Δt
        end
    end
    width = width + Δt
    width / (interval[end] - interval[1] +2Δt)
end

"""
    STTC(spiketrains::Vector{Vector{Float32}}, Δt::Float32, interval::AbstractVector=nothing)
Calculate the Spike Time Tiling Coefficient (STTC) matrix for a set of spike trains.
# Arguments
- `spiketrains`: A vector of vectors containing the spike times of each neuron.
- `Δt`: The time window for considering spikes as coincident.
- `interval`: The time interval over which to compute the STTC. If not provided, it will be inferred from the first and last events in the spike trains.
# Returns
- `sttc_matrix`: A matrix containing the STTC values between all pairs of spike trains.
"""
function STTC(spiketrains::Vector{Vector{Float32}}, Δt, interval=nothing)
    n = length(spiketrains)
    sttc_matrix = zeros(Float32, n, n)
    if isnothing(interval) 
        ss = vcat(spiketrains...)
        interval = (-Δt + minimum(ss):maximum(ss)+Δt)
    end 
    for i in ProgressBar(1:n)
    @inbounds @fastmath @simd for j in i+1:n
            if i==j
                sttc_value = 1
            elseif length(spiketrains[i]) == 0 || length(spiketrains[j]) == 0
                sttc_value = 0
            else
                sttc_value = STTC(spiketrains[i], spiketrains[j], Δt, interval)
            end
            sttc_matrix[i, j] = sttc_value
            sttc_matrix[j, i] = sttc_value
        end
    end
    return sttc_matrix
end

function STTC(pop::T; ΔT, interval) where T <: AbstractPopulation
    STTC(spiketimes(pop), ΔT, interval)
end

STTC(pop::T, ΔT::R, interval::V) where {T<:AbstractPopulation, R<: Real, V<:AbstractVector } = STTC(pop; ΔT, interval)




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
    gaussian_kernel(σ::Float64, length::Int)

Create a Gaussian kernel with standard deviation `σ` and specified `length`.
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
    gaussian_kernel_estimate(support_vector::Vector{Float64}, σ::Float64, length::Int)

Apply a Gaussian kernel estimate to a support vector with closed boundary conditions.

# Arguments
- `support_vector`: The input support vector.
- `σ`: Standard deviation of the Gaussian kernel.
- `length`: Length of the kernel.

# Returns
- `estimated_vector`: The estimated vector after applying the Gaussian kernel.
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



export gaussian_kernel_estimate, gaussian_kernel, asynchronous_state, is_attractor_state, STTC, tile_interval



# # Example support vector
# support_vector = [2.0, 2.0, 4.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 4.0, 5.0, 4.0, 3.0, 2.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 8.0, 10,]

# # Standard deviation of the Gaussian kernel
# σ = 1.0

# # Length of the kernel

# # Apply the Gaussian kernel estimate
# estimated_vector = gaussian_kernel_estimate(support_vector, 2.0, boundary=:continuous)
# rotated_array = circshift(estimated_vector, 10)
