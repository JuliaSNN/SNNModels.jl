using Interpolations
using StatsBase
using DSP


"""
    spiketimes(p; interval=nothing) -> Spiketimes

Return spike times for every neuron in `p` as a `Spiketimes`
(`Vector{Vector{Float32}}`), one inner vector per neuron.

Dispatches on the fire recording format stored in `p.records[:fire]`:
- **COO / dense** (`meta[:mode][:fire] == :dense`): reads the pre-allocated
  `times_buf` / `neurons_buf` flat arrays written by the sim loop. If `interval`
  is given, binary-search on the sorted buffer restricts the scan to
  O(log(N_spikes) + N_spikes_in_interval) instead of a full linear pass.
- **Legacy** (`push!`-based `:time` / `:neurons` dict): used when the population
  was recorded without the dense infrastructure (no `monitor!` call before the sim).

# Arguments
- `p`: an `AbstractPopulation` or `AbstractStimulus` with a `:fire` record.

# Keyword arguments
- `interval`: optional `(t0, t1)` tuple or any `AbstractRange`. Spikes satisfy
  `t > t0` and `t < t1` (both bounds strict). Pass `nothing` (default) to return
  all recorded spikes.

# Notes
- Neurons are indexed `1:p.N` in the output vector; empty inner vectors mean
  the neuron never fired or fired outside the requested interval.
- The returned `Spiketimes` is always freshly allocated (safe to mutate).
- In the legacy format fewer than two recorded spike events return empty trains.

# Example
```julia
using SpikingNeuralNetworks
@load_units
E = SNN.Poisson(N = 10, param = SNN.PoissonParameter(20Hz))
SNN.monitor!(E, [:fire])
model = SNN.compose(; E, silent = true)
sim!(model, 1s)
st = SNN.spiketimes(E)                          # all spikes
st2 = SNN.spiketimes(E; interval = (200ms, 500ms))
```
"""
function spiketimes(
    p::T;
    interval = nothing,
    kwargs...,
) where {T<:Union{AbstractPopulation,AbstractStimulus}}
    # COO fire format: flat parallel arrays (times_buf, neurons_buf), one entry per spike.
    rec = p.records[:fire]
    meta = get(p.records, :meta, nothing)
    is_dense = meta !== nothing &&
        get(get(meta, :mode, Dict()), :fire, :legacy) === :dense &&
        haskey(rec, :times_buf)
    if is_dense
        return _spiketimes_coo(p, rec, meta, interval)
    end

    _spiketimes = _init_spiketimes(p.N)
    firing_time = rec[:time]
    neurons = rec[:neurons]

    # @warn "No spikes in population"
    if length(firing_time) < 2
        return _spiketimes
    end
    if isnothing(interval)
        interval = (0, firing_time[end]+1)
    end
    tt0, tt1 = findfirst(x -> x > interval[1], firing_time),
    findlast(x -> x < interval[end], firing_time)
    if isnothing(tt0) || isnothing(tt1)
        return _spiketimes
    end
    for tt = tt0:tt1
        for n in neurons[tt]
            push!(_spiketimes[n], firing_time[tt])
        end
    end
    return _spiketimes
end

# Read COO fire buffers. `wp` = number of written spikes.
# times_buf[1:wp] is chronologically sorted (sim writes in order), so
# interval filtering uses binary search: O(log(wp)) + O(spikes_in_interval).
function _spiketimes_coo(p, rec, meta, interval)
    _spiketimes = _init_spiketimes(p.N)
    times_buf = rec[:times_buf]
    neurons_buf = rec[:neurons_buf]
    wp = length(times_buf) - get(meta[:allocated], :fire, 0)
    wp <= 0 && return _spiketimes
    if isnothing(interval)
        @inbounds for i in 1:wp
            push!(_spiketimes[neurons_buf[i]], times_buf[i])
        end
    else
        lo = Float32(interval[1])
        hi = Float32(interval[end])
        lo_idx = searchsortedlast(times_buf, lo, 1, wp, Base.Order.Forward) + 1
        hi_idx = searchsortedfirst(times_buf, hi, 1, wp, Base.Order.Forward) - 1
        @inbounds for i in lo_idx:hi_idx
            push!(_spiketimes[neurons_buf[i]], times_buf[i])
        end
    end
    return _spiketimes
end

function _init_spiketimes(N)
    return Spiketimes([Vector{Float32}() for _ in 1:N])
end



"""
    spiketimes(Ps; kwargs...) -> Spiketimes

Concatenate spike times from all populations in `Ps` into a single `Spiketimes`
vector. Neurons from the first population occupy indices `1:Ps[1].N`, the second
population `Ps[1].N+1 : Ps[1].N+Ps[2].N`, and so on.

`Ps` can be a `NamedTuple` (as returned by `model.pop`) or a `Vector` of
`AbstractPopulation`/`AbstractStimulus`. All `kwargs` are forwarded to the
single-population `spiketimes` dispatch (e.g. `interval`).

See also: `spiketimes_split` to keep each population's trains separate.
"""
function spiketimes(Ps::NamedTuple; kwargs...)
    st = Vector{Vector{Float32}}()
    for p in Ps
        append!(st, spiketimes(p; kwargs...))
    end
    return Spiketimes(st)
end

function spiketimes(Ps::Vector{T}; kwargs...) where {T<:Union{AbstractPopulation,AbstractStimulus}}
    st = Vector{Vector{Float32}}()
    for p in Ps
        append!(st, spiketimes(p; kwargs...))
    end
    return Spiketimes(st)
end

"""
    spiketimes_split(Ps; kwargs...) -> (Vector{Spiketimes}, Vector{String})

Return, for each element of `Ps` that has a `:fire` record, its `spiketimes(p; kwargs...)`,
together with the vector of the population names. Elements without a `:fire` record are skipped.
"""
function spiketimes_split(Ps; kwargs...)
    st_ps = Vector{Vector{Vector{Float32}}}()
    names = Vector{String}()
    for p in Ps
        haskey(p.records, :fire) || continue
        _st = spiketimes(p; kwargs...)
        push!(st_ps, Spiketimes(_st))
        push!(names, p.name)
    end
    return st_ps, names
end


"""
    spikecount(pop::AbstractPopulation, Trange::AbstractVector, neurons::Vector{Int})

Total number of spikes emitted by the neurons `neurons` of `pop` in the interval `Trange`
(open bounds, see `spiketimes`).
"""
function spikecount(
    pop::T,
    Trange::Q,
    neurons::Vector{Int},
) where {T<:AbstractPopulation,Q<:AbstractVector}
    return length.(spiketimes(pop, interval = Trange)[neurons]) |> sum
end

export spikecount

# spikecount(x::Spiketimes) = length.(x)

# function alpha_function(t::T; t0::T, τ::T) where {T<:Float32}
#     return exp64(- (t - t0) / τ) * Θ((t - t0))
# end

# """
#     Θ(x::Float64)

#     Heaviside function
# """
# Θ(x::Float32) = x > 0.0 ? x : 0.0


@doc raw"""
    alpha_function(t::Float32, τ::Float32)

Alpha function ``\alpha(t) = (t/\tau)\,e^{1 - t/\tau}`` for ``t \ge 0`` and ``0`` otherwise
(peak value 1 at ``t = \tau``). Used by `alpha_kernel`.
"""
function alpha_function(t::Float32, τ::Float32)
    if t >= 0
        return (t / τ) * exp(1 - t / τ)
    else
        return 0.0
    end
end

"""
    alpha_kernel(; τ = 25ms, interval, kwargs...)

Causal alpha-function kernel sampled with the step of `interval` on `0:step(interval):10τ`, and
normalised so that `sum(kernel) * step(interval) == 1` (unit area, in 1/ms). Default kernel of
`firing_rate`.
"""
function alpha_kernel(;τ=25ms, interval, kwargs...)
    kernel_length = τ * 10  # Length of the kernel in ms
    bin_width = step(interval) # kernel window in ms
    kernel_time = Float32.(0.0:bin_width:kernel_length)
    τ = Float32(τ)
    alpha_kernel = [alpha_function(t, τ) for t in kernel_time]
    alpha_kernel ./= sum(alpha_kernel) * bin_width
    return alpha_kernel
end
"""
    merge_spiketimes(spikes::Vector{Spiketimes}; start::Float32 = 0.0f0) -> Spiketimes
    merge_spiketimes(spikes::Spiketimes) -> Vector{Float32}

Merge the spike trains of several simulations of the same network, neuron by neuron.

With `start > 0`, the spikes of the `k`-th element are shifted by `start * (k - 1)` **in place**
(the input is modified) before merging, so that consecutive simulations are placed one after the
other. The result is sorted per neuron. The neurons are split over `Threads.nthreads()` tasks;
the function is not meant to be called from several threads at once.

The `Spiketimes` method returns all spikes of all neurons in one sorted vector.
"""
function merge_spiketimes(spikes::Vector{Spiketimes}; start::Float32=0.0f0)
    if start > 0.0f0
        for sp in eachindex(spikes)
            for n in eachindex(spikes[sp])
                spikes[sp][n] .+= start * (sp - 1)
            end
        end
    end
    neurons = [Vector{Float32}() for _ = 1:length(spikes[1])]
    neuron_ids = collect(1:length(spikes[1]))
    sub_indices = k_fold(neuron_ids, Threads.nthreads())
    sub_neurons = [neuron_ids[x] for x in sub_indices]
    Threads.@threads for p in eachindex(sub_indices)
        for spiketimes in spikes
            for (n, id) in zip(sub_indices[p], sub_neurons[p])
                push!(neurons[n], spiketimes[id]... )
            end
        end
    end
    return sort!.(neurons)
end


function k_fold(vector, k, do_shuffle = false)
    if do_shuffle
        ns = shuffle(1:length(vector))
    else
        ns = 1:length(vector)
    end
    b = length(ns) ÷ k
    indices=Vector{Vector{Int}}()
    for i = 1:(k-1)
        push!(indices, ns[((i-1)*b+1):(b*i)])
    end
    push!(indices, ns[(1+b*(k-1)):end])
    return indices
end


function merge_spiketimes(spikes::Spiketimes;)
    sort(vcat(spikes...))
end

"""
    firing_rate(spiketimes::Spiketimes; interval, kwargs...) -> (rates, interval)

    firing_rate(p::Union{AbstractPopulation,AbstractStimulus}; kwargs...)
    firing_rate(P, interval::AbstractRange; kwargs...)
    firing_rate(populations; mean_pop = false, kwargs...) -> (rates_per_pop, interval, names)

Estimate per-neuron instantaneous firing rates (Hz) from a `Spiketimes` object by
convolving each binned spike train (bin = `step(interval)`) with a kernel of unit area,
then multiplying by `1s`. Returns `(rates, interval)`.

`interval` should be an `AbstractRange` in milliseconds (e.g. `0:1ms:500ms`). It
sets the time grid for binning and convolution. If it is omitted, it is built as
`tt0:sampling:ttf` with the keyword arguments `sampling = 20ms`, `tt0 = 0` and `ttf` = last spike
time.

A population (or stimulus) is converted with `spiketimes` first. The method for a collection
of populations (e.g. `model.pop`) applies `spiketimes_split` and returns one rate object per
population with a `:fire` record (`mean_pop = true` averages each over its neurons).

# Keyword arguments
- `interval`: time grid in ms, e.g. `0f0:1f0:1000f0`.
- `kernel`: kernel function called as `kernel(; interval, kwargs...)` returning a
  `Vector` of weights. Default: `alpha_kernel` (causal alpha function, controlled
  by the `τ` kwarg, default `τ=25ms`).
- `neurons`: neuron subset. `:ALL` (default) processes every neuron; pass an
  `Int` or `Vector{Int}` to select a subset. Indexing is into `spiketimes`.
- `interpolate` (`true`): when `true`, wraps the `(N, T)` rate matrix in a
  `ScaledInterpolation` so rates can be evaluated at arbitrary `(neuron, time)`
  pairs via `rates(n, t)`. When `false`, returns a plain `Matrix{Float64}`
  of shape `(N_neurons, length(interval))`.
- `pop_average` (`false`): if `true`, averages over neurons (dim 1) and returns a
  plain `Vector` of length `length(interval)` regardless of `interpolate`.
- `time_average` (`false`): if `true`, skips convolution entirely and returns the
  mean firing rate per neuron (in Hz) over `interval` as a `Vector{Float32}`
  (spikes with `first(interval) < t <= last(interval)`; the `neurons` selection is not applied).
  Setting both `time_average` and `pop_average` collapses to a single scalar.

# Return types summary

| `interpolate` | `pop_average` | `time_average` | return type of `rates` |
|:---:|:---:|:---:|:---|
| `true`  | `false` | `false` | `ScaledInterpolation` — call as `rates(n, t)` |
| `false` | `false` | `false` | `Matrix{Float64}` shape `(N, T)` |
| any     | `true`  | `false` | `Vector` of length `T` |
| any     | `false` | `true`  | `Vector{Float32}` of length `N` (Hz) |
| any     | `true`  | `true`  | scalar `Float32` (Hz) |

# Example
```julia
using SpikingNeuralNetworks
@load_units
E = SNN.Poisson(N = 10, param = SNN.PoissonParameter(20Hz))
SNN.monitor!(E, [:fire])
sim!(SNN.compose(; E, silent = true), 2s)
st = SNN.spiketimes(E)
fr, r = SNN.firing_rate(st; interval = 0:1ms:2s, τ = 50ms)
fr(3, 500f0)                                   # neuron 3 at t = 500 ms
fr_avg, r = SNN.firing_rate(st; interval = 0:1ms:2s, pop_average = true)
mean_rates, _ = SNN.firing_rate(E; interval = 0:1ms:2s, time_average = true)
```
"""
function firing_rate(
    spiketimes::Spiketimes;
    interval::AbstractVector = [],
    interpolate = true,
    pop_average = false,
    time_average = false,
    neurons = :ALL,
    kernel = alpha_kernel,
    kwargs...,
)
    interval = _retrieve_interval(interval, spiketimes; kwargs...)
    neurons =
        neurons == :ALL ? eachindex(spiketimes) : (isa(neurons, Int) ? [neurons] : neurons)
    rates = nothing
    if time_average
        return time_average_fr(spiketimes, interval, pop_average), interval
    end

    if length(spiketimes) < 1
        rates = zeros(Float32, 0, length(interval))
    elseif all(isempty.(spiketimes))
        rates = zeros(Float32, length(spiketimes[neurons]), length(interval))
    else
        spiketimes = spiketimes[neurons]
        conv_kernel = kernel(;interval, kwargs...)
        my_rates = zeros(length(spiketimes), length(interval))
        Threads.@threads for n in eachindex(spiketimes)
            spike_train, _ = bin_spiketimes(spiketimes[n]; interval = interval, do_sparse = false)
            c = conv(spike_train, conv_kernel)
            @inbounds for t in eachindex(interval)
                my_rates[n, t] = c[t] * s
            end
        end
        rates = my_rates
    end

    if interpolate
        interp = get_interpolator(rates)
        rates = Interpolations.scale(
            Interpolations.interpolate(rates, interp),
            1:length(spiketimes),
            interval,
        )
    else
        rates = copy(rates)
    end

    if pop_average
        rates = mean(rates, dims = 1)[1, :]
    end
    return rates, interval
end

# `st` is the Spiketimes being processed; used to infer the span when the caller
# passes no interval. Previously this function mistakenly captured the exported
# `spiketimes` function from module scope instead of the local spike data.
function _retrieve_interval(interval, st; sampling = 20ms, ttf = -1, tt0 = -1, kwargs...)
    if isempty(interval)
        max_time =
            all(isempty.(st)) ? 1.0f0 : maximum(Iterators.flatten(st))
        tt0 = tt0 > 0 ? Float32(tt0) : 0.0f0
        ttf = ttf > 0 ? Float32(ttf) : max_time
        interval = tt0:sampling:ttf
    end
    return interval
end

function time_average_fr(spiketimes, interval, pop_average)
    lo = Float32(interval[1])
    hi = Float32(interval[end])
    dur_s = (hi - lo) / 1000f0
    rates = Vector{Float32}(undef, length(spiketimes))
    @inbounds for n in eachindex(spiketimes)
        count = 0
        for t in spiketimes[n]
            count += (t > lo) & (t <= hi)
        end
        rates[n] = dur_s > 0 ? Float32(count) / dur_s : 0f0
    end
    if pop_average
        m = mean(rates)
        return isnan(m) ? 0.0f0 : m
    end
    return rates
end

firing_rate(P, interval::T; kwargs...) where {T<:AbstractRange} =
    firing_rate(P; interval, kwargs...)

function firing_rate(
    population::T;
    kwargs...,
) where {T<:Union{AbstractPopulation,AbstractStimulus}}
    return firing_rate(spiketimes(population); kwargs...)
end

function firing_rate(populations; mean_pop = false, kwargs...)
    spiketimes_pop, names_pop = spiketimes_split(populations)
    fr_pop = []
    interval = nothing
    for n in eachindex(spiketimes_pop)
        rates, interval = firing_rate(spiketimes_pop[n]; pop_average = mean_pop, kwargs...)
        push!(fr_pop, rates)
    end
    return fr_pop, interval, names_pop
end

"""
    average_firing_rate(spiketimes::Spiketimes; interval = [])
    average_firing_rate(populations; interval)

`Spiketimes` method: per-neuron time average of the convolved rate returned by
`firing_rate(spiketimes; interval, interpolate = false)` (Hz).

The `populations` method (a population, a vector or a `NamedTuple` of populations) returns
the histogram of all spike times over the bin edges `interval` and the left bin edges
`interval[1:end-1]`. (Up to SNNModels 1.8.4 it failed with an `UndefVarError`: a local variable
shadowed the function `spiketimes`.)
"""
function average_firing_rate(
    spiketimes::Spiketimes;
    interval::AbstractVector = [],
)
    rates, interval = firing_rate(
        spiketimes;
        interval = interval,
        interpolate = false,
    )
    return mean.(rates)
end

function average_firing_rate(populations; interval)
    st = spiketimes(populations)
    return sort(vcat(st...)) |> x -> fit(Histogram, x, interval).weights,
    interval[1:(end-1)]
end

"""
    compute_cross_correlogram(spike_times1::Vector{Float32}, spike_times2 = Float32[];
                              bin_width = 1ms, max_lag = 100.0, shift_predictor = false)

Return `(lags, corr)`: the cross-correlogram (`DSP.xcorr`) of the spike trains of two neurons
binned with `bin_width` on the common interval `0:bin_width:(t_last + max_lag)`, or the
autocorrelogram if `spike_times2` is empty (with the zero-lag bin set to 0), for lags up to
`max_lag` ms. `corr[k]` counts the pairs with `t2 - t1` in the lag bin `lags[k]`. With
`shift_predictor = true` the second train is shifted by 1000 ms.

(Up to SNNModels 1.8.4 the function always threw: `bin_spiketimes` was called without the
required `interval` and the autocorrelation branch referred to an undefined `auto_corr`.)
"""
function compute_cross_correlogram(
    spike_times1::Vector{Float32},
    spike_times2::Vector{Float32} = Float32[];
    bin_width = 1.0ms,
    max_lag = 100.0,
    shift_predictor = false,
)
    auto = isempty(spike_times2)
    if !auto && shift_predictor
        spike_times2 = spike_times2 .+ 1000.0f0
    end
    interval = _correlogram_interval(spike_times1, spike_times2, bin_width, max_lag)
    spike_train1, _ = bin_spiketimes(spike_times1; interval, do_sparse = false)
    spike_train2 = auto ? spike_train1 : first(bin_spiketimes(spike_times2; interval, do_sparse = false))

    # Compute the auto-correlation (cross-correlogram with itself)
    _corr = xcorr(spike_train1, spike_train2)

    # Compute time lags
    bins = length(_corr) ÷ 2
    lags = ((-bins):bins) .* bin_width

    # Trim the lags and auto-correlation to the specified max_lag
    lag_mask = abs.(lags) .<= max_lag
    lags = lags[lag_mask]
    _corr = _corr[lag_mask]

    auto && (_corr[length(lags)÷2+1] = 0)

    return lags, _corr
end

# Common binning interval of two spike trains for the correlogram functions.
function _correlogram_interval(st1, st2, bin_width, max_lag)
    tmax = maximum(vcat(Float32[0], st1, st2)) + Float32(max_lag) + Float32(bin_width)
    return 0.0f0:Float32(bin_width):tmax
end

@doc raw"""
    compute_covariance_density(spike_times1::Vector{Float32}, spike_times2::Vector{Float32};
                               bin_width = 1ms, max_lag = 200ms)

Return `(lags, C)` with the covariance density
``C(\tau) = \mathrm{xcorr}(\tau) - \lambda_x \lambda_y\,\Delta\,n_{bins}`` (cross-correlogram
minus its value for independent trains with rates ``\lambda_x, \lambda_y``; ``\Delta`` =
`bin_width`).

Both trains are binned on the same interval as in `compute_cross_correlogram`. (Up to
SNNModels 1.8.4 the function always threw, see `compute_cross_correlogram`.)
"""
function compute_covariance_density(
    spike_times1::Vector{Float32},
    spike_times2::Vector{Float32};
    bin_width = 1ms,
    max_lag = 200ms,
)
    # Compute the cross-correlogram
    lags, cross_corr =
        compute_cross_correlogram(spike_times1, spike_times2; bin_width, max_lag)
    interval = _correlogram_interval(spike_times1, spike_times2, bin_width, max_lag)
    spike_train1, _ = bin_spiketimes(spike_times1; interval, do_sparse = false)
    spike_train2, _ = bin_spiketimes(spike_times2; interval, do_sparse = false)

    # Compute mean firing rates
    λ_x = mean(spike_train1) / bin_width
    λ_y = mean(spike_train2) / bin_width

    # Compute covariance density
    covariance_density = cross_corr .- (λ_x * λ_y * bin_width * length(spike_train1))

    return lags, covariance_density
end


"""
    bin_spiketimes(spike_times::Vector{Float32}; interval::AbstractRange, do_sparse = true)
    bin_spiketimes(spiketimes::Spiketimes; interval, do_sparse = true)
    bin_spiketimes(p::Union{AbstractPopulation,AbstractStimulus}; kwargs...)
    bin_spiketimes(P, interval::AbstractRange; kwargs...)
    bin_spiketimes(populations::NamedTuple; interval = nothing, do_sparse = true)

Count spikes in the bins of `interval` (bin width `step(interval)`, one bin per element of
`interval`; bin `k` covers `[first(interval) + (k-1)Δ, first(interval) + kΔ)`). Spikes at or
before `first(interval)` are ignored.

# Returns
- One train: `(counts, interval)` with `counts` a `SparseVector` (`do_sparse = true`) or a dense
  `Vector{Float64}` of length `length(interval)`.
- `Spiketimes` or a population: `(counts::Matrix{Float64}, interval)` of size
  `(N_neurons, length(interval))`.
- `NamedTuple` of populations: `(counts_per_population, interval, names)`, via
  `spiketimes_split`.

# Example
```julia
using SpikingNeuralNetworks
@load_units
st = SNN.Spiketimes([[1.5f0, 2.2f0, 7.0f0], Float32[3.0]])
counts, r = SNN.bin_spiketimes(st; interval = 0:1ms:10ms)   # 2 x 11 matrix
```
"""
function bin_spiketimes(
    spike_times::Vector{Float32};
    interval::AbstractRange,
    do_sparse = true,
)
    bin_width = step(interval)
    spike_train = zeros(length(interval))
    st = sort(spike_times) .- first(interval)
    first_st = findfirst(x -> x > 0, st) |> x-> isnothing(x) ? length(st)+1 : x
    last_st = findlast(x -> x < length(interval)*bin_width, st) |> x-> isnothing(x) ? length(st) : x
    for i in first_st:last_st
        index = floor(Int, st[i] / bin_width) + 1
        if index <= length(spike_train)
            spike_train[index] += 1.0
        end
    end
    if do_sparse
        return sparse(spike_train), interval
    else
        return spike_train, interval
    end
end

function bin_spiketimes(spike_times::Spiketimes; kwargs...)
    sample, r = bin_spiketimes(spike_times[1]; kwargs..., do_sparse = false)
    bin_array = zeros(length(spike_times), length(sample))
    tmap(eachindex(spike_times)) do n
        bin_array[n, :] = bin_spiketimes(spike_times[n]; kwargs...)[1]
    end
    return bin_array, r
end

bin_spiketimes(P::AbstractPopulation; kwargs...) = bin_spiketimes(spiketimes(P); kwargs...)
bin_spiketimes(P::AbstractStimulus; kwargs...) = bin_spiketimes(spiketimes(P); kwargs...)
bin_spiketimes(P, interval::T; kwargs...) where {T<:AbstractRange} =
    bin_spiketimes(P; interval, kwargs...)

function bin_spiketimes(populations::NamedTuple; interval=nothing, do_sparse = true)
    st_pops, names_pop = spiketimes_split(populations)
    ss = map(st->bin_spiketimes(st; interval, do_sparse)[1], st_pops)
    return ss, interval, names_pop
end

"""
    shift_spikes!(spiketimes::Spiketimes, delay::Number)

Add `delay` (ms) to every spike time, in place.
"""
function shift_spikes!(spiketimes::Spiketimes, delay::Number)
    for n in eachindex(spiketimes)
        spiketimes[n] .+= delay
    end
end


"""
    isi(spiketimes::Spiketimes) -> Vector{Vector{Float32}}
    isi(spiketimes::Vector{Float32}) -> Vector{Float32}
    isi(pop::AbstractPopulation; interval = nothing)

Inter-spike intervals (ms), i.e. `diff` of each (sorted) spike train.
"""
function isi(spiketimes::Spiketimes)
    return diff.(spiketimes)
end

function isi(spiketimes::Vector{Float32})
    return diff(spiketimes)
end

function isi(pop::T; interval = nothing) where {T<:AbstractPopulation}
    return spiketimes(pop; interval = interval) |> isi
end

# isi(spiketimes::NNSpikes, pop::Symbol) = read(spiketimes, pop) |> x -> diff.(x)

function CV(spikes::Spiketimes)
    intervals = isi(spikes;)
    cvs = sqrt.(var.(intervals) ./ (mean.(intervals) .^ 2))
    cvs[isnan.(cvs)] .= -0.0
    return cvs
end

"""
    spikes_in_interval(spiketimes::Spiketimes, interval, margin = [0, 0]; collapse = false) -> Spiketimes

Copy of the spikes with `interval[1] + margin[1] < t <= interval[end] + margin[2]` (spike
trains must be sorted). `collapse` is accepted and ignored.
"""
function spikes_in_interval(
    spiketimes::Spiketimes,
    interval,
    margin = [0, 0];
    collapse::Bool = false,
) 
    neurons = [Vector{Float32}() for x = 1:length(spiketimes)]
    @inbounds @fastmath for n in eachindex(neurons)
        ff = findfirst(x -> x > interval[1] + margin[1], spiketimes[n])
        ll = findlast(x -> x <= interval[end] + margin[2], spiketimes[n])
        if !isnothing(ff) && !isnothing(ll)
            append!(neurons[n], copy(spiketimes[n][ff:ll]))
        end
    end
    return neurons
end


"""
    spikes_in_intervals(spiketimes::Spiketimes, intervals::Vector{Vector{R}}; margin = [0, 0], floor = true)

Apply `spikes_in_interval` to each interval (in parallel with `tmap`) and return a
`Vector{Spiketimes}`. With `floor = true` the spikes of each interval are expressed relative to
the interval start (`interval_standard_spikes!`).
"""
function spikes_in_intervals(
    spiketimes::Spiketimes,
    intervals::Vector{Vector{R}};
    margin = [0, 0],
    floor = true,
) where {R<:Real}
    st = tmap(intervals) do interval
        spikes_in_interval(spiketimes, interval, margin)
    end
    (floor) && (interval_standard_spikes!(st, intervals; margin))
    return st
end

"""
    find_interval_indices(intervals::AbstractVector, interval::Vector)

Index range `x1:x2` of the vector of times `intervals`, with `x1` (`x2`) the first index with
value `>= interval[1]` (`>= interval[2]`).
"""
function find_interval_indices(
    intervals::AbstractVector{T},
    interval::Vector{T},
) where {T<:Real}
    x1 = findfirst(intervals .>= interval[1])
    x2 = findfirst(intervals .>= interval[2])
    return x1:x2
end


"""
    interval_standard_spikes(spiketimes::Spiketimes, interval::Vector; margin = [0, 0])

Express spike times relative to the interval start: subtract `interval[1] + margin[1]` from every
spike. Works on a copy; see `interval_standard_spikes!` for the in-place version.
"""
function interval_standard_spikes(spiketimes::Spiketimes, interval::Vector{R}; margin = [0, 0]) where {R<:Real}
    interval_standard_spikes!(deepcopy(spiketimes), interval; margin)
end

"""
    interval_standard_spikes!(spiketimes, interval::Vector; margin = [0, 0])
    interval_standard_spikes!(spiketimes::Vector{Spiketimes}, intervals::Vector{Vector}; margin = [0, 0])

In-place version of `interval_standard_spikes`: subtract `interval[1] + margin[1]` from every
spike time; for a `Vector{Spiketimes}`, element `i` is shifted by `intervals[i]`.
"""
function interval_standard_spikes!(
    spiketimes::Vector{Spiketimes},
    intervals::Vector{Vector{R}};
    margin = [0, 0]
) where {R<:Real}
    @assert length(spiketimes) == length(intervals)
    for i in eachindex(spiketimes)
        interval_standard_spikes!(spiketimes[i], intervals[i]; margin)
    end
end

function interval_standard_spikes!(spiketimes, interval::Vector{R}; margin = [0, 0]) where {R<:Real}
    for i in eachindex(spiketimes)
        spiketimes[i] .-= interval[1] + margin[1]
    end
    return spiketimes
end

# function gaussian_smooth(xs, signal; σ, n, padding=false)
#     Δx = step(xs)
#     gaussian_filter = gaussian(n, σ / Δx / 2) # Adjust σ based on bin size
#     gaussian_filter ./= sum(gaussian_filter)
#     @assert n % 2 == 1 "n must be odd for symmetric smoothing"
#     if padding
#         signal = vcat(fill(signal[1], n), signal, fill(signal[end], n))
#     end
#     n_start = (n - 1) ÷ 2 +1
#     n_end = n - n_start
#     new_pps = (n_start):length(signal)-n_end
#     smoothed_signal  = zeros(length(new_pps))
#     for i in new_pps
#         zz = 1+i-n_start:i+n_end
#         xx = i - n_start + 1
#         smoothed_signal[xx] = sum(signal[zz] .* gaussian_filter)
#     end

#     xvals = xs[n_start:length(xs)-n_start+1]
#     return smoothed_signal, xvals
# end
"""
    gaussian_smooth(xs, x, σ; skewed = :none) -> Vector

Apply a normalized Gaussian kernel to signal `x` sampled on grid `xs`.
Kernel half-width is `3σ` (truncated); boundary bins are renormalized by
accumulated kernel weight so edge values are not biased toward zero.

# Arguments
- `xs`: sample-position grid (used only for its step size `xs[2]-xs[1]`)
- `x`: signal to smooth, length `n`
- `sigma`: Gaussian standard deviation in the same units as `xs`
- `skewed`: `:none` (symmetric), `:left` (causal — uses only past samples),
  `:right` (anti-causal — uses only future samples)

# Returns
- smoothed signal, same length as `x`

"""
function gaussian_smooth(xs::RT, x::T, σ::R; skewed::Symbol=:none) where {T<:AbstractVector, R<:Real, RT<:AbstractVector}
    iszero(σ) && return copy(x)
    step_x = xs[2] - xs[1]
    half = ceil(Int, 3σ / step_x)
    offsets = if skewed === :left
        -half:0       # causal: only past samples
    elseif skewed === :right
        0:half        # anti-causal: only future samples
    else
        -half:half    # symmetric
    end
    kernel = exp.(.-(Float64.(offsets) .* step_x).^2 ./ (2σ^2))
    kernel ./= sum(kernel)
    n = length(x)
    out = similar(x)
    lo = first(offsets)
    s = 0.0
    w = 0.0
    @inbounds for i in 1:n
        s = 0.0
        w = 0.0
        @fastmath for (j, k) in enumerate(kernel)
            idx = i + lo + (j - 1)
            idx < 1 && continue
            idx > n && continue
            s += k * x[idx]
            w += k
        end
        out[i] = s / w
    end
    @assert length(out) == length(x)
    return out
end




@doc raw"""
    ISI_CV2(spiketime::Vector{Float32}; interval = nothing)
    ISI_CV2(spiketimes::Spiketimes; interval = nothing)
    ISI_CV2(pop::AbstractPopulation; interval = nothing)

Local coefficient of variation of the inter-spike intervals,
```math
CV_2 = \left\langle \frac{2\,|I_{i+1} - I_i|}{I_{i+1} + I_i} \right\rangle_i ,
```
averaged over consecutive ISI pairs of one train (0 if undefined, e.g. fewer than three
spikes). The `Spiketimes` method returns one value per neuron; the population method first
restricts the spikes to `interval`, the other methods ignore `interval`.

# References
Holt, G. R., Softky, W. R., Koch, C., & Douglas, R. J. (1996). Comparison of discharge
variability in vitro and in vivo in cat visual cortex neurons. Journal of Neurophysiology,
75(5), 1806–1814. https://doi.org/10.1152/jn.1996.75.5.1806
"""
function ISI_CV2(spiketime::Vector{Float32}; interval=nothing)
    ISI = diff(spiketime)
    CV2 = Float32[]
    for i in eachindex(ISI)
        i == 1 && continue
        x = 2(abs(ISI[i] - ISI[i-1]) / (ISI[i] + ISI[i-1]))
        push!(CV2, x)
    end
    _cv = mean(CV2)

    # _cv = sqrt(var(intervals)/mean(intervals)^2)
    return isnan(_cv) ? 0.0 : _cv
end



function ISI_CV2(x::Spiketimes; interval = nothing) 
    return ISI_CV2.(x, ; interval)
end

ISI_CV2(pop::T; interval = nothing) where {T<:AbstractPopulation} =
    spiketimes(pop; interval) |> ISI_CV2


export ISI_CV2


##

@doc raw"""
    ISI_CV(spiketime::Vector{Float32}; interval = nothing)
    ISI_CV(spiketimes::Spiketimes; interval = nothing)

Coefficient of variation of the inter-spike intervals, ``CV = \sigma_{ISI} / \mu_{ISI}``
(0 if undefined). The `Spiketimes` method returns one value per neuron; `interval` is ignored.

# Example
```julia
using SpikingNeuralNetworks
SNN.ISI_CV(Float32[0, 10, 20, 30])   # 0.0 (regular train)
```
"""
function ISI_CV(spiketime::Vector{Float32}; interval=nothing)
    ISI = diff(spiketime)
    cv = sqrt(var(ISI) / mean(ISI)^2)
    return isnan(cv) ? 0.0 : cv
end

function ISI_CV(spiketimes::Spiketimes; interval=nothing)
    return ISI_CV.(spiketimes; interval)    
end

export ISI_CV

"""
    FanoFactor(spiketime::Vector{Float32}; interval::AbstractRange)
    FanoFactor(spiketimes::Spiketimes; interval::AbstractRange)
    FanoFactor(pop::AbstractPopulation; interval)

Fano factor `var(counts) / mean(counts)` of the spike counts in the bins of `interval`
(see `bin_spiketimes`; the bin width is `step(interval)`), 0 if undefined. The `Spiketimes` and
population methods return one value per neuron.

`interval` is a required keyword (an `AbstractRange`). (Up to SNNModels 1.8.4 it defaulted to
`nothing`, and the population method built a tuple, both rejected by `bin_spiketimes`.)

# References
Softky, W. R., & Koch, C. (1993). The highly irregular firing of cortical cells is inconsistent
with temporal integration of random EPSPs. Journal of Neuroscience, 13(1), 334–350.
https://doi.org/10.1523/JNEUROSCI.13-01-00334.1993

# Example
```julia
using SpikingNeuralNetworks
@load_units
st = SNN.Spiketimes([Float32.(sort(rand(50) .* 1000))])
SNN.FanoFactor(st; interval = 0:100ms:1s)
```
"""
function FanoFactor(spiketime::Vector{Float32}; interval::AbstractRange)
    bins, r = bin_spiketimes(spiketime; interval) 
    ff  = var(bins) / mean(bins)
    isnan(ff) && (ff = 0.0)
    return ff
end

function FanoFactor(spiketimes::Spiketimes; interval::AbstractRange)
    return FanoFactor.(spiketimes; interval)
end

function FanoFactor(pop::T; interval::AbstractRange) where {T<:AbstractPopulation}
    st = spiketimes(pop)
    return FanoFactor.(st; interval)
end

export FanoFactor

"""
    st_order(spiketimes::Vector)
    st_order(spiketimes::Spiketimes, pop::Vector{Int}, intervals)
    st_order(spiketimes::Spiketimes, populations::Vector{Vector{Int}}, intervals, unique_pop = false)

Indices that sort the elements of `spiketimes` (`sort(eachindex(spiketimes), by = x -> spiketimes[x])`;
for a `Spiketimes`, trains are compared lexicographically, i.e. by first spike).

The methods with `pop`/`populations` sort the neurons of `pop` (each vector of
`populations`) by the time of their first spike inside `intervals` (a vector of `[start, end]`
pairs; neurons without such a spike come last) and return the sorted neuron indices. (Up to
SNNModels 1.8.4 they called the undefined `spike_statistics` and threw an `UndefVarError`.)
"""
function st_order(spiketimes::T) where {T<:Vector{}}
    ii = sort(eachindex(1:length(spiketimes)), by = x -> spiketimes[x])
    return ii
end

function st_order(spiketimes::Spiketimes, pop::Vector{Int}, intervals)
    first_spike = map(pop) do n
        t = Inf32
        for iv in intervals, s in spiketimes[n]
            iv[1] <= s <= iv[end] && (t = min(t, Float32(s)))
        end
        t
    end
    ii = sortperm(first_spike)
    return pop[ii]
end

function st_order(
    spiketimes::Spiketimes,
    populations::Vector{Vector{Int}},
    intervals::Vector{Vector{T}},
    unique_pop::Bool = false,
) where {T<:Real}
    return [st_order(spiketimes, population, intervals) for population in populations]
end

"""
    relative_time!(spiketimes::Spiketimes, start_time)

Subtract `start_time` from every spike time (the vectors are replaced in place) and return
`spiketimes`.
"""
function relative_time!(spiketimes::Spiketimes, start_time)
    neurons = 1:length(spiketimes)
    for n in neurons
        spiketimes[n] = spiketimes[n] .- start_time
    end
    return spiketimes
end



# Legacy rolling-window firing rate (disabled):
# function firing_rate(P, τ; dt = 0.1ms)
#     spikes = hcat(P.records[:fire]...)
#     time_span = round(Int, size(spikes, 2) * dt)
#     rates = zeros(P.N, time_span)
#     L = round(Int, time_span - τ) * 10
#     my_spikes = Matrix{Int}(spikes)
#     @fastmath @inbounds for s in axes(spikes, 1)
#         T = round(Int, τ / dt)
#         rates[s, round(Int, τ)+1:end] =
#             trolling_mean((@view my_spikes[s, :]), T)[1:10:L] ./ (dt / 1000)
#     end
#     return rates
# end

"""
    resample_spikes(X, Y)

If there are more than 200 000 points, return a random subsample (without replacement) of
200 000 pairs `(X[i], Y[i])` and warn; otherwise return `(X, Y)`. Used for raster plots.
"""
function resample_spikes(X, Y)
    if length(X) > 200_000
        s = ceil(Int, length(X) / 200_000)
        points = Vector{Int}(eachindex(X))
        points = sample(points, 200_000, replace = false)
        X = X[points]
        Y = Y[points]
        @warn "Subsampling raster plot, 1 out of $s spikes"
    end
    return X, Y
end


function rolling_mean(a, n::Int)
    @assert 1 <= n <= length(a)
    out = similar(a, length(a) - n + 1)
    out[1] = sum(a[1:n])
    for i in eachindex(out)[2:end]
        out[i] = out[i-1] - a[i-1] + a[i+n-1]
    end
    return out ./ n
end

function trolling_mean(a, n::Int)
    @assert 1 <= n <= length(a)
    nseg = Threads.nthreads()
    if nseg * n >= length(a)
        return rolling_mean(a, n)
    else
        out = similar(a, length(a) - n + 1)
        lseg = (length(out) - 1) ÷ nseg + 1
        segments = [(i * lseg + 1, min(length(out), (i + 1) * lseg)) for i = 0:(nseg-1)]
        for (start, stop) in segments
            out[start] = sum(a[start:(start+n-1)])
            for i = (start+1):stop
                out[i] = out[i-1] - a[i-1] + a[i+n-1]
            end
        end
        return out ./ n
    end
end


"""
    sample_spikes(N, rate::Vector, interval::AbstractRange; rate_factor = 1.0f0, dt = 0.125f0) -> Vector{Vector{Float32}}

Draw spike trains for `N` independent neurons from a time-varying rate (Bernoulli
approximation of an inhomogeneous Poisson process).

`rate[i]` is the rate in Hz (a plain number, multiplied by `Hz` internally) during the `i`-th
step of `interval`; each step of `interval` is divided in `step(interval) / dt` sub-steps, and in
every sub-step each neuron spikes with probability `rate[i] * Hz * dt * rate_factor`. Spike times
start at `first(interval) + dt`. `length(rate) == length(interval)` is required.

# Example
```julia
using SpikingNeuralNetworks
@load_units
st = SNN.sample_spikes(5, fill(10.0, 100), 0:10ms:990ms)   # 5 neurons at 10 Hz for 1 s
```
"""
function sample_spikes(
    N,
    rate::Vector,
    interval::R;
    rate_factor = 1.0f0,
    dt = 0.125f0,
) where {R<:AbstractRange}
    spiketimes = Vector{Float32}[[] for _ = 1:N]
    @assert length(rate) == length(interval)
    steps = step(interval) / dt
    t = dt + Float32(interval[1])
    for i in eachindex(interval)
        r = rate[i] * Hz
        for _ = 1:steps
            for n = 1:N
                if rand() < r * dt * rate_factor
                    push!(spiketimes[n], t)
                end
            end
            t = Float32(t + dt)
        end
    end
    spiketimes
end

"""
    infer_spikes(n_spikes::Vector{<:Integer}, interval::AbstractRange; dt = 0.125f0)

Place exactly `n_spikes[i]` spikes, at distinct random sub-steps of size `dt`, inside the `i`-th
bin of `interval` (`(interval[i], interval[i] + step(interval)]`). Returns a one-element
`Vector{Vector{Float32}}`.
"""
function infer_spikes(
    n_spikes::Vector{In},
    interval::R;
    dt = 0.125f0,
) where {R<:AbstractRange, In<:Integer}
    spiketimes = Vector{Float32}[[]]
    @assert length(n_spikes) == length(interval)
    step_int = step(interval)
    t = dt + Float32(interval[1])
    for i in eachindex(interval)
        t = Float32(interval[i])
        local_interval  = t .+(dt :dt:step_int) 
        r = n_spikes[i]
        if r > 0
            for spiketime in sample(local_interval, r; replace = false)
                push!(spiketimes[1], spiketime)
            end
        end
    end
    spiketimes
end

"""
    infer_spiketimes(rate::Matrix{<:Integer}, interval::AbstractRange; dt = 0.125f0, seed = nothing)

Apply `infer_spikes` to each row of the spike-count matrix `rate` (rows = neurons, columns = bins
of `interval`) and return one spike train per row. `seed` seeds the global RNG.
"""
function infer_spiketimes(
    rate::Matrix{In},
    interval::R;
    dt = 0.125f0,
    seed = nothing,
) where {R<:AbstractRange, In<:Integer}
    !isnothing(seed) && (Random.seed!(seed))
    inputs = Vector{Float32}[]
    for i = 1:size(rate, 1)
        for n in infer_spikes(rate[i, :], interval; dt = dt)
            push!(inputs, n)
        end
    end
    inputs
end

"""
    sample_inputs(N, rate::Matrix{<:Integer}, interval; dt = 0.125f0, rate_factor = 1.0f0, seed = nothing)
    sample_inputs(N, rate::Matrix{<:Real}, interval; dt = 0.125f0, rate_factor = 1.0f0, seed = nothing)
    sample_inputs(N, spikes::BitMatrix, interval; rate_factor, dt = 0.125f0, seed = nothing)

Build input spike trains from a matrix sampled on `interval` (columns = time bins):

- integer matrix: spike counts per bin, converted with `infer_spikes` (one train per row; `N`
  and `rate_factor` are ignored);
- real matrix: rates in Hz, converted with `sample_spikes(N, row, interval)` (`N` trains per row,
  concatenated);
- `BitMatrix`: spike indicators; each row gives the train `interval[findall(row)]`.

`seed` seeds the global RNG. Returns a `Vector{Vector{Float32}}`.
"""
function sample_inputs(
    N::Int, 
    rate::Matrix{In},
    interval::R;
    dt = 0.125f0,  
    rate_factor = 1.0f0,
    seed = nothing,
) where {R<:AbstractRange, In<:Integer}
    !isnothing(seed) && (Random.seed!(seed))
    inputs = Vector{Float32}[]
    for i = 1:size(rate, 1)
        for n in infer_spikes(rate[i, :], interval; dt = dt)
            push!(inputs, n)
        end
    end
    inputs
end


function sample_inputs(
    N,
    rate::Matrix{F},
    interval::R;
    dt = 0.125f0,
    rate_factor = 1.0f0,
    seed = nothing,
) where {R<:AbstractRange,F<:Real}
    !isnothing(seed) && (Random.seed!(seed))
    inputs = Vector{Float32}[]
    for i = 1:size(rate, 1)
        for n in sample_spikes(N, rate[i, :], interval; dt = dt, rate_factor = rate_factor)
            push!(inputs, n)
        end
    end
    inputs
end

function sample_inputs(
    N,
    spikes::BitMatrix,
    interval::R;
    rate_factor,
    dt = 0.125f0,
    seed = nothing,
) where {R<:AbstractRange}
    !isnothing(seed) && (Random.seed!(seed))
    inputs = Vector{Float32}[]
    @assert size(spikes, 2) == length(interval)
    for i = 1:size(spikes, 1)
        st_n = findall(spikes[i, :])
        push!(inputs, Float32.(interval[st_n]))
    end
    inputs
end



export spiketimes,
    merge_spiketimes,
    convolve,
    alpha_function,
    alpha_kernel,
    bin_spiketimes,
    compute_covariance_density,
    isi,
    ISI_CV2,
    firing_rate,
    average_firing_rate,
    spikes_in_interval,
    spikes_in_intervals,
    find_interval_indices,
    interval_standard_spikes,
    interval_standard_spikes!,
    relative_time!,
    st_order,
    sample_spikes,
    sample_inputs,
    infer_spiketimes,
    infer_spikes,
    resample_spikes,
    gaussian_smooth

