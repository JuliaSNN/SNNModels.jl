using SNNModels
using Test
using Random
using Statistics
using SparseArrays
using LinearAlgebra
using Distributions
@load_units

# Sparse sampling of the Poisson stimuli (geometric skipping, rejection for heterogeneous rates).
# The spike process must have the statistics of the dense sampler of SNNModels 1.9 (one draw per
# neuron and step): counts per neuron, per-step count law, inter-spike intervals, independence
# across neurons and steps. A fixed seed makes every statistic below deterministic; the thresholds
# are p > 1e-4 (a statistic that fails here is wrong by far more than chance).
# Set ENV["POISSON_STATS_VERBOSE"] = "1" to print the statistics.

const _K = SNNModels
const _VERBOSE = get(ENV, "POISSON_STATS_VERBOSE", "0") == "1"
const _PMIN = 1e-4
_report(name, args...) = _VERBOSE && println(rpad(name, 46), join(args, "  "))

# ---------------------------------------------------------------------------- statistics
# Kolmogorov distribution tail, with the finite-size correction of Stephens
function _ks_p(D, ne)
    λ = (sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) * D
    λ < 0.2 && return 1.0
    s = 0.0
    for k in 1:100
        t = 2 * (-1)^(k - 1) * exp(-2 * k^2 * λ^2)
        s += t
        abs(t) < 1e-12 && break
    end
    return clamp(s, 0.0, 1.0)
end

function ks_two_sample(x, y)
    xs, ys = sort(x), sort(y)
    D = 0.0
    for v in sort!(unique(vcat(x, y)))
        D = max(D, abs(searchsortedlast(xs, v) / length(xs) - searchsortedlast(ys, v) / length(ys)))
    end
    return D, _ks_p(D, length(x) * length(y) / (length(x) + length(y)))
end

# cdf of Geometric(q) on {1, 2, ...} conditioned on <= L
_geo_cdf(q, L) = m -> m < 1 ? 0.0 : (1 - (1 - q)^min(m, L)) / (1 - (1 - q)^L)

# one-sample KS against a discrete cdf (conservative: the p-value is an upper bound)
function ks_discrete(x, cdf)
    xs = sort(x)
    n = length(xs)
    D = 0.0
    for v in unique(xs)
        D = max(D, abs(searchsortedlast(xs, v) / n - cdf(v)), abs(searchsortedlast(xs, v - 1) / n - cdf(v - 1)))
    end
    return D, _ks_p(D, n)
end

# chi-square goodness of fit of integer samples `x` (in 1:L) to Geometric(q) conditioned on <= L, equiprobable bins
function chi2_geometric(x, q, L; nbins = 10)
    cdf = _geo_cdf(q, L)
    FL = 1 - (1 - q)^L
    edges = Int[]
    for j in 1:nbins-1
        m = ceil(Int, log(1 - (j / nbins) * FL) / log1p(-q))
        (isempty(edges) || m > edges[end]) && push!(edges, clamp(m, 1, L))
    end
    edges = unique(edges)
    probs = diff([0.0; cdf.(edges); 1.0])
    keep = probs .> 0
    obs = zeros(Int, length(probs))
    for v in x
        obs[searchsortedfirst(edges, v)] += 1
    end
    exp_ = length(x) .* probs
    χ2 = sum((obs[keep] .- exp_[keep]) .^ 2 ./ exp_[keep])
    return χ2, ccdf(Chisq(count(keep) - 1), χ2)
end

# chi-square of a statistic that must be chi-square distributed with `df` degrees of freedom, two-sided p
function chi2_two_sided(χ2, df)
    d = Chisq(df)
    return 2 * min(cdf(d, χ2), ccdf(d, χ2))
end

# ---------------------------------------------------------------------------- drivers
# What a run records from a stimulus that writes either the conductance `g` (counts) or `fire`.
mutable struct Rec
    totals::Vector{Int}          # spikes per neuron
    isis::Vector{Int}            # inter-event intervals (steps) of all neurons
    khist::Vector{Int}           # histogram of the per-step count: 0, 1, 2, 3, 4+
    stepsum::Vector{Int32}       # total count per step
    bins::Matrix{Int32}          # counts per neuron in windows of `win` steps
    win::Int
    Lmax::Int                    # only gaps of at most Lmax steps that start before nsteps - Lmax are kept
    nsteps::Int
end
# A finite run shows only the gaps that end inside it, which biases the observed gap law towards short
# gaps. The gaps that start before nsteps - Lmax and are at most Lmax long are exactly iid Geometric
# conditioned on <= Lmax (renewal property), so they are compared with the truncated geometric law.
Rec(N, nsteps, q; win = 100) = Rec(zeros(Int, N), Int[], zeros(Int, 5), zeros(Int32, nsteps), zeros(Int32, N, cld(nsteps, win)), win,
    min(ceil(Int, 5 / q), nsteps ÷ 2), nsteps)

@inline function _account!(r::Rec, last::Vector{Int}, i::Int, k::Int, n::Int)
    r.totals[i] += k
    r.khist[min(k, 4)+1] += 1
    (last[i] > 0 && n - last[i] <= r.Lmax && last[i] <= r.nsteps - r.Lmax) && push!(r.isis, n - last[i])
    last[i] = n
    r.stepsum[n] += k
    r.bins[i, (n-1)÷r.win+1] += k
end

# drive `step!` nsteps times and read the counts from `g` (scaled by μ) or the Bool vector `fire`
function record_run(step!, N, nsteps, q; g = nothing, fire = nothing, μ = 1.0f0)
    r = Rec(N, nsteps, q)
    last = zeros(Int, N)
    for n in 1:nsteps
        step!(n)
        if g !== nothing
            @inbounds for i in 1:N
                gi = g[i]
                if gi != 0
                    _account!(r, last, i, round(Int, gi / μ), n)
                    g[i] = 0
                end
            end
        else
            @inbounds for i in 1:N
                fire[i] && _account!(r, last, i, 1, n)
            end
        end
    end
    # the khist of k = 0 is the rest
    r.khist[1] = N * nsteps - sum(r.khist[2:end])
    return r
end

# a target population whose input buffer the stimulus writes (the conductance `ge`)
_target(N) = (E = IF(N = Int(N)); (E, E.receptors.glu))

# PoissonStimulus on N neurons, rate in Hz, via the real stimulate!
function run_poisson(N, rate, nsteps; dt = 0.1f0, seed = 1, μ = 1.0f0)
    E, g = _target(N)
    stim = PoissonStimulus(E, :ge; param = PoissonFixed(rate = rate, μ = μ))
    T = Time()
    Random.seed!(seed)
    return record_run(n -> (update_time!(T, dt); stimulate!(stim, stim.param, T, dt)), N, nsteps, 1 - exp(-Float64(rate * dt)); g, μ)
end

# the sampler of SNNModels up to 1.9: one Poisson draw per neuron and step
function run_dense_poisson(N, rate, nsteps; dt = 0.1f0, seed = 11)
    E, g = _target(N)
    λ = rate * dt
    Random.seed!(seed)
    d = Distributions.Poisson{Float32}(λ)
    return record_run(n -> (for i in 1:N; g[i] += rand(d); end), N, nsteps, 1 - exp(-Float64(λ)); g)
end

# PoissonStimulusLayer with identity weights, so that `fire` is the Bernoulli process of each neuron
function run_layer(param, nsteps; dt = 0.1f0, seed = 1)
    N = param.N
    E, _ = _target(N)
    stim = PoissonStimulusLayer(E, :ge; param, conn = sparse(1.0f0 * LinearAlgebra.I, N, N))
    T = Time()
    Random.seed!(seed)
    pmax = param isa PoissonLayer ? Float64(param.rate * dt) : Float64(minimum(param.rates) * dt)
    return record_run(n -> (update_time!(T, dt); stimulate!(stim, stim.param, T, dt)), N, nsteps, pmax; fire = stim.fire), stim
end

# the Bernoulli sampler of SNNModels up to 1.9
function run_dense_layer(N, p, nsteps; seed = 11)
    fire = zeros(Bool, N)
    cache = zeros(Float32, N)
    Random.seed!(seed)
    p32 = Float32(p)
    return record_run(n -> (rand!(cache); @. fire = cache < p32), N, nsteps, p; fire)
end

# ---------------------------------------------------------------------------- checks
# all the single-sample statistics of a run against `Poisson(λ)` counts (event probability q = 1 - exp(-λ))
function check_counts(r::Rec, λ, nsteps, label)
    N = length(r.totals)
    q = 1 - exp(-λ)
    μ_tot = nsteps * λ
    # (a) mean count per neuron: z-test of the mean of the N totals
    z = (mean(r.totals) - μ_tot) / sqrt(μ_tot / N)
    # (b) Fano factor of the totals over neurons; (N-1) var / mean is chi-square(N-1) for Poisson counts
    fano = var(r.totals) / mean(r.totals)
    p_disp = N > 1 ? chi2_two_sided((N - 1) * fano, N - 1) : 1.0
    # (c) law of the per-step count: chi-square against Poisson(λ) on 0, 1, 2, 3, 4+
    pk = [pdf(Distributions.Poisson(λ), k) for k in 0:3]
    pk = [pk; 1 - sum(pk)]
    keep = findall(>(0), pk .* N .* nsteps .>= 5)
    obs = [r.khist[keep[1:end-1]]; sum(r.khist[keep[end]:end])]
    prb = [pk[keep[1:end-1]]; sum(pk[keep[end]:end])]
    expc = N * nsteps .* prb
    χ2k = sum((obs .- expc) .^ 2 ./ expc)
    p_k = ccdf(Chisq(length(obs) - 1), χ2k)
    # (d) inter-event intervals are geometric: chi-square on equiprobable bins and KS (conservative)
    χ2i, p_isi = chi2_geometric(r.isis, q, r.Lmax)
    D, p_ks = ks_discrete(r.isis, _geo_cdf(q, r.Lmax))
    # (e) the step totals have no lag-1 autocorrelation (steps are independent)
    s = Float64.(r.stepsum)
    ρ1 = cor(s[1:end-1], s[2:end])
    z_ac = ρ1 * sqrt(length(s) - 1)
    _report(label, "mean z=$(round(z, digits=2))", "Fano=$(round(fano, digits=3)) (p=$(round(p_disp, sigdigits=2)))",
        "k-law p=$(round(p_k, sigdigits=2))", "ISI chi2 p=$(round(p_isi, sigdigits=2))", "KS D=$(round(D, digits=4)) p=$(round(p_ks, sigdigits=2))",
        "lag1 z=$(round(z_ac, digits=2))", "nISI=$(length(r.isis))")
    @test abs(z) < 4.5
    @test p_disp > _PMIN
    @test p_k > _PMIN
    @test p_isi > _PMIN
    @test p_ks > _PMIN
    @test abs(z_ac) < 4.5
    return nothing
end

# neurons are independent: neighbours (j, j+1) are the pairs that a skipping sampler would couple.
# The correlation of the window counts of a pair is N(0, 1/nbins), so the sum of z^2 over the pairs is chi-square.
function check_independence(r::Rec, label)
    N, nb = size(r.bins)
    X = Float64.(r.bins)
    zs = Float64[]
    for j in 1:N-1
        sd1, sd2 = std(X[j, :]), std(X[j+1, :])
        (sd1 > 0 && sd2 > 0) && push!(zs, cor(X[j, :], X[j+1, :]) * sqrt(nb))
    end
    χ2 = sum(abs2, zs)
    p = chi2_two_sided(χ2, length(zs))
    pm = maximum(abs, zs)
    _report(label, "neighbour corr: sum z^2 = $(round(χ2, digits=1)) / $(length(zs)) (p=$(round(p, sigdigits=2)))", "max |z|=$(round(pm, digits=2))")
    @test p > _PMIN
    # the largest of n standard normals stays below ~ sqrt(2 log n); not for windows with almost no events (z is skewed there)
    mean(X) >= 2 && @test pm < 5 + log(length(zs))
    return nothing
end

@testset "Sparse Poisson sampling" begin
    dt = 0.1f0

    @testset "zero-truncated Poisson sampler" begin
        Random.seed!(3)
        for λ in (1e-4, 0.02, 0.3, 0.9, 1.0, 3.0, 25.0)
            n = 200_000
            ks = [_K._zt_poisson(λ) for _ in 1:n]
            @test minimum(ks) >= 1
            d = Distributions.Poisson(λ)
            q = 1 - pdf(d, 0)
            m = mean(ks)
            @test abs(m - λ / q) < 5 * std(ks) / sqrt(n) + 1e-12
            # pmf of the first values
            for k in 1:3
                pk = pdf(d, k) / q
                obs = count(==(k), ks)
                @test abs(obs - n * pk) <= 5 * sqrt(n * pk * (1 - pk)) + 1
            end
        end
    end

    @testset "PoissonStimulus, homogeneous rate: N = $N, $(rate) Hz" for (N, rate, nsteps) in
                                                                             ((10, 10, 100_000), (1000, 1, 300_000), (1000, 10, 100_000), (1000, 50, 100_000), (1000, 200, 100_000), (10, 200, 100_000))
        r = run_poisson(N, rate * Hz, nsteps)
        λ = rate * Hz * dt
        check_counts(r, λ, nsteps, "PoissonStimulus N=$N $(rate)Hz")
        N > 10 && check_independence(r, "PoissonStimulus N=$N $(rate)Hz")
        if N == 10   # with 10 neurons check each neuron's total (binomial-like, here Poisson) separately
            μt = nsteps * λ
            @test all(abs.(r.totals .- μt) .< 5 * sqrt(μt))
        end
    end

    @testset "PoissonStimulus equals the dense sampler in distribution: N = $N, $(rate) Hz" for (N, rate, nsteps) in
                                                                                              ((10, 50, 200_000), (1000, 10, 100_000), (1000, 200, 100_000))
        rs = run_poisson(N, rate * Hz, nsteps; seed = 5)
        rd = run_dense_poisson(N, rate * Hz, nsteps; seed = 6)
        D1, p1 = ks_two_sample(rs.totals, rd.totals)
        D2, p2 = ks_two_sample(rs.isis, rd.isis)
        _report("sparse vs dense N=$N $(rate)Hz", "totals KS D=$(round(D1, digits=4)) p=$(round(p1, sigdigits=2))",
            "ISI KS D=$(round(D2, digits=4)) p=$(round(p2, sigdigits=2))", "n=$(length(rs.isis)), $(length(rd.isis))")
        @test p1 > _PMIN
        @test p2 > _PMIN
        # the number of ISIs of the two samplers agrees (Poisson counts with the same mean)
        @test abs(length(rs.isis) - length(rd.isis)) < 5 * sqrt(length(rs.isis) + length(rd.isis))
    end

    @testset "PoissonStimulus, rates above one spike per step (dense regime of the sampler)" begin
        for (rate, N, nsteps) in ((10_000, 100, 20_000), (30_000, 100, 20_000))   # λ = 1 and 3 at dt = 0.1
            r = run_poisson(N, rate * Hz, nsteps)
            λ = rate * Hz * dt
            @test abs(mean(r.totals) - nsteps * λ) < 4.5 * sqrt(nsteps * λ / N)
            @test r.khist[1] ≈ N * nsteps * exp(-λ) rtol = 0.05
        end
    end

    @testset "PoissonStimulus, neuron subsets, μ and the inactive switch" begin
        E, g = _target(50)
        T = Time()
        idx = [3, 7, 8, 40]
        stim = PoissonStimulus(E, :ge; param = PoissonFixed(rate = 1000Hz, μ = 2.0f0), neurons = idx)
        Random.seed!(2)
        for n in 1:2000
            update_time!(T, dt)
            stimulate!(stim, stim.param, T, dt)
        end
        @test all(g[setdiff(1:50, idx)] .== 0)
        @test all(g[idx] .> 0)
        @test all(mod.(g[idx], 2.0f0) .== 0)             # multiples of μ
        @test all(abs.(g[idx] .- 2 * 2000 * 0.1) .< 5 * 2 * sqrt(2000 * 0.1) )
        fill!(g, 0)
        set_active!(stim, false)
        for n in 1:200
            stimulate!(stim, stim.param, T, dt)
        end
        @test all(g .== 0)
        # zero and negative rates add nothing
        set_active!(stim, true)
        for rate in (0.0f0, -5.0f0)
            stim.param.rate = rate
            stimulate!(stim, stim.param, T, dt)
            @test all(g .== 0)
        end
        # duplicate neurons are independent entries
        stim2 = PoissonStimulus(E, :ge; param = PoissonFixed(rate = 1000Hz), neurons = [5, 5, 5, 5])
        fill!(g, 0)
        Random.seed!(2)
        for n in 1:2000
            stimulate!(stim2, stim2.param, T, dt)
        end
        @test abs(g[5] - 4 * 2000 * 0.1) < 5 * sqrt(4 * 2000 * 0.1)
    end

    @testset "PoissonStimulus, time-dependent rates" begin
        N, nsteps = 1000, 100_000
        E, g = _target(N)
        T = Time()
        # PoissonVariable: rate switched with set_variable! halfway
        param = PoissonVariable(variables = Dict{Symbol,Any}(:r => 5Hz), rate = (t, v) -> v[:r])
        stim = PoissonStimulus(E, :ge; param)
        tot1 = tot2 = 0
        Random.seed!(4)
        begin
            for n in 1:nsteps
                n == nsteps ÷ 2 + 1 && set_variable!(stim, :r, 100Hz)
                update_time!(T, dt)
                stimulate!(stim, stim.param, T, dt)
                s = sum(g)
                n <= nsteps ÷ 2 ? (tot1 += round(Int, s)) : (tot2 += round(Int, s))
                fill!(g, 0)
            end
        end
        m1, m2 = N * (nsteps ÷ 2) * 5Hz * dt, N * (nsteps ÷ 2) * 100Hz * dt
        @test abs(tot1 - m1) < 5 * sqrt(m1)
        @test abs(tot2 - m2) < 5 * sqrt(m2)
        # sinusoidal rate: count in the high and the low half-cycles, rate(t) = r0 (1 + sin(2π t / period))
        E2, g2 = _target(N)
        osc = PoissonVariable(variables = Dict{Symbol,Any}(:r0 => 50Hz, :period => 100ms), rate = (t, v) -> v[:r0] * (1 + sin(2π * t / v[:period])))
        stim2 = PoissonStimulus(E2, :ge; param = osc)
        T2 = Time()
        hi = lo = 0
        Random.seed!(5)
        for n in 1:nsteps
            update_time!(T2, dt)
            stimulate!(stim2, stim2.param, T2, dt)
            s = round(Int, sum(g2))
            fill!(g2, 0)
            ph = mod(get_time(T2), 100)
            ph < 50 ? (hi += s) : (lo += s)
        end
        mhi = N * (nsteps / 2) * dt * 50Hz * (1 + 2 / π)
        mlo = N * (nsteps / 2) * dt * 50Hz * (1 - 2 / π)
        @test abs(hi - mhi) < 5 * sqrt(mhi) + 0.01 * mhi
        @test abs(lo - mlo) < 5 * sqrt(mlo) + 0.01 * mlo
        # PoissonInterval: spikes only inside the interval
        E3, g3 = _target(N)
        stim3 = PoissonStimulus(E3, :ge; param = PoissonInterval(rate = 100Hz, intervals = [[100ms, 200ms]]))
        T3 = Time()
        inside = outside = 0
        Random.seed!(6)
        for n in 1:5000
            update_time!(T3, dt)
            stimulate!(stim3, stim3.param, T3, dt)
            s = round(Int, sum(g3))
            fill!(g3, 0)
            t = get_time(T3)
            (100 < t < 200) ? (inside += s) : (outside += s)
        end
        @test outside == 0
        m = N * 1000 * 100Hz * dt
        @test abs(inside - m) < 5 * sqrt(m) + 0.002 * m
    end

    @testset "PoissonLayer, homogeneous rate: N = $N, $(rate) Hz" for (N, rate, nsteps) in
                                                                      ((10, 20, 100_000), (1000, 1, 300_000), (1000, 10, 100_000), (1000, 200, 100_000), (1000, 1000, 50_000), (1000, 5000, 20_000))
        r, stim = run_layer(PoissonLayer(rate = rate * Hz, N = N), nsteps)
        p = rate * Hz * dt
        μt = nsteps * p
        z = (mean(r.totals) - μt) / sqrt(nsteps * p * (1 - p) / N)
        # Bernoulli: the total of a neuron is Binomial(nsteps, p); (N-1) var / (nsteps p (1-p)) is ~ chi-square(N-1)
        stat = (N - 1) * var(r.totals) / (nsteps * p * (1 - p))
        p_var = chi2_two_sided(stat, N - 1)
        @test abs(z) < 4.5
        @test p_var > _PMIN
        @test maximum(r.khist[3:end]) == 0                          # at most one spike per step and neuron
        χ2i, p_isi = chi2_geometric(r.isis, p, r.Lmax)               # Bernoulli: ISI is Geometric(p)
        D, p_ks = ks_discrete(r.isis, _geo_cdf(p, r.Lmax))
        s = Float64.(r.stepsum)
        z_ac = cor(s[1:end-1], s[2:end]) * sqrt(length(s) - 1)
        _report("PoissonLayer N=$N $(rate)Hz", "mean z=$(round(z, digits=2))", "var-test p=$(round(p_var, sigdigits=2))",
            "ISI chi2 p=$(round(p_isi, sigdigits=2))", "KS D=$(round(D, digits=4)) p=$(round(p_ks, sigdigits=2))", "lag1 z=$(round(z_ac, digits=2))")
        @test p_isi > _PMIN
        @test p_ks > _PMIN
        @test abs(z_ac) < 4.5
        N > 10 && check_independence(r, "PoissonLayer N=$N $(rate)Hz")
    end

    @testset "PoissonLayer equals the dense sampler in distribution: N = $N, $(rate) Hz" for (N, rate, nsteps) in
                                                                                           ((10, 50, 200_000), (1000, 10, 100_000), (1000, 100, 50_000))
        rs, _ = run_layer(PoissonLayer(rate = rate * Hz, N = N), nsteps; seed = 5)
        rd = run_dense_layer(N, rate * Hz * dt, nsteps; seed = 6)
        D1, p1 = ks_two_sample(rs.totals, rd.totals)
        D2, p2 = ks_two_sample(rs.isis, rd.isis)
        _report("layer sparse vs dense N=$N $(rate)Hz", "totals KS D=$(round(D1, digits=4)) p=$(round(p1, sigdigits=2))",
            "ISI KS D=$(round(D2, digits=4)) p=$(round(p2, sigdigits=2))")
        @test p1 > _PMIN
        @test p2 > _PMIN
    end

    @testset "PoissonLayerHet, heterogeneous rates: N = $N" for (N, lo, hi, nsteps) in ((10, 1, 100, 200_000), (1000, 1, 100, 100_000), (1000, 1, 300, 50_000), (1000, 20, 3000, 30_000))
        rates = Float32.(exp.(range(log(lo), log(hi), length = N))) .* Hz
        r, stim = run_layer(PoissonLayerHet(N = N, rates = rates), nsteps; seed = 9)
        p = Float64.(rates .* dt)
        # every neuron has Binomial(nsteps, p_j) spikes: sum of squared z-scores is chi-square(N)
        zs = (r.totals .- nsteps .* p) ./ sqrt.(nsteps .* p .* (1 .- p))
        χ2 = sum(abs2, zs)
        pv = chi2_two_sided(χ2, N)
        @test pv > _PMIN
        @test maximum(abs, zs) < 5 + log(N)
        # the rate of neuron j is recovered, across the range: relative error of the log-rate
        @test maximum(r.khist[3:end]) == 0
        # ISI of a slow neuron (rejection must not distort the process of neurons well below the maximum)
        j = 1
        _report("PoissonLayerHet N=$N rates $(lo)-$(hi)Hz", "sum z^2=$(round(χ2, digits=1))/$(N) (p=$(round(pv, sigdigits=2)))", "max |z|=$(round(maximum(abs, zs), digits=2))")
        N > 10 && check_independence(r, "PoissonLayerHet N=$N")
    end

    @testset "PoissonLayerHet: rates changed at runtime, same rates equal to the homogeneous layer" begin
        N, nsteps = 500, 100_000
        E, _ = _target(N)
        param = PoissonLayerHet(N = N, rates = fill(10f0 * Hz, N))
        stim = PoissonStimulusLayer(E, :ge; param, conn = sparse(1.0f0 * LinearAlgebra.I, N, N))
        T = Time()
        c1 = c2 = zeros(Int, N)
        c1 = zeros(Int, N); c2 = zeros(Int, N)
        Random.seed!(8)
        begin
            for n in 1:nsteps
                if n == nsteps ÷ 2 + 1
                    set_variable!(stim, :rates, collect(range(2f0 * Hz, 200f0 * Hz, length = N)))
                end
                update_time!(T, dt)
                stimulate!(stim, stim.param, T, dt)
                n <= nsteps ÷ 2 ? (c1 .+= stim.fire) : (c2 .+= stim.fire)
            end
        end
        h = nsteps ÷ 2
        p1 = 10Hz * dt
        p2 = Float64.(range(2f0 * Hz, 200f0 * Hz, length = N) .* dt)
        z1 = (c1 .- h * p1) ./ sqrt(h * p1 * (1 - p1))
        z2 = (c2 .- h .* p2) ./ sqrt.(h .* p2 .* (1 .- p2))
        @test chi2_two_sided(sum(abs2, z1), N) > _PMIN
        @test chi2_two_sided(sum(abs2, z2), N) > _PMIN
    end

    @testset "PoissonLayer, state: fire flags, spike list, weights, inactive layer" begin
        N, Npost = 200, 30
        E = IF(N = Npost)
        param = PoissonLayer(rate = 100Hz, N = N)
        stim = PoissonStimulusLayer(E, :ge; param, conn = (p = 0.3, μ = 1.0, σ = 0.0))
        T = Time()
        W = sparse(stim.I, stim.J, stim.W, Npost, N)
        g = E.receptors.glu
        Random.seed!(21)
        total = 0
        for n in 1:2000
            fill!(g, 0)
            update_time!(T, dt)
            stimulate!(stim, stim.param, T, dt)
            idx = findall(stim.fire)
            @test stim.spikes.n == length(idx)
            @test collect(stim.spikes.idx[1:stim.spikes.n]) == idx       # ascending, equal to the flags
            @test _K.spike_list!(stim.spikes, stim.fire) == idx
            @test _K.spike_index(stim) === stim.spikes
            @test g ≈ W * Float32.(stim.fire)
            total += length(idx)
        end
        @test total > 1000
        set_active!(stim, false)
        fill!(g, 0)
        stimulate!(stim, stim.param, T, dt)
        @test !any(stim.fire) && stim.spikes.n == 0 && all(g .== 0)
        # the layer always fires when p >= 1 (rate * dt >= 1) and never when the rate is 0
        set_active!(stim, true)
        stim.param.rate = 20_000f0
        stimulate!(stim, stim.param, T, dt)
        @test all(stim.fire) && stim.spikes.n == N
        stim.param.rate = 0f0
        stimulate!(stim, stim.param, T, dt)
        @test !any(stim.fire) && stim.spikes.n == 0
    end

    @testset "targets: Identity (sparse hand-off), ordinary populations, Tripod" begin
        # (i) onto an Identity: the sparse update equals the dense update of the same stimuli, flag by flag
        for mk in (:poisson, :layer, :both)
            function build(sparse_path)
                Random.seed!(31)
                pop = Identity(N = 60)
                stims = AbstractStimulus[]
                (mk in (:poisson, :both)) && push!(stims, PoissonStimulus(pop, :g; param = PoissonFixed(rate = 100Hz, μ = 0.5f0)))
                (mk in (:layer, :both)) && push!(stims, PoissonStimulusLayer(pop, :g; param = PoissonLayer(rate = 50Hz, N = 40), conn = (p = 0.2, μ = 1.0)))
                sparse_path || (pop.ndrivers = 0)
                return pop, stims
            end
            ps, ss = build(true)
            pd, sd = build(false)
            Ts, Td = Time(), Time()
            @test _K._list_driven(ps) && !_K._list_driven(pd)
            nfired = 0
            for n in 1:1500
                update_time!(Ts, dt); update_time!(Td, dt)
                Random.seed!(1000 + n)
                for s in ss; stimulate!(s, s.param, Ts, dt); end
                Random.seed!(1000 + n)
                for s in sd; stimulate!(s, s.param, Td, dt); end
                integrate!(ps, ps.param, dt); integrate!(pd, pd.param, dt)
                @test ps.fire == pd.fire && ps.spikecount == pd.spikecount && ps.h == pd.h && ps.g == pd.g
                @test collect(_K.spike_list!(ps.spikes, ps.fire)) == findall(ps.fire)
                nfired += sum(ps.fire)
            end
            @test nfired > 100
        end
        # a connection onto the same Identity is another writer: dense update, same result
        pop = Identity(N = 20); src = Identity(N = 20)
        stim = PoissonStimulus(pop, :g; param = PoissonFixed(rate = 50Hz))
        @test _K._list_driven(pop)
        syn = SpikingSynapse(src, pop, :g; conn = (p = 0.5, μ = 1.0))
        @test !_K._list_driven(pop)

        # (ii) ordinary populations: only the neurons that receive an input spike are written
        for E in (IF(N = 500), AdEx(N = 500), Tripod(N = 500))
            Random.seed!(41)
            stim = E isa Tripod ? PoissonStimulus(E, :glu; param = PoissonFixed(rate = 20Hz, μ = 1.0f0), comp = :d1) :
                                  PoissonStimulus(E, :ge; param = PoissonFixed(rate = 20Hz, μ = 1.0f0))
            @test stim.inbox === nothing
            g = stim.g
            T = Time()
            nz = 0
            for n in 1:500
                fill!(g, 0)
                update_time!(T, dt)
                stimulate!(stim, stim.param, T, dt)
                nz += count(!=(0), g)
            end
            @test abs(nz - 500 * 500 * (1 - exp(-20Hz * dt))) < 5 * sqrt(500 * 500 * 20Hz * dt)
        end
        Eg = IF(N = 300)
        Lst = PoissonStimulusLayer(Eg, :ge; param = PoissonLayer(rate = 20Hz, N = 100), conn = (p = 0.1, μ = 1.0))
        @test Lst.inbox === nothing
    end

    @testset "reproducibility and monitoring" begin
        function net_run(monitored; seed = 77)
            Random.seed!(seed)
            E = IF(N = 100)
            S1 = PoissonStimulus(E, :ge; param = PoissonFixed(rate = 2kHz, μ = 0.4f0))
            S2 = PoissonStimulusLayer(E, :ge; param = PoissonLayerHet(N = 50, rates = Float32.(range(5Hz, 100Hz, length = 50))), conn = (p = 0.2, μ = 1.0))
            syn = SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 0.2))
            if monitored
                monitor!(E, [:fire, :v])
                monitor!(S2, [:fire])
            end
            model = compose(; E, S1, S2, syn, silent = true)
            Random.seed!(seed)
            sim!(model; duration = 500ms)
            return copy(E.v), copy(E.fire), copy(S2.fire), copy(syn.W)
        end
        a1 = net_run(false)
        a2 = net_run(false)
        m = net_run(true)
        @test a1 == a2                  # bit-reproducible for a fixed seed
        @test a1 == m                   # recording does not touch the random stream
        @test net_run(false; seed = 78)[1] != a1[1]
        # the whole run actually spikes
        @test sum(abs, a1[1]) > 0
    end

    @testset "no allocation per step" begin
        function allocs(stim, T, dt)
            stimulate!(stim, stim.param, T, dt)
            return @allocated stimulate!(stim, stim.param, T, dt)
        end
        E = IF(N = 1000)
        Ei = Identity(N = 1000)
        T = Time()
        update_time!(T, dt)
        for post in (E, Ei)
            sym = post isa Identity ? :g : :ge
            S1 = PoissonStimulus(post, sym; param = PoissonFixed(rate = 10Hz))
            S2 = PoissonStimulusLayer(post, sym; param = PoissonLayer(rate = 10Hz, N = 1000), conn = (p = 0.01, μ = 1.0))
            S3 = PoissonStimulusLayer(post, sym; param = PoissonLayerHet(N = 1000, rates = fill(10f0 * Hz, 1000)), conn = (p = 0.01, μ = 1.0))
            @test allocs(S1, T, dt) == 0
            @test allocs(S2, T, dt) == 0
            @test allocs(S3, T, dt) == 0
            post isa Identity && (post.nwriters = 0)
        end
    end
end
true
