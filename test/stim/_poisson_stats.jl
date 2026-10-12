# Statistical helpers of the tests of the sparse Poisson samplers (the same statistics as test/stim/poisson_sparse.jl, kept in a module of
# their own so that the tests of the Poisson populations do not depend on the order in which the test files are included).
module PoissonStats
using Test, Statistics, Distributions
export _PMIN, _report, ks_two_sample, ks_discrete, chi2_geometric, chi2_two_sided, _geo_cdf, RunRecord, record_run, check_counts, check_independence
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
mutable struct RunRecord
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
RunRecord(N, nsteps, q; win = 100) = RunRecord(zeros(Int, N), Int[], zeros(Int, 5), zeros(Int32, nsteps), zeros(Int32, N, cld(nsteps, win)), win,
    min(ceil(Int, 5 / q), nsteps ÷ 2), nsteps)

@inline function _account!(r::RunRecord, last::Vector{Int}, i::Int, k::Int, n::Int)
    r.totals[i] += k
    r.khist[min(k, 4)+1] += 1
    (last[i] > 0 && n - last[i] <= r.Lmax && last[i] <= r.nsteps - r.Lmax) && push!(r.isis, n - last[i])
    last[i] = n
    r.stepsum[n] += k
    r.bins[i, (n-1)÷r.win+1] += k
end

# drive `step!` nsteps times and read the counts from `g` (scaled by μ) or the Bool vector `fire`
function record_run(step!, N, nsteps, q; g = nothing, fire = nothing, μ = 1.0f0)
    r = RunRecord(N, nsteps, q)
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

# ---------------------------------------------------------------------------- checks
# all the single-sample statistics of a run against `Poisson(λ)` counts (event probability q = 1 - exp(-λ))
function check_counts(r::RunRecord, λ, nsteps, label)
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
function check_independence(r::RunRecord, label)
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

end
