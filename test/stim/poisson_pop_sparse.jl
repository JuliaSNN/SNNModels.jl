using SNNModels
using Test
using Random
using Statistics
using SparseArrays
using Distributions
@load_units
isdefined(@__MODULE__, :PoissonStats) || include(joinpath(@__DIR__, "_poisson_stats.jl"))
const PS = PoissonStats      # qualified: other test files define the same helper names in Main

# Sparse sampling of the spiking Poisson populations `Poisson` (homogeneous and heterogeneous rates), `InhomogeneousPoisson` and
# `VariablePoisson` (geometric skipping, rejection for per-neuron probabilities): the same statistics as the dense sampler of
# SNNModels 1.9 (one uniform draw per neuron and step) - counts per neuron, inter-spike intervals, independence across neurons
# and steps, and the calibration of time-varying probabilities. A fixed seed makes the statistics deterministic; the thresholds are
# p > 1e-4. Set ENV["POISSON_STATS_VERBOSE"] = "1" to print them.

const _KP = SNNModels
const _DTP = 0.1f0

# drive `integrate!` of a population nsteps times and record `fire`
function run_pop(P, nsteps, q; seed = 1, dt = _DTP)
    N = Int(P.N)
    Random.seed!(seed)
    return PS.record_run(n -> SNNModels.integrate!(P, P.param, dt), N, nsteps, q; fire = P.fire)
end

# the dense Bernoulli sampler of SNNModels 1.9
function run_dense(N, p, nsteps; seed = 11)
    fire = zeros(Bool, N)
    cache = zeros(Float32, N)
    Random.seed!(seed)
    p32 = Float32(p)
    return PS.record_run(n -> (rand!(cache); @. fire = cache < p32), N, nsteps, p; fire)
end

# Bernoulli statistics of a run: per-neuron totals Binomial(nsteps, p), ISI geometric, no lag-1 correlation of the step counts
function check_bernoulli(r::PS.RunRecord, p, nsteps, label)
    N = length(r.totals)
    μt = nsteps * p
    z = (mean(r.totals) - μt) / sqrt(nsteps * p * (1 - p) / N)
    stat = (N - 1) * var(r.totals) / (nsteps * p * (1 - p))
    p_var = N > 1 ? PS.chi2_two_sided(stat, N - 1) : 1.0
    @test abs(z) < 4.5
    @test p_var > PS._PMIN
    @test maximum(r.khist[3:end]) == 0                      # at most one spike per step and neuron
    χ2i, p_isi = PS.chi2_geometric(r.isis, p, r.Lmax)
    D, p_ks = PS.ks_discrete(r.isis, PS._geo_cdf(p, r.Lmax))
    s = Float64.(r.stepsum)
    z_ac = cor(s[1:end-1], s[2:end]) * sqrt(length(s) - 1)
    PS._report(label, "mean z=$(round(z, digits=2))", "var-test p=$(round(p_var, sigdigits=2))", "ISI chi2 p=$(round(p_isi, sigdigits=2))",
        "KS D=$(round(D, digits=4)) p=$(round(p_ks, sigdigits=2))", "lag1 z=$(round(z_ac, digits=2))")
    @test p_isi > PS._PMIN
    @test p_ks > PS._PMIN
    @test abs(z_ac) < 4.5
    N > 10 && PS.check_independence(r, label)
    return nothing
end

# totals of observed spikes, of the spike probabilities and of their binomial variances, and the range of the probabilities
function inhom_totals(P, nsteps, seed)
    Random.seed!(seed)
    pspike, fire = P.pspike, P.fire
    tot_obs = 0; tot_p = 0.0; tot_var = 0.0; pmin = Inf; pmax = 0.0
    for n in 1:nsteps
        SNNModels.integrate!(P, P.param, _DTP)
        tot_obs += count(fire)
        for q in pspike
            tot_p += q; tot_var += q * (1 - q); pmin = min(pmin, q); pmax = max(pmax, q)
        end
    end
    return tot_obs, tot_p, tot_var, Float64(pmin), Float64(pmax)
end

# observed spikes, summed probabilities and summed variances of the neuron-steps whose probability is in each class of [pmin, pmax]
function inhom_classes(P, nsteps, seed, pmin, pmax, nb)
    Random.seed!(seed)
    pspike, fire = P.pspike, P.fire
    obs = zeros(Int, nb); sump = zeros(nb); sumvar = zeros(nb)
    w = nb / (pmax - pmin)
    for n in 1:nsteps
        SNNModels.integrate!(P, P.param, _DTP)
        for i in eachindex(pspike)
            q = Float64(pspike[i])
            k = clamp(floor(Int, (q - pmin) * w) + 1, 1, nb)
            obs[k] += fire[i]; sump[k] += q; sumvar[k] += q * (1 - q)
        end
    end
    return obs, sump, sumvar
end

@testset "Sparse Poisson populations" begin
    dt = _DTP

    @testset "Poisson, homogeneous rate: N = $N, $(rate) Hz" for (N, rate, nsteps) in
                                                                  ((10, 20, 100_000), (1000, 1, 300_000), (1000, 10, 100_000), (1000, 200, 100_000), (1000, 1000, 50_000), (1000, 3000, 20_000), (1, 50, 400_000))
        P = SNNModels.Poisson(N = N, param = PoissonParameter(rate * Hz))
        p = rate * Hz * dt
        r = run_pop(P, nsteps, p)
        check_bernoulli(r, p, nsteps, "Poisson N=$N $(rate)Hz")
    end

    @testset "Poisson equals the dense sampler in distribution: N = $N, $(rate) Hz" for (N, rate, nsteps) in ((10, 50, 200_000), (1000, 10, 100_000), (1000, 100, 50_000))
        P = SNNModels.Poisson(N = N, param = PoissonParameter(rate * Hz))
        rs = run_pop(P, nsteps, rate * Hz * dt; seed = 5)
        rd = run_dense(N, rate * Hz * dt, nsteps; seed = 6)
        D1, p1 = PS.ks_two_sample(rs.totals, rd.totals)
        D2, p2 = PS.ks_two_sample(rs.isis, rd.isis)
        PS._report("Poisson sparse vs dense N=$N $(rate)Hz", "totals KS D=$(round(D1, digits=4)) p=$(round(p1, sigdigits=2))", "ISI KS D=$(round(D2, digits=4)) p=$(round(p2, sigdigits=2))")
        @test p1 > PS._PMIN
        @test p2 > PS._PMIN
    end

    @testset "Poisson, heterogeneous rates: N = $N" for (N, lo, hi, nsteps) in ((10, 1, 100, 200_000), (1000, 1, 100, 100_000), (1000, 20, 3000, 30_000))
        rates = Float32.(exp.(range(log(lo), log(hi), length = N))) .* Hz
        P = SNNModels.Poisson(N = N, param = PoissonParameter(rates))
        r = run_pop(P, nsteps, Float64(minimum(rates) * dt); seed = 9)
        p = Float64.(rates .* dt)
        zs = (r.totals .- nsteps .* p) ./ sqrt.(nsteps .* p .* (1 .- p))
        χ2 = sum(abs2, zs)
        pv = PS.chi2_two_sided(χ2, N)
        @test pv > PS._PMIN
        @test maximum(abs, zs) < 5 + log(N)
        @test maximum(r.khist[3:end]) == 0
        PS._report("Poisson het N=$N rates $(lo)-$(hi)Hz", "sum z^2=$(round(χ2, digits=1))/$(N) (p=$(round(pv, sigdigits=2)))", "max |z|=$(round(maximum(abs, zs), digits=2))")
        N > 10 && PS.check_independence(r, "Poisson het N=$N")
    end

    @testset "Poisson, heterogeneous: rates changed at runtime, all equal rates equal the homogeneous one" begin
        N, nsteps = 500, 100_000
        P = SNNModels.Poisson(N = N, param = PoissonParameter(fill(10f0 * Hz, N)))
        c1 = zeros(Int, N); c2 = zeros(Int, N)
        Random.seed!(8)
        for n in 1:nsteps
            n == nsteps ÷ 2 + 1 && (P.param.rate .= collect(range(2f0 * Hz, 200f0 * Hz, length = N)))
            SNNModels.integrate!(P, P.param, dt)
            n <= nsteps ÷ 2 ? (c1 .+= P.fire) : (c2 .+= P.fire)
        end
        h = nsteps ÷ 2
        p1 = 10Hz * dt
        p2 = Float64.(range(2f0 * Hz, 200f0 * Hz, length = N) .* dt)
        z1 = (c1 .- h * p1) ./ sqrt(h * p1 * (1 - p1))
        z2 = (c2 .- h .* p2) ./ sqrt.(h .* p2 .* (1 .- p2))
        @test PS.chi2_two_sided(sum(abs2, z1), N) > PS._PMIN
        @test PS.chi2_two_sided(sum(abs2, z2), N) > PS._PMIN
    end

    @testset "Poisson: flags, spike list and the connections that read it" begin
        N, Npost = 200, 30
        for param in (PoissonParameter(100Hz), PoissonParameter(Float32.(range(10, 290, length = N)) .* Hz))
            P = SNNModels.Poisson(N = N, param = param)
            Random.seed!(21)
            total = 0
            for n in 1:1000
                SNNModels.integrate!(P, P.param, dt)
                idx = findall(P.fire)
                @test P.spikes.fed                                            # all rates here are below the dense threshold
                @test P.spikes.n == length(idx)
                @test collect(P.spikes.idx[1:P.spikes.n]) == idx              # ascending, equal to the flags
                @test _KP.spike_list!(P.spikes, P.fire) == idx
                total += length(idx)
            end
            @test total > 1000
            @test _KP.spike_index(P) === P.spikes && _KP.spike_index(P).fed
        end
        # a connection reads the list of the population and never scans `fire`: the input equals W * fire
        P = SNNModels.Poisson(N = N, param = PoissonParameter(80Hz))
        E = IF(N = Npost)
        Random.seed!(3)
        syn = SpikingSynapse(P, E, :ge; conn = (p = 0.3, μ = 1.0, σ = 0.0))
        SNNModels.ensure_transpose!(syn)   # J is built only when read
        @test syn.pre_spikes === P.spikes
        W = sparse(syn.I, syn.J, syn.W, Npost, N)
        T = Time()
        g = E.receptors.glu
        for n in 1:500
            fill!(g, 0)
            update_time!(T, dt)
            SNNModels.integrate!(P, P.param, dt)
            _KP.forward!(syn, syn.param, dt, T)
            @test g ≈ W * Float32.(P.fire)
        end
    end

    @testset "Poisson: switching between the sparse and the dense regime keeps flags and list consistent" begin
        N = 300
        P = SNNModels.Poisson(N = N, param = PoissonParameter(20Hz))
        H = SNNModels.Poisson(N = N, param = PoissonParameter(fill(20f0 * Hz, N)))
        Random.seed!(12)
        ok = true; nsp = zeros(Int, 3)
        for n in 1:3000
            rate = (20Hz, 800Hz, 0Hz)[1 + (n ÷ 7) % 3]
            P.param = PoissonParameter(rate); H.param.rate .= rate
            SNNModels.integrate!(P, P.param, dt); SNNModels.integrate!(H, H.param, dt)
            for Q in (P, H)
                idx = findall(Q.fire)
                ok &= _KP.spike_list!(Q.spikes, Q.fire) == idx
                ok &= Q.spikes.fed == (rate * dt < SNNModels._POP_DENSE_SPIKE_P)
                ok &= !Q.spikes.fed || collect(Q.spikes.idx[1:Q.spikes.n]) == idx
            end
            nsp[1 + (n ÷ 7) % 3] += count(P.fire)
        end
        @test ok
        @test nsp[3] == 0 && nsp[1] > 0 && nsp[2] > 5 * nsp[1]
    end

    @testset "regime switch of the fed list (Poisson, heterogeneous Poisson, InhomogeneousPoisson, VariablePoisson)" begin
        N = 250
        thr = SNNModels._POP_DENSE_SPIKE_P
        # sparse: the population feeds the list; dense: it does not, and `spike_list!` scans the flags; in both the list equals findall(fire)
        function check(Q, regime, label)
            idx = findall(Q.fire)
            @test _KP.spike_list!(Q.spikes, Q.fire) == idx
            @test Q.spikes.fed == (regime == :sparse)
            regime == :sparse && @test collect(Q.spikes.idx[1:Q.spikes.n]) == idx
            return length(idx)
        end
        # a connection that reads the population sees exactly the flags in either regime
        P = SNNModels.Poisson(N = N, param = PoissonParameter(10Hz))
        E = IF(N = 20)
        Random.seed!(5)
        syn = SpikingSynapse(P, E, :ge; conn = (p = 0.4, μ = 1.0, σ = 0.0))
        SNNModels.ensure_transpose!(syn)
        W = sparse(syn.I, syn.J, syn.W, 20, N)
        T = Time(); g = E.receptors.glu
        pre_rate = 10Hz
        nspk = Dict(:sparse => 0, :dense => 0)
        for n in 1:2400
            rate = (10Hz, 0.9f0 * thr / dt, 1.1f0 * thr / dt, 600Hz, 0Hz, 10Hz)[1 + (n - 1) ÷ 400]
            P.param = PoissonParameter(rate)
            regime = rate * dt >= thr ? :dense : :sparse
            fill!(g, 0); update_time!(T, dt)
            SNNModels.integrate!(P, P.param, dt)
            nspk[regime] += check(P, regime, "Poisson")
            _KP.forward!(syn, syn.param, dt, T)
            @test g ≈ W * Float32.(P.fire)
        end
        @test nspk[:dense] > 0 && nspk[:sparse] > 0
        # heterogeneous: the regime follows the largest rate of the population, also when it changes at runtime
        H = SNNModels.Poisson(N = N, param = PoissonParameter(fill(5f0 * Hz, N)))
        Random.seed!(6)
        for n in 1:1200
            H.param.rate[7] = (5Hz, 5Hz, 400Hz, 5Hz)[1 + (n - 1) ÷ 300]      # one fast neuron switches the whole population
            regime = H.param.rate[7] * dt >= thr ? :dense : :sparse
            SNNModels.integrate!(H, H.param, dt)
            check(H, regime, "Poisson het")
        end
        # InhomogeneousPoisson: the largest spike probability of the step decides (it moves with the noise); either way the list is right
        Q = InhomogeneousPoisson(N = N, param = InhomogeneousPoissonParam(r0 = 200Hz, β = 8.0f0))
        Random.seed!(7)
        seen = Set{Bool}()
        for n in 1:3000
            Q.r .= (n - 1) ÷ 250 % 2 == 0 ? 0f0 : 1500f0 * Hz      # the rate offset moves the largest probability across the threshold
            SNNModels.integrate!(Q, Q.param, dt)
            check(Q, maximum(Q.pspike) >= thr ? :dense : :sparse, "Inhomogeneous")
            push!(seen, Q.spikes.fed)
        end
        @test length(seen) == 2                    # both regimes visited by the noise
        # VariablePoisson: the total rate moves across the threshold at runtime
        V = VariablePoisson(N = N, param = VariablePoissonParameter(β = 0.0f0, r0 = 20Hz * N))
        Random.seed!(8)
        for n in 1:1200
            target = (20Hz, 20Hz, 600Hz, 20Hz)[1 + (n - 1) ÷ 300] * N
            V.r[1] = target - V.param.r0 / 2       # beta = 0: nu = r0 / 2 + r, so the total rate is `target`
            SNNModels.integrate!(V, V.param, dt)
            regime = V.Erate[1] / N * dt >= thr ? :dense : :sparse
            check(V, regime, "VariablePoisson")
        end
    end

    @testset "Poisson: zero rate, saturated rate, N = 0 and 1" begin
        P = SNNModels.Poisson(N = 50, param = PoissonParameter(0f0))
        for _ in 1:100; SNNModels.integrate!(P, P.param, dt); end
        @test !any(P.fire) && P.spikes.n == 0
        P = SNNModels.Poisson(N = 50, param = PoissonParameter(20_000f0))     # rate * dt = 2 >= 1
        SNNModels.integrate!(P, P.param, dt)
        @test all(P.fire) && !P.spikes.fed && _KP.spike_list!(P.spikes, P.fire) == 1:50      # dense regime: the connections scan the flags
        P.param = PoissonParameter(0f0)
        SNNModels.integrate!(P, P.param, dt)
        @test !any(P.fire) && P.spikes.n == 0 && P.spikes.fed                  # back to the sparse regime: all flags cleared
        P0 = SNNModels.Poisson(N = 0, param = PoissonParameter(10Hz))
        SNNModels.integrate!(P0, P0.param, dt)
        @test P0.spikes.n == 0
        H = SNNModels.Poisson(N = 40, param = PoissonParameter(fill(0f0, 40)))
        SNNModels.integrate!(H, H.param, dt)
        @test !any(H.fire)
        H = SNNModels.Poisson(N = 40, param = PoissonParameter(Float32.(vcat(zeros(20), fill(30_000, 20)))))
        SNNModels.integrate!(H, H.param, dt)
        @test H.fire == vcat(falses(20), trues(20))
        Hs = SNNModels.Poisson(N = 40, param = PoissonParameter(fill(10f0, 5)))
        @test_throws DimensionMismatch SNNModels.integrate!(Hs, Hs.param, dt)
    end

    @testset "InhomogeneousPoisson: constant rate (beta = 0): N = $N, $(rate) Hz" for (N, rate, nsteps) in ((1000, 10, 100_000), (1000, 100, 50_000), (10, 50, 100_000))
        P = InhomogeneousPoisson(N = N, param = InhomogeneousPoissonParam(r0 = rate * Hz, β = 0.0f0))
        P.r .= rate * Hz / 2                                     # beta = 0: rate = r0 / 2 + r = r0, a fixed point of the offset
        p = 1 - exp(-Float64(rate * Hz * dt))
        r = run_pop(P, nsteps, p)
        @test maximum(abs, P.r .- rate * Hz / 2) < 1e-3 * rate * Hz
        check_bernoulli(r, p, nsteps, "InhomogeneousPoisson N=$N $(rate)Hz")
    end

    @testset "InhomogeneousPoisson equals the dense sampler in distribution" begin
        N, nsteps = 1000, 100_000
        P = InhomogeneousPoisson(N = N, param = InhomogeneousPoissonParam(r0 = 30Hz, β = 0.0f0))
        P.r .= 15Hz
        p = 1 - exp(-Float64(30Hz * dt))
        rs = run_pop(P, nsteps, p; seed = 5)
        rd = run_dense(N, p, nsteps; seed = 6)
        D1, p1 = PS.ks_two_sample(rs.totals, rd.totals)
        D2, p2 = PS.ks_two_sample(rs.isis, rd.isis)
        @test p1 > PS._PMIN
        @test p2 > PS._PMIN
    end

    @testset "InhomogeneousPoisson: time-varying rates (beta > 0)" begin
        N, nsteps = 500, 60_000
        for (r0, β) in ((20Hz, 3.0f0), (50Hz, 10.0f0))
            mk() = InhomogeneousPoisson(N = N, param = InhomogeneousPoissonParam(r0 = r0, β = β, τ = 20ms))
            # calibration: the observed spikes against the probability `pspike` of the step (total)
            tot_obs, tot_p, tot_var, pmin_seen, pmax_seen = inhom_totals(mk(), nsteps, 17)
            z = (tot_obs - tot_p) / sqrt(tot_var)
            @test abs(z) < 4.5
            @test pmax_seen > 2 * pmin_seen                      # the rates really vary (otherwise the test is vacuous)
            PS._report("InhomogeneousPoisson r0=$(r0) beta=$β", "total z=$(round(z, digits=2))", "p in [$(round(pmin_seen, sigdigits=2)), $(round(pmax_seen, sigdigits=2))]", "rate $(round(tot_obs / N / (nsteps * dt * 1e-3), digits=1)) Hz")
            # calibration by class: the steps whose probability falls in each of 10 classes of the range, a binomial z per class
            obs, sump, sumvar = inhom_classes(mk(), nsteps, 19, pmin_seen, pmax_seen, 10)
            zs = [(obs[k] - sump[k]) / sqrt(sumvar[k]) for k in 1:10 if sumvar[k] > 20]
            @test length(zs) >= 3
            @test all(abs.(zs) .< 5)
            @test PS.chi2_two_sided(sum(abs2, zs), length(zs)) > PS._PMIN
        end
    end

    @testset "InhomogeneousPoisson: noise and rate offsets follow the model" begin
        # same random numbers (the first draw of a step is the uniform cache), the old formulas
        N = 257
        for (r0, β) in ((20Hz, 3.0f0), (0.5f0, 0.0f0), (100Hz, 40.0f0))
            P = InhomogeneousPoisson(N = N, param = InhomogeneousPoissonParam(r0 = r0, β = β, τ = 20ms))
            noise = zeros(Float32, N); r = fill(Float32(r0), N)
            R(x, v0 = 0.0f0) = x > 0.0f0 ? x : v0
            cache = zeros(Float32, N)
            ok = true; pok = true
            for n in 1:300
                Random.seed!(1000 + n); rand!(cache)
                cc = 1.0f0 - dt / 20ms
                for i in 1:N
                    re = cache[i] - 0.5f0
                    noise[i] = (noise[i] - re) * cc + re
                    Er = R(r0 ./ 2 * R(noise[i] * β, 1.0f0) + r[i], 0.0f0)
                    r[i] += (r0 - Er) / 400ms * dt
                    Float32(1 - exp(-Er * dt)) ≈ P.pspike[i] || (pok = false)
                end
                Random.seed!(1000 + n)
                SNNModels.integrate!(P, P.param, dt)
                ok &= isapprox(P.noise, noise; rtol = 1e-5, atol = 1e-6) && isapprox(P.r, r; rtol = 1e-5, atol = 1e-6)
            end
            @test ok
        end
    end

    @testset "InhomogeneousPoisson: flags, spike list, N = 0" begin
        N = 120
        P = InhomogeneousPoisson(N = N, param = InhomogeneousPoissonParam(r0 = 80Hz, β = 4.0f0))
        Random.seed!(2)
        total = 0
        for n in 1:1000
            SNNModels.integrate!(P, P.param, dt)
            idx = findall(P.fire)
            @test collect(P.spikes.idx[1:P.spikes.n]) == idx
            total += length(idx)
        end
        @test total > 500
        @test _KP.spike_index(P) === P.spikes
        P0 = InhomogeneousPoisson(N = 0, param = InhomogeneousPoissonParam())
        SNNModels.integrate!(P0, P0.param, dt)
        @test P0.spikes.n == 0
        # a rate so high that p >= 0.2 takes the dense path: still the right counts
        Pd = InhomogeneousPoisson(N = 400, param = InhomogeneousPoissonParam(r0 = 3000Hz, β = 0.0f0))
        Pd.r .= 1500Hz
        Random.seed!(4)
        tot = 0
        for n in 1:5000
            SNNModels.integrate!(Pd, Pd.param, dt)
            tot += count(Pd.fire)
        end
        pd = 1 - exp(-Float64(3000Hz * dt))
        @test abs(tot - 400 * 5000 * pd) < 5 * sqrt(400 * 5000 * pd * (1 - pd))
    end

    @testset "VariablePoisson: constant total rate (beta = 0): N = $N, $(rate) Hz per neuron" for (N, rate, nsteps) in ((1000, 20, 100_000), (100, 100, 100_000), (1000, 2, 200_000))
        P = VariablePoisson(N = N, param = VariablePoissonParameter(β = 0.0f0, r0 = rate * Hz * N))
        P.r .= rate * Hz * N / 2                                 # beta = 0: nu = r0 / 2 + r = r0, a fixed point
        p = Float64(rate * Hz * N * dt / N)
        r = run_pop(P, nsteps, p)
        @test P.Erate[1] ≈ rate * Hz * N
        check_bernoulli(r, p, nsteps, "VariablePoisson N=$N $(rate)Hz")
    end

    @testset "VariablePoisson: time-varying total rate (beta > 0)" begin
        N, nsteps = 500, 100_000
        for (r0, β) in ((20Hz * N, 3.0f0), (60Hz * N, 12.0f0))
            P = VariablePoisson(N = N, param = VariablePoissonParameter(β = β, τ = 20ms, r0 = r0))
            Random.seed!(23)
            counts = zeros(Int, nsteps); probs = zeros(Float64, nsteps); list_ok = true
            for n in 1:nsteps
                SNNModels.integrate!(P, P.param, dt)
                counts[n] = count(P.fire)
                list_ok &= P.spikes.n == counts[n]
                probs[n] = Float64(P.Erate[1] / N * dt)
            end
            @test list_ok
            # the spike count of a step is Binomial(N, p_t): standardised sum, and the same in the windows of low and high rate
            μ = N .* probs; v = N .* probs .* (1 .- probs)
            z = (sum(counts) - sum(μ)) / sqrt(sum(v))
            lo = probs .< median(probs)
            zlo = (sum(counts[lo]) - sum(μ[lo])) / sqrt(sum(v[lo]))
            zhi = (sum(counts[.!lo]) - sum(μ[.!lo])) / sqrt(sum(v[.!lo]))
            PS._report("VariablePoisson r0=$(r0) beta=$β", "z=$(round(z, digits=2)) zlo=$(round(zlo, digits=2)) zhi=$(round(zhi, digits=2))", "p in [$(minimum(probs)), $(maximum(probs))]")
            @test abs(z) < 4.5 && abs(zlo) < 4.5 && abs(zhi) < 4.5
            @test maximum(probs) > 1.5 * minimum(probs)
            # the rate of a step is shared by all neurons: no neuron is special (totals per neuron are identically distributed)
        end
        # neurons are exchangeable and independent: totals per neuron
        P = VariablePoisson(N = 200, param = VariablePoissonParameter(β = 0.0f0, r0 = 40Hz * 200))
        P.r .= 40Hz * 200 / 2
        r = run_pop(P, 100_000, Float64(40Hz * dt))
        PS.check_independence(r, "VariablePoisson exchangeable")
    end

    @testset "VariablePoisson: flags and spike list" begin
        P = VariablePoisson(N = 150, param = VariablePoissonParameter(β = 2.0f0, r0 = 60Hz * 150))
        Random.seed!(6)
        total = 0
        for n in 1:1000
            SNNModels.integrate!(P, P.param, dt)
            idx = findall(P.fire)
            @test collect(P.spikes.idx[1:P.spikes.n]) == idx
            total += length(idx)
        end
        @test total > 300
        @test _KP.spike_index(P) === P.spikes
    end

    @testset "networks: Poisson populations drive an IF population, seeded runs are reproducible" begin
        # record the spike count of E with an explicit loop so that the test does not depend on the recorder layout
        function net_spikes(seed)
            Random.seed!(seed)
            P = SNNModels.Poisson(N = 100, param = PoissonParameter(40Hz))
            Q = InhomogeneousPoisson(N = 50, param = InhomogeneousPoissonParam(r0 = 30Hz, β = 2.0f0))
            E = IF(N = 40)
            s1 = SpikingSynapse(P, E, :ge; conn = (p = 0.2, μ = 30.0))
            s2 = SpikingSynapse(Q, E, :ge; conn = (p = 0.2, μ = 30.0))
            T = Time(); n = 0
            for _ in 1:5000
                update_time!(T, dt)
                for pop in (P, Q); SNNModels.integrate!(pop, pop.param, dt); end
                for s in (s1, s2); _KP.forward!(s, s.param, dt, T); end
                SNNModels.integrate!(E, E.param, dt)
                n += count(E.fire)
            end
            return n
        end
        a = net_spikes(3); b = net_spikes(3); c = net_spikes(4)
        @test a == b
        @test a > 0
        @test a != c || a > 0
    end
end

true
