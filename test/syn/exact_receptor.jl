# SNNModels 2.2.0: exact one-step integration of the double-exponential receptor kinetics.
# Before 2.2.0 (and with `ReceptorSynapse(exact = false)`, used by `TripodPaper`) the update was
#     h += α w;  g = E(-dt/τd) (g + dt h);  h = E(-dt/τr) h
# whose sampled kernel is c(dt) times the exact one, so the peak conductance was c(dt) g0 instead of g0, with
#     c(dt, τr, τd) = α dt / (1 - exp(-α dt)),   α = 1/τr - 1/τd.
# Derivation: one spike of weight w gives h_k = w α e^{-(k-1)dt/τr}, hence
#     g_n = w α dt e^{-dt/τd} (e^{-n dt/τd} - e^{-n dt/τr}) / (e^{-dt/τd} - e^{-dt/τr}) = c (w) (e^{-n dt/τd} - e^{-n dt/τr}).
# Migration rule: a simulation made with parameter g0_old (legacy update, exact exponentials) is reproduced by the exact update with
#     g0_new = c(dt) g0_old.
# (a) peak = g0 and exact kernel, (b) closed form of c and equivalence of trajectories, (c) neuron level (AdEx, Tripod soma),
# (d) balanced E/I network: how long the spike trains stay identical, and the statistics.
using Test, Random, Statistics
using SNNModels
@load_units

const _ER = SNNModels
_cfac(dt, τr, τd) = (a = 1 / τr - 1 / τd; a * dt / (-expm1(-a * dt)))

# receptors of the four kinetics used by the library (AMPA, GABAa, NMDA, GABAb): (name, τr, τd)
const _ER_KIN = (("AMPA", 0.25, 2.0), ("GABAa", 0.3, 15.0), ("NMDA", 1.0, 100.0), ("GABAb", 30.0, 100.0))

_er_mk(FT, E, τr, τd, g0, tg; nmda = 0.0) = Receptor(T = FT, E_rev = E, τr = τr, τd = τd, g0 = g0, nmda = nmda, target = tg)
function _er_syn(FT, recs; glu = [1], gaba = Int[], exact = true, decay = 0)
    return ReceptorSynapse(ST = Vector{Receptor{FT}}, NMDAT = NMDAVoltageDependency{FT}, syn = recs,
        NMDA = NMDAVoltageDependency(T = FT), glu_receptors = glu, gaba_receptors = gaba, exact = exact, decay = decay)
end

# conductance trace (gsyn g, nS) of receptor 1 of `syn` driven by `w[k]` arriving at the beginning of step k; `P` is any population
const _ER_P = IF(N = 1)
function _er_trace(syn, FT, dt, w)
    sv = _ER.synaptic_variables(syn, 1, FT)
    rec = (glu = zeros(FT, 1), gaba = zeros(FT, 1))
    out = zeros(FT, length(w))
    for k in eachindex(w)
        rec.glu[1] = w[k]
        _ER.update_synapses!(_ER_P, syn, rec, sv, FT(dt), 1:1)
        out[k] = syn.syn[1].gsyn * sv.g[1, 1]
    end
    return out
end

# TEST-ONLY copy of the update of releases <= 2.1 (scalar, exact exponentials, i.e. `decay = 0`)
function _er_old_trace(τr, τd, g0, dt, w)
    α = 1 / τr - 1 / τd
    gsyn = g0 * norm_synapse(τr, τd)
    g = h = 0.0
    out = zeros(length(w))
    for k in eachindex(w)
        h += α * w[k]
        g = exp(-dt / τd) * (g + dt * h)
        h = exp(-dt / τr) * h
        out[k] = gsyn * g
    end
    return out
end

@testset "exact receptor kinetics (2.2.0)" verbose = true begin

    @testset "(a) a unit spike: peak conductance g0 and the exact kernel at every grid point" begin
        for (nm, τr, τd) in _ER_KIN, FT in (Float32, Float64), dt in (0.01, 0.05, 0.1, 0.125, 0.2)
            g0 = 0.8
            syn = _er_syn(FT, [_er_mk(FT, 0.0, τr, τd, g0, :glu)])
            r = syn.syn[1]
            nst = round(Int, 6τd / dt)
            tr = _er_trace(syn, FT, dt, [1; zeros(nst - 1)])
            peak = maximum(tr)
            tol = FT === Float32 ? 3e-4 : 1e-6
            # continuous peak is g0; on the grid it is at most slightly below (never above, up to rounding)
            @test peak <= g0 * (1 + tol)
            @test peak >= g0 * (1 - 5e-3)
            # kernel = g0 norm (e^{-t/τd} - e^{-t/τr}) with the stored constants
            t = dt .* (1:nst)
            ex = @. r.gsyn * (exp(-t * Float64(r.τd⁻)) - exp(-t * Float64(r.τr⁻)))
            @test maximum(abs.(Float64.(tr) .- ex)) <= (FT === Float32 ? 3e-4 : 1e-12) * g0
        end
    end

    @testset "(a) a spike of weight w and several receptors; MultiReceptorSynapse" begin
        for FT in (Float32, Float64)
            recs = [_er_mk(FT, 0.0, τr, τd, 0.7, :glu) for (_, τr, τd) in _ER_KIN]
            dt = 0.125
            syn = _er_syn(FT, recs; glu = [1, 2, 3, 4])
            msyn = MultiReceptorSynapse(ST = Vector{Receptor{FT}}, NMDAT = NMDAVoltageDependency{FT}, syn = recs, NMDA = NMDAVoltageDependency(T = FT),
                receptors = (glu = [1, 2, 3, 4],))
            for s in (syn, msyn)
                sv = _ER.synaptic_variables(s, 1, FT)
                rec = s isa ReceptorSynapse ? (glu = zeros(FT, 1), gaba = zeros(FT, 1)) : (glu = zeros(FT, 1),)
                pk = zeros(4)
                for k in 1:round(Int, 600 / dt)
                    k == 1 && (rec.glu[1] = 2.5)
                    _ER.update_synapses!(_ER_P, s, rec, sv, FT(dt), 1:1)
                    for n in 1:4; pk[n] = max(pk[n], s.syn[n].gsyn * sv.g[1, n]); end
                end
                @test all(pk .<= 2.5 * 0.7 * (1 + 3e-4)) && all(pk .>= 2.5 * 0.7 * (1 - 5e-3))
            end
        end
    end

    @testset "(a) the integrator is independent of the chunk and a receptor without time constants stays at zero" begin
        FT = Float64
        syn = _er_syn(FT, [_er_mk(FT, 0.0, 0.5, 5.0, 1.0, :glu), Receptor(T = FT)]; glu = [1, 2])
        sv = _ER.synaptic_variables(syn, 9, FT)
        rec = (glu = rand(MersenneTwister(1), FT, 9), gaba = zeros(FT, 9))
        _ER.update_synapses!(_ER_P, syn, rec, sv, 0.1, 1:9)
        @test all(isfinite, sv.g) && all(isfinite, sv.h) && all(==(0), sv.g[:, 2]) && all(==(0), sv.h[:, 2])
        sv2 = _ER.synaptic_variables(syn, 9, FT)
        rec2 = (glu = copy(rec.glu), gaba = zeros(FT, 9))
        rec2.glu .= rand(MersenneTwister(1), FT, 9)
        for r in (1:4, 5:9); _ER.update_synapses!(_ER_P, syn, rec2, sv2, 0.1, r); end
        @test sv2.g == sv.g && sv2.h == sv.h
    end

    @testset "(b) closed form of c(dt, τr, τd) and the migration rule g0_new = c g0_old" begin
        table = Dict(0.125 => (1.2346, 1.2180, 1.0632, 1.0015), 0.1 => (1.1852, 1.1722, 1.0503, 1.0012))
        for (dt, vals) in table, ((nm, τr, τd), v) in zip(_ER_KIN, vals)
            @test _cfac(dt, τr, τd) ≈ v atol = 5e-5          # the numbers of the changelog (four digits)
        end
        # c is the ratio of the legacy sampled kernel to the exact one, at every grid point, to 1e-6
        for (nm, τr, τd) in _ER_KIN, dt in (0.1, 0.125)
            nst = round(Int, 5τd / dt)
            old = _er_old_trace(τr, τd, 1.0, dt, [1.0; zeros(nst - 1)])
            t = dt .* (1:nst)
            ex = @. norm_synapse(τr, τd) * (exp(-t / τd) - exp(-t / τr))
            @test all(abs.(old ./ ex .- _cfac(dt, τr, τd)) .< 1e-6)
        end
        # the library legacy path (exact = false, decay = 0) is the test-only copy of the old update
        for (nm, τr, τd) in _ER_KIN
            dt = 0.125
            w = Float64.(rand(MersenneTwister(2), 400) .< 0.05) .* (0.5 .+ rand(MersenneTwister(3), 400))
            legacy = _er_syn(Float64, [_er_mk(Float64, 0.0, τr, τd, 0.9, :glu)]; exact = false, decay = 0)
            @test _er_trace(legacy, Float64, dt, w) ≈ _er_old_trace(τr, τd, 0.9, dt, w) rtol = 1e-12
        end
    end

    @testset "(b) old trajectories (g0_old) = new integrator with c g0_old, random spike trains, Float64" begin
        for (nm, τr, τd) in _ER_KIN, dt in (0.1, 0.125, 0.2)
            rng = MersenneTwister(11)
            w = Float64.(rand(rng, 4000) .< 0.03) .* (0.2 .+ 3 * rand(rng, 4000))
            g0 = 0.73
            old = _er_old_trace(τr, τd, g0, dt, w)
            new_scaled = _er_trace(_er_syn(Float64, [_er_mk(Float64, 0.0, τr, τd, g0 * _cfac(dt, τr, τd), :glu)]), Float64, dt, w)
            new_plain = _er_trace(_er_syn(Float64, [_er_mk(Float64, 0.0, τr, τd, g0, :glu)]), Float64, dt, w)
            @test new_scaled ≈ old rtol = 1e-5 atol = 1e-12
            @test maximum(abs.(new_scaled .- old)) <= 1e-10 * maximum(old)
            @test maximum(new_plain) < maximum(old) && sum(new_plain) < sum(old)    # the unscaled new kernel is weaker by c
            @test sum(old) / sum(new_plain) ≈ _cfac(dt, τr, τd) rtol = 1e-6
        end
    end

    # ---------------------------------------------------------------------------------------------------------------------------
    # (c) neuron level: the same seeded Poisson input (Bernoulli per step) into AdEx / Tripod soma receptors
    # ---------------------------------------------------------------------------------------------------------------------------
    _ER_REC(FT, mult; dt) = [
        _er_mk(FT, 0.0, 0.25, 2.0, 0.73 * mult(dt, 0.25, 2.0), :glu),
        _er_mk(FT, -75.0, 0.3, 15.0, 0.38 * mult(dt, 0.3, 15.0), :gaba),
    ]
    # `mode`: :legacy (old update, g0), :scaled (exact update, c g0), :plain (exact update, g0)
    function _er_mode_syn(FT, mode, dt)
        mult = mode === :scaled ? _cfac : (args...) -> 1.0
        return _er_syn(FT, _ER_REC(FT, mult; dt); glu = [1], gaba = [2], exact = mode !== :legacy, decay = 0)
    end
    function _er_input(seed, N, nsteps; rate_e = 4.0, rate_i = 1.5, we = 12.0, wi = 0.5, dt)
        rng = MersenneTwister(seed)
        pe, pi_ = rate_e * dt * 1e-3 * 100, rate_i * dt * 1e-3 * 100    # 100 independent Poisson afferents
        e = [we * count(<(pe / 100), rand(rng, 100)) for _ in 1:nsteps, _ in 1:N]
        i = [wi * count(<(pi_ / 100), rand(rng, 100)) for _ in 1:nsteps, _ in 1:N]
        return e, i
    end
    function _er_run(make, mode, FT, e, i, dt)
        p = make(FT, mode)
        N = size(e, 2)
        V = zeros(Float64, size(e, 1), N); S = Vector{Vector{Float64}}(undef, N); foreach(k -> S[k] = Float64[], 1:N)
        recs = p isa Tripod ? p.receptors_s : p.receptors
        vm = p isa Tripod ? p.v_s : p.v
        for k in axes(e, 1)
            recs.glu .= FT.(e[k, :]); recs.gaba .= FT.(i[k, :])
            _ER.integrate!(p, p.param, FT(dt))
            V[k, :] .= vm
            for n in findall(p.fire); push!(S[n], k * dt); end
        end
        return V, S
    end

    @testset "(c) AdEx with receptors" begin
        for FT in (Float64, Float32)
            dt = 0.125
            N, nsteps = 20, 4000
            e, i = _er_input(21, N, nsteps; dt)
            make(FT, mode) = begin
                p = AdEx(N = N, VFT = Vector{FT}, param = AdExParameter(FT = FT), spike = PostSpike(FT = FT), synapse = _er_mode_syn(FT, mode, dt))
                p.v .= FT(-65); p
            end
            V0, S0 = _er_run(make, :legacy, FT, e, i, dt)
            V1, S1 = _er_run(make, :scaled, FT, e, i, dt)
            V2, S2 = _er_run(make, :plain, FT, e, i, dt)
            n0, n1, n2 = sum(length, S0), sum(length, S1), sum(length, S2)
            dV = maximum(abs.(V0 .- V1))
            dts = maximum((isempty(a) && isempty(b)) ? 0.0 : (length(a) == length(b) ? maximum(abs.(a .- b); init = 0.0) : Inf) for (a, b) in zip(S0, S1))
            @info "AdEx $FT: spikes legacy $n0, exact c g0 $n1, exact g0 $n2; max |ΔV| $(dV) mV; max spike-time shift $(dts) ms"
            @test n0 > 50
            if FT === Float64
                @test dV < 1e-6 && dts == 0.0 && n0 == n1
            else
                @test dV < 0.5 && abs(n0 - n1) <= max(2, 0.01 * n0)
            end
            @test n2 < n0                       # the unscaled exact kernel is weaker (peak g0 instead of c g0)
        end
    end

    @testset "(c) Tripod soma with receptors" begin
        for FT in (Float64, Float32)
            dt = 0.125
            N, nsteps = 8, 4000
            e, i = _er_input(33, N, nsteps; dt)
            make(FT, mode) = begin
                s = _er_mode_syn(FT, mode, dt)
                p = Tripod(N = N, VFT = Vector{FT}, soma_syn = s, dend_syn = convert_precision(TripodDendSynapse, FT),
                    param = TripodParameter(ds = [250um, 150um]))
                p.v_s .= FT(-70.6); p.v_d1 .= FT(-70.6); p.v_d2 .= FT(-70.6)    # same initial state in every run
                p
            end
            V0, S0 = _er_run(make, :legacy, FT, e, i, dt)
            V1, S1 = _er_run(make, :scaled, FT, e, i, dt)
            V2, S2 = _er_run(make, :plain, FT, e, i, dt)
            n0, n1, n2 = sum(length, S0), sum(length, S1), sum(length, S2)
            dV = maximum(abs.(V0 .- V1))
            dts = maximum((isempty(a) && isempty(b)) ? 0.0 : (length(a) == length(b) ? maximum(abs.(a .- b); init = 0.0) : Inf) for (a, b) in zip(S0, S1))
            @info "Tripod soma $FT: spikes legacy $n0, exact c g0 $n1, exact g0 $n2; max |ΔV| $(dV) mV; max spike-time shift $(dts) ms"
            @test n0 > 20
            if FT === Float64
                @test dV < 1e-6 && dts == 0.0 && n0 == n1
            else
                @test dV < 0.5 && abs(n0 - n1) <= max(2, 0.01 * n0)
            end
            @test n2 < n0
        end
    end

    # ---------------------------------------------------------------------------------------------------------------------------
    # (d) balanced E/I network (COBA-type): 320 E + 80 I AdEx neurons, AMPA/GABAa, sparse random connectivity, seeded
    # ---------------------------------------------------------------------------------------------------------------------------
    # Constant heterogeneous drive (seeded), recurrent excitation and inhibition (no shared noise input: noise makes a network
    # reliable, and the comparison should be as sensitive as possible to a perturbation of the synaptic kernel).
    function _er_net(FT, mode; dt = 0.125, T = 3000.0, seed = 7, I0 = 850.0, μe = 30.0, μi = 60.0)
        Random.seed!(seed)
        mk() = _er_mode_syn(FT, mode, dt)
        ad, sp = AdExParameter(FT = FT), PostSpike(FT = FT)
        E = AdEx(N = 320, VFT = Vector{FT}, param = ad, spike = sp, synapse = mk(), name = "E")
        I = AdEx(N = 80, VFT = Vector{FT}, param = ad, spike = sp, synapse = mk(), name = "I")
        E.I .= I0 .* (0.8 .+ 0.4 .* rand(FT, 320)); I.I .= I0 .* (0.8 .+ 0.4 .* rand(FT, 80))
        ee = SpikingSynapse(E, E, :glu; conn = (p = 0.1, μ = μe))
        ei = SpikingSynapse(E, I, :glu; conn = (p = 0.1, μ = μe))
        ie = SpikingSynapse(I, E, :gaba; conn = (p = 0.1, μ = μi))
        ii = SpikingSynapse(I, I, :gaba; conn = (p = 0.1, μ = μi))
        model = compose(; E, I, ee, ei, ie, ii, silent = true)
        monitor!(E, [:fire]); monitor!(I, [:fire])
        sim!(; model, duration = T * ms, dt = dt * ms)
        return [Float64.(s) for s in spiketimes(E)], [Float64.(s) for s in spiketimes(I)], T
    end
    # Poisson-driven variant (1000 afferents at 10 Hz, 20 % connectivity): the rate depends on the synaptic strength; used for the
    # effect of the unscaled exact kernel
    function _er_net_poisson(FT, mode; dt = 0.125, T = 2000.0, seed = 7)
        Random.seed!(seed)
        mk() = _er_mode_syn(FT, mode, dt)
        ad, sp = AdExParameter(FT = FT), PostSpike(FT = FT)
        E = AdEx(N = 320, VFT = Vector{FT}, param = ad, spike = sp, synapse = mk(), name = "E")
        I = AdEx(N = 80, VFT = Vector{FT}, param = ad, spike = sp, synapse = mk(), name = "I")
        ext = PoissonLayer(10Hz, N = 1000)
        ee = SpikingSynapse(E, E, :glu; conn = (p = 0.1, μ = 0.5))
        ei = SpikingSynapse(E, I, :glu; conn = (p = 0.1, μ = 0.5))
        ie = SpikingSynapse(I, E, :gaba; conn = (p = 0.1, μ = 0.1))
        ii = SpikingSynapse(I, I, :gaba; conn = (p = 0.1, μ = 0.1))
        xe = Stimulus(ext, E, :glu; conn = (p = 0.2, μ = 5.0), name = "xe")
        xi = Stimulus(ext, I, :glu; conn = (p = 0.2, μ = 5.0), name = "xi")
        model = compose(; E, I, ee, ei, ie, ii, xe, xi, silent = true)
        monitor!(E, [:fire]); monitor!(I, [:fire])
        Random.seed!(seed + 1)
        sim!(; model, duration = T * ms, dt = dt * ms)
        return [Float64.(s) for s in spiketimes(E)], [Float64.(s) for s in spiketimes(I)], T
    end
    # first time (ms) at which two spike-train sets differ: the earlier of the first mismatching spikes over the neurons
    function _er_identical_until(A, B)
        t = Inf
        for (a, b) in zip(A, B)
            k = findfirst(j -> j > length(a) || j > length(b) || a[j] != b[j], 1:max(length(a), length(b)))
            k === nothing && continue
            ta = k <= length(a) ? a[k] : Inf; tb = k <= length(b) ? b[k] : Inf
            t = min(t, ta, tb)
        end
        return t
    end
    _er_rate(S, T) = sum(length, S) / length(S) / (T * 1e-3)
    function _er_cv(S)
        cvs = [begin d = diff(s); std(d) / mean(d) end for s in S if length(s) >= 4]
        return mean(cvs)
    end
    _er_first(S) = [isempty(s) ? NaN : s[1] for s in S]

    @testset "(d) balanced E/I network: legacy g0 vs exact c g0 vs exact g0" begin
        for FT in (Float64, Float32)
            E0, I0, T = _er_net(FT, :legacy)
            E0b, I0b, _ = _er_net(FT, :legacy)
            @test E0 == E0b && I0 == I0b                      # the seeded run is reproducible
            E1, I1, _ = _er_net(FT, :scaled)
            tid = min(_er_identical_until(E0, E1), _er_identical_until(I0, I1))
            r0, r1 = _er_rate(E0, T), _er_rate(E1, T)
            c0, c1 = _er_cv(E0), _er_cv(E1)
            f0, f1 = _er_first(E0), _er_first(E1)
            ok = .!isnan.(f0) .& .!isnan.(f1)
            # seeds 1-6: how long the spike trains stay identical (rounding-level perturbation: Float64 1e-14, Float32 1e-6)
            tids = map(1:6) do sd
                A0, B0, _ = _er_net(FT, :legacy; seed = sd, T = 2000.0)
                A1, B1, _ = _er_net(FT, :scaled; seed = sd, T = 2000.0)
                min(_er_identical_until(A0, A1), _er_identical_until(B0, B1))
            end
            @info "recurrent network $FT (320 E + 80 I, DC drive, T = $(T) ms): legacy g0 vs exact c g0: spike trains identical until $(tid) ms (seed 7); seeds 1-6 (2000 ms): $(tids) ms; E rate $(round(r0, digits = 3)) / $(round(r1, digits = 3)) Hz, ISI CV $(round(c0, digits = 3)) / $(round(c1, digits = 3)), first spikes max |Δ| $(maximum(abs.(f0[ok] .- f1[ok]); init = 0.0)) ms over $(count(ok)) neurons"
            @test r0 > 2
            @test FT === Float64 ? (tid > 1000 && all(>(1000), tids)) : (tid > 20 && all(>(20), tids))   # Float64: whole run; Float32: until a rounding-induced threshold flip
            @test abs(r1 - r0) / r0 < 0.05                      # same population rate
            @test abs(c1 - c0) < 0.1                            # same ISI statistics
            @test maximum(abs.(f0[ok] .- f1[ok]); init = 0.0) <= 1.0
            # Poisson-driven: the effect the migration removes
            P0, Q0, TP = _er_net_poisson(FT, :legacy)
            P1, Q1, _ = _er_net_poisson(FT, :scaled)
            P2, Q2, _ = _er_net_poisson(FT, :plain)
            tp = min(_er_identical_until(P0, P1), _er_identical_until(Q0, Q1))
            pr0, pr1, pr2 = _er_rate(P0, TP), _er_rate(P1, TP), _er_rate(P2, TP)
            @info "Poisson-driven network $FT: identical until $(tp) ms; E rate legacy g0 $(round(pr0, digits = 2)) Hz, exact c g0 $(round(pr1, digits = 2)) Hz, exact g0 (unscaled) $(round(pr2, digits = 2)) Hz (ratio $(round(pr2 / pr0, digits = 3))); ISI CV $(round(_er_cv(P0), digits = 3)) / $(round(_er_cv(P1), digits = 3)) / $(round(_er_cv(P2), digits = 3))"
            @test tp > 20 && abs(pr1 - pr0) / pr0 < 0.05
            @test pr2 < 0.8 * pr0                               # the unscaled exact kernel (peak g0 instead of c g0, c ≈ 1.23 for AMPA) is much weaker
        end
    end

end
true
