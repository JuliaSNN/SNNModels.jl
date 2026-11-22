# Per-synapse loops of the event-driven vSTDP (`current_weights`, `_ev_spikes!`): the vectorised loops (`@simd ivdep`) are
# bit-identical to the loops of SNNModels <= 2.3.0 (copied in vstdp_ivdep_old_kernel.jl). The loops write only at the loop
# index (W, snap, Wcur) and read the postsynaptic arrays through `I` (gathers): repeated postsynaptic indices in a column
# are harmless, and a test with such a column checks it. Every output and state variable is compared with `==`.
using SNNModels
using Test
using Random
@load_units

include(joinpath(@__DIR__, "vstdp_ivdep_old_kernel.jl"))

# ── Helpers ───────────────────────────────────────────────────────────────────

# Everything the rule writes, copied.
_iv_state(c) = (W = copy(c.W), x = copy(c.LTPVars.x), K = copy(c.LTPVars.K), snap = copy(c.LTPVars.snap),
                u = copy(c.LTPVars.u), v = copy(c.LTPVars.v), steps = copy(c.LTPVars.steps), tick = copy(c.LTPVars.tick),
                matstep = copy(c.LTPVars.matstep), pw = copy(c.LTPVars.pw), cache = copy(c.LTPVars.cache))

# `==` on every field; the cache holds NaN before the first step
_iv_same(a, b) = all(k === :cache ? isequal(a[k], b[k]) : a[k] == b[k] for k in keys(a))

# Connection matrix `Npost x Npre` with empty presynaptic columns (the first, a block in the middle, the last ten).
function _iv_matrix(Npost, Npre, p)
    return [(j == 1 || j in 11:20 || j > Npre - 10) ? 0.0 : (rand() < p ? 0.5 + rand() : 0.0) for i in 1:Npost, j in 1:Npre]
end

# A vSTDP synapse with a random but consistent pending state (amplitudes, accumulator, snapshots, q^n, filters), weights
# with exact zeros (+0 and -0) and empty columns. `Wmax` below the largest weights: the clamp is active. `dup = true`
# repeats postsynaptic indices inside the columns (every third synapse targets the neuron of the previous one).
function _iv_random(FT; Npre = 300, Npost = 200, Wmax = 1.2, seed = 1, dup = false)
    Random.seed!(seed)
    pre = Identity(N = Npre, name = "pre", VFT = Vector{FT})
    post = IF(N = Npost, VFT = Vector{FT}, name = "post")
    syn = SpikingSynapse(pre, post, :ge; conn = _iv_matrix(Npost, Npre, 0.2), name = "syn",
                         LTPParam = vSTDPParameter(FT = FT, Wmax = Wmax))
    if dup
        for j in 1:Npre, s in syn.colptr[j]+1:3:syn.colptr[j+1]-1
            syn.I[s] = syn.I[s-1]
        end
    end
    syn.W[1:9:end] .= 0
    syn.W[2:9:end] .= -0.0
    v = syn.LTPVars
    SNNModels._ev_check!(syn, v, syn.LTPParam, FT(0.125))
    v.x .= 0.1 .* rand(Npre)
    v.x[3:7:end] .= 0
    v.K .= 1e3 .* rand(Npost)
    v.snap .= rand(length(syn.W)) .* v.K[syn.I]
    v.u .= FT.(-80 .+ 40 .* rand(Npost))
    v.steps[1] = 400
    v.pw[1] = 0.8; v.pw[2] = 1 / 0.8
    return syn
end

# Presynaptic spikes of a step: empty columns and a random subset of the others.
_iv_spikes(Npre) = Int32[1; 12; Npre; randperm(MersenneTwister(5), Npre - 30)[1:Npre÷5] .+ 20]

# Spike trains given in advance onto 20 depolarised IF neurons (40 -> 20) with zero weights and empty columns; `rule`:
# the library rule or `OldKernelvSTDP` of it. The weights start at 2 and some reach `Wmax = 2.2`.
function _iv_trained_model(FT, old::Bool; Wmax = 2.2)
    Random.seed!(1)
    dt = 0.125f0; nsteps = 4000
    clock = accumulate((a, _) -> a + dt, 1:nsteps; init = 0.0f0)
    rng = MersenneTwister(4)
    idx = [(k, i) for k in 1:nsteps-1 for i in 1:40 if rand(rng) < 40 * dt * 1e-3]
    src = Identity(N = 40, name = "src", VFT = Vector{FT})
    post = IF(N = 20, param = IFParameter(FT = FT, El = -49mV), name = "post", VFT = Vector{FT})
    stim = SpikeTimeStimulusIdentity(src, :g, param = SpikeTimeParameter(FT.(clock[first.(idx)]), last.(idx)), name = "stim")
    rule = vSTDPParameter(FT = FT, θ_LTD = -60.0, θ_LTP = -55.0, A_LTP = 5e-3, A_LTD = 1e-3, Wmax = Wmax)
    Random.seed!(2)
    M = [j in (1, 17, 40) ? 0.0 : (rand() < 0.5 ? 2.0 : 0.0) for i in 1:20, j in 1:40]
    B = SpikingSynapse(src, post, :ge; conn = M, LTPParam = old ? OldKernelvSTDP(rule) : rule, name = "B")
    B.W[1:11:end] .= 0
    B.W[2:11:end] .= -0.0
    return compose(; src, post, B, stim, silent = true), B
end

# Two vSTDP synapses onto `E` (from `E` and from `P`) driven by Poisson inputs, library rule or old loops.
function _iv_net_model(FT, old::Bool)
    Random.seed!(7)
    E = IF(N = 250, VFT = Vector{FT}, name = "E")
    P = IF(N = 150, VFT = Vector{FT}, name = "P")
    rule = vSTDPParameter(FT = FT, θ_LTD = -65.0, θ_LTP = -55.0, A_LTP = 5e-3, A_LTD = 1e-3, Wmax = 1.8)
    r = old ? OldKernelvSTDP(rule) : rule
    EE = SpikingSynapse(E, E, :ge; conn = _iv_matrix(250, 250, 0.15), LTPParam = r, name = "EE")
    PE = SpikingSynapse(P, E, :ge; conn = _iv_matrix(250, 150, 0.10), LTPParam = r, name = "PE")
    EE.W[1:7:end] .= 0
    sE = PoissonStimulus(E, :ge; param = PoissonFixed(rate = 2kHz), name = "sE")
    sP = PoissonStimulus(P, :ge; param = PoissonFixed(rate = 3kHz), name = "sP")
    return compose(; E, P, EE, PE, sE, sP, silent = true), EE, PE
end

# library and old-loop models trained side by side in chunks; after each chunk the states and the current weights agree
function _iv_compare_runs(build, FT; chunks = 4, duration = 100ms)
    new, old = build(FT, false), build(FT, true)
    syns_new, syns_old = new[2:end], old[2:end]
    ok = true
    for k in 1:chunks
        Random.seed!(100 + k); train!(new[1], duration)
        Random.seed!(100 + k); train!(old[1], duration)
        for (a, b) in zip(syns_new, syns_old)
            ok &= _iv_same(_iv_state(a), _iv_state(b))
            ok &= plasticity_value(a, :W) == old_current_weights(b, b.LTPVars)
        end
    end
    return ok, syns_new
end

# ── current_weights (plasticity_value(syn, :W)) ───────────────────────────────

@testset "current_weights: Float32/Float64, clamp active and inactive, repeated indices" begin
    for FT in (Float32, Float64), Wmax in (1.2, 1e3), dup in (false, true)
        syn = _iv_random(FT; Wmax, dup)
        s0 = _iv_state(syn)
        new = copy(SNNModels.current_weights(syn, syn.LTPVars))
        @test _iv_same(_iv_state(syn), s0)                       # state unchanged
        @test eltype(new) == FT && new == old_current_weights(syn, syn.LTPVars)
        @test new != syn.W                                       # potentiation pending
        ncap = count(==(FT(Wmax)), new)
        Wmax < 2 ? (@test ncap > 0) : (@test ncap == 0)
        @test any(x -> iszero(x) && signbit(x), syn.W)           # -0 weights in the input
    end
end

@testset "current_weights: before the first step it is syn.W" begin
    pre = Identity(N = 10, name = "pre"); post = IF(N = 10, name = "post")
    syn = SpikingSynapse(pre, post, :ge; conn = (p = 0.5, μ = 1.0), LTPParam = vSTDPParameter(), name = "syn")
    @test SNNModels.current_weights(syn, syn.LTPVars) === syn.W
end

# ── Materialisation at the presynaptic spikes (_ev_spikes!) ───────────────────

@testset "_ev_spikes!: Float32/Float64, clamp active and inactive, empty columns, repeated indices" begin
    for FT in (Float32, Float64), Wmax in (1.2, 1e3), dup in (false, true)
        syn = _iv_random(FT; Wmax, dup)
        spikes = _iv_spikes(300)
        new, old = deepcopy(syn), deepcopy(syn)
        SNNModels._ev_spikes!(new, new.LTPVars, new.LTPParam, spikes)
        old_ev_spikes!(old, old.LTPVars, old.LTPParam, spikes)
        sn = _iv_state(new)
        @test _iv_same(sn, _iv_state(old))
        @test sn.W != syn.W && sn.snap != syn.LTPVars.snap && sn.x != syn.LTPVars.x   # the pass did something
        ncap = count(==(FT(Wmax)), sn.W)
        Wmax < 2 ? (@test ncap > 0) : (@test ncap == 0)
    end
end

# ── Whole runs: library loops against the old ones ─────────────────────────────

@testset "train! with spike times: library == old loops, Float32/Float64, clamp active and inactive" begin
    for FT in (Float32, Float64), Wmax in (2.2, 1e3)
        ok, (B,) = _iv_compare_runs((FT, old) -> _iv_trained_model(FT, old; Wmax), FT)
        @test ok
        v = B.LTPVars
        @test any(!iszero, v.K) && any(!iszero, v.snap) && any(!iszero, v.x)   # a pending state
        Wmax < 3 ? (@test count(==(FT(Wmax)), plasticity_value(B, :W)) > 0) : (@test maximum(B.W) < Wmax)
    end
end

@testset "train! with Poisson inputs, two vSTDP synapses: library == old loops, Float32/Float64" begin
    for FT in (Float32, Float64)
        ok, (EE, PE) = _iv_compare_runs(_iv_net_model, FT; duration = 75ms)
        @test ok
        @test EE.LTPVars.tick[1] > EE.LTPVars.cache[8]            # epoch restarts happened
        @test any(!iszero, EE.LTPVars.x) && any(!iszero, PE.LTPVars.x)
        @test any(==(FT(0.1)), EE.W)                              # LTD clamped some weights at Wmin
    end
end
