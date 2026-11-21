# Epoch restart of the event-driven vSTDP (`_ev_rebase!`, also run by `sync_weights!`): the fused, vectorised rebase is
# bit-identical to the serial rebase of SNNModels <= 2.3.0 (copied in vstdp_rebase_old_kernel.jl). The weights and the
# whole rule state (x, K, snapshots, step counter, powers of q) are compared with `==`.
using SNNModels
using Test
using Random
@load_units

include(joinpath(@__DIR__, "vstdp_rebase_old_kernel.jl"))

# ── Helpers ───────────────────────────────────────────────────────────────────

# Everything the rebase writes, copied.
_rb_state(c) = (W = copy(c.W), x = copy(c.LTPVars.x), K = copy(c.LTPVars.K), snap = copy(c.LTPVars.snap),
                steps = copy(c.LTPVars.steps), pw = copy(c.LTPVars.pw))

# Rebase a copy of `c` with the library and another copy with the old kernel; both final states.
function _rb_pair(c)
    new, old = deepcopy(c), deepcopy(c)
    SNNModels._ev_rebase!(new, new.LTPVars)
    old_ev_rebase!(old, old.LTPVars)
    return _rb_state(new), _rb_state(old)
end

# Connection matrix `Npost x Npre` with empty presynaptic columns (the first, a block in the middle, the last ten).
function _rb_matrix(Npost, Npre, p)
    return [(j == 1 || j in 11:20 || j > Npre - 10) ? 0.0 : (rand() < p ? 0.5 + rand() : 0.0) for i in 1:Npost, j in 1:Npre]
end

# A vSTDP synapse with a random but consistent pending state (amplitudes, accumulator, snapshots, q^n), weights with exact
# zeros (+0 and -0) and empty columns. `Wmax` below the largest weights: the clamp is active.
function _rb_random(FT; Npre = 300, Npost = 200, Wmax = 1.2, seed = 1)
    Random.seed!(seed)
    pre = Identity(N = Npre, name = "pre", VFT = Vector{FT})
    post = IF(N = Npost, VFT = Vector{FT}, name = "post")
    syn = SpikingSynapse(pre, post, :ge; conn = _rb_matrix(Npost, Npre, 0.2), name = "syn",
                         LTPParam = vSTDPParameter(FT = FT, Wmax = Wmax))
    syn.W[1:9:end] .= 0
    syn.W[2:9:end] .= -0.0
    v = syn.LTPVars
    SNNModels._ev_check!(syn, v, syn.LTPParam, FT(0.125))
    v.x .= 0.1 .* rand(Npre)
    v.x[3:7:end] .= 0
    v.K .= 1e3 .* rand(Npost)
    v.snap .= rand(length(syn.W)) .* v.K[syn.I]
    v.steps[1] = 400
    v.pw[1] = 0.8; v.pw[2] = 1 / 0.8
    return syn
end

# A trained vSTDP synapse (40 -> 20, spikes at given times, depolarised IF targets) with empty columns and zero weights;
# the weights start at 2 and `Wmax = 2.2` is reached by some of them (the zero weights are potentiated).
function _rb_trained(FT; Wmax = 2.2, duration = 400ms)
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
    B = SpikingSynapse(src, post, :ge; conn = M, LTPParam = rule, name = "B")
    B.W[1:11:end] .= 0
    model = compose(; src, post, B, stim, silent = true)
    Random.seed!(3)
    train!(model, duration)
    return model, B
end

# Two vSTDP synapses onto `E` (from `E` and from `P`) driven by Poisson inputs, normalised every 2 ms with `param`
# (`nothing`: no normalisation, the epoch restarts of the rule run instead).
function _rb_net(FT, param; duration = 300ms)
    Random.seed!(7)
    E = IF(N = 250, VFT = Vector{FT}, name = "E")
    P = IF(N = 150, VFT = Vector{FT}, name = "P")
    rule = vSTDPParameter(FT = FT, θ_LTD = -65.0, θ_LTP = -55.0, A_LTP = 5e-3, A_LTD = 1e-3, Wmax = 1.8)
    EE = SpikingSynapse(E, E, :ge; conn = _rb_matrix(250, 250, 0.15), LTPParam = rule, name = "EE")
    PE = SpikingSynapse(P, E, :ge; conn = _rb_matrix(250, 150, 0.10), LTPParam = rule, name = "PE")
    EE.W[1:7:end] .= 0
    sE = PoissonStimulus(E, :ge; param = PoissonFixed(rate = 2kHz), name = "sE")
    sP = PoissonStimulus(P, :ge; param = PoissonFixed(rate = 3kHz), name = "sP")
    if param === nothing
        model = compose(; E, P, EE, PE, sE, sP, silent = true)
    else
        norm = SynapseNormalization([EE, PE]; param)
        model = compose(; E, P, EE, PE, sE, sP, norm, silent = true)
    end
    Random.seed!(8)
    train!(model; duration)
    return model, EE, PE
end

_rb_same(a, b) = all(getfield(a, k) == getfield(b, k) for k in keys(a))

# ── One rebase on a given state ───────────────────────────────────────────────

@testset "random state: Float32/Float64, clamp active and inactive" begin
    for FT in (Float32, Float64), Wmax in (1.2, 1e3)
        syn = _rb_random(FT; Wmax)
        new, old = _rb_pair(syn)
        @test eltype(new.W) == FT
        @test _rb_same(new, old)
        @test new.W != syn.W                                     # the rebase did something
        ncap = count(==(FT(Wmax)), new.W)
        Wmax < 2 ? (@test ncap > 0) : (@test ncap == 0)          # some weights at the cap, or none
        @test any(iszero, new.W) && all(iszero, new.K) && all(iszero, new.snap)
    end
end

@testset "after train!: Float32/Float64" begin
    for FT in (Float32, Float64)
        _, B = _rb_trained(FT)
        v = B.LTPVars
        @test any(!iszero, v.K) && any(!iszero, v.snap) && any(!iszero, v.x)   # the premise: a pending state
        new, old = _rb_pair(B)
        @test _rb_same(new, old)
        @test new.W != B.W && count(==(FT(2.2)), new.W) > 0          # potentiation pending, some weights at the cap
        # sync_weights! is the same rebase
        twin = deepcopy(B)
        SNNModels.sync_weights!(twin)
        @test _rb_same(_rb_state(twin), new)
    end
end

# ── Whole runs: the library rebase against the old one at every normalisation ─

@testset "train!: vSTDP + normalisation every 2 ms, new rebase == old rebase" begin
    for FT in (Float32, Float64)
        _, EEn, PEn = _rb_net(FT, MultiplicativeNorm(τ = 2ms))
        _, EEo, PEo = _rb_net(FT, OldRebaseNorm(τ = 2ms))
        @test _rb_same(_rb_state(EEn), _rb_state(EEo)) && _rb_same(_rb_state(PEn), _rb_state(PEo))
        @test any(!iszero, EEn.LTPVars.x) && any(!iszero, PEn.LTPVars.x)    # presynaptic spikes
        @test EEn.W != _rb_net(FT, MultiplicativeNorm(τ = 2ms); duration = 0.125ms)[2].W
    end
end

# ── Saved and loaded model ────────────────────────────────────────────────────

@testset "saved/loaded model: new rebase == old rebase" begin
    model, _ = _rb_trained(Float32; duration = 100ms)
    dir = mktempdir()
    try
        save_model(model = model, path = dir, name = "rebase", info = (a = 1,))
        loaded = load_model(dir, "rebase", (a = 1,)).model
        Random.seed!(9)
        train!(loaded; duration = 100ms)
        B = loaded.syn.B
        @test any(!iszero, B.LTPVars.K)
        new, old = _rb_pair(B)
        @test _rb_same(new, old)
    finally
        rm(dir; recursive = true, force = true)
    end
end

# ── Threaded rebase (neurons_per_thread > 0): same bits for every setting ─────
# Meaningful with several threads (`julia -t 4`); with one thread the serial pass runs and the comparisons are trivial.

@testset "column blocks: every column exactly once, about nnz/nc synapses each" begin
    for colptr in (Int32[1, 5, 9, 9, 9], Int32[1, 1, 1, 4, 4, 10, 10], Int32[1, 1, 1, 1], _rb_random(Float32).colptr)
        ncol, nnz = length(colptr) - 1, Int(colptr[end]) - 1
        for nc in 1:8
            blocks = [SNNModels._col_chunk(colptr, nc, k) for k in 1:nc]
            @test reduce(vcat, collect.(blocks)) == 1:ncol
            nnz > 100 && @test all(abs(colptr[last(b)+1] - colptr[first(b)] - nnz / nc) <= nnz / 20 for b in blocks)
        end
    end
end

@testset "threaded: every neurons_per_thread == serial == old rebase (nthreads = $(Threads.nthreads()))" begin
    old_min = SNNModels.SYN_MIN_OPS[]
    SNNModels.SYN_MIN_OPS[] = 0          # thread even the small synapses of the test
    try
        for FT in (Float32, Float64), Wmax in (1.2, 1e3)
            syn = _rb_random(FT; Npre = 1500, Npost = 300, Wmax)
            serial, old = with_neurons_per_thread(() -> _rb_pair(syn), 0)
            @test _rb_same(serial, old)
            for npt in (128, 256, 512, 1024)
                new, _ = with_neurons_per_thread(() -> _rb_pair(syn), npt)
                if Threads.nthreads() > 1
                    @test with_neurons_per_thread(() -> SNNModels._col_nchunks(1500, length(syn.W)), npt) > 1   # threaded pass
                end
                @test _rb_same(new, serial)
            end
        end
    finally
        SNNModels.SYN_MIN_OPS[] = old_min
    end
end

@testset "threaded train!: epoch restarts and normalisation == serial == old rebase" begin
    old_min = SNNModels.SYN_MIN_OPS[]
    SNNModels.SYN_MIN_OPS[] = 0
    try
        _, EEo, PEo = _rb_net(Float32, OldRebaseNorm(τ = 2ms))
        _, EEs, PEs = with_neurons_per_thread(() -> _rb_net(Float32, nothing), 0)      # epoch restarts of the rule only
        for npt in (128, 512)
            _, EEn, PEn = with_neurons_per_thread(() -> _rb_net(Float32, MultiplicativeNorm(τ = 2ms)), npt)
            @test _rb_same(_rb_state(EEn), _rb_state(EEo)) && _rb_same(_rb_state(PEn), _rb_state(PEo))
            _, EEe, PEe = with_neurons_per_thread(() -> _rb_net(Float32, nothing), npt)
            @test _rb_same(_rb_state(EEe), _rb_state(EEs)) && _rb_same(_rb_state(PEe), _rb_state(PEs))
        end
    finally
        SNNModels.SYN_MIN_OPS[] = old_min
    end
end

@testset "serial pass: threading off, below SYN_MIN_OPS" begin
    with_neurons_per_thread(0) do
        @test SNNModels._col_nchunks(1500, 10^7) == 1
    end
    with_neurons_per_thread(512) do
        @test SNNModels._col_nchunks(1500, SNNModels.SYN_MIN_OPS[] - 1) == 1
        @test SNNModels._col_nchunks(1500, 10^7) == (Threads.nthreads() == 1 ? 1 : 3)
    end
end
