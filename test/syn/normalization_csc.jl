# SynapseNormalization kernels in CSC order: bit-identical to the row-by-row kernels through the transposed index
# (SNNModels <= 2.3.0, copied in normalization_old_kernels.jl). Weights, sums `W1` and factors `μ` are compared with `==`.
using SNNModels
using Test
using Random
@load_units

include(joinpath(@__DIR__, "normalization_old_kernels.jl"))

# ── Helpers ───────────────────────────────────────────────────────────────────

# Two synapses onto the same population `E` (from `E` and from `P`), random weights with exact zeros, and a block of
# postsynaptic neurons without any input (their columns of the matrix are empty).
function _norm_synapses(FT; N = 300, Npre = 200, seed = 1)
    Random.seed!(seed)
    E = IF(N = N, VFT = Vector{FT}, name = "E")
    P = IF(N = Npre, VFT = Vector{FT}, name = "P")
    M1 = [i > N - 20 ? 0.0 : (rand() < 0.15 ? 0.5 + rand() : 0.0) for i in 1:N, j in 1:N]
    M2 = [i > N - 20 ? 0.0 : (rand() < 0.10 ? 0.5 + rand() : 0.0) for i in 1:N, j in 1:Npre]
    EE = SpikingSynapse(E, E, :ge; conn = M1, LTPParam = STDPGerstner(), name = "EE")
    PE = SpikingSynapse(P, E, :ge; conn = M2, LTPParam = STDPGerstner(), name = "PE")
    for syn in (EE, PE)
        syn.W[1:7:end] .= 0          # exact zeros among the stored synapses
        syn.W .*= FT(0.5) .+ rand(FT, length(syn.W))
    end
    return E, P, EE, PE
end

# Build a normalization on `synapses` and one on a copy, perturb the weights of both identically (so that the step changes them),
# run one step with the new kernels on the first and with the old ones on the second; return both normalizations.
function _one_step_pair(param, synapses)
    copies = deepcopy(synapses)
    new = SynapseNormalization(synapses; param)
    old = SynapseNormalization(copies; param)
    Random.seed!(21)
    for (a, b) in zip(synapses, copies)
        FT = eltype(a.W)
        f = FT(0.7) .+ FT(0.6) .* rand(FT, length(a.W))
        a.W .*= f
        b.W .*= f
    end
    SNNModels.plasticity!(new, param)
    old_norm_step!(old, param isa AdditiveNorm)
    return new, old
end

_same(new, old) = new.W1 == old.W1 && new.μ == old.μ && all(a.W == b.W for (a, b) in zip(new.synapses, old.synapses))

# ── One step ──────────────────────────────────────────────────────────────────

@testset "one step: Float32/Float64, multiplicative/additive, two synapses" begin
    for FT in (Float32, Float64), param in (MultiplicativeNorm(τ = 10ms), AdditiveNorm(τ = 10ms))
        E, P, EE, PE = _norm_synapses(FT)
        new, old = _one_step_pair(param, [EE, PE])
        @test eltype(EE.W) == FT
        @test _same(new, old)
        @test any(iszero, new.W1)                                   # neurons without input are present
        @test any(!=(one(FT)), new.μ) && any(!=(zero(FT)), new.μ)   # the step did something
        # repeated steps stay identical
        for _ in 1:3
            SNNModels.plasticity!(new, param)
            old_norm_step!(old, param isa AdditiveNorm)
        end
        @test _same(new, old)
    end
end

@testset "one step: synapse without transpose is not given one" begin
    E = IF(N = 50)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.3, μ = 1.0))
    norm = SynapseNormalization([EE]; param = MultiplicativeNorm(τ = 10ms))
    EE.J = similar(EE.J, 0); EE.index = similar(EE.index, 0)   # dropped, as for a synapse whose rule does not read it
    SNNModels.plasticity!(norm, norm.param)
    @test !SNNModels.has_transpose(EE)
end

# ── Event-driven vSTDP: the step acts on the current (lazy) weights ───────────

@testset "vSTDP lazy weights" begin
    function build_vs(FT)
        dt = 0.125f0; nsteps = 4000
        clock = accumulate((a, _) -> a + dt, 1:nsteps; init = 0.0f0)
        rng = MersenneTwister(4)
        idx = [(k, i) for k in 1:nsteps-1 for i in 1:40 if rand(rng) < 40 * dt * 1e-3]
        src = Identity(N = 40, name = "src", VFT = Vector{FT})
        post = IF(N = 20, param = IFParameter(FT = FT, El = -49mV), name = "post", VFT = Vector{FT})
        stim = SpikeTimeStimulusIdentity(src, :g, param = SpikeTimeParameter(FT.(clock[first.(idx)]), last.(idx)), name = "stim")
        rule = vSTDPParameter(FT = FT, θ_LTD = -60.0, θ_LTP = -55.0, A_LTP = 5e-3, A_LTD = 1e-3)
        B = SpikingSynapse(src, post, :ge; conn = (p = 0.5, μ = 2.0), LTPParam = rule, name = "B")
        return compose(; src, post, B, stim, silent = true), B
    end
    for FT in (Float32, Float64), param in (MultiplicativeNorm(τ = 10ms), AdditiveNorm(τ = 10ms))
        Random.seed!(2); m1, B1 = build_vs(FT)
        Random.seed!(2); m2, B2 = build_vs(FT)
        n1 = SynapseNormalization([B1]; param)
        n2 = SynapseNormalization([B2]; param)
        Random.seed!(3); train!(m1, 400ms)
        Random.seed!(3); train!(m2, 400ms)
        @test B1.W == B2.W && B1.W != plasticity_value(B1, :W)      # the premise: W is stale
        SNNModels.plasticity!(n1, param)
        old_norm_step!(n2, param isa AdditiveNorm)
        @test plasticity_value(B1, :W) == plasticity_value(B2, :W)
        @test n1.W1 == n2.W1 && n1.μ == n2.μ
    end
end

# ── Training: STDP + normalisation over many steps ────────────────────────────

# Network with two plastic synapses onto `E`, normalized every 2 ms; `param` selects new (Multiplicative/AdditiveNorm) or old
# (OldNorm) kernels. The constructors draw from the global RNG: both twins are built from the same seed.
function _trained_net(FT, param; duration = 300ms)
    E, P, EE, PE = _norm_synapses(FT; N = 250, Npre = 150, seed = 7)
    sE = PoissonStimulus(E, :ge; param = PoissonFixed(rate = 2kHz), name = "sE")
    sP = PoissonStimulus(P, :ge; param = PoissonFixed(rate = 3kHz), name = "sP")
    norm = SynapseNormalization([EE, PE]; param)
    model = compose(; E, P, EE, PE, sE, sP, norm, silent = true)
    Random.seed!(8)
    train!(model; duration)
    return model, EE, PE, norm
end

@testset "train!: STDP + normalisation, new kernels == old kernels" begin
    for FT in (Float32, Float64), additive in (false, true)
        new_param = additive ? AdditiveNorm(τ = 2ms) : MultiplicativeNorm(τ = 2ms)
        _, EEn, PEn, nn = _trained_net(FT, new_param)
        _, EEo, PEo, no = _trained_net(FT, OldNorm(τ = 2ms, additive = additive))
        @test EEn.W == EEo.W && PEn.W == PEo.W
        @test nn.W1 == no.W1 && nn.μ == no.μ
        @test sum(nn.W1) > 0 && EEn.W != _norm_synapses(FT; N = 250, Npre = 150, seed = 7)[3].W   # the weights moved
    end
end

# ── Saved and loaded model ────────────────────────────────────────────────────

@testset "saved/loaded model: new step == old step" begin
    model, _, _, _ = _trained_net(Float32, MultiplicativeNorm(τ = 2ms); duration = 50ms)
    dir = mktempdir()
    try
        save_model(model = model, path = dir, name = "norm", info = (a = 1,))
        loaded = load_model(dir, "norm", (a = 1,)).model
        Random.seed!(9)
        train!(loaded; duration = 50ms)
        twin = deepcopy(loaded)
        norm, norm_old = loaded.syn.norm, twin.syn.norm
        for syn in norm.synapses
            syn.W .*= 1.1f0
        end
        for syn in norm_old.synapses
            syn.W .*= 1.1f0
        end
        SNNModels.plasticity!(norm, norm.param)
        old_norm_step!(norm_old, false)
        @test _same(norm, norm_old)
    finally
        rm(dir; recursive = true, force = true)
    end
end

# ── AggregateScaling: same CSC-order sums and per-synapse update ──────────────

@testset "AggregateScaling: new step == old step" begin
    for FT in (Float32, Float64)
        E, P, EE, PE = _norm_synapses(FT)
        copies = deepcopy([EE, PE])
        param = AggregateScalingParameter(E.N, 5Hz; Wmin = 0.01)
        new = AggregateScaling(E, [EE, PE]; param)
        old = AggregateScaling(E, copies; param)
        # a target sum and a rate state that move the weights, identical in both
        rates = rand(FT, E.N)
        for c in (new, old)
            c.WT .*= FT(0.8) .+ FT(0.4) .* rates
        end
        for _ in 1:3
            SNNModels.plasticity!(new, new.param)
            old_aggregate_step!(old, old.param)
        end
        @test new.Wt == old.Wt && new.μ == old.μ
        @test EE.W == copies[1].W && PE.W == copies[2].W
        @test any(!=(one(FT)), new.μ) && any(iszero, new.Wt)
    end
end

# ── Threaded passes (neurons_per_thread > 0): same bits for every setting ─────
# Meaningful with several threads (`julia -t 4`); with one thread the serial passes run and the comparisons are trivial.

@testset "threaded: every neurons_per_thread == serial == old kernels (nthreads = $(Threads.nthreads()))" begin
    old_min = SNNModels.SYN_MIN_OPS[]
    SNNModels.SYN_MIN_OPS[] = 0          # thread even the small synapses of the test
    try
        for FT in (Float32, Float64), param in (MultiplicativeNorm(τ = 10ms), AdditiveNorm(τ = 10ms))
            E, P, EE, PE = _norm_synapses(FT; N = 1500, Npre = 400, seed = 5)
            serial_new, old = _one_step_pair(param, deepcopy([EE, PE]))
            for npt in (128, 256, 512, 1024)
                syns = deepcopy([EE, PE])
                new = SynapseNormalization(syns; param)
                with_neurons_per_thread(npt) do
                    # the same perturbation as in `_one_step_pair`, from its seed
                    Random.seed!(21)
                    for a in syns
                        a.W .*= FT(0.7) .+ FT(0.6) .* rand(FT, length(a.W))
                    end
                    if Threads.nthreads() > 1
                        @test SNNModels._norm_chunk_ptr(syns[1], Int(E.N)) isa Matrix{Int32}   # the threaded passes run
                    end
                    SNNModels.plasticity!(new, param)
                end
                @test _same(new, serial_new) && _same(new, old)
            end
        end
    finally
        SNNModels.SYN_MIN_OPS[] = old_min
    end
end

@testset "threaded train!: STDP + normalisation == old kernels, serial run" begin
    old_min = SNNModels.SYN_MIN_OPS[]
    SNNModels.SYN_MIN_OPS[] = 0
    try
        for additive in (false, true)
            _, EEo, PEo, no = _trained_net(Float32, OldNorm(τ = 2ms, additive = additive))
            for npt in (128, 512)
                new_param = additive ? AdditiveNorm(τ = 2ms) : MultiplicativeNorm(τ = 2ms)
                _, EEn, PEn, nn = with_neurons_per_thread(() -> _trained_net(Float32, new_param), npt)
                @test EEn.W == EEo.W && PEn.W == PEo.W && nn.W1 == no.W1 && nn.μ == no.μ
            end
        end
    finally
        SNNModels.SYN_MIN_OPS[] = old_min
    end
end

@testset "threading off: serial passes, no pointer table" begin
    E, P, EE, PE = _norm_synapses(Float32; N = 1500, Npre = 400, seed = 5)
    norm = SynapseNormalization([EE, PE]; param = MultiplicativeNorm(τ = 10ms))
    builds = EE.chunkptr.builds
    with_neurons_per_thread(0) do
        @test SNNModels._norm_chunk_ptr(EE, Int(E.N)) === nothing
        SNNModels.plasticity!(norm, norm.param)
    end
    @test EE.chunkptr.builds == builds
end

@testset "threaded after rewiring: the cached pointer table follows the connectivity" begin
    old_min = SNNModels.SYN_MIN_OPS[]
    SNNModels.SYN_MIN_OPS[] = 0
    try
        E, P, EE, PE = _norm_synapses(Float32; N = 1500, Npre = 400, seed = 6)
        norm = SynapseNormalization([EE]; param = MultiplicativeNorm(τ = 10ms))
        with_neurons_per_thread(256) do
            SNNModels.plasticity!(norm, norm.param)             # builds the table
            Random.seed!(4)
            synaptic_turnover!(EE; p_rewire = 0.2)
            twin = deepcopy(norm)
            EE.W .*= 1.3f0; twin.synapses[1].W .*= 1.3f0
            SNNModels.plasticity!(norm, norm.param)
            old_norm_step!(twin, false)
            @test _same(norm, twin)
        end
    finally
        SNNModels.SYN_MIN_OPS[] = old_min
    end
end
