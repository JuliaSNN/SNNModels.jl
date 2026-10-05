# Regression tests for the bugs found by the documentation sweep (simulation loop level).
using SNNModels
using Test, Random, Statistics
@load_units

@testset "train! with IZ, HH, MorrisLecar and without connections" begin
    for P in (IZ(N = 3, param = IZParameter()), HH(N = 3), MorrisLecar(N = 3), IF(N = 3), AdEx(N = 3))
        @test train!([P]; duration = 5ms) isa SNNModels.Time
        @test sim!([P]; duration = 5ms) isa SNNModels.Time
    end
    E = IF(N = 5)
    @test train!([E], [EmptySynapse()]; duration = 5ms) isa SNNModels.Time
end

@testset "FORCE / PINning connections run under sim! and train!" begin
    Random.seed!(1)
    R = Rate(N = 100)
    F = FLSynapse(R, R; μ = 1.5, α = 1)
    dt = 0.125f0
    T = SNNModels.Time()
    err = Float32[]
    for k = 1:4000
        F.f = sin(2π * k * dt / 20)
        train!([R], [F], SNNModels.AbstractStimulus[], dt, T)
        push!(err, abs(F.z - F.f))
    end
    @test mean(err[end-400:end]) < mean(err[1:400])
    @test sim!([R], [F]; duration = 5ms) isa SNNModels.Time
    for C in (
        SNNModels.FLSparseSynapse(R, R; μ = 1.5, p = 0.2),
        PINningSynapse(R, R),
        SNNModels.PINningSparseSynapse(R, R; p = 0.2),
    )
        @test sim!([R], [C]; duration = 5ms) isa SNNModels.Time
        @test train!([R], [C]; duration = 5ms) isa SNNModels.Time
    end
    @test_throws ArgumentError SNNModels.FLSparseSynapse(R, R)
end

@testset "MorrisLecar, ExtendedIF, WilsonCowan receive connections" begin
    E = SNNModels.Poisson(N = 50, param = PoissonParameter(50Hz))
    M = MorrisLecar(N = 5)
    X = ExtendedIF(N = 5)
    sM = SpikingSynapse(E, M, :ge; conn = (p = 1.0, μ = 1.0))
    sX = SpikingSynapse(E, X, :glu; conn = (p = 1.0, μ = 1.0))
    sX2 = SpikingSynapse(E, X, :g_SST; conn = (p = 1.0, μ = 1.0))
    @test sM.g === M.ge
    @test sX.g === X.g_Exc && sX2.g === X.g_SST
    monitor!(M, [:ge]); monitor!(X, [:g_Exc])
    sim!([E, M, X], [sM, sX, sX2]; duration = 50ms)
    @test maximum(getvariable(M, :ge)) > 0
    @test maximum(getvariable(X, :g_Exc)) > 0
    @test_throws ArgumentError SpikingSynapse(E, M, :g_SST; conn = (p = 1.0, μ = 1.0))
    R = Rate(N = 10)
    W = WilsonCowan(N = 10)
    RW = RateSynapse(R, W; μ = 1.0, p = 0.5)
    @test RW.g === W.g
    @test sim!([R, W], [RW]; duration = 5ms) isa SNNModels.Time
end

@testset "HH and MorrisLecar: one spike flag per action potential" begin
    function flags_and_peaks(P, I, dt, T)
        P.I .= I
        nflag = 0; npeak = 0; vprev = P.v[1]; up = false
        for _ = 1:round(Int, T / dt)
            SNNModels.integrate!(P, P.param, Float32(dt))
            nflag += P.fire[1]
            # count action potentials as local maxima above the detection threshold
            if P.v[1] < vprev && up && vprev > (P isa HH ? -20.0f0 : 20.0f0)
                npeak += 1
            end
            up = P.v[1] > vprev
            vprev = P.v[1]
        end
        nflag, npeak
    end
    H = HH(N = 1); H.ge .= 0; H.gi .= 0
    nf, np = flags_and_peaks(H, 500.0f0, 0.01f0, 200.0f0)
    @test np > 2 && nf == np
    # Default MorrisLecar fires once and then stays depolarised under a constant current:
    # use three current pulses separated by pauses.
    M = MorrisLecar(N = 1)
    nf = 0; np = 0
    for _ = 1:3
        a, b = flags_and_peaks(M, 100.0f0, 0.05f0, 50.0f0); nf += a; np += b
        a, b = flags_and_peaks(M, 0.0f0, 0.05f0, 300.0f0); nf += a; np += b
    end
    @test nf == 3 # one action potential per pulse (peak > 50 mV)
    @test SNNModels.MorrisLecar_w_nullcline(0.0f0, M.param) ≈ 0.5f0
end

@testset "Rate input g is reset every step; RateSynapse/SpikeRateSynapse" begin
    R = Rate(N = 20)
    RR = RateSynapse(R, R; μ = 1.0, p = 1.0)
    T = SNNModels.Time()
    for _ = 1:10
        sim!([R], [RR], SNNModels.AbstractStimulus[], 0.125f0, T)
    end
    # after a step, g holds only the input of that step: W r
    Wm = zeros(Float32, 20, 20)
    for j = 1:20, s = RR.colptr[j]:(RR.colptr[j+1]-1)
        Wm[RR.I[s], j] = RR.W[s]
    end
    @test R.g ≈ Wm * R.r
    @test_throws ArgumentError RateSynapse(R, R)
    # a spike makes x jump by W (delta input), independently of dt
    for dt in (0.125f0, 0.05f0)
        E = Identity(N = 1)
        Q = Rate(N = 1); Q.x .= 0; Q.r .= 0
        S = SNNModels.SpikeRateSynapse(E, Q; μ = 1.0, p = 1.0)
        S.W .= 0.5f0
        E.fire .= true
        SNNModels.forward!(S, S.param, dt, SNNModels.Time())
        SNNModels.integrate!(Q, Q.param, dt)
        @test Q.x[1] ≈ 0.5f0
        @test Q.g[1] == 0
    end
    E = SNNModels.Poisson(N = 10, param = PoissonParameter(50Hz))
    S = SNNModels.SpikeRateSynapse(E, R; μ = 1.0, p = 0.5)
    @test train!([E, R], [S]; duration = 10ms) isa SNNModels.Time
end

@testset "Confavreux2025Synapse: a spike of weight w increments g by w" begin
    for dt in (0.125f0, 0.05f0)
        E = IF(N = 2, synapse = Confavreux2025Synapse())
        E.receptors.glu .= [2.0f0, 0.0f0]
        E.receptors.gaba .= [0.0f0, 3.0f0]
        SNNModels.update_synapses!(E, E.synapse, E.receptors, E.synvars, dt)
        @test E.synvars.gAMPA ≈ [2.0f0, 0.0f0]
        @test E.synvars.gGABA ≈ [0.0f0, 3.0f0]
    end
end

@testset "vSTDP: dt-independent rule, traces start at the membrane potential" begin
    function vstdp_dw(dt; Tsim = 30.0f0, vpost = -40.0f0)
        E = Identity(N = 1)
        P = AdEx(N = 1)
        syn = SpikingSynapse(E, P, :glu; conn = (p = 1.0, μ = 1.0), LTPParam = vSTDPParameter())
        W0 = copy(syn.W)
        T = SNNModels.Time()
        for k = 1:round(Int, Tsim / dt)
            E.fire .= (k == 1)
            P.v .= vpost
            SNNModels.plasticity!(syn, syn.LTPParam, syn.LTPVars, Float32(dt), T)
        end
        syn.W[1] - W0[1], syn.LTPVars.x[1]
    end
    dw1, x1 = vstdp_dw(0.125)
    dw2, x2 = vstdp_dw(0.0625)
    @test dw1 > 0 && isapprox(dw1, dw2; rtol = 0.05)
    @test isapprox(x1, x2; rtol = 0.05)
    # silent target at rest: no spurious LTD from 0 mV initial traces
    dw_rest, _ = vstdp_dw(0.125; vpost = -70.6f0)
    @test dw_rest == 0
end

@testset "AggregateScaling: rate estimate in 1/ms, dt-independent" begin
    function run_as(dt; rate_period = 10.0f0, Tsim = 1000.0f0)
        P = Identity(N = 5)
        E = IF(N = 2)
        S = SpikingSynapse(P, E, :ge; conn = (p = 1.0, μ = 2.0))
        A = AggregateScaling(E, [S]; param = AggregateScalingParameter(E.N, 50Hz))
        @test A.N == 2
        T = SNNModels.Time()
        nper = round(Int, rate_period / dt)
        for k = 1:round(Int, Tsim / dt)
            update_time!(T, Float32(dt))
            E.fire .= (k % nper == 0)          # 100 Hz
            SNNModels.forward!(A, A.param, Float32(dt), T)
            SNNModels.plasticity!(A, A.param, Float32(dt), T)
        end
        A.y[1], A.WT[1], sum(S.W[S.index[S.rowptr[1]:(S.rowptr[2]-1)]])
    end
    y1, WT1, W1 = run_as(0.125)
    y2, WT2, W2 = run_as(0.05)
    @test isapprox(y1, 0.1; rtol = 0.15) && isapprox(y2, 0.1; rtol = 0.15)  # 100 Hz in 1/ms
    @test isapprox(WT1, WT2; rtol = 0.02)
    @test WT1 < 10.0                       # rate above target: the target weight decreases
    @test isapprox(W1, WT1; rtol = 0.1)    # weights follow the target after rescaling
end

@testset "SynapseNormalization: additive and multiplicative restore the sum" begin
    for param in (AdditiveNorm(τ = 10ms), MultiplicativeNorm(τ = 10ms))
        E = IF(N = 20)
        EE = SpikingSynapse(E, E, :ge; conn = (p = 0.5, μ = 2.0))
        N = SynapseNormalization([EE]; param)
        W0 = copy(N.W0)
        EE.W .*= 1.5f0
        SNNModels.plasticity!(N, param)
        sums = zeros(Float32, 20)
        for i = 1:20, k = EE.rowptr[i]:(EE.rowptr[i+1]-1)
            sums[i] += EE.W[EE.index[k]]
        end
        @test sums ≈ W0 rtol = 1e-4
    end
    @test_throws UndefKeywordError SynapseNormalization(; synapses = [SpikingSynapse(IF(N = 2), IF(N = 2), :ge; conn = (p = 1.0, μ = 1.0))])
end

@testset "Turnover: RandomTurnover under train!, positive weights, state consistency" begin
    Random.seed!(2)
    E = IF(N = 30)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 2.0))
    TO = MetaPlasticity(RandomTurnover(τ = 5.0f0ms, threshold = 0.3f0, μ = 1.0f0), EE)
    I0 = copy(EE.I)
    EE.ρ .= range(0.1f0, 0.9f0, length = length(EE.ρ))
    @test train!([E], [EE, TO]; duration = 20ms) isa SNNModels.Time
    @test EE.I != I0
    @test all(EE.W .> 0)
    @test length(EE.ρ) == length(EE.W)
    # the default p_new works and per-synapse ρ follows its synapse
    nW = length(EE.W)
    synaptic_turnover!(EE; p_rewire = 0.5)
    @test length(EE.W) == nW == length(EE.ρ)
    # every synapse can be rewired even when free targets are scarce
    F = IF(N = 4)
    FF = SpikingSynapse(F, F, :ge; conn = (p = 0.75, μ = 1.0))
    synaptic_turnover!(FF; p_rewire = 1.0)
    @test length(FF.W) == length(FF.ρ)
end

@testset "connect!/update_sparse_matrix! keep per-synapse state; set_plasticity!" begin
    E = IF(N = 10)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 1.0))
    EE.ρ .= 0.5f0
    M = matrix(EE)
    i, j = findfirst(iszero, M - SNNModels.spdiagm(0 => ones(Float32, 10)) .* 0) |> Tuple
    # pick an absent synapse (i, j), i != j
    i, j = first((a, b) for a = 1:10, b = 1:10 if a != b && M[a, b] == 0)
    n0 = length(EE.W)
    connect!(EE, j, i, 3.0f0)
    @test length(EE.W) == length(EE.ρ) == n0 + 1
    @test matrix(EE)[i, j] == 3.0f0
    @test count(==(1.0f0), EE.ρ) == 1 && count(==(0.5f0), EE.ρ) == n0
    sim!([E], [EE]; duration = 5ms)
    # rebuilding from I, J, W keeps the matrix size even if the last neuron has no synapse
    G = IF(N = 5)
    w = SNNModels.sparse(Int32[1, 2], Int32[1, 1], Float32[1, 2], 5, 5)
    GG = SpikingSynapse(G, G, :ge; conn = w)
    update_sparse_matrix!(GG)
    @test length(GG.rowptr) == 6 && length(GG.colptr) == 6
    # set_plasticity!/has_plasticity on SpikingSynapse
    P = SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 1.0), LTPParam = STDPGerstner())
    @test has_plasticity(P)
    set_plasticity!(P, false)
    @test !has_plasticity(P)
    @test !has_plasticity(EE)
end
