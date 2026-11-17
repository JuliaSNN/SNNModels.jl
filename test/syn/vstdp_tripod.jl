#=
Voltage-based STDP on the dendrites of a Tripod
===============================================

A `SpikingSynapse` onto a compartment of a `Tripod` (or `BallAndStick`) reads the potential of that
compartment as `v_post`, so `vSTDPParameter` / `vSTDPDenseParameter` act on the local dendritic potential
(Bono & Clopath 2017; the Tripod plasticity code). Tests:
  - `v_post` is the potential of the targeted compartment;
  - with prescribed compartment potentials, the `bono_vstdp` preset on :d1 and :d2 equals the update of
    the Tripod code computed on the potential of that dendrite (the soma potential plays no role);
  - a simulated Tripod with Poisson inputs on both dendrites, `bono_vstdp` and one multiplicative
    normalisation per dendritic synapse: event-driven and dense rules give the same weights.
=#

using SNNModels
using Test
using Random
using Statistics
@load_units

include(joinpath(@__DIR__, "plasticity_presets_ref.jl"))

@testset "vSTDP on Tripod dendrites: the synapse reads the compartment potential" begin
    T = Tripod(N = 2)
    pre = IF(N = 4)
    for comp in (:s, :d1, :d2)
        syn = SpikingSynapse(pre, T, :glu, comp; conn = fill(1.0f0, 2, 4), LTPParam = bono_vstdp())
        @test syn.v_post === getfield(T, Symbol("v_", comp))
    end
    B = BallAndStick(N = 2)
    syn = SpikingSynapse(pre, B, :glu, :d; conn = fill(1.0f0, 2, 4), LTPParam = bono_vstdp(dense = true))
    @test syn.v_post === B.v_d
end

@testset "vSTDP on Tripod dendrites: prescribed potentials, equal to the Tripod code" begin
    npre, npost = 20, 3
    spikes, V1 = _pp_inputs(; npre, npost, seed = 21)
    _, V2 = _pp_inputs(; npre, npost, seed = 22)
    set = SNNModels._BONO_VSTDP
    D = round(Int, set.ε / _PDT)

    for dense in (false, true)
        T = Tripod(N = npost)
        pre = IF(N = npre)
        rule = bono_vstdp(; dense, Wmin = -100.0, Wmax = 100.0)
        s1 = SpikingSynapse(pre, T, :glu, :d1; conn = fill(1.0f0, npost, npre), LTPParam = rule)
        s2 = SpikingSynapse(pre, T, :glu, :d2; conn = fill(1.0f0, npost, npre), LTPParam = rule)
        ss = SpikingSynapse(pre, T, :glu, :s; conn = fill(1.0f0, npost, npre), LTPParam = rule)

        # the soma sits above every threshold: a rule reading it would potentiate at every step
        T.v_s .= 20.0f0
        _pp_run!(s1, pre, T.v_d1, spikes, V1)
        _pp_run!(s2, pre, T.v_d2, spikes, V2)
        @test all(==(20.0f0), T.v_s)

        R1 = _tripod_code_vstdp(spikes, V1, set; D, w0 = 1.0, npre)
        R2 = _tripod_code_vstdp(spikes, V2, set; D, w0 = 1.0, npre)
        W1, W2 = _pp_mat(s1), _pp_mat(s2)
        @test maximum(abs.(R1 .- 1)) > 1.0e-3 && maximum(abs.(R2 .- 1)) > 1.0e-3
        @test maximum(abs.(W1 .- R1)) < 1.0e-4 * maximum(abs.(R1 .- 1)) + 1.0e-6
        @test maximum(abs.(W2 .- R2)) < 1.0e-4 * maximum(abs.(R2 .- 1)) + 1.0e-6
        @test maximum(abs.(W1 .- W2)) > 1.0e-3     # each dendrite has its own filters

        # the same rule on the soma synapse reads the somatic potential (constant +20 mV): only potentiation
        _pp_run!(ss, pre, T.v_s, spikes, fill(20.0f0, npost, length(spikes)))
        @test minimum(_pp_mat(ss)) > 1.0
    end
end

@testset "vSTDP on Tripod dendrites: simulated Tripod, bono_vstdp, normalisation per dendrite" begin
    # 50 Poisson inputs at 10 Hz on each dendrite: the dendrites reach the LTP threshold (-15 mV) and the
    # soma fires. On d1 an event-driven and a dense synapse receive the same spikes and read the same
    # potential (they both drive the dendrite, with equal weights).
    Random.seed!(3)
    N = 50
    T = Tripod(N = 1)
    E1 = Poisson(N = N, param = PoissonParameter(10Hz))
    E2 = Poisson(N = N, param = PoissonParameter(10Hz))
    s1e = SpikingSynapse(E1, T, :glu, :d1; conn = fill(1.0f0, 1, N), LTPParam = bono_vstdp())
    s1d = SpikingSynapse(E1, T, :glu, :d1; conn = fill(1.0f0, 1, N), LTPParam = bono_vstdp(dense = true))
    s2 = SpikingSynapse(E2, T, :glu, :d2; conn = fill(1.0f0, 1, N), LTPParam = bono_vstdp())

    # one multiplicative normalisation per dendritic synapse, every 20 ms (the Tripod code: every 200 steps)
    n1e = SynapseNormalization([s1e]; param = MultiplicativeNorm(τ = 20ms))
    n1d = SynapseNormalization([s1d]; param = MultiplicativeNorm(τ = 20ms))
    n2 = SynapseNormalization([s2]; param = MultiplicativeNorm(τ = 20ms))
    monitor!(T, [:v_d1, :v_d2, :fire])
    model = compose(; T, E1, E2, s1e, s1d, s2, n1e, n1d, n2, silent = true)
    train!(model = model, duration = 2000ms, dt = 0.1ms)

    vd1, vd2 = getvariable(T, :v_d1), getvariable(T, :v_d2)
    @test count(>(-15.0f0), vd1) > 0 && count(>(-15.0f0), vd2) > 0      # LTP region reached on both dendrites
    @test length(spiketimes(T)[1]) > 0

    We, Wd, W2 = plasticity_value(s1e, :W), copy(s1d.W), plasticity_value(s2, :W)
    @test maximum(We) - minimum(We) > 0.1                                # the weights moved apart
    @test maximum(abs.(We .- Wd)) < 1.0e-5                               # event-driven == dense
    @test abs(sum(We) - N) < 0.02 * N && abs(sum(W2) - N) < 0.02 * N     # summed weight per dendrite kept
    @test maximum(abs.(We .- W2)) > 1.0e-3                               # the dendrites learn separately
end
