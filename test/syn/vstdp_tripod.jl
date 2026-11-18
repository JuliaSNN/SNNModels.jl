#=
Voltage-based STDP on the dendrites of a Tripod: a `SpikingSynapse` onto a compartment reads that
compartment's potential as `v_post`; a simulated Tripod with voltage STDP on both dendrites and one
multiplicative normalisation per dendritic synapse gives the same weights with the event-driven and
the dense rule.
=#

using SNNModels
using Test
using Random
@load_units

# voltage STDP with dendritic thresholds: potentiation above -15 mV (dendritic depolarisation), bounds 0.5-1.5
const _DEND_VSTDP = (A_LTD = 2.0e-4, A_LTP = 6.0e-4, θ_LTD = -69.0, θ_LTP = -15.0, τu = 35.0, τv = 35.0, delay_uv = 1.0,
                     Wmin = 0.5, Wmax = 1.5)

@testset "vSTDP on Tripod dendrites: the synapse reads the compartment potential" begin
    T = Tripod(N = 2)
    pre = IF(N = 4)
    for comp in (:s, :d1, :d2)
        syn = SpikingSynapse(pre, T, :glu, comp; conn = fill(1.0f0, 2, 4), LTPParam = vSTDPParameter(; _DEND_VSTDP...))
        @test syn.v_post === getfield(T, Symbol("v_", comp))
    end
    B = BallAndStick(N = 2)
    syn = SpikingSynapse(pre, B, :glu, :d; conn = fill(1.0f0, 2, 4), LTPParam = vSTDPDenseParameter(; _DEND_VSTDP...))
    @test syn.v_post === B.v_d
end

@testset "vSTDP on Tripod dendrites: simulated Tripod, normalisation per dendrite" begin
    # 50 Poisson inputs at 10 Hz on each dendrite: the dendrites reach the LTP threshold (-15 mV) and the
    # soma fires. On d1 an event-driven and a dense synapse receive the same spikes and read the same
    # potential (they both drive the dendrite, with equal weights).
    Random.seed!(3)
    N = 50
    T = Tripod(N = 1)
    E1 = SNNModels.Poisson(N = N, param = SNNModels.PoissonParameter(10Hz))   # qualified: Distributions.Poisson may be loaded
    E2 = SNNModels.Poisson(N = N, param = SNNModels.PoissonParameter(10Hz))
    s1e = SpikingSynapse(E1, T, :glu, :d1; conn = fill(1.0f0, 1, N), LTPParam = vSTDPParameter(; _DEND_VSTDP...))
    s1d = SpikingSynapse(E1, T, :glu, :d1; conn = fill(1.0f0, 1, N), LTPParam = vSTDPDenseParameter(; _DEND_VSTDP...))
    s2 = SpikingSynapse(E2, T, :glu, :d2; conn = fill(1.0f0, 1, N), LTPParam = vSTDPParameter(; _DEND_VSTDP...))

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
