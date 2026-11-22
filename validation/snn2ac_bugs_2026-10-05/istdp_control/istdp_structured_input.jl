# iSTDPRate fixed vs buggy under spatially structured external input.
# Run the same file in two environments (registered SNNModels 1.8.2 = fixed;
# branch control/istdp-bug-1.8.2 = buggy). Set LABEL before `include`.
# Profiles over excitatory index i = 1..800 (constant external current, pA):
#   :ensembles  two localized groups (151:250 and 551:650) at 40 pA, rest 0
#   :gradient   linear ramp 0 -> 40 pA along the index
# 800 E / 200 I conductance-based IF network, Poisson background, iSTDPRate on I->E
# (target 5 Hz), 20 s of train!, rates measured over the last 5 s.
using SNNModels, Random, Statistics, JLD2
@load_units

const NE, NI = 800, 200
const AMP = 40.0f0
profile(kind) = kind === :ensembles ? Float32[(151 <= i <= 250 || 551 <= i <= 650) ? AMP : 0 for i = 1:NE] :
                kind === :gradient  ? Float32[AMP * (i - 1) / (NE - 1) for i = 1:NE] : zeros(Float32, NE)

function build(Iext; plastic = true, seed = 3)
    Random.seed!(seed)
    E = IF(N = NE); Ipop = IF(N = NI)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.1f0, μ = 2.0f0))
    EI = SpikingSynapse(E, Ipop, :ge; conn = (p = 0.1f0, μ = 2.0f0))
    ltp = plastic ? iSTDPRate(η = 0.05f0, r = 5Hz, τy = 20ms, Wmin = 0.0f0, Wmax = 200.0f0) : NoLTP()
    IE = SpikingSynapse(Ipop, E, :gi; conn = (p = 0.1f0, μ = 20.0f0), LTPParam = ltp)
    II = SpikingSynapse(Ipop, Ipop, :gi; conn = (p = 0.1f0, μ = 10.0f0))
    nE = Stimulus(PoissonLayer(8Hz, N = 400), E, :ge; conn = (p = 0.1f0, μ = 3.0f0))
    nI = Stimulus(PoissonLayer(8Hz, N = 400), Ipop, :ge; conn = (p = 0.1f0, μ = 3.0f0))
    E.I .= Iext
    model = compose(; E, Ipop, EE, EI, IE, II, nE, nI, silent = true)
    monitor!(E, [:fire]; monitor_rate = 30Hz); monitor!(Ipop, [:fire]; monitor_rate = 60Hz)
    return model, E, Ipop, IE
end

rates(pop, t0, t1) = [count(t -> t0 < t <= t1, s) / ((t1 - t0) / 1000) for s in spiketimes(pop)]
# mean incoming inhibitory weight per E neuron, and mean outgoing weight per I neuron
in_w(IE)  = [mean(IE.W[IE.I .== i]) for i = 1:NE]
out_w(IE) = [mean(IE.W[IE.colptr[j]:IE.colptr[j+1]-1]) for j = 1:NI]

const DUR = 20s
results = Dict{String,Any}()
for kind in (:ensembles, :gradient)
    Iext = profile(kind)
    runs = LABEL == "fixed" ? ((true, "plastic"), (false, "static")) : ((true, "plastic"),)
    for (plastic, tag) in runs
        model, E, Ipop, IE = build(Iext; plastic)
        train!(model = model, duration = DUR, dt = 0.1f0, pbar = false)
        st = spiketimes(E)
        tc = [[count(t -> (k - 1) * 1000f0 < t <= k * 1000f0, s) for s in st] for k = 1:20]   # spikes per 1 s bin
        results["$(kind)_$(tag)"] = (; Iext, rE = rates(E, 15000f0, 20000f0), rI = rates(Ipop, 15000f0, 20000f0),
                                       win = in_w(IE), wout = out_w(IE), W = copy(IE.W), I = copy(IE.I), J = copy(IE.J),
                                       tc = reduce(hcat, tc))
        r = results["$(kind)_$(tag)"]
        println(LABEL, " ", kind, " ", tag, ": E ", round(mean(r.rE), digits = 2), " Hz (sd ", round(std(r.rE), digits = 2),
                "), corr(rate, input) = ", round(cor(r.rE, Iext), digits = 3))
    end
end
@save joinpath(@__DIR__, "structured_$(LABEL).jld2") results
