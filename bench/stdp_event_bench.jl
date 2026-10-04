# Micro-benchmark of the STDP plasticity step (not part of the test suite).
#
# Usage (single thread, any env that develops SNNModels):
#   julia --project=<env> -t 1 bench/stdp_event_bench.jl
# Runs on both the old clock-driven code (dev @ 614cfca) and the event-driven
# rewrite, so the same script gives the before/after numbers in claude/performance.md.
#
# Network: one Identity population of 4000 neurons, recurrent SpikingSynapse with
# p = 0.02 (about 3.2e5 synapses), dt = 0.125 ms. Spikes are forced: Bernoulli
# fire vectors at a fixed rate, pre-generated so the timing is the plasticity only.

using SNNModels
using Random
using Printf
@load_units

const N = 4000
const DT = 0.125f0
const STEPS = 10_000   # 1.25 s of simulated time

function fire_masks(rate_Hz, steps; seed = 1)
    Random.seed!(seed)
    p = rate_Hz * DT * 1.0f-3
    return rand(Float32, N, steps) .< p
end

function bench_plasticity(param, masks; reps = 3)
    Random.seed!(2)
    pop = Identity(N = N)
    syn = SpikingSynapse(pop, pop, :g; conn = (p = 0.02f0, μ = 5.0f0, σ = 0.5f0), LTPParam = param)
    T = Time()
    # warm-up (compilation)
    for n = 1:10
        pop.fire .= view(masks, :, n)
        update_time!(T, DT)
        plasticity!(syn, syn.param, DT, T)
    end
    best = Inf
    for _ = 1:reps
        t = @elapsed for n in axes(masks, 2)
            pop.fire .= view(masks, :, n)
            update_time!(T, DT)
            plasticity!(syn, syn.param, DT, T)
        end
        best = min(best, t)
    end
    # cost of the mask copy alone, subtracted from the result
    tcopy = @elapsed for n in axes(masks, 2)
        pop.fire .= view(masks, :, n)
        update_time!(T, DT)
    end
    return (best - tcopy), length(syn.W)
end

function bench_train(param, rate_Hz)
    Random.seed!(3)
    inp = Poisson(N = N, param = PoissonParameter(rate_Hz))
    post = IF(N = N, param = IFParameter())
    syn = SpikingSynapse(inp, post, :ge; conn = (p = 0.02f0, μ = 0.5f0), LTPParam = param)
    model = compose(inp = inp, post = post, syn = syn, silent = true)
    train!(model = model, duration = 50ms, dt = DT)   # warm-up
    t = @elapsed train!(model = model, duration = 1000ms, dt = DT)
    return t
end

rules = Any[("STDPGerstner", STDPGerstner(A_pre = 1.0f-3, A_post = -1.0f-3, Wmax = 10.0f0))]
isdefined(SNNModels, :STDPTriplet) &&
    push!(rules, ("STDPTriplet", getfield(SNNModels, :STDPTriplet)(Wmax = 10.0f0)))
isdefined(SNNModels, :STDPWeightDependent) &&
    push!(rules, ("STDPWeightDependent", getfield(SNNModels, :STDPWeightDependent)(Wmax = 10.0f0)))
push!(rules, ("STDPMexicanHat", STDPMexicanHat(Wmax = 10.0f0)))

println("threads = ", Threads.nthreads(), ", event-driven = ", isdefined(SNNModels, :STDPTriplet))
for rate in (5, 10, 20)
    masks = fire_masks(rate, STEPS)
    for (name, p) in rules
        t, nnz = bench_plasticity(p, masks)
        @printf("plasticity!  %-20s rate %3d Hz  nnz %d  %8.2f us/step\n",
                name, rate, nnz, 1e6 * t / STEPS)
    end
end
for rate in (5, 10, 20)
    t = bench_train(rules[1][2], rate * 1.0f0Hz)
    @printf("train! 1 s   %-20s input %3d Hz  %8.3f s\n", rules[1][1], rate, t)
    t = bench_train(SNNModels.NoLTP(), rate * 1.0f0Hz)
    @printf("train! 1 s   %-20s input %3d Hz  %8.3f s\n", "NoLTP", rate, t)
end
