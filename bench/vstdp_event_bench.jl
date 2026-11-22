# Micro-benchmark of the per-synapse loops of the event-driven vSTDP and of a whole train! step, not part of the test suite.
#
# Usage (any env that develops SNNModels), one thread:
#   julia --project=<env> -t 1 bench/vstdp_event_bench.jl
#
# Loops: the loops of SNNModels <= 2.3.0 (test-only copies in test/syn/vstdp_ivdep_old_kernel.jl) against the library ones,
# on one synapse 4800 -> 4800, p = 0.2 (4.6e6 synapses, about 960 per presynaptic neuron) with a random pending state,
# Float32 and Float64:
#   current_weights   all synapses (plasticity_value(syn, :W))
#   _ev_spikes!       outgoing synapses of 10 and 100 presynaptic spikes per call (400 different random sets)
# Step: train! of 4800 AdEx neurons driven by Poisson input (about 5 Hz) with vSTDP on E -> E (p = 0.2), with the library
# rule and with `OldKernelvSTDP` (old loops), the two runs alternated.
#
# Output: minimum over 7 repetitions; ns per synapse for the loops, us per step for train!; `identical` compares the
# results of the two versions.

using SNNModels
using Random
using Printf
@load_units

include(joinpath(@__DIR__, "..", "test", "syn", "vstdp_ivdep_old_kernel.jl"))

const REPS = 7

# ── Setup ─────────────────────────────────────────────────────────────────────

# One vSTDP synapse `N -> N` with connection probability `p` and a random pending state.
function build_synapse(FT, N, p; seed = 1)
    Random.seed!(seed)
    pre = Identity(N = N, name = "pre", VFT = Vector{FT})
    post = Identity(N = N, name = "post", VFT = Vector{FT})
    syn = SpikingSynapse(pre, post, :g; conn = (p = p, μ = 2.0, σ = 0.5), LTPParam = vSTDPParameter(FT = FT), name = "syn")
    v = syn.LTPVars
    SNNModels._ev_check!(syn, v, syn.LTPParam, FT(0.125))
    v.x .= 0.1 .* rand(N)
    v.K .= 100 .* rand(N)
    v.snap .= rand(length(syn.W)) .* v.K[syn.I]
    v.u .= FT.(-80 .+ 30 .* rand(N))
    return syn
end

# 4800 AdEx neurons, Poisson input, vSTDP on E -> E with the library rule or the old loops.
function build_network(FT, old::Bool)
    Random.seed!(1)
    E = AdEx(N = 4800, name = "E", VFT = Vector{FT})
    s = PoissonStimulus(E, :ge; param = PoissonFixed(rate = 2.4kHz, μ = 1.0), name = "s")
    rule = vSTDPParameter(FT = FT, Wmax = 0.02)
    EE = SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 0.01), LTPParam = old ? OldKernelvSTDP(rule) : rule, name = "EE")
    return compose(; E, EE, s, silent = true), E, EE
end

# ── Timing ────────────────────────────────────────────────────────────────────

# Minimum over the repetitions of the mean time (s) per call of `f(k)`, k = 1:ncall (one call first: compilation).
function time_calls(f, ncall)
    f(1)
    best = Inf
    for _ in 1:REPS
        best = min(best, @elapsed(for k in 1:ncall; f(k); end) / ncall)
    end
    return best
end

# ── Loops ─────────────────────────────────────────────────────────────────────

for FT in (Float32, Float64)
    syn = build_synapse(FT, 4800, 0.2)
    v = syn.LTPVars
    nsyn = length(syn.W)
    @printf("%s: %d synapses, %.0f per presynaptic neuron\n", FT, nsyn, nsyn / 4800)

    # current_weights: the state is not modified
    Wold = similar(syn.W)
    t_old = time_calls(_ -> old_current_weights(syn, v, Wold), 1)
    t_new = time_calls(_ -> SNNModels.current_weights(syn, v), 1)
    @printf("  current_weights   old %6.3f  new %6.3f ns/syn  identical=%s\n", t_old / nsyn * 1e9, t_new / nsyn * 1e9,
            SNNModels.current_weights(syn, v) == Wold)

    # presynaptic spikes: random sets of presynaptic neurons, each call on a different set (the state drifts, not restored)
    for nspk in (10, 100)
        Random.seed!(5)
        sets = [Int32.(randperm(4800)[1:nspk]) for _ in 1:400]
        nent = sum(sum(syn.colptr[j+1] - syn.colptr[j] for j in s) for s in sets) / length(sets)
        a, b = deepcopy(syn), deepcopy(syn)
        t_old = time_calls(k -> old_ev_spikes!(a, a.LTPVars, a.LTPParam, sets[k]), 400)
        t_new = time_calls(k -> SNNModels._ev_spikes!(b, b.LTPVars, b.LTPParam, sets[k]), 400)
        @printf("  _ev_spikes! %3d    old %6.3f  new %6.3f ns/syn  (%.1f / %.1f us per call)  identical=%s\n", nspk,
                t_old / nent * 1e9, t_new / nent * 1e9, t_old * 1e6, t_new * 1e6, a.W == b.W && a.LTPVars.snap == b.LTPVars.snap)
    end
end

# ── Whole step ────────────────────────────────────────────────────────────────

for FT in (Float32, Float64)
    m_new, E, EE_new = build_network(FT, false)
    m_old, _, EE_old = build_network(FT, true)
    Random.seed!(3); train!(m_new, 100ms)
    Random.seed!(3); train!(m_old, 100ms)
    best_new = best_old = Inf
    for r in 1:REPS
        Random.seed!(10 + r); best_new = min(best_new, @elapsed train!(m_new, 1s))
        Random.seed!(10 + r); best_old = min(best_old, @elapsed train!(m_old, 1s))
    end
    nsteps = round(Int, 1s / 0.125ms)
    identical = EE_new.W == EE_old.W && plasticity_value(EE_new, :W) == old_current_weights(EE_old, EE_old.LTPVars)
    # firing rate, from a monitored run after the timing
    monitor!(E, [:fire])
    train!(m_new, 500ms)
    rate = sum(length, spiketimes(E)) / 4800 / 0.5
    @printf("%s train! (4800 AdEx, %d E->E synapses, %.1f Hz): old %.2f  new %.2f us/step (old/new %.3f)  identical=%s\n",
            FT, length(EE_new.W), rate, best_old / nsteps * 1e6, best_new / nsteps * 1e6, best_old / best_new, identical)
end
