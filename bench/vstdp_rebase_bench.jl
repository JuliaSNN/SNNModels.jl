# Micro-benchmark of the epoch restart (rebase) of the event-driven vSTDP and of a SynapseNormalization step on a vSTDP
# synapse (the step materialises the lazy weights first), not part of the test suite.
#
# Usage (any env that develops SNNModels), one process per thread count:
#   for t in 1 2 4 6; do julia --project=<env> -t $t bench/vstdp_rebase_bench.jl; done
#
# Cases: 4800 -> 4800 neurons, p = 0.2 (4.6e6 synapses) and 8000 -> 8000, p = 0.2 (1.28e7 synapses), Float32, one synapse
# with a random pending state (amplitudes, accumulator, snapshots).
# Kernels: the serial rebase of SNNModels <= 2.3.0 (test-only copy in test/syn/vstdp_rebase_old_kernel.jl) and the library
# rebase with neurons_per_thread = 0 (off), 512 and 1024; a normalisation step with the old rebase followed by the library
# normalisation kernels, and the library step (new rebase and kernels). Every kernel must leave the state bit-identical to
# the old one.
#
# Output: one line per (case, kernel): ms per call and ns per synapse (minimum over the repetitions; the state is restored
# before each call, outside the timing).

using SNNModels
using Random
using Printf
@load_units

include(joinpath(@__DIR__, "..", "test", "syn", "vstdp_rebase_old_kernel.jl"))

const REPS = 7

# ── Setup ─────────────────────────────────────────────────────────────────────

# One vSTDP synapse `N -> N` with connection probability `p`, a normalisation on it, and a random pending state.
function build_case(N, p; seed = 1)
    Random.seed!(seed)
    pre = Identity(N = N, name = "pre")
    post = Identity(N = N, name = "post")
    syn = SpikingSynapse(pre, post, :g; conn = (p = p, μ = 2.0, σ = 0.5), LTPParam = vSTDPParameter(), name = "syn")
    norm = SynapseNormalization([syn]; param = MultiplicativeNorm(τ = 20ms))
    v = syn.LTPVars
    SNNModels._ev_check!(syn, v, syn.LTPParam, 0.125f0)
    v.x .= 0.1 .* rand(N)
    v.K .= 100 .* rand(N)
    v.snap .= rand(length(syn.W)) .* v.K[syn.I]
    v.steps[1] = 100
    v.pw[1] = 0.7; v.pw[2] = 1 / 0.7
    return syn, norm
end

# Everything the rebase and the normalisation write.
snapshot(syn) = (copy(syn.W), copy(syn.LTPVars.x), copy(syn.LTPVars.K), copy(syn.LTPVars.snap),
                 copy(syn.LTPVars.pw), copy(syn.LTPVars.steps))
restore!(syn, s) = foreach(copyto!, (syn.W, syn.LTPVars.x, syn.LTPVars.K, syn.LTPVars.snap, syn.LTPVars.pw, syn.LTPVars.steps), s)

# The library normalisation kernels without the rebase: `sync_weights!` is a no-op while `cache[1]` is NaN (bench only).
function norm_kernels_only!(syn, norm)
    dt = syn.LTPVars.cache[1]
    syn.LTPVars.cache[1] = NaN
    SNNModels.plasticity!(norm, norm.param)
    syn.LTPVars.cache[1] = dt
end

# ── Timing ────────────────────────────────────────────────────────────────────

# Minimum time (s) of `step!()`, the state restored to `s0` before each call (not timed); the state after one call.
function time_step(step!, syn, s0)
    restore!(syn, s0)
    step!()                                     # warm-up: compilation
    best = Inf
    for _ in 1:REPS
        restore!(syn, s0)
        best = min(best, @elapsed step!())
    end
    restore!(syn, s0)
    step!()
    return best, snapshot(syn)
end

function report(case, label, t, nsyn, ref, s)
    @printf("%-6s  t=%d  %-30s  %8.2f ms  %5.2f ns/syn  identical=%s\n",
            case, Threads.nthreads(), label, t * 1e3, t / nsyn * 1e9, s == ref)
end

# ── Run ───────────────────────────────────────────────────────────────────────

for (case, N, p) in (("4.6M", 4800, 0.2), ("12.8M", 8000, 0.2))
    syn, norm = build_case(N, p)
    s0 = snapshot(syn)
    nsyn = length(syn.W)
    @printf("%s: %d synapses\n", case, nsyn)

    # rebase alone
    t_old, ref = time_step(() -> old_ev_rebase!(syn, syn.LTPVars), syn, s0)
    report(case, "rebase old (serial)", t_old, nsyn, ref, ref)
    for npt in (0, 512, 1024)
        t, s = with_neurons_per_thread(npt) do
            time_step(() -> SNNModels._ev_rebase!(syn, syn.LTPVars), syn, s0)
        end
        report(case, "rebase new npt=$npt", t, nsyn, ref, s)
    end

    # normalisation step: old rebase + library kernels, then the library step
    for npt in (0, 512, 1024)
        t_o, ref_n = with_neurons_per_thread(npt) do
            time_step(() -> (old_ev_rebase!(syn, syn.LTPVars); norm_kernels_only!(syn, norm)), syn, s0)
        end
        report(case, "norm, old rebase npt=$npt", t_o, nsyn, ref_n, ref_n)
        t_n, s = with_neurons_per_thread(npt) do
            time_step(() -> SNNModels.plasticity!(norm, norm.param), syn, s0)
        end
        report(case, "norm, new rebase npt=$npt", t_n, nsyn, ref_n, s)
    end
end
