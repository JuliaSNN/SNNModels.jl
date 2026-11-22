# Micro-benchmark of one SynapseNormalization step (MultiplicativeNorm), not part of the test suite.
#
# Usage (any env that develops SNNModels), one process per thread count:
#   for t in 1 2 4 6; do julia --project=<env> -t $t bench/normalisation_bench.jl; done
#
# Cases: 4800 -> 4800 neurons, p = 0.2 (4.6e6 synapses) and 8000 -> 8000, p = 0.39 (2.5e7 synapses), one synapse.
# Kernels: the row-by-row gather through the transposed index of SNNModels <= 2.3.0 (test-only copy in
# test/syn/normalization_old_kernels.jl), the CSC-order passes on one thread (neurons_per_thread = 0) and the threaded
# CSC passes with neurons_per_thread = 512 and 1024. Every kernel must leave the weights bit-identical to the old one.
#
# Output: one line per (case, kernel): ms per step and ns per synapse (minimum over the repetitions; the weights are
# restored before each step, outside the timing), the build time of the sub-row pointer table (first threaded step, cached).

using SNNModels
using Random
using Printf
@load_units

include(joinpath(@__DIR__, "..", "test", "syn", "normalization_old_kernels.jl"))

const REPS = 7

# ── Setup ─────────────────────────────────────────────────────────────────────

# One synapse `N -> N` with connection probability `p`, a normalization on it, and perturbed weights (so the step works).
function build_case(N, p; seed = 1)
    Random.seed!(seed)
    pre = Identity(N = N, name = "pre")
    post = Identity(N = N, name = "post")
    syn = SpikingSynapse(pre, post, :g; conn = (p = p, μ = 2.0, σ = 0.5), name = "syn")
    norm = SynapseNormalization([syn]; param = MultiplicativeNorm(τ = 20ms))
    syn.W .*= 0.5f0 .+ rand(Float32, length(syn.W))
    return syn, norm
end

# ── Timing ────────────────────────────────────────────────────────────────────

# Minimum time (s) of `step!()`, the weights restored to `W0` before each call (not timed); the weights after one step.
function time_step(step!, syn, W0)
    copyto!(syn.W, W0)
    step!()                                     # warm-up: compilation, pointer table
    best = Inf
    for _ in 1:REPS
        copyto!(syn.W, W0)
        best = min(best, @elapsed step!())
    end
    copyto!(syn.W, W0)
    step!()
    return best, copy(syn.W)
end

function report(case, label, t, nsyn, Wref, W; build = NaN)
    @printf("%-10s  t=%d  %-22s  %8.2f ms  %6.2f ns/syn  identical=%s  ptr build %s\n",
            case, Threads.nthreads(), label, t * 1e3, t / nsyn * 1e9, W == Wref,
            isnan(build) ? "-" : @sprintf("%.1f ms", build * 1e3))
end

# ── Run ───────────────────────────────────────────────────────────────────────

for (case, N, p) in (("4.6M", 4800, 0.2), ("25M", 8000, 0.39))
    syn, norm = build_case(N, p)
    W0 = copy(syn.W)
    nsyn = length(W0)
    @printf("%s: %d synapses\n", case, nsyn)

    # reference: old kernels (they build the transpose once, outside the timing)
    t_old, Wref = time_step(() -> old_norm_step!(norm, false), syn, W0)
    report(case, "old (row gather)", t_old, nsyn, Wref, Wref)

    # CSC passes, threading off
    t_csc, W = with_neurons_per_thread(0) do
        time_step(() -> SNNModels.plasticity!(norm, norm.param), syn, W0)
    end
    report(case, "CSC serial", t_csc, nsyn, Wref, W)

    # CSC passes, threaded (with one thread the serial passes run)
    for npt in (512, 1024)
        t_thr, W = with_neurons_per_thread(npt) do
            time_step(() -> SNNModels.plasticity!(norm, norm.param), syn, W0)
        end
        report(case, "CSC npt=$npt", t_thr, nsyn, Wref, W; build = syn.chunkptr.buildtime)
    end
end
