# runtests.jl — SNNModels.jl regression suite
# Usage:  julia --project test/runtests.jl   (or Pkg.test())

using SNNModels
using Test
using SNNModels.Logging
using SNNModels.Interpolations
using Printf
@load_units

const _D = @__DIR__
const _section_times = Pair{String,Float64}[]

function _ts(label::String, relpath::String)
    t = time()
    @testset "$label" begin
        include(joinpath(_D, relpath))
    end
    push!(_section_times, label => time() - t)
end

function _inc(label::String, relpath::String)
    t = time()
    @test include(joinpath(_D, relpath))
    push!(_section_times, label => time() - t)
end

with_logger(ConsoleLogger(stderr, Logging.Error)) do

@testset "SNNModels — Full Test Suite" verbose = true begin

_suite_t0 = time()

# ── Utils ─────────────────────────────────────────────────────────────────────
@testset "Utils" verbose = true begin
    _ts("ctors",          "ctors.jl")
    _ts("macros",         "macros.jl")
    _ts("records",        "records.jl")
    _ts("record_api",     "utils/record_test.jl")
    _ts("structs",        "utils/structs_test.jl")
    _ts("util",           "utils/util_test.jl")
    _ts("io",             "utils/io_test.jl")
    _ts("sparse_matrix",  "utils/sparse_matrix_test.jl")
    _ts("sparse_gen",     "utils/sparse_matrix_gen_test.jl")
    _ts("spatial",        "utils/spatial_test.jl")
end

# ── Populations ───────────────────────────────────────────────────────────────
@testset "Populations" verbose = true begin
    _ts("parameters",   "pop/parameters.jl")
    _ts("poisson",      "pop/poisson.jl")
    _ts("dendrite",     "pop/dendrite.jl")
    _ts("spiketime",    "pop/spiketime.jl")
    _ts("morrislecar",  "pop/morrislecar.jl")
    _ts("wilsoncowan",  "pop/wilsoncowan.jl")
    _ts("hetrec",       "pop/hetrec.jl")
    _inc("if_neuron",    "pop/if_neuron.jl")
    _inc("adex_neuron",  "pop/adex_neuron.jl")
    _inc("iz_neuron",    "pop/iz_neuron.jl")
    _inc("hh_neuron",    "pop/hh_neuron.jl")
    _inc("tripod",       "pop/tripod.jl")
    _inc("ballandstick", "pop/ballandstick.jl")
    _ts("multicompartment_numerics", "pop/multicompartment_numerics.jl")
end

# ── Stimuli ───────────────────────────────────────────────────────────────────
@testset "Stimuli" verbose = true begin
    _ts("poisson",        "stim/poisson.jl")
    _ts("poisson_layer",  "stim/poisson_layer.jl")
    _ts("current",        "stim/current.jl")
    _ts("current_subset", "stim/current_subset_and_receptors.jl")
    _ts("timed",          "stim/timed.jl")
    _ts("balanced",       "stim/balanced.jl")
end

# ── Synapses ──────────────────────────────────────────────────────────────────
@testset "Synapses" verbose = true begin
    _ts("spiking_synapse",   "syn/spiking_synapse.jl")
    _ts("plasticity_params", "syn/plasticity_params.jl")
    _ts("with_plasticity",   "syn/with_plasticity.jl")
    _ts("float32",           "syn/float32.jl")
    _ts("stdp_event",        "syn/stdp_event.jl")
    _ts("istdp_kernel",      "syn/istdp_kernel.jl")
    _ts("metaplasticity",    "syn/metaplasticity.jl")
end

# ── Analysis ──────────────────────────────────────────────────────────────────
@testset "Analysis" verbose = true begin
    _ts("sttc",        "analysis/sttc.jl")
    _ts("spikes",      "analysis/spikes_test.jl")
    _ts("populations", "analysis/populations_test.jl")
end

# ── Simulation control ────────────────────────────────────────────────────────
_ts("sim_control", "sim/sim_control.jl")
_ts("sweep_fixes", "sim/sweep_fixes.jl")

# ── Networks ──────────────────────────────────────────────────────────────────
@testset "Networks" verbose = true begin
    _ts("ei_network", "network/ei_network.jl")
    _inc("if_net",     "network/if_net.jl")
    _inc("chain",      "network/chain.jl")
    _inc("iz_net",     "network/iz_net.jl")
    _inc("hh_net",     "network/hh_net.jl")
    _inc("oja",        "network/oja.jl")
    _inc("rate_net",   "network/rate_net.jl")
    _inc("stdp_demo",  "network/stdp_demo.jl")
end

# ── Timing report ─────────────────────────────────────────────────────────────
_total = time() - _suite_t0
println("\n", "─"^55)
println("  TIMING REPORT  (slowest first)")
println("─"^55)
for (lbl, dt) in sort(_section_times; by = x -> x.second, rev = true)
    bar = "█" ^ clamp(round(Int, dt * 3), 0, 30)
    @printf("  %-24s %6.2f s  %s\n", lbl, dt, bar)
end
println("─"^55)
@printf("  %-24s %6.2f s\n", "TOTAL", _total)
println("─"^55, "\n")

end # @testset

end # with_logger
