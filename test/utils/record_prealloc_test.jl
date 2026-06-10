# record_prealloc_test.jl — standalone snapshot suite for the pre-allocated
# recording-buffer refactor (manifesto: ~/.claude/code-manifests/record-prealloc.md).
#
# This file is NOT wired into runtests.jl (scope decision Q1). Run it directly:
#
#     # Phase A — capture legacy behaviour BEFORE implementing the dense path:
#     SNN_RECORD_SNAPSHOT_MODE=save julia --project test/utils/record_prealloc_test.jl
#
#     # Phase B — after implementing, assert the new path matches the artifacts:
#     SNN_RECORD_SNAPSHOT_MODE=compare julia --project test/utils/record_prealloc_test.jl
#
# Default mode (no env var) is "compare".
#
# Each snapshot exercises one of the storage paths the refactor touches:
#   - scalar / Vector{Float32}      → dense buffer, time axis last
#   - Matrix{Float32} (N, comp)     → dense buffer, 3D, time axis last
#   - :fire                         → flat CSR (times + neurons)
#   - sampling-rate downsampling    → per-key step_count gate
#   - chunked sims                  → buffer persists / grows across sim! calls

using SNNModels
using Test
using JLD2
using Statistics
@load_units

const SNAP_DIR = abspath(joinpath(@__DIR__, "..", "artifacts", "record_prealloc"))
const SNAP_MODE = get(ENV, "SNN_RECORD_SNAPSHOT_MODE", "compare")
mkpath(SNAP_DIR)

# Deterministic seeding so legacy and post-impl runs produce identical traces.
import Random
const SNAP_SEED = 20240601

_artifact_path(name) = joinpath(SNAP_DIR, "$(name)_legacy.jld2")

# Save (Phase A) or compare (Phase B) a named bundle of arrays.
# `data` is a Dict{String,Any} of plain arrays / numbers (JLD2-friendly).
function snapshot!(name::String, data::Dict{String,<:Any})
    path = _artifact_path(name)
    if SNAP_MODE == "save"
        JLD2.jldopen(path, "w") do f
            for (k, v) in data
                f[k] = v
            end
        end
        @info "saved snapshot" name path
        return nothing
    else
        @assert isfile(path) "Missing legacy artifact $path — run with SNN_RECORD_SNAPSHOT_MODE=save first."
        ref = JLD2.load(path)
        @testset "$name" begin
            for (k, v) in data
                @test haskey(ref, k)
                rv = ref[k]
                if v isa AbstractArray && eltype(v) <: Real
                    @test size(v) == size(rv)
                    @test collect(v) ≈ collect(rv) rtol = 1e-4 atol = 1e-5
                else
                    @test v == rv
                end
            end
        end
        return nothing
    end
end

# ────────────────────────────────────────────────────────────────────────────
# snap_voltage — IF (N=100): dense Vector path + fire CSR
# ────────────────────────────────────────────────────────────────────────────
function run_snap_voltage()
    Random.seed!(SNAP_SEED)
    pop = IF(N = 100, param = IFParameter(El = -49mV))
    model = compose(pop = pop, silent = true)
    monitor!(pop, [:v, :fire]; sr = 1000Hz)
    sim!(model, 500ms)
    v = getvariable(pop, :v)
    st = spiketimes(pop)
    data = Dict{String,Any}(
        "v"          => collect(v),
        "v_size"     => collect(size(v)),
        "n_spikes"   => sum(length, st),
        "spike_sum"  => sum(sum, st; init = 0.0f0),
    )
    snapshot!("snap_voltage", data)
end

# ────────────────────────────────────────────────────────────────────────────
# snap_receptors — AdEx (N=50): ge/gi live in synvars (Vector path)
# ────────────────────────────────────────────────────────────────────────────
function run_snap_receptors()
    Random.seed!(SNAP_SEED)
    pop = AdEx(N = 50, synapse = DoubleExpSynapse())
    model = compose(pop = pop, silent = true)
    monitor!(pop, [:ge, :gi]; sr = 1000Hz, variables = :synvars)
    sim!(model, 300ms)
    ge = getvariable(pop, :synvars_ge)
    gi = getvariable(pop, :synvars_gi)
    data = Dict{String,Any}(
        "ge"      => collect(ge),
        "gi"      => collect(gi),
        "ge_size" => collect(size(ge)),
        "gi_size" => collect(size(gi)),
    )
    snapshot!("snap_receptors", data)
end

# ────────────────────────────────────────────────────────────────────────────
# snap_matrix — Tripod (N=20): `is` is a Matrix{Float32}(N, comp), exercises
# the dense 3D buffer path (snapshot shape (N, comp) → buffer (N, comp, T)).
# ────────────────────────────────────────────────────────────────────────────
function run_snap_matrix()
    Random.seed!(SNAP_SEED)
    pop = Tripod(N = 20)
    model = compose(pop = pop, silent = true)
    monitor!(pop, [:is]; sr = 1000Hz)
    sim!(model, 200ms)
    is = getvariable(pop, :is)
    data = Dict{String,Any}(
        "is"      => collect(is),
        "is_size" => collect(size(is)),
    )
    snapshot!("snap_matrix", data)
end

# ────────────────────────────────────────────────────────────────────────────
# snap_spikes — Poisson (N=200, rate=20Hz): fire CSR total count + per-neuron
# ────────────────────────────────────────────────────────────────────────────
function run_snap_spikes()
    Random.seed!(SNAP_SEED)
    pop = Poisson(N = 200, param = PoissonParameter(20Hz))
    model = compose(pop = pop, silent = true)
    monitor!(pop, [:fire]; sr = 1000Hz)
    sim!(model, 1000ms)
    st = spiketimes(pop)
    counts = Float32[length(s) for s in st]
    data = Dict{String,Any}(
        "n_spikes"   => sum(length, st),
        "counts"     => counts,
        "mean_count" => mean(counts),
    )
    snapshot!("snap_spikes", data)
end

# ────────────────────────────────────────────────────────────────────────────
# snap_multirate — AdEx (N=100): :v at sr=500Hz + fire. Correct downsampling.
# ────────────────────────────────────────────────────────────────────────────
function run_snap_multirate()
    Random.seed!(SNAP_SEED)
    pop = AdEx(N = 100, synapse = DoubleExpSynapse())
    model = compose(pop = pop, silent = true)
    monitor!(pop, [:v]; sr = 500Hz)
    monitor!(pop, [:fire]; sr = 1000Hz)
    sim!(model, 400ms)
    v = getvariable(pop, :v)
    st = spiketimes(pop)
    data = Dict{String,Any}(
        "v"        => collect(v),
        "v_ncols"  => size(v, ndims(v)),
        "n_spikes" => sum(length, st),
    )
    snapshot!("snap_multirate", data)
end

# ────────────────────────────────────────────────────────────────────────────
# snap_chunked — IF (N=50): sim!(100ms)×10 must equal sim!(1s) (one shot).
# Compares chunked vs single-run within this run (no artifact needed), and also
# snapshots the chunked result for cross-version stability.
# ────────────────────────────────────────────────────────────────────────────
function run_snap_chunked()
    # chunked run
    Random.seed!(SNAP_SEED)
    pop_c = IF(N = 50, param = IFParameter(El = -49mV))
    model_c = compose(pop = pop_c, silent = true)
    monitor!(pop_c, [:v]; sr = 1000Hz)
    for _ in 1:10
        sim!(model_c, 100ms)
    end
    v_chunked = collect(getvariable(pop_c, :v))

    # single run
    Random.seed!(SNAP_SEED)
    pop_s = IF(N = 50, param = IFParameter(El = -49mV))
    model_s = compose(pop = pop_s, silent = true)
    monitor!(pop_s, [:v]; sr = 1000Hz)
    sim!(model_s, 1000ms)
    v_single = collect(getvariable(pop_s, :v))

    @testset "snap_chunked — chunked == single" begin
        @test size(v_chunked) == size(v_single)
        @test v_chunked ≈ v_single rtol = 1e-4 atol = 1e-5
    end

    data = Dict{String,Any}(
        "v_chunked"  => v_chunked,
        "v_single"   => v_single,
        "size"       => collect(size(v_chunked)),
    )
    snapshot!("snap_chunked", data)
end

@testset "record_prealloc snapshots (mode=$(SNAP_MODE))" begin
    run_snap_voltage()
    run_snap_receptors()
    run_snap_matrix()
    run_snap_spikes()
    run_snap_multirate()
    run_snap_chunked()
end

println("\nSnapshot suite finished in mode=$(SNAP_MODE). Artifacts at:\n  $SNAP_DIR")
