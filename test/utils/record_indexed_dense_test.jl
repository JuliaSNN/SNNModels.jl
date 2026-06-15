# Branch-specific tests for fix/record-indexed-dense
# NOT included in test/runtests.jl — run manually:
#   julia --project=test test/utils/record_indexed_dense_test.jl
#
# Tests:
#   1. Indexed-subset monitor! uses dense path, not legacy.
#   2. getvariable returns correct subset shape and is a zero-copy view.
#   3. Indexed values match full-population recording at the same neurons.
#   4. Full-pop and indexed monitors co-exist on the same population.
#   5. Legacy getvariable path still readable (backward compat with SNNsave
#      artifacts saved before the dense refactor).

using SNNModels
using Test
using JLD2
import Random
@load_units

const ARTIFACT_DIR = abspath(joinpath(@__DIR__, "..", "artifacts", "record_prealloc"))

# ── Helpers ───────────────────────────────────────────────────────────────────

function _model(; N = 50, seed = 42)
    Random.seed!(seed)
    pop   = IF(N = N, param = IFParameter(El = -49mV))
    model = compose(pop = pop, silent = true)
    return model, pop
end

_nsteps(T, sr, dt = 0.125ms) =
    floor(Int, floor(Int, T / dt) / max(1, floor(Int, 1.0 / sr / dt))) + 1  # +1 for record_zero! at t=0

# ── 1. Indexed subset → dense mode ───────────────────────────────────────────

@testset "indexed subset: monitor! sets dense mode" begin
    _, pop = _model()
    ind = [1, 5, 10, 20]
    monitor!(pop, [(:v, ind)]; sr = 1000Hz)

    meta = pop.records[:meta]
    @test get(meta[:mode], :v, :legacy) === :dense
    @test meta[:snapshot_size][:v] == (length(ind),)
end

# ── 2. Shape and view type ────────────────────────────────────────────────────

@testset "getvariable indexed subset: shape and zero-copy view" begin
    model, pop = _model(N = 50)
    ind = [2, 10, 30]
    T   = 100ms
    sr  = 1000Hz

    monitor!(pop, [(:v, ind)]; sr = sr)
    sim!(model, T)

    v = getvariable(pop, :v)
    nsteps = _nsteps(T, sr)

    @test size(v, 1) == length(ind)
    @test size(v, 2) == nsteps
    @test v isa SubArray                 # selectdim view, not a freshly allocated array
end

# ── 3. Indexed values match full-population at same neurons ───────────────────

@testset "indexed values consistent with full-pop recording" begin
    ind = [3, 7, 15]
    T   = 60ms
    sr  = 2000Hz

    model_full, pop_full = _model(N = 20, seed = 99)
    model_ind,  pop_ind  = _model(N = 20, seed = 99)

    monitor!(pop_full, [:v]; sr = sr)
    sim!(model_full, T)

    monitor!(pop_ind, [(:v, ind)]; sr = sr)
    sim!(model_ind, T)

    v_full = getvariable(pop_full, :v)
    v_ind  = getvariable(pop_ind,  :v)

    @test collect(v_ind) ≈ collect(v_full[ind, :])
end

# ── 4. Full-pop and indexed monitors co-exist ─────────────────────────────────

@testset "full-pop dense and indexed dense co-exist on same population" begin
    model, pop = _model(N = 40)
    ind = [1, 2, 3]
    T   = 80ms
    sr  = 500Hz

    monitor!(pop, [:v]; sr = sr)
    monitor!(pop, [(:w, ind)]; sr = sr)
    sim!(model, T)

    v = getvariable(pop, :v)
    w = getvariable(pop, :w)
    nsteps = _nsteps(T, sr)

    @test size(v, 1) == 40
    @test size(w, 1) == length(ind)
    @test size(v, 2) == nsteps
    @test size(w, 2) == nsteps
    @test w isa SubArray
end

# ── 5. Legacy getvariable: backward compat with pre-refactor artifacts ─────────
# The artifact snap_voltage_legacy.jld2 was generated from a push!-based
# recording (Vector{Vector{Float32}} per step). We reconstruct that format
# and verify getvariable still returns the same matrix — so SNNload data
# from before this refactor is still readable.

@testset "legacy getvariable: artifact round-trip" begin
    art_path = joinpath(ARTIFACT_DIR, "snap_voltage_legacy.jld2")
    @assert isfile(art_path) "Missing artifact $art_path — run record_prealloc_test.jl in save mode first."

    ref = JLD2.load(art_path)
    v_ref = ref["v"]::Matrix{Float32}   # shape (N, nsteps)
    N, nsteps = size(v_ref)

    # Re-build a synthetic population just for its records dict.
    _, pop = _model(N = N)
    SNNModels._init_records!(pop.records)
    # Inject legacy-format data (as SNNload would produce from old JLD2 files).
    legacy_vec = [v_ref[:, t] for t in 1:nsteps]
    pop.records[:v_leg] = legacy_vec
    push!(pop.records[:data], :v_leg)
    pop.records[:sr][:v_leg]         = 1000f0
    pop.records[:start_time][:v_leg] = 0f0
    pop.records[:end_time][:v_leg]   = Float32(nsteps * 0.125)
    pop.records[:meta][:mode][:v_leg] = :legacy

    v_out = getvariable(pop, :v_leg)

    @test size(v_out) == (N, nsteps)
    @test v_out ≈ v_ref
end
