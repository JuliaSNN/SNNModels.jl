using SNNModels
using Test
using Interpolations
@load_units

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

function _small_model(; El = -49mV, N = 20, T = 200ms)
    pop   = IF(N = N, param = IFParameter(El = El))
    model = compose(pop = pop, silent = true)
    return model, pop
end

function _run(model, pop, syms, T = 200ms; sr = 1000Hz, variables = :none)
    if variables == :none
        monitor!(pop, syms; sr = sr)
    else
        monitor!(pop, syms; sr = sr, variables = variables)
    end
    sim!(model, T)
    return model, pop
end

# expected number of recorded steps at sr over duration T, dt=0.125ms
_nsteps(T, sr) = length(0ms : (1/sr/1ms)*ms : T) - 1

# ────────────────────────────────────────────────────────────────────────────
# monitor! — setup
# ────────────────────────────────────────────────────────────────────────────

@testset "monitor! — fire field" begin
    _, pop = _small_model()
    monitor!(pop, [:fire])
    @test haskey(pop.records, :fire)
    @test haskey(pop.records[:fire], :time)
    @test haskey(pop.records[:fire], :neurons)
end

@testset "monitor! — direct field :v" begin
    _, pop = _small_model()
    monitor!(pop, [:v])
    @test haskey(pop.records, :v)
    @test pop.records[:v] isa Vector
end

@testset "monitor! — multiple fields" begin
    _, pop = _small_model()
    monitor!(pop, [:v, :fire])
    @test haskey(pop.records, :v)
    @test haskey(pop.records, :fire)
end

@testset "monitor! — variables= prefix" begin
    _, pop = _small_model()
    monitor!(pop, [:glu]; variables = :receptors)
    @test haskey(pop.records, :receptors_glu)
end

@testset "monitor! — duplicate key is idempotent" begin
    _, pop = _small_model()
    monitor!(pop, [:v])
    n_keys_before = length(pop.records[:v])
    monitor!(pop, [:v])  # should not add a second entry
    @test haskey(pop.records, :v)   # still there
    n_keys_after = length(pop.records[:v])
    @test n_keys_before == n_keys_after  # no extra data
end

@testset "monitor! — missing field silently skipped" begin
    _, pop = _small_model()
    monitor!(pop, [:nonexistent_field_xyz])
    @test !haskey(pop.records, :nonexistent_field_xyz)
end

# ────────────────────────────────────────────────────────────────────────────
# record! via sim! — data written correctly
# ────────────────────────────────────────────────────────────────────────────

@testset "record! — :v written at each recorded step" begin
    model, pop = _small_model()
    sr = 1000Hz
    monitor!(pop, [:v]; sr = sr)
    sim!(model, 200ms)
    v_rec = pop.records[:v]
    @test length(v_rec) > 0
    @test all(x -> length(x) == pop.N, v_rec)  # each snapshot is N-length
end

@testset "record! — sampling rate respected" begin
    model, pop = _small_model()
    monitor!(pop, [:v]; sr = 500Hz)  # 1 sample per 2ms
    sim!(model, 100ms)
    # at dt=0.125ms, period = floor(1/500Hz / 0.125ms) = floor(16) = 16 steps per sample
    # over 100ms = 800 steps → expect ~50 samples
    n = length(pop.records[:v])
    @test 40 <= n <= 60  # loose bound (record_zero! adds one at t=0)
end

@testset "record! — :fire records spikes" begin
    model, pop = _small_model(El = -49mV)  # spontaneous firing
    monitor!(pop, [:fire])
    sim!(model, 500ms)
    @test length(pop.records[:fire][:time]) > 0
    @test length(pop.records[:fire][:neurons]) > 0
end

@testset "record! — variables= prefix records from sub-field" begin
    model, pop = _small_model()
    monitor!(pop, [:glu]; variables = :receptors)
    sim!(model, 100ms)
    rec = pop.records[:receptors_glu]
    @test length(rec) > 0
    @test all(x -> length(x) == pop.N, rec)
end

@testset "record! — start_time and end_time populated" begin
    model, pop = _small_model()
    monitor!(pop, [:v])
    sim!(model, 100ms)
    @test haskey(pop.records[:start_time], :v)
    @test haskey(pop.records[:end_time], :v)
    @test pop.records[:start_time][:v] >= 0f0
    @test pop.records[:end_time][:v] > pop.records[:start_time][:v]
    @test pop.records[:end_time][:v] ≈ 100f0 rtol=0.01
end

# ────────────────────────────────────────────────────────────────────────────
# clear_records! (P5) — time bounds reset
# ────────────────────────────────────────────────────────────────────────────

@testset "clear_records! — data cleared" begin
    model, pop = _small_model()
    monitor!(pop, [:v, :fire])
    sim!(model, 100ms)
    @test length(pop.records[:v]) > 0
    clear_records!(pop)
    @test length(pop.records[:v]) == 0
    @test length(pop.records[:fire][:time]) == 0
end

@testset "clear_records! — start_time / end_time reset (P5)" begin
    model, pop = _small_model()
    monitor!(pop, [:v])
    sim!(model, 100ms)
    @test haskey(pop.records[:start_time], :v)
    clear_records!(pop)
    # after clear, time bounds must be gone so next sim gets fresh ones
    @test !haskey(pop.records[:start_time], :v)
    @test !haskey(pop.records[:end_time], :v)
end

@testset "clear_records! + continue sim — time bounds refreshed" begin
    model, pop = _small_model()
    monitor!(pop, [:v])
    sim!(model, 100ms)
    clear_records!(pop)
    sim!(model, 50ms)
    # new start_time should be ~100ms (where new sim begins), not 0
    @test haskey(pop.records[:start_time], :v)
    @test pop.records[:end_time][:v] > pop.records[:start_time][:v]
end

# ────────────────────────────────────────────────────────────────────────────
# record_step (P4) — guard against divide-by-zero
# ────────────────────────────────────────────────────────────────────────────

@testset "record_step — sr > 1/dt does not crash (P4)" begin
    model, pop = _small_model()
    # dt=0.125ms → max meaningful sr = 8kHz; use 100kHz to force period < 1 before fix
    @test_nowarn begin
        monitor!(pop, [:v]; sr = 100000Hz)
        sim!(model, 10ms)
    end
    @test length(pop.records[:v]) > 0  # at least some samples recorded
end

# ────────────────────────────────────────────────────────────────────────────
# record() retrieval
# ────────────────────────────────────────────────────────────────────────────

@testset "record() — interpolated :v" begin
    model, pop = _small_model()
    monitor!(pop, [:v]; sr = 1000Hz)
    sim!(model, 200ms)
    v = record(pop, :v)
    @test v isa Interpolations.ScaledInterpolation
    @test size(v, 1) == pop.N
end

@testset "record() — :fire returns interpolant" begin
    model, pop = _small_model(El = -49mV)
    monitor!(pop, [:fire])
    sim!(model, 300ms)
    interval = 0ms:1ms:300ms
    v = record(pop, :fire; interval = interval)
    @test v isa Interpolations.ScaledInterpolation
end

@testset "getvariable() — returns (N, T) matrix" begin
    model, pop = _small_model()
    monitor!(pop, [:v]; sr = 1000Hz)
    sim!(model, 100ms)
    mat = getvariable(pop, :v)
    @test mat isa Matrix
    @test size(mat, 1) == pop.N
    @test size(mat, 2) > 0
end
