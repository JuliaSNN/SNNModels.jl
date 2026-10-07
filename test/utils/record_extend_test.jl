# Recording buffers extended across sim! chunks (SNNModels 1.9.1): geometric growth, data
# identical to a single run, one warning per recording.
using SNNModels, Test, Random
using SNNModels.Logging
@load_units

function _tonic(; N = 50)
    Random.seed!(11)                                   # IF starts from random membrane potentials
    pop = IF(N = N, param = IFParameter(El = -49mV))   # El above threshold: deterministic tonic firing
    return pop, compose(pop = pop, silent = true)
end

@testset "chunked recording: same data, geometric growth, one warning" begin
    # reference: one 2 s run
    ref, mref = _tonic()
    monitor!(ref, [:fire, :v]; sr = 1kHz)
    sim!(mref, 2000ms)

    # 40 chunks of 50 ms, :fire sized for 1 Hz so every chunk would need room
    pop, model = _tonic()
    monitor!(pop, [:fire, :v]; sr = 1kHz, monitor_rate = 1Hz)
    logger = Test.TestLogger(min_level = Logging.Warn)
    with_logger(logger) do
        for _ = 1:40
            sim!(model, 50ms)
        end
    end

    @test spiketimes(pop) == spiketimes(ref)
    @test record(pop, :v) == record(ref, :v)

    n = sum(length, spiketimes(pop))
    cap = length(pop.records[:fire][:times_buf])
    @test n > 1000                    # the test exercises many extensions
    @test cap < 2n + 1000             # geometric growth: at most about 2x the spikes held

    msgs = [string(l.message) for l in logger.logs]
    @test count(m -> occursin("Fire COO buffer", m), msgs) == 1
    @test count(m -> occursin("Dense recording buffer for v", m), msgs) <= 1
end

@testset "_extend_fire! grows geometrically and keeps the written spikes" begin
    fire = Dict{Symbol,AbstractVector}(:times_buf => Float32[1, 2, 3, 0], :neurons_buf => [1, 2, 3, 0])
    meta = Dict{Symbol,Any}(:allocated => Dict(:fire => 1), :grew => Dict(:fire => false))
    @test_logs (:warn, r"Fire COO buffer extended") SNNModels._extend_fire!(fire, meta, 1)
    @test length(fire[:times_buf]) == 8                     # max(4 + 1, 2 * 4)
    @test fire[:times_buf][1:3] == Float32[1, 2, 3] && fire[:neurons_buf][1:3] == [1, 2, 3]
    @test meta[:allocated][:fire] == 5                      # 8 slots, 3 written
    @test meta[:grew][:fire]
    @test_logs SNNModels._extend_fire!(fire, meta, 10)      # latched: no second warning
    @test length(fire[:times_buf]) == 18                    # max(8 + 10, 16)
end
