using SNNModels
using Test
using SNNModels.Logging
@load_units

# monitor! on a key that is already monitored updates the settings that differ, with an @info note
@testset "monitor! - re-monitoring updates settings" begin
    meta(p) = p.records[:meta]

    @testset "sr, monitor_time, monitor_rate" begin
        E = IF(N = 10)
        monitor!(E, [:v, :fire]; sr = 1kHz)
        @test E.records[:sr][:v] == Float32(1kHz)
        @test_logs (:info, r"Monitor settings of :v.*sr 1000.0 Hz -> 2000.0 Hz.*monitor_time unset -> 500.0 ms") min_level = Logging.Info monitor!(
            E, [:v]; sr = 2kHz, monitor_time = 500ms,
        )
        @test E.records[:sr][:v] == Float32(2kHz)
        @test meta(E)[:monitor_time][:v] == 500f0
        @test meta(E)[:period][:v] == 4          # 1 / (2 kHz) / 0.125 ms
        @test_logs (:info, r"Monitor settings of :fire.*monitor_time unset -> 500.0 ms, monitor_rate unset -> 50.0 Hz") min_level = Logging.Info monitor!(
            E, [:fire]; monitor_time = 500ms, monitor_rate = 50Hz,
        )
        @test meta(E)[:monitor_rate][:fire] == 50f0
        # same values again, or settings left out: no message and nothing changes
        @test_logs min_level = Logging.Info monitor!(E, [:v, :fire]; sr = 2kHz, monitor_time = 500ms, monitor_rate = 50Hz)
        @test_logs min_level = Logging.Info monitor!(E, [:v, :fire])
        @test E.records[:sr][:v] == Float32(2kHz)
        @test meta(E)[:monitor_time][:v] == 500f0
        @test E.records[:data] == [:v, :fire]
        # a rate in plain Hz is still rejected
        @test_throws ArgumentError monitor!(E, [:fire]; monitor_rate = 50)
    end

    @testset "the new settings are used by the next sim!" begin
        E = IF(N = 10)
        monitor!(E, [:v]; sr = 1kHz)
        monitor!(E, [:v]; sr = 500Hz)
        with_logger(NullLogger()) do
            sim!([E]; duration = 100ms)
        end
        v = getvariable(E, :v)
        @test size(v, 2) == 51                   # t = 0 and one sample every 2 ms
    end

    @testset "collections keep their 200 Hz default" begin
        E = IF(N = 5)
        I = IF(N = 5)
        monitor!([E, I], [:v])
        @test E.records[:sr][:v] == Float32(200Hz)
        monitor!([E, I], [:v]; sr = 1kHz)
        @test I.records[:sr][:v] == Float32(1kHz)
        monitor!((; E, I), [:v])                 # omitted: unchanged
        @test E.records[:sr][:v] == Float32(1kHz)
    end
end
