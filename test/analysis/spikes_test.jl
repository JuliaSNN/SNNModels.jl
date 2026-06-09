using SNNModels
using Test
using SparseArrays
@load_units

# Build a small driven IF network and run it once for all tests.
function _spike_model()
    pop   = IF(N = 20, param = IFParameter(El = -49mV))
    model = compose(pop = pop, silent = true)
    monitor!(pop, [:fire])
    sim!(model, 500ms)
    return model, pop
end

const _SPIKE_MODEL, _SPIKE_POP = _spike_model()

@testset "Spike analysis" begin

    @testset "spiketimes — returns Spiketimes of length N" begin
        st = spiketimes(_SPIKE_POP)
        @test st isa Spiketimes
        @test length(st) == _SPIKE_POP.N
        # with constant drive some neurons must have spiked
        @test any(!isempty, st)
    end

    @testset "spiketimes — interval kwarg" begin
        st_full = spiketimes(_SPIKE_POP)
        st_half = spiketimes(_SPIKE_POP; interval = (0, 250))
        # all spikes in the half interval must be ≤ 250ms
        for n in eachindex(st_half)
            @test all(st_half[n] .<= 250)
        end
        # full interval has at least as many spikes
        n_full = sum(length, st_full)
        n_half = sum(length, st_half)
        @test n_full >= n_half
    end

    @testset "spikes_in_interval — filters by window" begin
        st = spiketimes(_SPIKE_POP)
        win = [100.0f0, 300.0f0]
        si  = spikes_in_interval(st, win)
        for n in eachindex(si)
            @test all(100 .< si[n] .<= 300)
        end
    end

    @testset "firing_rate — returns interpolant + range" begin
        st = spiketimes(_SPIKE_POP)
        fr, r = firing_rate(st; interval = 0:1ms:500ms)
        @test r ≈ 0:1ms:500ms
        @test all(isfinite, fr(1:_SPIKE_POP.N, r))
    end

    @testset "firing_rate — time_average" begin
        st = spiketimes(_SPIKE_POP)
        fr_avg, r = firing_rate(st; interval = 0:1ms:500ms, time_average = true)
        @test length(fr_avg) == _SPIKE_POP.N
        @test all(fr_avg .>= 0)
    end

    @testset "firing_rate — pop_average" begin
        st = spiketimes(_SPIKE_POP)
        fr, r = firing_rate(st; interval = 0:1ms:500ms, pop_average = true)
        # pop_average=true: mean(rates, dims=1)[1,:] → returns a plain Vector, not callable
        @test fr isa Vector
        @test length(fr) == length(r)
        @test all(isfinite, fr)
    end

    @testset "bin_spiketimes — returns sparse + range" begin
        st = spiketimes(_SPIKE_POP)
        n1 = findfirst(!isempty, st)
        if !isnothing(n1)
            sp, r = bin_spiketimes(st[n1]; interval = 0:10ms:500ms)
            @test sp isa SparseVector
            @test length(sp) == length(0:10ms:500ms)
        end
    end

    @testset "bin_spiketimes — do_sparse=false" begin
        st = spiketimes(_SPIKE_POP)
        n1 = findfirst(!isempty, st)
        if !isnothing(n1)
            sp, r = bin_spiketimes(st[n1]; interval = 0:10ms:500ms, do_sparse = false)
            @test sp isa Vector
        end
    end

    @testset "gaussian_smooth — same length, finite values" begin
        xs = collect(0.0:1.0:100.0)
        x  = randn(Float32, length(xs))
        σ  = 3.0
        y  = gaussian_smooth(xs, x, σ)
        @test length(y) == length(x)
        @test all(isfinite, y)
    end

    @testset "ISI_CV2 — Spiketimes input" begin
        st  = spiketimes(_SPIKE_POP)
        cv2 = ISI_CV2(st)
        @test length(cv2) == _SPIKE_POP.N
        @test all(x -> x >= 0, cv2)
        # for neurons with < 3 spikes, CV2 is 0 by convention
        for (cv, sp) in zip(cv2, st)
            length(sp) < 3 && @test cv == 0
        end
    end

    @testset "ISI_CV2 — population dispatch" begin
        cv2 = ISI_CV2(_SPIKE_POP)
        @test length(cv2) == _SPIKE_POP.N
        @test all(x -> x >= 0, cv2)
    end

end
true
