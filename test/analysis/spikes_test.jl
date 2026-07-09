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

    # Synthetic spiketimes: evenly spaced spikes at known rate.
    # firing_rate convolution should recover a value close to the true rate.
    @testset "firing_rate — synthetic 10 Hz neuron (interpolated)" begin
        rate_hz  = 10.0f0   # Hz = spikes/second
        duration = 1000.0f0 # ms
        # Evenly spaced spikes every 100ms
        st = Spiketimes([collect(100.0f0:100.0f0:900.0f0)])
        fr, r = firing_rate(st; interval = 0f0:1f0:duration, τ = 50ms)
        # evaluate at the middle of the train where rate should be stable
        mid_rate = fr(1, 500f0)
        @test mid_rate > 0
        @test isfinite(mid_rate)
    end

    @testset "firing_rate — all-silent spiketimes, shape check" begin
        st = Spiketimes([Float32[], Float32[], Float32[]])
        fr, r = firing_rate(st; interval = 0f0:1f0:200f0)
        mat = fr(1:3, r)
        @test size(mat) == (3, length(r))
        @test all(mat .== 0)
    end

    @testset "firing_rate — population dispatch returns same shape" begin
        st = spiketimes(_SPIKE_POP)
        fr1, r1 = firing_rate(st; interval = 0:1ms:500ms)
        fr2, r2 = firing_rate(_SPIKE_POP; interval = 0:1ms:500ms)
        @test r1 ≈ r2
        @test fr1(1:_SPIKE_POP.N, r1) ≈ fr2(1:_SPIKE_POP.N, r2)
    end

    @testset "firing_rate — non-interpolated returns Matrix" begin
        st = spiketimes(_SPIKE_POP)
        fr, r = firing_rate(st; interval = 0:1ms:500ms, interpolate = false)
        @test fr isa Matrix
        @test size(fr) == (_SPIKE_POP.N, length(r))
        @test all(isfinite, fr)
    end

    # ── Fix regression tests ──────────────────────────────────────────────────

    @testset "Fix 3 — _spiketimes_coo binary search: interval boundaries exact" begin
        # All spikes exactly on boundary must respect strict (lo, hi) semantics.
        st_full = spiketimes(_SPIKE_POP)
        # interval with strict bounds: t > lo and t < hi
        lo, hi = 100f0, 400f0
        st_win = spiketimes(_SPIKE_POP; interval = (lo, hi))
        for n in eachindex(st_win)
            @test all(t -> t > lo && t < hi, st_win[n])
        end
        # subset must not exceed full count
        @test sum(length, st_win) <= sum(length, st_full)
    end

    @testset "Fix 4 — firing_rate matrix shape without intermediate vector" begin
        st = spiketimes(_SPIKE_POP)
        fr, r = firing_rate(st; interval = 0:1ms:500ms, interpolate = false)
        @test fr isa Matrix
        @test size(fr, 1) == _SPIKE_POP.N
        @test size(fr, 2) == length(r)
        @test all(isfinite, fr)
        @test all(fr .>= -1e-10)  # conv can produce tiny fp negatives at boundaries
    end

    @testset "Fix 5 — time_average_fr direct count matches spikes_in_interval" begin
        st = spiketimes(_SPIKE_POP)
        interval = 0:1ms:500ms
        fr_ta, _ = firing_rate(st; interval, time_average = true)
        # manual reference: count spikes in (lo, hi] per neuron / duration_s
        lo, hi = Float32(interval[1]), Float32(interval[end])
        dur_s = (hi - lo) / 1000f0
        ref = [count(t -> t > lo && t <= hi, st[n]) / dur_s for n in eachindex(st)]
        @test fr_ta ≈ Float32.(ref)  atol=1f-4
    end

    @testset "Fix 7 — _init_spiketimes correct length and type" begin
        st = spiketimes(_SPIKE_POP)
        @test st isa Spiketimes
        @test length(st) == _SPIKE_POP.N
        @test all(s -> s isa Vector{Float32}, st)
    end

    @testset "Fix 8 — spiketimes(NamedTuple/Vector) no vcat O(N²): neuron count correct" begin
        pop1 = IF(N = 5, param = IFParameter(El = -49mV))
        pop2 = IF(N = 7, param = IFParameter(El = -49mV))
        m = compose(pop1 = pop1, pop2 = pop2, silent = true)
        monitor!(pop1, [:fire]); monitor!(pop2, [:fire])
        sim!(m, 200ms)
        # NamedTuple dispatch
        st_nt = spiketimes(m.pop)
        @test length(st_nt) == pop1.N + pop2.N
        # Vector dispatch
        st_v = spiketimes([pop1, pop2])
        @test length(st_v) == pop1.N + pop2.N
        @test all(s -> s isa Vector{Float32}, st_v)
    end

    # ── Bug verification tests ────────────────────────────────────────────────

    @testset "Bug 1 — _retrieve_interval: firing_rate with no interval succeeds" begin
        # _retrieve_interval auto-computes the interval from the spike times when none
        # is provided: tt0=0, ttf=max(spike times), step=20ms.
        st = Spiketimes([Float32[100f0, 200f0, 300f0], Float32[150f0, 250f0]])
        fr, r = @test_nowarn firing_rate(st)
        @test fr isa AbstractArray
        @test last(r) ≈ 300f0
    end

    @testset "Bug 2 — pop_average+interpolate returns correct Vector" begin
        # With interpolate=true (default), rates becomes a ScaledInterpolation.
        # mean(rates, dims=1)[1,:] must still produce a finite Vector of length == length(r).
        st = Spiketimes([Float32[100f0, 200f0, 300f0] for _ in 1:5])
        fr, r = firing_rate(st; interval = 0f0:1f0:400f0, pop_average = true)
        @test fr isa Vector
        @test length(fr) == length(r)
        @test all(isfinite, fr)
    end

end
true
