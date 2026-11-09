# Tripod: hyperpolarisation of the soma after a burst is the unbounded adaptation current, w_s / gl below the leak reversal
# (documented in `Tripod`), independent of dt, and bounded by it.
using SNNModels
using Test

function _tripod_burst(Is, Id; dt, Ton = 200f0, T = 800f0)
    E = Tripod(N = 1)
    vmin = Inf; wmax = -Inf; n = 0
    for k = 1:round(Int, T / dt)
        t = k * dt
        E.I .= t < Ton ? Is : 0
        E.I_d .= t < Ton ? Id : 0
        SNNModels.integrate!(E, E.param, dt)
        wmax = max(wmax, E.w_s[1]); n += E.fire[1]
        t > Ton && (vmin = min(vmin, E.v_s[1]))
    end
    return (; vmin, wmax, n, w_end = E.w_s[1], vs_end = E.v_s[1], El = E.adex.El, gl = E.adex.gl)
end

@testset "Tripod afterhyperpolarisation" begin
    r = _tripod_burst(3000f0, 3000f0; dt = 0.1f0)
    @test r.n > 30
    @test r.vmin < r.El - 30                        # a strong burst undershoots the leak reversal by tens of mV ...
    @test r.vmin > r.El - r.wmax / r.gl - 1         # ... never by more than w_max / gl
    @test r.vmin < r.El - 0.3 * r.wmax / r.gl       # and by a large part of it
    @test r.w_end < 0.02 * r.wmax && abs(r.vs_end - r.El) < 1   # recovery after the burst
    @testset "does not depend on dt" begin
        for dt in (0.025f0, 0.5f0)
            q = _tripod_burst(3000f0, 3000f0; dt)
            @test abs(q.vmin - r.vmin) < 5
        end
    end
    @testset "scales with the burst" begin
        v = [_tripod_burst(I, 0f0; dt = 0.1f0).vmin for I in (500f0, 1000f0, 3000f0, 5000f0)]
        @test issorted(v; rev = true)
    end
end
true
