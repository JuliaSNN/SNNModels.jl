# Izhikevich neuron stays bounded: default state, regular spiking, I = 10 (reproduction of the divergence to -1e17 mV).
using SNNModels
using Test

function _iz_extremes(E, dt, T)
    vmin, vmax, n = Inf, -Inf, 0
    for _ = 1:round(Int, T / dt)
        SNNModels.integrate!(E, E.param, dt)
        vmin = min(vmin, minimum(E.v)); vmax = max(vmax, maximum(E.v)); n += count(E.fire)
    end
    return vmin, vmax, n
end

@testset "IZ stability" begin
    rs = IZParameter(a = 0.02, b = 0.2, c = -65, d = 8)
    @testset "default conductances are zero" begin
        E = IZ(N = 5)
        @test all(iszero, E.ge) && all(iszero, E.gi)
    end
    @testset "regular spiking, I = 10, default state" begin
        for dt in (0.1f0, 0.25f0, 0.5f0, 1.0f0)
            E = IZ(N = 3, param = rs)
            E.I .= 10
            vmin, vmax, n = _iz_extremes(E, dt, 500)
            @test -90 < vmin
            @test vmax <= 30 + 1   # reset after the step that crosses 30 mV
            @test 5 <= n / 3 <= 20   # about 12 spikes in 500 ms
        end
    end
    @testset "Float64 population" begin
        E = IZ(N = 2, param = IZParameter(FT = Float64, a = 0.02, b = 0.2, c = -65, d = 8), VFT = Vector{Float64})
        E.I .= 10
        vmin, vmax, n = _iz_extremes(E, 0.5, 500)
        @test -90 < vmin && n > 0
    end
    @testset "no input: rests" begin
        E = IZ(N = 2, param = rs)
        vmin, vmax, n = _iz_extremes(E, 0.5, 500)
        @test n == 0 && vmin > -75   # the fixed point is near -70 mV
    end
end
true
