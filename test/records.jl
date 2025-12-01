E = IF(; N = 3200, param = IFParameter(; El = -49mV))
monitor!(E, [:fire, :v])
I = IF(; N = 800, param = IFParameter(; El = -49mV))
EE = SpikingSynapse(E, E, :ge, conn = (μ = 60 * 0.27 / 10, ρ = 0.02))
EI = SpikingSynapse(E, I, :ge, conn = (μ = 60 * 0.27 / 10, ρ = 0.02))
IE = SpikingSynapse(I, E, :gi, conn = (μ = -20 * 4.5 / 10, ρ = 0.02))
II = SpikingSynapse(I, I, :gi, conn = (μ = -20 * 4.5 / 10, ρ = 0.02))
P = [E, I]
C = [EE, EI, IE, II]
model = compose(; E, I, EE, EI, IE, II, silent = true, name = "E/I network")

monitor!([E, I], [:fire])
sim!(model; duration = 1second)
train!(model; duration = 1second)

true
using Interpolations
@load_units
interval = 1s:10ms:2s

## make a test for all cases of record
@testset "Record" begin
    v = record(E, :fire; interval = interval)
    @test size(v, 2) == length(interval)
    @test v isa Interpolations.ScaledInterpolation

    v, r_v = record(E, :fire; range = true, interval = interval)
    @test size(v, 2) == length(r_v)
    @test v isa Interpolations.ScaledInterpolation

    v = record(E, :v)
    @test size(v, 2) == length(0:1ms:get_time(model))
    v, r_v = record(E, :v; range = true)
    @test size(v, 2) == length(r_v)
    @test v isa Interpolations.ScaledInterpolation
end
