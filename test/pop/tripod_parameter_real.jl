# TripodParameter / BallAndStickParameter accept any Real for the dendrite lengths `ds` and the diameter `d`.
using Test, SNNModels
@load_units

@testset "TripodParameter accepts Float64 lengths and diameter" begin
    ref = TripodParameter(ds = [200um, 400um], d = 4um)
    p64 = TripodParameter(ds = [Float64(200um), Float64(400um)], d = Float64(4um))
    @test p64.ds == ref.ds
    @test p64.d == ref.d
    @test TripodParameter(ds = [200f-4, 400e-4]).ds == ref.ds          # mixed Float32 / Float64
    @test TripodParameter(ds = [(150e-4, 300e-4), 400e-4]).ds[2] === 400f-4  # tuple and scalar
    @test BallAndStickParameter(ds = [150e-4], d = 3e-4).ds == Union{Float32,Tuple}[150um]
    T = Tripod(N = 3, param = p64)
    @test T.d1.l == fill(200um, 3) && T.d2.l == fill(400um, 3)
end
