# `_tanh_inl` and `_cosh_inl` (Float32 tanh and cosh with an inlined exp, used by the vectorised neuron loops) must return the bits of Base.
using Random
@testset "inlined tanh and cosh are bit-identical to Base" begin
    Random.seed!(7)
    same(a, b) = (isnan(a) && isnan(b)) || reinterpret(UInt32, a) == reinterpret(UInt32, b)
    special = Float32[0, -0.0, 1, -1, 1f-30, -1f-30, 1f-12, 2f-12, 0.5, -0.5, 1.3862944, 1.3862945, 9, -9, 17.9, 18, 18.1, 88.7, 88.72283, 88.8, 100, -100,
                      floatmax(Float32), -floatmax(Float32), Inf32, -Inf32, NaN32, nextfloat(0f0), prevfloat(0f0)]
    for x in special
        @test same(SNNModels._tanh_inl(x), tanh(x))
        @test same(SNNModels._cosh_inl(x), cosh(x))
    end
    xs = reinterpret.(Float32, rand(UInt32, 2_000_000))                  # all magnitudes
    @test all(i -> same(SNNModels._tanh_inl(xs[i]), tanh(xs[i])), eachindex(xs))
    @test all(i -> same(SNNModels._cosh_inl(xs[i]), cosh(xs[i])), eachindex(xs))
    xs = 12f0 .* randn(Float32, 2_000_000)                               # the range where the neuron models live
    @test all(i -> same(SNNModels._tanh_inl(xs[i]), tanh(xs[i])), eachindex(xs))
    @test all(i -> same(SNNModels._cosh_inl(xs[i]), cosh(xs[i])), eachindex(xs))
    @test SNNModels._tanh_inl(0.3) == tanh(0.3)                           # other precisions fall back to Base
    @test SNNModels._cosh_inl(0.3) == cosh(0.3)
end
true
