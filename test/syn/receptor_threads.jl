# Receptor registry (`_predefined`) used from several threads: no ConcurrencyViolationError, no duplicate entries.
# Meaningful with several threads, e.g. `julia -t 4 --project=test test/syn/receptor_threads.jl`.
using Test, SNNModels
const SM = SNNModels
@load_units

@testset "receptor registry is thread safe" begin
    n0 = length(SM._PREDEFINED)
    errs = Channel{Any}(Inf)
    Threads.@threads for i in 1:2000
        try
            r = SM._predefined(Receptor; E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.7311 + 1e-3 * (i % 8), target = :glu)
            SM._from_definition(r, Float64)
            Tripod(N = 2)
        catch e
            put!(errs, e)
        end
    end
    close(errs)
    @test isempty(collect(errs))
    @test length(SM._PREDEFINED) - n0 == 8      # 8 distinct definitions, each registered once
end
