using SNNModels
using Test
using MetaGraphs
using Graphs: nv, ne
@load_units

@testset "Simulation control" begin

    # IFParameter(El=-49mV) → spontaneous firing (close to Vt=-50mV)
    _active_param = IFParameter(El = -49mV)

    function _model()
        E     = IF(N = 10, param = _active_param)
        model = compose(pop = E, silent = true)
        monitor!(E, [:fire])
        return model, E
    end

    @testset "get_time — zero before sim!" begin
        model, _ = _model()
        @test get_time(model) ≈ 0.0f0
    end

    @testset "get_time — advances after sim!" begin
        model, _ = _model()
        sim!(model, 100ms)
        @test get_time(model) ≈ 100f0
    end

    @testset "reset_time! — resets to zero" begin
        model, _ = _model()
        sim!(model, 200ms)
        @test get_time(model) ≈ 200f0
        reset_time!(model)
        @test get_time(model) ≈ 0.0f0
    end

    @testset "clear_records! — empties fire record" begin
        model, E = _model()
        sim!(model, 300ms)
        @test length(E.records[:fire][:time]) > 0
        clear_records!(E)
        @test length(E.records[:fire][:time]) == 0
    end

    @testset "monitor! — sr keyword stored" begin
        E = IF(N = 5)
        monitor!(E, [:v]; sr = 500Hz)
        @test E.records[:sr][:v] ≈ 500Hz   # stored in units Hz = 1/second
    end

    @testset "monitor! — multiple fields" begin
        E = IF(N = 5)
        monitor!(E, [:v, :fire]; sr = 1000Hz)
        @test haskey(E.records, :v)
        @test haskey(E.records, :fire)
    end

    @testset "graph(model) — returns MetaDiGraph" begin
        E   = IF(N = 10, param = _active_param)
        I   = IF(N = 5,  param = _active_param)
        syn = SpikingSynapse(E, I, :ge; conn = (p = 0.5f0, μ = 1.0f0))
        model = compose(E = E, I = I, syn = syn, silent = true)
        g = graph(model)
        @test g isa MetaDiGraph
        @test nv(g) >= 2
        @test ne(g) >= 1
    end

end
true
