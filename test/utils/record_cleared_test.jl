using SNNModels
using Test
@load_units

# record() on data that is absent gives an ArgumentError naming the available keys (it was a bare KeyError after load_model)
@testset "record - cleared or missing records" begin
    E = IF(N = 20, name = "E")
    I = IF(N = 10, name = "I")
    syn = SpikingSynapse(E, I, :ge; conn = (p = 0.5, μ = 2.0))
    monitor!(syn, [:W])
    monitor!(E, [:v])
    model = compose(; E, I, syn, silent = true)
    sim!(model, 50ms)
    @test record(syn, :W) !== nothing

    dir = mktempdir()
    save_model(; model, path = dir, name = "rec", info = (a = 1,))

    # model file: records cleared, the monitoring set-up is kept
    cleared = load_model(dir, "rec", (a = 1,)).model
    err = try; record(cleared.syn.syn, :W); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin(":W", sprint(showerror, err)) || occursin("`W`", sprint(showerror, err))
    @test occursin("load_data", sprint(showerror, err))
    err = try; record(cleared.pop.E, :v, interpolate = false); nothing; catch e; e; end
    @test err isa ArgumentError

    # data file: the samples are present
    kept = load_data(dir, "rec", (a = 1,)).model
    @test size(record(kept.syn.syn, :W, interpolate = false), 2) > 1
    @test record(kept.syn.syn, :W) !== nothing

    # a key that is not monitored: the message lists what is
    err = try; record(E, :nothing_here); nothing; catch e; e; end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("nothing_here", msg)
    @test occursin(":v", msg)
end
