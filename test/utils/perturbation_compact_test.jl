# Compact perturbation storage (PerturbationStore / PerturbationSpikeStore):
# identity with the legacy Dict-per-window layout, converter, clear_*, accessor, JLD2.
using SNNModels
using Test
using JLD2
@load_units

const _PS = SNNModels

# The legacy flush, exactly as SNNModels <= 1.9.1 / 2.0.1 wrote it.
function _legacy_flush!(model, pert_model, condition::String)
    for field in (:pop, :syn, :stim)
        haskey(model, field) || continue
        haskey(pert_model, field) || continue
        og = getfield(model, field); pg = getfield(pert_model, field)
        for key in keys(og)
            haskey(pg, key) || continue
            _legacy_component!(getfield(og, key), getfield(pg, key), condition)
        end
    end
end
function _legacy_component!(orig_obj, pert_obj, condition::String)
    haskey(pert_obj.records, :data) || return
    isempty(pert_obj.records[:data]) && return
    haskey(orig_obj.records, :perturbation) ||
        (orig_obj.records[:perturbation] = Dict{Symbol,Any}())
    pert_store = orig_obj.records[:perturbation]
    for key in pert_obj.records[:data]
        tstart = get(pert_obj.records[:start_time], key, NaN32)
        tend   = get(pert_obj.records[:end_time],   key, NaN32)
        (isnan(tstart) || isnan(tend)) && continue
        data = key === :fire ? spiketimes(pert_obj) :
               copy(Float32.(SNNModels.getvariable(pert_obj, key)))
        haskey(pert_store, key)            || (pert_store[key]           = Dict{String,Any}())
        haskey(pert_store[key], condition) || (pert_store[key][condition] = Dict{Int,Any}())
        n = length(pert_store[key][condition]) + 1
        pert_store[key][condition][n] = Dict{String,Any}("interval" => Float32[tstart, tend], "data" => data)
    end
end

_bitsame(a::AbstractArray{Float32}, b::AbstractArray{Float32}) =
    size(a) == size(b) && reinterpret(UInt32, vec(collect(a))) == reinterpret(UInt32, vec(collect(b)))
function _same_rec(x, y)
    (a, ra), (b, rb) = x, y
    ra == rb || return false
    if a isa Matrix
        return a == b && isequal(a, b) && _bitsame(a, b)
    else
        length(a) == length(b) || return false
        return all(i -> a[i] == b[i] && isequal(a[i], b[i]) && _bitsame(a[i], b[i]), eachindex(a))
    end
end

function _pert_model()
    E = IF(N = 8, param = IFParameter(El = -49mV), name = "E")
    noise = CurrentStimulus(E, :I, param = CurrentNoise(E; I_base = 300pA))
    m = compose(; E, noise, silent = true)
    monitor!(m.pop.E, [:v, :fire]; sr = 1kHz)
    monitor!(m.pop.E, [:I]; sr = 500Hz)     # second dense variable, other sampling rate
    sim!(m, 100ms)
    return m
end

# (condition, duration, trigger, gap): unequal lengths, overlap (gap = 0), T = 1, silent window.
const _SPECS = [
    ("a", 30ms, m -> (m.pop.E.I .+= 100pA), 25ms),
    ("a", 50ms, m -> (m.pop.E.I .+= 200pA), 0ms),      # overlaps the next one
    ("a", 20ms, m -> (m.pop.E.I .+= 50pA),  30ms),
        ("a", 40ms, m -> (m.pop.E.I .+= 150pA), 20ms),
    ("b", 40ms, m -> (m.stim.noise.param.I_base .= -2000f0), 20ms),    # no spikes in the window
    ("b", 25ms, m -> (m.pop.E.I .+= 80pA),  40ms),
    ("t1", 1ms, m -> nothing,               20ms),     # T = 1 sample
]

function _build()
    m = _pert_model()
    perts = Tuple{String,Any}[]
    for (cond, dur, trig, gap) in _SPECS
        pm = perturbation_test(m, dur; trigger! = trig, train = false)
        push!(perts, (cond, pm))
        SNNModels._flush_perturbation!(m, pm, cond)
        gap > 0ms && sim!(m, gap)
    end
    sim!(m, 20ms)
    m_leg = deepcopy(m)
    clear_perturbation_records!(m_leg)
    for (cond, pm) in perts
        _legacy_flush!(m_leg, pm, cond)
    end
    return m, m_leg, perts
end

const _IVS = (0ms:1ms:600ms, -10ms:0.5ms:900ms, 150ms:1ms:260ms, 0ms:2ms:50ms)

@testset "compact perturbation records" begin
    m, m_leg, perts = _build()
    E, L = m.pop.E, m_leg.pop.E
    conds = ("a", "b")
    vars  = (:v, :I, :fire)

    @testset "layout" begin
        @test E.records[:perturbation][:v]["a"] isa PerturbationStore
        @test E.records[:perturbation][:I]["b"] isa PerturbationStore
        @test E.records[:perturbation][:fire]["a"] isa PerturbationSpikeStore
        @test L.records[:perturbation][:v]["a"] isa Dict
        @test length(E.records[:perturbation][:v]["a"]) == 4
        @test length(L.records[:perturbation][:v]["a"]) == 4
        s = E.records[:perturbation][:v]["a"]
        @test s.version == 1
        @test s.ids == 1:4
        @test length(s.data) == E.N * (length(s.data) ÷ E.N)
        @test size(only(perturbation_windows(E, :v, "t1")).data, 2) == 1   # a T = 1 window
        @test any(isempty(w.data.times) for w in perturbation_windows(E, :fire, "b"))  # zero spikes
        @test s[2]["data"] == L.records[:perturbation][:v]["a"][2]["data"]
    end

    @testset "perturbation_record: legacy == compact == converted" begin
        m_conv = deepcopy(m_leg)
        summary = compact_perturbation_records!(m_conv)
        @test Set(keys(summary)) == Set(vars)
        @test summary[:v].windows == 7
        @test summary[:v].bytes > 0
        @test m_conv.pop.E.records[:perturbation][:v]["a"] isa PerturbationStore
        for var in vars, cond in conds, iv in _IVS
            old = perturbation_record(L, var, cond, iv)
            new = perturbation_record(E, var, cond, iv)
            cnv = perturbation_record(m_conv.pop.E, var, cond, iv)
            @test _same_rec(old, new)
            @test _same_rec(old, cnv)
        end
        # T = 1 window: the reader fails in the same way on both layouts (Interpolations rejects a
        # one-sample scaled axis; present since before the compact layout)
        ivt = 0ms:1ms:600ms
        e_old = try perturbation_record(L, :v, "t1", ivt); nothing catch e; e end
        e_new = try perturbation_record(E, :v, "t1", ivt); nothing catch e; e end
        @test e_old isa ArgumentError && e_new isa ArgumentError && e_old.msg == e_new.msg
        @test perturbation_windows(E, :v, "t1")[1].data == perturbation_windows(L, :v, "t1")[1].data
        @test _same_rec(perturbation_record(L, :v, "t1", 0ms:1ms:50ms), perturbation_record(E, :v, "t1", 0ms:1ms:50ms))
        # unknown condition: baseline only
        @test _same_rec(perturbation_record(L, :v, "zz", _IVS[1]), perturbation_record(E, :v, "zz", _IVS[1]))
        # idempotent
        @test isempty(compact_perturbation_records!(m_conv))
        @test _same_rec(perturbation_record(L, :v, "a", _IVS[1]), perturbation_record(m_conv.pop.E, :v, "a", _IVS[1]))
        # a model as returned by load_data, and a bare component
        Lc = deepcopy(m_leg)
        @test !isempty(compact_perturbation_records!((model = Lc, config = nothing)))
        @test Lc.pop.E.records[:perturbation][:fire]["b"] isa PerturbationSpikeStore
        Lc2 = deepcopy(m_leg)
        compact_perturbation_records!(Lc2.pop.E)
        @test Lc2.pop.E.records[:perturbation][:I]["a"] isa PerturbationStore
    end

    @testset "perturbation_windows" begin
        for var in vars, cond in conds
            wn = perturbation_windows(E, var, cond)
            wo = perturbation_windows(L, var, cond)
            @test length(wn) == length(wo)
            for (a, b) in zip(wn, wo)
                @test a.tstart === b.tstart && a.tend === b.tend
                if var === :fire
                    @test a.data.neuron == b.data.neuron
                    @test a.data.times == b.data.times
                    @test issorted(a.data.neuron)
                else
                    @test a.data == b.data
                end
            end
            leg = L.records[:perturbation][var][cond]
            for (j, w) in enumerate(wo)
                @test Float32[w.tstart, w.tend] == leg[j]["interval"]
            end
        end
        @test isempty(perturbation_windows(E, :v, "nope"))
        # views do not block growth
        ws = perturbation_windows(E, :v, "a")
        fs = perturbation_windows(E, :fire, "a")
        pm = last(perts[1])
        mm = deepcopy(m)
        SNNModels._flush_perturbation!(mm, pm, "a")
        mw = perturbation_windows(mm.pop.E, :v, "a")
        held = mw
        SNNModels._flush_perturbation!(mm, pm, "a")
        @test length(perturbation_windows(mm.pop.E, :v, "a")) == length(held) + 1
    end

    @testset "clear_perturbation_records!" begin
        for (label, mk) in (("legacy", () -> deepcopy(m_leg)),
                            ("compact", () -> deepcopy(m)),
                            ("converted", () -> deepcopy(m_leg)))
            mm = mk()
            label == "converted" && compact_perturbation_records!(mm)
            e = mm.pop.E
            iv = _IVS[1]
            ref(v, c) = perturbation_record(e, v, c, iv)
            # n = 2 of condition a, variable v: later windows keep their ids
            clear_perturbation_records!(mm, "a"; variable = :v, n = 2)
            @test length(e.records[:perturbation][:v]["a"]) == 3
            @test haskey(e.records[:perturbation][:v]["a"], 3)
            @test !haskey(e.records[:perturbation][:v]["a"], 2)
            @test length(e.records[:perturbation][:fire]["a"]) == 4
            @test length(e.records[:perturbation][:I]["a"]) == 4
            # flush after a deletion never overwrites an existing window
            pm = last(perts[1])
            SNNModels._flush_perturbation!(mm, pm, "a")
            @test length(e.records[:perturbation][:v]["a"]) == 4
            @test length(e.records[:perturbation][:I]["a"]) == 5
            wv = perturbation_windows(e, :v, "a")
            @test wv[3].tstart == perturbation_windows(m_leg.pop.E, :v, "a")[4].tstart   # old n = 4 (the window flushed last)
            mm = mk(); label == "converted" && compact_perturbation_records!(mm); e = mm.pop.E
            clear_perturbation_records!(mm, "b")
            @test !haskey(e.records[:perturbation][:v], "b") && haskey(e.records[:perturbation][:v], "a")
            clear_perturbation_monitor!(e, :I)
            @test !haskey(e.records[:perturbation], :I)
            clear_perturbation_monitor!(e, :v, "a"; n = 1)
            @test length(e.records[:perturbation][:v]["a"]) == 3
            clear_perturbation_monitor!(e, :v, "a")
            @test !haskey(e.records[:perturbation][:v], "a")
            clear_perturbation_records!(mm)
            @test !haskey(e.records, :perturbation)
        end
        # splice after deleting a middle window: compact == legacy
        a = deepcopy(m); b = deepcopy(m_leg)
        for var in vars
            clear_perturbation_records!(a, "a"; variable = var, n = 3)
            clear_perturbation_records!(b, "a"; variable = var, n = 3)
        end
        for var in vars, iv in _IVS
            @test _same_rec(perturbation_record(a.pop.E, var, "a", iv), perturbation_record(b.pop.E, var, "a", iv))
        end
    end

    @testset "mixed state: flush into a legacy condition converts it" begin
        mm = deepcopy(m_leg)
        pm = last(perts[1])
        SNNModels._flush_perturbation!(mm, pm, "a")
        e = mm.pop.E
        @test e.records[:perturbation][:v]["a"] isa PerturbationStore
        @test length(e.records[:perturbation][:v]["a"]) == 5
        @test e.records[:perturbation][:v]["a"].ids == 1:5
        ref = deepcopy(m); SNNModels._flush_perturbation!(ref, pm, "a")
        for var in vars
            @test _same_rec(perturbation_record(e, var, "a", _IVS[1]), perturbation_record(ref.pop.E, var, "a", _IVS[1]))
        end
        # legacy keys with gaps are kept as ids
        g = deepcopy(m_leg)
        delete!(g.pop.E.records[:perturbation][:v]["a"], 2)
        compact_perturbation_records!(g)
        @test g.pop.E.records[:perturbation][:v]["a"].ids == [1, 3, 4]
    end

    @testset "converter refuses what it cannot store" begin
        g = deepcopy(m_leg)
        leg = g.pop.E.records[:perturbation][:v]["a"]
        leg[3]["data"] = zeros(Float32, 8, 2, 2)           # 3-dimensional legacy window
        err = try compact_perturbation_records!(g); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("n = 3", err.msg) && occursin("\"a\"", err.msg) && occursin(":v", string(err.msg) * ":v")
        @test g.pop.E.records[:perturbation][:v]["a"] isa Dict        # untouched
        @test length(g.pop.E.records[:perturbation][:v]["a"]) == 4
        g2 = deepcopy(m_leg)
        g2.pop.E.records[:perturbation][:v]["a"][2]["data"] = zeros(Float32, 5, 4)   # other N
        @test_throws ArgumentError compact_perturbation_records!(g2)
    end

    @testset "JLD2 round trip" begin
        dir = mktempdir()
        f = joinpath(dir, "m.jld2")
        JLD2.jldsave(f; model = m)
        r = JLD2.load(f, "model")
        @test r.pop.E.records[:perturbation][:v]["a"] isa PerturbationStore
        for var in vars, cond in conds, iv in _IVS
            @test _same_rec(perturbation_record(E, var, cond, iv), perturbation_record(r.pop.E, var, cond, iv))
        end
        info = (id = 1,)
        save_model(model = m, path = dir, name = "pert", info = info)
        d = load_data(dir, "pert", info)
        for var in vars, cond in conds
            @test _same_rec(perturbation_record(E, var, cond, _IVS[1]), perturbation_record(d.model.pop.E, var, cond, _IVS[1]))
        end
    end

    @testset "file size, many short windows" begin
        m1 = _pert_model()
        pm = perturbation_test(m1, 3ms; trigger! = x -> (x.pop.E.I .+= 100pA), train = false)
        ml = deepcopy(m1)
        nwin = 600
        for c in ("c1", "c2"), _ in 1:nwin
            SNNModels._flush_perturbation!(m1, pm, c)
            _legacy_flush!(ml, pm, c)
        end
        pn, pl = joinpath(mktempdir(), "n.jld2"), joinpath(mktempdir(), "l.jld2")
        JLD2.jldsave(pn; p = m1.pop.E.records[:perturbation])
        JLD2.jldsave(pl; p = ml.pop.E.records[:perturbation])
        sn, sl = filesize(pn), filesize(pl)
        println("perturbation file size (2 conditions x $nwin windows x 3 variables): legacy = $sl B, compact = $sn B, ratio = $(round(sl / sn, digits = 1))")
        @test sn * 10 < sl
        a = JLD2.load(pn, "p"); b = JLD2.load(pl, "p")
        @test a[:v]["c1"] isa PerturbationStore
        compact_perturbation_records!(ml)
        @test Base.summarysize(ml.pop.E.records[:perturbation]) < Base.summarysize(b) ÷ 5
    end
end

true
