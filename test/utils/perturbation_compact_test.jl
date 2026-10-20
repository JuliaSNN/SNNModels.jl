# Compact perturbation storage (PerturbationStore / PerturbationSpikeStore):
# identity with the legacy Dict-per-window layout (written by a test-local copy of the old flush and
# read by a test-local copy of the old reader), the one-time converter, the errors raised on legacy
# stores, window removal and never-reused ids, accessor, JLD2.
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

# The reader of SNNModels <= 1.9.1 / 2.0.1, exactly, for the legacy layout.
function _old_record(obj, variable::Symbol, condition::String, interval::AbstractRange)
    variable === :fire ? _old_record_fire(obj, condition, interval) :
                         _old_record_dense(obj, variable, condition, interval)
end
function _old_record_dense(obj, variable::Symbol, condition::String, interval::AbstractRange)
    !haskey(obj.records, variable) && return (fill(0.0f0, (0, length(interval))), interval)
    v_interp, r_full = _PS.interpolated_record(obj, variable)
    t_lo   = max(Float32(first(interval)), Float32(first(r_full)))
    t_hi   = min(Float32(last(interval)),  Float32(last(r_full)))
    eval_r = range(t_lo, t_hi; step = Float32(step(interval)))
    N      = size(v_interp, 1)
    result = Float32.(v_interp(1:N, eval_r))
    pert       = get(obj.records, :perturbation, nothing)
    isnothing(pert) && return result, eval_r
    var_store  = get(pert, variable, nothing)
    isnothing(var_store) && return result, eval_r
    cond_store = get(var_store, condition, nothing)
    isnothing(cond_store) && return result, eval_r
    eval_r_vec = collect(eval_r)
    for (_, entry) in sort(collect(cond_store); by = x -> x[1])
        tstart, tend = entry["interval"]
        pert_data    = entry["data"]::Matrix{Float32}
        t_lo_idx = searchsortedfirst(eval_r_vec, Float32(tstart))
        t_hi_idx = searchsortedlast(eval_r_vec,  Float32(tend))
        n_win    = t_hi_idx - t_lo_idx + 1
        n_win > 0 || continue
        T_pert = size(pert_data, 2)
        pert_t = range(Float32(tstart), Float32(tend); length = T_pert)
        ax     = map(1:(ndims(pert_data)-1)) do i; axes(pert_data, i); end
        itp    = _PS.scale(_PS.interpolate(pert_data, _PS.get_interpolator(pert_data)), ax..., pert_t)
        result[:, t_lo_idx:t_hi_idx] .= itp(ax..., eval_r[t_lo_idx:t_hi_idx])
    end
    return result, eval_r
end
function _old_record_fire(obj, condition::String, interval::AbstractRange)
    result = deepcopy(spiketimes(obj; interval))
    pert       = get(obj.records, :perturbation, nothing)
    isnothing(pert) && return result, interval
    var_store  = get(pert, :fire, nothing)
    isnothing(var_store) && return result, interval
    cond_store = get(var_store, condition, nothing)
    isnothing(cond_store) && return result, interval
    for (_, entry) in sort(collect(cond_store); by = x -> x[1])
        tstart, tend = entry["interval"]
        pert_st = entry["data"]::Spiketimes
        @inbounds for n in eachindex(result)
            filter!(t -> !(t > tstart && t < tend), result[n])
            for t in pert_st[n]
                t > tstart && t < tend && push!(result[n], t)
            end
            sort!(result[n])
        end
    end
    return result, interval
end

# windows of a legacy store, in key order, in the format of perturbation_windows
function _legwins(obj, var, cond)
    cs = obj.records[:perturbation][var][cond]
    return [(tstart = Float32(cs[k]["interval"][1]), tend = Float32(cs[k]["interval"][2]),
             data = var === :fire ? _PS._flatten_spikes(cs[k]["data"]) : cs[k]["data"])
            for k in sort!(collect(keys(cs)))]
end

# structural invariants of a store
function _consistent(s)
    nwin = length(s.offsets) - 1
    ok = s.offsets[1] == 1 && issorted(s.offsets) &&
         length(s.tstart) == nwin && length(s.tend) == nwin && length(s.ids) == nwin &&
         issorted(s.ids) && allunique(s.ids) && (isempty(s.ids) || s.next_id > maximum(s.ids))
    if s isa PerturbationStore
        ok &= length(s.data) == (s.offsets[end] - 1) * s.N
    else
        ok &= length(s.times) == s.offsets[end] - 1 && length(s.neuron) == length(s.times)
    end
    return ok
end

# minimal objects to exercise refusals of a flush
struct MockObj
    name::String
    records::Dict{Symbol,Any}
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

# minimal object carrying one store, to read it through the public accessor
struct PerturbationHolder
    records::Dict{Symbol,Any}
end
PerturbationHolder(s) = PerturbationHolder(Dict{Symbol,Any}(:perturbation => Dict{Symbol,Any}(:v => Dict{String,Any}("c" => s))))

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
            old = _old_record(L, var, cond, iv)
            new = perturbation_record(E, var, cond, iv)
            cnv = perturbation_record(m_conv.pop.E, var, cond, iv)
            @test _same_rec(old, new)
            @test _same_rec(old, cnv)
        end
        # T = 1 window: the reader fails in the same way on both layouts (Interpolations rejects a
        # one-sample scaled axis; present since before the compact layout)
        ivt = 0ms:1ms:600ms
        e_old = try _old_record(L, :v, "t1", ivt); nothing catch e; e end
        e_new = try perturbation_record(E, :v, "t1", ivt); nothing catch e; e end
        @test e_old isa ArgumentError && e_new isa ArgumentError && e_old.msg == e_new.msg
        @test perturbation_windows(E, :v, "t1")[1].data == _legwins(L, :v, "t1")[1].data
        @test _same_rec(_old_record(L, :v, "t1", 0ms:1ms:50ms), perturbation_record(E, :v, "t1", 0ms:1ms:50ms))
        # unknown condition: baseline only
        @test _same_rec(_old_record(L, :v, "zz", _IVS[1]), perturbation_record(E, :v, "zz", _IVS[1]))
        # idempotent
        @test isempty(compact_perturbation_records!(m_conv))
        @test _same_rec(_old_record(L, :v, "a", _IVS[1]), perturbation_record(m_conv.pop.E, :v, "a", _IVS[1]))
        @test m_conv.pop.E.records[:perturbation][:v]["a"].next_id == 5
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
            wo = _legwins(L, var, cond)
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

    @testset "clear_perturbation_records! (compact and converted stores)" begin
        for (label, mk) in (("compact", () -> deepcopy(m)),
                            ("converted", () -> (g = deepcopy(m_leg); compact_perturbation_records!(g); g)))
            mm = mk()
            e = mm.pop.E
            # n = 2 of condition a, variable v: later windows keep their ids
            clear_perturbation_records!(mm, "a"; variable = :v, n = 2)
            @test length(e.records[:perturbation][:v]["a"]) == 3
            @test e.records[:perturbation][:v]["a"].ids == [1, 3, 4]
            @test !haskey(e.records[:perturbation][:v]["a"], 2)
            @test length(e.records[:perturbation][:fire]["a"]) == 4
            @test length(e.records[:perturbation][:I]["a"]) == 4
            # flush after a deletion never overwrites an existing window
            pm = last(perts[1])
            SNNModels._flush_perturbation!(mm, pm, "a")
            @test e.records[:perturbation][:v]["a"].ids == [1, 3, 4, 5]
            @test length(e.records[:perturbation][:I]["a"]) == 5
            wv = perturbation_windows(e, :v, "a")
            @test wv[3].tstart == _legwins(m_leg.pop.E, :v, "a")[4].tstart   # old n = 4
            mm = mk(); e = mm.pop.E
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
        # clearing a whole condition / variable / everything still works on a legacy store
        g = deepcopy(m_leg); e = g.pop.E
        clear_perturbation_records!(g, "b")
        @test !haskey(e.records[:perturbation][:v], "b") && e.records[:perturbation][:v]["a"] isa Dict
        clear_perturbation_monitor!(e, :I)
        @test !haskey(e.records[:perturbation], :I)
        clear_perturbation_monitor!(e, :v, "a")
        @test !haskey(e.records[:perturbation][:v], "a")
        clear_perturbation_records!(g)
        @test !haskey(e.records, :perturbation)
    end

    @testset "window removal: first, middle, last, all" begin
        for seq in ([1], [2], [3], [4], [1, 4], [4, 1], [2, 3], [3, 2, 1], [2, 4, 1, 3])
            a = deepcopy(m); b = deepcopy(m_leg)
            for n in seq
                for var in vars
                    clear_perturbation_records!(a, "a"; variable = var, n = n)
                    delete!(b.pop.E.records[:perturbation][var]["a"], n)
                end
                for var in vars
                    s = a.pop.E.records[:perturbation][var]["a"]
                    @test _consistent(s)
                    @test s.ids == sort!(collect(keys(b.pop.E.records[:perturbation][var]["a"])))
                    @test s.next_id == 5
                    for iv in _IVS
                        @test _same_rec(perturbation_record(a.pop.E, var, "a", iv), _old_record(b.pop.E, var, "a", iv))
                    end
                end
                for (w, wl) in zip(perturbation_windows(a.pop.E, :v, "a"), _legwins(b.pop.E, :v, "a"))
                    @test w.tstart === wl.tstart && w.tend === wl.tend && w.data == wl.data
                end
            end
            # the other conditions are untouched
            @test _same_rec(perturbation_record(a.pop.E, :v, "b", _IVS[1]), _old_record(b.pop.E, :v, "b", _IVS[1]))
        end
        # deleting a window that does not exist changes nothing
        a = deepcopy(m)
        clear_perturbation_records!(a, "a"; variable = :v, n = 99)
        @test a.pop.E.records[:perturbation][:v]["a"].ids == 1:4
        # delete all one by one, then flush again: ids continue, nothing is reused
        pm = last(perts[1])
        a = deepcopy(m)
        for n in 1:4, var in vars
            clear_perturbation_records!(a, "a"; variable = var, n = n)
        end
        for var in vars
            s = a.pop.E.records[:perturbation][var]["a"]
            @test isempty(s) && s.offsets == [1] && s.next_id == 5
            @test isempty(perturbation_windows(a.pop.E, var, "a"))
            @test _same_rec(perturbation_record(a.pop.E, var, "a", _IVS[1]),
                            perturbation_record(a.pop.E, var, "zz", _IVS[1]))     # baseline only
        end
        SNNModels._flush_perturbation!(a, pm, "a")
        SNNModels._flush_perturbation!(a, pm, "a")
        r = deepcopy(m_leg); clear_perturbation_records!(r)
        _legacy_flush!(r, pm, "a")
        for var in vars
            s = a.pop.E.records[:perturbation][var]["a"]
            @test s.ids == [5, 6] && s.next_id == 7 && _consistent(s)
        end
        r2 = deepcopy(r); _legacy_flush!(r2, pm, "a")
        for var in vars, iv in _IVS
            @test _same_rec(perturbation_record(a.pop.E, var, "a", iv), _old_record(r2.pop.E, var, "a", iv))
        end
        # the counter survives a JLD2 round trip
        f = joinpath(mktempdir(), "ids.jld2")
        JLD2.jldsave(f; model = a)
        rl = JLD2.load(f, "model")
        @test rl.pop.E.records[:perturbation][:v]["a"].next_id == 7
        clear_perturbation_records!(rl, "a"; variable = :v, n = 5)
        clear_perturbation_records!(rl, "a"; variable = :v, n = 6)
        SNNModels._flush_perturbation!(rl, pm, "a")
        @test rl.pop.E.records[:perturbation][:v]["a"].ids == [7]
    end

    @testset "legacy records: errors, no mixed state" begin
        # flush into a legacy (variable, condition): error naming object, variable, condition; nothing written
        for var in vars
            g = deepcopy(m)
            e = g.pop.E
            e.records[:perturbation][var]["a"] = deepcopy(L.records[:perturbation][var]["a"])
            before = Dict(v => length(e.records[:perturbation][v]["a"]) for v in vars)
            pm = last(perts[1])
            err = try SNNModels._flush_perturbation!(g, pm, "a"); nothing catch ex; ex end
            @test err isa ArgumentError
            @test occursin("IF \"E\"", err.msg) && occursin("variable :$var,", err.msg) &&
                  occursin("condition \"a\"", err.msg) && occursin("pre-1.9.2 layout", err.msg) &&
                  occursin("compact_perturbation_records!", err.msg)
            @test e.records[:perturbation][var]["a"] isa Dict
            @test all(length(e.records[:perturbation][v]["a"]) == before[v] for v in vars)   # atomic
            @test_throws ArgumentError SNNModels._flush_component!(e, pm.pop.E, "a")
        end
        # readers
        for var in vars
            err = try perturbation_windows(L, var, "a"); nothing catch ex; ex end
            @test err isa ArgumentError && occursin("variable :$var,", err.msg) && occursin("condition \"a\"", err.msg)
            @test_throws ArgumentError perturbation_record(L, var, "a", _IVS[1])
            @test perturbation_record(L, var, "zz", _IVS[1]) isa Tuple          # no stored condition: baseline
        end
        @test isempty(perturbation_windows(L, :v, "zz"))
        # deleting one window of a legacy store
        g = deepcopy(m_leg)
        err = try clear_perturbation_records!(g, "a"; variable = :v, n = 2); nothing catch ex; ex end
        @test err isa ArgumentError && occursin("variable :v,", err.msg) && occursin("condition \"a\"", err.msg) &&
              occursin("compact_perturbation_records!", err.msg)
        @test length(g.pop.E.records[:perturbation][:v]["a"]) == 4
        @test_throws ArgumentError clear_perturbation_monitor!(g.pop.E, :v, "a"; n = 1)
        # after the conversion everything works
        compact_perturbation_records!(g)
        clear_perturbation_records!(g, "a"; variable = :v, n = 2)
        @test g.pop.E.records[:perturbation][:v]["a"].ids == [1, 3, 4]
        # legacy keys with gaps are kept as ids, next_id = max(keys) + 1
        h = deepcopy(m_leg)
        delete!(h.pop.E.records[:perturbation][:v]["a"], 2)
        compact_perturbation_records!(h)
        @test h.pop.E.records[:perturbation][:v]["a"].ids == [1, 3, 4]
        @test h.pop.E.records[:perturbation][:v]["a"].next_id == 5
        h = deepcopy(m_leg)
        delete!(h.pop.E.records[:perturbation][:v]["a"], 4)
        compact_perturbation_records!(h)
        @test h.pop.E.records[:perturbation][:v]["a"].next_id == 4
        SNNModels._flush_perturbation!(h, last(perts[1]), "a")
        @test h.pop.E.records[:perturbation][:v]["a"].ids == [1, 2, 3, 4]
        # a foreign value (interim NamedTuple layout) in place of a store
        nt = (data = zeros(Float32, 8, 3), offsets = [1, 4], tstart = Float32[0], tend = Float32[1])
        for var in vars
            g = deepcopy(m); e = g.pop.E
            e.records[:perturbation][var]["a"] = nt
            for f in (() -> perturbation_windows(e, var, "a"),
                      () -> perturbation_record(e, var, "a", _IVS[1]),
                      () -> SNNModels._flush_perturbation!(g, last(perts[1]), "a"),
                      () -> clear_perturbation_records!(g, "a"; variable = var, n = 1),
                      () -> compact_perturbation_records!(g))
                err = try f(); nothing catch ex; ex end
                @test err isa ArgumentError
                @test occursin("NamedTuple", err.msg)
                @test occursin("\"a\"", err.msg) && occursin("$var", err.msg)
            end
            @test e.records[:perturbation][var]["a"] === nt          # untouched
            clear_perturbation_records!(g, "a"; variable = var)       # whole condition: allowed
            @test !haskey(e.records[:perturbation][var], "a")
        end
        # such a value survives a JLD2 round trip (only readers/flush refuse it)
        g = deepcopy(m); g.pop.E.records[:perturbation][:v]["a"] = nt
        f = joinpath(mktempdir(), "nt.jld2")
        JLD2.jldsave(f; model = g)
        @test JLD2.load(f, "model").pop.E.records[:perturbation][:v]["a"].offsets == [1, 4]
        # N-dimensional dense window: refused at flush time, nothing stored
        mk_meta() = Dict{Symbol,Any}(:mode => Dict(:v => :dense), :allocated => Dict(:v => 0))
        pobj = MockObj("P", Dict{Symbol,Any}(:meta => mk_meta(), :v => zeros(Float32, 3, 2, 5),
                                              :data => [:v], :start_time => Dict(:v => 0f0), :end_time => Dict(:v => 1f0)))
        oobj = MockObj("O", Dict{Symbol,Any}())
        err = try SNNModels._flush_component!(oobj, pobj, "c3d"); nothing catch ex; ex end
        @test err isa ArgumentError && occursin("O", err.msg) && occursin("variable :v,", err.msg) &&
              occursin("condition \"c3d\"", err.msg) && occursin("3 dimensions", err.msg)
        @test !haskey(oobj.records, :perturbation)
    end

    @testset "converter refuses what it cannot store" begin
        g = deepcopy(m_leg)
        leg = g.pop.E.records[:perturbation][:v]["a"]
        leg[3]["data"] = zeros(Float32, 8, 2, 2)           # 3-dimensional legacy window
        err = try compact_perturbation_records!(g); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("n = 3", err.msg) && occursin("\"a\"", err.msg) && occursin("variable v", err.msg) &&
              occursin("IF \"E\"", err.msg)
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
        @test r.pop.E.records[:perturbation][:v]["a"].next_id == 5
        for var in vars, cond in conds, iv in _IVS
            @test _same_rec(perturbation_record(E, var, cond, iv), perturbation_record(r.pop.E, var, cond, iv))
        end
        info = (id = 1,)
        save_model(model = m, path = dir, name = "pert", info = info)
        d = load_data(dir, "pert", info)
        @test d.model.pop.E.records[:perturbation][:fire]["a"].next_id == 5
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

    @testset "CSR offsets, T = 0 windows" begin
        s = PerturbationStore(3)
        @test s.offsets == [1] && isempty(s) && length(s) == 0 && s.next_id == 1
        SNNModels._push_window!(s, rand(Float32, 3, 4), 1f0, 2f0)
        SNNModels._push_window!(s, zeros(Float32, 3, 0), 2f0, 3f0)      # T = 0
        SNNModels._push_window!(s, rand(Float32, 3, 2), 3f0, 4f0)
        @test s.offsets == [1, 5, 5, 7] && length(s) == 3
        ws = perturbation_windows(PerturbationHolder(s), :v, "c")
        @test [size(w.data) for w in ws] == [(3, 4), (3, 0), (3, 2)]
        keep = copy(ws[3].data)
        SNNModels._delete_window!(s, 1)
        @test s.offsets == [1, 1, 3] && s.ids == [2, 3] && length(s.data) == 6
        @test perturbation_windows(PerturbationHolder(s), :v, "c")[2].data == keep
        SNNModels._delete_window!(s, 2); SNNModels._delete_window!(s, 3)
        @test isempty(s) && s.offsets == [1] && isempty(s.data)
        @test s.next_id == 4                                            # ids are not reused
        SNNModels._push_window!(s, rand(Float32, 3, 1), 4f0, 5f0)
        @test s.ids == [4] && s.next_id == 5
        sp = PerturbationSpikeStore(2)
        SNNModels._push_window!(sp, [Float32[], Float32[]], 1f0, 2f0)   # zero spikes
        SNNModels._push_window!(sp, [Float32[1.5f0], Float32[2f0, 2.5f0]], 2f0, 3f0)
        @test sp.offsets == [1, 1, 4] && sp.neuron == Int32[1, 2, 2]
        SNNModels._delete_window!(sp, 1)
        @test sp.offsets == [1, 4] && sp.ids == [2] && sp.next_id == 3
    end

    @testset "population with N = 0" begin
        E0 = IF(N = 0, name = "E0")
        m0 = compose(; E0, silent = true)
        monitor!(m0.pop.E0, [:v, :fire]; sr = 1kHz)
        sim!(m0, 50ms)
        perts0 = Tuple{String,Any}[]
        for d in (20ms, 30ms)
            pm = perturbation_test(m0, d; train = false)
            push!(perts0, ("c", pm))
            SNNModels._flush_perturbation!(m0, pm, "c")
            sim!(m0, 10ms)
        end
        # 0.5 ms: no dense sample, only a :fire window
        pm = perturbation_test(m0, 0.5ms; train = false)
        SNNModels._flush_perturbation!(m0, pm, "c"); push!(perts0, ("c", pm))
        sim!(m0, 10ms)
        l0 = deepcopy(m0); clear_perturbation_records!(l0)
        for (c, pm) in perts0
            _legacy_flush!(l0, pm, c)
        end
        e, l = m0.pop.E0, l0.pop.E0
        @test e.records[:perturbation][:v]["c"] isa PerturbationStore
        @test length(e.records[:perturbation][:v]["c"]) == 2
        @test length(e.records[:perturbation][:fire]["c"]) == 3
        for var in (:v, :fire)
            @test _same_rec(perturbation_record(e, var, "c", 0ms:1ms:150ms), _old_record(l, var, "c", 0ms:1ms:150ms))
            @test length(perturbation_windows(e, var, "c")) == length(_legwins(l, var, "c"))
        end
        @test [size(w.data) for w in perturbation_windows(e, :v, "c")] == [(0, 20), (0, 30)]
        # converter
        compact_perturbation_records!(l0)
        @test l.records[:perturbation][:v]["c"] isa PerturbationStore
        @test l.records[:perturbation][:v]["c"].offsets == e.records[:perturbation][:v]["c"].offsets
        # flush after conversion, clear
        SNNModels._flush_perturbation!(l0, last(perts0[1]), "c")
        @test length(l.records[:perturbation][:v]["c"]) == 3
        clear_perturbation_records!(l0, "c"; variable = :v, n = 1)
        @test l.records[:perturbation][:v]["c"].ids == [2, 3]
        # JLD2
        f = joinpath(mktempdir(), "n0.jld2")
        JLD2.jldsave(f; model = m0)
        r = JLD2.load(f, "model")
        for var in (:v, :fire)
            @test _same_rec(perturbation_record(e, var, "c", 0ms:1ms:150ms), perturbation_record(r.pop.E0, var, "c", 0ms:1ms:150ms))
        end
        clear_perturbation_records!(m0)
        @test !haskey(e.records, :perturbation)
    end

    @testset "layout version" begin
        bad = PerturbationStore(2, 3, Float32[], [1], Float32[], Float32[], Int[], 1)
        @test_throws ArgumentError SNNModels._push_window!(bad, rand(Float32, 3, 2), 1f0, 2f0)
        err = try SNNModels._push_window!(bad, rand(Float32, 3, 2), 1f0, 2f0); nothing catch e; e end
        @test occursin("version 2", err.msg) && occursin("upgrade SNNModels", err.msg)
        mm = deepcopy(m)
        mm.pop.E.records[:perturbation][:v]["a"] = bad
        @test_throws ArgumentError perturbation_windows(mm.pop.E, :v, "a")
        @test_throws ArgumentError perturbation_record(mm.pop.E, :v, "a", 0ms:1ms:100ms)
        @test_throws ArgumentError clear_perturbation_records!(mm, "a"; variable = :v, n = 1)
        badf = PerturbationSpikeStore(2, 2, Float32[], Int32[], [1], Float32[], Float32[], Int[], 1)
        mm.pop.E.records[:perturbation][:fire]["a"] = badf
        @test_throws ArgumentError perturbation_record(mm.pop.E, :fire, "a", 0ms:1ms:100ms)
    end
end

true
