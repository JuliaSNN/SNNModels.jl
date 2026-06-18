using SNNModels
using Test
@load_units

# ── Shared fixtures ───────────────────────────────────────────────────────────

function _make_model()
    exc = IF(N = 10, param = IFParameter(El = -49mV))
    pv  = IF(N = 5,  param = IFParameter(El = -49mV))
    syn = SpikingSynapse(exc, exc, :glu; conn = (p = 0.5f0, μ = 1nS))
    model = compose(; exc, pv, exc_exc = syn, silent = true)
    monitor!(model.pop.exc, [:v, :fire]; sr = 1kHz)
    sim!(model, 300ms)
    return model
end

const _MODEL = _make_model()

# Deep-freeze a reference copy BEFORE any modelcopy calls touch _MODEL.
# All "original unchanged" tests compare against BASELINE, not against live _MODEL.
const BASELINE = deepcopy(_MODEL)

# ── modelcopy ─────────────────────────────────────────────────────────────────

@testset "modelcopy" begin

    @testset "structure preserved" begin
        ck = modelcopy(_MODEL)
        @test ck isa NamedTuple
        @test haskey(ck.pop, :exc) && haskey(ck.pop, :pv)
        @test haskey(ck.syn, :exc_exc)
    end

    @testset "state independent: copy.v !== original.v (deep copy)" begin
        ck = modelcopy(_MODEL)
        @test ck.pop.exc.v !== _MODEL.pop.exc.v
        @test ck.syn.exc_exc.W !== _MODEL.syn.exc_exc.W
        # But values match at copy time
        @test ck.pop.exc.v == _MODEL.pop.exc.v
    end

    @testset "cross-references preserved in copy: syn.fireJ === copy.pop.exc.fire" begin
        ck = modelcopy(_MODEL)
        # modelcopy tracks identity via IdDict, so aliasing is preserved within the copy
        @test ck.syn.exc_exc.fireJ === ck.pop.exc.fire
        # and independent from original
        @test ck.syn.exc_exc.fireJ !== _MODEL.pop.exc.fire
    end

    @testset "records empty after copy (no data, only schema)" begin
        ck = modelcopy(_MODEL)
        @test !haskey(ck.pop.exc.records, :v) || isempty(ck.pop.exc.records[:v])
        @test !haskey(ck.pop.exc.records, :fire) ||
              isempty(get(ck.pop.exc.records[:fire], :times_buf, Float32[]))
    end

    @testset "monitoring schema copied to new model" begin
        ck = modelcopy(_MODEL)
        @test haskey(ck.pop.exc.records, :data)
        @test :v    ∈ ck.pop.exc.records[:data]
        @test :fire ∈ ck.pop.exc.records[:data]
        @test ck.pop.exc.records[:sr][:v] == _MODEL.pop.exc.records[:sr][:v]
    end

    @testset "BASELINE records intact: modelcopy does not touch original" begin
        _ = modelcopy(_MODEL)
        n_baseline = sum(length, spiketimes(BASELINE.pop.exc))
        n_after    = sum(length, spiketimes(_MODEL.pop.exc))
        @test n_baseline == n_after
    end

    @testset "copy has independent time" begin
        ck = modelcopy(_MODEL)
        @test_nowarn sim!(ck, 50ms)
        @test get_time(ck) > get_time(_MODEL)
        @test get_time(_MODEL) == get_time(BASELINE)
    end

end

# ── perturbation_test — structure and return value ────────────────────────────

@testset "perturbation_test" begin

    @testset "add_records=nothing returns pert model" begin
        ck = modelcopy(_MODEL)
        result = perturbation_test(ck, 50ms, identity)
        @test result isa NamedTuple
        @test haskey(result.pop, :exc)
    end

    @testset "returned pert model has recordings" begin
        ck = modelcopy(_MODEL)
        result = perturbation_test(ck, 50ms, identity)
        st = spiketimes(result.pop.exc)
        @test st isa Spiketimes
        @test length(st) == ck.pop.exc.N
    end

    @testset "from_state skips modelcopy — uses provided model directly" begin
        ck = modelcopy(_MODEL)
        result = perturbation_test(_MODEL, 50ms, identity; from_state = ck)
        @test result isa NamedTuple
        @test result === ck      # same object, not a copy
    end

    @testset "add_records stores data under :perturbation" begin
        ck = modelcopy(_MODEL)
        perturbation_test(ck, 100ms, identity; add_records = "cond_a")
        @test haskey(ck.pop.exc.records, :perturbation)
        @test haskey(ck.pop.exc.records[:perturbation], :v)
        @test haskey(ck.pop.exc.records[:perturbation][:v], "cond_a")
        entry = ck.pop.exc.records[:perturbation][:v]["cond_a"][1]
        @test haskey(entry, "interval")
        @test haskey(entry, "data")
        @test entry["data"] isa Matrix{Float32}
        @test size(entry["data"], 1) == ck.pop.exc.N
    end

    @testset ":fire stored as Spiketimes" begin
        ck = modelcopy(_MODEL)
        perturbation_test(ck, 100ms, identity; add_records = "fire_test")
        entry = ck.pop.exc.records[:perturbation][:fire]["fire_test"][1]
        @test entry["data"] isa Spiketimes
        @test length(entry["data"]) == ck.pop.exc.N
    end

    @testset "n auto-increments across repeated calls" begin
        ck = modelcopy(_MODEL)
        clear_perturbation_records!(ck)
        for _ in 1:3
            perturbation_test(ck, 30ms, identity; add_records = "rep")
        end
        @test length(ck.pop.exc.records[:perturbation][:v]["rep"]) == 3
    end

    @testset "train=false uses sim! (no plasticity)" begin
        ck = modelcopy(_MODEL)
        result = perturbation_test(ck, 50ms, identity; train = false)
        @test result isa NamedTuple
    end

    @testset "BASELINE records unchanged after perturbation_test" begin
        ck = modelcopy(_MODEL)
        perturbation_test(ck, 50ms, identity; add_records = "check")
        n_base  = sum(length, spiketimes(BASELINE.pop.exc))
        n_model = sum(length, spiketimes(_MODEL.pop.exc))
        @test n_base == n_model
    end

end

# ── perturbation_record ───────────────────────────────────────────────────────
#
# Workflow: perturbation runs FIRST (from current state at t=T), then baseline
# sim! advances the archive model for the same window [T, T+simtime].
# Both pert and baseline cover the same absolute time range → splice aligns.

@testset "perturbation_record" begin

    function _pert_fixture()
        exc = IF(N = 6, param = IFParameter(El = -49mV))
        m   = compose(; exc, silent = true)
        monitor!(m.pop.exc, [:v, :fire]; sr = 1kHz)
        perturbation_test(m, 200ms, identity; add_records = "ident")
        sim!(m, 200ms)
        return m
    end

    m = _pert_fixture()

    @testset "returns baseline when condition has no data" begin
        v, r = perturbation_record(m.pop.exc, :v, "no_such_cond", 0:1ms:200ms)
        @test v isa Matrix{Float32}
        @test size(v, 1) == m.pop.exc.N
        @test length(r) == size(v, 2)
    end

    @testset "returns Matrix{Float32} for continuous variable" begin
        v, r = perturbation_record(m.pop.exc, :v, "ident", 0:1ms:200ms)
        @test v isa Matrix{Float32}
        @test size(v, 1) == m.pop.exc.N
        @test all(isfinite, v)
    end

    @testset "interval clipped to recorded baseline range" begin
        v, r = perturbation_record(m.pop.exc, :v, "ident", 0:1ms:200ms)
        @test first(r) >= 0f0
        @test last(r)  <= 200f0
        @test length(r) == size(v, 2)
    end

    @testset "spliced and baseline have same shape" begin
        v_base, r_base = perturbation_record(m.pop.exc, :v, "no_such_cond", 0:1ms:200ms)
        v_pert, r_pert = perturbation_record(m.pop.exc, :v, "ident",        0:1ms:200ms)
        @test size(v_base) == size(v_pert)
        @test r_base == r_pert
    end

    @testset ":fire returns Spiketimes" begin
        st, r = perturbation_record(m.pop.exc, :fire, "ident", 0:1ms:200ms)
        @test st isa Spiketimes
        @test length(st) == m.pop.exc.N
    end

    @testset ":fire spikes in perturbed window replaced by pert data" begin
        exc2 = IF(N = 4, param = IFParameter(El = -49mV))
        m2   = compose(; exc2, silent = true)
        monitor!(m2.pop.exc2, [:fire]; sr = 1kHz)

        # from_state: independent model with silenced neurons
        fs = compose(; exc2 = IF(N = 4, param = IFParameter(El = -49mV)), silent = true)
        monitor!(fs.pop.exc2, [:fire]; sr = 1kHz)
        fs.pop.exc2.I .= -2000f0

        perturbation_test(m2, 150ms, identity; from_state = fs, add_records = "silence")
        sim!(m2, 150ms)

        entry  = m2.pop.exc2.records[:perturbation][:fire]["silence"][1]
        tstart, tend = entry["interval"]

        pert_st = entry["data"]::Spiketimes
        @test all(isempty, pert_st)

        base_st = spiketimes(m2.pop.exc2)
        @test any(!isempty, base_st)

        st_spliced, _ = perturbation_record(m2.pop.exc2, :fire, "silence",
                                            Float32(tstart):1f0:Float32(tend))
        for n in eachindex(st_spliced)
            @test all(t -> !(t > tstart && t < tend), st_spliced[n])
        end
    end

end

# ── clear functions ───────────────────────────────────────────────────────────

@testset "clear_perturbation_records!" begin

    function _setup_pert()
        ck = modelcopy(_MODEL)
        clear_perturbation_records!(ck)
        perturbation_test(ck, 30ms, identity; add_records = "A")
        perturbation_test(ck, 30ms, identity; add_records = "A")
        perturbation_test(ck, 30ms, identity; add_records = "B")
        return ck
    end

    @testset "clear all" begin
        ck = _setup_pert()
        clear_perturbation_records!(ck)
        @test !haskey(ck.pop.exc.records, :perturbation)
    end

    @testset "clear by condition leaves other conditions" begin
        ck = _setup_pert()
        clear_perturbation_records!(ck, "A")
        p = ck.pop.exc.records[:perturbation]
        @test !haskey(p[:v], "A")
        @test  haskey(p[:v], "B")
    end

    @testset "clear by condition + variable leaves other variables" begin
        ck = _setup_pert()
        clear_perturbation_records!(ck, "A"; variable = :v)
        p = ck.pop.exc.records[:perturbation]
        @test !haskey(get(p, :v,    Dict()), "A")
        @test  haskey(get(p, :fire, Dict()), "A")
    end

    @testset "clear by condition + n leaves other n" begin
        ck = _setup_pert()
        clear_perturbation_records!(ck, "A"; n = 1)
        p = ck.pop.exc.records[:perturbation][:v]["A"]
        @test !haskey(p, 1)
        @test  haskey(p, 2)
    end

end

@testset "clear_perturbation_monitor!" begin

    function _setup_monitor()
        ck = modelcopy(_MODEL)
        clear_perturbation_records!(ck)
        perturbation_test(ck, 30ms, identity; add_records = "X")
        return ck
    end

    @testset "clear variable removes it, leaves others" begin
        ck = _setup_monitor()
        clear_perturbation_monitor!(ck.pop.exc, :v)
        p = get(ck.pop.exc.records, :perturbation, Dict())
        @test !haskey(p, :v)
        @test  haskey(p, :fire)
    end

    @testset "clear variable + condition" begin
        ck = _setup_monitor()
        clear_perturbation_monitor!(ck.pop.exc, :v, "X")
        p = get(ck.pop.exc.records, :perturbation, Dict())
        @test !haskey(get(p, :v, Dict()), "X")
    end

    @testset "clear variable + condition + n" begin
        ck = modelcopy(_MODEL)
        clear_perturbation_records!(ck)
        perturbation_test(ck, 30ms, identity; add_records = "Y")
        perturbation_test(ck, 30ms, identity; add_records = "Y")
        clear_perturbation_monitor!(ck.pop.exc, :v, "Y"; n = 1)
        p = ck.pop.exc.records[:perturbation][:v]["Y"]
        @test !haskey(p, 1)
        @test  haskey(p, 2)
    end

end

true
