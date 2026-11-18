# Multipod with the library utilities: the checks that decided its layout (one concrete type per number of dendrites, with
# named fields `d1..dN`, generated when SNNModels is loaded).
#   (a) save_model / load_model / load_data round trip, in this process and in a fresh Julia process that only runs
#       `using SNNModels` (JLD2 rebuilds the population from the name of its type);
#   (b) check_loaded_model;
#   (c) monitor! / record of :v_s, :v_d<k>, :fire and receptor variables;
#   (d) sim!/train! in threaded chunks equal to the single-thread run (also in multipod.jl);
#   (e) convert_precision of the synapse models between Float32 and Float64, Float64 populations;
#   (f) change_plasticity / with_plasticity on a SpikingSynapse onto :d3;
#   (g) deepcopy and modelcopy;
#   (h) compose, graph and name-based access.
using Test
using Random
using SNNModels
@load_units

# a small model: an IF population drives d3 of a 4-dendrite Multipod; constant currents, so that the run is deterministic
function _mpio_model(; VFT = Vector{Float32}, plastic = false)
    Random.seed!(31)
    FT = eltype(VFT)
    M = Multipod(N = 40, name = "M", VFT = VFT, param = MultipodParameter(ds = [400um, 300um, 200um, 150um]))
    M.I .= range(FT(300), FT(900), length = 40)
    pre = IF(N = 30, name = "pre", VFT = VFT, param = IFParameter(FT = FT))
    pre.I .= range(FT(250), FT(450), length = 30)
    kw = plastic ? (LTPParam = vSTDPParameter(FT = FT),) : (;)
    to_d3 = SpikingSynapse(pre, M, :glu, :d3; conn = (p = 0.5, μ = 4.0), kw...)
    to_s = SpikingSynapse(pre, M, :gaba, :s; conn = (p = 0.2, μ = 1.0))
    return compose(; pre, M, to_d3, to_s, silent = true)
end

_mpio_state(M) = (copy(M.v_s), copy(M.w_s), [copy(getfield(M, Symbol(:v_d, k))) for k in 1:ndendrites(M)]..., copy(M.tabs),
                  copy(M.synvars_d3.g), copy(M.θ))

# state digest printed by both processes: the bits of every state vector
_mpio_digest(M) = string(hash((M.v_s, M.w_s, M.v_d1, M.v_d2, M.v_d3, M.v_d4, M.tabs, M.synvars_d3.g, M.θ)))

@testset "Multipod: save/load and library utilities (gate of the generated-type layout)" begin
    dir = mktempdir()
    info = (case = "multipod", N_d = 4)

    # ---- (c) monitor! / record --------------------------------------------------------------------------------------------
    model = _mpio_model()
    M = model.pop.M
    monitor!(M, [:v_s, :v_d1, :v_d4, :fire])
    monitor!(M, [:g], variables = :synvars_d3)
    sim!(; model, duration = 200ms)
    @test size(getvariable(M, :v_d4)) == (40, 201)
    @test size(getvariable(M, :synvars_d3_g), 1) == 40
    @test maximum(getvariable(M, :synvars_d3_g)) > 0                     # the input reaches the d3 receptors
    @test sum(length, spiketimes(M)) > 0
    v_d1, r_t = record(M, :v_d1; range = true)                            # interpolated in time
    @test v_d1(3, 100.0) ≈ getvariable(M, :v_d1)[3, 101] rtol = 1e-5
    @test record(M, :v_d1; interpolate = false) == getvariable(M, :v_d1)

    # ---- (a), (b) save, load in the same session ----------------------------------------------------------------------------
    save_model(; model, path = dir, name = "mp", info)
    data = load_data(dir, "mp", info)
    mdl = load_model(dir, "mp", info)
    for loaded in (data, mdl)
        Ml = loaded.model.pop.M
        @test typeof(Ml) == typeof(M) && Ml isa Multipod{4}
        @test _mpio_state(Ml) == _mpio_state(M)
        @test Ml.d3.gax == M.d3.gax && Ml.param.ds == M.param.ds && Ml.param.type isa MultipodNeuron
        @test check_loaded_model(loaded.model) === loaded.model
        # the loaded synapse still writes the d3 buffer of the loaded population
        @test loaded.model.syn.to_d3.g === Ml.receptors_d3.glu && loaded.model.syn.to_d3.v_post === Ml.v_d3
    end
    @test getvariable(data.model.pop.M, :v_d4) == getvariable(M, :v_d4)   # the data file keeps the records
    @test isempty(mdl.model.pop.M.records[:v_d4])                          # the model file has the records cleared

    # the loaded model continues as the original
    ref = deepcopy(model)
    sim!(; model = ref, duration = 100ms)
    sim!(; model = mdl.model, duration = 100ms)
    @test _mpio_state(mdl.model.pop.M) == _mpio_state(ref.pop.M)

    # ---- (a) load in a fresh process (only `using SNNModels`) ---------------------------------------------------------------
    file = SNNpath(dir, "mp", info, :model, 0)
    @test isfile(file)
    Mf = load_model(dir, "mp", info).model.pop.M
    for _ in 1:500
        SNNModels.integrate!(Mf, Mf.param, 0.1f0)
    end
    expected = string(nameof(typeof(Mf)), " ", ndendrites(Mf), " ", _mpio_digest(Mf))
    script = """
        using SNNModels
        M = SNNModels.load_model($(repr(dir)), "mp", $(repr(info))).model.pop.M
        check_loaded_model(M)
        M isa Multipod{4} || error("not a Multipod{4}")
        for _ in 1:500
            SNNModels.integrate!(M, M.param, 0.1f0)
        end
        print("RESULT ", nameof(typeof(M)), " ", ndendrites(M), " ",
              hash((M.v_s, M.w_s, M.v_d1, M.v_d2, M.v_d3, M.v_d4, M.tabs, M.synvars_d3.g, M.θ)))
        """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t 1 -e $script`
    out = read(pipeline(cmd; stderr = devnull), String)
    got = strip(last(split(out, "RESULT ")))
    @info "fresh-process load" expected got
    @test got == expected

    # ---- (b) a file with a placeholder for a missing Multipod layout is refused ---------------------------------------------
    ph = SNNModels.JLD2.ReconstructedMutable{:Multipod4,(:a,),Tuple{Int}}(Any[1])
    @test_throws ArgumentError check_loaded_model((; model = (; pop = (; M = ph))))

    # ---- (d) threads: chunked sim!/train! equal to the serial run ----------------------------------------------------------
    for run! in (sim!, train!)
        a = _mpio_model(plastic = true); b = deepcopy(a)
        run!(; model = a, duration = 150ms, neurons_per_thread = 0)
        run!(; model = b, duration = 150ms, neurons_per_thread = 8)
        @test _mpio_state(a.pop.M) == _mpio_state(b.pop.M)
        SNNModels.sync_weights!(a.syn.to_d3); SNNModels.sync_weights!(b.syn.to_d3)
        @test a.syn.to_d3.W == b.syn.to_d3.W
    end

    # ---- (e) precision ------------------------------------------------------------------------------------------------------
    m64 = _mpio_model(VFT = Vector{Float64})
    @test float_type(m64.pop.M) === Float64 && eltype(m64.pop.M.v_d4) === Float64
    sim!(; model = m64, duration = 100ms)
    @test all(isfinite, m64.pop.M.v_d3)
    M32 = model.pop.M
    s64 = convert_precision(M32.dend_syn, Float64)
    @test float_type(s64) === Float64 && convert_precision(s64, Float32).syn == M32.dend_syn.syn
    M64 = Multipod(N = 2, VFT = Vector{Float64}, param = M32.param, adex = convert_precision(M32.adex, Float64),
                   spike = convert_precision(M32.spike, Float64), soma_syn = convert_precision(M32.soma_syn, Float64), dend_syn = s64)
    @test M64 isa Multipod{4} && M64.adex.El == Float64(M32.adex.El)
    save_model(; model = m64, path = dir, name = "mp64", info)
    @test eltype(load_model(dir, "mp64", info).model.pop.M.v_d4) === Float64

    # ---- (f) change_plasticity / with_plasticity onto :d3 -------------------------------------------------------------------
    base = _mpio_model()
    plastic = change_plasticity(base, :to_d3; LTP = vSTDPParameter())
    @test plastic.syn.to_d3.v_post === base.pop.M.v_d3 && plastic.syn.to_d3.g === base.pop.M.receptors_d3.glu
    W0 = copy(plastic.syn.to_d3.W)
    train!(; model = plastic, duration = 300ms)
    SNNModels.sync_weights!(plastic.syn.to_d3)
    @test plastic.syn.to_d3.W != W0
    frozen = change_plasticity(plastic, :to_d3; LTP = NoLTP(), STP = NoSTP())
    W1 = copy(frozen.syn.to_d3.W)
    train!(; model = frozen, duration = 100ms)
    @test frozen.syn.to_d3.W == W1
    w = with_plasticity(base.syn.to_d3; LTP = vSTDPParameter())
    @test w.v_post === base.pop.M.v_d3

    # ---- (g) deepcopy and modelcopy -----------------------------------------------------------------------------------------
    c1 = deepcopy(model)
    c2 = modelcopy(model)
    for c in (c1, c2)
        @test typeof(c.pop.M) == typeof(model.pop.M) && _mpio_state(c.pop.M) == _mpio_state(model.pop.M)
        @test c.pop.M.v_d3 !== model.pop.M.v_d3 && c.syn.to_d3.g === c.pop.M.receptors_d3.glu   # copied, references kept inside
    end
    @test isempty(c2.pop.M.records[:v_d4])                                  # modelcopy drops the recorded data

    # ---- (h) compose, graph, name-based access ------------------------------------------------------------------------------
    g = graph(model)
    @test SNNModels.Graphs.nv(g) == 2 && SNNModels.Graphs.ne(g) >= 1
    targets = reduce(vcat, [SNNModels.MetaGraphs.get_prop(g, e, :target) for e in SNNModels.Graphs.edges(g)])
    @test "glu_d3" in targets && "gaba_s" in targets
    @test model.pop.M.name == "M" && model.syn.to_d3.targets[:post] == model.pop.M.id
    @test M.d2 === getproperty(M, :d2) && :v_d4 in propertynames(M) && hasproperty(M, :receptors_d4)
    m2 = compose(model, (; extra = IF(N = 3, name = "extra")); silent = true)
    @test m2.pop.M === model.pop.M
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Multipods with more than 8 dendrites: types declared at run time (declare_multipod)
# ---------------------------------------------------------------------------------------------------------------------------------

# run `script` in a fresh Julia process with the test environment; returns its standard output
_mpio_fresh(script) = read(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t 1 -e $script`;
                                    stderr = devnull), String)

@testset "Multipod with 12 dendrites: save here, load in a fresh process" begin
    dir = mktempdir()
    info = (case = "multipod12",)
    M = Multipod(N = 6, name = "M", param = MultipodParameter(ds = [Float32((100 + 25k) * um) for k in 1:12]))   # declares Multipod12
    M.I .= 700f0
    M.receptors_d12.glu .= 3f0
    for _ in 1:200
        SNNModels.integrate!(M, M.param, 0.1f0)
    end
    model = compose(; M, silent = true)
    save_model(; model, path = dir, name = "m12", info)
    file = SNNpath(dir, "m12", info, :model, 0)
    expected = string(hash((M.v_s, M.v_d12, M.w_s)))
    load_expr = "SNNModels.load_model($(repr(dir)), \"m12\", $(repr(info))).model.pop.M"
    digest = "hash((M.v_s, M.v_d12, M.w_s))"

    # (1) load_model reads the type table of the file and declares Multipod12 itself
    out = _mpio_fresh("""
        using SNNModels
        M = $load_expr
        print("RESULT ", nameof(typeof(M)), " ", ndendrites(M), " ", $digest)
        """)
    @test strip(last(split(out, "RESULT "))) == "Multipod12 12 $expected"

    # (2) without a readable type table (JLD2 < 0.6) the placeholders of a first read are declared and the file is read again
    out = _mpio_fresh("""
        using SNNModels
        @eval SNNModels _file_multipods(file::AbstractString) = nothing
        M = $load_expr
        print("RESULT ", nameof(typeof(M)), " ", $digest)
        """)
    @test strip(last(split(out, "RESULT "))) == "Multipod12 $expected"

    # (3) another reader (JLD2.load) gets a placeholder; check_loaded_model names the declaration; after it the file loads
    out = _mpio_fresh("""
        using SNNModels, JLD2
        x = JLD2.load($(repr(file)))
        msg = try; check_loaded_model(x; what = "file"); "no error"; catch e; e.msg; end
        println("MESSAGE ", msg)
        declare_multipod(12)
        M = JLD2.load($(repr(file)))["model"].pop.M
        print("RESULT ", nameof(typeof(M)), " ", $digest)
        """)
    msg = strip(match(r"MESSAGE (.*)", out).captures[1])
    @test msg == "The file contains a Multipod with 12 dendrites whose type is not defined in this session; " *
                 "call SNNModels.declare_multipod(12) before loading, then load again."
    @test strip(last(split(out, "RESULT "))) == "Multipod12 $expected"

    # in this session: the message for an undeclared number of dendrites, the generic one for a preloaded type
    ph(name) = SNNModels.JLD2.ReconstructedMutable{name,(:a,),Tuple{Int}}(Any[1])
    nested = (; model = (; pop = (; M = ph(Symbol("Multipod40{Vector{Float32}}")))))
    err = try; check_loaded_model(nested); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("Multipod with 40 dendrites", err.msg) && occursin("declare_multipod(40)", err.msg)
    err = try; check_loaded_model(ph(:Multipod4)); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("1.9.1", err.msg) && !occursin("declare_multipod", err.msg)
    @test SNNModels._file_multipods(file) == [12]
end
