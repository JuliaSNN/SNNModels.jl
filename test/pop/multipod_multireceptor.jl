# MultiReceptorSynapse on the compartments of Tripod and Multipod: every receptor is driven by the input buffer named by its
# `target` (:AMPA, :NMDA, :GABAa, :GABAb), so a connection addresses a receptor group of one compartment directly,
# `SpikingSynapse(pre, post, :NMDA, :d3)`.
#   1. Multipod with two dendrites == Tripod bit for bit with MultiReceptorSynapse on soma and dendrites (HeunScheme; the audit
#      protocols). PaperScheme needs `ReceptorSynapse` (the published scheme uses its legacy receptor update, `exact = false`,
#      which MultiReceptorSynapse does not have): both populations refuse it.
#   2. MultiReceptorSynapse with the receptors of the standard split (targets :glu and :gaba) == ReceptorSynapse (`exact = true`,
#      same NMDA block) bit for bit.
#   3. Multipod with 3 and 5 dendrites: receptor-group targets on each dendrite, recording of the receptor variables of each
#      dendrite, threading invariance, Float64, save/load (also in a fresh process), change_plasticity onto a receptor-group target.
using Test
using Random
using SNNModels
@load_units

# ---------------------------------------------------------------------------------------------------------------------------------
# Receptor sets with one target per receptor
# ---------------------------------------------------------------------------------------------------------------------------------

# a copy of receptor `r` with input group `target` (every field copied, so the kinetics are bit-identical)
_mr_retarget(r::Receptor, target::Symbol) = Receptor(; name = r.name, E_rev = r.E_rev, τr = r.τr, τd = r.τd, g0 = r.g0, gsyn = r.gsyn,
                                                       α = r.α, τr⁻ = r.τr⁻, τd⁻ = r.τd⁻, nmda = r.nmda, target)

const _MR_SOMA_TARGETS = (:AMPA, :GABAa)
const _MR_DEND_TARGETS = (:AMPA, :NMDA, :GABAa, :GABAb)

_mr_soma(; targets = _MR_SOMA_TARGETS) = MultiReceptorSynapse(
    syn = Receptors((_mr_retarget(r, t) for (r, t) in zip(TripodSomaSynapse.syn, targets))...), NMDA = TripodSomaSynapse.NMDA)
_mr_dend(; targets = _MR_DEND_TARGETS) = MultiReceptorSynapse(
    syn = Receptors((_mr_retarget(r, t) for (r, t) in zip(TripodDendSynapse.syn, targets))...), NMDA = TripodDendSynapse.NMDA)

# Inputs of the protocols as glu/gaba events (rows: glu s, d1, d2, gaba s, d1, d2); with MultiReceptorSynapse a glu event goes to
# every excitatory group of the compartment (AMPA and NMDA), a gaba event to every inhibitory group.
_mr_groups(P, comp::Symbol, kind::Symbol) = begin
    buf = getfield(P, Symbol(:receptors_, comp))
    names = kind === :glu ? (:glu, :AMPA, :NMDA) : (:gaba, :GABAa, :GABAb)
    Tuple(getfield(buf, n) for n in names if hasproperty(buf, n))
end
const _MR_ROWS = ((:s, :glu), (:d1, :glu), (:d2, :glu), (:s, :gaba), (:d1, :gaba), (:d2, :gaba))

function _mr_events(nsteps; seed = 2024)
    rng = Xoshiro(seed)
    X = zeros(Float32, 6, nsteps)
    rates = (0.05, 0.08, 0.08, 0.01, 0.01, 0.01)      # events per step: strong enough to make the neurons fire
    for tt in 1:nsteps, r in 1:6
        rand(rng) < rates[r] && (X[r, tt] = 6f0)
    end
    return X
end

const _MR_PROTOCOLS = (
    step_sub = (n = 3000, I = t -> 50 <= t < 250 ? 300f0 : 0f0, inputs = nothing, plateau = false),
    epsp_d1 = (n = 2000, I = nothing, inputs = (r, tt) -> (r == 2 && tt == 500) ? 1f0 : 0f0, plateau = false),
    step_supra = (n = 5000, I = t -> 50 <= t < 450 ? 1200f0 : 0f0, inputs = nothing, plateau = false),
    poisson_1s = (n = 10000, I = nothing, inputs = let X = _mr_events(10000); (r, tt) -> X[r, tt]; end, plateau = false),
    plateau = (n = 10000, I = nothing, inputs = (r, tt) -> (r == 2 && tt == 150) ? 100f0 : 0f0, plateau = true),
)

function _mr_run(P, prot)
    trace = Vector{NTuple{6,Vector}}(undef, prot.n)
    spikes = Tuple{Int,Int}[]
    for tt in 1:prot.n
        if prot.inputs !== nothing
            for (r, (comp, kind)) in enumerate(_MR_ROWS)
                w = prot.inputs(r, tt)
                w != 0 && foreach(b -> b .= w, _mr_groups(P, comp, kind))
            end
        end
        prot.I === nothing || (P.I .= prot.I(tt * 0.1f0))
        SNNModels.integrate!(P, P.param, convert(eltype(P.v_s), 0.1))
        for i in 1:Int(P.N)
            P.fire[i] && push!(spikes, (tt, i))
        end
        trace[tt] = (copy(P.v_s), copy(P.v_d1), copy(P.v_d2), copy(P.w_s), copy(P.θ), copy(P.tabs))
    end
    return trace, spikes
end

function _mr_rest!(P)
    El = P.adex.El
    P.v_s .= El
    for f in fieldnames(typeof(P))
        startswith(string(f), "v_d") && fill!(getfield(P, f), El)
    end
    P.w_s .= 0; P.θ .= P.adex.Vt
    return P
end

_mr_maxdiff(a, b) = maximum(maximum(maximum(abs.(Float64.(x) .- Float64.(y))) for (x, y) in zip(sa[1:4], sb[1:4])) for (sa, sb) in zip(a, b))

const MR_RESULTS = Dict{String,Tuple{Float64,Int}}()

# ---------------------------------------------------------------------------------------------------------------------------------
# 1. Multipod (2 dendrites) == Tripod with MultiReceptorSynapse
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "MultiReceptorSynapse: Multipod N_d = 2 == Tripod (bit for bit)" begin
    for VFT in (Vector{Float32}, Vector{Float64}), (pname, prot) in pairs(_MR_PROTOCOLS)
        FT = eltype(VFT)
        lengths = prot.plateau ? [250um, 250um] : [400um, 150um]
        kw = (soma_syn = convert_precision(_mr_soma(), FT), dend_syn = convert_precision(_mr_dend(), FT), VFT = VFT, N = 3)
        T = _mr_rest!(Tripod(; param = TripodParameter(ds = lengths), kw...))
        M = _mr_rest!(Multipod(; param = MultipodParameter(ds = lengths), kw...))
        @test keys(M.receptors_d1) == keys(T.receptors_d1) && Set(keys(M.receptors_d2)) == Set(_MR_DEND_TARGETS)
        tT, sT = _mr_run(T, prot)
        tM, sM = _mr_run(M, prot)
        @test sT == sM
        @test all(tT[k] == tM[k] for k in eachindex(tT))
        d = _mr_maxdiff(tT, tM)
        MR_RESULTS["tripod-vs-multipod/$(FT)/$(pname)"] = (d, length(sT))
        @test d == 0
    end
    @test MR_RESULTS["tripod-vs-multipod/Float32/step_supra"][2] > 0 && MR_RESULTS["tripod-vs-multipod/Float32/poisson_1s"][2] > 0
    # PaperScheme is defined on ReceptorSynapse only
    @test_throws ArgumentError Tripod(N = 1, scheme = PaperScheme(), soma_syn = _mr_soma(), dend_syn = _mr_dend())
    @test_throws ArgumentError Multipod(N = 1, scheme = PaperScheme(), soma_syn = _mr_soma(), dend_syn = _mr_dend())
end

# ---------------------------------------------------------------------------------------------------------------------------------
# 2. MultiReceptorSynapse with the glu/gaba split == ReceptorSynapse
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "MultiReceptorSynapse with targets :glu/:gaba == ReceptorSynapse (exact kinetics)" begin
    soma = _mr_soma(targets = (:glu, :gaba))
    dend = _mr_dend(targets = (:glu, :glu, :gaba, :gaba))
    @test Set(keys(soma.receptors)) == Set((:glu, :gaba)) && Set(keys(dend.receptors)) == Set((:glu, :gaba))
    for (pname, prot) in pairs(_MR_PROTOCOLS), P in (Tripod, Multipod)
        lengths = prot.plateau ? [250um, 250um] : [400um, 150um]
        param = P === Tripod ? TripodParameter(ds = lengths) : MultipodParameter(ds = lengths)
        A = _mr_rest!(P(N = 3; param))                                              # ReceptorSynapse, exact = true
        B = _mr_rest!(P(N = 3; param, soma_syn = soma, dend_syn = dend))           # MultiReceptorSynapse
        @test A.soma_syn.exact && A.dend_syn.exact && A.dend_syn.NMDA == B.dend_syn.NMDA
        tA, sA = _mr_run(A, prot)
        tB, sB = _mr_run(B, prot)
        d = _mr_maxdiff(tA, tB)
        MR_RESULTS["receptor-vs-multireceptor/$(nameof(P))/$(pname)"] = (d, length(sA))
        @test sA == sB && d == 0
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------
# 3. Multipod with 3 and 5 dendrites
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "MultiReceptorSynapse on Multipod with $D dendrites" for D in (3, 5)
    lengths = [Float32((150 + 50k) * um) for k in 1:D]
    mk(; kw...) = Multipod(; N = 40, name = "M", param = MultipodParameter(ds = lengths), soma_syn = _mr_soma(), dend_syn = _mr_dend(), kw...)
    nmda = findfirst(==(:NMDA), [r.target for r in _mr_dend().syn])
    gabab = findfirst(==(:GABAb), [r.target for r in _mr_dend().syn])

    # targets: (group, compartment) -> that buffer and that receptor column only
    M = _mr_rest!(mk())
    pre = IF(N = 10, name = "pre")
    for k in 1:D, group in _MR_DEND_TARGETS
        syn = SpikingSynapse(pre, M, group, Symbol(:d, k); conn = (p = 1.0, μ = 1.0))
        @test syn.g === getfield(getfield(M, Symbol(:receptors_d, k)), group)
        @test syn.v_post === getfield(M, Symbol(:v_d, k)) && syn.targets[:sym] == "$(group)_d$(k)"
    end
    @test SpikingSynapse(pre, M, :AMPA, :s; conn = (p = 1.0, μ = 1.0)).g === M.receptors_s.AMPA
    @test_throws ArgumentError SpikingSynapse(pre, M, :NMDA, :s; conn = (p = 1.0, μ = 1.0))    # no NMDA group on the soma
    @test_throws ArgumentError SpikingSynapse(pre, M, :glu, :d1; conn = (p = 1.0, μ = 1.0))    # no glu buffer: groups only
    M.receptors_d3.NMDA .= 5f0
    getfield(M, Symbol(:receptors_d, D)).GABAb .= 5f0
    SNNModels.integrate!(M, M.param, 0.1f0)
    for k in 1:D
        g = getfield(M, Symbol(:synvars_d, k)).g
        for n in axes(g, 2)
            expected = (k == 3 && n == nmda) || (k == D && n == gabab)
            @test all(>(0), g[:, n]) == expected
            expected || @test all(==(0), g[:, n])
        end
    end
    @test all(==(0), M.synvars_s.g)

    # recording of the receptor variables of each dendrite, in a network run
    M = mk()
    M.I .= range(200f0, 900f0, length = 40)
    pre = IF(N = 20, name = "pre"); pre.I .= range(250f0, 450f0, length = 20)
    syns = NamedTuple(Symbol(:nmda_d, k) => SpikingSynapse(pre, M, :NMDA, Symbol(:d, k); conn = (p = 0.3, μ = 2.0)) for k in 1:D)
    model = compose(; pre, M, syns..., silent = true)
    for k in 1:D
        monitor!(M, [:g], variables = Symbol(:synvars_d, k))
    end
    monitor!(M, [:v_s, Symbol(:v_d, D), :fire])
    sim!(; model, duration = 200ms)
    for k in 1:D
        gk = getvariable(M, Symbol(:synvars_d, k, :_g))
        @test size(gk, 1) == 40 && maximum(gk[:, nmda, :]) > 0
        @test maximum(gk[:, 1, :]) == 0                                           # AMPA of d_k: no input
    end

    # threading invariance (sim! and train!)
    for run! in (sim!, train!)
        a = deepcopy(model); b = deepcopy(model)
        run!(; model = a, duration = 100ms, neurons_per_thread = 0)
        run!(; model = b, duration = 100ms, neurons_per_thread = 8)
        Ma, Mb = a.pop.M, b.pop.M
        @test Ma.v_s == Mb.v_s && Ma.w_s == Mb.w_s && all(getfield(Ma, Symbol(:synvars_d, k)).g == getfield(Mb, Symbol(:synvars_d, k)).g for k in 1:D)
    end

    # Float64: the same trajectory up to rounding
    M32 = _mr_rest!(mk())
    M64 = _mr_rest!(mk(VFT = Vector{Float64}, soma_syn = convert_precision(_mr_soma(), Float64), dend_syn = convert_precision(_mr_dend(), Float64)))
    @test eltype(M64.synvars_d2.g) === Float64 && M64.dend_syn.syn[2].τd === Float64(TripodDendSynapse.syn[2].τd)
    for P in (M32, M64)
        P.I .= 200
        for tt in 1:2000
            tt % 50 == 0 && (P.receptors_d2.NMDA .= 2; P.receptors_d1.AMPA .= 2)
            SNNModels.integrate!(P, P.param, P === M32 ? 0.1f0 : 0.1)
        end
    end
    @test maximum(abs.(M32.v_d2 .- M64.v_d2)) < 1e-2 && maximum(abs.(M32.v_s .- M64.v_s)) < 1e-2

    # change_plasticity onto a receptor-group target
    plastic = change_plasticity(model, :nmda_d1; LTP = vSTDPParameter())
    @test plastic.syn.nmda_d1.g === M.receptors_d1.NMDA && plastic.syn.nmda_d1.v_post === M.v_d1
    W0 = copy(plastic.syn.nmda_d1.W)
    train!(; model = plastic, duration = 300ms)
    SNNModels.sync_weights!(plastic.syn.nmda_d1)
    @test plastic.syn.nmda_d1.W != W0

    # save / load, same session and fresh process
    dir = mktempdir()
    info = (case = "multireceptor", D = D)
    save_model(; model, path = dir, name = "mr", info)
    loaded = load_model(dir, "mr", info).model
    Ml = loaded.pop.M
    @test typeof(Ml) == typeof(M) && Ml.dend_syn.receptors == M.dend_syn.receptors && keys(Ml.receptors_d2) == keys(M.receptors_d2)
    @test Ml.v_s == M.v_s && Ml.synvars_d3.g == M.synvars_d3.g && loaded.syn.nmda_d2.g === Ml.receptors_d2.NMDA
    expected = string(hash((Ml.v_s, Ml.synvars_d3.g)))
    script = """
        using SNNModels
        M = SNNModels.load_model($(repr(dir)), "mr", $(repr(info))).model.pop.M
        print("RESULT ", hash((M.v_s, M.synvars_d3.g)), " ", keys(M.receptors_d$(D)))
        """
    out = read(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t 1 -e $script`; stderr = devnull), String)
    got = split(strip(last(split(out, "RESULT "))), " "; limit = 2)
    @test got[1] == expected
    @test occursin("NMDA", got[2])
end

for k in sort(collect(keys(MR_RESULTS)))
    println("  multireceptor  ", rpad(k, 44), " max|Δ| = ", MR_RESULTS[k][1], "  spikes = ", MR_RESULTS[k][2])
end
