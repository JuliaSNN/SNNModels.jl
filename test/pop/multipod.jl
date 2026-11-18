# Multipod: construction for 1 to 6 dendrites, parameter validation, cable constants, receptor targets, generality checks
# (disconnected extra dendrites, symmetric dendrites), Float32/Float64, plasticity onto a dendrite (vSTDP on :d3), threading
# invariance and a balanced-network smoke test with 3 and 5 dendrites.
# The bit-for-bit equality with Tripod for two dendrites is in multipod_tripod_equivalence.jl; save/load and the other
# library utilities in multipod_io.jl.
using Test
using Random
using SNNModels
@load_units

# ---------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------------------

_mp_dends(M) = Tuple(getfield(M, Symbol(:d, k)) for k in 1:ndendrites(M))
_mp_vds(M) = Tuple(getfield(M, Symbol(:v_d, k)) for k in 1:ndendrites(M))

# every compartment at El, no adaptation, threshold at rest: a deterministic initial state
function _mp_rest!(P)
    El = P.adex.El
    P.v_s .= El
    for f in fieldnames(typeof(P))
        startswith(string(f), "v_d") && fill!(getfield(P, f), El)
    end
    P.w_s .= 0
    P.θ .= P.adex.Vt
    return P
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "Multipod construction, 1 to 6 dendrites" begin
    for D in 1:6
        ls = [(150 + 50k) * um for k in 1:D]
        M = Multipod(N = 7, param = MultipodParameter(ds = ls))
        T = typeof(M)
        @test M isa Multipod{D} && M isa Multipod && M isa SNNModels.AbstractDendriteIF
        @test nameof(T) === Symbol(:Multipod, D) && ndendrites(M) == D
        @test M.name == "Multipod" && M.N == 7 && M.scheme isa HeunScheme
        # named fields d1..dD, v_d1..v_dD, synvars_d1.., receptors_d1.. and nothing beyond D
        for k in 1:D, pre in (:d, :v_d, :synvars_d, :receptors_d)
            @test hasfield(T, Symbol(pre, k))
        end
        @test !hasfield(T, Symbol(:d, D + 1)) && !hasfield(T, Symbol(:v_d, D + 1))
        @test all(length(v) == 7 for v in _mp_vds(M))
        @test size(M.Δv) == (7, D + 2) && size(M.Δv_temp) == (7, D + 2) && size(M.is) == (7, D + 1) && size(M.v_pred) == (7, D + 1)
        @test [round(d.l[1] / um) for d in _mp_dends(M)] == [150 + 50k for k in 1:D]
        @test M.param.geometry == [(:s => Symbol(:d, k)) for k in 1:D] && M.param.type isa MultipodNeuron
        # the type-parameter constructor and Population(param)
        @test Multipod{D}(N = 2) isa Multipod{D}
        @test ndendrites(Multipod{D}(N = 2)) == D
        @test Population(MultipodParameter(ds = ls); N = 3) isa Multipod{D}
        # it integrates (default receptors, default scheme)
        M.I .= 600f0
        for _ in 1:200
            SNNModels.integrate!(M, M.param, 0.1f0)
        end
        @test all(isfinite, M.v_s) && all(v -> all(isfinite, v), _mp_vds(M))
    end
    @test ndendrites(Multipod(N = 2)) == 3                                 # the default: three dendrites of 200-400 um
    @test Multipod(N = 2).d3.l[1] >= 200um - 1f-6
    @test Population(DendNeuronParameter(ds = [300um, 300um, 300um, 300um]); N = 2) isa Multipod{4}
    @test SNNModels.MULTIPOD_PRELOADED == 8 && Multipod{8}(N = 1) isa Multipod{8}
end

@testset "Multipod with more than 8 dendrites: declare_multipod" begin
    @test SNNModels.MULTIPOD_MAX_DENDRITES == 64
    # declared on demand by the constructors, idempotent
    M = Multipod(N = 3, param = MultipodParameter(ds = fill(300um, 12)))
    @test M isa Multipod{12} && nameof(typeof(M)) === :Multipod12 && hasfield(typeof(M), :v_d12)
    @test declare_multipod(12) === typeof(M).name.wrapper
    @test declare_multipod(12) === declare_multipod(12)
    @test declare_multipod(3) === SNNModels.Multipod3                         # a preloaded type is returned as it is
    T20 = declare_multipod(20)
    @test invokelatest(T20; N = 2, param = MultipodParameter(ds = fill(200um, 20))) isa Multipod{20}
    @test Multipod{9}(N = 1) isa Multipod{9}
    @test Population(DendNeuronParameter(ds = fill(300um, 10)); N = 2) isa Multipod{10}
    @test_throws ArgumentError declare_multipod(0)
    @test_throws ArgumentError declare_multipod(65)
    # it integrates and connects like the others
    M.I .= 800f0
    M.receptors_d12.glu .= 5f0
    for _ in 1:300
        SNNModels.integrate!(M, M.param, 0.1f0)
    end
    @test all(isfinite, M.v_s) && maximum(M.v_d12) > maximum(M.v_d11) - 1f-3
    @test SpikingSynapse(IF(N = 2), M, :glu, :d12; conn = (p = 1.0, μ = 1.0)).g === M.receptors_d12.glu
    # concurrent declarations of the same type give one type
    Ts = fetch.([Threads.@spawn declare_multipod(30) for _ in 1:8])
    @test all(T -> T === Ts[1], Ts)
end

@testset "Multipod parameters: validation and backward compatibility" begin
    # number of dendrites
    @test_throws ArgumentError MultipodParameter(ds = [])
    @test_throws ArgumentError MultipodParameter(ds = fill(200um, 65))
    @test_throws ErrorException DendNeuronParameter(ds = fill(200um, 65))
    @test_throws ArgumentError Multipod{65}(N = 1)
    # star topology only
    @test_throws ArgumentError MultipodParameter(ds = [200um, 200um, 200um], geometry = [(:s => :d1), (:d1 => :d2), (:s => :d3)])
    @test_throws ArgumentError MultipodParameter(ds = [200um, 200um, 200um], geometry = [(:s => :d1), (:s => :d2)])
    @test MultipodParameter(ds = [200um, 200um, 200um], geometry = [(:s => :d3), (:s => :d1), (:s => :d2)]).geometry ==
          [(:s => :d3), (:s => :d1), (:s => :d2)]
    # diameters
    @test_throws ArgumentError MultipodParameter(ds = [200um, 200um], d = [3um])
    @test_throws ArgumentError MultipodParameter(ds = [200um => 3um, 200um], d = [3um, 4um])
    # the type and the parameter must agree on the number of dendrites
    @test_throws ArgumentError SNNModels.Multipod3(N = 1, param = MultipodParameter(ds = [200um, 200um]))
    # Tripod and BallAndStick keep their own number of dendrites
    @test DendNeuronParameter(ds = [200um]).type isa SNNModels.BallAndStickNeuron
    @test DendNeuronParameter(ds = [200um, 300um]).type isa SNNModels.TripodNeuron
    @test DendNeuronParameter(ds = [200um, 300um, 100um]).type isa MultipodNeuron
    @test DendNeuronParameter(ds = [200um]).geometry == [(:s => :d1), (:s => :d2)]          # unchanged default
    @test DendNeuronParameter(ds = [200um, 300um, 100um]).geometry == [(:s => :d1), (:s => :d2), (:s => :d3)]
    @test_throws ArgumentError Tripod(N = 1, param = DendNeuronParameter(ds = [200um, 300um, 100um]))
    @test_throws ArgumentError BallAndStick(N = 1, param = DendNeuronParameter(ds = [200um, 300um, 100um]))
    @test Population(TripodParameter(ds = [200um, 300um]); N = 2) isa Tripod
    @test Population(BallAndStickParameter(ds = [200um]); N = 2) isa BallAndStick
    # a Multipod with two dendrites is a Multipod, not a Tripod
    @test Population(MultipodParameter(ds = [200um, 300um]); N = 2) isa Multipod{2}
    # `ds` entries: numbers, ranges, pairs; stored in Float32
    p = MultipodParameter(ds = [0.03, (150um, 300um), 200um => 3um, (100um, 200um) => 2um])
    @test p.ds[1] === 0.03f0 && p.ds[2] == (150um, 300um) && p.ds[3] === (200um => Float32(3um)) && p.ds[4].first == (100um, 200um)
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Cable constants
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "Multipod cable constants vs the formulas" begin
    for FT in (Float32, Float64), phys in (human_dend, mouse_dend)
        ref(l, d) = FT === Float32 ? create_dendrite(l; d, physiology = phys) : SNNModels._create_dendrite_decimal(l; d, physiology = phys)
        ls = [400um, 300um, 250um, 150um, 100um]
        ds = [4um, 3um, 5um, 2um, 4um]
        # per-dendrite diameters, given as a vector or as pairs: the same dendrites
        M = Multipod(N = 3, VFT = Vector{FT}, param = MultipodParameter(ds = ls, d = ds, physiology = phys))
        Mp = Multipod(N = 3, VFT = Vector{FT}, param = MultipodParameter(ds = [l => d for (l, d) in zip(ls, ds)], physiology = phys))
        for (k, (dend, dendp)) in enumerate(zip(_mp_dends(M), _mp_dends(Mp)))
            r = ref(ls[k], ds[k])
            # exactly the values of create_dendrite (the derivation of Tripod)
            @test all(==(FT(r.gax)), dend.gax) && all(==(FT(r.gm)), dend.gm) && all(==(FT(r.C)), dend.C)
            @test dend.gax == dendp.gax && dend.C == dendp.C
            # and the cable formulas: g_ax = pi d^2 / (4 Ri l), g_m = pi d l / Rd, C = Cd pi d l
            @test dend.gax[1] ≈ π * ds[k]^2 / (4 * phys.Ri * ls[k]) rtol = 1e-5
            @test dend.gm[1] ≈ π * ds[k] * ls[k] / phys.Rd rtol = 1e-5
            @test dend.C[1] ≈ phys.Cd * π * ds[k] * ls[k] rtol = 1e-5
            @test dend.l[1] ≈ ls[k] rtol = 1e-6
            @test dend.d[1] ≈ ds[k] rtol = 1e-6
        end
        # a common diameter
        M4 = Multipod(N = 2, VFT = Vector{FT}, param = MultipodParameter(ds = ls, d = 3um, physiology = phys))
        @test all(d -> d.d[1] ≈ 3um, _mp_dends(M4))
        @test M4.d2.gax[1] == FT(ref(300um, 3um).gax)
    end
    # the paper numbers (Table 3 of the Tripod paper, human, 400 um, 4 um): 1.292 nS, 15.708 nS, 25.13 pF
    M = MultipodPaper(N = 1, lengths = [400um, 400um, 400um])
    @test all(d -> isapprox(d.gm[1], 1.292; atol = 1e-3) && isapprox(d.gax[1], 15.708; atol = 1e-3) &&
                   isapprox(d.C[1], 25.13; atol = 1e-2), _mp_dends(M))
    # dendritic leak reversal: El_dend or the somatic El
    @test all(d -> all(==(-70.6f0), d.El), _mp_dends(M))
    M = Multipod(N = 2, adex = AdExParameter(El = -60mV), param = MultipodParameter(ds = [200um, 200um, 200um]))
    @test all(d -> all(==(-60f0), d.El), _mp_dends(M))
    M = Multipod(N = 2, adex = AdExParameter(El = -60mV), param = MultipodParameter(ds = [200um, 200um, 200um], El_dend = -70mV))
    @test all(d -> all(==(-70f0), d.El), _mp_dends(M))
    # random lengths, one draw per neuron and dendrite
    M = Multipod(N = 300, param = MultipodParameter(ds = fill((150um, 400um), 4)))
    @test all(d -> length(unique(d.l)) > 20 && all(150um - 1f-6 .<= d.l .<= 400um + 1f-6), _mp_dends(M))
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Generality: extra disconnected dendrites, symmetric dendrites, MultipodPaper keywords
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "Multipod: disconnected extra dendrites leave Tripod unchanged (bit for bit)" begin
    # dendrites of length 0 have g_ax = 0: they add 0 to the axial sum, so the soma and d1, d2 follow Tripod exactly
    for paper in (false, true)
        T = paper ? TripodPaper(N = 3, config = :distal_proximal) : Tripod(N = 3, param = TripodParameter(ds = [400um, 150um]))
        M = paper ? MultipodPaper(N = 3, lengths = [400um, 150um, 0um, 0um]) :
            Multipod(N = 3, param = MultipodParameter(ds = [400um, 150um, 0um, 0um]))
        _mp_rest!(T); _mp_rest!(M)
        @test all(==(0), M.d3.gax) && all(==(0), M.d4.gax)
        rng = Xoshiro(3)
        for tt in 1:5000
            w = rand(rng) < 0.05 ? 3f0 : 0f0
            for P in (T, M)
                P.receptors_d1.glu .= w
                P.receptors_s.glu .= w / 2
                P.I .= 50 <= tt * 0.1 < 300 ? 700f0 : 0f0
                SNNModels.integrate!(P, P.param, 0.1f0)
            end
            @test T.v_s == M.v_s && T.v_d1 == M.v_d1 && T.v_d2 == M.v_d2 && T.w_s == M.w_s && T.fire == M.fire
        end
    end
end

@testset "Disconnected dendrites (length 0): semantics" begin
    M = Multipod(N = 2, param = MultipodParameter(ds = [300um, 300um, 0um]))
    @test all(==(0), M.d3.gax) && all(==(1), M.d3.gm) && all(==(1), M.d3.C) && all(==(-1), M.d3.l)
    _mp_rest!(M)
    # the isolated dendrite is integrated: from -40 mV it relaxes to El with τ = C / gm = 1 ms
    M.v_d3 .= -40f0
    SNNModels.integrate!(M, M.param, 0.1f0)
    @test M.v_d3[1] ≈ -70.6 + 30.6 * exp(-0.1) rtol = 1e-3
    # a synapse onto it is accepted with a warning; its input moves d3 only
    pre = IF(N = 2)
    @test_logs (:warn, r"2 of 2 neurons .* dendrite `d3` disconnected") SpikingSynapse(pre, M, :glu, :d3; conn = (p = 1.0, μ = 1.0))
    @test_logs SpikingSynapse(pre, M, :glu, :d2; conn = (p = 1.0, μ = 1.0))      # connected dendrite: no warning
    A = Multipod(N = 2, param = MultipodParameter(ds = [300um, 300um, 0um])); _mp_rest!(A)
    B = deepcopy(A)
    A.receptors_d3.glu .= 50f0
    for _ in 1:500
        SNNModels.integrate!(A, A.param, 0.1f0)
        SNNModels.integrate!(B, B.param, 0.1f0)
    end
    @test A.v_s == B.v_s && A.v_d1 == B.v_d1 && A.v_d2 == B.v_d2 && A.v_d3 != B.v_d3
    # set_dendrite_length!: the values of create_dendrite, in both precisions, and back
    for VFT in (Vector{Float32}, Vector{Float64})
        FT = eltype(VFT)
        P = Multipod(N = 4, VFT = VFT, param = MultipodParameter(ds = [400um, 300um, 200um]))
        ref = Multipod(N = 4, VFT = VFT, param = MultipodParameter(ds = [400um, 250um, 0um]))
        set_dendrite_length!(P, 2, 250um)
        set_dendrite_length!(P, 3, 0um; neurons = 2:3)
        @test P.d2.gax == ref.d2.gax && P.d2.C == ref.d2.C && P.d2.gm == ref.d2.gm && P.d2.l == ref.d2.l
        @test P.d3.gax[2:3] == ref.d3.gax[2:3] && P.d3.l[2:3] == ref.d3.l[2:3] && P.d3.gax[1] > 0
        set_dendrite_length!(P, 3, 200um; neurons = 2:3)
        @test all(==(P.d3.gax[1]), P.d3.gax)
        set_dendrite_length!(P, 1, 400um; d = 3um)
        @test P.d1.gax[1] ≈ (3 / 4)^2 * ref.d1.gax[1] rtol = 1e-5
    end
    @test_throws ArgumentError set_dendrite_length!(M, 4, 100um)
    T = Tripod(N = 2); set_dendrite_length!(T, 2, 0um; neurons = 1:1)
    @test T.d2.gax[1] == 0 && T.d2.gax[2] > 0
    B1 = BallAndStick(N = 2); set_dendrite_length!(B1, 1, 0um)
    @test all(==(0), B1.d.gax)
end

# Each neuron of a heterogeneous population follows the population of one neuron with its configuration (bit for bit); the
# neurons with d3, d4 disconnected follow Tripod.
@testset "Heterogeneous population: per-neuron disconnected dendrites ($scheme)" for scheme in (:heun, :paper)
    N = 12
    lengths = [400um, 150um, 300um, 250um]
    configs = [isodd(i) ? (3, 4) : (i % 4 == 0 ? (2,) : ()) for i in 1:N]       # dendrites off, per neuron
    build(n) = scheme === :paper ? MultipodPaper(N = n, lengths = lengths) : Multipod(N = n, param = MultipodParameter(ds = lengths))
    P = build(N)
    for i in 1:N, k in configs[i]
        set_dendrite_length!(P, k, 0um; neurons = i:i)
    end
    _mp_rest!(P)
    singles = map(1:N) do i
        Q = build(1)
        foreach(k -> set_dendrite_length!(Q, k, 0um), configs[i])
        _mp_rest!(Q)
    end
    tripods = [scheme === :paper ? TripodPaper(N = 1, config = :distal_proximal) :
               Tripod(N = 1, param = TripodParameter(ds = [400um, 150um])) for i in 1:N if isodd(i)]
    foreach(_mp_rest!, tripods)
    # per-neuron inputs: Poisson events on soma, d1, d3 and a current step
    rng = Xoshiro(17)
    X = [rand(rng) < 0.03 ? 3f0 * rand(rng, Float32) : 0f0 for _ in 1:3, _ in 1:N, _ in 1:4000]
    current(tt, i) = 50 <= tt * 0.1 < 250 ? 300f0 + 60f0 * i : 0f0
    ok = true
    for tt in 1:4000
        for i in 1:N
            P.receptors_s.glu[i] = X[1, i, tt]; P.receptors_d1.glu[i] = X[2, i, tt]; P.receptors_d3.glu[i] = X[3, i, tt]
            P.I[i] = current(tt, i)
            Q = singles[i]
            Q.receptors_s.glu[1] = X[1, i, tt]; Q.receptors_d1.glu[1] = X[2, i, tt]; Q.receptors_d3.glu[1] = X[3, i, tt]
            Q.I[1] = current(tt, i)
            SNNModels.integrate!(Q, Q.param, 0.1f0)
        end
        for (j, i) in enumerate(1:2:N)
            Tr = tripods[j]
            Tr.receptors_s.glu[1] = X[1, i, tt]; Tr.receptors_d1.glu[1] = X[2, i, tt]   # d3 is off: its input is not seen
            Tr.I[1] = current(tt, i)
            SNNModels.integrate!(Tr, Tr.param, 0.1f0)
        end
        SNNModels.integrate!(P, P.param, 0.1f0)
        for i in 1:N
            Q = singles[i]
            ok &= P.v_s[i] == Q.v_s[1] && P.w_s[i] == Q.w_s[1] && P.tabs[i] == Q.tabs[1] && P.fire[i] == Q.fire[1]
            ok &= all(getfield(P, Symbol(:v_d, k))[i] == getfield(Q, Symbol(:v_d, k))[1] for k in 1:4)
        end
        for (j, i) in enumerate(1:2:N)
            ok &= P.v_s[i] == tripods[j].v_s[1] && P.v_d1[i] == tripods[j].v_d1[1] && P.v_d2[i] == tripods[j].v_d2[1] &&
                  P.w_s[i] == tripods[j].w_s[1]
        end
    end
    @test ok
    @test any(P.w_s .> 0)                                     # the run contains spikes
end

@testset "Multipod: symmetric dendrites with the same input stay equal" begin
    M = Multipod(N = 2, param = MultipodParameter(ds = fill(250um, 5)))
    _mp_rest!(M)
    for tt in 1:3000
        for k in 1:5
            getfield(M, Symbol(:receptors_d, k)).glu .= tt % 100 == 0 ? 4f0 : 0f0
        end
        SNNModels.integrate!(M, M.param, 0.1f0)
    end
    @test all(v -> v == M.v_d1, _mp_vds(M))
    @test M.v_d1 != M.v_s
end

@testset "MultipodPaper keywords" begin
    M = MultipodPaper(N = 3, lengths = [400um, 250um, 150um])
    @test M isa Multipod{3} && M.scheme isa PaperScheme && M.name == "MultipodPaper" && M.Vspike == -50.4f0
    @test (M.adex.C, M.adex.gl, M.adex.El, M.adex.Vt, M.adex.b) == (281f0, 40f0, -70.6f0, -50.4f0, 80.5f0)
    @test all(==(-70.6f0), M.v_s) && all(v -> all(==(-70.6f0), v), _mp_vds(M)) && all(==(0), M.w_s)
    @test !M.dend_syn.exact && M.dend_syn.NMDA.approx == 32
    Mm = MultipodPaper(N = 1, lengths = [400um, 250um, 150um], physiology = :mouse)
    @test Mm.param.physiology === mouse_dend && Mm.dend_syn.syn[2].τd == 100f0
    @test MultipodPaper(N = 1, lengths = [400um, 400um], d = [3um, 4um]).d1.gax[1] / M.d1.gax[1] ≈ (3 / 4)^2 rtol = 1e-5
    @test MultipodPaper(N = 1, name = "x").name == "x"
    @test ndendrites(MultipodPaper(N = 1)) == 2                                   # default: the lengths of TripodPaper()
    @test_throws ArgumentError MultipodPaper(physiology = :rat)
    @test_throws ArgumentError MultipodPaper(lengths = [(150um, 300um), 200um])     # fixed lengths only
    @test_throws ArgumentError Multipod(N = 1, scheme = PaperScheme(), soma_syn = DeltaSynapse())
    M = MultipodPaper(N = 1)
    @test_logs (:warn, r"dt = 0.1 ms") SNNModels.integrate!(M, M.param, 0.125f0)
    # spike sequence of the published code: spike step at 21 mV, 9 backpropagation steps at 20 mV, 21 refractory steps
    M = MultipodPaper(N = 1, lengths = [400um, 250um, 150um, 150um])
    M.v_s .= -50.4f0
    vs = Float32[]
    for _ in 1:40
        SNNModels.integrate!(M, M.param, 0.1f0)
        push!(vs, M.v_s[1])
    end
    @test vs[1] == 21f0 && all(==(20f0), vs[2:10]) && all(==(-70.6f0), vs[11:31]) && vs[32] != -70.6f0
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Receptor targets
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "Multipod receptor targets :s, :d1..:dN" begin
    M = Multipod(N = 20, param = MultipodParameter(ds = fill(300um, 5)))
    _mp_rest!(M)
    pre = IF(N = 10)
    syns = Dict(t => SpikingSynapse(pre, M, :glu, t; conn = (p = 1.0, μ = 1.0)) for t in (:s, :d1, :d3, :d5))
    @test syns[:d3].targets[:sym] == "glu_d3" && syns[:d5].targets[:sym] == "glu_d5"
    # the synapse writes the buffer of its compartment and reads the potential of that compartment
    @test syns[:d5].g === M.receptors_d5.glu && syns[:d3].g === M.receptors_d3.glu && syns[:s].g === M.receptors_s.glu
    @test syns[:d3].v_post === M.v_d3
    @test SpikingSynapse(pre, M, :gaba, :d2; conn = (p = 1.0, μ = 1.0)).g === M.receptors_d2.gaba
    @test_throws ArgumentError SpikingSynapse(pre, M, :glu, :d6; conn = (p = 1.0, μ = 1.0))
    @test_throws ArgumentError SpikingSynapse(pre, M, :glu, :d; conn = (p = 1.0, μ = 1.0))
    # an event on d4 depolarises d4 most, then the soma; NMDA keeps d4 up longer than an AMPA-only receptor would
    M.receptors_d4.glu .= 20f0
    vmax = fill(-Inf32, 6)
    for _ in 1:300
        SNNModels.integrate!(M, M.param, 0.1f0)
        for (k, v) in enumerate((M.v_s, M.v_d1, M.v_d2, M.v_d3, M.v_d4, M.v_d5))
            vmax[k] = max(vmax[k], v[1])
        end
    end
    @test argmax(vmax) == 5 && vmax[1] > vmax[2] && vmax[2] ≈ vmax[3] && vmax[3] ≈ vmax[6]
    @test M.synvars_d4.g[1, 2] > 0 && M.synvars_d3.g[1, 2] == 0           # NMDA conductance of d4 only
    # MultiReceptorSynapse and a synapse model without chunk methods (serial path)
    Mr = Multipod(N = 4, param = MultipodParameter(ds = fill(300um, 3)), soma_syn = MultiReceptorSynapse(), dend_syn = MultiReceptorSynapse())
    Mr.receptors_d2.glu .= 5f0
    for _ in 1:100; SNNModels.integrate!(Mr, Mr.param, 0.1f0); end
    @test all(isfinite, Mr.v_d2)
    Ms = Multipod(N = 4, param = MultipodParameter(ds = fill(300um, 3)), soma_syn = SingleExpSynapse(), dend_syn = SingleExpSynapse())
    Ms.I .= 500f0
    for _ in 1:100; SNNModels.integrate!(Ms, Ms.param, 0.1f0); end
    @test all(isfinite, Ms.v_s)
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "Multipod Float32 / Float64" begin
    for D in (1, 3, 6)
        M32 = Multipod(N = 4, param = MultipodParameter(ds = fill(300um, D)))
        M64 = Multipod(N = 4, VFT = Vector{Float64}, param = MultipodParameter(ds = fill(300um, D)))
        @test eltype(M64.v_s) === Float64 && eltype(M64.v_d1) === Float64 && M64.adex isa AdExParameter{Float64}
        @test float_type(M64) === Float64 && float_type(M32) === Float32
        @test eltype(getfield(M64, Symbol(:d, D)).gax) === Float64 && eltype(M64.Δv) === Float64
        # the same trajectory up to rounding (subthreshold drive)
        _mp_rest!(M32); _mp_rest!(M64)
        for P in (M32, M64)
            P.I .= 200
            P.receptors_d1.glu .= 2
            for _ in 1:2000; SNNModels.integrate!(P, P.param, P === M32 ? 0.1f0 : 0.1); end
        end
        @test maximum(abs.(M32.v_s .- M64.v_s)) < 1e-2 && maximum(abs.(M32.v_d1 .- M64.v_d1)) < 1e-2
        # one precision per population; convert_precision builds the synapse models of the other one
        @test_throws ArgumentError Multipod(N = 2, VFT = Vector{Float64}, param = MultipodParameter(ds = fill(300um, D)), soma_syn = TripodSomaSynapse)
        M2 = Multipod(N = 2, VFT = Vector{Float64}, param = MultipodParameter(ds = fill(300um, D)),
                      soma_syn = convert_precision(M32.soma_syn, Float64), dend_syn = convert_precision(M32.dend_syn, Float64))
        @test M2.soma_syn.syn == M64.soma_syn.syn && convert_precision(M2.dend_syn, Float32).syn == M32.dend_syn.syn
    end
    # PaperScheme in Float64
    M = MultipodPaper(N = 2, lengths = [400um, 150um, 150um], VFT = Vector{Float64})
    M.I .= 1200.0
    nsp = 0
    for _ in 1:3000; SNNModels.integrate!(M, M.param, 0.1); nsp += count(M.fire); end
    @test nsp > 10
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Plasticity on a dendritic synapse
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "Multipod: vSTDP on a synapse onto :d3" begin
    Random.seed!(21)
    M = Multipod(N = 30, param = MultipodParameter(ds = [300um, 300um, 200um, 400um]))
    M.I .= 600f0
    pre = IF(N = 40)
    pre.I .= range(250f0, 400f0, length = 40)
    syn = SpikingSynapse(pre, M, :glu, :d3; conn = (p = 0.5, μ = 5.0), LTPParam = vSTDPParameter())
    @test syn.v_post === M.v_d3                                    # the rule reads the potential of d3
    model = compose(; pre, M, syn, silent = true)
    monitor!(M, [:fire]; monitor_rate = 100Hz)
    W0 = copy(syn.W)
    train!(; model, duration = 500ms)
    SNNModels.sync_weights!(syn)
    @test syn.W != W0 && all(isfinite, syn.W)
    @test sum(length, spiketimes(M)) > 0
    # sim! leaves the weights alone
    W1 = copy(syn.W)
    sim!(; model, duration = 100ms)
    SNNModels.sync_weights!(syn)
    @test syn.W == W1
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Threading invariance
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "Multipod threading invariance (neurons_per_thread)" begin
    for (D, paper) in ((3, false), (5, false), (4, true))
        Random.seed!(8)
        M = paper ? MultipodPaper(N = 501, lengths = [400um, 300um, 200um, 150um]) :
            Multipod(N = 501, param = MultipodParameter(ds = fill((150um, 400um), D)))
        M.I .= range(0f0, 1500f0, length = 501)
        pre = IF(N = 100)
        pre.I .= range(250f0, 450f0, length = 100)
        syn = SpikingSynapse(pre, M, :glu, Symbol(:d, D); conn = (p = 0.2, μ = 3.0))
        model = compose(; pre, M, syn, silent = true)
        a = deepcopy(model); b = deepcopy(model)
        sim!(; model = a, duration = 200ms, dt = 0.1ms, neurons_per_thread = 0)
        sim!(; model = b, duration = 200ms, dt = 0.1ms, neurons_per_thread = 32)
        Ma, Mb = a.pop.M, b.pop.M
        @test Ma.v_s == Mb.v_s && Ma.w_s == Mb.w_s && Ma.tabs == Mb.tabs
        @test all(getfield(Ma, Symbol(:v_d, k)) == getfield(Mb, Symbol(:v_d, k)) for k in 1:ndendrites(Ma))
        @test any(Ma.tabs .!= 0) || any(Ma.w_s .> 0)                # the runs contain spikes
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Network smoke test
# ---------------------------------------------------------------------------------------------------------------------------------

@testset "Multipod balanced network, $D dendrites" for D in (3, 5)
    Random.seed!(D)
    E = Multipod(N = 200, name = "E", param = MultipodParameter(ds = fill((150um, 400um), D)))
    I = IF(N = 50, name = "I", param = IFParameter(τm = 10ms, R = 0.06, El = -55mV))
    # recurrent excitation onto every dendrite (total weight independent of D), inhibition onto the soma and the first dendrite
    EE = NamedTuple(Symbol(:EE_d, k) => SpikingSynapse(E, E, :glu, Symbol(:d, k); conn = (p = 0.05, μ = 12.0 / D)) for k in 1:D)
    EI = SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 1.0))
    IE_s = SpikingSynapse(I, E, :gaba, :s; conn = (p = 0.2, μ = 4.0))
    IE_d = SpikingSynapse(I, E, :gaba, :d1; conn = (p = 0.2, μ = 4.0))
    II = SpikingSynapse(I, I, :gi; conn = (p = 0.2, μ = 2.0))
    # background: Poisson input on every dendrite and on the inhibitory population
    stims = NamedTuple(Symbol(:noise_d, k) => Stimulus(PoissonLayer(20Hz, N = 100), E, :glu, Symbol(:d, k); conn = (μ = 18.0 / D, ρ = 0.5))
                       for k in 1:D)
    noise_I = Stimulus(PoissonLayer(20Hz, N = 100), I, :ge; conn = (μ = 1.5, ρ = 0.5))
    model = compose(; E, I, EE..., EI, IE_s, IE_d, II, stims..., noise_I, silent = true)
    monitor!(E, [:fire, :v_s, Symbol(:v_d, D)]; monitor_rate = 200Hz)
    monitor!(I, [:fire]; monitor_rate = 500Hz)
    sim!(; model, duration = 500ms)
    rE = sum(length, spiketimes(E)) / (200 * 0.5)
    rI = sum(length, spiketimes(I)) / (50 * 0.5)
    @info "Multipod network" D rate_E = rE rate_I = rI
    @test 0 < rE < 200 && rI > 0
    @test all(isfinite, E.v_s) && all(k -> all(isfinite, getfield(E, Symbol(:v_d, k))), 1:D)
    @test size(getvariable(E, Symbol(:v_d, D)), 1) == 200
end
