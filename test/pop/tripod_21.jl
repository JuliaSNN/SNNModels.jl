# SNNModels 2.1.0: Tripod / BallAndStick with the soma clamped, dendrite diameter, fixed paper configurations, mouse synapses,
# NMDA-block and decay options, `TripodPaper` (reproduction of the published code, aquaresima/tripod_neuron @ 8698e1a).
# Reference traces of the original code (test/data) were produced by running that code unmodified (validation/tripod_audit of the
# JuliaSNN publication repository, scripts/run_orig.jl) on the audit protocols.
using Test
using SNNModels
@load_units

_data(name) = joinpath(@__DIR__, "..", "data", name)
function _read_matrix(name)
    rows = [parse.(Float64, split(l)) for l in eachline(_data(name)) if !startswith(l, "#") && !isempty(strip(l))]
    return reduce(vcat, (permutedims(r) for r in rows))
end

# ---------------------------------------------------------------------------------------------------------------------------------
# 1. The dendrites are integrated in full while the soma is clamped
# ---------------------------------------------------------------------------------------------------------------------------------
@testset "Clamped soma: dendrites keep leak, synaptic and axial currents" begin
    dt = 0.1f0
    spike = PostSpike(τabs = 2ms, up = 1ms, At = 0mV, AP_membrane = 20mV)
    n_abs = 20
    # synaptic current of the Tripod dendrite synapse with `g` (conductance, before gsyn), by the definition of the model
    function isyn(syn, g, v)
        s = 0.0
        for (n, r) in enumerate(syn.syn)
            B = r.nmda == 0 ? 1.0 : nmda_gating(Float64(v), NMDAVoltageDependency(T = Float64, b = 3.36, k = -0.077, mg = 1.0))
            s += r.gsyn * g[n] * (v - r.E_rev) * B
        end
        return clamp(s, -1500, 1500)
    end
    for tabs0 in (31, 21, 3)         # first backpropagation step, last backpropagation step, refractory
        T = Tripod(N = 1, param = TripodParameter(ds = [250um, 150um]), spike = spike)
        T.v_s .= -70.6f0; T.v_d1 .= -45f0; T.v_d2 .= -62f0; T.w_s .= 10f0; T.θ .= T.adex.Vt
        T.tabs .= tabs0 + 1              # `tabs - 1` is the counter of this step
        T.synvars_d1.g[1, :] .= (0.4f0, 0.8f0, 0.1f0, 0.05f0)     # AMPA, NMDA, GABAa, GABAb conductances
        T.synvars_d2.g[1, :] .= (0.0f0, 0.0f0, 0.3f0, 0.0f0)
        vsc = Float64(tabs0 > n_abs ? 20.0f0 : T.adex.Vr)
        # reference: one Heun step of each dendrite with the soma at `vsc`, with the conductances after the synapse update
        T2 = deepcopy(T)
        SNNModels.update_synapses!(T2, T2.dend_syn, T2.receptors_d1, T2.synvars_d1, dt)
        SNNModels.update_synapses!(T2, T2.dend_syn, T2.receptors_d2, T2.synvars_d2, dt)
        w0 = T.w_s[1]
        SNNModels.integrate!(T, T.param, dt)
        for (vd0, d, sv, vd_new) in ((-45.0, T2.d1, T2.synvars_d1, T.v_d1[1]), (-62.0, T2.d2, T2.synvars_d2, T.v_d2[1]))
            f(v) = ((d.El[1] - v) * d.gm[1] - isyn(T2.dend_syn, sv.g[1, :], v) + d.gax[1] * (vsc - v)) / d.C[1]
            k1 = f(vd0); k2 = f(vd0 + dt * k1)
            @test vd_new ≈ vd0 + dt / 2 * (k1 + k2) rtol = 2e-5
        end
        @test T.v_s[1] == Float32(vsc)                     # clamped
        @test T.w_s[1] == w0                               # w frozen
        @test T.tabs[1] == tabs0
        @test !T.fire[1]
    end
    # the same for BallAndStick
    B = BallAndStick(N = 1, param = BallAndStickParameter(ds = [250um]), spike = spike)
    B.v_s .= -70.6f0; B.v_d .= -45f0; B.w_s .= 10f0; B.θ .= B.adex.Vt; B.tabs .= 4
    B.synvars_d.g[1, :] .= (0.0f0, 1.0f0, 0.0f0, 0.0f0)
    vd0 = B.v_d[1]
    B2 = deepcopy(B)
    SNNModels.integrate!(B, B.param, dt)
    # without the synaptic current the dendrite relaxes only by leak and axial current: the NMDA conductance makes the difference
    B2.synvars_d.g .= 0f0
    SNNModels.integrate!(B2, B2.param, dt)
    @test B.v_d[1] != B2.v_d[1]
    @test B.v_d[1] > B2.v_d[1]            # NMDA (E = 0 mV) depolarises the dendrite at -45 mV
    # leak: with the soma clamped at El and no synapses the dendrite at rest does not move; away from it the leak acts
    # (before 2.1 the clamp step was axial only)
    B3 = BallAndStick(N = 1, param = BallAndStickParameter(ds = [250um]), spike = spike)
    B3.v_s .= -70.6f0; B3.v_d .= -40f0; B3.θ .= B3.adex.Vt; B3.tabs .= 4
    gax, gm, C = B3.d.gax[1], B3.d.gm[1], B3.d.C[1]
    SNNModels.integrate!(B3, B3.param, dt)
    k1 = ((-70.6 + 40) * gm + gax * (-70.6 + 40)) / C
    k2 = ((-70.6 - (-40 + dt * k1)) * gm + gax * (-70.6 - (-40 + dt * k1))) / C
    @test B3.v_d[1] ≈ -40 + dt / 2 * (k1 + k2) rtol = 1e-5
    axial_only = -40 + dt * (-70.6 + 40) * gax / C
    @test B3.v_d[1] != axial_only
end

# parameters of the audit protocols for the original code (human, d = 4 um, dt 0.1 ms) with the numerics of the library defaults
const _ORIG_SOMA = ReceptorSynapse(glu_receptors = [1], gaba_receptors = [2], NMDA = EyalNMDA, syn = TripodSomaSynapsePaper.syn)
const _ORIG_DEND = ReceptorSynapse(glu_receptors = [1, 2], gaba_receptors = [3, 4], NMDA = EyalNMDA, syn = TripodDendSynapsePaper.syn)
function _plateau_default(l, eff; n = 10000)
    T = Tripod(N = 1, param = TripodParameter(ds = [l * um, l * um]), adex = AdExParameter(Vt = -50.4mV),
               spike = PostSpike(AP_membrane = 20mV, up = 1ms, τabs = 2ms, At = 0mV), soma_syn = _ORIG_SOMA, dend_syn = _ORIG_DEND,
               Vspike = -50.4mV)
    El = T.adex.El; T.v_s .= El; T.v_d1 .= El; T.v_d2 .= El; T.w_s .= 0; T.θ .= T.adex.Vt
    cnt = 0; nsp = 0
    for tt in 1:n
        T.receptors_d1.glu[1] = tt == 150 ? Float32(eff) : 0f0
        SNNModels.integrate!(T, T.param, 0.1f0)
        nsp += T.fire[1]
        tt >= 150 && T.v_d1[1] >= -60f0 && (cnt += 1)
    end
    return 0.1 * cnt, nsp
end

@testset "Fig. 8 plateau is not cut by the somatic spike" begin
    # One NMDA event of weight 100 on d1 (both dendrites of length l), plateau = time with V_d1 >= -60 mV. The reference values are
    # those of the switchable reference stepper of the audit (ref.jl: SNNModels numerics with the original's refractory handling,
    # i.e. dendrites integrated in full during the clamp). In 2.0.3 the plateau of l = 250 um was 28.7 ms (one somatic spike
    # ended it), the original code gives 58.4 ms (its exp32 Mg block makes it longer than with the library's exp256).
    ref = Dict(400 => 67.2, 250 => 54.2, 150 => 13.6)      # ms, ref.jl, SNNM numerics + original refractory
    old = Dict(400 => 67.2, 250 => 28.7, 150 => 13.2)      # ms, 2.0.3 (axial-only dendrites during the clamp)
    for l in (400, 250, 150)
        pl, nsp = _plateau_default(l, 100)
        @test abs(pl - ref[l]) <= 0.5
        l == 250 && @test nsp == 1 && pl > old[l] + 20       # the spike no longer ends the plateau
        l == 400 && @test nsp == 0 && pl == old[l]
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------
# 2. Mouse synapses, NMDA block option and decay option
# ---------------------------------------------------------------------------------------------------------------------------------
@testset "Mouse synapse set (DuarteGluDend)" begin
    ampa, nmda = DuarteGluDend.AMPA, DuarteGluDend.NMDA
    @test (ampa.E_rev, ampa.τr, ampa.τd, ampa.g0) == (0.0f0, 0.25f0, 2.0f0, 0.73f0)
    @test (nmda.E_rev, nmda.τr, nmda.τd, nmda.g0, nmda.nmda) == (0.0f0, 0.99f0, 100.0f0, 0.159f0, 1.0f0)
    S = TripodDendSynapseMouse
    @test S.syn[1] === ampa && S.syn[2] === nmda && S.glu_receptors == [1, 2] && S.gaba_receptors == [3, 4]
    @test S.NMDA === EyalNMDA                        # the published code uses the human block for the mouse as well
    @test MouseNMDA.b == 3.57f0 && MouseNMDA.k == -0.062f0 && MouseNMDA.approx == 256
    # the peak conductance of a unit weight is g0: NMDA (peak of the double exponential times gsyn)
    @test nmda.gsyn ≈ 0.159f0 * SNNModels.norm_synapse(0.99f0, 100.0f0) rtol = 1e-5
    # selectable for Tripod; the default does not change
    @test Tripod(N = 1).dend_syn === TripodDendSynapse
    T = Tripod(N = 2, dend_syn = TripodDendSynapseMouse, param = TripodParameter(ds = distal_proximal, physiology = mouse_dend))
    @test T.dend_syn.syn[2].τd == 100.0f0
    T.receptors_d1.glu[1] = 5f0
    vmax = -Inf32
    for _ in 1:100; SNNModels.integrate!(T, T.param, 0.1f0); vmax = max(vmax, T.v_d1[1]); end
    @test vmax > -69 && all(isfinite, T.v_d1)
    # the NMDA tail is longer than in the human set (τd 100 vs 35 ms): the conductance after 50 ms
    Th = Tripod(N = 1, param = TripodParameter(ds = distal_proximal))
    Tm = Tripod(N = 1, param = TripodParameter(ds = distal_proximal), dend_syn = TripodDendSynapseMouse)
    for P in (Th, Tm); P.receptors_d1.glu[1] = 1f0; end
    for _ in 1:500; SNNModels.integrate!(Th, Th.param, 0.1f0); SNNModels.integrate!(Tm, Tm.param, 0.1f0); end
    @test Tm.synvars_d1.g[1, 2] > 1.5 * Th.synvars_d1.g[1, 2]
    # Float64
    S64 = convert_precision(TripodDendSynapseMouse, Float64)
    @test S64.syn[2].τr === 0.99 && S64.syn[2].g0 === 0.159
end

@testset "NMDA block exponential option (approx) and receptor decay option" begin
    # B(-70) and B(-40): exp32 of the original, exp256 (default), exact
    @test nmda_gating(-70f0, EyalNMDA32) ≈ 0.0225 atol = 5e-5
    @test nmda_gating(-40f0, EyalNMDA32) ≈ 0.1508 atol = 5e-5
    @test nmda_gating(-70f0, EyalNMDA) ≈ 0.0160 atol = 5e-5
    @test nmda_gating(-70f0, NMDAVoltageDependency(approx = 0)) ≈ 1 / (1 + exp(-0.077 * -70) / 3.36) rtol = 1e-5
    @test nmda_gating(-70f0, NMDAVoltageDependency(approx = 64)) != nmda_gating(-70f0, EyalNMDA)
    @test NMDAVoltageDependency().approx == 256 && EyalNMDA.approx == 256 && SomaNMDA.approx == 256     # default unchanged
    @test_throws ArgumentError NMDAVoltageDependency(approx = 100)
    @test_throws ArgumentError ReceptorSynapse(decay = 3)
    # the receptor current uses the selected block
    function current(approx)
        syn = ReceptorSynapse(; syn = TripodDendSynapse.syn, NMDA = NMDAVoltageDependency(approx = approx), glu_receptors = [1, 2], gaba_receptors = [3, 4])
        T = Tripod(N = 1, dend_syn = syn)
        T.synvars_d1.g[1, 2] = 1f0
        out = zeros(Float32, 1)
        SNNModels.synaptic_current!(T, T.dend_syn, T.synvars_d1, Float32[-70.0], out)
        return out[1]
    end
    @test current(32) < current(256) < current(0) < 0       # inward currents: a larger block factor gives a larger magnitude
    @test current(32) / current(256) ≈ nmda_gating(-70f0, EyalNMDA32) / nmda_gating(-70f0, EyalNMDA) rtol = 1e-5
    # decay: one step after a unit input, g = E(-dt/τd) (dt h) with h = α
    for (decay, E) in ((64, SNNModels.exp64), (32, SNNModels.exp32o), (256, SNNModels.exp256), (0, exp))
        syn = ReceptorSynapse(; syn = TripodDendSynapse.syn, glu_receptors = [1, 2], gaba_receptors = [3, 4], decay = decay)
        T = Tripod(N = 3, dend_syn = syn)
        T.receptors_d1.glu .= 1f0
        SNNModels.update_synapses!(T, syn, T.receptors_d1, T.synvars_d1, 0.1f0, 1:3)
        r = syn.syn[1]
        h = r.α
        @test T.synvars_d1.g[2, 1] ≈ E(-0.1f0 * r.τd⁻) * 0.1f0 * h rtol = 1e-6
        @test T.synvars_d1.h[2, 1] ≈ E(-0.1f0 * r.τr⁻) * h rtol = 1e-6
        @test all(==(0), T.receptors_d1.glu)
    end
    # exp32o maps inputs below -10 to 0 (the exp32 of the original code)
    @test SNNModels.exp32o(-11f0) == 0f0 && SNNModels.exp32o(0f0) == 1f0 && SNNModels.exp32o(-1f0) ≈ (1 - 1 / 32f0)^32
    # Float64 conversion keeps the options
    @test convert_precision(TripodDendSynapsePaper, Float64).decay == 32
    @test convert_precision(TripodDendSynapsePaper, Float64).NMDA.approx == 32
    @test convert_precision(NMDAVoltageDependency(approx = 64, b = 3.0f0), Float64).approx == 64
end

# ---------------------------------------------------------------------------------------------------------------------------------
# 3. Dendrite diameter
# ---------------------------------------------------------------------------------------------------------------------------------
@testset "Dendrite diameter" begin
    for FT in (Float32, Float64)
        mk = (; d...) -> Tripod(N = 4, VFT = Vector{FT}, param = TripodParameter(ds = [160um, 300um]; d...))
        ref(l; kw...) = FT === Float32 ? create_dendrite(l; kw...) : SNNModels._create_dendrite_decimal(l; kw...)
        # the default is 4 um and reproduces the geometry of 2.0 exactly (create_dendrite default)
        T0 = mk()
        @test TripodParameter().d == 4um
        for (dd, l) in ((T0.d1, 160um), (T0.d2, 300um))
            r = ref(l)
            @test all(==(FT(r.gm)), dd.gm) && all(==(FT(r.gax)), dd.gax) && all(==(FT(r.C)), dd.C) && all(==(FT(r.d)), dd.d)
        end
        @test mk(d = 4um).d1.gax == T0.d1.gax
        # another diameter follows the cable formulas: gax ~ d^2, gm ~ d, C ~ d at fixed length
        T6 = mk(d = 6um)
        for (a, b) in ((T6.d1, T0.d1), (T6.d2, T0.d2))
            @test a.gax[1] / b.gax[1] ≈ (6 / 4)^2 rtol = 1e-5
            @test a.gm[1] / b.gm[1] ≈ 6 / 4 rtol = 1e-5
            @test a.C[1] / b.C[1] ≈ 6 / 4 rtol = 1e-5
            @test all(x -> isapprox(x, FT(6um); rtol = 1e-6), a.d)
        end
        # explicit values: G_axial = pi d^2 / (4 Ri l)
        @test T6.d1.gax[1] ≈ G_axial(Ri = 200Ω * cm, d = 6um, l = 160um) rtol = 1e-5
        @test T6.d1.gm[1] ≈ G_mem(Rd = 38907Ω * cm^2, d = 6um, l = 160um) rtol = 1e-5
        @test T6.d1.C[1] ≈ C_mem(Cd = 0.5μF / cm^2, d = 6um, l = 160um) rtol = 1e-5
        # BallAndStick
        B0 = BallAndStick(N = 2, VFT = Vector{FT}, param = BallAndStickParameter(ds = [200um]))
        B3 = BallAndStick(N = 2, VFT = Vector{FT}, param = BallAndStickParameter(ds = [200um], d = 3um))
        @test B3.d.gax[1] / B0.d.gax[1] ≈ (3 / 4)^2 rtol = 1e-5
        @test B3.d.C[1] / B0.d.C[1] ≈ 3 / 4 rtol = 1e-5
        @test all(==(FT(ref(200um).gax)), B0.d.gax)
    end
    # the diameter changes the dynamics: a thicker dendrite is coupled more strongly (larger gax) and has more leak
    @test DendNeuronParameter(d = 5um).d == 5um
end

# ---------------------------------------------------------------------------------------------------------------------------------
# 4. Fixed configurations, lengths above 500 um
# ---------------------------------------------------------------------------------------------------------------------------------
@testset "Fixed configurations and length bound" begin
    ls(T) = (round(T.d1.l[1] / um), round(T.d2.l[1] / um))
    for (ds, expected) in ((distal_distal, (400, 400)), (proximal_proximal, (150, 150)), (proximal_distal, (150, 400)),
                           (distal_proximal, (400, 150)))
        T = Tripod(N = 5, param = TripodParameter(ds = ds))
        @test ls(T) == expected
        @test all(==(T.d1.l[1]), T.d1.l) && all(==(T.d2.l[1]), T.d2.l)        # no randomness
        @test T.d1.gax[1] ≈ create_dendrite(expected[1] * um).gax rtol = 1e-6
    end
    @test distal_proximal == [400um, 150um] && proximal_distal == [150um, 400um]
    # random presets (the former `proximal_distal` / `proximal_proximal`)
    @test random_proximal_distal == [(150um, 400um), (150um, 400um)]
    @test random_proximal_proximal == [(150um, 300um), (150um, 300um)]
    T = Tripod(N = 200, param = TripodParameter(ds = random_proximal_distal))
    @test length(unique(T.d1.l)) > 20 && all(150um - 1f-6 .<= T.d1.l .<= 400um + 1f-6)
    T = Tripod(N = 200, param = TripodParameter(ds = random_proximal_proximal))
    @test length(unique(T.d2.l)) > 20 && all(150um - 1f-6 .<= T.d2.l .<= 300um + 1f-6)
    # lengths above 500 um (paper Fig. 3 uses 550 um)
    d = create_dendrite(550um)
    @test d.gax ≈ G_axial(Ri = 200Ω * cm, d = 4um, l = 550um) rtol = 1e-5
    @test Tripod(N = 1, param = TripodParameter(ds = [550um, 550um])).d1.gm[1] > 0
    @test create_dendrite(2, (500um, 700um)).l[1] <= 700um
    @test_throws ErrorException create_dendrite(400)          # a length without its unit (400 cm)
    @test_throws ErrorException create_dendrite(1500um)
    # soma-only control: disconnected dendrites
    T = Tripod(N = 2, param = TripodParameter(ds = soma_only))
    @test all(==(0), T.d1.gax) && all(==(0), T.d2.gax)
    T.I .= 1500f0
    nsp = 0
    for _ in 1:2000; SNNModels.integrate!(T, T.param, 0.1f0); nsp += count(T.fire); end
    @test nsp > 0
end

# ---------------------------------------------------------------------------------------------------------------------------------
# 5. TripodPaper
# ---------------------------------------------------------------------------------------------------------------------------------
function _run_paper(T, n; inputs = nothing, I = nothing, rec = nothing)
    spikes = Float32[]
    for tt in 1:n
        if inputs !== nothing
            for (r, buf) in enumerate((T.receptors_s.glu, T.receptors_d1.glu, T.receptors_d2.glu, T.receptors_s.gaba, T.receptors_d1.gaba, T.receptors_d2.gaba))
                v = inputs(r, tt)
                v != 0 && (buf[1] = v)
            end
        end
        I === nothing || (T.I[1] = I(tt * 0.1f0))
        SNNModels.integrate!(T, T.param, 0.1f0)
        T.fire[1] && push!(spikes, tt * 0.1f0)
        rec === nothing || push!(rec, (T.v_s[1], T.v_d1[1], T.v_d2[1]))
    end
    return spikes
end

@testset "TripodPaper: configuration" begin
    T = TripodPaper(N = 3)
    @test T isa Tripod && T.scheme isa PaperScheme && T.scheme.AP_spike == 21f0
    a = T.adex
    @test (a.C, a.gl, a.El, a.Vr, a.ΔT, a.Vt, a.a, a.b, a.τw) == (281f0, 40f0, -70.6f0, -70.6f0, 2f0, -50.4f0, 4f0, 80.5f0, 144f0)
    @test T.Vspike == -50.4f0
    sp = T.spike
    @test (sp.AP_membrane, sp.up, sp.τabs, sp.At) == (20f0, 1f0, 2.1f0, 0f0)
    @test all(==(-70.6f0), T.v_s) && all(==(-70.6f0), T.v_d1) && all(==(-70.6f0), T.v_d2) && all(==(0), T.w_s)
    @test all(==(-50.4f0), T.θ)
    @test T.param.d == 4um && T.param.physiology === human_dend
    @test (round(T.d1.l[1] / um), round(T.d2.l[1] / um)) == (400, 150)               # default configuration of the original
    for (config, ls) in ((:distal_distal, (400, 400)), (:proximal_proximal, (150, 150)), (:proximal_distal, (150, 400)),
                         (:distal_proximal, (400, 150)))
        Tc = TripodPaper(N = 1; config)
        @test (round(Tc.d1.l[1] / um), round(Tc.d2.l[1] / um)) == ls
    end
    @test all(==(0), TripodPaper(N = 1, config = :soma_only).d1.gax)
    # the human cable constants of the paper (Table 3, 400 um): 1.292 nS, 15.708 nS, 25.13 pF
    T400 = TripodPaper(N = 1, config = :distal_distal)
    @test T400.d1.gm[1] ≈ 1.292 atol = 1e-3
    @test T400.d1.gax[1] ≈ 15.708 atol = 1e-3
    @test T400.d1.C[1] ≈ 25.13 atol = 1e-2
    # receptors of the original code
    @test T.soma_syn === TripodSomaSynapsePaper || T.soma_syn == TripodSomaSynapsePaper
    s = T.soma_syn.syn
    @test (s[1].E_rev, s[1].τr, s[1].τd, s[1].g0) == (0f0, 0.25f0, 2f0, 0.73f0)
    @test (s[2].E_rev, s[2].τr, s[2].τd, s[2].g0) == (-75f0, 0.3f0, 15f0, 0.38f0)
    dsyn = T.dend_syn.syn
    @test [r.E_rev for r in dsyn] == [0f0, 0f0, -75f0, -90f0]
    @test [r.τr for r in dsyn] == [0.26f0, 8f0, 4.8f0, 30f0] && [r.τd for r in dsyn] == [2f0, 35f0, 29f0, 400f0]
    @test T.dend_syn.NMDA.approx == 32 && T.dend_syn.decay == 32 && T.soma_syn.decay == 32
    @test nmda_gating(-70f0, T.dend_syn.NMDA) ≈ 0.0225 atol = 5e-5
    @test nmda_gating(-40f0, T.dend_syn.NMDA) ≈ 0.1508 atol = 5e-5
    # mouse
    M = TripodPaper(N = 1, physiology = :mouse)
    @test M.param.physiology === mouse_dend
    @test M.dend_syn.syn[1].τr == 0.25f0 && M.dend_syn.syn[2].τr == 0.99f0 && M.dend_syn.syn[2].τd == 100f0 && M.dend_syn.syn[2].g0 == 0.159f0
    @test M.dend_syn.syn[3].E_rev == -75f0 && M.dend_syn.NMDA.b == 3.36f0
    @test M.d1.C[1] ≈ 2 * T.d1.C[1] rtol = 1e-5        # Cd 1 vs 0.5 uF/cm2
    # keywords
    @test TripodPaper(N = 2, d = 3um).d1.gax[1] / T.d1.gax[1] ≈ (3 / 4)^2 rtol = 1e-5
    @test TripodPaper(N = 2, name = "x").name == "x"
    @test_throws ArgumentError TripodPaper(physiology = :rat)
    @test_throws ArgumentError TripodPaper(config = :wide)
    @test_throws ArgumentError Tripod(N = 1, scheme = PaperScheme(), soma_syn = DeltaSynapse())
    # Float64 runs the same scheme
    T64 = TripodPaper(N = 2, VFT = Vector{Float64})
    @test eltype(T64.v_s) === Float64 && T64.adex isa AdExParameter{Float64}
    T64.I .= 1200.0
    nsp = 0
    for _ in 1:3000; SNNModels.integrate!(T64, T64.param, 0.1); nsp += count(T64.fire); end
    @test nsp > 10
    # default Tripod unchanged
    @test Tripod(N = 1).scheme isa HeunScheme
end

@testset "TripodPaper: spike sequence and dt warning" begin
    # spike step, 9 backpropagation steps at 20 mV, 21 refractory steps at Vr (original: 21 mV spike step, `last_event` 30)
    T = TripodPaper(N = 1)
    T.v_s .= -50.4f0                      # at the threshold: it fires on the next step (rule `V_s >= Vspike` before integrating)
    vs = Float32[]
    for _ in 1:60
        SNNModels.integrate!(T, T.param, 0.1f0)
        push!(vs, T.v_s[1])
        T.fire[1] && @test length(vs) == 1
    end
    @test vs[1] == 21f0
    @test all(==(20f0), vs[2:10])
    @test all(==(-70.6f0), vs[11:31])
    @test vs[32] != -70.6f0 && vs[32] < -60                 # free integration again
    @test T.w_s[1] > 70          # w += b on the spike step, frozen during the clamp, then slow decay
    # dt other than 0.1 ms warns
    T = TripodPaper(N = 1)
    @test_logs (:warn, r"dt = 0.1 ms") SNNModels.integrate!(T, T.param, 0.125f0)
end

@testset "TripodPaper vs original code (reference traces, dt 0.1 ms, identical inputs)" begin
    # subthreshold current step
    ref = _read_matrix("tripod_orig_step_sub.txt")
    T = TripodPaper(N = 1); rec = Tuple{Float32,Float32,Float32}[]
    _run_paper(T, size(ref, 1); I = t -> 50 <= t < 250 ? 300f0 : 0f0, rec)
    dv = maximum(abs(rec[i][c] - ref[i, c]) for i in axes(ref, 1), c in 1:3)
    @test dv < 0.05                                      # acceptance; achieved < 1e-3
    @test dv < 1e-3
    # single EPSP on d1
    ref = _read_matrix("tripod_orig_epsp.txt")
    T = TripodPaper(N = 1); rec = Tuple{Float32,Float32,Float32}[]
    _run_paper(T, size(ref, 1); inputs = (r, tt) -> (r == 2 && tt == 500) ? 1f0 : 0f0, rec)
    dv = maximum(abs(rec[i][c] - ref[i, c]) for i in axes(ref, 1), c in 1:3)
    @test dv < 0.05 && dv < 1e-3
    # 1200 pA step: same spike count, all spikes within 0.2 ms
    sref = vec(_read_matrix("tripod_orig_step_supra_spikes.txt"))
    T = TripodPaper(N = 1)
    sp = _run_paper(T, 5000; I = t -> 50 <= t < 450 ? 1200f0 : 0f0)
    @test length(sp) == length(sref) == 15
    @test maximum(abs.(sp .- sref)) < 0.2
    # frozen Poisson input (first 500 ms): same spike count, spikes within 1 ms
    ev = _read_matrix("tripod_poisson_events.txt")
    X = zeros(Float32, 6, 5000)
    for k in axes(ev, 1); X[Int(ev[k, 1]), Int(ev[k, 2])] = Float32(ev[k, 3]); end
    sref = vec(_read_matrix("tripod_orig_poisson_spikes.txt"))
    T = TripodPaper(N = 1)
    sp = _run_paper(T, 5000; inputs = (r, tt) -> X[r, tt])
    @test length(sp) == length(sref) == 22
    @test count(abs.(sp .- sref) .<= 1) >= 0.9 * length(sref)
    @test maximum(abs.(sp .- sref)) < 0.2                # achieved: identical spike times
    # Fig. 8 plateau durations (event of weight 100 on d1, both dendrites of length l); original code: 71.1, 58.4, 28.7 ms
    for (l, orig) in ((400, 71.1), (250, 58.4), (150, 28.7))
        T = TripodPaper(N = 1, param = TripodParameter(ds = [l * um, l * um]))
        cnt = 0
        for tt in 1:10000
            T.receptors_d1.glu[1] = tt == 150 ? 100f0 : 0f0
            SNNModels.integrate!(T, T.param, 0.1f0)
            tt >= 150 && T.v_d1[1] >= -60f0 && (cnt += 1)
        end
        @test abs(0.1 * cnt - orig) <= 2.0
    end
end

@testset "TripodPaper in a network (chunks, sim!)" begin
    N = 64
    T = TripodPaper(N = N)
    E = PoissonLayer(20Hz, N = 100)
    syn = Stimulus(E, T, :glu, :d1; conn = (μ = 8.0, ρ = 1), name = "noise_d1")
    synI = Stimulus(E, T, :glu, :s; conn = (μ = 4.0, ρ = 1), name = "noise_s")
    model = compose(; T, syn, synI, silent = true)
    monitor!(T, [:fire])
    sim!(; model, duration = 300ms, dt = 0.1ms)
    @test sum(length, spiketimes(T)) > 0
    @test all(isfinite, T.v_s) && all(isfinite, T.v_d1)
end
