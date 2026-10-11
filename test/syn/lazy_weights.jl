using SNNModels
using Test
using Random
@load_units

# Deferred weight updates (the event-driven vSTDPParameter): `syn.W` is stored lazily and is read through the same accessor as
# the lazily decayed traces, `plasticity_value(syn, :W)` / `trace_values(syn, :W)`, with `sync_traces!(syn)` to
# materialise it. Every reader of `syn.W` in the library goes through the accessor or syncs first.

# a Float64 network driven by given spikes and a given postsynaptic potential (no neuron model): `step!` is one train! step
function _lazy_net(Rule; FT = Float64, Npre = 12, Npost = 6, nsteps = 6000, seed = 3, kw = (;))
    rng = MersenneTwister(seed)
    pre = Identity(N = Npre, VFT = Vector{FT})
    post = Identity(N = Npost, VFT = Vector{FT})
    times = FT[]; neurons = Int[]
    for j in 1:Npre, n in 1:nsteps
        rand(rng) < 0.002 && (push!(times, n * 0.1); push!(neurons, j))
    end
    order = sortperm(times)
    stim = SpikeTimeStimulusIdentity(pre, :g; param = SpikeTimeStimulusParameter(times[order], neurons[order]))
    rule = Rule(FT = FT, A_LTD = 8e-4, A_LTP = 5e-3, θ_LTD = -70.0, θ_LTP = -49.0, τu = 20.0, τv = 7.0, τx = 15.0,
                Wmin = 0.0, Wmax = 1.0; kw...)
    syn = SpikingSynapse(pre, post, :g; conn = fill(0.5, Npost, Npre), LTPParam = rule)
    V = FT[-70 + 25 * (0.5 + 0.5 * sin(2π * n * 0.1 / (150 + 37i))) ^ 8 for i in 1:Npost, n in 1:nsteps]   # crosses θ_LTP
    return (; pre, post, stim, syn, V, nsteps, T = Time{FT}(), dt = FT(0.1))
end

function _lazy_run!(net, steps = net.nsteps - 1; from = 1)
    (; pre, post, stim, syn, V, T, dt) = net
    for n in from:from+steps-1
        syn.v_post .= view(V, :, n + 1)
        SNNModels.train!([pre, post], [syn], [stim], dt, T)
    end
    return net
end

@testset "deferred weights through the trace accessor" begin
    @testset "declaration" begin
        @test :W in SNNModels.lazy_variables(vSTDPVariables(Npre = 2, Npost = 2))
        @test :W ∉ SNNModels.lazy_variables(vSTDPDenseVariables(Npre = 2, Npost = 2))
        @test :W ∉ SNNModels.lazy_variables(STDPVariables(Npre = 2, Npost = 2))
        @test :W ∉ SNNModels.lazy_variables(NoVariables())
    end

    @testset "values, sync, dense agreement (Float64)" begin
        ev = _lazy_run!(_lazy_net(vSTDPParameter))
        de = _lazy_run!(_lazy_net(vSTDPDenseParameter))
        We = copy(plasticity_value(ev.syn, :W))
        @test trace_values(ev.syn, :W) == We
        @test SNNModels.current_weights(ev.syn) == We            # the thin wrapper
        @test maximum(abs.(de.syn.W .- 0.5)) > 0.05             # the weights moved
        @test maximum(abs.(We .- de.syn.W)) < 1e-9               # Float64: event and dense agree to the accumulation error of the epoch
        @test maximum(abs.(ev.syn.W .- We)) > 0                  # stored weights are stale
        @test plasticity_value(ev.syn, :W) === plasticity_value(ev.syn, :W)   # one scratch vector, no allocation
        sync_traces!(ev.syn)
        @test ev.syn.W == We                                     # sync_traces! materialises the weights
        @test plasticity_value(ev.syn, :W) == We
        # the dense rule and the other rules: the accessor is the field itself
        @test plasticity_value(de.syn, :W) === de.syn.W
        # fields other than :W are not read through the two-argument form
        @test_throws ArgumentError plasticity_value(ev.syn, :ρ)
    end

    @testset "matrix reads the current weights" begin
        ev = _lazy_run!(_lazy_net(vSTDPParameter), 5000)
        W = plasticity_value(ev.syn, :W)
        M = matrix(ev.syn)
        SNNModels.ensure_transpose!(ev.syn)   # J is built only when read (vSTDP does not)
        @test all(M[ev.syn.I[s], ev.syn.J[s]] == W[s] for s in eachindex(W))
        @test maximum(abs.(ev.syn.W .- W)) > 0                  # while syn.W is stale
    end

    @testset "monitor!(:W) records values; a recorded run equals an unrecorded one" begin
        a, b = _lazy_net(vSTDPParameter), _lazy_net(vSTDPParameter)
        monitor!(b.syn, [:W], sr = 100Hz)
        # drive the clock-less loop by train!: use the model path for the recorder
        ma = compose(; a.pre, a.post, a.stim, a.syn, silent = true)
        mb = compose(; b.pre, b.post, b.stim, b.syn, silent = true)
        for n in 1:2000
            a.syn.v_post .= view(a.V, :, n + 1); train!(ma, 0.1; dt = 0.1)
            b.syn.v_post .= view(b.V, :, n + 1); train!(mb, 0.1; dt = 0.1)
        end
        @test plasticity_value(a.syn, :W) == plasticity_value(b.syn, :W)
        @test a.syn.W == b.syn.W                                 # bit-identical stored state
        rec = record(b.syn, :W)
        @test size(rec, 1) == length(b.syn.W) && eltype(rec) == Float64
        @test rec[:, end] == plasticity_value(b.syn, :W)
    end

    @testset "a deferred weight without accessor fails loudly" begin
        # a rule that declares :W lazy but defines no way to read it
        struct _BadRule <: SNNModels.LTPParameter
            active::Vector{Bool}
        end
        struct _BadVars <: SNNModels.LTPVariables
            active::Vector{Bool}
        end
        SNNModels.plasticityvariables(::_BadRule, Npre, Npost) = _BadVars([true])
        SNNModels.lazy_variables(::_BadVars) = (:W,)
        SNNModels.plasticity!(c, ::_BadRule, ::_BadVars, dt::AbstractFloat, T) = nothing
        pre, post = Identity(N = 3), Identity(N = 3)
        syn = SpikingSynapse(pre, post, :g; conn = fill(1.0, 3, 3), LTPParam = _BadRule([true]))
        @test_throws ArgumentError plasticity_value(syn, :W)
        @test_throws ArgumentError trace_values(syn, :W)
        @test_throws ArgumentError sync_traces!(syn)
        @test_throws ArgumentError monitor!(syn, [:W])           # at monitor! time, not in the middle of a run
        @test_throws ArgumentError matrix(syn)
        # and a lazily stored field of the variables without accessor, as before
        SNNModels.lazy_variables(::_BadVars) = (:active,)
        @test_throws ArgumentError plasticity_value(syn, :LTPVars, :active, 0.0)
    end

    @testset "library readers and writers materialise first" begin
        # changing the rule keeps the pending update
        ev = _lazy_run!(_lazy_net(vSTDPParameter), 5000)
        We = copy(plasticity_value(ev.syn, :W))
        syn2 = with_plasticity(ev.syn; LTP = STDPGerstner(FT = Float64))
        @test syn2.W == We
        # update_weights! sets a value: no pending potentiation is added on top of it
        ev = _lazy_run!(_lazy_net(vSTDPParameter), 5000)
        j = 1; s = ev.syn.colptr[j]; i = ev.syn.I[s]
        update_weights!(ev.syn, j, i, 0.25)
        @test plasticity_value(ev.syn, :W)[s] == 0.25
        # update_sparse_matrix! (rewiring) acts on the current weights
        ev = _lazy_run!(_lazy_net(vSTDPParameter), 5000)
        We = copy(plasticity_value(ev.syn, :W))
        SNNModels.update_sparse_matrix!(ev.syn)
        @test sort(ev.syn.W) == sort(We)
        # normalisation rescales the current weights: the summed input is restored
        ev = _lazy_net(vSTDPParameter)
        norm = SynapseNormalization([ev.syn]; param = MultiplicativeNorm(τ = 10ms))
        _lazy_run!(ev, 5000)
        SNNModels.plasticity!(norm, norm.param)
        Wn = plasticity_value(ev.syn, :W)
        for i in 1:length(ev.syn.fireI)
            ss = [s for s in eachindex(Wn) if ev.syn.I[s] == i]
            @test sum(Wn[ss]) ≈ sum(norm.W0[i]) rtol = 1e-10
        end
        # saving: the saved weights are the current ones
        ev = _lazy_run!(_lazy_net(vSTDPParameter), 5000)
        We = copy(plasticity_value(ev.syn, :W))
        model = compose(; ev.pre, ev.post, ev.stim, ev.syn, silent = true)
        dir = mktempdir()
        file = SNNModels.SNNsave(model; path = dir, name = "lazy", info = (; a = 1), type = :model)
        @test ev.syn.W == We
        saved = SNNModels.JLD2.load(file)["model"]
        @test saved.syn.syn.W == We
        # a copy of a model continues the run exactly (the deferred state is copied, not materialised)
        a = _lazy_run!(_lazy_net(vSTDPParameter), 2000)
        b = (; a..., syn = SNNModels.modelcopy(a.syn), pre = SNNModels.modelcopy(a.pre), post = SNNModels.modelcopy(a.post))
        @test plasticity_value(b.syn, :W) == plasticity_value(a.syn, :W)
        @test b.syn.W == a.syn.W
    end

    @testset "Float32 default" begin
        ev = _lazy_run!(_lazy_net(vSTDPParameter; FT = Float32), 5000)
        We = copy(plasticity_value(ev.syn, :W))
        @test eltype(We) == Float32
        sync_traces!(ev.syn)
        @test ev.syn.W == We
    end
end
true
