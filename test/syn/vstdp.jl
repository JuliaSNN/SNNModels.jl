using SNNModels
using Test
using Random
@load_units

# Event-driven voltage STDP (`vSTDPParameter`) against the dense rule (`vSTDPDenseParameter`) on identical,
# deterministic inputs (spikes on given steps, the postsynaptic potential given step by step), and against an
# independent Float64 per-step reference. The dense rule accumulates 1e5 Float32 additions, so the two differ
# by Float32 rounding of the dense rule: the comparison with the Float64 reference shows which one is closer.

const _VSE = (A_LTD = 8.0f-4, A_LTP = 5.0f-4, θ_LTD = -70.0f0, θ_LTP = -49.0f0, τu = 20.0f0, τv = 7.0f0, τx = 15.0f0,
              Wmin = 0.0f0, Wmax = 1.0f0)

# inputs of the validation protocol (stdp_continuous): 5 Hz presynaptic spikes on integer steps, a sinusoidal
# potential with a spike waveform (+90 mV for 1 ms, then -8 mV decaying with 15 ms) at the postsynaptic spikes
function _vse_inputs(; npre = 20, npost = 10, nsteps = 60000, dt = 0.1, seed = 11, rate = 5.0, silent = 0:-1, until = 0)
    rng = MersenneTwister(seed)
    spikes = [Int[] for _ in 1:nsteps]                       # presynaptic neurons that fire at step n
    for j in 1:npre, n in 1:nsteps-1
        rand(rng) < rate * dt * 1.0e-3 || continue
        (j in silent && n <= until) && continue
        push!(spikes[n], j)
    end
    V = zeros(Float32, npost, nsteps)
    for i in 1:npost
        P = 200 + 37i
        post = [n for n in 1:nsteps if rand(rng) < 8.0 * dt * 1.0e-3]
        for n in 1:nsteps
            t = (n - 1) * dt
            v = -70 + 4sin(2π * t / P)
            for m in post
                m <= n || continue
                τ = t - (m - 1) * dt
                τ > 150 && continue
                v += τ < 1 ? 90.0 : -8exp(-(τ - 1) / 15)
            end
            V[i, n] = v
        end
    end
    return spikes, V
end

# manual stepping through train!(P, C, S, dt, T): `dts(n)` is the step of call n; weights read by `readW`
function _vse_manual(rule, spikes, V; npre = 20, w0 = 0.5f0, dts = n -> 0.1f0, every = 0, w0vec = nothing)
    nsteps, npost = length(spikes), size(V, 1)
    pre, post = IF(N = npre), IF(N = npost)               # neither is integrated (not in the population list)
    syn = SpikingSynapse(pre, post, :ge; conn = fill(w0, npost, npre), LTPParam = rule)
    w0vec === nothing || (syn.W .= w0vec)
    syn.g = zeros(Float32, npost)
    post.v .= view(V, :, 1)
    syn.LTPVars.u .= view(V, :, 1); syn.LTPVars.v .= view(V, :, 1); syn.LTPVars.initialized[1] = true
    T = Time()
    traj = Vector{Float32}[]
    for n in 1:nsteps-1
        post.v .= view(V, :, n + 1)
        fill!(pre.fire, false); pre.fire[spikes[n]] .= true
        train!(SNNModels.AbstractPopulation[], [syn], SNNModels.AbstractStimulus[], dts(n), T)
        every > 0 && n % every == 0 && push!(traj, copy(plasticity_value(syn, :W)))
    end
    return syn, traj
end

# independent Float64 per-step reference of the rule (order: x, u and v, LTD at the spikes, LTP), constant dt
function _vse_ref64(spikes, V, P; npre = 20, w0 = 0.5, dt = 0.1)
    npost = size(V, 1)
    W = fill(Float64(w0), npost, npre)
    x = zeros(npre); u = Float64.(V[:, 1]); v = copy(u)
    for n in 1:length(spikes)-1
        Vn = Float64.(@view V[:, n + 1])
        @. x -= dt * x / P.τx
        for j in spikes[n]; x[j] += 1 / P.τx; end
        @. u += dt * (-u + Vn) / P.τu
        @. v += dt * (-v + Vn) / P.τv
        for j in spikes[n], i in 1:npost
            W[i, j] = max(P.Wmin, W[i, j] - P.A_LTD * max(u[i] - P.θ_LTD, 0))
        end
        for i in 1:npost
            g = max(v[i] - P.θ_LTD, 0) * max(Vn[i] - P.θ_LTP, 0)
            g == 0 && continue
            for j in 1:npre
                W[i, j] = min(P.Wmax, W[i, j] + dt * P.A_LTP * x[j] * g)
            end
        end
    end
    return W
end

_vse_dense(; kw...) = vSTDPDenseParameter(; merge(_VSE, NamedTuple(kw))...)
_vse_event(; kw...) = vSTDPParameter(; merge(_VSE, NamedTuple(kw))...)
# weights as a (post, pre) matrix
_vse_mat(syn, W = plasticity_value(syn, :W)) = (SNNModels.ensure_transpose!(syn); M = zeros(Float64, length(syn.fireI), length(syn.fireJ)); for s in eachindex(W); M[syn.I[s], syn.J[s]] = W[s]; end; M)

@testset "vSTDPParameter" begin
    spikes, V = _vse_inputs()
    @test sum(length, spikes) > 300
    @test count(>(-49.0f0), V) > 200                    # the potential crosses θ_LTP

    @testset "event-driven equals dense (Float32 rounding of the dense rule)" begin
        sd, td = _vse_manual(_vse_dense(), spikes, V; every = 100)
        se, te = _vse_manual(_vse_event(), spikes, V; every = 100)
        Wd, We = copy(sd.W), copy(plasticity_value(se, :W))
        change = maximum(abs.(Wd .- 0.5f0))
        @test change > 0.1                              # the weights moved
        @test maximum(abs.(We .- Wd)) < 2.0e-5 * change
        @test maximum(maximum(abs.(a .- b)) for (a, b) in zip(td, te)) < 2.0e-5 * change
        @test all(0.0f0 .<= We .<= 1.0f0)
        @test all(all(0.0f0 .<= w .<= 1.0f0) for w in te)
        # the stored weights are not the current ones until synchronised
        @test maximum(abs.(se.W .- We)) > 0
        sync_traces!(se)
        @test se.W == We
        @test plasticity_value(se, :W) == We                 # synchronising again changes nothing
        # independent Float64 reference: the event-driven weights are at least as close as the dense ones
        R = _vse_ref64(spikes, V, _VSE)
        ed, ee = maximum(abs.(_vse_mat(sd, Wd) .- R)), maximum(abs.(_vse_mat(se, We) .- R))
        @test ee < 2.0e-6
        @test ee <= ed + 1.0e-7
        @info "vSTDP event vs dense" max_dW = maximum(abs.(We .- Wd)) trajectory = maximum(maximum(abs.(a .- b)) for (a, b) in zip(td, te)) change event_vs_f64 = ee dense_vs_f64 = ed
    end

    @testset "epoch length: tiny, default, large" begin
        sd, _ = _vse_manual(_vse_dense(), spikes, V)
        change = maximum(abs.(sd.W .- 0.5f0))
        for epoch in (0.01f0, 0.2f0, 10.0f0, 25.0f0)
            se, _ = _vse_manual(_vse_event(epoch = epoch), spikes, V)
            @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 2.0e-5 * change
        end
    end

    @testset "silent synapses for longer than one epoch" begin
        sp, Vs = _vse_inputs(silent = 1:6, until = 40000)   # 4 s without presynaptic spikes = 2.7 default epochs
        @test all(isempty(filter(<=(6), s)) for s in sp[1:40000])
        sd, _ = _vse_manual(_vse_dense(), sp, Vs)
        for epoch in (0.5f0, 10.0f0)
            se, _ = _vse_manual(_vse_event(epoch = epoch), sp, Vs)
            @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 2.0e-5 * maximum(abs.(sd.W .- 0.5f0))
            # weights of the silent neurons are untouched (no spike, no LTP), then change with their first spikes
        end
        se, _ = _vse_manual(_vse_event(), sp[1:39000], Vs[:, 1:39000])
        SNNModels.ensure_transpose!(se)
        @test all(plasticity_value(se, :W)[se.J .<= 6] .== 0.5f0)
    end

    @testset "weight bounds: Wmax and Wmin" begin
        sd, _ = _vse_manual(_vse_dense(A_LTP = 4.0f-3), spikes, V)
        se, _ = _vse_manual(_vse_event(A_LTP = 4.0f-3), spikes, V)
        @test count(==(1.0f0), sd.W) > 20                # many weights sit at Wmax
        @test count(==(1.0f0), plasticity_value(se, :W)) > 20
        @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 1.0e-4
        # strong LTD pushes weights to Wmin at the spikes, LTP then adds from there
        sd, _ = _vse_manual(_vse_dense(A_LTD = 0.05f0, A_LTP = 1.0f-3), spikes, V)
        se, _ = _vse_manual(_vse_event(A_LTD = 0.05f0, A_LTP = 1.0f-3), spikes, V)
        @test count(==(0.0f0), sd.W) > 5
        @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 1.0e-4
        @test all(0.0f0 .<= plasticity_value(se, :W) .<= 1.0f0)
        # an initial weight above Wmax is clamped as the dense rule does at its first step
        w0 = fill(1.5f0, 200)
        sd, _ = _vse_manual(_vse_dense(), spikes[1:300], V[:, 1:300], w0vec = w0)
        se, _ = _vse_manual(_vse_event(), spikes[1:300], V[:, 1:300], w0vec = w0)
        @test maximum(sd.W) <= 1.0f0 && maximum(plasticity_value(se, :W)) <= 1.0f0
    end

    @testset "dt changed in the middle of the run" begin
        dts = n -> n <= 25000 ? 0.1f0 : 0.05f0
        sd, _ = _vse_manual(_vse_dense(), spikes, V; dts)
        se, _ = _vse_manual(_vse_event(), spikes, V; dts)
        @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 2.0e-5 * maximum(abs.(sd.W .- 0.5f0))
        # a parameter that scales the accumulation (A_LTP) changed in the middle of the run
        pre, post = IF(N = 20), IF(N = 10)
        function mk(T)
            syn = SpikingSynapse(pre, post, :ge; conn = fill(0.5f0, 10, 20), LTPParam = T(; _VSE...))
            syn.g = zeros(Float32, 10); syn.LTPVars.initialized[1] = true
            syn.LTPVars.u .= V[:, 1]; syn.LTPVars.v .= V[:, 1]
            return syn
        end
        function run2!(syn, T)
            t = Time()
            for n in 1:40000
                post.v .= view(V, :, n + 1); fill!(pre.fire, false); pre.fire[spikes[n]] .= true
                n == 20001 && (syn.LTPParam = T(; merge(_VSE, (A_LTP = 1.0f-3,))...))
                train!(SNNModels.AbstractPopulation[], [syn], SNNModels.AbstractStimulus[], 0.1f0, t)
            end
            return syn
        end
        a, b = run2!(mk(vSTDPDenseParameter), vSTDPDenseParameter), run2!(mk(vSTDPParameter), vSTDPParameter)
        @test maximum(abs.(plasticity_value(b, :W) .- a.W)) < 2.0e-5 * maximum(abs.(a.W .- 0.5f0))
    end

    @testset "validation protocol inputs of stdp_continuous (if present)" begin
        vd = joinpath(@__DIR__, "..", "..", "..", "..", "papers", "JuliaSNN_publication", "validation")
        fs = joinpath(vd, "stdp", "data", "spikes_pre.txt")
        fv = joinpath(vd, "stdp_continuous", "data", "vpost_f32.bin")
        if isfile(fs) && isfile(fv)
            nsteps = 200000
            raw = [parse.(Int, split(l)) for l in eachline(fs) if !isempty(l)]
            sp = [Int[] for _ in 1:nsteps]
            for (k, j) in raw; push!(sp[k], j + 1); end
            Vp = Matrix{Float32}(undef, 10, nsteps); read!(fv, Vp)
            kw = (A_LTD = 8.0f-4, A_LTP = 5.0f-4)
            sd, td = _vse_manual(_vse_dense(; kw...), sp, Vp; every = 1000)
            se, te = _vse_manual(_vse_event(; kw...), sp, Vp; every = 1000)
            change = maximum(abs.(sd.W .- 0.5f0))
            @test maximum(abs.(plasticity_value(se, :W) .- sd.W)) < 2.0e-5 * change
            @test maximum(maximum(abs.(a .- b)) for (a, b) in zip(td, te)) < 2.0e-5 * change
            @info "protocol (200 synapses, 20 s)" final = maximum(abs.(plasticity_value(se, :W) .- sd.W)) change
        else
            @test_skip false
        end
    end

    @testset "parameter checks" begin
        @test_throws ArgumentError plasticityvariables(vSTDPParameter(A_LTP = -1.0f-3), 5, 4)
        @test_throws ArgumentError plasticityvariables(vSTDPParameter(epoch = 0.0f0), 5, 4)
        @test_throws ArgumentError plasticityvariables(vSTDPParameter(epoch = 30.0f0), 5, 4)
        @test plasticityvariables(vSTDPParameter(), 5, 4) isa vSTDPVariables
        pre, post = IF(N = 3), IF(N = 2)
        syn = SpikingSynapse(pre, post, :ge; conn = fill(0.5f0, 2, 3), LTPParam = vSTDPParameter(τx = 0.05f0))
        syn.g = zeros(Float32, 2)
        @test_throws ArgumentError train!(SNNModels.AbstractPopulation[], [syn], SNNModels.AbstractStimulus[], 0.1f0, Time())
    end
end

# ---- records and monitoring: a Poisson pre population, the potential of the target overwritten at every step ----

function _vse_model(rule; monitor = nothing, nsteps = 5900, seed = 5)
    Random.seed!(seed)
    pre = SNNModels.Poisson(N = 12, param = SNNModels.PoissonParameter(40Hz))
    post = SNNModels.IF(N = 6)
    syn = SpikingSynapse(pre, post, :ge; conn = fill(0.5f0, 6, 12), LTPParam = rule)
    syn.g = zeros(Float32, 6)
    syn.LTPVars.initialized[1] = true
    isnothing(monitor) || monitor(syn)
    k = Ref(0)
    step!(; kw...) = (k[] += 1; n = k[]; for i in 1:6
            post.v[i] = Float32(-70 + 40 * (sin(0.003n + i) > 0.3) + 6sin(0.01n * i))
        end)
    Random.seed!(seed + 1)
    train!([pre], [syn], SNNModels.AbstractStimulus[]; duration = nsteps * 0.1f0, dt = 0.1f0, perturbation! = step!)
    return syn
end

@testset "vSTDPParameter: records and monitors" begin
    rule(T; kw...) = T(; A_LTD = 8.0f-4, A_LTP = 2.0f-3, θ_LTP = -49.0f0, Wmin = 0.0f0, Wmax = 1.0f0, kw...)
    mon = s -> (monitor!(s, [:W], sr = 1kHz); monitor!(s, [:u, :v, :x], :LTPVars, sr = 1kHz))
    se = _vse_model(rule(vSTDPParameter))
    sm = _vse_model(rule(vSTDPParameter), monitor = mon)
    sd = _vse_model(rule(vSTDPDenseParameter), monitor = mon)
    @test maximum(abs.(plasticity_value(se, :W) .- 0.5f0)) > 0.05
    @testset "monitored equals unmonitored, bit for bit" begin
        @test sm.W == se.W
        @test plasticity_value(sm, :W) == plasticity_value(se, :W)
        @test sm.LTPVars.K == se.LTPVars.K && sm.LTPVars.snap == se.LTPVars.snap && sm.LTPVars.x == se.LTPVars.x
        @test sm.LTPVars.u == se.LTPVars.u && sm.LTPVars.v == se.LTPVars.v
    end
    @testset "records are the current values" begin
        Wr = SNNModels.getvariable(sm, :W)
        Wd = SNNModels.getvariable(sd, :W)
        @test size(Wr) == size(Wd)
        @test size(Wr, 2) > 10
        @test maximum(abs.(Wr .- Wd)) < 1.0e-4
        @test Wr[:, end] == plasticity_value(sm, :W)
        @test maximum(abs.(Wr[:, end] .- sm.W)) > 0           # the stored weights are not what is recorded
        xr = SNNModels.getvariable(sm, :LTPVars_x)
        xd = SNNModels.getvariable(sd, :LTPVars_x)
        @test size(xr) == size(xd) && maximum(xr) > 0.01
        @test xr ≈ xd rtol = 1.0e-4 atol = 1.0e-7
        @test xr[:, end] ≈ plasticity_value(sm, :LTPVars, :x, 0.0f0) rtol = 1.0e-5
        @test SNNModels.getvariable(sm, :LTPVars_u) ≈ SNNModels.getvariable(sd, :LTPVars_u) rtol = 1.0e-5
    end
    @testset "sync_traces! materialises the weights" begin
        s2 = _vse_model(rule(vSTDPParameter))
        Wc = copy(plasticity_value(s2, :W))
        sync_traces!(s2)
        @test s2.W == Wc
        @test s2.LTPVars.steps[1] == 0 && all(==(0.0), s2.LTPVars.K) && all(==(0.0), s2.LTPVars.snap)
    end
    @testset "no allocation per step" begin
        pre, post = IF(N = 40), IF(N = 10)
        syn = SpikingSynapse(pre, post, :ge; conn = fill(0.5f0, 10, 40), LTPParam = rule(vSTDPParameter; epoch = 0.2f0))
        syn.g = zeros(Float32, 10); post.v .= -50f0
        T = Time()
        function steps!(n)
            for k in 1:n
                post.v .= -60f0 + 20f0 * (k % 50 < 5)
                fill!(pre.fire, false); pre.fire[(k % 40) + 1] = true
                train!(SNNModels.AbstractPopulation[], [syn], SNNModels.AbstractStimulus[], 0.1f0, T)
            end
        end
        steps!(2000)                                          # compile, first-use allocation of the snapshots
        c = syn.LTPVars
        @test @allocated(plasticity!(syn, syn.LTPParam, c, 0.1f0, T)) == 0
        @test @allocated(update_traces!(syn, syn.LTPParam, c, 0.1f0, T)) == 0
        plasticity_value(syn, :W)
        @test @allocated(plasticity_value(syn, :W)) == 0
    end
end
