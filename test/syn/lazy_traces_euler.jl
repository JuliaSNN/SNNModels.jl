using SNNModels
using Test
using Random
@load_units

# Euler-integrated trace rules stored lazily: iSTDPRate, STDPMexicanHat, STDPSymmetric, STDPAntiSymmetric.
# Their traces follow x <- x (1 - dt/τ) (forward Euler, not exp(-dt/τ)); the decay is applied on one scalar per
# trace type (`scale`), with the Euler factor as the per-step factor, so the recursion is the same as the dense
# one. Checked here: (1) the recorded traces against an independent dense Float64 recursion, (2) the weights against
# the dense Float32 implementations of SNNModels 1.9.1 (reference kernels below, run beside the rule), (3) monitored
# and unmonitored runs give bit-identical weights, (4) `sync_traces!`/`trace_values`.

const _KE = SNNModels
const _DT_EU = 0.125f0
const _NPRE_EU, _NPOST_EU = 12, 6

# ---- independent dense references -------------------------------------------------------------------------

# Euler trace after each step. `order = :decay_then_add`: x <- x f + s (Mexican hat, structured, iSTDPRate);
# the value after step n includes the spike of step n.
function _eu_trace(spikes::AbstractMatrix{Bool}, τ, dt)
    N, nsteps = size(spikes)
    f = 1 - Float64(dt) / Float64(τ)
    x = zeros(N)
    out = zeros(N, nsteps)
    for n in 1:nsteps
        x .= x .* f .+ spikes[:, n]
        out[:, n] = x
    end
    return out
end

# dense Float32 kernels of the previous implementation (every neuron decayed at every step)
function _eu_ref_istdprate!(W, tpre, tpost, c, p, dt)
    (; colptr, I, J, fireI, fireJ) = c
    for j in eachindex(fireJ)
        tpre[j] += dt * (-tpre[j]) / p.τy
        if fireJ[j]
            tpre[j] += 1
            for s = colptr[j]:(colptr[j+1]-1)
                W[s] = clamp(W[s] + p.η * (tpost[I[s]] - 2 * p.r * p.τy), p.Wmin, p.Wmax)
            end
        end
    end
    for i in eachindex(fireI)
        tpost[i] += dt * (-tpost[i]) / p.τy
        if fireI[i]
            tpost[i] += 1
            for s in eachindex(I)
                I[s] == i && (W[s] = clamp(W[s] + p.η * tpre[J[s]], p.Wmin, p.Wmax))
            end
        end
    end
end

function _eu_ref_mexican!(W, tpre, tpost, c, p, dt)
    (; colptr, I, J, fireI, fireJ) = c
    mh(x) = (r = (1 - x) * exp(-x / sqrt(2)); isnan(r) ? 0.0f0 : r)
    tpost .+= dt .* (-tpost) ./ p.τ
    tpost[findall(fireI)] .+= 1
    tpre .+= dt .* (-tpre) ./ p.τ
    tpre[findall(fireJ)] .+= 1
    for j in findall(fireJ), s = colptr[j]:(colptr[j+1]-1)
        abs(tpost[I[s]] * tpre[j]) > 0 && (W[s] += p.A * mh(log(tpre[j] / tpost[I[s]])^2))
    end
    for s in eachindex(I)
        fireI[I[s]] && abs(tpost[I[s]] * tpre[J[s]]) > 0 && (W[s] += p.A * mh(log(tpre[J[s]] / tpost[I[s]])^2))
    end
    W .= clamp.(W, p.Wmin, p.Wmax)
end

function _eu_ref_structured!(W, st, c, p, dt)
    (; colptr, I, J, fireI, fireJ) = c
    sym = p isa STDPSymmetric
    for j in findall(fireJ), s = colptr[j]:(colptr[j+1]-1)
        i = I[s]
        W[s] += sym ? p.αpre + (p.A_x / 2p.τ_x * st.to_x[i] - p.A_y / 2p.τ_y * st.to_y[i]) :
                      p.αpre - p.A_y / p.τ_y * st.to_y[i]
    end
    for s in eachindex(I)
        fireI[I[s]] || continue
        j = J[s]
        W[s] += sym ? p.αpost + (p.A_x / 2p.τ_x * st.tr_x[j] - p.A_y / 2p.τ_y * st.tr_y[j]) :
                      p.αpost + p.A_x / p.τ_x * st.tr_x[j]
    end
    st.to_y .+= dt .* (-st.to_y) ./ p.τ_y
    st.to_y[findall(fireI)] .+= 1
    st.tr_x .+= dt .* (-st.tr_x) ./ p.τ_x
    st.tr_x[findall(fireJ)] .+= 1
    if sym
        st.to_x .+= dt .* (-st.to_x) ./ p.τ_x
        st.to_x[findall(fireI)] .+= 1
        st.tr_y .+= dt .* (-st.tr_y) ./ p.τ_y
        st.tr_y[findall(fireJ)] .+= 1
    end
    for s in eachindex(W)
        W[s] = clamp(W[s], p.Wmin, p.Wmax)
    end
end

# ---- a driven pre (Poisson) -> post (IF) connection, spikes of both populations logged ----

function _eu_run(; ltp, nsteps = 600, seed = 4, monitor = nothing, μ = 40.0, pre_rate = 80Hz)
    Random.seed!(seed)
    pre = SNNModels.Poisson(N = _NPRE_EU, param = SNNModels.PoissonParameter(pre_rate))
    post = SNNModels.IF(N = _NPOST_EU)
    Random.seed!(seed + 1)
    syn = SpikingSynapse(pre, post, :ge; conn = (p = 0.6, μ = μ), LTPParam = ltp)
    isnothing(monitor) || monitor(syn)
    pf, qf = falses(_NPRE_EU, nsteps), falses(_NPOST_EU, nsteps)
    k = Ref(0)
    spy(; kw...) = (k[] > 0 && (pf[:, k[]] .= pre.fire; qf[:, k[]] .= post.fire); k[] += 1)
    Random.seed!(seed + 2)
    train!([pre, post], [syn], SNNModels.AbstractStimulus[]; duration = nsteps * _DT_EU, dt = _DT_EU, perturbation! = spy)
    pf[:, nsteps] .= pre.fire; qf[:, nsteps] .= post.fire
    return syn, pre, post, pf, qf
end

_eu_samples(syn, key) = Float64.(SNNModels.getvariable(syn, Symbol(:LTPVars_, key)))[:, 2:end]

# name => (constructor, traces (key, side, τ), weight bounds)
const _EU_RULES = (
    "iSTDPRate" => (() -> iSTDPRate(η = 0.02f0, r = 5Hz, τy = 20ms, Wmin = 0.0f0, Wmax = 80.0f0),
                    ((:tpre, :pre, 20.0), (:tpost, :post, 20.0))),
    "STDPMexicanHat" => (() -> STDPMexicanHat(A = 0.05f0, τ = 17ms, Wmin = 0.0f0, Wmax = 80.0f0),
                         ((:tpre, :pre, 17.0), (:tpost, :post, 17.0))),
    "STDPSymmetric" => (() -> STDPSymmetric(A_x = 0.5f0, A_y = 0.4f0, τ_x = 15ms, τ_y = 60ms, αpre = 0.001f0, αpost = -0.001f0,
                                            Wmin = 0.0f0, Wmax = 80.0f0),
                        ((:tr_x, :pre, 15.0), (:tr_y, :pre, 60.0), (:to_x, :post, 15.0), (:to_y, :post, 60.0))),
    "STDPAntiSymmetric" => (() -> STDPAntiSymmetric(A_x = 0.5f0, A_y = 0.45f0, τ_x = 15ms, τ_y = 40ms, αpre = 0.001f0,
                                                    Wmin = 0.0f0, Wmax = 80.0f0),
                            ((:tr_x, :pre, 15.0), (:to_y, :post, 40.0))),
)

@testset "Euler traces stored lazily" begin

    for (name, (mk, traces)) in _EU_RULES
        @testset "$name: monitor! records the trace values" begin
            keys = collect(first.(traces))
            syn, pre, post, pf, qf = _eu_run(ltp = mk(), monitor = s -> monitor!(s, keys, :LTPVars, sr = 10kHz))
            @test sum(pf) > 50 && sum(qf) > 20
            @test any(!=(1.0), syn.LTPVars.scale)
            for (key, side, τ) in traces
                ref = _eu_trace(side === :pre ? pf : qf, τ, _DT_EU)
                rec = _eu_samples(syn, key)
                @test size(rec) == size(ref)
                @test maximum(rec) > 0.1
                @test rec ≈ ref rtol = 1.0e-5 atol = 1.0e-12
                @test !(Float64.(getfield(syn.LTPVars, key)) ≈ ref[:, end])    # the stored values are not the values
            end
            @test syn.W != _eu_run(ltp = mk(), nsteps = 1)[1].W                 # plasticity acted
        end

        @testset "$name: bit-identical weights, monitored or not" begin
            keys = collect(first.(traces))
            a = _eu_run(ltp = mk())[1]
            b = _eu_run(ltp = mk(), monitor = s -> monitor!(s, keys, :LTPVars, sr = 10kHz))[1]
            @test a.W == b.W
            for k in propertynames(a.LTPVars)
                @test getfield(a.LTPVars, k) == getfield(b.LTPVars, k)
            end
        end

        @testset "$name: sync_traces! gives the values of trace_values" begin
            syn = _eu_run(ltp = mk())[1]
            v = syn.LTPVars
            keys = first.(traces)
            stored, scale = [copy(getfield(v, k)) for k in keys], copy(v.scale)
            vals = [trace_values(syn, k) for k in keys]
            @test [copy(getfield(v, k)) for k in keys] == stored && v.scale == scale
            @test any(!=(1.0), v.scale)
            @test sync_traces!(syn) === syn
            @test all(==(1.0), v.scale)
            @test [getfield(v, k) for k in keys] == vals
            @test all(k -> plasticity_value(syn, :LTPVars, k, 0.0f0) == getfield(v, k), keys)
        end
    end

    @testset "weights agree with the dense Float32 Euler kernels of 1.9.1" begin
        Npre, Npost = 50, 40
        for (name, (mk, _)) in _EU_RULES
            @testset "$name" begin
                Random.seed!(7)
                p = mk()
                pre, post = Identity(N = Npre), Identity(N = Npost)
                syn = SpikingSynapse(pre, post, :g; conn = (p = 0.3f0, μ = 5.0f0, σ = 1.0f0), LTPParam = p)
                Wd = copy(syn.W)
                clamp!(Wd, p.Wmin, p.Wmax)
                tpre, tpost = zeros(Float32, Npre), zeros(Float32, Npost)
                st = (to_x = zeros(Float32, Npost), to_y = zeros(Float32, Npost),
                      tr_x = zeros(Float32, Npre), tr_y = zeros(Float32, Npre))
                T = Time()
                Random.seed!(13)
                for n in 1:1500
                    pre.fire .= rand(Npre) .< 0.05; post.fire .= rand(Npost) .< 0.05
                    update_time!(T, _DT_EU)
                    plasticity!(syn, syn.param, _DT_EU, T)
                    if p isa iSTDPRate
                        _eu_ref_istdprate!(Wd, tpre, tpost, syn, p, _DT_EU)
                    elseif p isa STDPMexicanHat
                        _eu_ref_mexican!(Wd, tpre, tpost, syn, p, _DT_EU)
                    else
                        _eu_ref_structured!(Wd, st, syn, p, _DT_EU)
                    end
                end
                d = maximum(abs.(syn.W .- Wd))
                @debug "$name: max |dW| against the dense Float32 kernel" d maximum(abs.(Wd .- 5.0f0))
                @test maximum(abs.(Wd .- 5.0f0)) > 0.05            # plasticity acted
                @test d < 2.0f-6 * maximum(abs.(Wd))               # a few Float32 ulps of the largest weight
                @test syn.W ≈ Wd rtol = 1.0f-5 atol = 1.0f-5
            end
        end
    end

    @testset "a change of dt or of a time constant keeps the decay of the steps already run (iSTDPRate)" begin
        Random.seed!(5)
        pre, post = Identity(N = 20), Identity(N = 15)
        rule = iSTDPRate(η = 0.1f0, r = 5Hz, τy = 20ms, Wmin = 0.0f0, Wmax = 20.0f0)
        syn = SpikingSynapse(pre, post, :g; conn = (p = 0.5f0, μ = 5.0f0), LTPParam = rule)
        dts = n -> n <= 300 ? 0.125f0 : n <= 450 ? 0.25f0 : 0.0625f0
        T = Time()
        xpre, xpost = zeros(20), zeros(15)
        Random.seed!(9)
        for n in 1:700
            pf, qf = rand(20) .< 0.05, rand(15) .< 0.05
            pre.fire .= pf; post.fire .= qf
            dt = dts(n)
            update_time!(T, dt)
            plasticity!(syn, syn.param, dt, T)
            f = 1 - Float64(dt) / 20.0
            xpre .= xpre .* f .+ pf
            xpost .= xpost .* f .+ qf
        end
        @test Float64.(trace_values(syn, :tpre)) ≈ xpre rtol = 1.0e-5 atol = 1.0e-12
        @test Float64.(trace_values(syn, :tpost)) ≈ xpost rtol = 1.0e-5 atol = 1.0e-12
    end

    @testset "iSTDPPotential keeps its dense traces" begin
        syn = _eu_run(ltp = iSTDPPotential(η = 0.05f0, v0 = -55mV, τy = 20ms, Wmin = 0.0f0, Wmax = 80.0f0), nsteps = 200,
                      monitor = s -> monitor!(s, [:tpre, :tpost], :LTPVars, sr = 10kHz))[1]
        @test all(==(1.0), syn.LTPVars.scale)
        @test trace_values(syn, :tpre) === syn.LTPVars.tpre
        @test plasticity_value(syn, :LTPVars, :tpost, 0.0f0) === syn.LTPVars.tpost
    end
end
