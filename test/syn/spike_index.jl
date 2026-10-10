using SNNModels
using Test
using Random
@load_units

# Spike-index lists: forward! and the event-driven plasticity kernels walk the list of the
# neurons that fired (one scan of the Bool fire vector per step) instead of scanning the
# whole vector in every kernel. The list path must give the same result as the dense scans.

const _K = SNNModels
const _DT_SI = 0.125f0

# no spike, all spikes, and two random firing patterns
_patterns(N, rate) = (falses(N), trues(N), rand(N) .< rate, rand(N) .< rate)

# dense reference of one STDPGerstner step (the kernels with the Bool fire vectors)
function _dense_gerstner!(r, p, v, t, dt)
    _K._initial_clamp!(r.W, v.initialized, p.Wmin, p.Wmax)
    _K._pre_spike_pass!((w, i, j) -> p.A_post * v.tpost[i], r.W, r.colptr, r.I, r.fireJ, p.Wmin, p.Wmax)
    _K._post_spike_pass!((w, i, j) -> p.A_pre * v.tpre[j], r.W, r.rowptr, r.index, r.J, r.fireI, p.Wmin, p.Wmax)
    _K._spike_increment!(v.tpre, v.last_pre, r.fireJ, t)
    _K._spike_increment!(v.tpost, v.last_post, r.fireI, t)
    _K._rescale!(v.tpre, Float32(exp(-dt / p.τpre)))
    _K._rescale!(v.tpost, Float32(exp(-dt / p.τpost)))
end

# dense reference of forward!
function _dense_forward!(g, c)
    for j in eachindex(c.fireJ)
        c.fireJ[j] || continue
        for s in c.colptr[j]:(c.colptr[j+1]-1)
            g[c.I[s]] += c.W[s] * c.ρ[s]
        end
    end
end

@testset "spike index list" begin

    @testset "flags to list" begin
        Random.seed!(3)
        for N in (0, 1, 7, 8, 9, 64, 100, 1001), rate in (0.0, 0.02, 0.5, 1.0)
            fire = rand(N) .< rate
            idx = Vector{Int32}(undef, N)
            n = _K._flags_to_list!(idx, fire)
            @test idx[1:n] == findall(fire)
            # generic AbstractVector{Bool} method
            n2 = _K._flags_to_list!(idx, view(fire, 1:N))
            @test idx[1:n2] == findall(fire)
        end
    end

    @testset "SpikeIndex reuse" begin
        fire = [false, true, false, true]
        s = _K.SpikeIndex(4)
        # inactive: every call rescans, a modified fire vector is always seen
        @test _K.spike_list!(s, fire) == [2, 4]
        fire[1] = true
        @test _K.spike_list!(s, fire) == [1, 2, 4]
        # active: the list of the step is reused until released
        _K._arm!(s)
        @test _K.spike_list!(s, fire) == [1, 2, 4]
        fire[1] = false
        @test _K.spike_list!(s, fire) == [1, 2, 4]
        _K._disarm!(s)
        @test _K.spike_list!(s, fire) == [2, 4]
        # no spikes, all spikes
        @test isempty(_K.spike_list!(s, falses(4)))
        @test _K.spike_list!(s, trues(4)) == 1:4
    end

    @testset "forward! and plasticity! equal the dense scans" begin
        Npre, Npost = 61, 43
        rules = (STDPGerstner(A_pre = 0.05, A_post = -0.04, Wmax = 20.0f0, Wmin = 0.0f0),
                 STDPTriplet(A2_plus = 0.01, A3_plus = 0.02, A2_minus = 0.01, A3_minus = 0.005,
                             Wmax = 20.0f0, Wmin = 0.0f0),
                 STDPWeightDependent(Wmax = 20.0f0, Wmin = 0.0f0))
        for rule in rules
            Random.seed!(5)
            pre, post = Identity(N = Npre), Identity(N = Npost)
            a = SpikingSynapse(pre, post, :g; conn = (p = 0.3f0, μ = 5.0f0, σ = 1.0f0), LTPParam = rule)
            Random.seed!(5)
            pre2, post2 = Identity(N = Npre), Identity(N = Npost)
            b = SpikingSynapse(pre2, post2, :g; conn = (p = 0.3f0, μ = 5.0f0, σ = 1.0f0), LTPParam = rule)
            @test a.W == b.W
            Ta, Tb = Time(), Time()
            Random.seed!(11)
            # steps: empty, full, random; a runs with the list armed as in train!, b rescans at every call
            for n in 1:300
                if n == 1
                    fa, fb = falses(Npre), falses(Npost)
                elseif n == 2
                    fa, fb = trues(Npre), trues(Npost)
                else
                    fa, fb = rand(Npre) .< 0.05, rand(Npost) .< 0.05
                end
                pre.fire .= fa; post.fire .= fb; pre2.fire .= fa; post2.fire .= fb
                update_time!(Ta, _DT_SI); update_time!(Tb, _DT_SI)
                _K.prepare_spikes!(a)
                update_traces!(a, a.param, _DT_SI, Ta)
                _K.forward!(a, a.param, _DT_SI, Ta)
                plasticity!(a, a.param, _DT_SI, Ta)
                _K.release_spikes!(a)
                update_traces!(b, b.param, _DT_SI, Tb)
                _K.forward!(b, b.param, _DT_SI, Tb)
                plasticity!(b, b.param, _DT_SI, Tb)
            end
            @test a.W == b.W
            @test a.g == b.g
            @test maximum(abs.(a.W .- 5.0f0)) > 0.1   # plasticity acted
            @test a.pre_spikes.active == false && a.post_spikes.active == false
        end
    end

    @testset "list kernels against the dense STDPGerstner step" begin
        Npre, Npost = 50, 40
        p = STDPGerstner(A_pre = 0.05, A_post = -0.04, Wmax = 20.0f0, Wmin = 0.0f0)
        Random.seed!(7)
        pre, post = Identity(N = Npre), Identity(N = Npost)
        syn = SpikingSynapse(pre, post, :g; conn = (p = 0.3f0, μ = 5.0f0, σ = 1.0f0), LTPParam = p)
        r = (; syn.rowptr, syn.colptr, syn.I, syn.J, syn.index, W = copy(syn.W), syn.fireI, syn.fireJ)
        vref = plasticityvariables(p, Npre, Npost)
        T = Time()
        Random.seed!(13)
        for n in 1:400
            if n == 1
                pre.fire .= false; post.fire .= false
            elseif n == 2
                pre.fire .= true; post.fire .= true
            else
                pre.fire .= rand(Npre) .< 0.05; post.fire .= rand(Npost) .< 0.05
            end
            update_time!(T, _DT_SI)
            plasticity!(syn, syn.param, _DT_SI, T)
            _dense_gerstner!(r, p, vref, get_time(T), _DT_SI)
        end
        # the rule decays its traces lazily, the reference multiplies them at every step: equal to Float32 rounding
        @test syn.W ≈ r.W rtol = 1.0f-5
        @test trace_values(syn, :tpre) ≈ vref.tpre rtol = 1.0f-5 atol = 1.0f-12
        @test trace_values(syn, :tpost) ≈ vref.tpost rtol = 1.0f-5 atol = 1.0f-12
    end

    @testset "forward! equals the dense loop, with and without delay" begin
        Random.seed!(17)
        pre, post = Identity(N = 30), Identity(N = 20)
        c = SpikingSynapse(pre, post, :g; conn = (p = 0.4f0, μ = 2.0f0, σ = 0.5f0))
        T = Time()
        for pat in _patterns(30, 0.1)
            pre.fire .= pat
            c.g .= 0
            g = zeros(Float32, 20)
            _K.forward!(c, c.param, _DT_SI, T)
            _dense_forward!(g, c)
            @test c.g == g
        end
        # delay path: the same increments arrive, queued
        Random.seed!(17)
        cd = SpikingSynapse(pre, post, :g; conn = (p = 0.4f0, μ = 2.0f0, σ = 0.5f0),
                            delay_dist = SNNModels.Distributions.Dirac(0.0))
        pre.fire .= rand(30) .< 0.3
        cd.g .= 0
        g = zeros(Float32, 20)
        update_time!(T, _DT_SI)
        _K.forward!(cd, cd.param, _DT_SI, T)
        _dense_forward!(g, cd)
        @test cd.g ≈ g
    end

    @testset "short-term plasticity update_traces! uses the list" begin
        for stp in (MarkramSTPParameter(), MarkramSTPParameter(Wmax = 10.0f0))
            Random.seed!(19)
            pre, post = Identity(N = 40), Identity(N = 25)
            a = SpikingSynapse(pre, post, :g; conn = (p = 0.3f0, μ = 2.0f0, σ = 0.5f0), STPParam = stp)
            Random.seed!(19)
            pre2, post2 = Identity(N = 40), Identity(N = 25)
            b = SpikingSynapse(pre2, post2, :g; conn = (p = 0.3f0, μ = 2.0f0, σ = 0.5f0), STPParam = stp)
            Ta, Tb = Time(), Time()
            Random.seed!(23)
            for n in 1:200
                f = n == 1 ? falses(40) : n == 2 ? trues(40) : rand(40) .< 0.05
                pre.fire .= f; pre2.fire .= f
                update_time!(Ta, _DT_SI); update_time!(Tb, _DT_SI)
                _K.prepare_spikes!(a)
                update_traces!(a, a.param, _DT_SI, Ta)
                _K.forward!(a, a.param, _DT_SI, Ta)
                plasticity!(a, a.param, _DT_SI, Ta)
                _K.release_spikes!(a)
                update_traces!(b, b.param, _DT_SI, Tb)
                _K.forward!(b, b.param, _DT_SI, Tb)
                plasticity!(b, b.param, _DT_SI, Tb)
            end
            @test a.ρ == b.ρ
            @test a.g == b.g
            @test any(a.ρ .!= 1.0f0)
        end
    end

    @testset "delay ring buffer" begin
        D = SNNModels.Distributions
        # delays are rounded up to whole steps: 0.8 ms at dt = 0.125 ms arrives 7 steps after the spike
        pre, post = Identity(N = 1), Identity(N = 1)
        c = SpikingSynapse(pre, post, :g; conn = fill(2.0f0, 1, 1), delay_dist = D.Dirac(0.8))
        T = Time()
        arrivals = Int[]
        for n in 1:20
            pre.fire[1] = n == 3
            update_time!(T, _DT_SI)
            _K.forward!(c, c.param, _DT_SI, T)
            c.g[1] != 0 && (push!(arrivals, n); @test c.g[1] == 2.0f0; c.g[1] = 0)
        end
        @test arrivals == [3 + 7]
        @test c.param.delaysteps == Int32[7]
        # exact multiples of dt are not rounded up (0.75 ms = 6 steps)
        c = SpikingSynapse(pre, post, :g; conn = fill(2.0f0, 1, 1), delay_dist = D.Dirac(0.75))
        T = Time(); arrivals = Int[]
        for n in 1:20
            pre.fire[1] = n == 3
            update_time!(T, _DT_SI)
            _K.forward!(c, c.param, _DT_SI, T)
            c.g[1] != 0 && (push!(arrivals, n); c.g[1] = 0)
        end
        @test arrivals == [3 + 6]

        # random network against the sorted-queue algorithm of 1.9.1 (arrival time <= t), same step arrivals
        Random.seed!(29)
        Npre, Npost = 25, 18
        pre, post = Identity(N = Npre), Identity(N = Npost)
        c = SpikingSynapse(pre, post, :g; conn = (p = 0.4f0, μ = 1.0f0, σ = 0.3f0), delay_dist = D.Uniform(0.3, 1.7))
        queue = [Tuple{Float32,Float32}[] for _ in 1:Npost]
        T = Time()
        for n in 1:300
            pre.fire .= n == 1 ? falses(Npre) : rand(Npre) .< 0.1
            update_time!(T, _DT_SI)
            t = get_time(T)
            for j in findall(pre.fire), s in c.colptr[j]:(c.colptr[j+1]-1)
                push!(queue[c.I[s]], (t + c.param.delaytime[s], c.W[s] * c.ρ[s]))
            end
            gref = zeros(Float32, Npost)
            for i in 1:Npost
                for (ta, w) in sort(queue[i]; by = first)
                    ta <= t && (gref[i] += w)
                end
                filter!(q -> q[1] > t, queue[i])
            end
            c.g .= 0
            _K.forward!(c, c.param, _DT_SI, T)
            @test c.g ≈ gref rtol = 1e-5 atol = 1e-6
        end
    end

    @testset "decay: cached factors, stale check and subnormal flush" begin
        c = fill(NaN, 5)
        f1 = _K._decay_factors!(c, 0.125f0, 20.0f0, 30.0f0)
        @test f1 == (Float32(exp(-0.125f0 / 20.0f0)), Float32(exp(-0.125f0 / 30.0f0)))
        c[4] = 99.0   # cache hit returns the stored value, no recomputation
        @test _K._decay_factors!(c, 0.125f0, 20.0f0, 30.0f0)[1] == 99.0f0
        @test !_K._decay_stale(c, 0.125f0, 20.0f0, 30.0f0)
        @test _K._decay_stale(c, 0.125f0, 21.0f0, 30.0f0)
        @test _K._decay_stale(c, 0.25f0, 20.0f0, 30.0f0)
        @test _K._decay_stale(fill(NaN, 5), 0.125f0, 20.0f0, 30.0f0)
        # a changed τ or dt recomputes
        @test _K._decay_factors!(c, 0.125f0, 25.0f0, 30.0f0)[1] == Float32(exp(-0.125f0 / 25.0f0))
        @test _K._decay_factors!(c, 0.25f0, 25.0f0, 30.0f0)[1] == Float32(exp(-0.25f0 / 25.0f0))
        x = Float32[1.0, 1.0f-38, 1.0f-40, 0.0, -1.0f-40]
        _K._rescale!(x, 0.5f0)
        @test x == Float32[0.5, 0.0, 0.0, 0.0, 0.0]
    end
end
