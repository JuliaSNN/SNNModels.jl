using SNNModels
using Test
using Random
@load_units

# Stimulus-to-population spike hand-off: an Identity population whose input is written only by
# SpikeTimeStimulus updates just the neurons the stimulus wrote and exposes the list of the
# neurons that fired (shared with the connections that read it) instead of being scanned.
# The result must equal the dense update, flag by flag and bit by bit.

const _K = SNNModels
const _DT_SH = 0.125f0

# stimulus spikes on a regular grid of steps, several per step, repeated neurons included
function _spike_param(Nin, nspikes, nsteps; seed = 1)
    rng = MersenneTwister(seed)
    steps = rand(rng, 1:nsteps, nspikes)
    return SpikeTimeParameter(Float32.(steps .* _DT_SH .- 0.01f0), rand(rng, 1:Nin, nspikes))
end

# an Identity driven by a stimulus; `sparse = false` forces the dense update of the same network
function _driven(; Nin = 20, N = 30, conn = (p = 0.2, μ = 1.0), param, sparse = true, seed = 2)
    Random.seed!(seed)
    pop = Identity(N = N)
    stim = SpikeTimeStimulus(pop, :g; param, conn)
    sparse || (pop.ndrivers = 0)
    return pop, stim
end

function _step!(pop, stim, T)
    update_time!(T, _DT_SH)
    stimulate!(stim, stim.param, T, _DT_SH)
    integrate!(pop, pop.param, _DT_SH)
end

@testset "Spike hand-off, stimulus to Identity" begin
    @testset "sparse update equals the dense update" begin
        for (Nin, N, conn) in ((20, 30, (p = 0.2, μ = 1.0)), (5, 7, (p = 0.9, μ = 0.5)), (8, 8, (p = 1.0, μ = 1.0)))
            param = _spike_param(Nin, 400, 300)
            ps, ss = _driven(; Nin, N, conn, param = deepcopy(param))
            pd, sd = _driven(; Nin, N, conn, param = deepcopy(param), sparse = false)
            Ts, Td = Time(), Time()
            nfired = 0
            for n in 1:320
                _step!(ps, ss, Ts); _step!(pd, sd, Td)
                @test ps.fire == pd.fire
                @test ps.spikecount == pd.spikecount
                @test ps.h == pd.h
                @test ps.g == pd.g
                @test ss.fire == sd.fire
                @test findall(ss.fire) == sort(unique(ss.spiking.idx[1:ss.spiking.n]))
                nfired += sum(ps.fire)
            end
            @test nfired > 0
            @test _K._list_driven(ps) && !_K._list_driven(pd)
        end
    end

    @testset "the fed list is the ascending list of the fired neurons" begin
        param = _spike_param(20, 400, 300)
        pop, stim = _driven(; param)
        T = Time()
        for n in 1:320
            _step!(pop, stim, T)
            @test pop.spikes.fed
            @test _K.spike_list!(pop.spikes, pop.fire) == findall(pop.fire)
            @test issorted(_K.spike_list!(pop.spikes, pop.fire))
        end
    end

    @testset "connections share the list of the population and are unchanged" begin
        param = _spike_param(20, 400, 300)
        pre, stim = _driven(; N = 20, param)
        post = Identity(N = 10)
        Wc = Float32.(rand(MersenneTwister(5), 10, 20) .* (rand(MersenneTwister(6), 10, 20) .< 0.5))
        syn = SpikingSynapse(pre, post, :g; conn = Wc)
        @test syn.pre_spikes === pre.spikes
        @test syn.post_spikes === post.spikes
        P, C, S = [pre, post], [syn], [stim]
        # the reference network forces the dense update of `pre`
        pre_d, stim_d = _driven(; N = 20, param = deepcopy(param), sparse = false)
        post_d = Identity(N = 10)
        syn_d = SpikingSynapse(pre_d, post_d, :g; conn = Wc)
        T, Td = Time(), Time()
        for n in 1:320
            sim!(P, C, S, _DT_SH, T)
            sim!([pre_d, post_d], [syn_d], [stim_d], _DT_SH, Td)
            @test post.h == post_d.h
            @test post.fire == post_d.fire
        end
        @test sum(post.h) > 0
    end

    @testset "another writer of the input switches the population to the dense update" begin
        param = _spike_param(20, 400, 300)
        pop, stim = _driven(; param)
        other = _K.Poisson(N = 5, param = _K.PoissonParameter(200Hz))
        syn = SpikingSynapse(other, pop, :g; conn = (p = 1.0, μ = 1.0))
        @test !_K._list_driven(pop)
        T = Time()
        for n in 1:100
            update_time!(T, _DT_SH)
            stimulate!(stim, stim.param, T, _DT_SH)
            integrate!(other, other.param, _DT_SH)
            _K.forward!(syn, syn.param, _DT_SH, T)
            integrate!(pop, pop.param, _DT_SH)
            @test !pop.spikes.fed
            @test all(pop.g .== 0)
            @test pop.fire == (pop.spikecount .> 0)
        end
        @test sum(pop.h) > 0
    end

    @testset "overflow of the inbox falls back to the dense update" begin
        # every input spike writes all 6 neurons: more writes than neurons in one step
        param = SpikeTimeParameter(Float32[0.1, 0.1, 0.1, 0.2, 0.3], [1, 2, 3, 1, 2])
        ps, ss = _driven(; Nin = 3, N = 6, conn = (p = 1.0, μ = 1.0), param = deepcopy(param))
        pd, sd = _driven(; Nin = 3, N = 6, conn = (p = 1.0, μ = 1.0), param = deepcopy(param), sparse = false)
        Ts, Td = Time(), Time()
        for n in 1:6
            _step!(ps, ss, Ts); _step!(pd, sd, Td)
            @test ps.h == pd.h
            @test ps.fire == pd.fire
            @test _K.spike_list!(ps.spikes, ps.fire) == findall(pd.fire)
        end
        @test sum(ps.h) == 30
    end

    @testset "negative and cancelling input does not fire and leaves no residue" begin
        pop = Identity(N = 4)
        W = Float32[1 -1 0; -1 0 0; 0 0 -2; 1 1 0]
        stim = SpikeTimeStimulus(pop, :g; param = SpikeTimeParameter(Float32[0.1, 0.1, 0.3], [1, 2, 3]), conn = W)
        T = Time()
        _step!(pop, stim, T)   # t = 0.125: neurons 1, 2 spike; input to Identity 1: 1 - 1 = 0, 2: -1 + 0, 4: 1 + 1
        @test pop.fire == [false, false, false, true]
        @test pop.spikecount == Float32[0, 0, 0, 2]
        @test pop.h == Float32[0, -1, 0, 2]
        _step!(pop, stim, T)
        _step!(pop, stim, T)   # t = 0.375: neuron 3 spikes, input -2 to Identity 3
        @test !any(pop.fire)
        @test pop.h == Float32[0, -1, -2, 2]
        @test all(pop.g .== 0)
        @test isempty(_K.spike_list!(pop.spikes, pop.fire))
    end

    @testset "the update resumes after a dense step" begin
        param = _spike_param(20, 400, 300)
        pop, stim = _driven(; param)
        ref, sref = _driven(; param = deepcopy(param), sparse = false)
        T, Tr = Time(), Time()
        for n in 1:300
            n == 100 && (pop.ndrivers = 0)          # three dense steps ...
            n == 103 && (pop.ndrivers = pop.nwriters)  # ... then sparse again
            _step!(pop, stim, T); _step!(ref, sref, Tr)
            @test pop.fire == ref.fire && pop.h == ref.h && pop.spikecount == ref.spikecount
            @test _K.spike_list!(pop.spikes, pop.fire) == findall(ref.fire)
        end
    end
end
