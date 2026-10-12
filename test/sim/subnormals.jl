using SNNModels
using Test
@load_units

# sim!/train! run with flush-to-zero (`flush_subnormals = true`): decaying variables of quiet populations and silent
# synapses never become subnormal (they stop at a tiny normal value), spikes are unchanged, the thread mode is restored.
@testset "Flush-to-zero in sim!/train!" begin
    nsub(a) = count(issubnormal, a)
    noc = SNNModels.AbstractConnection[SNNModels.EmptySynapse()]
    nos = SNNModels.AbstractStimulus[SNNModels.EmptyStimulus()]

    @testset "quiet Rate population" begin
        # x decays by 0.9 per step from about 1: subnormal after about 830 steps and stuck there under IEEE rounding
        mk() = (p = Rate(N = 64); p.x .= range(-1f0, 1f0, length = 64); p.I .= 0; p)
        p = mk()
        sim!(SNNModels.AbstractPopulation[p], noc, nos; dt = 0.1f0, duration = 300f0, flush_subnormals = false)
        @test nsub(p.x) > 0                                   # the regime the fix is about exists
        q = mk()
        sim!(SNNModels.AbstractPopulation[q], noc, nos; dt = 0.1f0, duration = 300f0)
        @test nsub(q.x) == 0 && nsub(q.r) == 0
        @test maximum(abs, q.x) <= 10 * floatmin(Float32)     # stops at most floatmin * τ / dt
        @test maximum(abs.(q.x .- p.x)) <= 10 * floatmin(Float32)
    end

    @testset "silent CurrentSynapse, train!" begin
        mk() = (E = IF(N = 32, synapse = CurrentSynapse(τe = 5ms, τi = 5ms)); E.I .= 0; E.synvars.ge .= 1f0; E.synvars.gi .= 2f0; E)
        E = mk()
        train!(SNNModels.AbstractPopulation[E], noc, nos; dt = 0.1f0, duration = 600f0, flush_subnormals = false)
        @test nsub(E.synvars.ge) > 0 && nsub(E.synvars.gi) > 0
        F = mk()
        train!(SNNModels.AbstractPopulation[F], noc, nos; dt = 0.1f0, duration = 600f0)
        @test nsub(F.synvars.ge) == 0 && nsub(F.synvars.gi) == 0 && nsub(F.syn_curr) == 0
        @test F.v == E.v                                      # membrane unchanged
    end

    @testset "spikes identical with and without flush" begin
        function net()
            E = IF(N = 200, synapse = CurrentSynapse(τe = 5ms, τi = 5ms), param = IFParameter(El = -49mV))
            E.v .= range(-60f0, -50f0, length = 200)
            EE = SpikingSynapse(E, E, :ge; conn = (p = 0.05, μ = 2.0), name = "EE")
            monitor!(E, [:fire])
            return E, EE
        end
        E1, C1 = net(); E2, C2 = deepcopy((E1, C1))
        sim!(SNNModels.AbstractPopulation[E1], SNNModels.AbstractConnection[C1]; dt = 0.1f0, duration = 500f0, flush_subnormals = false)
        sim!(SNNModels.AbstractPopulation[E2], SNNModels.AbstractConnection[C2]; dt = 0.1f0, duration = 500f0)
        @test spiketimes(E1) == spiketimes(E2)
        @test E1.v == E2.v
    end

    @testset "thread mode restored" begin
        p = Rate(N = 8)
        set_zero_subnormals(false)
        sim!(SNNModels.AbstractPopulation[p], noc, nos; dt = 0.1f0, duration = 1f0)
        @test get_zero_subnormals() == false
        @test_throws ErrorException sim!(SNNModels.AbstractPopulation[p], noc, nos; dt = 0.1f0, duration = 1f0,
                                         perturbation! = (; kw...) -> error("stop"))
        @test get_zero_subnormals() == false
        @test SNNModels.with_flush_subnormals(get_zero_subnormals) == true
        @test SNNModels.with_flush_subnormals(get_zero_subnormals, false) == false
    end
end
