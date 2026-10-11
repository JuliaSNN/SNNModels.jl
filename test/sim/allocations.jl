using SNNModels
using Test
using Random
@load_units

# `sim!` and `train!` allocate nothing per step after warm-up: the bytes of a run do not depend on its length, only a
# per-call constant remains (the component vectors of the model form, the record buffers check, the `Ref` of `dt`).
# The step functions are called with dynamic dispatch (abstract component vectors); before 1.10 an unboxed `dt`
# argument or return value of those calls was boxed, 16 B per stimulus, population and connection and step.
@testset "Allocations per step" begin
    function _if_net()
        Random.seed!(1)
        E = IF(N = 40, synapse = CurrentSynapse(τe = 5ms, τi = 5ms), param = IFParameter(El = -49mV), name = "E")
        I = IF(N = 10, synapse = CurrentSynapse(τe = 5ms, τi = 5ms), param = IFParameter(El = -49mV), name = "I")
        E.v .= range(-60f0, -50f0, length = 40)
        EE = SpikingSynapse(E, E, :ge; conn = (p = 0.2, μ = 2.0))
        EI = SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 2.0))
        IE = SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 2.0))
        compose(; E, I, EE, EI, IE, silent = true)
    end
    function _poisson_net()
        Random.seed!(2)
        E = IF(N = 40, synapse = CurrentSynapse(τe = 5ms, τi = 5ms), name = "E")
        stim = PoissonStimulus(E, :ge; param = PoissonFixed(rate = 2kHz))
        EE = SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 0.5))
        compose(; E, EE, stim, silent = true)
    end
    function _tripod_net()
        Random.seed!(3)
        E = Tripod(N = 10, name = "E")
        stim = PoissonStimulus(E, :he; comp = :d1, param = PoissonFixed(rate = 2kHz))
        EE = SpikingSynapse(E, E, :glu, :d1; conn = (p = 0.2, μ = 2.0))
        compose(; E, EE, stim, silent = true)
    end

    # function barriers: `@allocated` of a call with local arguments
    _alloc_sim(m, d) = @allocated sim!(m, d)
    _alloc_train(m, d) = @allocated train!(m, d)
    _alloc_step(P, C, S, r, T) = @allocated SNNModels._sim_step!(P, C, S, r, T)
    _alloc_tstep(P, C, S, r, T) = @allocated SNNModels._train_step!(P, C, S, r, T)

    dt = 0.125f0
    n1, n2 = 10, 400
    for (name, mk) in (("IF + CurrentSynapse", _if_net), ("PoissonStimulus", _poisson_net), ("Tripod", _tripod_net))
        @testset "$name" begin
            m = mk()
            for _ in 1:2   # warm-up (compilation)
                sim!(m, n1 * dt); train!(m, n1 * dt)
                _alloc_sim(m, dt); _alloc_train(m, dt)
            end
            b1 = minimum(_alloc_sim(m, n1 * dt) for _ in 1:3)
            b2 = minimum(_alloc_sim(m, n2 * dt) for _ in 1:3)
            @test (b2 - b1) / (n2 - n1) < 1          # no per-step allocation (was 48 to 80 B per step)
            @test b2 < 4096                          # bounded per-call constant
            t1 = minimum(_alloc_train(m, n1 * dt) for _ in 1:3)
            t2 = minimum(_alloc_train(m, n2 * dt) for _ in 1:3)
            @test (t2 - t1) / (n2 - n1) < 1
            @test t2 < 4096
            # the step the loop runs: nothing at all
            P, C, S = SNNModels._args_model((), m)
            r = Ref(dt)
            _alloc_step(P, C, S, r, m.time); _alloc_tstep(P, C, S, r, m.time)
            @test _alloc_step(P, C, S, r, m.time) == 0
            @test _alloc_tstep(P, C, S, r, m.time) == 0
        end
    end
end
