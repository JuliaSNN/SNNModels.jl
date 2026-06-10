using SNNModels
using Test
@load_units

# IFParameter(El=-49mV) → spontaneous firing (Vt=-50mV, so leak drives toward threshold)
const _ei_active = IFParameter(El = -49mV)
const _ei_inhib  = IFParameter(El = -49mV)

@testset "E/I network integration" begin

    @testset "basic E/I IF network — spikes produced" begin
        E = IF(N = 40, param = _ei_active, name = "E")
        I = IF(N = 10, param = _ei_inhib,  name = "I")

        ee = SpikingSynapse(E, E, :ge; conn = (p = 0.2f0, μ = 0.5f0))
        ei = SpikingSynapse(E, I, :ge; conn = (p = 0.5f0, μ = 1.0f0))
        ie = SpikingSynapse(I, E, :gi; conn = (p = 0.5f0, μ = 2.0f0))

        model = compose(E = E, I = I, ee = ee, ei = ei, ie = ie, silent = true)
        monitor!(E, [:fire])
        monitor!(I, [:fire])

        sim!(model, 500ms)

        @test sum(length, spiketimes(E)) > 0
        @test sum(length, spiketimes(I)) > 0
    end

    @testset "E/I network — time advances correctly" begin
        E   = IF(N = 20, param = _ei_active, name = "E")
        I   = IF(N = 5,  param = _ei_inhib,  name = "I")
        syn = SpikingSynapse(E, I, :ge; conn = (p = 0.5f0, μ = 1.0f0))
        model = compose(E = E, I = I, syn = syn, silent = true)
        sim!(model, 200ms)
        @test get_time(model) ≈ 200f0
    end

    @testset "E/I network — inhibition reduces E rate" begin
        # Excitatory-only baseline
        E1    = IF(N = 30, param = _ei_active, name = "E1")
        m1    = compose(E = E1, silent = true)
        monitor!(E1, [:fire])
        sim!(m1, 500ms)
        n_noinhib = sum(length, spiketimes(E1))

        # Add inhibitory feedback
        E2 = IF(N = 30, param = _ei_active, name = "E2")
        I2 = IF(N = 10, param = _ei_inhib,  name = "I2")
        ei2 = SpikingSynapse(E2, I2, :ge; conn = (p = 0.8f0, μ = 1.5f0))
        ie2 = SpikingSynapse(I2, E2, :gi; conn = (p = 0.8f0, μ = 3.0f0))
        m2  = compose(E = E2, I = I2, ei = ei2, ie = ie2, silent = true)
        monitor!(E2, [:fire])
        sim!(m2, 500ms)
        n_inhib = sum(length, spiketimes(E2))

        @test n_noinhib >= n_inhib
    end

end
true
