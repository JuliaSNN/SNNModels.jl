E1 = IF(; N = 1, param = IFParameter(; El = -65mV, Vr = -55mV))
E2 = IF(; N = 1, param = IFParameter(; El = -65mV, Vr = -55mV))
EE = SpikingSynapse(
    E1,
    E2,
    :ge;
    μ = 60 * 0.27 / 10,
    p = 1,
    delay_dist = Uniform(1ms, 5ms),
)
# PositiveUniform(10ms/0.125, 0.1ms./0.125))

# monitor!([E, I], [:fire])
# sim!(P, C; duration = 1second)
monitor!(EE, [:g, :W])
#
monitor!(E1, [:v, :fire])
monitor!(E2, [:v, :fire])
sim!([E1, E2], [EE]; duration = 1second, dt = 0.125)
E1.v[1] = -20mV
# E1.fire[1] = 1
sim!([E1, E2], [EE]; duration = 1second, dt = 0.125)

vecplot(E1, :v, r = 990:0.01:1.1second, dt = 0.125, neurons = 1:1)
vecplot(EE, :g, r = 990:0.01:1.1second, dt = 0.125, neurons = 1:1)
vecplot(E2, :v, r = 990:0.01:1.1second, dt = 0.125, neurons = 1:1)
