using SNNPlots
using SpikingNeuralNetworks
@load_units

inputs = Poisson(; N = 1000)
inputs.param = PoissonParameter(; rate = 1Hz)

neurons = IF(; N = 1)
neurons.param = IFParameterSingleExponential(;
    τm = 10ms,
    τe = 5ms,
    El = -74mV,
    E_e = 0mV,
    Vt = -54mV,
    Vr = -60mV,
)

S = SpikingSynapse(
    inputs,
    neurons,
    :ge;
    μ = 0.01,
    p = 1.0,
    param = vSTDPParameter(; Wmax = 1.0),
)

model = compose(; inputs, neurons, S)
# histogram(S.W / S.param.Wmax; nbins = 20)
# monitor!(S, [(:W, [1, 2])])
@time train!(; model, duration = 100second)

scatter(S.W / S.param.Wmax)
histogram(S.W / S.param.Wmax; nbins = 20)
# plot(hcat(getrecord(S, :W)...)' / S.param.Wmax)
# heatmap(full(sparse(S.I, S.J, S.W / S.param.Wmax)))
