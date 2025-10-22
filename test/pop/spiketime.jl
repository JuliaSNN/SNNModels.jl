spiketime = [1000ms, 1100ms]
neurons = [1, 2]
inputs = SpikeTimeParameter(spiketime, neurons)
w = zeros(Float32, 2, 2)
w[2, 1] = 1.0f0

st = Identity(N = max_neuron(inputs))
stim = SpikeTimeStimulusIdentity(st, :g, param = inputs)
syn = SpikingSynapse(st, st, :g, conn = w, LTPParam = STDPGerstner())
model = compose(pop = st, stim = stim, syn = syn, silent = true)
monitor!(model.pop..., [:fire])
# monitor!(model.syn..., [:tpre, :tpost, :W])
train!(model = model, duration = 3000ms, dt = 0.1ms)
true

# # vecplot(model.syn..., [:tpre, :tpost], neurons = 1:2, r = 0s:3s)
# # plot(getvariable(model.syn..., :W)')
# raster(model.pop, [0s, 3s])

# @info model.syn[1].W .- 1
# ΔWs[i] = S
