using SNNPlots
using SpikingNeuralNetworks
@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random


E = AdEx(; N = 2000, param = AdExParameter(; El = -40mV))
I = IF(; N = 500, param = IFParameter())
G = Rate(; N = 100)
EE = SpikingSynapse(E, E, :ge; μ = 10, p = 0.02)
EI = SpikingSynapse(E, I, :ge; μ = 40, p = 0.02)
IE = SpikingSynapse(I, E, :gi; μ = -50, p = 0.02)
II = SpikingSynapse(I, I, :gi; μ = -10, p = 0.02)
# EG = SpikeRateSynapse(E, G; μ = 1.0, p = 0.02)
# GG = RateSynapse(G, G; μ = 1.2, p = 1.0)
P = [E, G, I]
C = [EE, EI, IE, II, EG]
# C = [EE, EG, GG]

monitor!([E, I], [:fire])
monitor!(G, [(:r)])
sim!(P, C; duration = 4second)
raster(E, [3.4, 4] .* 10e3)
vecplot(G, :r, 10:20)

# Random.seed!(101)
# E = AdEx(;N = 100, param = AdExParameter(;El=-40mV))
# EE = SpikingSynapse(E, E, :ge; μ=10, p = 0.02)
# EG = SpikeRateSynapse(E, G; μ = 1., p = 1.0)
# monitor!(E, [:fire])
# sim!(P, C; duration = 4second)
# raster([E], [900, 1000])
# plot!(xlims=(100,1000))
