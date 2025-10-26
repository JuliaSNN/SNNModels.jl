using SNNModels
using BenchmarkTools
using SpikingNeuralNetworks
using Random
@load_units


##
Random.seed!(1234)
my_rec = SNN.SomaReceptors
# zz = Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38, target = :zz) 
# push!(my_rec, zz)

my_syn = SNN.ReceptorSynapse(
    syn=my_rec,
    NMDA=SNN.SomaNMDA,
)

E = Population(AdExParameter(), N = 100, synapse = my_syn, spike = PostSpike())

poisson_exc = PoissonLayer(
    10.2Hz,    # Mean firing rate (Hz) 
    N = 1000, # Neurons in the Poisson Layer
)
poisson_inh = PoissonLayer(
    3Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)

exc_conn = (
    p = .20f0,  # Probability of connecting to a neuron
    μ = 3.0f0,  # Synaptic strength (nS)
)
inh_conn = (
    p = .20f0,   # Probability of connecting to a neuron
    μ = 4.0f0,   # Synaptic strength (nS)
)
# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc = Stimulus(poisson_exc, E, :glu, conn = exc_conn, name = "noiseE")
stim_inh = Stimulus(poisson_inh, E, :gaba, conn = inh_conn, name = "noiseI")


model = compose(; E, stim_exc, stim_inh, silent = true)
@btime sim!(model, 5s)
##
# # @profview sim!(model, 50s)
#
SNN.monitor!(E, [:fire, :v])
SNN.monitor!(E, [:g], variables=:synvars)
sim!(model, 5s)
# @profview sim!(model, 50s)
# SNN.raster(model.pop, [4s, 5s])
p = SNN.vecplot!(p, E, :v, neurons = 1, r = 4s:5s, add_spikes = true)
# p = plot()
# p = SNN.vecplot!(p, E, :synvars_g, neurons = 1, r = 4s:5s, add_spikes = true, sym_id=2)
# using Plots
# plot!(ylims=:auto)
#   
#   240.436 ms (799501 allocations: 15.86 MiB) on DellTower
##

using SNNModels
@load_units

Random.seed!(1234)

dend_neuron = SNN.DendNeuronParameter(; spike = SNN.PostSpike(At = 0, τA = 30ms, up = 1))
E = SNNModels.Population(dend_neuron, N = 1)
E  = SNN.Tripod(N=1)

poisson_exc = PoissonLayer(
    10.2Hz,    # Mean firing rate (Hz) 
    N = 100, # Neurons in the Poisson Layer
)
poisson_inh = PoissonLayer(
    3Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)

exc_conn = (
    p = 1.0f0,  # Probability of connecting to a neuron
    μ = 15.5,  # Synaptic strength (nS)
)
inh_conn = (
    p = 1.0f0,   # Probability of connecting to a neuron
    μ = 4.0,   # Synaptic strength (nS)
)
# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc1 = Stimulus(poisson_exc, E, :glu, :d1, conn = exc_conn, name = "noiseE")
stim_inh1 = Stimulus(poisson_inh, E, :gaba, :d1, conn = inh_conn, name = "noiseI")
stim_exc2 = Stimulus(poisson_exc, E, :glu, :d2, conn = exc_conn, name = "noiseE")
stim_inh2 = Stimulus(poisson_inh, E, :gaba, :d2, conn = inh_conn, name = "noiseI")

model = compose(; E, stim_exc1, stim_inh1, stim_exc2, stim_inh2, silent = true)
SNN.monitor!(E, [:fire, :v_d1, :v_s, :v_d2])

# sim!(model, 0.01s)
#
sim!(model, 5s)
SNN.raster(model.pop, [4s, 5s])
SNN.vecplot(E, :v_d1, neurons = 1, r = 4s:5s)
SNN.vecplot(E, :v_d2, neurons = 1, r = 4s:5s)
SNN.vecplot(E, :v_s, neurons = 1, r = 3.7s:5s, add_spikes = true)
##

@profview sim!(model, 50s)
@btime sim!(model, 10s)
#   735.852 ms (799501 allocations: 15.86 MiB) on DellTower
##
