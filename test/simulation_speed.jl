using SNNModels
using BenchmarkTools
@load_units

E = Population(AdExParameter(), N=100, synapse=DoubleExpSynapse(), spike=PostSpike())

poisson_exc = PoissonLayer(
    10.2Hz,    # Mean firing rate (Hz) 
    N = 1000, # Neurons in the Poisson Layer
)
poisson_inh = PoissonLayer(
    3Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)

exc_conn = 
(
    p = 1.0f0,  # Probability of connecting to a neuron
    μ = 1.0f0,  # Synaptic strength (nS)
)
inh_conn = ( 
        p = 1.0f0,   # Probability of connecting to a neuron
    μ = 4.0f0,   # Synaptic strength (nS)
)
# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc = Stimulus(poisson_exc, E, :glu, conn=exc_conn, name = "noiseE")
stim_inh = Stimulus(poisson_inh, E, :gaba, conn=inh_conn, name = "noiseI")


model = compose(; E, stim_exc, stim_inh, silent=true)
# @profview sim!(model, 50s)
@btime sim!(model, 10s)
#   
#   240.436 ms (799501 allocations: 15.86 MiB) on DellTower
##

using SNNModels
@load_units

Random.seed!(1234)

E = SNNModels.Tripod(N=1)

poisson_exc = PoissonLayer(
    10.2Hz,    # Mean firing rate (Hz) 
    N = 200, # Neurons in the Poisson Layer
)
poisson_inh = PoissonLayer(
    3Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)

exc_conn = 
(
    p = 1.0f0,  # Probability of connecting to a neuron
    μ = 7.0,  # Synaptic strength (nS)
)
inh_conn = ( 
    p = 1.0f0,   # Probability of connecting to a neuron
    μ = 4.0,   # Synaptic strength (nS)
)
# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc1 = Stimulus(poisson_exc, E, :glu, :d1, conn=exc_conn, name = "noiseE")
stim_inh1 = Stimulus(poisson_inh, E, :gaba, :d1, conn=inh_conn, name = "noiseI")
stim_exc2 = Stimulus(poisson_exc, E, :glu, :d2, conn=exc_conn, name = "noiseE")
stim_inh2 = Stimulus(poisson_inh, E, :gaba, :d2, conn=inh_conn, name = "noiseI")

model = compose(; E, stim_exc1, stim_inh1, stim_exc2, stim_inh2, silent=true)
SNN.monitor!(E, [:fire, :v_d1, :v_s, :v_d2])
sim!(model, 5s)
SNN.raster(model.pop,    [4s, 5s])
SNN.vecplot(E, :v_d1, neurons=1, r=1s:5s)
SNN.vecplot(E, :v_d2, neurons=1, r=1s:5s)
SNN.vecplot(E, :v_s, neurons=1, r=1s:5s)
##

@profview sim!(model, 50s)
@btime sim!(model, 10s)
#   735.852 ms (799501 allocations: 15.86 MiB) on DellTower
##
