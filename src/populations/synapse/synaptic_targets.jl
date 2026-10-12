"""
    synaptic_target(targets::Dict, post::AbstractPopulation, sym::Symbol, target = nothing) -> (g, v_post)

Resolve where a connection onto `post` delivers its spikes. Called by the connection
constructors (e.g. `SpikingSynapse(pre, post, sym, target; conn)`); each population type
implements a method.

Returns the input buffer `g` (a `Vector{Float32}` of length `post.N`, into which `forward!`
adds the weights of the presynaptic spikes) and the membrane potential vector `v_post` of the
target compartment (used by voltage-dependent plasticity rules). It also records in `targets`
the entry `:sym` (name of the target, e.g. `:glu` or `"glu_d1"`) used by `print_model`/`graph`.

- Generalized IF point neurons (`IF`, `AdEx`, ...): `sym` is mapped by `get_synapse_symbol`
  (`:ge`, `:he` -> `:glu`; `:gi`, `:hi` -> `:gaba`; other symbols unchanged, e.g. the receptor
  groups of a `MultiReceptorSynapse`) and `g = post.receptors.<sym>`; `target` is ignored.
- Dendritic neurons (`Tripod`, `BallAndStick`): `target` is the compartment symbol
  (`:s`, `:d1`, `:d2` for `Tripod`; `:s`, `:d` for `BallAndStick`), `g` is the buffer
  `post.receptors_<target>.<sym>` and `v_post = post.v_<target>`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 10, param = SNN.PoissonParameter(10Hz))
T = SNN.Tripod(N = 5)
syn = SNN.SpikingSynapse(E, T, :glu, :d1; conn = (p = 0.5, μ = 1.0))
syn.targets[:sym]   # "glu_d1"
```
"""
synaptic_target
