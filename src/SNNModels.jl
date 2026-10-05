"""
    SNNModels

Core library of the JuliaSNN ecosystem: neuron population models, synapse and receptor models,
connections with long- and short-term plasticity, metaplasticity, stimuli, recording, analysis
and input/output utilities.

A network is a `NamedTuple` assembled with `compose` (fields `pop`, `syn`, `stim`, `name`,
`time`) and simulated with `sim!` (no plasticity) or `train!` (plasticity active). Quantities
are plain `Float32` in the unit system loaded by `@load_units` (ms, mV, pA, pF, nS, cm).
Users normally load it through `SpikingNeuralNetworks`, which re-exports it as
`SNN.SNNModels`.
"""
module SNNModels

using DrWatson
import Dates: now
using JLD2
using LinearAlgebra
using SparseArrays
using Requires
using Documenter
using IterTools
using UnPack
using Random
using Logging
using StaticArrays
using ProgressBars
using Parameters
using LoopVectorization
using ThreadTools
using Distributions
using Graphs, MetaGraphs

include("utils/macros.jl")
include("utils/unit.jl")
include("utils/structs.jl")
include("populations/populations.jl")
include("connections/connections.jl")
include("stimuli/stimuli.jl")

include("utils/util.jl")
include("utils/copying.jl")
include("utils/io.jl")
include("utils/graph.jl")
include("utils/record.jl")
include("utils/main.jl")
include("utils/sparse_matrix.jl")
include("utils/spatial.jl")
include("utils/perturbation.jl")
include("analysis/spikes.jl")
include("analysis/populations.jl")
include("analysis/targets.jl")

end
