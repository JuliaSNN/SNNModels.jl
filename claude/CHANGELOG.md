# SNNModels changelog (branch notes)

## core-clock-sparse

### `sparse_matrix` without a dense matrix
- All NamedTuple rules (`:Fixed`, `:FixedIn`, `:FixedOut`, `:Bernoulli`, `:PowerLaw`) build
  CSC directly in O(nnz) memory; return type `SparseMatrixCSC{Float32,Int}` as before.
- Behaviour change: seeded networks do not reproduce the old realisations (different random
  stream). Statistics are the same (tested against `sparse_matrix_dense_legacy`).
- Behaviour change: recurrent `SpikingSynapse(pre, pre, ...)` no longer stores zero-weight
  autapses (they were removed only by value before, and could be potentiated).
- `SpikeTimeStimulusIdentity` uses a sparse identity.

### iSTDP kernels
- iSTDPRate/iSTDPPotential: the postsynaptic pass reassigned its loop variable
  (`st = index[st]`) inside `@turbo`, which LoopVectorization does not support; iSTDPRate
  diverged from a plain loop by up to 22 pF. Now plain `@inbounds @simd` loops.

The integer-clock work lives on the local branch `wip-clock`.
