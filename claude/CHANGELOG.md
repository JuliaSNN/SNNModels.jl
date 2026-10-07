# SNNModels changelog (branch notes)

## 1.9.1: recording buffers across chunks
- `_extend_fire!` and `_extend_dense!` (called when a new `sim!`/`train!` chunk needs more room)
  grew by exactly the missing amount, so a run of many chunks reallocated and copied the whole
  buffer at almost every chunk (quadratic copying, garbage per chunk). They now grow
  geometrically (`max(old + extra, 2 old)`), and warn once per recording, sharing the latch of
  the mid-loop overflow path. Reported by SNN2AC.

## fix/sweep-bugs: membrane pair rule (1.9.0)
- `AdExParameter`, `IFParameter`: `C`, `gl`, `R`, `τm` always consistent. Construction from a
  pair (or the default pair); a single value is an error; more values must agree within 0.1 %.
  One changed value follows a fixed rule (τm keeps gl; C keeps gl; gl/R keeps C) in `@update!`,
  `setproperty!` (mutable AdEx), `with_membrane`, `make_heterogeneous`.
- Reason: AdEx/IF integrate with τm, R; Tripod/BallAndStick with C, gl. `@update! adex.τm` had no
  effect on Tripod; `IFParameter(τm = x)` silently combined with the default R.
- `@snn_kw` keyword constructors pass their fields through `snn_kw_finalize(T, nt)` (identity by
  default) before building the struct.

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
