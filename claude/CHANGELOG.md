# SNNModels changelog (branch notes)

## 1.9.2: compact perturbation records
- `perturbation_test(...; add_records = cond)` stored every window as its own
  `Dict("interval", "data")` under `records[:perturbation][var][cond][n]`. With 1,320 presentations x 13 conditions
  x several variables that is 17,160 small objects per population and variable, JLD2 files of 35-41 GB and a load
  that peaked at 83 GB (reported by SNN2AC_optim). The windows of one `(variable, condition)` are now appended to
  one ragged store: `PerturbationStore` (dense: `N x sum(T)` Float32 kept as a flat vector, CSR `offsets`, `tstart`,
  `tend`, `ids`) and `PerturbationSpikeStore` (`:fire`: flat `times` and `neuron`, CSR `offsets`).
  Both carry a `version` field (1). See the docstrings.
- `perturbation_record` returns bit-identical values to the old layout (same window order, clipping,
  interpolation); it reads both layouts.
- New: `perturbation_windows(obj, var, cond)` (tstart, tend, data views per window, insertion order),
  `compact_perturbation_records!(model_or_obj)` (converts old records in place, one variable at a time,
  checked bit by bit against the old dictionaries before they are dropped; idempotent; returns a summary).
  Downstream code that read `entry["interval"]` / `entry["data"]` should use `perturbation_windows`;
  `store[n]` still returns such a dictionary (a copy).
- Bug fix: `n` was `length(dict) + 1`, so after `clear_perturbation_records!(...; n = 2)` of `{1,2,3}` the next
  flush overwrote window 3. Window numbers (`ids`) are now never reused (`maximum + 1`); `n` in `clear_*`
  refers to that number. Also fixed for the old layout. Flushing into a condition that still holds the old layout
  converts it first, so one `(variable, condition)` never has two stores.
- Compatibility: records that are compact (written, or converted by `compact_perturbation_records!`, with 1.9.2) cannot be read by
  the `perturbation_record` of 1.9.1 or earlier: every process that reads such files needs >= 1.9.2. Stores carry a `version` field
  (1); readers and writers throw an `ArgumentError` for another value. `offsets` has `length(store) + 1` entries (CSR style), so
  objects with 0 rows and windows with 0 columns work.
- Not changed: a window of one sample (`T = 1`, e.g. `perturbation_test(model, 1ms)` at 1 kHz) makes
  `perturbation_record` throw an `ArgumentError` from Interpolations, as before.

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
