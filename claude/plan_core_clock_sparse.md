# Plan: integer clock, dense-free sparse_matrix, iSTDP loop fix

Status (scope change requested by the user): the clock change was moved off this branch
to the local branch `wip-clock` (to be redone together with the spike-buffer redesign).
`core-clock-sparse` holds Change 2 (sparse_matrix) and Change 3 (iSTDP) only.

Branch `core-clock-sparse` from `event-stdp`. Local only, no push. Untracked plan file.

## Impact analysis (codegraph CLI + grep)

- `update_time!` (10 symbols): record.jl (2 methods), main.jl `train!`/`sim!`/`initialize!`,
  test/syn/stdp_event.jl. No callers outside SNNModels.
- `reset_time!` (4): record.jl, test/sim/sim_control.jl, SNN.jl tutorial HetRec_layer.jl.
- `get_time` (155, mostly other packages and the unrelated Auryn C++ method): every caller
  treats it as Float32 ms (STDP traces, STP, poisson rates, timed stimuli, records,
  pbar). Return type is kept Float32 -> no API change; new `get_time64` returns Float64.
- `.t[1]` direct access: only record.jl and test/utils/structs_test.jl (read only).
- `.tt[1]`: record.jl, structs_test.jl (`== Int32(800)`, value-compatible with Int64).
  SNNUtils `.tt` hits are a different struct (trackers), unaffected.
- `get_step` (record_step, record_sym_dense!, aggregate_scaling, turnover): all do
  `tt % period`; changing the return from Float32 to Int64 is value-identical below 2^24
  steps and fixes the phase error above it.
- `get_dt` only in `record_step` (legacy recorder). `T.dt` was never updated by
  `update_time!`, so it was stale (0.125) when sim! ran at another dt. It now tracks the
  actual dt: legacy-recorder period becomes correct (bug fix, behaviour change).
- `times_buf`: written in record.jl, read in analysis/spikes.jl (searchsorted on Float32),
  copying.jl, perturbation_test.jl. Keeping it Float32 but writing the integer-derived time
  Float32(t_base + (tt-tt_base)*dt64) gives the exact values "Float32(step*dt) at read time"
  would give, needs no reader / on-disk change, and stays correct across dt changes
  (a pure step buffer would need per-chunk dt bookkeeping). Chosen as least invasive.
- `Time(...)` constructors: main.jl / util.jl defaults, tests, SNN.jl examples (`SNN.Time()`).
- SNNsave/SNNload: plain JLD2/DrWatson save of the model NamedTuple; Time layout changes,
  so SNNload gets a JLD2 typemap `Upgrade` for `SNNModels.Time` with an `rconvert` that
  accepts both the old (t, tt::Int32, dt) and the new layout.
- `sparse_matrix` (93 symbols): SpikingSynapse, SpikeTimeStimulus, PoissonStimulusLayer
  (both constructors), hetrec (dense M' path), tests, examples. All consume via `dsparse`,
  which accepts any SparseMatrixCSC; VIT = Vector{Int}, so index type stays Int.
  PoissonStimulusLayer has no other dense matrix (randcache is length N); it is fixed by
  this change. SpikeTimeStimulusIdentity builds `Matrix(I(N))` (dense N x N): switch to a
  sparse identity.
- Autapse removal `w[diagind(w)] .= 0` currently leaves explicit stored zeros (synapses
  with W = 0 that plasticity can grow). Replace by structural removal.

No public-API break: `get_time` stays Float32, `Time` fields t/tt/dt keep names, `tt`
widens Int32 -> Int64, `get_step` returns Int64 (was Float32 of an integer).

## Change 1: integer clock
Time fields: t (Float32 mirror), tt (Int64 step counter, ground truth), dt (Float32),
t_base::Float64, tt_base::Int64, dt64::Float64 (shortest-decimal Float64 of dt so that
100000 * 0.1 == 10000.0). Time(ms) = t_base + (tt - tt_base) * dt64, never accumulated.
On a dt change at update_time!, rebase: t_base = current time, tt_base = tt. Copy, reset,
constructors and JLD2 upgrade handled. Tests: 10 s exact, 5+5 == 10, dt 0.1 -> 0.125 chain,
save/load mid-run, SpikeTimeStimulus at 5000 ms -> step 50000 (fails on base), recorded spike
times on the dt grid.

## Change 2: sparse_matrix without dense matrix
Rules: :Fixed/:FixedIn (in-degree Npre - round((1-rho)Npre)), :FixedOut, :Bernoulli,
:PowerLaw (Pareto out-degree). Direct CSC: Bernoulli by geometric skip sampling over the
column-major index space, fixed rules by per-row/column sampling without replacement, weights
drawn per non-zero in Float32 from dist(|mu|, sigma), draws <= 0 dropped (as before), sign
applied afterwards. Old generator kept as `sparse_matrix_dense_legacy` (not exported) for a
statistical comparison test. Scaling test at 1e5 x 1e5, p = 1e-3.

## Change 3: iSTDP loop fix
Replace `@turbo for st ...; st = index[st]` (loop var reassigned) by
`@inbounds @simd for k ...; s = index[k]` in iSTDPRate and iSTDPPotential; regression test
against a plain reference loop.

## Checkpoints
1. test: failing clock tests on base (committed with the fix if failing tests are not wanted alone)
2. feat: integer clock
3. feat: dense-free sparse_matrix + tests + perf notes
4. fix: iSTDP loops + test
