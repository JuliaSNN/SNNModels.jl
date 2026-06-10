# Record System Refactor — Pre-allocated Matrix Storage

## Context

This plan describes the next step in the records system refactor. Earlier fixes (P2–P7) addressed
correctness and hot-path efficiency in `record!`. This plan addresses the allocation problem: the
append-based `Vector{Vector{T}}` storage generates thousands of heap allocations during simulation,
creating GC pressure proportional to `N_neurons × sr × duration`.

The goal is to replace append-based storage with a single pre-allocated `Matrix{T}` per monitored
key, so that the hot path in `record!` is allocation-free.

The relevant source file is `src/utils/record.jl`. The relevant entry points in the sim loop are
`src/utils/main.jl` (`sim!` and `train!`).

---

## What already exists (do not touch)

After the P2–P7 fixes, `records` is a flat `Dict{Symbol, Any}` with this structure:

- `records[:v]`, `records[:ge]`, etc. — data containers (currently `Vector{Vector{Float32}}`)
- `records[:fire]` — event dict (`Dict(:time => Vector{Float32}, :neurons => Vector{Vector{Int}})`)
- `records[:sr]` — `Dict{Symbol, Float32}` sampling rates per key
- `records[:indices]` — `Dict{Symbol, Vector{Int}}` optional neuron subsets per key
- `records[:start_time]`, `records[:end_time]` — `Dict{Symbol, Float32}` time bounds
- `records[:variables]` — `Vector{Symbol}` variable-group names (for `variables=` mechanism)
- `records[:data]` — `Vector{Symbol}` ordered dispatch list (P2 fix)
- `records[:meta]` — `Dict` containing `var_map` for `variables=` dispatch (P2 fix)

Do not change any of these keys or their semantics.

---

## What changes

### 1. Add a fill pointer to `meta`

`records[:meta]` already exists (added in P2). Add one more entry:

```
records[:meta][:fill_ptr] = Dict{Symbol, Int}()
```

This stores, per key, how many samples have been written so far. It is the write cursor into
the pre-allocated matrix. It starts at 0, increments by 1 on each recorded sample, resets to 0
on `clear_records!`.

The fill pointer is separate from the time step counter (`get_step(T)`) because: (a) the step
counter does not reset between sim calls, whereas the fill pointer resets on `clear_records!`;
(b) the fill pointer must track actual writes, not total elapsed steps.

### 2. Add `record_time` keyword to `monitor!`

```
monitor!(pop, [:v]; sr = 1000Hz, record_time = nothing)
```

- `record_time = nothing` (default): deferred mode — no matrix allocated at monitor time.
  The container remains `nothing` or an empty sentinel until the first `sim!`/`train!` call.
- `record_time = T` (e.g. `10s`): eager mode — allocate `Matrix{Float32}(undef, N, T_steps)`
  immediately, where `T_steps = max(1, floor(Int, T * sr / 1000))`.

In `monitor!`, store `record_time` in `records[:meta][:record_time][key]` so that
`preallocate_records!` can find it later.

Initialize the fill pointer entry: `records[:meta][:fill_ptr][key] = 0`.

### 3. Add `preallocate_records!(model, duration)` called at `sim!`/`train!` time

This function iterates all monitored components in the model. For each key in `records[:data]`
(excluding `:fire`):

- If the container is already a `Matrix` and has enough remaining capacity
  (`size(mat, 2) - fill_ptr >= steps_needed`): do nothing.
- If the container is already a `Matrix` but does not have enough capacity: extend it
  (see §4 below).
- If the container is `nothing` or a `Vector{}` (deferred mode): allocate a new
  `Matrix{Float32}(undef, N, steps_needed)` where `N` is the field length
  (`length(getfield(obj, key))` or from `indices` if a subset was specified) and
  `steps_needed = max(1, floor(Int, duration / get_dt(...) / period))`.

The point of calling this at `sim!`/`train!` time (not at `monitor!` time) is that `duration`
is only known at that point. This covers the default case with no `record_time` argument.

Call `preallocate_records!` once, before the main time loop, in both `sim!` (line ~224 in
`main.jl`) and `train!` (line ~109 in `main.jl`), right after `record_zero!`.

### 4. Matrix extension strategy

When the fill pointer reaches the matrix capacity (overflow), extend the matrix:

- Allocate a new matrix of size `(N, max(2 × old_capacity, old_capacity + steps_needed))`.
- Copy the existing data into the first `fill_ptr` columns.
- Replace `records[key]` with the new matrix.
- The old matrix becomes GC garbage — this is a one-time cost per overflow, not per sample.

This is the same doubling strategy Julia uses for `Vector`. Amortized over a full simulation,
the total number of copies is O(N × T), equal to exactly one copy of the final data, regardless
of how many extensions happen. This is much cheaper than N_samples individual push! allocations.

The overflow case in practice: a `sim!(model, 1ms)` called after a full 10s run will trigger
one extension from 10s capacity to 20s capacity. The 1ms of data is written without further
allocation. A subsequent `sim!(model, 10s)` will fit in the remaining 10s of capacity.

### 5. Rewrite `_record_sym` for the `Vector{T}` case

Currently:
```
push!(records, my_record[ind])   # allocates new Vector{Float32}(N) every sample
```

After: detect whether `records` is a `Matrix` or a `Vector{Vector{T}}` and branch:

- If `Matrix`: read `fill_ptr` from `meta`, increment it, write in-place:
  `records[eachindex(ind), fill_ptr] .= my_record[ind]` (or equivalent column slice).
- If `Vector{Vector{T}}` (fallback / legacy): keep the existing `push!` path.

The branching should be on the container type (dispatch), not a runtime flag. Define a new
method of `_record_sym` that takes `Matrix{T}` as the second argument alongside the existing
method that takes `Vector{Vector{T}}`. Julia dispatch selects the right one based on what is
stored in `records[key]`.

The `fill_ptr` must be accessible inside `_record_sym`. Since `_record_sym` currently receives
`obj.records[key]` (the container) but not `meta`, the cleanest way is to pass `fill_ptr` as
an additional argument from `record_sym!`, which already has access to `obj.records[:meta]`.

So the signature of the matrix-path `_record_sym` gains one parameter:
`_record_sym(source, container::Matrix{T}, ind, col::Int)`.

The caller (`record_sym!`) is responsible for reading and incrementing the fill pointer.
After incrementing, it passes the new value as `col` to `_record_sym`.

### 6. Update `record_sym!` to manage fill_ptr

`record_sym!` currently:
1. Checks `record_step` (sampling gate).
2. Resolves `ind`.
3. Calls `_record_sym(source, obj.records[key], ind)`.

After:
1. Same sampling gate check.
2. Same `ind` resolution.
3. If `obj.records[key] isa Matrix`: increment `meta[:fill_ptr][key]`, call matrix-path
   `_record_sym` with the new fill_ptr as `col`.
4. If `obj.records[key]` is a vector-of-vectors: call existing `_record_sym` (unchanged).

The type check (`isa Matrix`) is a single runtime branch per recorded step per monitored key.
It adds negligible overhead and can be eliminated later if the push! path is removed entirely.

### 7. Update `_clear` / `clear_records!`

Currently `_clear` calls `empty!(val)` on data containers, which releases all the push!-allocated
vectors to the GC.

After: for a `Matrix` container, reset the fill pointer to 0 instead of emptying or freeing the
matrix. The matrix remains allocated and ready for the next sim call.

```
if records[key] isa Matrix
    meta[:fill_ptr][key] = 0
    # optionally: fill!(records[key], zero(eltype(records[key])))
else
    empty!(records[key])
end
```

Do not free the matrix. The whole point is to reuse it across `clear_records!` + new `sim!`
cycles without re-allocation.

### 8. Update `getvariable` to return matrix slice

Currently: `hcat(rec...)` — O(N×T) allocation at read time.

After: if `rec isa Matrix`, return `rec[:, 1:fill_ptr]`. This is still an O(N×T) allocation
(you want an owned copy, not a view into the live buffer), but it happens once at read time,
not thousands of times during simulation.

The existing `hcat(rec...)` path remains for the `Vector{Vector{T}}` fallback case and for any
legacy containers.

### 9. Leave unchanged

- `:fire` recording: event-based, spike count is unknown in advance. Keep `push!` into
  `Dict(:time => Vector{Float32}, :neurons => Vector{Vector{Int}})`.
- The three complex `_record_sym` overloads: `Array{T,3}`, `Vector{Vector{T}}`,
  `Matrix{T}` source. In practice, monitored fields are always leaf `Vector{Float32}` or
  scalar `Float32` — these overloads are not exercised. Leave them using `push!`.
- The scalar `T<:Real` overload: already allocates into a flat `Vector{T}`, not
  `Vector{Vector{T}}`. Pre-allocating a `Vector{T}` of fixed length is straightforward but
  low priority since scalars are cheap. Leave for now; can apply the same fill_ptr pattern
  later if needed.
- All public API signatures: `monitor!`, `record(p, sym)`, `getvariable`, `clear_records!`,
  `interpolated_record`, `get_measure_interval`. Callers see the same interface.
- The `variables=` dispatch mechanism (P2 fix via `meta[:var_map]`). Untouched.

---

## Correctness invariants to preserve

1. **Segmented sims without clear**: `sim!(10s); sim!(1ms); sim!(10s)` — all data accumulates
   in the same matrix. `fill_ptr` increases monotonically. `getvariable` returns all 20080
   samples. Extension handles the capacity boundary transparently.

2. **Clear + resim**: `sim!(10s); clear_records!(); sim!(10s)` — fill_ptr resets to 0, matrix
   is reused. `getvariable` returns the second run's 10000 samples only. `start_time`/`end_time`
   reset as before (P5 fix already handles this).

3. **Looped short sims**: `for _ in 1:1000; sim!(model, 1ms); end` — `preallocate_records!`
   allocates for 1ms on the first call, extends by doubling on each overflow. After a few
   doublings the matrix is large enough to absorb the remaining iterations without further
   allocation. This is much cheaper than 1000 × 80 = 80000 push! allocations.

4. **`record_zero!` at t=0**: already calls `record!` before the main loop. This writes the
   first sample (fill_ptr goes 0→1). `preallocate_records!` must be called before `record_zero!`
   so the matrix exists when the first write happens.

---

## Summary of changes by file

| File | Change |
|---|---|
| `src/utils/record.jl` — `monitor!` | Add `record_time` kwarg; init `meta[:fill_ptr]` and `meta[:record_time]` per key |
| `src/utils/record.jl` — `preallocate_records!` | New function; allocates or extends matrices based on duration and sr |
| `src/utils/record.jl` — `record_sym!` | Read/increment fill_ptr; pass `col` to matrix-path `_record_sym` |
| `src/utils/record.jl` — `_record_sym` | Add matrix-path method: write to `container[ind, col]` |
| `src/utils/record.jl` — `_clear` | Reset fill_ptr instead of `empty!` for matrix containers |
| `src/utils/record.jl` — `getvariable` | Return `rec[:, 1:fill_ptr]` for matrix containers |
| `src/utils/main.jl` — `sim!` | Call `preallocate_records!(model, duration)` before `record_zero!` |
| `src/utils/main.jl` — `train!` | Same |
