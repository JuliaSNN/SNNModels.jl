# STDP Rules — Inconsistencies Memo

Two incompatible trace-update schemes coexist in the codebase. This memo documents what each rule does and where the schemes diverge.

---

## Scheme A — Event-driven, Auryn style (exact multiplicative decay)

Used by: `STDPGerstner`, `STDPConfavreux2025`, `STDPWeightDependent`, `STDPTriplet`
(shared loops in `sparse_plasticity/STDP_kernels.jl`).

Per step: (1) for each presynaptic spike, walk its outgoing synapses (`colptr`) and apply
the pre-spike update; (2) for each postsynaptic spike, walk its incoming synapses
(`rowptr`/`index`) and apply the post-spike update; clamp only touched weights; (3) add 1
to the traces of the neurons that fired; (4) multiply every trace by the precomputed
`exp(-dt/τ)`. Traces are therefore read before this step's spikes are added (Auryn and
Brian2 convention). Cost O(N) multiplies plus O(spikes x fan-out), no exp per neuron, no
O(|W|) scan. Serial (no threading).

- Mathematically exact decay (up to Float32 rounding of the factor).
- `last_pre` / `last_post` store spike times (informational).
- The ephemeral `Δpre` / `Δpost` buffers were removed.
- All weights are clamped once on the first step (`initialized` flag), then only touched ones.

---

## Scheme B — Continuous Euler (per-dt leaky integration)

Used by: `STDPMexicanHat`, `STDPAntiSymmetric`, `STDPSymmetric`, `iSTDPRate`, `iSTDPPotential`, `vSTDPParameter`

Traces decay every dt via `@turbo` Euler step, and are incremented on spike:

```julia
@turbo for i in eachindex(fireI)
    tpost[i] += dt * (-tpost[i]) / τ
end
@simd for i in findall(fireI)
    tpost[i] += 1
end
```

- Euler integration error proportional to `dt/τ`; at default `dt=0.1ms` and `τ=20ms`, error ≈ 0.5% per step — acceptable.
- No `last_pre`/`last_post` storage.
- Weight update triggered only on spike events (row/col pointer loops checking `fireJ[j]` or `fireI[i]`).

---

## Inconsistency 1 and 2 — resolved (event-stdp)

`STDPGerstner` / `STDPConfavreux2025` no longer scan all synapses each dt and no longer keep
`Δpre`/`Δpost` buffers. `STDPMexicanHat` and `STDPSymmetric`/`STDPAntiSymmetric` no longer
clamp all of `W` each step. STDPGerstner's former A^2 amplitude (trace incremented by A and
multiplied by A again) is fixed; its default `A_post` is now negative.

---

## Inconsistency 3 — Weight update scope

- Scheme A: resolved. Weights are updated only through the spike passes (colptr for pre spikes, rowptr/index for post spikes); no scan of untouched synapses.
- Scheme B (`iSTDPRate`): weight updated only inside `if fireJ[j]` / `if fireI[i]` blocks, using colptr/rowptr to reach affected synapses. No scan of untouched synapses.
- Scheme B (`vSTDP`): update runs via `Threads.@threads` over `eachindex(fireJ)` chunks — same structure as Scheme A but uses col-pointer to reach synapses rather than iterating all of `W`.

---

## Inconsistency 4 — `fireI` absent in `vSTDP`

`plasticity!` for `vSTDPParameter` unpacks `fireJ` but **not** `fireI`:
```julia
@unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, index = c
```
LTD (`u` trace) is read via `u[I[s]]` (post-neuron index), but `fireI` is never checked — LTD is applied on every pre-spike regardless of whether the post-neuron fired. This is consistent with the Clopath 2010 rule, where the LTP term requires post-spike gating but LTD only requires pre-spike and the slow `u` trace. **Not a bug**, but differs from Gerstner-style paired rules.

---

## Conventions common to the rules

- Plasticity runs only under `train!`; `sim!` never calls `update_traces!`/`plasticity!`.
- `STDPGerstner` sign convention: signed amplitudes used once, `A_pre > 0` LTP and `A_post < 0`
  LTD by default (before 1.8.2 the effective amplitude was `A^2`).
- Same-step spikes: Scheme A rules read traces before this step's increments, so a pre and a
  post spike in the same step do not interact. Scheme B rules increment their traces during the pass (STDPMexicanHat before both passes; iSTDP between the pre and post pass), so same-step spikes may interact.

---

## Summary table

| Rule | Trace update | Weight scan | Extra state |
|---|---|---|---|
| STDPGerstner | exact decay per step (Scheme A) | on spike (colptr+rowptr) | last_pre, last_post, initialized |
| STDPConfavreux2025 | exact decay per step (Scheme A) | on spike (colptr+rowptr) | last_pre, last_post, initialized |
| STDPWeightDependent | exact decay per step (Scheme A) | on spike (colptr+rowptr) | last_pre, last_post, initialized |
| STDPTriplet | exact decay per step, 4 traces | on spike (colptr+rowptr) | r1, r2, o1, o2, last_pre, last_post |
| STDPMexicanHat | Euler per-dt | on spike (colptr+rowptr), touched clamp | initialized |
| STDPAntiSymmetric | Euler per-dt | on spike (colptr+rowptr), touched clamp | initialized |
| STDPSymmetric | Euler per-dt | on spike (colptr+rowptr), touched clamp | initialized |
| iSTDPRate | Euler per-dt | on spike (colptr+rowptr) | — |
| iSTDPPotential | Euler per-dt | on spike (colptr+rowptr) | — |
| vSTDPParameter | Euler per-dt | on pre-spike (colptr) | — |

---

## Recommendations

1. Done: event-driven weight updates for the trace rules; `Δpre`/`Δpost` removed.
2. Scheme B rules could use the same exact multiplicative decay (`_decay!`) instead of Euler;
   not done to keep their definitions unchanged.
3. Validation against analytic results, Brian2 and Auryn:
   `papers/JuliaSNN_publication/validation/stdp/` in the umbrella repo.
