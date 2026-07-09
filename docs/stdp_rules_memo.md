# STDP Rules — Inconsistencies Memo

Two incompatible trace-update schemes coexist in the codebase. This memo documents what each rule does and where the schemes diverge.

---

## Scheme A — Event-driven (exact exponential decay)

Used by: `STDPGerstner`, `STDPConfavreux2025`

Traces are **not** updated every dt. Instead, on each spike event the trace is decayed exactly from the last spike time:

```julia
if fireJ[j]
    tpre[j] = tpre[j] * exp(-(t - last_pre[j]) / τpre) + A_pre
    last_pre[j] = t
end
Δpre[j] = t > last_pre[j] ? tpre[j] * exp(-(t - last_pre[j]) / τpre) : 0f0
```

- Mathematically exact: no Euler integration error.
- Requires storing `last_pre` / `last_post` vectors (extra memory).
- `Δpre` / `Δpost` are ephemeral "current value" buffers recomputed each step; they are not persistent state.
- Weight update runs in a threaded loop over all synapses (`Threads.@threads`), checking `fireI[i]` / `fireJ[j]`.

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

## Inconsistency 1 — Mixed weight-update triggers

`STDPGerstner` / `STDPConfavreux2025` update weights for **every synapse** at every dt (threaded loop over `eachindex(W)`), even when no spike occurred (they check `fireI[i]` / `fireJ[j]` inside the loop). This is safe but wasteful when spike rates are low.

Scheme B rules scan only the spiking neurons via row/col pointer, which is O(Npre·p) on a spike, O(0) otherwise. This is the standard efficient approach.

---

## Inconsistency 2 — Δ buffers in Scheme A

`STDPGerstner` / `STDPConfavreux2025` maintain `Δpre` / `Δpost` vectors in `STDPVariables` as ephemeral per-step values. These are not traces — they are recalculated every dt and should not be read outside of `plasticity!`. Scheme B rules have no such ephemeral state. The naming is misleading (`Δpre` sounds like a weight change, not a trace value).

---

## Inconsistency 3 — Weight update scope

- Scheme A: all synapses scanned each dt, weight updated only when `fireI` or `fireJ`.
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

## Summary table

| Rule | Trace update | Weight scan | Extra state |
|---|---|---|---|
| STDPGerstner | event-driven exact | all W each dt | last_pre, last_post, Δpre, Δpost |
| STDPConfavreux2025 | event-driven exact | all W each dt | last_pre, last_post, Δpre, Δpost |
| STDPMexicanHat | Euler per-dt | on pre-spike (colptr+rowptr) | — |
| STDPAntiSymmetric | Euler per-dt | on spike (colptr+rowptr) | — |
| STDPSymmetric | Euler per-dt | on spike (colptr+rowptr) | — |
| iSTDPRate | Euler per-dt | on spike (colptr+rowptr) | — |
| iSTDPPotential | Euler per-dt | on spike (colptr+rowptr) | — |
| vSTDPParameter | Euler per-dt | on pre-spike (colptr) | — |

---

## Recommendations

1. **Low priority:** Refactor `STDPGerstner` / `STDPConfavreux2025` to use colptr/rowptr weight update (avoids O(|W|) scan each dt). Only matters at low firing rates with large, dense weight matrices.
2. **Rename `Δpre`/`Δpost`** to `tpre_now`/`tpost_now` to clarify these are current trace values, not weight deltas.
3. **No functional change needed** for Scheme B rules — Euler error is negligible at `dt=0.1ms`.
