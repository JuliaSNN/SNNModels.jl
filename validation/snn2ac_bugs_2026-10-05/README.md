# SNNModels bugs that affect SNN2AC_Optim: validation scripts, data and memo figures (2026-10-05)

Side commit of SNNModels.jl, annotated tag `snn2ac-bugs-2026-10-05`, based on the 1.9.0 release commit
(`e94ddca`, the one carrying the STP-order and Tripod fixes). It is not part of any release and not on `main`:
the registered versions are tree-hash bound, so nothing here touches them.

The scripts, small data files (JLD2) and figures document four SNNModels bugs that changed SNN2AC_Optim
results (iSTDPRate kernel, STP update order, Tripod soma, connectivity generator). The SNN2AC_Optim log
`CLAUDE/logs/2026-10-05.md` ("SNNModels bugs affecting SNN2AC (memo)") gives, per bug, the commit with the wrong
code, the fix commit or version, the code difference and the effect on SNN2AC. Total size of this folder: 21 MB.

## What each folder measures

| Folder | Measures | Output |
|---|---|---|
| `istdp_control/` | iSTDPRate kernel and structured-input test, fixed vs buggy (registered 1.8.2 vs the library control branch) | `result_*.jld2`, `structured_*.jld2`, `*.png` |
| `sweep_bugs/` | fixed vs unfixed comparison of the sweep bug fixes: Tripod, STP order, connectivity, membrane pair rule, other models (own `README.md`) | `data/`, `figures/` |
| `memo_snn2ac/` | the four memo figures: STDP Gerstner kernel, sparse generator, STP update order, Tripod soma | `data/`, `figures/` |

## Provenance

The scripts were written in the JuliaSNN project repository (clone without remote, folder
`papers/JuliaSNN_publication/validation/`) and copied to SNN2AC_Optim on 2026-11-17 (commit `bc062a1` there, branch
`bugfix_2026-10-05`, now deleted). Commits in the JuliaSNN project repository, newest first:

- `istdp_control/`: `2505fca`, `8a19f1b`, `14c153f`
- `sweep_bugs/`: `d4b1f9c` (2026-10-11), `629796c`, `2fbca76`, `7be80d9`, `4cdf657`
- `memo_snn2ac/`: `149f9d7` (complete), `2e3e7ae`, `d4d6570`

The scripts refer to themselves as `scripts/validation/snnmodels_bugs_2026-10-05/...` (their path in SNN2AC_Optim);
here the folder is `validation/snn2ac_bugs_2026-10-05/`. The scripts are unchanged.

## Old and new code

- Old code: SNNModels <= 1.8.4 (STP, Tripod) and <= 1.8.1 (iSTDP, connectivity). New code: 1.9.0 (1.8.2 for the
  iSTDP and connectivity fixes).
- In `sweep_bugs/run_all.sh` every variant is a Julia environment (Project.toml and Manifest.toml dev-linking
  SNNModels, SNNPlots, SNNUtils; the hash is the pin): `unfixed` = SNNModels commit `779f5a8` (`docs/sweep`),
  `unfixedIs`, `dtonly`, `glonly` (scratch commits listed in the script) and `fixed` = branch `fix/sweep-bugs`.
  These environments are not in this repository: set `ENVS=<dir>` to a folder holding them (the default path in
  `run_all.sh` points to a deleted scratch folder).
- The scripts record `779f5a8` as having the behaviour of the released 1.8.4. Checked on 2026-11-22: `779f5a8` is an
  ancestor of v1.9.0, 47 commits after v1.8.1 and not an ancestor of v1.8.4 (`dfe16de`), and `git diff 779f5a8 v1.8.4
  -- src` is not empty (77 files). The identity of behaviour for the four bugs was asserted when the scripts were
  written and is not a file-level identity of the sources.
- `istdp_control/istdp_cmp.jl` runs identically in the fixed (registered 1.8.2) and the buggy (library control branch)
  environments; the outputs are `result_fixed.jld2` and `result_buggy.jld2`.

## How to run

Every script takes `julia --project=<env>` (the environment named in its header). The plotting scripts need only
SNNPlots, CairoMakie and JLD2 and redraw the figures from the stored JLD2 files, so the figures can be regenerated
without the old environments, e.g. `julia --project=<env with SNNModels >= 1.9, SNNPlots, CairoMakie, JLD2>
validation/snn2ac_bugs_2026-10-05/memo_snn2ac/plot_memo_stp.jl`. Re-running the measurements needs the old and new
library versions above. Not re-run from this tag.

## Layout

```
istdp_control/   istdp_cmp.jl, istdp_structured_input.jl, plot_istdp_control.jl, result_*.jld2, structured_*.jld2, *.png
sweep_bugs/      measure_*.jl, summarize_*.jl, plot_*.jl, network_model.jl, run_all.sh, README.md, data/, figures/
memo_snn2ac/     memo_figures.jl, plot_memo_stp.jl, plot_memo_tripod.jl, data/, figures/
```
