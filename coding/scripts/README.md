# `scripts/` — bash glue

Thin shell wrappers; no logic of their own beyond chaining the Python
entry points and computing the right paths.

## `run_overnight.sh`

End-to-end "generate the canonical pool + the two summary plots" run.
~1 h on 12 cores for the BFS pool (65 536 trials, scales 2..2048),
seconds for the downstream analyses.

```bash
bash coding/scripts/run_overnight.sh
```

Chains:
1. `coding/simulators/fractal_dim_pool_sim.py` with seed `20260518`
   → `simulation_data/fractal_dim_pool.{npz,meta.json}`
2. `coding/analysis/regime_sweep.py` → `images/fig_regime_sweep.png`
3. `coding/analysis/error_vs_time.py` → `images/fig_error_vs_time.png`

Stdout is teed to `<repo>/overnight.log`.

## `run_gif.sh`

Opens `images/lattices_merged.gif` fullscreen via `eog`. Used live
during the talk to show the merged three-lattice animation produced by
`demos/perc_animate_merged.py`.

```bash
bash coding/scripts/run_gif.sh
```
