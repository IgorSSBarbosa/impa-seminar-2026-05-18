# `simulators/` — heavy BFS pool generators

The three scripts here produce the pre-computed pools that the
`coding/analysis/` plots consume. They are the only files in the
codebase whose wall-time is measured in hours.

| script | purpose | output |
| --- | --- | --- |
| `fractal_dim_pool_sim.py` | Single-threaded clipped-arena BFS. Reference implementation; kept for cross-checks against the parallel variant. | `simulation_data/fractal_dim_pool.npz` (+ `.meta.json`) |
| `fractal_dim_pool_sim_par.py` | Multi-process clipped-arena BFS. Same numerics as the single-threaded one (same seed → same pool, modulo trial order). Use this for production. | same |
| `fractal_dim_sim.py` | Older per-lattice CSV-driven simulator (hex / square / triangular). Output feeds `analysis/plot_fractal_dim.py` for the universality figure. | `simulation_data/fractal_dim_data.csv` (+ `.meta.json`) |

## Output schema (pool variants)

`fractal_dim_pool.npz` contains two arrays:

```
V_pool : int64 [n_trials, len(scales)]    — cluster volume at each scale
scales : int64 [len(scales)]              — the L∞-radii used
```

`fractal_dim_pool.meta.json` carries seed / scales / `elapsed_seconds`
/ `bfs_cpu_seconds` / `n_trials` / method (`"bfs"` for clipped-arena,
`"uf"` for the legacy union-find that the talk no longer uses).

## Reproducing the canonical pool

```bash
# from repo root
bash coding/scripts/run_overnight.sh
```

Equivalent direct call (without the regime-sweep / error-vs-time
follow-up steps that `run_overnight.sh` chains on):

```bash
python3 coding/simulators/fractal_dim_pool_sim_par.py \
    --method bfs --trials 65536 \
    --scales 2 4 8 16 32 64 128 256 512 1024 2048 \
    --seed 20260518 \
    --out simulation_data/fractal_dim_pool.npz
```

## Future work

See `notes/leath_simulator_plan.md` for a planned Leath-style lazy-growth
rewrite that removes the arena memory bottleneck (would unlock
$r_{\max} \geq 2^{14}$ on a 31 GB machine).
