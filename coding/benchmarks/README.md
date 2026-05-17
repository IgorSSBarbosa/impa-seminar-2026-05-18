# `benchmarks/` — micro-benchmarks

Kernel-level timings. No plotting; writes a CSV for offline inspection.

## `bench_l0.py`

Times the L0 BFS kernel (the inner loop that powers
`simulators/fractal_dim_pool_sim*.py`) across scales
$r \in \{2, 4, \dots, 1024\}$, both for the Numba-compiled clipped-arena
variant and the SciPy/Numpy fallback. Used to justify the choice of the
clipped-arena method over the legacy union-find variant in the 2026-05-16
performance pass.

```bash
python3 coding/benchmarks/bench_l0.py
```

Output: `simulation_data/bench_l0.csv` with columns
`backend, scale, trials, seconds, sites_per_sec`.

The current pool's `--method bfs` choice was decided based on this
benchmark — see also `notes/leath_simulator_plan.md` for the next
planned optimisation.
