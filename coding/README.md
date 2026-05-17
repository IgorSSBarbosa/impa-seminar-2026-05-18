# `coding/` — pipeline & layout

All Python for the IMPA talk lives here. The folder is organised by
**role**, not by feature, so it is easy to tell what each file is
responsible for at a glance:

```
coding/
├── lib/           shared utilities (OLS, palettes, footers) — pure functions
├── simulators/    heavy BFS-based pool simulators (multi-hour wall time)
├── analysis/      everything that reads a pool NPZ and emits a CSV / PNG
├── benchmarks/    micro-benchmarks (kernel timings, no plotting)
├── demos/         interactive / animation scripts shown live in the talk
├── mean_field/    self-contained d = 2..8 mean-field cluster animation
└── scripts/       bash glue (overnight runs, viewer launchers)
```

## Module-level rules

- `lib/` may not import from anything else in `coding/`. Everything else
  may import `from lib.stats import ...` and `from lib.plotting import ...`.
- `analysis/` and `benchmarks/` scripts insert
  `coding/` into `sys.path` at top of file, so the `from lib.X` imports
  work no matter where you invoke them from.
- Output paths are computed from `Path(__file__)`, never from the
  caller's CWD. All write into the project-root `simulation_data/` and
  `images/` folders.
- `simulators/` write NPZ pools (`V_pool[trial, scale_index]`) + a
  `.meta.json` sidecar with seed, scales, elapsed time, method.
- `analysis/` reads the NPZ + meta, writes a CSV (recoverable summary)
  + a PNG (talk-grade figure).

## Pipeline

```
       simulators/                           analysis/
       -----------                           ---------
seed ► fractal_dim_pool_sim_par.py  ──┐
       (parallel BFS, 65k trials,    │   regime_sweep.py        ► fig_regime_sweep.png
       writes the canonical pool ►   ├─► masterpiece_plot.py    ► fig_masterpiece.png
       fractal_dim_pool.npz +        │   mae_vs_time_plot.py    ► fig_mae_vs_time.png
       fractal_dim_pool.meta.json)   │   error_vs_time.py       ► fig_error_vs_time.png
                                     │   scale_of_scales.py     ► fig_scale_of_scales.png
                                     └─► verify_error_vs_time.py (sanity check, no plot)

       fractal_dim_sim.py  ──────────────► plot_fractal_dim.py  ► fig_fractal_dim.png
       (per-lattice CSV variant — used                              (hex/sq/tri universality)
        for the universality plot)
```

The whole §7–§8 figure pack for the talk comes from
`fractal_dim_pool.npz` via the four `analysis/*_plot.py` scripts.

## Reproducing everything from scratch

```bash
# from repo root
bash coding/scripts/run_overnight.sh   # ~1 h on 12 cores, writes the pool
                                        # and runs the two summary analyses

# the remaining plots are seconds-level on top of the existing pool:
python3 coding/analysis/masterpiece_plot.py
python3 coding/analysis/mae_vs_time_plot.py
python3 coding/analysis/scale_of_scales.py
python3 coding/analysis/regime_sweep.py
python3 coding/analysis/error_vs_time.py
```

## Demos

See [`demos/README.md`](demos/README.md). Three live demos:

- `perc_interactive.py` — Matplotlib slider over $p$ on a square lattice.
- `perc_animate_merged.py` — synchronised hex/square/triangular sweep.
- `../mean_field/main.py` — zoom-out + rotating camera through d = 2..8.

## Adding a new analysis script

1. Drop it into `analysis/`.
2. Put `import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
   *before* any `from lib.X import Y`. (Other `analysis/` files use the
   same pattern — copy from there.)
3. Compute paths off `Path(__file__).resolve().parent.parent.parent`
   (= repo root) so the script runs from any CWD.
4. Write outputs into `<repo>/images/` and `<repo>/simulation_data/`,
   never alongside the script.
