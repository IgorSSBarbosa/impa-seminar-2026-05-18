# `lib/` — shared helpers

Pure-function utilities used by everything under `coding/analysis/`,
`coding/benchmarks/`, and `coding/simulators/`. `lib/` may not import
from any other folder in `coding/`.

## `stats.py`

```python
from lib.stats import DF_THEORY, ols_slope, loglog_fit
```

- `DF_THEORY` — universal 2D site-percolation fractal dimension, $91/48$.
- `ols_slope(log_x, log_y) -> float` — OLS slope when inputs are
  already on log scale.
- `loglog_fit(x, y) -> (slope, intercept_log)` — OLS fit of
  $\log y = \text{slope}\cdot \log x + \text{intercept}$ for raw
  positive inputs. Recover the prefactor with `np.exp(intercept)`.

## `plotting.py`

Slide-grade conventions so every figure shares the same look:

- `REGIME_COLORS`, `REGIME_LATEX` — per-$m_0$-schedule colours / LaTeX
  labels (used by `regime_sweep.py`, `masterpiece_plot.py`,
  `mae_vs_time_plot.py`, `error_vs_time.py`).
- `LATTICE_COLOR`, `LATTICE_LABEL`, `LATTICE_ORDER` — for the
  universality plot in `plot_fractal_dim.py`.
- `apply_grid(ax, log=False)` — dotted major-grid; pass `log=True` for
  log-log axes.
- `footer(fig, text, y=0.01)` — italicised low-contrast footer line.
- `pool_footer_text(meta)` / `sim_footer_text(meta)` — standard
  "pool elapsed / trials / seed / scales" or "N / trials / scales /
  elapsed / seed" footer text from a pool / per-lattice meta dict.
- `format_duration(seconds)` — render `5667` as `1h34m27s`.
