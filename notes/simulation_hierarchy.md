# Simulation hierarchy

## Persistent decisions (do not lose)

1. **End-to-end re-run is mandatory** once the L0–L4 hierarchy is finalised.
   Every figure that appears on a slide must be regenerated from the final
   pipeline in a single sweep — no mixing-and-matching of artifacts from
   intermediate iterations. The canonical recipe lives in
   `coding/run_overnight.sh`.
2. **Shared statistical / plotting helpers live in their own module(s).**
   OLS slope, weighted regression, axis-formatting, footers, colour palette,
   etc., live in `coding/lib/` (Python package) so the regime sweep, the
   verifier, and the masterpiece plot all import the same functions. No
   copy-paste of `ols_slope` across files.
3. **L0 method chosen: clipped BFS.** ~20× faster than the legacy full-grid
   union-find at N=4096. Selectable via `--method {bfs,uf,old}` for
   cross-checks. See `coding/bench_l0.py` and the L0 row in the table below.


The pipeline that feeds the §6 "error vs simulation time" and "masterpiece"
plots is layered. Each level is a loop over the level below it. Naming and
semantics fixed here so all scripts and docs stay consistent.

| Level | What it produces | Notation | Lives in |
|---|---|---|---|
| **L0** | one trial → one realisation of $V(r)$ for every kept scale | one row of `V_pool` | per-trial sampler (clipped BFS or full Bernoulli + union-find) |
| **L1** | n trials averaged → one point $\bar V(r_k)$ on the log-log plot | one entry of `mv = chunk.mean(axis=0)` | inside `regime_sweep.py` |
| **L2** | m log-log points → one OLS estimate $\hat d_f$ — **this is the unit whose wall-clock time we report** | one element of `slopes[r]` | inside `regime_sweep.py` |
| **L3** | `log_logPlot_trials` replicas of L2 → mean $\hat d_f$, std, mean time per L2 → one point on `error_vs_time` | one row of `regime_sweep.csv` | one row of the CSV |
| **L4** | sweep a budget multiplier `B ∈ {1, 2, 4, 8}` → one curve per regime in the "masterpiece" plot | the plot itself | `error_vs_time.py` / masterpiece script |

## Open design decisions (deferred — discuss when we get to that level)

- **L3 α** — fix `log_logPlot_trials` at one value (e.g. 64) across the grid,
  vs. let it equal `n_total // n` (currently the case).
- **L3 β** — report `time_seconds = n · τ_avg` (deterministic, one global τ)
  vs. per-L2 wall-clock averaged inside L3 (matches BFS variance).
- **L4 γ** — which axis does B grow on: n only, m only, or both? Do all four
  regimes share the same `(n(B), m(B))` and only differ in `m_0(·)`?
- **L4 δ** — confirm the four regime schedules:
  1. `m_0 = 1`           (constant)
  2. `m_0 = ⌊α m⌋`       (fixed fraction, α = ?)
  3. `m_0 = ½ log_ρ(n m³)` (theoretical optimum from the article)
  4. `m - m_0 = const`   (only the largest scales; needs ≥ 2 to fit OLS)
