"""MAE-vs-time companion to `masterpiece_plot.py`.

Same four $m_0$-schedules, same panel-per-$n$ layout, but reports

    y-axis :  <|gamma_hat - d_f|>   (mean absolute error across blocks)
    x-axis :  estimated wall-time in seconds, log scale

The wall-time axis is the theoretical budget B(n, m, m_0) = n * rho^((m_0+m)*d_f)
rescaled by a single calibration constant derived from the pool meta:

    c  =  elapsed_seconds_pool / (n_trials_pool * rho^(k_max_pool * d_f))

so that running the entire pool would, in budget units, cost
n_trials_pool * rho^(k_max_pool * d_f) and, in seconds, elapsed_seconds_pool.

This is not per-trial timing (we don't have that) — it's an apples-to-apples
"what would this (n, m, m_0) cell cost if you ran an independent BFS just for
it, on the same hardware as the pool" number.

Original budget-axis / RMSE plot stays around as a backup: see
images/fig_masterpiece.png.

Inputs : simulation_data/fractal_dim_pool.npz (+ .meta.json)
Outputs: simulation_data/mae_vs_time.csv
         images/fig_mae_vs_time.png
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `from lib.*` works

from lib.stats import DF_THEORY, ols_slope
from lib.plotting import (
    REGIME_COLORS, REGIME_LATEX, apply_grid, footer, format_duration,
    pool_footer_text,
)


HERE     = Path(__file__).resolve().parent          # presentation/coding/analysis
ROOT     = HERE.parent.parent                       # presentation/
DATA_DIR = ROOT / "simulation_data"
IMG_DIR  = ROOT / "images"
POOL     = DATA_DIR / "fractal_dim_pool.npz"
META     = DATA_DIR / "fractal_dim_pool.meta.json"


RHO = 2


def _m0_log_rho(m, n):
    return max(1, math.ceil(0.5 * math.log2(n * m**3)))


def _m0_min_arm(m, n):
    return max(1, m - 2)


REGIMES = [
    ("const m0=1", lambda m, n: 1,                       "o"),
    ("alpha=1/2",  lambda m, n: max(1, m // 2),          "^"),
    ("log_rho",    _m0_log_rho,                          "s"),
    ("min_arm",    _m0_min_arm,                          "D"),
]

M_VALUES = [3, 4, 5, 6, 7, 8]
N_VALUES = [16, 64, 256, 1024]

CI_Z = 1.96


def budget(n: int, m_0: int, m: int, df: float = DF_THEORY) -> float:
    k_max = m_0 + m
    return float(n * RHO ** (k_max * df))


def main():
    if not POOL.exists():
        raise SystemExit(f"missing {POOL}; run fractal_dim_pool_sim_par.py first")

    data    = np.load(POOL, allow_pickle=False)
    V_pool  = data["V_pool"]
    scales  = data["scales"].astype(np.int64)
    n_total = V_pool.shape[0]
    n_scales_avail = len(scales)

    meta = json.loads(META.read_text())
    print(f"pool: n_total={n_total}  N={meta['N']}  "
          f"scales={[int(s) for s in scales]}")
    print(f"      pool elapsed {format_duration(meta.get('elapsed_seconds'))}\n")

    # ── Wall-time calibration ────────────────────────────────────────────────
    # The pool's largest scale fixes its per-trial cost in budget units.
    k_max_pool = int(round(math.log(int(scales[-1]), RHO)))
    budget_pool = n_total * RHO ** (k_max_pool * DF_THEORY)
    elapsed_pool = float(meta["elapsed_seconds"])
    c_sec_per_bu = elapsed_pool / budget_pool
    print(f"calibration: pool k_max={k_max_pool}  budget_pool={budget_pool:.3e}")
    print(f"             c = {c_sec_per_bu:.3e}  seconds per budget unit\n")

    rows: list[dict] = []
    for key, m0_fn, _marker in REGIMES:
        print(f"=== regime: {key} ===")
        for n in N_VALUES:
            for m in M_VALUES:
                m_0 = int(m0_fn(m, n))
                if m_0 + m > n_scales_avail:
                    print(f"  n={n:5d}  m={m}  m_0={m_0}  needs {m_0+m} scales — SKIP")
                    continue
                R = n_total // n
                if R < 3:
                    print(f"  n={n:5d}  m={m}  m_0={m_0}  only R={R} blocks — SKIP")
                    continue

                use   = slice(m_0, m_0 + m)
                log_r = np.log(scales[use].astype(np.float64))

                sl = np.empty(R, dtype=np.float64)
                bad = 0
                for r in range(R):
                    chunk = V_pool[r * n : (r + 1) * n, use]
                    mv    = chunk.mean(axis=0)
                    if np.any(mv <= 0):
                        sl[r] = np.nan
                        bad += 1
                        continue
                    sl[r] = ols_slope(log_r, np.log(mv))
                sl = sl[~np.isnan(sl)]
                if len(sl) < 2:
                    continue

                abs_err = np.abs(sl - DF_THEORY)
                mae     = float(abs_err.mean())
                se_mae  = float(abs_err.std(ddof=1) / np.sqrt(len(sl)))
                B       = budget(n, m_0, m)
                wall_s  = c_sec_per_bu * B
                rows.append({
                    "regime": key, "n": n, "m": m, "m_0": m_0,
                    "blocks": len(sl), "dropped_blocks": bad,
                    "mae": mae, "se_mae": se_mae,
                    "budget": B, "wall_seconds": wall_s,
                    "k_max": m_0 + m,
                })
                print(f"  n={n:5d}  m={m}  m_0={m_0:2d}  R={len(sl):4d}  "
                      f"mae={mae:.4f}±{CI_Z*se_mae:.4f}  "
                      f"wall≈{format_duration(wall_s)}")
        print()

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = DATA_DIR / "mae_vs_time.csv"
    fields = ["regime", "n", "m", "m_0", "k_max", "blocks", "dropped_blocks",
              "mae", "se_mae", "budget", "wall_seconds"]
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}")

    # ── Plot ────────────────────────────────────────────────────────────────
    n_panels = len(N_VALUES)
    ncols    = 2
    nrows    = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(11.0, 4.4 * nrows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    panels = dict(zip(N_VALUES, axes_flat))
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    all_mae = np.array([r["mae"] for r in rows]) if rows else np.array([1.0])
    y_lo, y_hi = float(all_mae.min()) * 0.85, float(all_mae.max()) * 1.15

    for n in N_VALUES:
        ax = panels[n]
        for key, _m0_fn, marker in REGIMES:
            sub = sorted([r for r in rows if r["regime"] == key and r["n"] == n],
                         key=lambda r: r["m"])
            if len(sub) < 2:
                continue
            x    = np.array([r["wall_seconds"] for r in sub])
            mae  = np.array([r["mae"]          for r in sub])
            se   = np.array([r["se_mae"]       for r in sub])

            ci_half   = CI_Z * se
            lower_err = np.minimum(ci_half, 0.5 * mae)
            upper_err = ci_half
            yerr      = np.vstack([lower_err, upper_err])

            color = REGIME_COLORS[key]
            ax.errorbar(x, mae, yerr=yerr,
                        linestyle="-", marker=marker,
                        color=color,
                        lw=1.4, ms=7, mew=1.2,
                        mfc=color, mec=color,
                        label=REGIME_LATEX[key],
                        capsize=4, elinewidth=1.1, capthick=1.1,
                        ecolor=color)

            for r, b, y in zip(sub, x, mae):
                ax.annotate(f"m={r['m']}", (b, y),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7, color=color, alpha=0.85)

        R_avail = n_total // n
        ax.set_title(rf"$n = {n}$    (R = {R_avail} blocks)", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        apply_grid(ax, log=True)
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9,
                  ncols=1, columnspacing=1.0)

    fig.supxlabel(r"simulation time in seconds  (log scale)",
                  fontsize=11, y=0.045)
    fig.supylabel(r"$\langle\,|\hat d_f - d_f|\,\rangle$  (log scale)",
                  fontsize=11)
    fig.suptitle(
        rf"$\langle\,|\hat d_f - d_f|\,\rangle$ vs simulation time; "
        rf"one panel per $n$, four $m_0$-schedules overlaid   "
        rf"(square, $N={meta['N']}$, pool ${{=}}{n_total}$)",
        fontsize=12,
    )

    fig.tight_layout(rect=(0.02, 0.10, 1, 0.95))
    footer(fig,
           pool_footer_text(meta)
           + f"   wall-time calibration: c = {c_sec_per_bu:.2e} s/budget-unit",
           y=0.005)

    out = IMG_DIR / "fig_mae_vs_time.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
