"""
Regime sweep for §6 of the IMPA talk.

Compares four schedules for choosing the drop count m_0 as a function of the
regression width m:

  - constant:  m_0 = 1
  - alpha 1/4: m_0 = floor(m/4)
  - alpha 1/3: m_0 = floor(m/3)
  - alpha 1/2: m_0 = floor(m/2)

For each (regime, n, m) on the geometric/linear grid, partition the pool into
R = floor(n_total / n) disjoint chunks of size n, compute the OLS slope of
log(V̄) vs log(r) on the kept scales, and report mean ± std across replicas.

Inputs:  presentation18-05-2026/fractal_dim_pool.npz  (+ .meta.json)
Outputs: presentation18-05-2026/fig_regime_sweep.png
         presentation18-05-2026/regime_sweep.csv
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.stats import DF_THEORY, ols_slope
from lib.plotting import (
    REGIME_COLORS, apply_grid, footer, pool_footer_text,
)

HERE = Path(__file__).resolve().parent           # presentation/coding
ROOT = HERE.parent                                # presentation/
DATA_DIR = ROOT / "simulation_data"
IMG_DIR  = ROOT / "images"
POOL = DATA_DIR / "fractal_dim_pool.npz"
META = DATA_DIR / "fractal_dim_pool.meta.json"

# (label, m_0(m))  -- colour comes from REGIME_COLORS keyed by label
REGIMES = [
    ("const m0=1", lambda m: 1),
    ("alpha=1/4",  lambda m: m // 4),
    ("alpha=1/3",  lambda m: m // 3),
    ("alpha=1/2",  lambda m: m // 2),
]

NM_GRID = [(32, 3), (64, 4), (128, 5), (256, 6), (512, 7)]

# L3: each cell wants this many disjoint L2 replicas. R is the *target*; we
# fall back to floor(n_total / n) if the pool is too small (with a warning).
LOG_LOGPLOT_TRIALS = 64
R_MIN              = 3


def main():
    if not POOL.exists():
        raise SystemExit(f"missing {POOL}; run fractal_dim_pool_sim.py first")

    data    = np.load(POOL, allow_pickle=False)
    V_pool  = data["V_pool"]
    scales  = data["scales"].astype(np.int64)
    n_total = V_pool.shape[0]
    lattice = str(data["lattice"])
    n_scales_avail = len(scales)

    meta = json.loads(META.read_text())
    sec_per_trial = meta["elapsed_seconds"] / meta["n_trials"]

    print(f"pool: n_total={n_total} lattice={lattice}  N={meta['N']}")
    print(f"      scales={[int(s) for s in scales]}")
    print(f"      sec/trial = {sec_per_trial:.4f}\n")

    rows = []
    for label, m0_fn in REGIMES:
        color = REGIME_COLORS[label]
        print(f"=== regime: {label} ===")
        for n, m in NM_GRID:
            m_0 = int(m0_fn(m))
            if m_0 + m > n_scales_avail:
                print(f"  (n={n}, m={m})  m_0={m_0}  "
                      f"needs {m_0+m} scales, pool has {n_scales_avail} — SKIP")
                continue
            R_avail = n_total // n
            R = min(LOG_LOGPLOT_TRIALS, R_avail)
            if R < R_MIN:
                print(f"  (n={n}, m={m})  m_0={m_0}  only R={R} replicas — SKIP")
                continue
            if R < LOG_LOGPLOT_TRIALS:
                print(f"  (n={n}, m={m})  WARN: pool gives only "
                      f"{R_avail} chunks, target was {LOG_LOGPLOT_TRIALS}")
            use   = slice(m_0, m_0 + m)
            log_r = np.log(scales[use].astype(np.float64))
            sl    = np.empty(R)
            for r in range(R):
                chunk = V_pool[r * n : (r + 1) * n, use]
                mv    = chunk.mean(axis=0)
                if np.any(mv <= 0):
                    sl[r] = np.nan
                    continue
                sl[r] = ols_slope(log_r, np.log(mv))
            sl    = sl[~np.isnan(sl)]
            if len(sl) == 0:
                continue
            bias        = float(sl.mean() - DF_THEORY)
            std         = float(sl.std(ddof=1)) if len(sl) > 1 else 0.0
            rmse        = float(np.sqrt(((sl - DF_THEORY) ** 2).mean()))
            mean_L2_t   = float(n * sec_per_trial)          # one L2 replica
            total_t     = float(len(sl) * mean_L2_t)        # whole L3 cell
            rows.append({
                "regime": label,
                "n": n, "m": m, "m_0": m_0,
                "log_logPlot_trials": len(sl),
                "mean_slope": float(sl.mean()),
                "bias": bias, "std": std, "rmse": rmse,
                "mean_L2_time_s": mean_L2_t,
                "total_time_s":   total_t,
                "scales_used": [int(s) for s in scales[use]],
            })
            print(f"  (n={n:4d}, m={m})  m_0={m_0}  R={len(sl):3d}  "
                  f"mean={sl.mean():.4f}  bias={bias:+.4f}  std={std:.4f}  "
                  f"L2={mean_L2_t:7.2f}s  total={total_t:8.1f}s  "
                  f"scales={list(scales[use])}")
        print()

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = DATA_DIR / "regime_sweep.csv"
    fields = ["regime", "n", "m", "m_0", "log_logPlot_trials",
              "mean_slope", "bias", "std", "rmse",
              "mean_L2_time_s", "total_time_s", "sec_per_trial", "scales_used"]
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in fields if k in r},
                        "sec_per_trial": f"{sec_per_trial:.6f}",
                        "scales_used": r["scales_used"]})
    print(f"wrote {csv_path}")

    # ── plot: mean ± std vs n, one curve per regime ──────────────────────────
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for label, _ in REGIMES:
        color = REGIME_COLORS[label]
        sub = [r for r in rows if r["regime"] == label]
        if not sub:
            continue
        sub.sort(key=lambda r: r["n"])
        ns    = np.array([r["n"]          for r in sub])
        ms    = np.array([r["m"]          for r in sub])
        m0s   = np.array([r["m_0"]        for r in sub])
        means = np.array([r["mean_slope"] for r in sub])
        stds  = np.array([r["std"]        for r in sub])
        ax.errorbar(ns, means, yerr=stds, fmt="o-", capsize=3,
                    color=color, lw=1.5, label=label)
        for x, y, mi, m0i in zip(ns, means, ms, m0s):
            ax.annotate(f"m={mi},m$_0$={m0i}", (x, y),
                        xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=color)

    ax.axhline(DF_THEORY, color="#444", ls="--", lw=1.2,
               label=rf"$d_f = 91/48 \approx {DF_THEORY:.4f}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"trials per replica  $n$  (geometric)")
    ax.set_ylabel(r"$\hat d_f$")
    ax.set_title(rf"Regime sweep   (square, $N={meta['N']}$, "
                 rf"pool $={n_total}$, $\tau\approx{sec_per_trial:.3f}$ s/trial)")
    apply_grid(ax, log=True)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    footer(fig, pool_footer_text(meta, scales=scales))

    out = IMG_DIR / "fig_regime_sweep.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
