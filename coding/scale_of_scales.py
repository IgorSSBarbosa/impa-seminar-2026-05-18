"""
Scale-of-scales study (Phase 1) for §6 of the IMPA talk.

Question: how does the RMSE of the log-log fractal-dimension estimator scale
with the trial budget n, when m_0 is fixed to a constant?

Method: load the per-trial V pool, partition it into R disjoint chunks of size
n (for each n on a log-spaced grid), run OLS on each chunk to get a replica
slope, then plot RMSE(n) vs n on log-log.

Expected shape: variance-dominated regime → slope -1/2; then plateau at the
finite-size bias floor (since m_0 = const ≠ const × m kills only the leading
correction term, leaving an O(rho^{-m_0}/m^2) bias that does not vanish with n).

Inputs:  presentation18-05-2026/fractal_dim_pool.npz
Outputs: presentation18-05-2026/fig_scale_of_scales.png
         presentation18-05-2026/scale_of_scales.csv
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.stats import DF_THEORY, ols_slope
from lib.plotting import apply_grid, footer, pool_footer_text

HERE = Path(__file__).resolve().parent           # presentation/coding
ROOT = HERE.parent                                # presentation/
DATA_DIR = ROOT / "simulation_data"
IMG_DIR  = ROOT / "images"
POOL = DATA_DIR / "fractal_dim_pool.npz"

M0 = 1                                     # drop the smallest scale (constant)
# (n, m) pairs: n geometric, m linear  →  m/n → 0
NM_GRID = [(64, 4), (128, 5), (256, 6), (512, 7), (1024, 8)]
R_MIN_REPLICAS = 3


def main():
    if not POOL.exists():
        raise SystemExit(f"missing {POOL}; run fractal_dim_pool_sim.py first")

    data = np.load(POOL, allow_pickle=False)
    V_pool = data["V_pool"]                # (n_total, n_scales)
    scales = data["scales"].astype(np.int64)
    n_total = V_pool.shape[0]
    lattice = str(data["lattice"])

    n_scales_avail = len(scales)
    print(f"pool: n_total={n_total} lattice={lattice}  scales={list(scales)}")

    rows = []
    for n, m in NM_GRID:
        if M0 + m > n_scales_avail:
            print(f"  (n={n}, m={m}): m_0+m={M0+m} exceeds pool scales — skipping")
            continue
        use   = slice(M0, M0 + m)
        log_r = np.log(scales[use].astype(np.float64))
        R = n_total // n
        if R < R_MIN_REPLICAS:
            print(f"  (n={n}, m={m}): only R={R} replicas — skipping")
            continue
        slopes = np.empty(R)
        for r in range(R):
            chunk = V_pool[r * n : (r + 1) * n, use]
            mean_V = chunk.mean(axis=0)
            if np.any(mean_V <= 0):
                slopes[r] = np.nan
                continue
            slopes[r] = ols_slope(log_r, np.log(mean_V))
        slopes = slopes[~np.isnan(slopes)]
        bias    = slopes.mean() - DF_THEORY
        rmse    = np.sqrt(((slopes - DF_THEORY) ** 2).mean())
        std     = slopes.std(ddof=1) if len(slopes) > 1 else 0.0
        rows.append((n, m, R, slopes.mean(), bias, std, rmse,
                     [int(s) for s in scales[use]]))
        print(f"  (n={n:4d}, m={m})  R={len(slopes):3d}  "
              f"mean_slope={slopes.mean():.4f}  bias={bias:+.4f}  "
              f"std={std:.4f}  rmse={rmse:.4f}  scales={list(scales[use])}")

    csv_path = DATA_DIR / "scale_of_scales.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["n", "m", "R", "mean_slope", "bias", "std", "rmse", "scales"])
        for n, m, R, ms, b, s, rm, sc in rows:
            w.writerow([n, m, R, ms, b, s, rm, sc])
    print(f"wrote {csv_path}")

    ns      = np.array([r[0] for r in rows])
    ms      = np.array([r[1] for r in rows])
    means   = np.array([r[3] for r in rows])
    biases  = np.array([r[4] for r in rows])
    stds    = np.array([r[5] for r in rows])
    rmses   = np.array([r[6] for r in rows])

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.errorbar(ns, means, yerr=stds, fmt="o-", capsize=3, color="#7b68ee",
                label=rf"$\hat d_f$ (mean $\pm$ std over $R$ replicas)")
    ax.axhline(DF_THEORY, color="#444", ls="--", lw=1.2,
               label=rf"$d_f = 91/48 \approx {DF_THEORY:.4f}$")
    for x, y, mi in zip(ns, means, ms):
        ax.annotate(f"m={mi}", (x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color="#555")
    ax.set_xscale("log")
    ax.set_xlabel("trials per replica  $n$  (geometric)")
    ax.set_ylabel(r"$\hat d_f$")
    ax.set_title(rf"Estimator $\hat d_f$ vs budget   "
                 rf"($m_0={M0}$, $m$ grows linearly with $\log n$)")
    apply_grid(ax, log=True)
    ax.legend(loc="best", fontsize=9)

    meta = json.loads((DATA_DIR / "fractal_dim_pool.meta.json").read_text())
    fig.suptitle(
        rf"Scale-of-scales (Phase 1): fixed $m_0 = {M0}$ — "
        rf"{lattice} lattice, $N = {meta['N']}$, pool $= {n_total}$ trials",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    footer(fig, pool_footer_text(meta, scales=scales))

    out = IMG_DIR / "fig_scale_of_scales.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
