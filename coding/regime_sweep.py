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

HERE = Path(__file__).resolve().parent           # presentation/coding
ROOT = HERE.parent                                # presentation/
DATA_DIR = ROOT / "simulation_data"
IMG_DIR  = ROOT / "images"
POOL = DATA_DIR / "fractal_dim_pool.npz"
META = DATA_DIR / "fractal_dim_pool.meta.json"

DF_THEORY = 91.0 / 48.0

# (label, color, m_0(m))
REGIMES = [
    ("const m0=1", "#888888", lambda m: 1),
    ("alpha=1/4",  "#1f9d55", lambda m: m // 4),
    ("alpha=1/3",  "#7b68ee", lambda m: m // 3),
    ("alpha=1/2",  "#cc3333", lambda m: m // 2),
]

NM_GRID = [(32, 3), (64, 4), (128, 5), (256, 6), (512, 7)]
R_MIN   = 3


def ols_slope(log_r, log_y):
    A = np.vstack([log_r, np.ones_like(log_r)]).T
    slope, _ = np.linalg.lstsq(A, log_y, rcond=None)[0]
    return slope


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
    for label, color, m0_fn in REGIMES:
        print(f"=== regime: {label} ===")
        for n, m in NM_GRID:
            m_0 = int(m0_fn(m))
            if m_0 + m > n_scales_avail:
                print(f"  (n={n}, m={m})  m_0={m_0}  "
                      f"needs {m_0+m} scales, pool has {n_scales_avail} — SKIP")
                continue
            R = n_total // n
            if R < R_MIN:
                print(f"  (n={n}, m={m})  m_0={m_0}  only R={R} replicas — SKIP")
                continue
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
            bias  = float(sl.mean() - DF_THEORY)
            std   = float(sl.std(ddof=1)) if len(sl) > 1 else 0.0
            rmse  = float(np.sqrt(((sl - DF_THEORY) ** 2).mean()))
            t_sec = float(n * sec_per_trial)
            rows.append({
                "regime": label, "color": color,
                "n": n, "m": m, "m_0": m_0, "R": len(sl),
                "mean_slope": float(sl.mean()),
                "bias": bias, "std": std, "rmse": rmse,
                "time_seconds": t_sec,
                "scales_used": [int(s) for s in scales[use]],
            })
            print(f"  (n={n:4d}, m={m})  m_0={m_0}  R={len(sl):3d}  "
                  f"mean={sl.mean():.4f}  bias={bias:+.4f}  std={std:.4f}  "
                  f"t={t_sec:7.2f}s  scales={list(scales[use])}")
        print()

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = DATA_DIR / "regime_sweep.csv"
    fields = ["regime", "n", "m", "m_0", "R", "mean_slope", "bias", "std",
              "rmse", "time_seconds", "sec_per_trial", "scales_used"]
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
    for label, color, _ in REGIMES:
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
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.text(0.5, 0.01,
             f"pool elapsed ≈ {meta['elapsed_seconds']}s   seed = {meta['seed']}   "
             f"pool scales = {[int(s) for s in scales]}",
             ha="center", va="bottom", fontsize=9, color="#444", fontstyle="italic")

    out = IMG_DIR / "fig_regime_sweep.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
