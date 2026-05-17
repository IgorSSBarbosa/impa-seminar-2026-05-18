"""
L4 masterpiece plot for §6 of the IMPA talk.

Compares four schedules for $m_0(m, n)$ on a single log-log plot:

  R1  m_0 = 1                                 (baseline, drops nothing)
  R2  m_0 = floor(m/2)                        (linear)
  R3  m_0 = ceil( log_rho(n m^3) / 2 )        (matches the CLT-optimal threshold)
  R4  m_0 = m - 2                             (drop almost everything; only keep
                                              the largest two)

For each (regime, m, n) cell, we partition the pool into R = floor(n_total / n)
disjoint blocks of size n, fit the OLS slope on log(V̄) vs log(rho^k) over the
kept scales k = m_0+1, ..., m_0+m, and record:

  RMSE^2 = bias^2 + variance      (across blocks)

The x-axis is a *theoretical* budget B(n, m, m_0) := n * tau_{m_0+m}, where
the per-trial BFS cost is modelled as

  tau_k = rho^(k * d_f)     (cluster-size scaling at criticality)

— this is the cost model the asymptotic CLT uses, and it's also what governs
the bias/variance trade-off in §4. Switching to an arena-bound cost rho^{2k}
would shift curves rightward but not change the relative ordering.

Layout: one panel per n (the experimentalist's budget axis), with the four
regimes overlaid in each panel. This lets a viewer compare regimes head-to-head
at a fixed trial count.

Inputs : simulation_data/fractal_dim_pool.npz (+ .meta.json)
Outputs: simulation_data/masterpiece.csv
         images/fig_masterpiece.png
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

from lib.stats import DF_THEORY, ols_slope
from lib.plotting import (
    REGIME_COLORS, REGIME_LATEX, apply_grid, footer, format_duration,
    pool_footer_text,
)


HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent
DATA_DIR = ROOT / "simulation_data"
IMG_DIR  = ROOT / "images"
POOL     = DATA_DIR / "fractal_dim_pool.npz"
META     = DATA_DIR / "fractal_dim_pool.meta.json"


# ── Regime definitions ──────────────────────────────────────────────────────
# Each entry: (key, m_0(m, n), marker). The key indexes into REGIME_COLORS /
# REGIME_LATEX in lib.plotting.
RHO = 2  # scale ratio of the pool

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

# Sweep grid
M_VALUES = [3, 4, 5, 6, 7, 8]
N_VALUES = [16, 64, 256, 1024, 4096]

CI_Z = 1.96  # 95% Gaussian half-width on the mean


def budget(n: int, m_0: int, m: int, df: float = DF_THEORY) -> float:
    """Theoretical simulation budget: n trials * cost of the largest scale used."""
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

                mean_sl = float(sl.mean())
                bias    = mean_sl - DF_THEORY
                std     = float(sl.std(ddof=1))
                rmse    = float(np.sqrt(((sl - DF_THEORY) ** 2).mean()))
                B       = budget(n, m_0, m)
                rows.append({
                    "regime": key, "n": n, "m": m, "m_0": m_0,
                    "blocks": len(sl), "dropped_blocks": bad,
                    "mean_slope": mean_sl, "bias": bias,
                    "std": std, "rmse": rmse, "budget": B,
                    "k_max": m_0 + m,
                })
                print(f"  n={n:5d}  m={m}  m_0={m_0:2d}  R={len(sl):4d}  "
                      f"mean={mean_sl:.4f}  bias={bias:+.4f}  "
                      f"rmse={rmse:.4f}  B={B:.3e}")
        print()

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = DATA_DIR / "masterpiece.csv"
    fields = ["regime", "n", "m", "m_0", "k_max", "blocks", "dropped_blocks",
              "mean_slope", "bias", "std", "rmse", "budget"]
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}")

    # ── Plot: panels by n; regimes overlaid in each panel ───────────────────
    # One panel per n ∈ N_VALUES. Within each panel, four curves (one per
    # regime), each parameterised by m. The x-axis is the theoretical budget
    # B(n, m, m_0(m, n)) — monotone-increasing in m for every regime.
    n_panels = len(N_VALUES)
    ncols    = 3
    nrows    = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13.5, 4.2 * nrows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    panels = dict(zip(N_VALUES, axes_flat))
    # Hide spare panels (e.g. the 6th in a 2×3 grid when we only have 5 n's).
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    # Common y-range across all panels for fair visual comparison.
    all_rmse = np.array([r["rmse"] for r in rows]) if rows else np.array([1.0])
    y_lo, y_hi = float(all_rmse.min()) * 0.85, float(all_rmse.max()) * 1.15

    slope_by_regime: dict[str, float] = {}

    for n in N_VALUES:
        ax = panels[n]
        for key, _m0_fn, marker in REGIMES:
            sub = sorted([r for r in rows if r["regime"] == key and r["n"] == n],
                         key=lambda r: r["m"])
            if len(sub) < 2:
                continue
            B    = np.array([r["budget"] for r in sub])
            rmse = np.array([r["rmse"]   for r in sub])
            Rs   = np.array([r["blocks"] for r in sub])
            stds = np.array([r["std"]    for r in sub])

            # 95% CI on hat_gamma, clipped on the lower side for log axis safety
            ci_half   = CI_Z * stds / np.sqrt(Rs)
            lower_err = np.minimum(ci_half, 0.95 * rmse)
            upper_err = ci_half
            yerr      = np.vstack([lower_err, upper_err])

            color = REGIME_COLORS[key]
            ax.errorbar(B, rmse, yerr=yerr,
                        linestyle="-", marker=marker,
                        color=color,
                        lw=1.4, ms=7, mew=1.2,
                        mfc=color, mec=color,
                        label=REGIME_LATEX[key],
                        capsize=3, elinewidth=0.9, ecolor=color)

            # annotate each point with its m
            for r, b, y in zip(sub, B, rmse):
                ax.annotate(f"m={r['m']}", (b, y),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7, color=color, alpha=0.85)

        # Number of blocks available at this n (constant per panel).
        R_avail = n_total // n
        ax.set_title(rf"$n = {n}$    (R = {R_avail} blocks)", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        apply_grid(ax, log=True)
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9,
                  ncols=1, columnspacing=1.0)

    # Per-regime overall cloud slope (across all (n,m) cells), for the printout.
    for key, _m0_fn, _marker in REGIMES:
        sub_all = [r for r in rows if r["regime"] == key]
        if len(sub_all) >= 2:
            B_all    = np.array([r["budget"] for r in sub_all])
            rmse_all = np.array([r["rmse"]   for r in sub_all])
            try:
                slope_by_regime[key] = ols_slope(np.log(B_all), np.log(rmse_all))
            except Exception:
                pass

    print("\nRMSE vs B, OLS slopes on log–log (whole cloud per regime):")
    for k, v in slope_by_regime.items():
        print(f"  {k:12s}: {v:+.4f}")
    if slope_by_regime:
        best = min(slope_by_regime, key=slope_by_regime.get)
        print(f"  steepest cloud-slope: {best}  ({slope_by_regime[best]:+.4f})")

    # Shared axis labels.
    fig.supxlabel(r"theoretical budget  $B(n, m, m_0) = n \cdot \rho^{(m_0+m)\, d_f}$  (log)",
                  fontsize=11, y=0.045)
    fig.supylabel(r"$\mathrm{RMSE}(\hat\gamma) = \sqrt{\mathrm{bias}^2 + \mathrm{var}}$  (log)",
                  fontsize=11)
    fig.suptitle(
        rf"L4 masterpiece — RMSE vs budget; one panel per $n$, four $m_0$-schedules overlaid   "
        rf"(square, $N={meta['N']}$, pool ${{=}}{n_total}$)",
        fontsize=12,
    )

    # Layout: leave ~10% at bottom for the supxlabel + footer, ~5% at top
    # for the suptitle, and a small left margin for the supylabel.
    fig.tight_layout(rect=(0.02, 0.10, 1, 0.95))
    footer(fig, pool_footer_text(meta), y=0.005)

    out = IMG_DIR / "fig_masterpiece.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
