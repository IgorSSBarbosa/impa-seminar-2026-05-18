"""
Error-vs-simulation-time plot for §6 of the IMPA talk.

Reads `simulation_data/regime_sweep.csv` (produced by `regime_sweep.py`) and
plots |bias| = |mean(hat d_f) - d_f| against the simulation time per replica
(`time_seconds` column == n * sec_per_trial) on log-log, with one curve per
regime. Each point is annotated with its (m, m_0). The OLS slope of each
regime's log|bias|-log time curve is printed and shown in the legend.

"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `from lib.*` works

from lib.stats import DF_THEORY, loglog_fit
from lib.plotting import (
    REGIME_COLORS, REGIME_LATEX, apply_grid, footer, pool_footer_text,
)

HERE = Path(__file__).resolve().parent           # presentation/coding/analysis
ROOT = HERE.parent.parent                         # presentation/
DATA_DIR = ROOT / "simulation_data"
IMG_DIR  = ROOT / "images"
CSV  = DATA_DIR / "regime_sweep.csv"
META = DATA_DIR / "fractal_dim_pool.meta.json"

# (regime_key, marker, plot_zorder)
# Colours come from REGIME_COLORS, labels from REGIME_LATEX (lib.plotting).
REGIME_STYLE = [
    ("alpha=1/2",  "^", 3),
    ("const m0=1", "X", 10),    # plotted last, on top
]
LABEL_OFFSET = {
    "const m0=1": (-6, -16),
    "alpha=1/2":  (+6, +8),
}

CI_Z = 1.96  # 95% Gaussian half-width


def load_rows():
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "regime":             r["regime"],
                "n":                  int(r["n"]),
                "m":                  int(r["m"]),
                "m_0":                int(r["m_0"]),
                "log_logPlot_trials": int(r["log_logPlot_trials"]),
                "mean_slope":         float(r["mean_slope"]),
                "bias":               float(r["bias"]),
                "std":                float(r["std"]),
                "rmse":               float(r["rmse"]),
                "mean_L2_time_s":     float(r["mean_L2_time_s"]),
                "total_time_s":       float(r["total_time_s"]),
                "sec_per_trial":      float(r["sec_per_trial"]),
            })
    return rows


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit(f"{CSV} is empty; run regime_sweep.py first")
    meta = json.loads(META.read_text())
    sec_per_trial = rows[0]["sec_per_trial"]

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    slope_by_regime = {}
    for regime, marker, zorder in REGIME_STYLE:
        color = REGIME_COLORS[regime]
        sub = sorted([r for r in rows if r["regime"] == regime],
                     key=lambda r: r["mean_L2_time_s"])
        if not sub:
            continue
        times = np.array([r["mean_L2_time_s"]      for r in sub])
        absb  = np.array([abs(r["bias"])           for r in sub])
        stds  = np.array([r["std"]                 for r in sub])
        ntrials = np.array([r["log_logPlot_trials"] for r in sub])
        if np.any(absb <= 0):
            continue
        # OLS slope on log–log (only if ≥2 points)
        if len(sub) >= 2:
            slope, _ = loglog_fit(times, absb)
            slope_by_regime[regime] = slope
            label = REGIME_LATEX[regime] + rf"   slope = {slope:+.2f}"
        else:
            label = REGIME_LATEX[regime]

        # 95% CI half-width on hat_d_f - d_f: ±z * std / sqrt(N_trials).
        ci_half = CI_Z * stds / np.sqrt(ntrials)
        # Asymmetric bars on log axis: clip lower bar so it doesn't reach zero.
        lower_err = np.minimum(ci_half, 0.95 * absb)
        upper_err = ci_half
        yerr = np.vstack([lower_err, upper_err])

        ax.errorbar(times, absb, yerr=yerr,
                    linestyle="-", marker=marker, color=color,
                    lw=1.6, ms=9, mew=1.6,
                    mfc=(color if marker != "X" else "none"),
                    mec=color, zorder=zorder, label=label,
                    capsize=4, elinewidth=1.0, ecolor=color)

        dx, dy = LABEL_OFFSET[regime]
        for r, t, b in zip(sub, times, absb):
            ax.annotate(f"m={r['m']},m$_0$={r['m_0']}", (t, b),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color=color, zorder=zorder + 1)

    print("|bias| vs time, OLS slopes on log–log:")
    for k, v in slope_by_regime.items():
        print(f"  {k}: {v:+.4f}")
    if slope_by_regime:
        best = min(slope_by_regime, key=slope_by_regime.get)
        print(f"  steepest decay: {best}  (slope {slope_by_regime[best]:+.4f})")

    ax.set_xlabel(r"simulation time per replica  $t \approx n \cdot \tau$  (s, log)")
    ax.set_ylabel(r"$|\,\hat d_f - d_f\,|$  (log)")
    ax.set_title(rf"Error vs simulation time   "
                 rf"(square, $N={meta['N']}$, "
                 rf"$\tau\approx{sec_per_trial:.3f}$ s/trial)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    apply_grid(ax, log=True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    footer(fig, pool_footer_text(meta))

    out = IMG_DIR / "fig_error_vs_time.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
