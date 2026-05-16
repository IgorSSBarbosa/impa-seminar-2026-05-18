"""
Error-vs-simulation-time plot for §6 of the IMPA talk.

Reads `simulation_data/regime_sweep.csv` (produced by `regime_sweep.py`) and
plots |bias| = |mean(hat d_f) - d_f| against the simulation time per replica
(`time_seconds` column == n * sec_per_trial) on log-log, with one curve per
regime. Each point is annotated with its (m, m_0). The OLS slope of each
regime's log|bias|-log time curve is printed and shown in the legend.

Note on overlapping points: for several values of m in the grid the constant
regime m_0 = 1 numerically coincides with alpha=1/4 (since floor(m/4) = 1 for
m in {4,...,7}). To keep both visible we use a distinct marker per regime and
plot the constant regime last with higher z-order. Markers also get a slight
horizontal jitter to disambiguate identical points.
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
CSV  = DATA_DIR / "regime_sweep.csv"
META = DATA_DIR / "fractal_dim_pool.meta.json"

DF_THEORY = 91.0 / 48.0

# (regime_key, color, marker, jitter_log_x, plot_zorder)
REGIME_STYLE = [
    ("alpha=1/4",  "#1f9d55", "o",  0.00, 3),
    ("alpha=1/3",  "#7b68ee", "s", +0.03, 3),
    ("alpha=1/2",  "#cc3333", "^", +0.06, 3),
    ("const m0=1", "#222222", "X", -0.03, 10),    # plotted last, on top
]
LABELS = {
    "const m0=1": r"$m_0 = 1$ (const)",
    "alpha=1/4":  r"$m_0 = \lfloor m/4 \rfloor$",
    "alpha=1/3":  r"$m_0 = \lfloor m/3 \rfloor$",
    "alpha=1/2":  r"$m_0 = \lfloor m/2 \rfloor$",
}
LABEL_OFFSET = {
    "const m0=1": (-6, -16),
    "alpha=1/4":  (+6, +8),
    "alpha=1/3":  (+6, -16),
    "alpha=1/2":  (+6, +8),
}


def load_rows():
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "regime":        r["regime"],
                "n":             int(r["n"]),
                "m":             int(r["m"]),
                "m_0":           int(r["m_0"]),
                "R":             int(r["R"]),
                "mean_slope":    float(r["mean_slope"]),
                "bias":          float(r["bias"]),
                "std":           float(r["std"]),
                "rmse":          float(r["rmse"]),
                "time_seconds":  float(r["time_seconds"]),
                "sec_per_trial": float(r["sec_per_trial"]),
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
    for regime, color, marker, jitter, zorder in REGIME_STYLE:
        sub = sorted([r for r in rows if r["regime"] == regime],
                     key=lambda r: r["time_seconds"])
        if not sub:
            continue
        times = np.array([r["time_seconds"] for r in sub])
        absb  = np.array([abs(r["bias"])    for r in sub])
        if np.any(absb <= 0):
            continue
        # OLS slope on log–log (only if ≥2 points)
        if len(sub) >= 2:
            slope = np.polyfit(np.log(times), np.log(absb), 1)[0]
            slope_by_regime[regime] = slope
            label = LABELS[regime] + rf"   slope = {slope:+.2f}"
        else:
            label = LABELS[regime]

        # apply small log-x jitter so coincident points don't fully overlap
        times_plot = times * (10.0 ** jitter)
        ax.loglog(times_plot, absb,
                  linestyle="-", marker=marker, color=color,
                  lw=1.6, ms=9, mew=1.6,
                  mfc=(color if marker != "X" else "none"),
                  mec=color, zorder=zorder, label=label)

        dx, dy = LABEL_OFFSET[regime]
        for r, t, b in zip(sub, times_plot, absb):
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
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.text(0.5, 0.01,
             f"pool elapsed ≈ {meta['elapsed_seconds']}s   "
             f"pool trials = {meta['n_trials']}   "
             f"seed = {meta['seed']}   pool scales = {meta['scales']}",
             ha="center", va="bottom", fontsize=9, color="#444", fontstyle="italic")

    out = IMG_DIR / "fig_error_vs_time.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
