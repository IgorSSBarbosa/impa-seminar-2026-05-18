"""
Plots for §2 (universality) and §3 (running example) of the IMPA seminar.

Reads `fractal_dim_data.csv` (+ sidecar `fractal_dim_data.meta.json`) produced by
`uniform_coupling/fractal_dim_sim.py`, and writes side-by-side linear / log-log
plots with a metadata footer.

  - fig_universality.png       all three lattices  (linear | log-log)
  - fig_running_example.png    square lattice alone  (linear | log-log)
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
    LATTICE_ORDER, LATTICE_LABEL, LATTICE_COLOR,
    apply_grid, footer, sim_footer_text,
)

HERE = Path(__file__).resolve().parent           # presentation/coding/analysis
ROOT = HERE.parent.parent                         # presentation/
DATA_DIR = ROOT / "simulation_data"
IMG_DIR  = ROOT / "images"
DATA = DATA_DIR / "fractal_dim_data.csv"
META = DATA_DIR / "fractal_dim_data.meta.json"


def load_data() -> dict:
    rows = {lat: {"r": [], "mean_V": [], "var_V": [], "n_trials": []}
            for lat in LATTICE_ORDER}
    with DATA.open() as f:
        for rec in csv.DictReader(f):
            lat = rec["lattice"]
            rows[lat]["r"].append(float(rec["r"]))
            rows[lat]["mean_V"].append(float(rec["mean_V"]))
            rows[lat]["var_V"].append(float(rec["var_V"]))
            rows[lat]["n_trials"].append(float(rec["n_trials"]))
    out = {}
    for lat, cols in rows.items():
        idx = np.argsort(cols["r"])
        out[lat] = {k: np.asarray(v)[idx] for k, v in cols.items()}
    return out


def load_meta() -> dict:
    return json.loads(META.read_text()) if META.exists() else {}


def ols_loglog(r: np.ndarray, mean_V: np.ndarray):
    """Thin wrapper for backward compatibility: returns (slope, log_a0)."""
    return loglog_fit(r, mean_V)


def _draw_one_panel(ax, data: dict, lattices, *, log_scale: bool,
                    show_fits: bool, ylabel: str,
                    marker_scale: float = 1.0):
    ms      = 6.0 * marker_scale
    capsize = 2.5 * marker_scale
    for lat in lattices:
        sub = data[lat]
        r  = sub["r"]
        mV = sub["mean_V"]
        se = np.sqrt(sub["var_V"] / sub["n_trials"])
        slope, log_a0 = ols_loglog(r, mV)
        label = LATTICE_LABEL[lat]
        if show_fits:
            label += f"   slope = {slope:.3f}"
        ax.errorbar(r, mV, yerr=se,
                    fmt="o", markersize=ms, capsize=capsize,
                    color=LATTICE_COLOR[lat], ecolor=LATTICE_COLOR[lat],
                    label=label, zorder=3)
        if show_fits:
            rr = np.array([r.min(), r.max()])
            ax.plot(rr, np.exp(log_a0) * rr ** slope,
                    "-", color=LATTICE_COLOR[lat], alpha=0.65,
                    lw=1.4, zorder=2)
    if log_scale:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$r$  (log)")
        ax.set_ylabel(ylabel + "  (log)")
        ax.set_title("log–log scale", fontsize=11)
    else:
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(ylabel)
        ax.set_title("linear scale", fontsize=11)
    apply_grid(ax, log=log_scale)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)


def plot_universality(data: dict, meta: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ylab = r"$\overline{V}(r)$"
    _draw_one_panel(axes[0], data, LATTICE_ORDER,
                    log_scale=False, show_fits=False, ylabel=ylab)
    _draw_one_panel(axes[1], data, LATTICE_ORDER,
                    log_scale=True, show_fits=True, ylabel=ylab)
    fig.suptitle(
        r"Fractal dimension at $p_c$ — universality across lattices"
        f"   (theory: $d_f = 91/48 \\approx {DF_THEORY:.4f}$)",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    footer(fig, sim_footer_text(meta), y=0.015)

    out = IMG_DIR / "fig_universality.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_running_example(data: dict, meta: dict, lattice: str = "square",
                         *, log_marker_scale: float = 1.0,
                         filename: str = "fig_running_example.png") -> Path:
    """Right panel (log–log) marker size scaled by `log_marker_scale`.

    The small-dots variant (scale = 1/3) is intended to illustrate the
    "big dots make linear plots look more linear" pedagogical point in §3.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ylab = r"$\overline{V}(r)$"
    _draw_one_panel(axes[0], data, [lattice],
                    log_scale=False, show_fits=False, ylabel=ylab)
    _draw_one_panel(axes[1], data, [lattice],
                    log_scale=True, show_fits=True, ylabel=ylab,
                    marker_scale=log_marker_scale)
    fig.suptitle(
        f"{LATTICE_LABEL[lattice]} — cluster volume vs. box size at $p_c$"
        f"   (asymptotic theory: $d_f \\approx {DF_THEORY:.4f}$)",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    footer(fig, sim_footer_text(meta), y=0.015)

    out = IMG_DIR / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run fractal_dim_sim.py first")
    data = load_data()
    meta = load_meta()
    print(f"  wrote {plot_universality(data, meta)}")
    print(f"  wrote {plot_running_example(data, meta, lattice='square')}")
    # Small-dots variant for the "big dots ⇒ deceptively-linear" point in §3.
    print(f"  wrote {plot_running_example(data, meta, lattice='square', log_marker_scale=1/3, filename='fig_running_example_small_dots.png')}")


if __name__ == "__main__":
    main()
