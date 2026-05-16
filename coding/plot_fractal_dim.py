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

HERE = Path(__file__).resolve().parent           # presentation/coding
ROOT = HERE.parent                                # presentation/
DATA_DIR = ROOT / "simulation_data"
IMG_DIR  = ROOT / "images"
DATA = DATA_DIR / "fractal_dim_data.csv"
META = DATA_DIR / "fractal_dim_data.meta.json"

LATTICE_ORDER = ["hexagonal", "square", "triangular"]
LATTICE_LABEL = {
    "hexagonal":  "Honeycomb (3-conn)",
    "square":     "Square (4-conn)",
    "triangular": "Triangular (6-conn)",
}
LATTICE_COLOR = {
    "hexagonal":  "#3cb371",
    "square":     "#7b68ee",
    "triangular": "#ff7f50",
}
DF_THEORY = 91.0 / 48.0  # universal 2D site-percolation fractal dimension


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
    x = np.log(r)
    y = np.log(mean_V)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, log_a0 = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, log_a0


def metadata_footer(meta: dict) -> str:
    if not meta:
        return ""
    return (
        f"N = {meta.get('N','?')}   "
        f"trials = {meta.get('n_trials','?')}   "
        f"scales = {meta.get('n_scales','?')}  "
        f"({min(meta.get('scales',[0]))}…{max(meta.get('scales',[0]))})   "
        f"elapsed ≈ {meta.get('elapsed_seconds','?')}s   "
        f"seed = {meta.get('seed','?')}"
    )


def _draw_one_panel(ax, data: dict, lattices, *, log_scale: bool,
                    show_fits: bool, ylabel: str):
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
                    fmt="o", markersize=6, capsize=2.5,
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
    ax.grid(True, which="both", ls=":", alpha=0.4)
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
    fig.text(0.5, 0.015, metadata_footer(meta),
             ha="center", va="bottom", fontsize=9,
             color="#444", fontstyle="italic")

    out = IMG_DIR / "fig_universality.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_running_example(data: dict, meta: dict, lattice: str = "square") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ylab = r"$\overline{V}(r)$"
    _draw_one_panel(axes[0], data, [lattice],
                    log_scale=False, show_fits=False, ylabel=ylab)
    _draw_one_panel(axes[1], data, [lattice],
                    log_scale=True, show_fits=True, ylabel=ylab)
    fig.suptitle(
        f"{LATTICE_LABEL[lattice]} — cluster volume vs. box size at $p_c$"
        f"   (asymptotic theory: $d_f \\approx {DF_THEORY:.4f}$)",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.5, 0.015, metadata_footer(meta),
             ha="center", va="bottom", fontsize=9,
             color="#444", fontstyle="italic")

    out = IMG_DIR / "fig_running_example.png"
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


if __name__ == "__main__":
    main()
