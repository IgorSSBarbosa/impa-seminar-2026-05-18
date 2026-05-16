"""Slide-grade plotting conventions for the IMPA seminar deck.

Centralised so every figure shares the same look (palette, grid, footer).
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt


# ── Regime colours / labels (used by regime_sweep.py and error_vs_time.py) ───

REGIME_COLORS: dict[str, str] = {
    "const m0=1": "#222222",
    "alpha=1/4":  "#1f9d55",
    "alpha=1/3":  "#7b68ee",
    "alpha=1/2":  "#cc3333",
    # Reserved for the L4 masterpiece plot:
    "log_rho":    "#ff8c1a",
    "min_arm":    "#2b6cb0",
}

REGIME_LATEX: dict[str, str] = {
    "const m0=1": r"$m_0 = 1$ (const)",
    "alpha=1/4":  r"$m_0 = \lfloor m/4 \rfloor$",
    "alpha=1/3":  r"$m_0 = \lfloor m/3 \rfloor$",
    "alpha=1/2":  r"$m_0 = \lfloor m/2 \rfloor$",
    "log_rho":    r"$m_0 = \lfloor \tfrac12 \log_\rho(nm^3) \rfloor$",
    "min_arm":    r"$m - m_0 = 2$",
}


# ── Lattice colours / labels (universality plot in plot_fractal_dim.py) ──────

LATTICE_ORDER = ["hexagonal", "square", "triangular"]

LATTICE_LABEL: dict[str, str] = {
    "hexagonal":  "Honeycomb (3-conn)",
    "square":     "Square (4-conn)",
    "triangular": "Triangular (6-conn)",
}

LATTICE_COLOR: dict[str, str] = {
    "hexagonal":  "#3cb371",
    "square":     "#7b68ee",
    "triangular": "#ff7f50",
}


# ── Grid / footer helpers ────────────────────────────────────────────────────

def apply_grid(ax, *, log: bool = False) -> None:
    """Standard dotted grid; pass `log=True` for log–log axes."""
    which = "both" if log else "major"
    ax.grid(True, which=which, ls=":", alpha=0.4)


def footer(fig, text: str, *, y: float = 0.01) -> None:
    """Italicised, low-contrast footer line — used for run metadata."""
    fig.text(0.5, y, text,
             ha="center", va="bottom", fontsize=9,
             color="#444", fontstyle="italic")


def pool_footer_text(meta: dict, *, scales: Iterable[int] | None = None) -> str:
    """Standard 'pool elapsed / trials / seed / scales' line.

    `meta` is the dict loaded from the `.meta.json` sidecar of a pool NPZ.
    `scales` may be passed explicitly when the meta dict doesn't carry it.
    """
    sc = list(scales) if scales is not None else meta.get("scales", [])
    parts = [
        f"pool elapsed ≈ {meta.get('elapsed_seconds','?')}s",
        f"trials = {meta.get('n_trials','?')}",
        f"seed = {meta.get('seed','?')}",
        f"scales = {[int(s) for s in sc]}",
    ]
    method = meta.get("method")
    if method:
        parts.insert(0, f"method = {method}")
    return "   ".join(parts)


def sim_footer_text(meta: dict) -> str:
    """Footer for figures generated from `fractal_dim_data.csv` (per-lattice
    aggregated sim, not the pool). Plays the same role as `pool_footer_text`
    but for the older CSV-driven plots in `plot_fractal_dim.py`."""
    sc = meta.get("scales", [])
    sc_range = (
        f"({min(sc)}…{max(sc)})" if sc else ""
    )
    return (
        f"N = {meta.get('N','?')}   "
        f"trials = {meta.get('n_trials','?')}   "
        f"scales = {meta.get('n_scales','?')}  {sc_range}   "
        f"elapsed ≈ {meta.get('elapsed_seconds','?')}s   "
        f"seed = {meta.get('seed','?')}"
    )
