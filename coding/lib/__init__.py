"""Shared utilities for the IMPA-talk simulation/plotting pipeline.

Submodules:
    stats     — numerical helpers (OLS slope, log-log fit, theory constants).
    plotting  — slide-grade plot conventions (palettes, footers, grid styling).

All scripts under `coding/` should import from here rather than re-defining
`ols_slope` / `DF_THEORY` / colour palettes locally.
"""
