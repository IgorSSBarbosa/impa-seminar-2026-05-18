"""Numerical helpers for the log-log fractal-dimension estimator.

Two OLS-on-log-log entry points, depending on whether the caller has the
inputs already log-transformed:

    ols_slope(log_x, log_y)   -> float                   (slope only)
    loglog_fit(x, y)          -> (slope, intercept_log)  (raw values in)

Both are thin wrappers around `np.linalg.lstsq`. They share the implementation
so a future weighted variant only needs to touch one place.
"""

from __future__ import annotations

import numpy as np

# Universal 2D site-percolation fractal dimension (Stauffer & Aharony).
DF_THEORY: float = 91.0 / 48.0


def _fit_log(log_x: np.ndarray, log_y: np.ndarray) -> tuple[float, float]:
    """OLS fit of `log_y = slope * log_x + intercept`. Returns (slope, intercept)."""
    log_x = np.asarray(log_x, dtype=np.float64)
    log_y = np.asarray(log_y, dtype=np.float64)
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    sol, *_ = np.linalg.lstsq(A, log_y, rcond=None)
    slope, intercept = sol[0], sol[1]
    return float(slope), float(intercept)


def ols_slope(log_x, log_y) -> float:
    """OLS slope of `log_y` vs `log_x` (inputs are already on log scale)."""
    slope, _ = _fit_log(log_x, log_y)
    return slope


def loglog_fit(x, y) -> tuple[float, float]:
    """OLS fit of log(y) vs log(x) for raw positive inputs.

    Returns (slope, intercept_in_log_space). Recover the prefactor via
    `a0 = np.exp(intercept)`.
    """
    return _fit_log(np.log(np.asarray(x, dtype=np.float64)),
                    np.log(np.asarray(y, dtype=np.float64)))
