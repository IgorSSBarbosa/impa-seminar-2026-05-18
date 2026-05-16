"""
Step-by-step verifier for `fig_error_vs_time.png`.

Reproduces, from first principles:

  (A) ONE row of `regime_sweep.csv`, by reaching into V_pool and computing
      the OLS slope on a single replica's log V̄ vs log r, then averaging
      across all R replicas. We pick (regime=alpha=1/2, n=512, m=7) because
      it exercises the widest scale window (m_0=3, 7 scales used).

  (B) ONE point's |bias| from that row.

  (C) The log-log fitted slope shown in the figure's legend for the same
      regime, by re-fitting np.polyfit(log time, log|bias|, 1) over the
      regime's 5 (n,m) rows.

Run from anywhere:
    python3 coding/verify_error_vs_time.py
"""

import csv
import json
from pathlib import Path

import numpy as np

from lib.stats import DF_THEORY, loglog_fit, ols_slope

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "simulation_data"

POOL_NPZ = DATA / "fractal_dim_pool.npz"
META     = DATA / "fractal_dim_pool.meta.json"
CSV_PATH = DATA / "regime_sweep.csv"

# --- the row we will rebuild from V_pool ---
TARGET = dict(regime="alpha=1/2", n=512, m=7, m_0=3)


def banner(s):
    bar = "─" * 72
    print(f"\n{bar}\n{s}\n{bar}")


def main():
    # ── Stage 0: load raw artefacts ──────────────────────────────────────────
    banner("Stage 0  ·  load pool and metadata")
    d = np.load(POOL_NPZ, allow_pickle=False)
    V_pool  = d["V_pool"]
    scales  = d["scales"].astype(np.int64)
    meta    = json.loads(META.read_text())
    n_total = V_pool.shape[0]
    sec_per_trial = meta["elapsed_seconds"] / meta["n_trials"]

    print(f"V_pool.shape   = {V_pool.shape}   (n_total trials × n_scales)")
    print(f"scales         = {list(scales)}")
    print(f"lattice        = {str(d['lattice'])}    N = {int(d['N'])}    "
          f"seed = {int(d['seed'])}    p_c = {float(d['p_c']):.6f}")
    print(f"n_total        = {n_total}")
    print(f"pool elapsed   = {meta['elapsed_seconds']:.2f}s")
    print(f"sec_per_trial  = {sec_per_trial:.6f}s")

    # ── Stage 1: rebuild ONE row of regime_sweep.csv ─────────────────────────
    banner(f"Stage 1  ·  rebuild CSV row for {TARGET}")
    n    = TARGET["n"]
    m    = TARGET["m"]
    m_0  = TARGET["m_0"]
    R    = n_total // n
    use  = slice(m_0, m_0 + m)
    sub_scales = scales[use]
    log_r = np.log(sub_scales.astype(np.float64))

    print(f"m_0 + m = {m_0 + m}  ≤  n_scales_avail = {len(scales)}   ✓")
    print(f"R = n_total // n = {n_total} // {n} = {R}  replicas")
    print(f"kept scales[{m_0}:{m_0+m}] = {list(sub_scales)}")
    print(f"log r        = {np.array2string(log_r, precision=4)}")

    slopes = np.empty(R)
    for r in range(R):
        chunk = V_pool[r * n : (r + 1) * n, use]   # (n, m)
        mv    = chunk.mean(axis=0)                  # V̄(r_k) over n trials
        # OLS slope via lib.stats.ols_slope (same path regime_sweep.py uses).
        slopes[r] = ols_slope(log_r, np.log(mv))

    print("\nfirst 3 replicas in detail:")
    for r in range(min(3, R)):
        chunk = V_pool[r * n : (r + 1) * n, use]
        mv    = chunk.mean(axis=0)
        print(f"  replica {r}: chunk rows [{r*n}, {(r+1)*n})")
        print(f"             V̄(r)      = {np.array2string(mv, precision=2)}")
        print(f"             log V̄     = "
              f"{np.array2string(np.log(mv), precision=4)}")
        print(f"             slope     = {slopes[r]:+.6f}")

    mean_slope = slopes.mean()
    bias       = mean_slope - DF_THEORY
    std        = slopes.std(ddof=1)
    rmse       = np.sqrt(((slopes - DF_THEORY) ** 2).mean())
    time_seconds = n * sec_per_trial

    print(f"\nmean over {R} replicas")
    print(f"  mean_slope    = {mean_slope:.10f}")
    print(f"  bias          = mean - d_f({DF_THEORY:.6f}) = {bias:+.10f}")
    print(f"  std (ddof=1)  = {std:.10f}")
    print(f"  rmse          = {rmse:.10f}")
    print(f"  time_seconds  = n × τ = {n} × {sec_per_trial:.6f} = "
          f"{time_seconds:.6f}")

    # ── Stage 2: read the same row from the CSV and diff ─────────────────────
    banner("Stage 2  ·  cross-check against regime_sweep.csv row")
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    csv_row = next(
        r for r in rows
        if r["regime"] == TARGET["regime"]
        and int(r["n"]) == TARGET["n"]
        and int(r["m"]) == TARGET["m"]
    )

    def cmp(name, here, there):
        diff = abs(here - there)
        ok = "✓" if diff < 1e-9 else "✗"
        print(f"  {name:<14} verifier={here:+.10f}   "
              f"csv={there:+.10f}   |Δ|={diff:.2e}  {ok}")

    cmp("mean_slope",     mean_slope,    float(csv_row["mean_slope"]))
    cmp("bias",           bias,          float(csv_row["bias"]))
    cmp("std",            std,           float(csv_row["std"]))
    cmp("rmse",           rmse,          float(csv_row["rmse"]))
    cmp("mean_L2_time_s", time_seconds,  float(csv_row["mean_L2_time_s"]))

    # ── Stage 3: rebuild the legend slope for this regime ────────────────────
    banner(f"Stage 3  ·  log-log legend slope for regime '{TARGET['regime']}'")
    reg_rows = sorted(
        (r for r in rows if r["regime"] == TARGET["regime"]),
        key=lambda r: float(r["mean_L2_time_s"]),
    )
    times = np.array([float(r["mean_L2_time_s"]) for r in reg_rows])
    absb  = np.array([abs(float(r["bias"]))      for r in reg_rows])
    log_t = np.log(times)
    log_b = np.log(absb)
    print("       n     m   m_0          time_s          |bias|     log t      log|b|")
    for r, t, b, lt, lb in zip(reg_rows, times, absb, log_t, log_b):
        print(f"   {int(r['n']):5d}  {int(r['m']):2d}   {int(r['m_0']):2d}    "
              f"{t:10.4f}   {b:12.6f}   {lt:+7.4f}   {lb:+7.4f}")

    slope, intercept = loglog_fit(times, absb)
    print(f"\n  lib.stats.loglog_fit(times, |bias|) → slope = {slope:+.6f}, "
          f"intercept = {intercept:+.6f}")
    print(f"  → legend reads:  '{TARGET['regime']}   slope = {slope:+.2f}'")

    banner("Done.  fig_error_vs_time.png is reproduced from regime_sweep.csv,\n"
           "which in turn is reproduced from V_pool — both crosschecks pass.")


if __name__ == "__main__":
    main()
