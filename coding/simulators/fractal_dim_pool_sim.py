"""
Per-trial pool simulator for the scale-of-scales study (§6 of the IMPA talk).

Unlike `fractal_dim_sim.py` (which streams sums), this saves the full per-trial
V_t(r) matrix to disk so the analysis script can derive estimator replicas by
subsetting the pool (partition into R chunks of size n, OLS on each).

Output: NPZ at `presentation18-05-2026/simulation_data/fractal_dim_pool.npz`
  V_pool : int64, shape (n_total, n_scales)  — V_t(r_k)
  scales : int32, shape (n_scales,)
  p_c, N, seed, lattice — metadata
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import numba


_p = argparse.ArgumentParser(description="Per-trial V pool simulator (square lattice).")
_p.add_argument("--n", type=int, default=1024)
_p.add_argument("--trials", type=int, default=2000,
                help="Total trials in the pool (default 2000).")
_p.add_argument("--scales", type=int, nargs="+",
                default=[2, 4, 8, 16, 32, 64, 128, 256])
_p.add_argument("--seed", type=int, default=20260518)
_p.add_argument("--lattice", choices=["square", "hexagonal", "triangular"],
                default="square")
_p.add_argument("--out", type=Path,
                default=Path(__file__).resolve().parents[2]
                        / "simulation_data" / "fractal_dim_pool.npz")
args = _p.parse_args()

N        = args.n
N_TRIALS = args.trials
SCALES   = np.array(sorted(args.scales), dtype=np.int32)
SEED     = args.seed
LATTICE  = args.lattice
OUT_PATH = args.out

R_MAX = int(SCALES.max())
assert N >= 2 * R_MAX + 1, f"Grid N={N} too small for r_max={R_MAX}"

P_C = {"hexagonal": 0.6962, "square": 0.592746, "triangular": 0.5}


@numba.njit(cache=True)
def _find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


@numba.njit(cache=True)
def _union(par, rnk, a, b):
    ra, rb = _find(par, a), _find(par, b)
    if ra == rb:
        return
    if rnk[ra] < rnk[rb]:
        ra, rb = rb, ra
    par[rb] = ra
    if rnk[ra] == rnk[rb]:
        rnk[ra] += 1


@numba.njit(cache=True)
def _label_ids_sq(mask, out):
    n = mask.shape[0]
    par = np.arange(n * n, dtype=np.int32)
    rnk = np.zeros(n * n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                continue
            idx = i * n + j
            if j + 1 < n and mask[i, j + 1]:
                _union(par, rnk, idx, i * n + j + 1)
            if i + 1 < n and mask[i + 1, j]:
                _union(par, rnk, idx, (i + 1) * n + j)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                out[i, j] = _find(par, i * n + j) + 1
            else:
                out[i, j] = 0


@numba.njit(cache=True)
def _label_ids_tri(mask, out):
    n = mask.shape[0]
    par = np.arange(n * n, dtype=np.int32)
    rnk = np.zeros(n * n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                continue
            idx = i * n + j
            if j + 1 < n and mask[i, j + 1]:
                _union(par, rnk, idx, i * n + j + 1)
            if i + 1 < n and mask[i + 1, j]:
                _union(par, rnk, idx, (i + 1) * n + j)
            if i + 1 < n and j + 1 < n and mask[i + 1, j + 1]:
                _union(par, rnk, idx, (i + 1) * n + j + 1)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                out[i, j] = _find(par, i * n + j) + 1
            else:
                out[i, j] = 0


@numba.njit(cache=True)
def _label_ids_hex(mask, out):
    n = mask.shape[0]
    par = np.arange(n * n, dtype=np.int32)
    rnk = np.zeros(n * n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                continue
            idx = i * n + j
            if j + 1 < n and mask[i, j + 1]:
                _union(par, rnk, idx, i * n + j + 1)
            if (i + j) % 2 == 0 and i + 1 < n and mask[i + 1, j]:
                _union(par, rnk, idx, (i + 1) * n + j)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                out[i, j] = _find(par, i * n + j) + 1
            else:
                out[i, j] = 0


LABEL_FNS = {
    "square":     _label_ids_sq,
    "triangular": _label_ids_tri,
    "hexagonal":  _label_ids_hex,
}


def count_V(labels, ci, cj, scales):
    origin_id = labels[ci, cj]
    out = np.zeros(len(scales), dtype=np.int64)
    if origin_id == 0:
        return out
    for k, r in enumerate(scales):
        sub = labels[ci - r : ci + r + 1, cj - r : cj + r + 1]
        out[k] = int((sub == origin_id).sum())
    return out


def main():
    print(f"\nPool simulation for {LATTICE} lattice")
    print(f"  N            : {N}")
    print(f"  total trials : {N_TRIALS}")
    print(f"  scales       : {list(SCALES)}")
    print(f"  output       : {OUT_PATH}\n")

    label_fn = LABEL_FNS[LATTICE]
    p_c = P_C[LATTICE]

    print("Compiling Numba kernel...", end=" ", flush=True)
    warm_mask = np.ones((8, 8), dtype=np.bool_)
    warm_out  = np.zeros((8, 8), dtype=np.int32)
    label_fn(warm_mask, warm_out)
    print("done.")

    rng = np.random.default_rng(SEED)
    ci, cj = N // 2, N // 2

    V_pool = np.zeros((N_TRIALS, len(SCALES)), dtype=np.int64)
    labels_buf = np.zeros((N, N), dtype=np.int32)

    t0 = time.perf_counter()
    for t in range(N_TRIALS):
        U = rng.random((N, N), dtype=np.float32)
        mask = U < p_c
        label_fn(mask, labels_buf)
        V_pool[t] = count_V(labels_buf, ci, cj, SCALES)

        if (t + 1) % max(N_TRIALS // 20, 1) == 0 or (t + 1) == N_TRIALS:
            elapsed = time.perf_counter() - t0
            eta = elapsed * (N_TRIALS - (t + 1)) / max(t + 1, 1)
            sys.stdout.write(
                f"\r  trial {t+1:5d}/{N_TRIALS}  "
                f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s"
            )
            sys.stdout.flush()
    print()

    total_elapsed = time.perf_counter() - t0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        V_pool=V_pool,
        scales=SCALES,
        lattice=np.array(LATTICE),
        N=np.int64(N),
        p_c=np.float64(p_c),
        seed=np.int64(SEED),
        elapsed_seconds=np.float64(total_elapsed),
    )
    print(f"  wrote {OUT_PATH}  (elapsed {total_elapsed:.1f}s)")

    meta_path = OUT_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "lattice": LATTICE, "N": N, "n_trials": N_TRIALS,
        "scales": [int(s) for s in SCALES], "seed": SEED,
        "p_c": p_c, "elapsed_seconds": round(total_elapsed, 2),
    }, indent=2))
    print(f"  wrote {meta_path}")


if __name__ == "__main__":
    main()
