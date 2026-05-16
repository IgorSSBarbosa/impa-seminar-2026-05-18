"""
Fractal-dimension simulation at p_c on three lattices.

For each trial t = 1..N_TRIALS:
  1. Generate a single uniform random matrix U_t ∈ [0,1]^{N×N}.
  2. For each lattice (hex / square / triangular), build mask = U_t < p_c(lattice).
  3. Label the connected components (union-find on the lattice graph).
  4. Find the cluster id of the central site (ci, cj).
     If unoccupied (probability 1 - p_c), then V(r) = 0 for every r.
  5. For each scale r in SCALES, count cluster sites inside the L_∞ box
     of half-side r centered at (ci, cj).  That is V_t^(lattice)(r).

The U is shared across lattices within a trial (visual sync / variance reduction);
trials themselves are independent (independent U_t).

Output: CSV at `presentation18-05-2026/fractal_dim_data.csv` with columns
  lattice, r, mean_V, var_V, n_trials.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import numba


# ── CLI ────────────────────────────────────────────────────────────────────────

_p = argparse.ArgumentParser(description="Fractal-dim simulation at p_c.")
_p.add_argument("--n", type=int, default=1024,
                help="Grid side N (default 1024). Must exceed 2*max(scales)+1.")
_p.add_argument("--trials", type=int, default=500,
                help="Trials per lattice (default 500).")
_p.add_argument("--scales", type=int, nargs="+",
                default=[2, 4, 8, 16, 32, 64, 128, 256],
                help="Box half-sides r (default geometric base-2 up to 256).")
_p.add_argument("--seed", type=int, default=20260518)
_p.add_argument("--out", type=Path,
                default=Path(__file__).resolve().parents[2]
                        / "simulation_data" / "fractal_dim_data.csv")
args = _p.parse_args()

N         = args.n
N_TRIALS  = args.trials
SCALES    = np.array(sorted(args.scales), dtype=np.int32)
SEED      = args.seed
OUT_PATH  = args.out

R_MAX = int(SCALES.max())
assert N >= 2 * R_MAX + 1, f"Grid N={N} too small for r_max={R_MAX}"

P_C = {
    "hexagonal":  0.6962,
    "square":     0.592746,
    "triangular": 0.5,
}


# ── Union-find with cluster-id output ─────────────────────────────────────────

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


def count_V(labels: np.ndarray, ci: int, cj: int, scales: np.ndarray) -> np.ndarray:
    """V[k] = # sites in origin's cluster within L_inf <= scales[k] around (ci,cj).
    Returns zeros if (ci,cj) is unoccupied."""
    origin_id = labels[ci, cj]
    out = np.zeros(len(scales), dtype=np.int64)
    if origin_id == 0:
        return out
    for k, r in enumerate(scales):
        sub = labels[ci - r : ci + r + 1, cj - r : cj + r + 1]
        out[k] = int((sub == origin_id).sum())
    return out


def main():
    print(f"\nFractal-dimension simulation")
    print(f"  N            : {N}")
    print(f"  trials/latt  : {N_TRIALS}")
    print(f"  scales       : {list(SCALES)}")
    print(f"  output       : {OUT_PATH}\n")

    # Warm up Numba JIT.
    print("Compiling Numba kernels...", end=" ", flush=True)
    warm_mask = np.ones((8, 8), dtype=np.bool_)
    warm_out  = np.zeros((8, 8), dtype=np.int32)
    for fn in LABEL_FNS.values():
        fn(warm_mask, warm_out)
    print("done.")

    rng = np.random.default_rng(SEED)
    ci, cj = N // 2, N // 2

    # Running sums per lattice for mean/var via Welford-equivalent (sum, sumsq).
    sum_V    = {k: np.zeros(len(SCALES), dtype=np.float64) for k in LABEL_FNS}
    sum_V2   = {k: np.zeros(len(SCALES), dtype=np.float64) for k in LABEL_FNS}

    labels_buf = np.zeros((N, N), dtype=np.int32)

    t0 = time.perf_counter()
    for t in range(1, N_TRIALS + 1):
        U = rng.random((N, N), dtype=np.float32)
        for lattice, fn in LABEL_FNS.items():
            mask = U < P_C[lattice]
            fn(mask, labels_buf)
            V = count_V(labels_buf, ci, cj, SCALES).astype(np.float64)
            sum_V[lattice]  += V
            sum_V2[lattice] += V * V

        if t % max(N_TRIALS // 20, 1) == 0 or t == N_TRIALS:
            elapsed = time.perf_counter() - t0
            eta = elapsed * (N_TRIALS - t) / max(t, 1)
            sys.stdout.write(
                f"\r  trial {t:5d}/{N_TRIALS}  "
                f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s"
            )
            sys.stdout.flush()
    print()

    total_elapsed = time.perf_counter() - t0

    # Compute mean / variance.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write("lattice,r,mean_V,var_V,n_trials\n")
        for lattice in ("hexagonal", "square", "triangular"):
            mean = sum_V[lattice] / N_TRIALS
            var = np.maximum(sum_V2[lattice] / N_TRIALS - mean ** 2, 0.0)
            for k, r in enumerate(SCALES):
                f.write(f"{lattice},{int(r)},{mean[k]:.6f},{var[k]:.6f},{N_TRIALS}\n")
    print(f"  wrote {OUT_PATH}")

    meta_path = OUT_PATH.with_suffix(".meta.json")
    meta = {
        "N": N,
        "n_trials": N_TRIALS,
        "scales": [int(s) for s in SCALES],
        "n_scales": len(SCALES),
        "seed": SEED,
        "p_c": P_C,
        "elapsed_seconds": round(total_elapsed, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {meta_path}  (elapsed {total_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
