"""
L0 benchmark: which single-trial method generates V(r) fastest?

Three candidate L0 trial functions, all producing the *box-clipped* V(r):
  V(r) = # sites in the L_inf box of half-side r centered at the origin,
         in the connected component of the origin restricted to the same arena.

  old_4096 : N=4096 grid, full Bernoulli + union-find labels everything,
             count V via slicing.  (current production code's cost.)
  box_uf   : N=2*r_max+1=2049 grid, box-only Bernoulli + union-find on the box,
             count V via slicing.
  box_bfs  : N=2*r_max+1=2049 grid, box-only Bernoulli + BFS from origin
             (4-neighbor, clipped to box), tally visited sites by L_inf
             distance to origin, prefix-sum to get V(r).

box_uf and box_bfs use the same definition and the same RNG seed for the
box's mask, so they MUST agree on V(r) trial-by-trial — the script asserts.
old_4096 uses a different (larger) arena, so its V(r) differs slightly.

Output:
  - prints per-method timing + assertion result
  - simulation_data/bench_l0.csv (one row per method)
"""

import csv
import time
from pathlib import Path

import numpy as np
import numba


HERE = Path(__file__).resolve().parent           # presentation/coding/benchmarks
ROOT = HERE.parent.parent                         # presentation/
DATA_DIR = ROOT / "simulation_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

P_C    = 0.592746
SCALES = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=np.int64)
R_MAX  = int(SCALES.max())
N_BIG  = 4096                  # old method's arena
N_BOX  = 2 * R_MAX + 1         # 2049

N_TRIALS = 50                  # benchmark sample size
SEED     = 20260518


# ─────────────────────────────────────────────────────────────────────────────
# Union-find (shared by old_4096 and box_uf)
# ─────────────────────────────────────────────────────────────────────────────

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
def label_sq(mask, out):
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


def count_V_from_labels(labels, ci, cj, scales):
    origin_id = labels[ci, cj]
    out = np.zeros(len(scales), dtype=np.int64)
    if origin_id == 0:
        return out
    for k, r in enumerate(scales):
        sub = labels[ci - r : ci + r + 1, cj - r : cj + r + 1]
        out[k] = int((sub == origin_id).sum())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Clipped BFS
# ─────────────────────────────────────────────────────────────────────────────

@numba.njit(cache=True)
def bfs_count_V(mask, scales, out, visited, queue, count_by_d):
    """Box-clipped BFS from the origin (center of mask)."""
    n = mask.shape[0]
    r_max = (n - 1) // 2
    origin = r_max

    # reset buffers
    for k in range(len(scales)):
        out[k] = 0
    for i in range(n):
        for j in range(n):
            visited[i, j] = False
    for d in range(r_max + 1):
        count_by_d[d] = 0

    if not mask[origin, origin]:
        return

    count_by_d[0] = 1
    visited[origin, origin] = True
    queue[0, 0] = origin
    queue[0, 1] = origin
    head = 0
    tail = 1

    while head < tail:
        i = queue[head, 0]
        j = queue[head, 1]
        head += 1
        # 4-neighbour expansion, clipped to grid
        for nb in range(4):
            if nb == 0:
                ni = i + 1; nj = j
            elif nb == 1:
                ni = i - 1; nj = j
            elif nb == 2:
                ni = i;     nj = j + 1
            else:
                ni = i;     nj = j - 1
            if 0 <= ni < n and 0 <= nj < n and mask[ni, nj] and not visited[ni, nj]:
                visited[ni, nj] = True
                di = ni - origin
                dj = nj - origin
                if di < 0: di = -di
                if dj < 0: dj = -dj
                d = di if di > dj else dj
                count_by_d[d] += 1
                queue[tail, 0] = ni
                queue[tail, 1] = nj
                tail += 1

    cum = 0
    k = 0
    n_scales = len(scales)
    for d in range(r_max + 1):
        cum += count_by_d[d]
        if k < n_scales and d == scales[k]:
            out[k] = cum
            k += 1


# ─────────────────────────────────────────────────────────────────────────────
# Bench drivers
# ─────────────────────────────────────────────────────────────────────────────

def run_old_4096(n_trials, seed):
    """Per-trial: gen 4096^2 uniforms → mask → label N^2 → count V via slices."""
    rng = np.random.default_rng(seed)
    ci = cj = N_BIG // 2
    labels = np.zeros((N_BIG, N_BIG), dtype=np.int32)
    Vs = np.zeros((n_trials, len(SCALES)), dtype=np.int64)
    t0 = time.perf_counter()
    for t in range(n_trials):
        U = rng.random((N_BIG, N_BIG), dtype=np.float32)
        mask = U < P_C
        label_sq(mask, labels)
        Vs[t] = count_V_from_labels(labels, ci, cj, SCALES)
    return Vs, time.perf_counter() - t0


def run_box_uf(n_trials, seed):
    """Per-trial: gen 2049^2 uniforms → mask → label box → count V via slices."""
    rng = np.random.default_rng(seed)
    ci = cj = R_MAX
    labels = np.zeros((N_BOX, N_BOX), dtype=np.int32)
    Vs = np.zeros((n_trials, len(SCALES)), dtype=np.int64)
    t0 = time.perf_counter()
    for t in range(n_trials):
        U = rng.random((N_BOX, N_BOX), dtype=np.float32)
        mask = U < P_C
        label_sq(mask, labels)
        Vs[t] = count_V_from_labels(labels, ci, cj, SCALES)
    return Vs, time.perf_counter() - t0


def run_box_bfs(n_trials, seed):
    """Per-trial: gen 2049^2 uniforms → mask → BFS from center → prefix-sum."""
    rng = np.random.default_rng(seed)
    visited     = np.zeros((N_BOX, N_BOX), dtype=np.bool_)
    queue       = np.zeros((N_BOX * N_BOX, 2), dtype=np.int32)
    count_by_d  = np.zeros(R_MAX + 1, dtype=np.int64)
    out         = np.zeros(len(SCALES), dtype=np.int64)
    Vs          = np.zeros((n_trials, len(SCALES)), dtype=np.int64)
    t0 = time.perf_counter()
    for t in range(n_trials):
        U = rng.random((N_BOX, N_BOX), dtype=np.float32)
        mask = U < P_C
        bfs_count_V(mask, SCALES, out, visited, queue, count_by_d)
        Vs[t] = out
    return Vs, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"L0 benchmark: {N_TRIALS} trials per method, "
          f"scales = {list(SCALES)}, p_c = {P_C}\n")

    # Warm up Numba
    print("compiling numba kernels...", end=" ", flush=True)
    warm_mask = (np.random.default_rng(0).random((16, 16)) < P_C)
    warm_lab  = np.zeros((16, 16), dtype=np.int32)
    label_sq(warm_mask, warm_lab)
    bfs_count_V(
        warm_mask,
        np.array([1, 2, 4], dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.zeros((16, 16), dtype=np.bool_),
        np.zeros((16 * 16, 2), dtype=np.int32),
        np.zeros(8 + 1, dtype=np.int64),
    )
    print("done.\n")

    print(f"running old_4096  (arena {N_BIG}x{N_BIG}) ...", flush=True)
    V_old,  t_old  = run_old_4096(N_TRIALS, SEED)

    print(f"running box_uf    (arena {N_BOX}x{N_BOX}) ...", flush=True)
    V_uf,   t_uf   = run_box_uf(N_TRIALS, SEED)

    print(f"running box_bfs   (arena {N_BOX}x{N_BOX}) ...", flush=True)
    V_bfs,  t_bfs  = run_box_bfs(N_TRIALS, SEED)

    # Correctness check: box_uf and box_bfs share the seed and arena,
    # so their per-trial V's must agree exactly.
    same = np.array_equal(V_uf, V_bfs)
    print(f"\nbox_uf == box_bfs (per-trial V(r)) : {same}")
    if not same:
        diff = np.where(V_uf != V_bfs)
        raise SystemExit(f"BFS / UF disagree at indices {diff} — bug")

    # Some sanity numbers from V_bfs (cluster sizes at biggest scale)
    V_max = V_bfs[:, -1]
    occ_frac = float((V_max > 0).mean())
    print(f"origin occupied fraction (box arena)  : {occ_frac:.3f}  "
          f"(theory {P_C})")
    print(f"mean cluster size in box (r=r_max)    : {V_max.mean():.1f}")
    print(f"max  cluster size in box (r=r_max)    : {V_max.max()}")

    # ── timing table ──────────────────────────────────────────────────────────
    rows = []
    print(f"\n{'method':12s}  {'total (s)':>10s}  {'per-trial (s)':>14s}  "
          f"{'×speedup':>10s}")
    print("-" * 52)
    for name, total in [("old_4096", t_old), ("box_uf", t_uf), ("box_bfs", t_bfs)]:
        per = total / N_TRIALS
        sp  = t_old / total
        rows.append({
            "method":   name,
            "n_trials": N_TRIALS,
            "total_s":  round(total, 4),
            "per_trial_s": round(per, 6),
            "speedup_vs_old": round(sp, 3),
        })
        print(f"{name:12s}  {total:10.3f}  {per:14.6f}  {sp:10.3f}x")

    csv_path = DATA_DIR / "bench_l0.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "n_trials", "total_s", "per_trial_s",
                        "speedup_vs_old"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
