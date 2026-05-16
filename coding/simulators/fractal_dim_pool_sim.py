"""
Per-trial pool simulator for the scale-of-scales study (§6 of the IMPA talk).

Saves the full per-trial V_t(r) matrix to disk so the downstream regime-sweep
script can derive estimator replicas by chunking the pool and running OLS on
each chunk.

There are three L0 trial methods available (see `coding/bench_l0.py` for the
head-to-head benchmark that produced the defaults):

  bfs  (default, ~20x faster than `old` on N=4096):
        arena = 2*r_max+1, origin at center, BFS from origin clipped to the
        box. V(r) is the count of cluster sites within L_inf box of half-side r.
        Square lattice only for now.

  uf:   arena = 2*r_max+1, origin at center, union-find labels the whole box,
        V(r) counted via slicing. Box-clipped semantics — identical V(r) to
        `bfs` for the same seed. Available on hex/square/tri.

  old:  arena = --n (e.g. 4096), origin at center, union-find labels the whole
        grid. V(r) uses *full-grid* cluster membership restricted to the box
        — sites can connect via detours outside the box. Different (slightly
        larger) V(r) than `bfs`/`uf`. This is the legacy method, kept for
        cross-checks.

Output: NPZ at `presentation18-05-2026/simulation_data/fractal_dim_pool.npz`
  V_pool : int64, shape (n_total, n_scales)
  scales : int32, shape (n_scales,)
  method : 'bfs' | 'uf' | 'old'
  lattice, N (arena side), p_c, seed, elapsed_seconds — metadata
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import numba


# ─── CLI ─────────────────────────────────────────────────────────────────────

_p = argparse.ArgumentParser(description="Per-trial V pool simulator.")
_p.add_argument("--method", choices=["bfs", "uf", "old"], default="bfs",
                help="L0 trial method (default: bfs, the fastest).")
_p.add_argument("--n", type=int, default=None,
                help="Arena side (only used for --method old; auto-set to "
                     "2*r_max+1 for bfs/uf).")
_p.add_argument("--trials", type=int, default=2000,
                help="Total trials in the pool (default 2000).")
_p.add_argument("--scales", type=int, nargs="+",
                default=[2, 4, 8, 16, 32, 64, 128, 256])
_p.add_argument("--seed", type=int, default=20260518)
_p.add_argument("--lattice", choices=["square", "hexagonal", "triangular"],
                default="square")
_p.add_argument("--batch-size", type=int, default=2048,
                help="Trials per checkpoint flush (default 2048). After each "
                     "batch the partial pool is written to a .ckpt.npz; on "
                     "restart the simulator picks up from there.")
_p.add_argument("--out", type=Path,
                default=Path(__file__).resolve().parents[2]
                        / "simulation_data" / "fractal_dim_pool.npz")
args = _p.parse_args()

METHOD     = args.method
N_TRIALS   = args.trials
SCALES     = np.array(sorted(args.scales), dtype=np.int32)
SEED       = args.seed
LATTICE    = args.lattice
BATCH_SIZE = max(1, args.batch_size)
OUT_PATH   = args.out
CKPT_PATH  = OUT_PATH.with_suffix(".ckpt.npz")

R_MAX = int(SCALES.max())

if METHOD == "old":
    if args.n is None:
        raise SystemExit("--method old requires --n (arena side, e.g. 4096).")
    N = int(args.n)
    if N < 2 * R_MAX + 1:
        raise SystemExit(f"Grid N={N} too small for r_max={R_MAX}; "
                         f"need N >= {2*R_MAX+1}.")
else:
    # box-clipped methods: arena = 2*r_max+1 always
    N = 2 * R_MAX + 1
    if args.n is not None and args.n != N:
        print(f"warning: --n={args.n} ignored for --method {METHOD}; "
              f"arena = 2*r_max+1 = {N}.")

P_C_MAP = {"hexagonal": 0.6962, "square": 0.592746, "triangular": 0.5}
P_C = P_C_MAP[LATTICE]

if METHOD == "bfs" and LATTICE != "square":
    raise SystemExit("--method bfs only implemented for square lattice; "
                     "use --method uf for hex/tri.")


# ─── Union-find (used by `uf` and `old`) ─────────────────────────────────────

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


def count_V_from_labels(labels, ci, cj, scales):
    origin_id = labels[ci, cj]
    out = np.zeros(len(scales), dtype=np.int64)
    if origin_id == 0:
        return out
    for k, r in enumerate(scales):
        sub = labels[ci - r : ci + r + 1, cj - r : cj + r + 1]
        out[k] = int((sub == origin_id).sum())
    return out


# ─── Clipped BFS (used by `bfs`, square only) ────────────────────────────────

@numba.njit(cache=True)
def _bfs_count_V_sq(mask, scales, out, visited, queue, count_by_d):
    """Box-clipped BFS from the centre of `mask` (square lattice, 4-neighbour)."""
    n = mask.shape[0]
    r_max = (n - 1) // 2
    origin = r_max

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


# ─── Per-trial drivers ───────────────────────────────────────────────────────

def make_trial_fn(method, lattice, N, scales):
    """Return a closure trial_fn(U) -> V (np.ndarray of len(scales))."""
    scales_i64 = scales.astype(np.int64)
    ci = cj = N // 2

    if method == "bfs":
        visited    = np.zeros((N, N), dtype=np.bool_)
        queue      = np.zeros((N * N, 2), dtype=np.int32)
        count_by_d = np.zeros(R_MAX + 1, dtype=np.int64)
        out        = np.zeros(len(scales), dtype=np.int64)

        def trial(U):
            mask = U < P_C
            _bfs_count_V_sq(mask, scales_i64, out, visited, queue, count_by_d)
            return out.copy()
        return trial

    # uf and old share the same labels-then-slice pipeline; only the arena size
    # changes (caller picks N appropriately).
    label_fn = LABEL_FNS[lattice]
    labels   = np.zeros((N, N), dtype=np.int32)

    def trial(U):
        mask = U < P_C
        label_fn(mask, labels)
        return count_V_from_labels(labels, ci, cj, scales)
    return trial


# ─── Numba warm-up ───────────────────────────────────────────────────────────

def warmup():
    warm_mask = np.ones((8, 8), dtype=np.bool_)
    warm_out  = np.zeros((8, 8), dtype=np.int32)
    for fn in LABEL_FNS.values():
        fn(warm_mask, warm_out)
    _bfs_count_V_sq(
        warm_mask,
        np.array([1, 2], dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        np.zeros((8, 8), dtype=np.bool_),
        np.zeros((8 * 8, 2), dtype=np.int32),
        np.zeros(4 + 1, dtype=np.int64),
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def _fingerprint():
    """Identify a run: anything in here must match a checkpoint to reuse it."""
    return {
        "method":   METHOD,
        "lattice":  LATTICE,
        "N":        int(N),
        "scales":   [int(s) for s in SCALES],
        "seed":     int(SEED),
        "trials":   int(N_TRIALS),
        "batch":    int(BATCH_SIZE),
    }


def _try_resume():
    """Return (V_pool, n_done, elapsed_so_far) — fresh state if no usable ckpt."""
    V_pool = np.zeros((N_TRIALS, len(SCALES)), dtype=np.int64)
    if not CKPT_PATH.exists():
        return V_pool, 0, 0.0
    try:
        ck = np.load(CKPT_PATH, allow_pickle=False)
        fp_saved = json.loads(str(ck["fingerprint"]))
        if fp_saved != _fingerprint():
            print(f"  checkpoint at {CKPT_PATH} has different parameters "
                  f"— ignoring and starting fresh.")
            return V_pool, 0, 0.0
        n_done = int(ck["n_done"])
        V_pool[:n_done] = ck["V_pool"][:n_done]
        elapsed = float(ck["elapsed_seconds"])
        print(f"  resuming from checkpoint: {n_done}/{N_TRIALS} trials "
              f"already done ({elapsed:.1f}s of prior compute).")
        return V_pool, n_done, elapsed
    except Exception as e:
        print(f"  could not read checkpoint ({e}); starting fresh.")
        return V_pool, 0, 0.0


def _write_checkpoint(V_pool, n_done, elapsed):
    """Atomic-ish: write to tmp, rename. Avoids half-written .npz on Ctrl-C."""
    tmp = CKPT_PATH.with_suffix(".ckpt.tmp.npz")
    np.savez(
        tmp,
        V_pool=V_pool[:n_done],
        n_done=np.int64(n_done),
        elapsed_seconds=np.float64(elapsed),
        fingerprint=np.array(json.dumps(_fingerprint())),
    )
    tmp.replace(CKPT_PATH)


def main():
    print(f"\nPool simulation")
    print(f"  method       : {METHOD}")
    print(f"  lattice      : {LATTICE}")
    print(f"  arena N      : {N}   (origin at ({N//2},{N//2}))")
    print(f"  trials       : {N_TRIALS}")
    print(f"  scales       : {list(SCALES)}")
    print(f"  p_c          : {P_C}")
    print(f"  seed         : {SEED}")
    print(f"  batch_size   : {BATCH_SIZE}")
    print(f"  out          : {OUT_PATH}")
    print(f"  ckpt         : {CKPT_PATH}\n")

    print("Compiling Numba kernels...", end=" ", flush=True)
    warmup()
    print("done.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    trial = make_trial_fn(METHOD, LATTICE, N, SCALES)
    V_pool, n_done, prior_elapsed = _try_resume()

    # One batch = one independent sub-RNG, spawned deterministically from SEED.
    # Resuming skips already-done batches but reuses the same spawn list, so
    # each batch's randomness is reproducible regardless of where we restart.
    n_batches = (N_TRIALS + BATCH_SIZE - 1) // BATCH_SIZE
    batch_seeds = np.random.SeedSequence(SEED).spawn(n_batches)

    # Reusable U buffer — avoid allocating ~N*N*4 bytes per trial.
    U = np.empty((N, N), dtype=np.float32)

    t0 = time.perf_counter()
    for b in range(n_batches):
        b_lo = b * BATCH_SIZE
        b_hi = min(b_lo + BATCH_SIZE, N_TRIALS)
        if b_hi <= n_done:
            continue                          # batch already covered by ckpt
        rng_b = np.random.default_rng(batch_seeds[b])
        # If we resume in the middle of a batch, redo the whole batch — the
        # RNG is per-batch so re-running is deterministic and the previously
        # saved trials in this batch get correctly overwritten.
        for t in range(b_lo, b_hi):
            rng_b.random(out=U, dtype=np.float32)
            V_pool[t] = trial(U)
        n_done = b_hi

        elapsed = prior_elapsed + (time.perf_counter() - t0)
        _write_checkpoint(V_pool, n_done, elapsed)

        eta = elapsed * (N_TRIALS - n_done) / max(n_done, 1)
        sys.stdout.write(
            f"\r  batch {b+1:3d}/{n_batches}  trial {n_done:5d}/{N_TRIALS}  "
            f"elapsed={elapsed:7.1f}s  eta={eta:7.1f}s"
        )
        sys.stdout.flush()
    print()

    total_elapsed = prior_elapsed + (time.perf_counter() - t0)

    np.savez_compressed(
        OUT_PATH,
        V_pool=V_pool,
        scales=SCALES,
        method=np.array(METHOD),
        lattice=np.array(LATTICE),
        N=np.int64(N),
        p_c=np.float64(P_C),
        seed=np.int64(SEED),
        elapsed_seconds=np.float64(total_elapsed),
    )
    print(f"  wrote {OUT_PATH}  (elapsed {total_elapsed:.1f}s, "
          f"{total_elapsed / N_TRIALS * 1000:.1f} ms/trial)")

    meta_path = OUT_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "method": METHOD, "lattice": LATTICE, "N": N, "n_trials": N_TRIALS,
        "scales": [int(s) for s in SCALES], "seed": SEED, "p_c": P_C,
        "batch_size": BATCH_SIZE,
        "elapsed_seconds": round(total_elapsed, 2),
    }, indent=2))
    print(f"  wrote {meta_path}")

    try:
        CKPT_PATH.unlink()
        print(f"  removed checkpoint {CKPT_PATH}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
