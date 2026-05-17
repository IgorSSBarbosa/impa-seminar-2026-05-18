"""
Parallel per-trial pool simulator (square lattice, BFS method only).

Same outputs and checkpoint format as `fractal_dim_pool_sim.py` so the two are
interchangeable: a partial pool produced here can be resumed serially and vice
versa. The compute model is a `multiprocessing.Pool` with a fixed worker count
held throughout the run; each worker allocates its own arena `U`, BFS scratch
arrays and numba-cached kernel once at initialisation.

Determinism: each batch is seeded from `np.random.SeedSequence(SEED).spawn(...)`,
identical to the serial driver. Workers process batches in any order; we slot
results back into the pool by batch index.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import numba


# ─── BFS kernel (copied verbatim from fractal_dim_pool_sim.py) ───────────────

@numba.njit(cache=True)
def _bfs_count_V_sq(mask, scales, out, visited, queue, count_by_d):
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


# ─── Worker (module-level so it pickles) ─────────────────────────────────────
# Per-process state is held in module globals, initialised once per worker.

_W = {}  # holds N, R_MAX, SCALES, P_C and the scratch arrays


def _worker_init(N, R_MAX, scales_list, p_c):
    scales = np.array(scales_list, dtype=np.int64)
    _W["N"]          = N
    _W["R_MAX"]      = R_MAX
    _W["SCALES"]     = scales
    _W["P_C"]        = float(p_c)
    _W["U"]          = np.empty((N, N), dtype=np.float32)
    _W["visited"]    = np.zeros((N, N), dtype=np.bool_)
    _W["queue"]      = np.zeros((N * N, 2), dtype=np.int32)
    _W["count_by_d"] = np.zeros(R_MAX + 1, dtype=np.int64)
    _W["out"]        = np.zeros(len(scales), dtype=np.int64)
    # warm the numba kernel so the first real batch isn't paying compile time
    warm_mask = np.ones((8, 8), dtype=np.bool_)
    _bfs_count_V_sq(
        warm_mask,
        np.array([1, 2], dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        np.zeros((8, 8), dtype=np.bool_),
        np.zeros((8 * 8, 2), dtype=np.int32),
        np.zeros(4 + 1, dtype=np.int64),
    )


def _worker_run_batch(task):
    """task = (batch_idx, seed_state, n_in_batch)
    Returns (batch_idx, V_batch, wall_seconds)."""
    batch_idx, seed_state, n_in_batch = task
    seed_seq = np.random.SeedSequence(entropy=seed_state["entropy"],
                                      spawn_key=tuple(seed_state["spawn_key"]),
                                      pool_size=seed_state["pool_size"])
    rng = np.random.default_rng(seed_seq)

    N        = _W["N"]
    scales   = _W["SCALES"]
    p_c      = _W["P_C"]
    U        = _W["U"]
    visited  = _W["visited"]
    queue    = _W["queue"]
    count_by_d = _W["count_by_d"]
    out      = _W["out"]

    V_batch = np.zeros((n_in_batch, len(scales)), dtype=np.int64)
    t0 = time.perf_counter()
    for t in range(n_in_batch):
        rng.random(out=U, dtype=np.float32)
        mask = U < p_c
        _bfs_count_V_sq(mask, scales, out, visited, queue, count_by_d)
        V_batch[t] = out
    wall = time.perf_counter() - t0
    return batch_idx, V_batch, wall


def _seed_state(seq):
    return {
        "entropy":   seq.entropy,
        "spawn_key": list(seq.spawn_key),
        "pool_size": seq.pool_size,
    }


# ─── CLI / orchestration ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Parallel BFS pool simulator.")
    p.add_argument("--trials", type=int, default=65536)
    p.add_argument("--scales", type=int, nargs="+",
                   default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--batch-size", type=int, default=512,
                   help="Trials per batch dispatched to a worker. Smaller = "
                        "better load balance, more IPC overhead. Default 512.")
    p.add_argument("--ckpt-every", type=int, default=4,
                   help="Flush a checkpoint every N completed batches.")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[2]
                           / "simulation_data" / "fractal_dim_pool.npz")
    return p.parse_args()


def fingerprint(method, lattice, N, scales, seed, trials, batch_size):
    return {
        "method":  method,
        "lattice": lattice,
        "N":       int(N),
        "scales":  [int(s) for s in scales],
        "seed":    int(seed),
        "trials":  int(trials),
        "batch":   int(batch_size),
    }


def try_resume(ckpt_path, N_TRIALS, n_scales, fp_self):
    V_pool = np.zeros((N_TRIALS, n_scales), dtype=np.int64)
    done_mask = np.zeros(N_TRIALS, dtype=np.bool_)
    if not ckpt_path.exists():
        return V_pool, done_mask, 0.0
    try:
        ck = np.load(ckpt_path, allow_pickle=False)
        fp_saved = json.loads(str(ck["fingerprint"]))
        if fp_saved != fp_self:
            print(f"  checkpoint at {ckpt_path} has different parameters "
                  f"— ignoring and starting fresh.")
            return V_pool, done_mask, 0.0
        n_done = int(ck["n_done"])
        V_pool[:n_done] = ck["V_pool"][:n_done]
        done_mask[:n_done] = True
        elapsed = float(ck["elapsed_seconds"])
        print(f"  resuming: {n_done}/{N_TRIALS} trials done "
              f"({elapsed:.1f}s prior compute).")
        return V_pool, done_mask, elapsed
    except Exception as e:
        print(f"  could not read checkpoint ({e}); starting fresh.")
        return V_pool, done_mask, 0.0


def write_checkpoint(ckpt_path, V_pool, n_done, elapsed, fp_self):
    tmp = ckpt_path.with_suffix(".ckpt.tmp.npz")
    np.savez(
        tmp,
        V_pool=V_pool[:n_done],
        n_done=np.int64(n_done),
        elapsed_seconds=np.float64(elapsed),
        fingerprint=np.array(json.dumps(fp_self)),
    )
    tmp.replace(ckpt_path)


def main():
    args = parse_args()
    SCALES   = np.array(sorted(args.scales), dtype=np.int32)
    R_MAX    = int(SCALES.max())
    N        = 2 * R_MAX + 1
    P_C      = 0.592746
    N_TRIALS = args.trials
    BATCH    = args.batch_size
    OUT_PATH = args.out
    CKPT     = OUT_PATH.with_suffix(".ckpt.npz")

    fp_self = fingerprint("bfs", "square", N, SCALES, args.seed, N_TRIALS, BATCH)

    print("\nParallel pool simulation")
    print(f"  workers      : {args.workers}")
    print(f"  arena N      : {N}   (origin at ({N//2},{N//2}))")
    print(f"  trials       : {N_TRIALS}")
    print(f"  scales       : {list(SCALES)}")
    print(f"  seed         : {args.seed}")
    print(f"  batch_size   : {BATCH}")
    print(f"  per-worker U : {N*N*4 / 1024**2:.1f} MB  "
          f"(+ visited/queue ≈ {N*N*(1+8) / 1024**2:.1f} MB more)")
    print(f"  out          : {OUT_PATH}")
    print(f"  ckpt         : {CKPT}\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    V_pool, done_mask, prior_elapsed = try_resume(
        CKPT, N_TRIALS, len(SCALES), fp_self)

    n_batches = (N_TRIALS + BATCH - 1) // BATCH
    batch_seeds = np.random.SeedSequence(args.seed).spawn(n_batches)

    tasks = []
    for b in range(n_batches):
        b_lo = b * BATCH
        b_hi = min(b_lo + BATCH, N_TRIALS)
        if done_mask[b_lo:b_hi].all():
            continue
        tasks.append((b, _seed_state(batch_seeds[b]), b_hi - b_lo))

    if not tasks:
        print("  nothing to do (pool already complete).")
    else:
        print(f"  dispatching {len(tasks)} batches to {args.workers} workers...")

    n_done = int(done_mask.sum())
    t0 = time.perf_counter()
    completed_since_ckpt = 0
    bfs_seconds = 0.0  # cumulative worker CPU time on BFS

    with mp.Pool(processes=args.workers,
                 initializer=_worker_init,
                 initargs=(N, R_MAX, list(SCALES), P_C)) as pool:
        for batch_idx, V_batch, wall in pool.imap_unordered(
                _worker_run_batch, tasks, chunksize=1):
            b_lo = batch_idx * BATCH
            b_hi = b_lo + V_batch.shape[0]
            V_pool[b_lo:b_hi] = V_batch
            done_mask[b_lo:b_hi] = True
            n_done = int(done_mask.sum())
            bfs_seconds += wall

            completed_since_ckpt += 1
            elapsed = prior_elapsed + (time.perf_counter() - t0)
            eta = elapsed * (N_TRIALS - n_done) / max(n_done, 1)
            speedup = bfs_seconds / max(time.perf_counter() - t0, 1e-9)
            sys.stdout.write(
                f"\r  done {n_done:5d}/{N_TRIALS}  "
                f"elapsed={elapsed:7.1f}s  eta={eta:7.1f}s  "
                f"effective_speedup={speedup:4.1f}x   "
            )
            sys.stdout.flush()

            if completed_since_ckpt >= args.ckpt_every:
                write_checkpoint(CKPT, V_pool, n_done, elapsed, fp_self)
                completed_since_ckpt = 0
    print()

    total_elapsed = prior_elapsed + (time.perf_counter() - t0)
    np.savez_compressed(
        OUT_PATH,
        V_pool=V_pool,
        scales=SCALES,
        method=np.array("bfs"),
        lattice=np.array("square"),
        N=np.int64(N),
        p_c=np.float64(P_C),
        seed=np.int64(args.seed),
        elapsed_seconds=np.float64(total_elapsed),
    )
    print(f"  wrote {OUT_PATH}  (wall {total_elapsed:.1f}s, "
          f"{total_elapsed / max(N_TRIALS,1) * 1000:.2f} ms/trial wall, "
          f"effective speedup {bfs_seconds / max(total_elapsed - prior_elapsed, 1e-9):.2f}x)")

    meta_path = OUT_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "method": "bfs", "lattice": "square", "N": N, "n_trials": N_TRIALS,
        "scales": [int(s) for s in SCALES], "seed": int(args.seed), "p_c": P_C,
        "batch_size": BATCH,
        "workers": args.workers,
        "elapsed_seconds": round(total_elapsed, 2),
        "bfs_cpu_seconds": round(bfs_seconds, 2),
    }, indent=2))
    print(f"  wrote {meta_path}")

    try:
        CKPT.unlink()
        print(f"  removed checkpoint {CKPT}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
