# Leath-style lazy-growth simulator — design notes (future work)

**Status.** Not implemented as of 2026-05-17. Captured here so a future pass
can pick it up without re-deriving. Motivation: the current arena-BFS
simulator in `coding/simulators/fractal_dim_pool_sim.py` is bottlenecked by
**memory** (queue scales as N² ≈ r_max²) and **per-trial reset cost** (the
visited array zero-out is O(N²) regardless of cluster size). This forces the
pool's largest scale to be at most ≈ r_max, where V is heavily clipped by the
arena edge — the bias we observed on 2026-05-17 in `fig_masterpiece.png` at
scale 4096.

A Leath / lazy-growth variant removes both bottlenecks and unlocks
r_max ≥ 2^14 within current hardware (31 GB RAM, 12 cores).

## Algorithm

Standard Leath cluster growth (Leath 1976, *Phys. Rev. B* 14):

```
initialise:
    visited = { (0,0): occupied? }  # roll Bernoulli(p_c) lazily on first touch
    cluster = { (0,0) } if origin occupied else ∅
    frontier = deque([(0,0)]) if origin occupied else empty
    count_by_d = zeros(r_max + 1, int64)
    if origin occupied: count_by_d[0] = 1

while frontier not empty:
    (i, j) = frontier.popleft()
    for (ni, nj) in 4-neighbours of (i, j):
        d = max(|ni|, |nj|)             # L∞ distance from origin
        if d > r_max: continue          # box-clipping stop rule
        if (ni, nj) in visited: continue
        occupied = bernoulli(p_c)       # lazy: only sample on first touch
        visited[(ni, nj)] = occupied
        if occupied:
            cluster.add((ni, nj))
            frontier.append((ni, nj))
            count_by_d[d] += 1

V(rho^k) = cumsum(count_by_d)[rho^k] for each scale k    # same as today
```

Stop rule: BFS halts when the frontier empties (cluster died) or when no
frontier site lies at distance ≤ r_max (cluster is fully outside the box).
This gives identical V(r) statistics to the current arena-BFS *for r ≤ r_max*
**but with no arena allocated outside the cluster**.

## Cost vs current implementation, at p_c (square lattice, d_f = 91/48)

| metric | current (arena BFS) | Leath |
| --- | --- | --- |
| per-trial RNG | N² ≈ 4 r_max² floats | ~4 × cluster_size Bernoullis |
| per-trial reset | N² bool zero-outs | none |
| per-trial BFS work | O(cluster_size) | O(cluster_size) |
| queue mem (worst case) | N² × 8 bytes | cluster_size × ~32 bytes |
| visited mem (worst case) | N² × 1 byte | cluster_size × ~16 bytes (hash) |

At r_max=4096: cluster_size ≈ 4096^{1.896} ≈ 8.5 M sites.
- Current: 4 × 67 M ≈ 270 M RNG floats per trial, 67 M bool zero-outs, 537 MB queue.
- Leath: 34 M Bernoullis per trial (one per BFS edge sampled), 0 zero-outs,
  ~270 MB queue+visited at peak.

Expected speedup: **5–10×** at r_max=4096, larger at r_max ≥ 2^14.

## Numba implementation gotcha — the visited set

Python `dict`/`set` work in object mode but kill performance in numba's
nopython mode. Three workable strategies:

**(1) Linear-probing fixed-size hash table.**
Pre-allocate two parallel arrays `keys: int64[M]` and `vals: bool[M]` with
M = next power of two ≥ 4 × upper-bound-on-cluster-size. Encode `(i,j)` as
`i * stride + j` with `stride = 2 r_max + 1` (so i, j ∈ [-r_max, r_max] pack
into one int64). Hash with `mix64(key) & (M-1)`, linear probe on collision.
**Recommended**: simple, numba-friendly, ~16 bytes/entry. For r_max=2^14,
upper-bound cluster ≈ 2^14^{1.896} ≈ 6.3 × 10^7 → M = 2^28 → 4 GB. Too big.

**(2) Numba-typed `Dict`.**
`from numba.typed import Dict; Dict.empty(key_type=types.int64, value_type=types.b1)`.
Works inside `@njit`, but ~3× slower than (1) and same memory ballpark. Use
this for a v1 prototype, then swap to (1) for production.

**(3) Sparse coordinate sets via sorted arrays.**
For BFS this is bad — random lookups are O(log n). Skip.

**Recommended path:** ship (2) as v1 (5 lines of code, works immediately),
benchmark, then move hot path to (1) if needed. For r_max ≤ 2^13 (cluster
≈ 2 × 10^7), (1) with M = 2^25 ≈ 500 MB per worker is fine on 31 GB / 12
workers. For r_max = 2^14, drop to 6 workers.

## Bernoulli sampling on demand

`numba.njit` has `np.random.random()` available. The compiler hoists the RNG
state per-call. Cleanest pattern:

```python
@numba.njit(cache=True)
def _grow_cluster(seed, r_max, p_c, count_by_d):
    rng_state = ...  # seed the per-call generator (see below)
    ...
    if rng_state.next_uniform() < p_c:
        ...
```

Numba doesn't expose `np.random.Generator` cleanly inside njit. Workarounds:
- Use `np.random.seed(seed)` at the top of each njit call (global state — fine
  if one trial = one njit call, which is our case).
- Or use `xoshiro256` hand-rolled in pure numba (200 lines, 2x faster).

Start with `np.random.seed(seed)` + `np.random.random()`. Replace with
xoshiro if profiling shows RNG is the bottleneck.

## Validation plan

Before promoting the Leath sim to the canonical pool source:

1. **Reproducibility.** Same seed → identical V(r) values trial-by-trial vs
   the current sim, at r_max where both are unbiased (e.g., r_max=512).
   Caveat: per-trial trajectories will differ because the BFS visit order
   differs; only the aggregated V(r) per trial should match. Actually that's
   also not guaranteed without careful seeding — the order in which we sample
   neighbours matters. **Better validation: per-scale mean and variance over
   N=10⁴ trials must match within 2 SE.**

2. **Cross-scale ratios.** V(ρ^{k+1})/V(ρ^k) at k far from r_max should be
   2^{d_f} = 3.72 ± noise. Compare to the new pool's table on 2026-05-17.

3. **Clipping gone.** Run Leath at r_max=2^14, harvest V(4096). Mean V(4096)
   should be ≈ 1.07 × (Leath at r_max=2^13) — confirms scale 4096 is no
   longer at the frontier (the 7% gap is just the same finite-size a₁/i
   correction that all interior scales pay).

## Migration path

1. New file `coding/simulators/fractal_dim_pool_sim_leath.py` — copies the
   parallel-orchestration skeleton from `fractal_dim_pool_sim_par.py`, swaps
   the kernel for Leath.
2. Same checkpoint format and output schema as today, so downstream scripts
   (`regime_sweep.py`, `error_vs_time.py`, `masterpiece_plot.py`) need zero
   changes.
3. Run a small Leath pool (10⁴ trials, r_max=2048) and compare to the
   r11-archived pool. Expect identical aggregate statistics.
4. Promote to canonical when validated; archive the arena-BFS pool as
   `fractal_dim_pool.bfs.npz` for cross-checks.

## Estimated implementation time

- v1 with `numba.typed.Dict`: ~2 h coding + 1 h validation
- Hash-table v2 (only if v1 too slow): +2 h
- Pool re-run at r_max=2^14, 65k trials, 12 workers: ~6 h wall (estimate;
  could be 2–10 h depending on RNG efficiency)

Total realistic estimate: **1 day of focused work** to swap simulators and
re-generate the pool / plots / talk material.

## Related opportunistic improvements (cheap, do alongside)

- Drop the U[i,j] uniform array entirely (Leath samples Bernoulli directly).
- Pack `count_by_d` as `int32` (cluster never exceeds 2^31 sites in our range).
- Use the radius information to short-circuit BFS once the cluster is
  confirmed fully inside r_max (peripheral frontier all unoccupied).
