"""
BFS-based cluster exploration from the origin on Z^d at critical probability.

Strategy
--------
Instead of allocating an N^d grid and running global Union-Find, we grow the
cluster with BFS from origin (0,...,0).  For each occupied site we visit its
2d neighbours, flipping a biased coin (prob p_c) to decide occupation.
Only sites within L-inf radius R are explored.

Why this is the right object
-----------------------------
The cluster grown this way is a sample of the *incipient infinite cluster*
(IIC) — the canonical object that exhibits fractal / mean-field scaling.

Complexity
----------
Memory and time are O(d × cluster_size), NOT O(N^d).
Key optimisation: when moving from site s along axis d_idx by ±1,
only the changed coordinate can exceed R — radius check is O(1).
"""

import numpy as np
from collections import deque


def explore_cluster(dim, p_c, R, min_cluster=200, max_retries=30, rng=None, verbose=True):
    """
    Explore the critical percolation cluster from the origin.

    Parameters
    ----------
    dim         : int
    p_c         : float — site occupation probability
    R           : int   — L-inf exploration radius
    min_cluster : int   — minimum accepted cluster size
    max_retries : int
    rng         : np.random.Generator, optional
    verbose     : bool

    Returns
    -------
    dict:
        'dim', 'coords', 'depths', 'adjacency', 'size', 'max_depth'
        'coords' is sorted ascending by BFS depth (origin first).
    """
    if rng is None:
        rng = np.random.default_rng()

    origin    = tuple([0] * dim)
    last_size = 0

    for attempt in range(max_retries):
        if verbose:
            print(f"    attempt {attempt + 1}/{max_retries} ... ", end="", flush=True)

        visited = {}   # coord_tuple -> bool (occupied?)
        depths  = {}   # coord_tuple -> BFS depth (occupied only)
        adj     = {}   # coord_tuple -> set of occupied neighbours

        # Check origin
        if rng.random() >= p_c:
            if verbose:
                print("origin empty, retrying")
            continue

        visited[origin] = True
        depths[origin]  = 0
        adj[origin]     = set()
        queue           = deque([origin])

        while queue:
            site      = queue.popleft()
            depth     = depths[site]
            site_list = list(site)

            for d_idx in range(dim):
                orig_coord = site_list[d_idx]
                for delta in (-1, 1):
                    new_coord = orig_coord + delta

                    # O(1) radius check: only changed coordinate can exceed R
                    if abs(new_coord) > R:
                        continue

                    site_list[d_idx] = new_coord
                    nb               = tuple(site_list)
                    site_list[d_idx] = orig_coord      # restore

                    if nb in visited:
                        if visited[nb]:                # occupied -> record edge
                            adj[site].add(nb)
                            adj[nb].add(site)
                        continue

                    occ         = rng.random() < p_c
                    visited[nb] = occ

                    if occ:
                        depths[nb] = depth + 1
                        adj[nb]    = {site}
                        adj[site].add(nb)
                        queue.append(nb)

        last_size = len(depths)

        if verbose:
            print(f"size={last_size:,}", end="")

        if last_size >= min_cluster:
            if verbose:
                print("  OK")
            coords_sorted = sorted(depths.keys(), key=lambda c: depths[c])
            return {
                'dim':       dim,
                'coords':    coords_sorted,
                'depths':    depths,
                'adjacency': {k: list(v) for k, v in adj.items()},
                'size':      last_size,
                'max_depth': max(depths.values()),
            }
        else:
            if verbose:
                print("  (too small, retrying)")

    raise RuntimeError(
        f"Could not find cluster >= {min_cluster} sites for dim={dim} "
        f"after {max_retries} attempts (last size: {last_size}). "
        f"Reduce min_cluster in critical_params.py or increase max_retries."
    )