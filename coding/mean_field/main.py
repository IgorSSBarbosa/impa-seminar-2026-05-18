#!/usr/bin/env python3
"""
Percolation Mean-Field Animation
=================================
Generates a GIF showing site percolation clusters at criticality
for dimensions 2 through 8, illustrating the transition to mean-field
behaviour at and above the upper critical dimension d = 6.

Each dimension segment:
  1. Zoom-in hold  — tight view around the origin
  2. Smooth zoom-out — reveals the full cluster
  3. Freeze (5 s)  — full cluster, camera rotates for d >= 3

Cluster algorithm:
  BFS from the origin (0,...,0) on Z^d with site probability p_c.
  This samples the Incipient Infinite Cluster (IIC).

Usage
-----
    python main.py                  # defaults: seed=42, all dimensions
    python main.py --seed 7         # different realisation
    python main.py --dims 2 3 4     # only those dimensions

Output
------
    /home/usuario/Claude_workspace/percolation_animations/percolation_meanfield.gif
"""

import os
import sys
import argparse
import time
import numpy as np

# Ensure project root is on sys.path regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from percolation.critical_params import DIMENSION_CONFIGS, ANIM_FPS
from percolation.bfs_cluster     import explore_cluster
from visualization.animator      import build_animation

ANIM_DIR    = os.path.join(BASE_DIR, 'percolation_animations')
OUTPUT_FILE = os.path.join(ANIM_DIR, 'percolation_meanfield.gif')


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate percolation mean-field animation (dimensions 2-8).')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (default: 42)')
    p.add_argument('--dims', nargs='+', type=int, default=list(range(2, 9)),
                   help='Dimensions to include (default: 2 3 4 5 6 7 8)')
    return p.parse_args()


def header(text):
    print(f'\n{"="*60}')
    print(f'  {text}')
    print(f'{"="*60}')


def main():
    args = parse_args()
    os.makedirs(ANIM_DIR, exist_ok=True)

    header('Percolation Mean-Field Animation')
    print(f'  Dimensions : {args.dims}')
    print(f'  Seed       : {args.seed}')
    print(f'  Output     : {OUTPUT_FILE}')

    rng = np.random.default_rng(seed=args.seed)

    # ------------------------------------------------------------------ #
    # Phase 1: BFS cluster exploration                                    #
    # ------------------------------------------------------------------ #
    header('Phase 1 — Cluster exploration (BFS from origin)')

    t0       = time.time()
    clusters = {}

    for dim in args.dims:
        cfg = DIMENSION_CONFIGS[dim]
        print(f'\n  [d={dim}]  p_c={cfg["p_c"]:.6f}  '
              f'R={cfg["R"]}  min_size={cfg["min_cluster"]}')

        cluster       = explore_cluster(
            dim=dim,
            p_c=cfg['p_c'],
            R=cfg['R'],
            min_cluster=cfg['min_cluster'],
            max_retries=cfg['max_retries'],
            rng=rng,
            verbose=True,
        )
        clusters[dim] = cluster
        print(f'  -> {cluster["size"]:,} sites, max BFS depth = {cluster["max_depth"]}')

    t1 = time.time()
    print(f'\n  Cluster exploration done in {t1 - t0:.1f} s')

    # ------------------------------------------------------------------ #
    # Phase 2: Rendering + GIF assembly                                   #
    # ------------------------------------------------------------------ #
    header('Phase 2 — Rendering frames & assembling GIF')

    build_animation(clusters, OUTPUT_FILE, fps=ANIM_FPS)

    t2 = time.time()
    header('Done')
    print(f'  Total time : {t2 - t0:.1f} s  ({(t2 - t0)/60:.1f} min)')
    print(f'  Animation  : {OUTPUT_FILE}')
    print()


if __name__ == '__main__':
    main()