#!/bin/bash
# Overnight scale-of-scales sweep (Plan B, 2026-05-16).
# Generates a Plan-B pool (clipped-BFS, arena 4097, scales up to r=2048,
# 65 536 trials) then runs the regime sweep and error-vs-time analyses.
# Wall clock with --method bfs: ~45-60 min for the pool (vs many hours
# on the old union-find path). See coding/benchmarks/bench_l0.py for the
# L0 benchmark.
#
# Pool size 65 536 = 64 × 1024 so the n=1024 cell still gets 64 disjoint
# L2 replicas (LOG_LOGPLOT_TRIALS in analysis/regime_sweep.py).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"         # presentation/coding/scripts
CODING="$(cd "${HERE}/.." && pwd)"                            # presentation/coding
PRES="$(cd "${CODING}/.." && pwd)"                            # presentation/
DATA_DIR="${PRES}/simulation_data"
mkdir -p "${DATA_DIR}"

LOG="${PRES}/overnight.log"
echo "===== overnight run started $(date) =====" | tee "${LOG}"

echo "[1/3] pool simulation: --method bfs, 65536 trials, scales=2..2048" | tee -a "${LOG}"
python3 -u "${CODING}/simulators/fractal_dim_pool_sim.py" \
    --method bfs --trials 65536 \
    --scales 2 4 8 16 32 64 128 256 512 1024 2048 \
    --seed 20260518 \
    --out "${DATA_DIR}/fractal_dim_pool.npz" 2>&1 | tee -a "${LOG}"

echo "[2/3] regime sweep" | tee -a "${LOG}"
python3 -u "${CODING}/analysis/regime_sweep.py" 2>&1 | tee -a "${LOG}"

echo "[3/3] error vs time" | tee -a "${LOG}"
python3 -u "${CODING}/analysis/error_vs_time.py" 2>&1 | tee -a "${LOG}"

echo "===== overnight run finished $(date) =====" | tee -a "${LOG}"
