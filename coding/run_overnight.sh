#!/bin/bash
# Overnight scale-of-scales sweep.
# Generates a big pool (N=4096, scales up to r=1024, 5000 trials) then runs the
# regime sweep and error-vs-time analyses. Total: ~70 min.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # presentation/coding
PRES="$(cd "${HERE}/.." && pwd)"                         # presentation/
DATA_DIR="${PRES}/simulation_data"
mkdir -p "${DATA_DIR}"

LOG="${PRES}/overnight.log"
echo "===== overnight run started $(date) =====" | tee "${LOG}"

echo "[1/3] pool simulation: N=4096, 5000 trials, scales=2..1024" | tee -a "${LOG}"
python3 -u "${HERE}/simulators/fractal_dim_pool_sim.py" \
    --n 4096 --trials 5000 \
    --scales 2 4 8 16 32 64 128 256 512 1024 \
    --seed 20260518 \
    --out "${DATA_DIR}/fractal_dim_pool.npz" 2>&1 | tee -a "${LOG}"

echo "[2/3] regime sweep" | tee -a "${LOG}"
python3 -u "${HERE}/regime_sweep.py" 2>&1 | tee -a "${LOG}"

echo "[3/3] error vs time" | tee -a "${LOG}"
python3 -u "${HERE}/error_vs_time.py" 2>&1 | tee -a "${LOG}"

echo "===== overnight run finished $(date) =====" | tee -a "${LOG}"
