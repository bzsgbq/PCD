#!/usr/bin/env bash
# Path-2 four-task perturbation grid.
# - Reacher: 5-seed grid over {0,1,2,3,42}, --seed=s --solver_seed=s (mirrors Hydra coupling).
# - Two-Room, Push-T, Cube: single-seed grid at seed=42, --solver_seed=42.
# - All cells: A0,A1,A2,A3,A4,A5,A0_H10,A2_H10.
# - All runs: --solver_batch_size 1, num_eval=50.
set -e
# shellcheck disable=SC1091
source ${LEWM_VENV}/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
cd "$(dirname "$0")/././."
LOG_DIR="phase1/two_room_planner_ablation/logs"
mkdir -p "$LOG_DIR"

CELLS="A0,A1,A2,A3,A4,A5,A0_H10,A2_H10"

# Single-seed (seed=42) grid for Two-Room, Push-T, Cube.
for TASK in tworoom pusht cube; do
 echo "=== GRID task=${TASK} seed=42 cells=${CELLS} ==="
 python phase1/two_room_planner_ablation/scripts/run_ablation.py \
 --task "$TASK" --cells "$CELLS" --num_eval 50 \
 --seed 42 --solver_seed 42 --solver_batch_size 1 \
 > "$LOG_DIR/grid_${TASK}_seed42.log" 2>&1
done

# 5-seed grid for Reacher.
for SEED in 0 1 2 3 42; do
 echo "=== GRID task=reacher seed=${SEED} solver_seed=${SEED} cells=${CELLS} ==="
 python phase1/two_room_planner_ablation/scripts/run_ablation.py \
 --task reacher --cells "$CELLS" --num_eval 50 \
 --seed "$SEED" --solver_seed "$SEED" --solver_batch_size 1 \
 > "$LOG_DIR/grid_reacher_seed${SEED}.log" 2>&1
done

echo "Path-2 four-task grid complete."
