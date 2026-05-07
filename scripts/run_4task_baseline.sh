#!/usr/bin/env bash
# Four-task official-baseline (A0) reproduction.
# num_eval=50, seed 42, published CEM (H=5, RH=5, terminal cost, N=300, 30 iters, topk=30).
# Compare to LeWM-reported 87 / 86 / 96 / 74 for Two-Room / Reacher / Push-T / Cube.
set -e
# shellcheck disable=SC1091
source ${LEWM_VENV}/bin/activate
# CPU thread limits per operational note.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
cd "$(dirname "$0")/././."
LOG_DIR="phase1/two_room_planner_ablation/logs"
mkdir -p "$LOG_DIR"

for TASK in tworoom reacher pusht cube; do
 echo "=== A0 reproduction: task=${TASK}, num_eval=50 ==="
 python phase1/two_room_planner_ablation/scripts/run_ablation.py \
 --task "$TASK" --cells A0 --num_eval 50 --seed 42 \
 > "$LOG_DIR/a0_repro_${TASK}.log" 2>&1
done

echo "Four-task A0 reproduction complete."
