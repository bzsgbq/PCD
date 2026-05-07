#!/usr/bin/env bash
# Parallel Path-2 grid — 7 remaining (task, seed) combos across 4 GPUs (TwoRoom already done).
# User asked to parallelize per-GPU; round-robin distribute combos so each GPU runs its
# assigned combos serially while all GPUs run in parallel. Each combo = run_ablation.py
# with --cells A0,A1,A2,A3,A4,A5,A0_H10,A2_H10 --num_eval 50 --solver_batch_size 1.
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

# 7 remaining combos. Round-robin distribution across 4 GPUs:
# GPU 0: pusht@42, reacher@2
# GPU 1: cube@42, reacher@3
# GPU 2: reacher@0, reacher@42
# GPU 3: reacher@1
GPU0_COMBOS=("pusht 42" "reacher 2")
GPU1_COMBOS=("cube 42" "reacher 3")
GPU2_COMBOS=("reacher 0" "reacher 42")
GPU3_COMBOS=("reacher 1")

run_combos_on_gpu() {
 local gpu=$1
 shift
 local combos=("$@")
 for combo in "${combos[@]}"; do
 set -- $combo
 local TASK=$1
 local SEED=$2
 local LOGFILE="${LOG_DIR}/grid_${TASK}_seed${SEED}_gpu${gpu}.log"
 echo "[GPU${gpu}] === task=${TASK} seed=${SEED} ==="
 CUDA_VISIBLE_DEVICES=${gpu} python phase1/two_room_planner_ablation/scripts/run_ablation.py \
 --task "$TASK" --cells "$CELLS" --num_eval 50 \
 --seed "$SEED" --solver_seed "$SEED" --solver_batch_size 1 \
 > "$LOGFILE" 2>&1
 echo "[GPU${gpu}] done task=${TASK} seed=${SEED}"
 done
}

echo "Launching 4-GPU parallel grid at $(date -Iseconds)"

run_combos_on_gpu 0 "${GPU0_COMBOS[@]}" &
PID0=$!
run_combos_on_gpu 1 "${GPU1_COMBOS[@]}" &
PID1=$!
run_combos_on_gpu 2 "${GPU2_COMBOS[@]}" &
PID2=$!
run_combos_on_gpu 3 "${GPU3_COMBOS[@]}" &
PID3=$!

echo "PIDs: 0=$PID0 1=$PID1 2=$PID2 3=$PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "Parallel grid complete at $(date -Iseconds)"
