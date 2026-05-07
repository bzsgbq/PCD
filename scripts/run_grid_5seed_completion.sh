#!/usr/bin/env bash
# 5-seed grid completion. Runs the 12 missing (task, seed) combos:
# tworoom, pusht, cube × seeds {0, 1, 2, 3} (seed 42 already exists).
# Distributed round-robin across GPUs 0, 2, 3 (GPU 1 is in use by another user).
# Each combo: --num_eval 50, --solver_batch_size 1, coupled --seed s --solver_seed s,
# cells A0.A5 + A0_H10, A2_H10. CPU thread limits enforced.
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

# 12 combos, distributed across GPUs 0/2/3 (GPU 1 in use by another user).
# Distribution prioritizes balance: tworoom is the slowest (~3.6h/seed),
# pusht/cube are ~2.5h/seed each.
#
# GPU 0: tworoom@0, pusht@0, cube@0, pusht@2 (3.6+2.5+2.5+2.5 ≈ 11.1h)
# GPU 2: tworoom@1, pusht@1, cube@1, pusht@3 (3.6+2.5+2.5+2.5 ≈ 11.1h)
# GPU 3: tworoom@2, tworoom@3, cube@2, cube@3 (3.6+3.6+2.5+2.5 ≈ 12.2h)
GPU0_COMBOS=("tworoom 0" "pusht 0" "cube 0" "pusht 2")
GPU2_COMBOS=("tworoom 1" "pusht 1" "cube 1" "pusht 3")
GPU3_COMBOS=("tworoom 2" "tworoom 3" "cube 2" "cube 3")

run_combos_on_gpu() {
 local gpu=$1
 shift
 local combos=("$@")
 for combo in "${combos[@]}"; do
 set -- $combo
 local TASK=$1
 local SEED=$2
 local LOGFILE="${LOG_DIR}/grid5seed_${TASK}_seed${SEED}_gpu${gpu}.log"
 echo "[GPU${gpu}] === task=${TASK} seed=${SEED} start=$(date -Iseconds) ==="
 CUDA_VISIBLE_DEVICES=${gpu} python phase1/two_room_planner_ablation/scripts/run_ablation.py \
 --task "$TASK" --cells "$CELLS" --num_eval 50 \
 --seed "$SEED" --solver_seed "$SEED" --solver_batch_size 1 \
 > "$LOGFILE" 2>&1
 echo "[GPU${gpu}] done task=${TASK} seed=${SEED} end=$(date -Iseconds)"
 done
}

echo "Launching 3-GPU 5-seed-completion grid at $(date -Iseconds)"

run_combos_on_gpu 0 "${GPU0_COMBOS[@]}" &
PID0=$!
run_combos_on_gpu 2 "${GPU2_COMBOS[@]}" &
PID2=$!
run_combos_on_gpu 3 "${GPU3_COMBOS[@]}" &
PID3=$!

echo "PIDs: 0=$PID0 2=$PID2 3=$PID3"
wait $PID0 $PID2 $PID3
echo "5-seed grid completion at $(date -Iseconds)"
