#!/usr/bin/env bash
# Cube hard-case paired solver-seed sweep.
# Three persistent failures from the grid (0/8 across A2/A3/A4/A5/A2_H10):
# idx 11: episode 2767 start 101
# idx 17: episode 4434 start 15
# idx 38: episode 7815 start 98
# Cells: A0, A4, A5, A2_H10 at the published budget (N=300).
# Seeds 1.5 paired across cells per episode.
#
# Distribution across 4 GPUs: 3 episodes × 4 cells = 12 (episode, cell) jobs.
# Each job runs 5 seeds serially within a python process. Round-robin across GPUs.
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

# 12 (episode, start, cell) jobs; round-robin across GPUs.
JOBS=(
 "2767 101 A0" "2767 101 A4" "2767 101 A5" "2767 101 A2_H10"
 "4434 15 A0" "4434 15 A4" "4434 15 A5" "4434 15 A2_H10"
 "7815 98 A0" "7815 98 A4" "7815 98 A5" "7815 98 A2_H10"
)

run_jobs_on_gpu() {
 local gpu=$1
 shift
 local jobs=("$@")
 for job in "${jobs[@]}"; do
 set -- $job
 local EP=$1; local ST=$2; local CELL=$3
 local LOGFILE="${LOG_DIR}/cube_hardcase_ep${EP}_${CELL}_gpu${gpu}.log"
 echo "[GPU${gpu}] === ep=${EP} start=${ST} cell=${CELL} ==="
 CUDA_VISIBLE_DEVICES=${gpu} python phase1/two_room_planner_ablation/scripts/seed_sweep_4339.py \
 --task cube --episode "$EP" --start "$ST" --cell "$CELL" \
 --seeds 1 2 3 4 5 \
 > "$LOGFILE" 2>&1
 echo "[GPU${gpu}] done ep=${EP} cell=${CELL}"
 done
}

# Round-robin assignment: 12 jobs / 4 GPUs = 3 jobs each.
GPU0_JOBS=("${JOBS[0]}" "${JOBS[4]}" "${JOBS[8]}")
GPU1_JOBS=("${JOBS[1]}" "${JOBS[5]}" "${JOBS[9]}")
GPU2_JOBS=("${JOBS[2]}" "${JOBS[6]}" "${JOBS[10]}")
GPU3_JOBS=("${JOBS[3]}" "${JOBS[7]}" "${JOBS[11]}")

echo "Launching 4-GPU Cube hard-case sweep at $(date -Iseconds)"

run_jobs_on_gpu 0 "${GPU0_JOBS[@]}" &
run_jobs_on_gpu 1 "${GPU1_JOBS[@]}" &
run_jobs_on_gpu 2 "${GPU2_JOBS[@]}" &
run_jobs_on_gpu 3 "${GPU3_JOBS[@]}" &
wait

echo "Cube hard-case sweep complete at $(date -Iseconds)"
