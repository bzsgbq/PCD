#!/usr/bin/env bash
# launcher (Reproduction launcher):
# - 12 missing (task, seed) combos: tworoom/pusht/cube x seeds {0,1,2,3}.
# - solver_batch_size=1 (matches existing Reacher 5-seed pool).
# - num_eval=50; coupled --seed s --solver_seed s.
# - Multi-process per GPU on GPUs 0/2/3 (skip GPU 1; Wan2.1 + cmpa procs there).
# - VRAM-safe slot allocation: GPU0=4, GPU2=3, GPU3=3 = 10 simultaneous slots.
# - 12 combos -> 2 slots run a 2-combo chain; the other 8 slots run 1 combo each.
#
# Per slot: separate background bash process so all workers on a GPU run in
# parallel. Each combo's stdout/stderr lands in
# logs/grid5seed_<task>_seed<s>_gpu<g>_slot<n>.log.
set -e
# shellcheck disable=SC1091
source ${LEWM_VENV}/bin/activate

# Per-process CPU thread caps. Total active processes = 10 ours + 12 cmpa = 22.
# Keep our procs at 4 threads each so we don't worsen cmpa's CPU contention.
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

run_one() {
 local gpu=$1
 local slot=$2
 local task=$3
 local seed=$4
 local logfile="${LOG_DIR}/grid5seed_${task}_seed${seed}_gpu${gpu}_slot${slot}.log"
 echo "[GPU${gpu}/slot${slot}] >>> task=${task} seed=${seed} start=$(date -Iseconds)"
 CUDA_VISIBLE_DEVICES=${gpu} python phase1/two_room_planner_ablation/scripts/run_ablation.py \
 --task "$task" --cells "$CELLS" --num_eval 50 \
 --seed "$seed" --solver_seed "$seed" --solver_batch_size 1 \
 > "$logfile" 2>&1
 echo "[GPU${gpu}/slot${slot}] <<< task=${task} seed=${seed} end=$(date -Iseconds)"
}

# Each chain function runs its assigned combos serially.
chain_g0_s0() { run_one 0 0 tworoom 0; }
chain_g0_s1() { run_one 0 1 pusht 0; }
chain_g0_s2() { run_one 0 2 cube 0; }
chain_g0_s3() { run_one 0 3 tworoom 3; }

chain_g2_s0() { run_one 2 0 tworoom 1; run_one 2 0 cube 3; }
chain_g2_s1() { run_one 2 1 pusht 1; }
chain_g2_s2() { run_one 2 2 cube 1; }

chain_g3_s0() { run_one 3 0 tworoom 2; run_one 3 0 pusht 3; }
chain_g3_s1() { run_one 3 1 pusht 2; }
chain_g3_s2() { run_one 3 2 cube 2; }

echo "Launching Path-D 5-seed grid at $(date -Iseconds)"
echo "Layout: GPU0=4 slots, GPU2=3 slots, GPU3=3 slots; 12 combos in 10 simultaneous workers"

chain_g0_s0 & PID0_0=$!
chain_g0_s1 & PID0_1=$!
chain_g0_s2 & PID0_2=$!
chain_g0_s3 & PID0_3=$!
chain_g2_s0 & PID2_0=$!
chain_g2_s1 & PID2_1=$!
chain_g2_s2 & PID2_2=$!
chain_g3_s0 & PID3_0=$!
chain_g3_s1 & PID3_1=$!
chain_g3_s2 & PID3_2=$!

echo "PIDs: G0[$PID0_0,$PID0_1,$PID0_2,$PID0_3] G2[$PID2_0,$PID2_1,$PID2_2] G3[$PID3_0,$PID3_1,$PID3_2]"

wait $PID0_0 $PID0_1 $PID0_2 $PID0_3 $PID2_0 $PID2_1 $PID2_2 $PID3_0 $PID3_1 $PID3_2

echo "Path-D 5-seed grid complete at $(date -Iseconds)"
