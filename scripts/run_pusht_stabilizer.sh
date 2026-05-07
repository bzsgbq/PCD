#!/usr/bin/env bash
# Push-T 20-seed stabilizer for the cross-task taxonomy.
# Cells: vanilla N=300, vanilla N=1000, AMB-topK no-same-ep N=300; for episodes 8314 (start=3) and 14108 (start=61).
set -e
# Activate venv whose stable_worldmodel resolves checkpoints via ~/.stable_worldmodel/<task> symlinks.
# shellcheck disable=SC1091
source ${LEWM_VENV}/bin/activate
# CPU thread limits. Defaults to 4; override at the call site, e.g.
# OMP_NUM_THREADS=16 TORCH_NUM_THREADS=16 bash run_pusht_stabilizer.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
cd "$(dirname "$0")/././."
LOG_DIR="phase1/two_room_planner_ablation/logs"
mkdir -p "$LOG_DIR"

EPISODES_STARTS=("8314 3" "14108 61")
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

for es in "${EPISODES_STARTS[@]}"; do
 set -- $es
 EP=$1; START=$2

 echo "=== ep=$EP start=$START vanilla A5 N=300 ==="
 python phase1/two_room_planner_ablation/scripts/seed_sweep_4339.py \
 --task pusht --episode "$EP" --start "$START" --cell A5 \
 --num_samples 300 --seeds $SEEDS \
 > "$LOG_DIR/stab_pusht_ep${EP}_a5_n300.log" 2>&1

 echo "=== ep=$EP start=$START vanilla A5 N=1000 ==="
 python phase1/two_room_planner_ablation/scripts/seed_sweep_4339.py \
 --task pusht --episode "$EP" --start "$START" --cell A5 \
 --num_samples 1000 --seeds $SEEDS \
 > "$LOG_DIR/stab_pusht_ep${EP}_a5_n1000.log" 2>&1

 echo "=== ep=$EP start=$START AMB-topK no-same-ep N=300 ==="
 python phase1/two_room_planner_ablation/scripts/dataset_bridge.py \
 --task pusht --episode "$EP" --start "$START" --cell A5 \
 --mode bridge --n_bridges 10 --n_db_episodes 200 --exclude_eval_ep \
 --seeds $SEEDS \
 > "$LOG_DIR/stab_pusht_ep${EP}_amb_topk.log" 2>&1
done

echo "Push-T stabilizer complete."
