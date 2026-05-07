#!/usr/bin/env bash
# Multi-seed Reacher A0 reproduction.
# Uses the Hydra wrapper so cfg.seed varies BOTH the eval-set sampler AND
# the CEM solver seed (the official solver/cem.yaml has seed: ${seed}).
# run_ablation.py's --seed only controls eval-set sampling; solver seed is
# hardcoded 42, so we use the Hydra path here.
set -e
# shellcheck disable=SC1091
source ${LEWM_VENV}/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
cd "$(dirname "$0")/./././."
LOG_DIR="phase1/two_room_planner_ablation/logs"
mkdir -p "$LOG_DIR"

# Run 5 seeds total: seeds 0.3 plus 42 as anchor.
# Note: do not mask Hydra failures with `|| true`. Verify each
# completed seed produced a metrics line before declaring it done.
for SEED in 0 1 2 3 42; do
 echo "=== Hydra Reacher A0 seed=${SEED}, num_eval=50, batch_size=1 ==="
 python phase1/two_room_planner_ablation/scripts/hydra_control/eval_no_video.py \
 --config-name=reacher policy=reacher/lewm seed="$SEED" \
 > "$LOG_DIR/hydra_reacher_seed_${SEED}.log" 2>&1
 if ! grep -q "'success_rate':" "$LOG_DIR/hydra_reacher_seed_${SEED}.log"; then
 echo " !!! seed=${SEED}: no 'success_rate' in log; aborting" >&2
 exit 1
 fi
 echo " done seed=${SEED}"
done

echo "Multi-seed Reacher sweep complete."
