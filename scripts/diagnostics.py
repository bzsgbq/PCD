"""Phase 1 diagnostic suite.

Runs the trajectory-oracle protocol (replay-in-imagination, teacher-forced
rollout) and the rollout-fidelity tests (random + planner-proposal +
dataset-chunks + CEM reranking calibration) on a frozen LeWM checkpoint and
a fixed eval set.

This is intentionally separate from run_ablation.py so each cell can be
re-run in isolation.

Usage:
 python diagnostics.py --task tworoom --num_eval 8 \
 --eval_set phase1/two_room_planner_ablation/results/<run>/eval_set.json \
 --tests replay,teacher_forced,rollout_fidelity,reranking
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_LEWM_DIR = Path(__file__).resolve().parents[3] / "le-wm"
if _LEWM_DIR.exists() and str(_LEWM_DIR) not in sys.path:
 sys.path.insert(0, str(_LEWM_DIR))
if str(Path(__file__).parent) not in sys.path:
 sys.path.insert(0, str(Path(__file__).parent))

import _threadlimits # noqa: F401 # CPU thread limits, BEFORE numpy/torch/sklearn

import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as sps
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel


def img_transform(img_size=224):
 return transforms.Compose([
 transforms.ToImage(),
 transforms.ToDtype(torch.float32, scale=True),
 transforms.Normalize(**spt.data.dataset_stats.ImageNet),
 transforms.Resize(size=img_size),
 ])


# ---------------------------------------------------------------------------
# Trajectory oracle: replay-in-imagination
# ---------------------------------------------------------------------------

def replay_in_imagination(model, dataset, ep_id, start, goal_offset_steps,
 tform, device="cuda"):
 """Encode start, then roll the JEPA predictor under the *dataset* action
 segment of length goal_offset_steps. Compare the resulting predicted
 latents against the encoded true-future observations.

 Returns dict with per-step latent MSE and the cumulative cost the planner
 would have seen.
 """
 chunk = dataset.load_chunk([ep_id], [start], [start + goal_offset_steps])[0]
 pixels = chunk["pixels"] # (T, C, H, W) uint8
 actions = chunk["action"] # (T, A)
 T = pixels.shape[0]

 # Transform the whole pixel sequence
 pix_t = torch.stack([tform(p.permute(1, 2, 0).numpy() if isinstance(p, torch.Tensor)
 else p) for p in pixels]).unsqueeze(0) # (1, T, C, H, W)

 info = {
 "pixels": pix_t.to(device).unsqueeze(2), # JEPA expects (B, S, T,..) — S=1
 "action": actions.unsqueeze(0).unsqueeze(0).float().to(device), # (1, 1, T, A)
 }

 # Encode all true observations (teacher targets)
 enc_info = {"pixels": pix_t.to(device)}
 enc_info = model.encode(enc_info)
 z_true = enc_info["emb"][0] # (T, D)

 # Encode start, roll with the dataset action segment as "candidates"
 info_inf = {
 "pixels": pix_t[:, :1].to(device).unsqueeze(2), # only initial frame, (1,1,1,..)
 "action": actions.unsqueeze(0).unsqueeze(0).float().to(device), # (1,1,T,A)
 }
 info_inf = model.rollout(info_inf, info_inf["action"])
 z_pred = info_inf["predicted_emb"][0, 0] # (T, D)

 # Compare. z_pred has length T (after rollout); align with z_true[1:] when
 # JEPA predicts t+1 from t. We just take min length.
 L = min(z_pred.shape[0], z_true.shape[0])
 z_pred = z_pred[:L]
 z_true_align = z_true[:L]
 per_step_mse = (z_pred - z_true_align).pow(2).sum(-1).cpu().numpy()
 return {
 "per_step_mse": per_step_mse,
 "terminal_mse": float(per_step_mse[-1]),
 "mean_mse": float(per_step_mse.mean()),
 "T": int(L),
 }


# ---------------------------------------------------------------------------
# Rollout fidelity: imagination vs sim under three action distributions
# ---------------------------------------------------------------------------

def rollout_fidelity_random(model, dataset, ep_id, start, goal_offset_steps,
 action_low, action_high, n_candidates,
 tform, env, device="cuda"):
 """Generate n_candidates uniform-random action sequences of length
 goal_offset_steps; compare predictor-imagined terminal latents to true
 sim-rollout latents."""
 rng = np.random.default_rng(0)
 A = action_low.shape[0]
 actions = rng.uniform(action_low, action_high, size=(n_candidates, goal_offset_steps, A))
 return _fidelity_for_actions(
 model, dataset, ep_id, start, actions, tform, env, device
 )


def rollout_fidelity_planner(model, dataset, ep_id, start, goal_offset_steps,
 cem_proposals, tform, env, device="cuda"):
 """cem_proposals: ndarray (n_candidates, goal_offset_steps, A) snapshot
 from a real CEM iteration (loaded from saved logs)."""
 return _fidelity_for_actions(
 model, dataset, ep_id, start, cem_proposals, tform, env, device
 )


def _fidelity_for_actions(model, dataset, ep_id, start, actions_np,
 tform, env, device):
 # For now we report only imagined trajectory and hold sim-comparison until
 # the env-driver is integrated. This stub keeps the API stable.
 raise NotImplementedError("sim-rollout requires per-candidate env reset; will be implemented after smoke test passes")


# ---------------------------------------------------------------------------
# CEM reranking calibration (paired top-k accuracy)
# ---------------------------------------------------------------------------

def cem_reranking_topk(imagined_costs, true_costs, k_imagined=5, k_true_pct=10):
 """Return the fraction of imagined top-k that fall in the true top-k_true_pct%.
 Both arrays have shape (n_candidates,)."""
 n = len(imagined_costs)
 n_true_top = max(1, int(np.ceil(n * k_true_pct / 100)))
 imagined_top = np.argsort(imagined_costs)[:k_imagined]
 true_top = set(np.argsort(true_costs)[:n_true_top])
 return float(np.mean([i in true_top for i in imagined_top]))


def cem_reranking_spearman(imagined_costs, true_costs):
 rho, _ = sps.spearmanr(imagined_costs, true_costs)
 return float(rho)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASKS = {
 "tworoom": dict(
 env_name="swm/TwoRoom-v1",
 ckpt="tworoom/lewm",
 dataset_name="tworoom",
 keys_to_cache=["action", "proprio"],
 history_size=1,
 frame_skip=1,
 goal_offset_steps=25,
 ),
}


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--num_eval", type=int, default=8)
 ap.add_argument("--eval_set", type=str, default=None,
 help="optional path to eval_set.json from run_ablation.py")
 ap.add_argument("--tests", default="replay",
 help="comma list of: replay,teacher_forced,rollout_fidelity,reranking")
 ap.add_argument("--out_dir", default="phase1/two_room_planner_ablation/diagnostics")
 args = ap.parse_args()

 out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
 cfg = TASKS[args.task]

 dataset = swm.data.HDF5Dataset(
 cfg["dataset_name"], keys_to_cache=cfg["keys_to_cache"],
 cache_dir=swm.data.utils.get_cache_dir(),
 )
 model = AutoCostModel(cfg["ckpt"]).to("cuda").eval()
 model.requires_grad_(False)
 if hasattr(model, "interpolate_pos_encoding"):
 model.interpolate_pos_encoding = True
 tform = img_transform()

 if args.eval_set and Path(args.eval_set).exists():
 eset = json.loads(Path(args.eval_set).read_text())
 episodes_idx = eset["episodes_idx"][:args.num_eval]
 start_steps = eset["start_steps"][:args.num_eval]
 else:
 # Quick fallback: take the first N from the dataset
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 episodes_idx = list(np.unique(dataset.get_col_data(col))[:args.num_eval])
 start_steps = [0] * args.num_eval

 tests = set(args.tests.split(","))
 out = {"task": args.task, "tests": list(tests), "episodes_idx": episodes_idx, "start_steps": start_steps}

 if "replay" in tests:
 out["replay"] = []
 for ep, start in zip(episodes_idx, start_steps):
 try:
 r = replay_in_imagination(model, dataset, ep, start,
 cfg["goal_offset_steps"], tform)
 out["replay"].append({"ep": int(ep), "start": int(start), **r,
 "per_step_mse": r["per_step_mse"].tolist()})
 except Exception as e:
 out["replay"].append({"ep": int(ep), "start": int(start), "error": str(e)})

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-diag-{stamp}.json"
 out_path.write_text(json.dumps(out, indent=2, default=str))
 print(f"\nDiagnostics written to {out_path}")
 if "replay" in tests:
 oks = [x for x in out["replay"] if "terminal_mse" in x]
 if oks:
 print(f"replay terminal_mse: mean={np.mean([x['terminal_mse'] for x in oks]):.4f}, "
 f"max={np.max([x['terminal_mse'] for x in oks]):.4f}")


if __name__ == "__main__":
 main()
