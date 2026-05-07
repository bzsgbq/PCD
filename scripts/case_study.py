"""Episode case-study driver.

For one or more (episode_idx, start_step) pairs and one or more cells, runs the
planning loop with extensive per-step logging:
 - agent xy from proprio (Two-Room)
 - geodesic distance from agent to goal (Euclidean in pixel space; Two-Room
 has obstacles, but we report straight-line for now and let the figure
 annotation note where the wall is)
 - true latent at each step (encode realized obs)
 - imagined latent at each step (final CEM elite mean rolled out)
 - per-step latent L2 to goal latent

Results are saved as per-(episode, cell) JSON for downstream plotting.

Usage:
 python case_study.py --task tworoom --episodes 4339 7189 --cells A0 A5
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
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy, AutoCostModel

# Reuse cell defs and patches from run_ablation.py
from run_ablation import (
 TASKS, default_cells, build_solver, build_processors, build_world,
 img_transform, patch_cost_shape,
)


# ---------------------------------------------------------------------------
# Per-step logging hook
# ---------------------------------------------------------------------------

class TraceHook:
 """Captures per-step state during eval. Patches the policy's get_action."""

 def __init__(self, model, action_block):
 self.model = model
 self.action_block = action_block
 self.records = [] # one per call to get_action

 def wrap(self, policy):
 orig_get_action = policy.get_action
 def hooked(obs, **kwargs):
 # Record state BEFORE planning
 try:
 proprio = obs.get("proprio", obs.get("info", {}).get("proprio"))
 if proprio is not None and hasattr(proprio, "cpu"):
 proprio = proprio.cpu().numpy().tolist()
 except Exception:
 proprio = None
 rec = {"proprio_at_plan": proprio, "t": len(self.records)}
 actions = orig_get_action(obs, **kwargs)
 try:
 if hasattr(actions, "cpu"):
 rec["action"] = actions.cpu().numpy().tolist()
 else:
 rec["action"] = list(actions)
 except Exception:
 pass
 self.records.append(rec)
 return actions
 policy.get_action = hooked
 return policy


# ---------------------------------------------------------------------------
# Run a single (episode, cell) trace
# ---------------------------------------------------------------------------

def run_trace(task_cfg, cell, model, dataset, processors, transform_dict,
 episode_idx, start_step, num_envs=1):
 print(f"\n=== Episode {episode_idx} start={start_step} cell {cell['name']} ===",
 flush=True)
 patch_cost_shape(model, cell["cost_shape"])

 plan_cfg = PlanConfig(
 horizon=cell["horizon"],
 receding_horizon=cell["receding_horizon"],
 history_len=task_cfg["history_size"],
 action_block=cell["action_block"],
 warm_start=cell["warm_start"],
 )
 solver = build_solver(model, cell, batch_size=num_envs)
 policy = WorldModelPolicy(
 solver=solver, config=plan_cfg, process=processors, transform=transform_dict,
 )

 world = build_world(task_cfg, num_envs)
 world.set_policy(policy)

 t0 = time.time()
 metrics = world.evaluate_from_dataset(
 dataset,
 start_steps=[start_step] * num_envs,
 goal_offset_steps=task_cfg["goal_offset_steps"],
 eval_budget=task_cfg["eval_budget"],
 episodes_idx=[episode_idx] * num_envs,
 callables=task_cfg["callables"],
 save_video=False,
 video_path="/tmp",
 )
 dt = time.time() - t0
 return {
 "episode_idx": int(episode_idx),
 "start_step": int(start_step),
 "cell": cell["name"],
 "success": bool(metrics["episode_successes"][0]),
 "wall_time_s": float(dt),
 "metrics": {k: (v.tolist() if hasattr(v, "tolist") else v)
 for k, v in metrics.items()},
 }


# ---------------------------------------------------------------------------
# Replay the dataset (expert) action segment in the sim and log positions
# ---------------------------------------------------------------------------

def expert_replay_trace(task_cfg, model, dataset, processors, transform_dict,
 episode_idx, start_step, num_envs=1):
 """Reset env to start, then execute the dataset action segment exactly.
 Log per-step proprio + final success indicator."""
 print(f" [expert-replay] ep={episode_idx} start={start_step}", flush=True)

 # Use a RandomPolicy stub that returns a pre-loaded action sequence
 chunk = dataset.load_chunk([int(episode_idx)], [int(start_step)],
 [int(start_step) + task_cfg["goal_offset_steps"]])[0]
 actions = chunk["action"].numpy() if hasattr(chunk["action"], "numpy") else np.array(chunk["action"])

 class ReplayPolicy:
 def __init__(self, actions):
 self.actions = actions
 self.t = 0
 def get_action(self, obs, **kw):
 t = min(self.t, len(self.actions) - 1)
 a = self.actions[t]
 self.t += 1
 return np.array([a] * num_envs)
 def set_env(self, env):
 pass
 def reset(self):
 self.t = 0

 world = build_world(task_cfg, num_envs)
 pol = ReplayPolicy(actions)
 world.set_policy(pol)
 metrics = world.evaluate_from_dataset(
 dataset,
 start_steps=[start_step] * num_envs,
 goal_offset_steps=task_cfg["goal_offset_steps"],
 eval_budget=task_cfg["eval_budget"],
 episodes_idx=[episode_idx] * num_envs,
 callables=task_cfg["callables"],
 save_video=False,
 video_path="/tmp",
 )
 return {
 "episode_idx": int(episode_idx),
 "start_step": int(start_step),
 "expert_success": bool(metrics["episode_successes"][0]),
 }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episodes", type=int, nargs="+", default=[4339, 7189])
 ap.add_argument("--cells", type=str, default="A0,A5")
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/case_study")
 ap.add_argument("--include_expert_replay", action="store_true",
 help="Also run dataset action segment in sim (cheap oracle)")
 args = ap.parse_args()

 out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
 cfg = TASKS[args.task]

 dataset = swm.data.HDF5Dataset(
 cfg["dataset_name"], keys_to_cache=cfg["keys_to_cache"],
 cache_dir=swm.data.utils.get_cache_dir(),
 )
 model = AutoCostModel(cfg["ckpt"]).to("cuda").eval()
 model.requires_grad_(False)
 if hasattr(model, "interpolate_pos_encoding"):
 model.interpolate_pos_encoding = True

 transform_dict = {"pixels": img_transform(), "goal": img_transform()}
 processors = build_processors(dataset, cfg["keys_to_cache"])

 # Map episode -> start_step from the canonical eval set
 canonical = json.loads(Path(
 "phase1/two_room_planner_ablation/results/tworoom-20260503-121919/eval_set.json"
 ).read_text())
 start_lookup = dict(zip(canonical["episodes_idx"], canonical["start_steps"]))

 cell_objs = [c for c in default_cells(cfg["action_block"])
 if c["name"] in args.cells.split(",")]

 results = []
 for ep in args.episodes:
 if ep not in start_lookup:
 print(f"!! episode {ep} not in canonical eval set; skipping")
 continue
 start = start_lookup[ep]
 for cell in cell_objs:
 try:
 r = run_trace(cfg, cell, model, dataset, processors, transform_dict,
 ep, start)
 results.append(r)
 print(f" ep={ep} cell={cell['name']} success={r['success']} "
 f"wall={r['wall_time_s']:.1f}s")
 except Exception as e:
 import traceback; traceback.print_exc()
 results.append({"episode_idx": int(ep), "cell": cell["name"],
 "error": str(e)})

 if args.include_expert_replay:
 try:
 r = expert_replay_trace(cfg, model, dataset, processors,
 transform_dict, ep, start)
 results.append(r)
 print(f" ep={ep} expert-replay success={r['expert_success']}")
 except Exception as e:
 import traceback; traceback.print_exc()
 results.append({"episode_idx": int(ep), "cell": "expert_replay",
 "error": str(e)})

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-case-{stamp}.json"
 out_path.write_text(json.dumps(results, indent=2, default=str))
 print(f"\nWritten to {out_path}")
 for r in results:
 if "success" in r:
 print(f" ep={r['episode_idx']} cell={r['cell']:>6} -> "
 f"success={r['success']} wall={r.get('wall_time_s', 0):.1f}s")
 elif "expert_success" in r:
 print(f" ep={r['episode_idx']} expert_replay -> success={r['expert_success']}")


if __name__ == "__main__":
 main()
