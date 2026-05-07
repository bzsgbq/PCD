"""Phase 1 ablation driver: TwoRoom planner sweep with paired seeds and no video.

Each cell defines (solver, cost_shape, plan_config) on top of a frozen LeWM
checkpoint. We use the same evaluation episodes and start indices across all
cells so that cell-to-cell deltas are not confounded by goal-set variance.

Cells (mirrors):
 A0 CEM horizon=5 RH=5 cost=terminal (paper baseline)
 A1 CEM horizon=5 RH=1 cost=terminal
 A2 CEM horizon=5 RH=1 cost=cumulative
 A3 iCEM horizon=5 RH=1 cost=cumulative
 A4 iCEM horizon=5 RH=1 cost=cumulative warm_start=True
 A5 iCEM horizon=10 RH=1 cost=cumulative warm_start=True

Usage:
 python run_ablation.py --task tworoom --num_eval 16 --cells A0,A1,A2,A3,A4,A5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# LeWM checkpoints pickle classes from le-wm/jepa.py and le-wm/module.py.
# Make them importable.
_LEWM_DIR = Path(__file__).resolve().parents[3] / "le-wm"
if _LEWM_DIR.exists() and str(_LEWM_DIR) not in sys.path:
 sys.path.insert(0, str(_LEWM_DIR))
if str(Path(__file__).parent) not in sys.path:
 sys.path.insert(0, str(Path(__file__).parent))

import _threadlimits # noqa: F401 # CPU thread limits, BEFORE numpy/torch/sklearn

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy, AutoCostModel
from stable_worldmodel.solver import CEMSolver, ICEMSolver

# ---------------------------------------------------------------------------
# Task definitions (mirror le-wm/config/eval/*.yaml)
# ---------------------------------------------------------------------------

TASKS = {
 "tworoom": dict(
 env_name="swm/TwoRoom-v1",
 ckpt="tworoom/lewm",
 dataset_name="tworoom",
 keys_to_cache=["action", "proprio"],
 callables=[
 {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
 {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
 ],
 action_block=5, # frame_skip is 1 in world; planner action_block=5 == repeat 5x
 history_size=1,
 frame_skip=1,
 goal_offset_steps=25,
 eval_budget=50,
 ),
 "pusht": dict(
 env_name="swm/PushT-v1",
 ckpt="pusht/lewm",
 dataset_name="pusht_expert_train",
 keys_to_cache=["action", "proprio", "state"],
 callables=[
 {"method": "_set_state", "args": {"state": {"value": "state"}}},
 {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
 ],
 action_block=5,
 history_size=1,
 frame_skip=1,
 goal_offset_steps=25,
 eval_budget=50,
 ),
 "reacher": dict(
 env_name="swm/ReacherDMControl-v0",
 ckpt="reacher/lewm",
 dataset_name="dmc/reacher_random",
 # Official le-wm/config/eval/reacher.yaml caches only ["action"]; callables read
 # qpos/qvel via the dataset's default column loader. Keeping the cache list
 # aligned with the official config avoids silent processor-mutation differences
 # in WorldModelPolicy._prepare_info.
 keys_to_cache=["action"],
 task="qpos_match",
 callables=[
 {"method": "set_state", "args": {"qpos": {"value": "qpos"},
 "qvel": {"value": "qvel"}}},
 {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
 ],
 action_block=5,
 history_size=1,
 frame_skip=1,
 goal_offset_steps=25,
 eval_budget=50,
 ),
 "cube": dict(
 env_name="swm/OGBCube-v0",
 ckpt="cube/lewm",
 dataset_name="ogbench/cube_single_expert",
 # Official le-wm/config/eval/cube.yaml caches only ["action"]. The set_state /
 # set_target_pos callables still read qpos/qvel/privileged_block_0_* via the
 # dataset's default column loader; we don't need processors for those columns.
 # Keeping the cache list aligned with the official config.
 keys_to_cache=["action"],
 env_type="single",
 ob_type="states",
 terminate_at_goal=True,
 callables=[
 {"method": "set_state",
 "args": {"qpos": {"value": "qpos"},
 "qvel": {"value": "qvel"}}},
 {"method": "set_target_pos",
 "args": {"cube_id": {"value": 0, "in_dataset": False},
 "target_pos": {"value": "goal_privileged_block_0_pos"},
 "target_quat": {"value": "goal_privileged_block_0_quat"}}},
 ],
 action_block=5,
 history_size=1,
 frame_skip=1,
 goal_offset_steps=25,
 eval_budget=50,
 ),
}


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

def base_solver_kwargs(batch_size, num_samples=300, n_steps=30, topk=30, seed=42):
 return dict(
 batch_size=batch_size,
 num_samples=num_samples,
 var_scale=1.0,
 n_steps=n_steps,
 topk=topk,
 device="cuda",
 seed=seed,
 )


def make_cell(name, solver, horizon, receding_horizon, action_block,
 cost_shape, warm_start, num_samples, n_steps, topk,
 noise_beta=2.0, alpha=0.1, n_elite_keep=5):
 return dict(
 name=name,
 solver=solver, # 'cem' or 'icem'
 horizon=horizon,
 receding_horizon=receding_horizon,
 action_block=action_block,
 cost_shape=cost_shape, # 'terminal' or 'cumulative'
 warm_start=warm_start,
 num_samples=num_samples,
 n_steps=n_steps,
 topk=topk,
 noise_beta=noise_beta,
 alpha=alpha,
 n_elite_keep=n_elite_keep,
 )


def default_cells(action_block):
 return [
 make_cell("A0", "cem", 5, 5, action_block, "terminal", False, 300, 30, 30),
 make_cell("A1", "cem", 5, 1, action_block, "terminal", False, 300, 30, 30),
 make_cell("A2", "cem", 5, 1, action_block, "cumulative", False, 300, 30, 30),
 make_cell("A3", "icem", 5, 1, action_block, "cumulative", False, 300, 30, 30),
 make_cell("A4", "icem", 5, 1, action_block, "cumulative", True, 300, 30, 30),
 make_cell("A5", "icem", 10, 1, action_block, "cumulative", True, 300, 30, 30),
 # horizon-isolation counterfactuals 
 make_cell("A0_H10", "cem", 10, 5, action_block, "terminal", False, 300, 30, 30),
 make_cell("A2_H10", "cem", 10, 1, action_block, "cumulative", False, 300, 30, 30),
 ]


# ---------------------------------------------------------------------------
# Cost-shape patch
#
# The LeWM JEPA's terminal criterion (le-wm/jepa.py) is:
# cost = F.mse_loss(pred_emb[.., -1:, :],
# goal_emb[.., -1:, :].detach(),
# reduction='none').sum(dim=tuple(range(2, pred_emb.ndim)))
# Our cumulative version evaluates MSE at every predicted time-step and sums.
# ---------------------------------------------------------------------------

def _broadcast_goal(goal_emb, pred_emb):
 """goal_emb may be (B, T_g, D) or (B, S, T_g, D); pred_emb is (B, S, T_p, D).
 Return goal_emb_terminal of shape (B, S, 1, D) suitable for broadcasting/expand."""
 g = goal_emb[.., -1:, :] # (.., 1, D)
 if g.dim() == pred_emb.dim() - 1: # missing the S dim
 g = g.unsqueeze(1) # (B, 1, 1, D)
 g = g.detach()
 target_shape = list(pred_emb.shape)
 # Force goal terminal to have T=1 so broadcasting works either way
 target_shape[-2] = 1
 g = g.expand(*target_shape)
 return g


def terminal_criterion(self, info_dict):
 pred_emb = info_dict["predicted_emb"] # (B, S, T_pred, D)
 pred_term = pred_emb[.., -1:, :] # (B, S, 1, D)
 goal_term = _broadcast_goal(info_dict["goal_emb"], pred_emb) # (B, S, 1, D)
 cost = F.mse_loss(pred_term, goal_term, reduction="none").sum(
 dim=tuple(range(2, pred_emb.ndim))
 ) # (B, S)
 return cost


def cumulative_criterion(self, info_dict):
 pred_emb = info_dict["predicted_emb"] # (B, S, T_pred, D)
 goal_term = _broadcast_goal(info_dict["goal_emb"], pred_emb) # (B, S, 1, D)
 goal_full = goal_term.expand_as(pred_emb) # broadcast over T_pred
 cost = F.mse_loss(pred_emb, goal_full, reduction="none").sum(
 dim=tuple(range(2, pred_emb.ndim))
 ) # (B, S)
 return cost


def patch_cost_shape(model, shape):
 if shape == "terminal":
 model.criterion = types.MethodType(terminal_criterion, model)
 elif shape == "cumulative":
 model.criterion = types.MethodType(cumulative_criterion, model)
 else:
 raise ValueError(f"unknown cost shape: {shape}")
 return model


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def img_transform(img_size=224):
 return transforms.Compose([
 transforms.ToImage(),
 transforms.ToDtype(torch.float32, scale=True),
 transforms.Normalize(**spt.data.dataset_stats.ImageNet),
 transforms.Resize(size=img_size),
 ])


def get_episodes_length(dataset, episodes):
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 epi = dataset.get_col_data(col)
 sti = dataset.get_col_data("step_idx")
 return np.array([np.max(sti[epi == ep]) + 1 for ep in episodes])


def sample_eval_set(dataset, num_eval, goal_offset_steps, seed):
 """Identical sampling logic to eval.py — we reuse it so that paired seeds
 map to the same (ep_id, start_step) tuples across cells."""
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
 episode_len = get_episodes_length(dataset, ep_indices)
 max_start_idx = episode_len - goal_offset_steps - 1
 max_start_per_row = np.array(
 [max_start_idx[np.where(ep_indices == ep)[0][0]] for ep in dataset.get_col_data(col)]
 )
 valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
 valid_indices = np.nonzero(valid_mask)[0]
 g = np.random.default_rng(seed)
 chosen = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
 chosen = np.sort(valid_indices[chosen])
 rows = dataset.get_row_data(chosen)
 return rows[col].tolist(), rows["step_idx"].tolist()


def build_world(cfg, num_envs):
 return swm.World(
 env_name=cfg["env_name"],
 num_envs=num_envs,
 image_shape=(224, 224),
 history_size=cfg["history_size"],
 frame_skip=cfg["frame_skip"],
 max_episode_steps=2 * cfg["eval_budget"],
 **{k: v for k, v in cfg.items() if k in ("env_type", "ob_type", "multiview",
 "width", "height", "visualize_info",
 "terminate_at_goal", "task")},
 )


def build_processors(dataset, keys_to_cache):
 process = {}
 for col in keys_to_cache:
 if col == "pixels":
 continue
 proc = preprocessing.StandardScaler()
 data = dataset.get_col_data(col)
 data = data[~np.isnan(data).any(axis=1)]
 proc.fit(data)
 process[col] = proc
 if col != "action":
 process[f"goal_{col}"] = proc
 return process


def _icem_configure_with_action_block(orig_configure):
 """Wrap ICEMSolver.configure to repeat _action_low/_action_high by
 action_block, matching the candidate dimensionality used downstream."""
 def configure(self, *, action_space, n_envs, config):
 orig_configure(self, action_space=action_space, n_envs=n_envs, config=config)
 if self._action_low is not None and config.action_block > 1:
 self._action_low = self._action_low.repeat(config.action_block)
 self._action_high = self._action_high.repeat(config.action_block)
 return configure


def build_solver(model, cell, batch_size, solver_seed=42):
 base = base_solver_kwargs(
 batch_size=batch_size,
 num_samples=cell["num_samples"],
 n_steps=cell["n_steps"],
 topk=cell["topk"],
 seed=solver_seed,
 )
 if cell["solver"] == "cem":
 return CEMSolver(model=model, **base)
 if cell["solver"] == "icem":
 # Patch upstream bug: _action_low shape doesn't account for action_block.
 if not getattr(ICEMSolver, "_patched_for_action_block", False):
 ICEMSolver.configure = _icem_configure_with_action_block(ICEMSolver.configure)
 ICEMSolver._patched_for_action_block = True
 return ICEMSolver(
 model=model,
 noise_beta=cell["noise_beta"],
 alpha=cell["alpha"],
 n_elite_keep=cell["n_elite_keep"],
 return_mean=True,
 **base,
 )
 raise ValueError(cell["solver"])


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------

def run_cell(task_cfg, cell, model, dataset, processors, transform_dict,
 episodes_idx, start_steps, num_envs, save_video, video_path,
 solver_batch_size=None, solver_seed=42):
 """Run one ablation cell.

 solver_batch_size: if None, defaults to num_envs (batched grid execution).
 Pass 1 for exact-official reproduction (matches le-wm/config/eval/solver/cem.yaml).

 solver_seed: CEM/iCEM solver seed. The Hydra eval couples this to the
 eval-set sampler seed via solver/cem.yaml: seed: ${seed}; pass --seed=s
 --solver_seed=s on the CLI to reproduce that coupling.
 """
 sbs = solver_batch_size if solver_batch_size is not None else num_envs
 print(f"\n=== Cell {cell['name']} solver={cell['solver']} "
 f"H={cell['horizon']} RH={cell['receding_horizon']} "
 f"cost={cell['cost_shape']} warm={cell['warm_start']} "
 f"solver_batch_size={sbs} solver_seed={solver_seed} ===", flush=True)

 patch_cost_shape(model, cell["cost_shape"])

 plan_cfg = PlanConfig(
 horizon=cell["horizon"],
 receding_horizon=cell["receding_horizon"],
 history_len=task_cfg["history_size"],
 action_block=cell["action_block"],
 warm_start=cell["warm_start"],
 )
 solver = build_solver(model, cell, batch_size=sbs, solver_seed=solver_seed)
 policy = WorldModelPolicy(
 solver=solver, config=plan_cfg, process=processors, transform=transform_dict,
 )

 world = build_world(task_cfg, num_envs)
 world.set_policy(policy)

 t0 = time.time()
 metrics = world.evaluate_from_dataset(
 dataset,
 start_steps=start_steps,
 goal_offset_steps=task_cfg["goal_offset_steps"],
 eval_budget=task_cfg["eval_budget"],
 episodes_idx=episodes_idx,
 callables=task_cfg["callables"],
 save_video=save_video,
 video_path=video_path,
 )
 dt = time.time() - t0
 metrics = {
 k: (v.tolist() if hasattr(v, "tolist") else v)
 for k, v in metrics.items()
 }
 metrics["wall_time_s"] = dt
 metrics["plan_len_env_steps"] = cell["horizon"] * cell["action_block"]
 metrics["cell"] = cell["name"]
 return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom", choices=list(TASKS.keys()))
 ap.add_argument("--num_eval", type=int, default=16)
 ap.add_argument("--seed", type=int, default=42, help="seed for eval-set sampling")
 ap.add_argument("--solver_seed", type=int, default=42,
 help="CEM/iCEM solver seed. The Hydra eval couples this to --seed via "
 "solver/cem.yaml: seed: ${seed}; pass --seed=s --solver_seed=s to "
 "reproduce the official coupling. Default 42 preserves prior behavior.")
 ap.add_argument("--solver_batch_size", type=int, default=None,
 help="CEM/iCEM solver batch_size. Default None = use num_eval (batched grid). "
 "Pass 1 for exact-official reproduction (matches le-wm/config/eval/solver/cem.yaml).")
 ap.add_argument("--cells", type=str, default="A0,A1,A2,A3,A4,A5",
 help="comma-separated cell names to run")
 ap.add_argument("--out_dir", type=str, default="phase1/two_room_planner_ablation")
 ap.add_argument("--save_video", action="store_true")
 args = ap.parse_args()

 out_dir = Path(args.out_dir)
 (out_dir / "logs").mkdir(parents=True, exist_ok=True)
 (out_dir / "results").mkdir(parents=True, exist_ok=True)
 video_path = out_dir / "videos"
 video_path.mkdir(parents=True, exist_ok=True)

 task_cfg = TASKS[args.task]

 # Load dataset and frozen LeWM checkpoint
 dataset = swm.data.HDF5Dataset(
 task_cfg["dataset_name"],
 keys_to_cache=task_cfg["keys_to_cache"],
 cache_dir=swm.data.utils.get_cache_dir(),
 )
 print(f"loaded dataset {task_cfg['dataset_name']}", flush=True)

 model = AutoCostModel(task_cfg["ckpt"]).to("cuda").eval()
 model.requires_grad_(False)
 if hasattr(model, "interpolate_pos_encoding"):
 model.interpolate_pos_encoding = True
 print(f"loaded checkpoint {task_cfg['ckpt']}", flush=True)

 # Paired sampling — same eval set for every cell
 episodes_idx, start_steps = sample_eval_set(
 dataset, args.num_eval, task_cfg["goal_offset_steps"], args.seed,
 )
 print(f"eval set: {args.num_eval} episodes, sampled with seed={args.seed}", flush=True)
 print(f" episodes_idx[:5] = {episodes_idx[:5]}")
 print(f" start_steps[:5] = {start_steps[:5]}")

 transform_dict = {"pixels": img_transform(), "goal": img_transform()}
 processors = build_processors(dataset, task_cfg["keys_to_cache"])

 cells = [c for c in default_cells(task_cfg["action_block"])
 if c["name"] in args.cells.split(",")]

 results = []
 for cell in cells:
 try:
 m = run_cell(task_cfg, cell, model, dataset, processors, transform_dict,
 episodes_idx, start_steps, args.num_eval,
 save_video=args.save_video, video_path=str(video_path),
 solver_batch_size=args.solver_batch_size,
 solver_seed=args.solver_seed)
 results.append(m)
 print(f"--> cell {cell['name']}: success_rate={m.get('success_rate')!r} "
 f"wall={m['wall_time_s']:.1f}s")
 except Exception as e:
 import traceback
 print(f"!!! cell {cell['name']} FAILED:")
 traceback.print_exc()
 results.append({"cell": cell["name"], "error": str(e)})

 # Persist
 stamp = time.strftime("%Y%m%d-%H%M%S")
 rdir = out_dir / "results" / f"{args.task}-{stamp}"
 rdir.mkdir(parents=True, exist_ok=True)
 (rdir / "config.json").write_text(json.dumps({
 "task": args.task, "num_eval": args.num_eval,
 "seed": args.seed, "solver_seed": args.solver_seed,
 "solver_batch_size": args.solver_batch_size, # None means batched-grid (= num_eval)
 "cells": cells, "task_cfg": task_cfg,
 }, indent=2, default=str))
 (rdir / "results.json").write_text(json.dumps(results, indent=2, default=str))
 (rdir / "eval_set.json").write_text(json.dumps({
 "episodes_idx": list(episodes_idx), "start_steps": list(start_steps),
 }, indent=2, default=str))

 # CSV summary
 rows = ["cell,solver,horizon,RH,cost,warm,success_rate,wall_s"]
 for cell, m in zip(cells, results):
 sr = m.get("success_rate", "ERR")
 wt = m.get("wall_time_s", "")
 rows.append(",".join(map(str, [
 cell["name"], cell["solver"], cell["horizon"], cell["receding_horizon"],
 cell["cost_shape"], cell["warm_start"], sr, wt,
 ])))
 (rdir / "summary.csv").write_text("\n".join(rows) + "\n")

 print(f"\nResults written to {rdir}/")
 for line in rows:
 print(line)

 # Propagate per-cell failures so shell launchers (set -e) don't silently
 # mark a partially-failed grid as success (note). Diagnostic
 # JSON is still written above for forensics.
 failed = [r for r in results if "error" in r]
 if failed:
 names = ", ".join(r.get("cell", "?") for r in failed)
 print(f"\n!!! {len(failed)} cell(s) failed: {names}", flush=True)
 raise SystemExit(1)


if __name__ == "__main__":
 main()
