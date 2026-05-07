"""Probe-proxy oracle-state cost.

Train a linear regressor z -> agent_xy from offline data (cheap, ~1 min).
At plan time, replace JEPA's criterion with: cost = euclidean(probe(z_pred), goal_xy).
Tests whether geometry-aware cost (not raw latent L2) breaks the seed
fragility on episode 4339.

This is the "oracle-state geodesic cost" diagnostic reviewer requested. It's
not literal oracle (we still use the predictor, not the sim, to roll forward),
but it replaces the latent-L2 cost with a *task-relevant* distance, which is
the closest cheap surrogate to true geodesic.
"""
from __future__ import annotations
import argparse, json, os, sys, time, types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_LEWM_DIR = Path(__file__).resolve().parents[3] / "le-wm"
sys.path.insert(0, str(_LEWM_DIR))
sys.path.insert(0, str(Path(__file__).parent))

import _threadlimits # noqa: F401 # CPU thread limits, BEFORE numpy/torch/sklearn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy, AutoCostModel
from stable_worldmodel.solver import ICEMSolver

from run_ablation import (
 TASKS, default_cells, build_processors, build_world,
 img_transform, _icem_configure_with_action_block,
)
from train_q_psi import encode_dataset_sample


# ---------------------------------------------------------------------------
# Train linear probe z -> proprio on a sample of frames
# ---------------------------------------------------------------------------

@torch.inference_mode()
def fit_probe(model, dataset, n_episodes, frame_stride, tform, device="cuda"):
 print(f"== Encoding {n_episodes} eps for probe ==", flush=True)
 cache = encode_dataset_sample(model, dataset, n_episodes, frame_stride,
 25, 5, tform, device)
 # Now load proprio for the same frames
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 Z, P = [], []
 for ep, c in cache.items():
 steps = c["steps"]
 chunk = dataset.load_chunk([int(ep)], [0], [int(steps.max() + 1)])[0]
 proprio_full = chunk["proprio"]
 if torch.is_tensor(proprio_full):
 proprio_full = proprio_full.numpy()
 proprio_sub = proprio_full[steps]
 Z.append(c["latents"]); P.append(proprio_sub)
 Z = np.concatenate(Z, 0)
 P = np.concatenate(P, 0)
 print(f" fit Ridge on {Z.shape[0]} samples (z dim={Z.shape[1]}, p dim={P.shape[1]})",
 flush=True)
 reg = Ridge(alpha=1e-2)
 reg.fit(Z, P)
 pred = reg.predict(Z)
 err = np.sqrt(((pred - P)**2).sum(-1))
 print(f" probe train MSE: mean px err = {err.mean():.2f}, max = {err.max():.2f}",
 flush=True)
 W = torch.tensor(reg.coef_, dtype=torch.float32, device=device) # (P, Z)
 b = torch.tensor(reg.intercept_, dtype=torch.float32, device=device) # (P,)
 return W, b, float(err.mean())


# ---------------------------------------------------------------------------
# Probe-cost criteria
# ---------------------------------------------------------------------------

def make_probe_terminal(W, b, goal_xy_callable):
 def crit(self, info_dict):
 pred_emb = info_dict["predicted_emb"] # (B, S, T, D)
 z = pred_emb[.., -1, :] # (B, S, D)
 # z @ W.T + b → predicted xy of shape (B, S, 2)
 xy_pred = z @ W.T + b
 goal_xy = goal_xy_callable(info_dict) # (B, 2)
 if goal_xy.dim() == xy_pred.dim() - 1:
 goal_xy = goal_xy.unsqueeze(1)
 cost = ((xy_pred - goal_xy.expand_as(xy_pred)) ** 2).sum(-1) # (B, S)
 return cost
 return crit


def make_probe_cumulative(W, b, goal_xy_callable):
 def crit(self, info_dict):
 pred_emb = info_dict["predicted_emb"] # (B, S, T, D)
 # Apply W,b to each time-step
 xy_pred_all = pred_emb @ W.T + b # (B, S, T, 2)
 goal_xy = goal_xy_callable(info_dict) # (B, 2)
 if goal_xy.dim() == 2:
 goal_xy = goal_xy.unsqueeze(1).unsqueeze(1) # (B, 1, 1, 2)
 elif goal_xy.dim() == 3:
 goal_xy = goal_xy.unsqueeze(2) # (B, S, 1, 2)
 cost = ((xy_pred_all - goal_xy) ** 2).sum(-1).sum(-1) # sum over T, then xy
 return cost
 return crit


def goal_xy_from_info(info_dict):
 """Pull goal proprio from info_dict if available, else use goal-emb decoded.
 The TwoRoom env exposes 'goal_proprio'."""
 if "goal_proprio" in info_dict:
 gp = info_dict["goal_proprio"]
 if torch.is_tensor(gp):
 return gp[:, 0] if gp.dim() == 3 else gp
 raise KeyError("goal_proprio not in info_dict")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--cell", default="A5")
 ap.add_argument("--cost", choices=["terminal", "cumulative"], default="cumulative")
 ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10])
 ap.add_argument("--n_probe_eps", type=int, default=200)
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/diagnostics")
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

 if not getattr(ICEMSolver, "_patched_for_action_block", False):
 ICEMSolver.configure = _icem_configure_with_action_block(ICEMSolver.configure)
 ICEMSolver._patched_for_action_block = True

 tform = img_transform()
 W, b, train_err = fit_probe(model, dataset, args.n_probe_eps, 5, tform)

 if args.cost == "terminal":
 model.criterion = types.MethodType(make_probe_terminal(W, b, goal_xy_from_info), model)
 else:
 model.criterion = types.MethodType(make_probe_cumulative(W, b, goal_xy_from_info), model)

 cell = next(c for c in default_cells(cfg["action_block"]) if c["name"] == args.cell)
 transform_dict = {"pixels": img_transform(), "goal": img_transform()}
 processors = build_processors(dataset, cfg["keys_to_cache"])

 results = []
 for s in args.seeds:
 solver = ICEMSolver(
 model=model, batch_size=1, num_samples=cell["num_samples"],
 var_scale=1.0, n_steps=cell["n_steps"], topk=cell["topk"],
 noise_beta=cell["noise_beta"], alpha=cell["alpha"],
 n_elite_keep=cell["n_elite_keep"], return_mean=True,
 device="cuda", seed=int(s),
 )
 plan_cfg = PlanConfig(
 horizon=cell["horizon"], receding_horizon=cell["receding_horizon"],
 history_len=cfg["history_size"], action_block=cell["action_block"],
 warm_start=cell["warm_start"],
 )
 policy = WorldModelPolicy(solver=solver, config=plan_cfg,
 process=processors, transform=transform_dict)
 world = build_world(cfg, num_envs=1)
 world.set_policy(policy)
 t0 = time.time()
 m = world.evaluate_from_dataset(
 dataset, start_steps=[args.start], goal_offset_steps=cfg["goal_offset_steps"],
 eval_budget=cfg["eval_budget"], episodes_idx=[args.episode],
 callables=cfg["callables"], save_video=False, video_path="/tmp",
 )
 dt = time.time() - t0
 success = bool(m["episode_successes"][0])
 results.append({"seed": int(s), "success": success, "wall_s": dt})
 print(f" seed={s} success={success} wall={dt:.1f}s", flush=True)

 rate = 100.0 * sum(r["success"] for r in results) / len(results)
 print(f"\nProbe-{args.cost} on episode {args.episode} ({args.cell}): "
 f"{sum(r['success'] for r in results)}/{len(results)} = {rate:.1f}% "
 f"across {len(results)} CEM seeds")

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-probe-{args.cost}-ep{args.episode}-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "cell": args.cell, "cost_variant": args.cost,
 "probe_train_pixel_err_mean": train_err,
 "results": results,
 "success_count": sum(r["success"] for r in results),
 "success_rate_pct": rate,
 }, indent=2))
 print(f"Written to {out_path}")


if __name__ == "__main__":
 main()
