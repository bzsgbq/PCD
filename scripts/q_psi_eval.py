"""Evaluate Q_psi as the planner cost on a single episode across CEM seeds.

Replaces the JEPA's criterion with -log Q_psi(z_pred, z_g, h_remaining).
Then runs the A5-style protocol (iCEM, RH=1, warm_start=True, H=10) on
episode 4339 across 10 CEM seeds. Compare the success rate to A5's
5/10 = 50% baseline.

Two cost variants:
 - terminal: cost = -log Q_psi(z_pred_terminal, z_goal_terminal, H)
 - cumulative: cost = sum over t of -log Q_psi(z_pred_t, z_goal_terminal, H-t)
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
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy, AutoCostModel
from stable_worldmodel.solver import ICEMSolver

from run_ablation import (
 TASKS, default_cells, build_processors, build_world,
 img_transform, _icem_configure_with_action_block,
)
from train_q_psi import QPsi


# ---------------------------------------------------------------------------
# Q-as-cost criterion — patches JEPA.criterion to use Q_psi
# ---------------------------------------------------------------------------

def make_q_terminal_criterion(q_psi):
 def crit(self, info_dict):
 pred_emb = info_dict["predicted_emb"] # (B, S, T_pred, D)
 goal_emb = info_dict["goal_emb"] # (B, T_g, D) or (B, S, T_g, D)
 z_pred = pred_emb[.., -1, :] # (B, S, D)
 # Add S dim to goal if needed
 g = goal_emb[.., -1, :] # (.., D)
 if g.dim() == z_pred.dim() - 1: # missing S
 g = g.unsqueeze(1)
 goal_t = g.expand_as(z_pred) # (B, S, D)
 B, S, D = z_pred.shape
 h = float(pred_emb.shape[-2]) # latent horizon
 h_tensor = torch.full((B*S,), h, device=z_pred.device)
 logits = q_psi(z_pred.reshape(B*S, D), goal_t.reshape(B*S, D), h_tensor)
 # -log P(reach) = softplus(-logit)
 cost = F.softplus(-logits).reshape(B, S)
 return cost
 return crit


def make_q_cumulative_criterion(q_psi):
 def crit(self, info_dict):
 pred_emb = info_dict["predicted_emb"] # (B, S, T_pred, D)
 goal_emb = info_dict["goal_emb"]
 g = goal_emb[.., -1, :]
 if g.dim() == pred_emb.dim() - 2:
 g = g.unsqueeze(1)
 B, S, T_pred, D = pred_emb.shape
 cost = torch.zeros(B, S, device=pred_emb.device)
 goal_t_template = g.expand(B, S, D)
 for t in range(T_pred):
 z_t = pred_emb[.., t, :] # (B, S, D)
 z_g = goal_t_template
 h = float(T_pred - t)
 h_tensor = torch.full((B*S,), h, device=pred_emb.device)
 logits = q_psi(z_t.reshape(B*S, D), z_g.reshape(B*S, D), h_tensor)
 cost = cost + F.softplus(-logits).reshape(B, S)
 return cost
 return crit


# ---------------------------------------------------------------------------
# Main: run the A5 protocol with Q-cost on a single episode across seeds
# ---------------------------------------------------------------------------

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--cell", default="A5")
 ap.add_argument("--cost", choices=["terminal", "cumulative"], default="cumulative")
 ap.add_argument("--q_ckpt", required=True)
 ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10])
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/diagnostics")
 args = ap.parse_args()

 out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
 cfg = TASKS[args.task]

 # Load Q_psi
 ckpt = torch.load(args.q_ckpt, weights_only=False, map_location="cuda")
 qcfg = ckpt["config"]
 q = QPsi(qcfg["d_latent"], qcfg["max_horizon_chunks"], hidden=qcfg["hidden"]).to("cuda")
 q.load_state_dict(ckpt["state_dict"])
 q.eval()
 q.requires_grad_(False)
 n_pos = qcfg.get("n_pos", qcfg.get("n_pos_train", "?"))
 n_neg = qcfg.get("n_neg", qcfg.get("n_neg_train", "?"))
 print(f"Loaded Q_psi: d={qcfg['d_latent']}, max_h={qcfg['max_horizon_chunks']},"
 f" hidden={qcfg['hidden']}, trained on {n_pos}+{n_neg} pairs")

 # Load world model
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

 # Patch criterion
 if args.cost == "terminal":
 model.criterion = types.MethodType(make_q_terminal_criterion(q), model)
 else:
 model.criterion = types.MethodType(make_q_cumulative_criterion(q), model)

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
 print(f"\nQ_psi-{args.cost} on episode {args.episode} ({args.cell} protocol): "
 f"{sum(r['success'] for r in results)}/{len(results)} = {rate:.1f}% "
 f"across {len(results)} CEM seeds")

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-q-cost-{args.cost}-ep{args.episode}-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "cell": args.cell, "cost_variant": args.cost,
 "q_ckpt": args.q_ckpt,
 "results": results,
 "success_count": sum(r["success"] for r in results),
 "success_rate_pct": rate,
 }, indent=2))
 print(f"Written to {out_path}")


if __name__ == "__main__":
 main()
