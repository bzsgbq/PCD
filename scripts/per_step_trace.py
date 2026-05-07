"""Per-step trace for episode 4339.

Walks the env step-by-step, logging:
 - true proprio (agent xy etc.) at every env step
 - true latent (encode current pixels obs) at every env step
 - imagined latent at each plan step (final CEM elite mean rolled out)
 - per-plan-step CEM imagined terminal cost for the elite

Output is one JSON file per (episode, cell). Designed for plotting later.

Doesn't try to dump all 300 candidates per CEM iter — that's a separate
sim-replay job. This script just produces the geodesic + latent curves you
need to argue "the elite trajectory drifted past the door" or "the elite
trajectory hit the wall."
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
_LEWM_DIR = Path(__file__).resolve().parents[3] / "le-wm"
sys.path.insert(0, str(_LEWM_DIR))
sys.path.insert(0, str(Path(__file__).parent))

import _threadlimits # noqa: F401 # CPU thread limits, BEFORE numpy/torch/sklearn

import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy, AutoCostModel
from stable_worldmodel.solver import CEMSolver, ICEMSolver

from run_ablation import (
 TASKS, default_cells, build_processors, build_world,
 img_transform, patch_cost_shape, _icem_configure_with_action_block,
)


# ---------------------------------------------------------------------------
# Hooks: capture per-step world.infos (proprio, pixels) and per-plan elite
# ---------------------------------------------------------------------------

class TraceState:
 def __init__(self):
 self.env_steps = [] # one per env.step() call
 self.plan_events = [] # one per planner.solve() call


def hook_world_step(world, state):
 orig_step = world.step
 def hooked():
 out = orig_step()
 # After step: world.infos has 'proprio', 'pixels', etc.
 rec = {"t": len(state.env_steps)}
 try:
 for k, v in world.infos.items():
 if torch.is_tensor(v):
 rec[k] = v[0].cpu().numpy().tolist()
 elif isinstance(v, np.ndarray):
 rec[k] = v[0].tolist() if v.ndim > 1 else v.tolist()
 # skip dict-of-pixels too big to log per step
 except Exception as e:
 rec["err"] = str(e)
 state.env_steps.append(rec)
 return out
 world.step = hooked
 return world


def hook_solver_solve(solver, state, model):
 orig_solve = solver.solve
 @torch.inference_mode()
 def hooked(info_dict, init_action=None):
 # Snapshot init pixels and proprio before solve
 ev = {"plan_idx": len(state.plan_events)}
 try:
 ev["init_proprio"] = info_dict.get("proprio")[0].cpu().numpy().tolist() \
 if torch.is_tensor(info_dict.get("proprio", None)) else None
 except Exception:
 ev["init_proprio"] = None
 out = orig_solve(info_dict, init_action=init_action)
 # Capture the final elite (mean of action distribution)
 # `outputs["mean"]` history is recorded by base solver; final is last entry.
 try:
 mean_history = out.get("mean", [])
 if mean_history:
 ev["final_mean"] = mean_history[-1].cpu().numpy().tolist()
 except Exception as e:
 ev["mean_err"] = str(e)
 # Capture the last cost array if available
 try:
 costs_hist = out.get("costs", [])
 if costs_hist:
 ev["final_topk_costs"] = costs_hist[-1].cpu().numpy().tolist() \
 if torch.is_tensor(costs_hist[-1]) else costs_hist[-1]
 except Exception:
 pass
 state.plan_events.append(ev)
 return out
 solver.solve = hooked
 return solver


# ---------------------------------------------------------------------------
# Run a single trace
# ---------------------------------------------------------------------------

def run_trace(task_cfg, cell, model, dataset, processors, transform_dict,
 episode_idx, start_step):
 print(f"\n=== Trace ep={episode_idx} start={start_step} cell={cell['name']} ===",
 flush=True)
 patch_cost_shape(model, cell["cost_shape"])

 if not getattr(ICEMSolver, "_patched_for_action_block", False):
 ICEMSolver.configure = _icem_configure_with_action_block(ICEMSolver.configure)
 ICEMSolver._patched_for_action_block = True

 plan_cfg = PlanConfig(
 horizon=cell["horizon"], receding_horizon=cell["receding_horizon"],
 history_len=task_cfg["history_size"], action_block=cell["action_block"],
 warm_start=cell["warm_start"],
 )

 if cell["solver"] == "cem":
 solver = CEMSolver(model=model, batch_size=1, num_samples=cell["num_samples"],
 var_scale=1.0, n_steps=cell["n_steps"], topk=cell["topk"],
 device="cuda", seed=42)
 else:
 solver = ICEMSolver(model=model, batch_size=1, num_samples=cell["num_samples"],
 var_scale=1.0, n_steps=cell["n_steps"], topk=cell["topk"],
 noise_beta=cell["noise_beta"], alpha=cell["alpha"],
 n_elite_keep=cell["n_elite_keep"], return_mean=True,
 device="cuda", seed=42)

 state = TraceState()
 solver = hook_solver_solve(solver, state, model)
 policy = WorldModelPolicy(solver=solver, config=plan_cfg, process=processors,
 transform=transform_dict)
 world = build_world(task_cfg, num_envs=1)
 world = hook_world_step(world, state)
 world.set_policy(policy)

 t0 = time.time()
 metrics = world.evaluate_from_dataset(
 dataset, start_steps=[start_step], goal_offset_steps=task_cfg["goal_offset_steps"],
 eval_budget=task_cfg["eval_budget"], episodes_idx=[episode_idx],
 callables=task_cfg["callables"], save_video=False, video_path="/tmp",
 )
 dt = time.time() - t0

 return {
 "episode_idx": int(episode_idx), "start_step": int(start_step),
 "cell": cell["name"], "wall_s": dt,
 "success": bool(metrics["episode_successes"][0]),
 "env_steps": state.env_steps,
 "plan_events": state.plan_events,
 }


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episodes", type=int, nargs="+", default=[4339])
 ap.add_argument("--cells", type=str, default="A0,A5")
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/case_study")
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

 eset = json.loads(Path(
 "phase1/two_room_planner_ablation/results/tworoom-20260503-121919/eval_set.json"
 ).read_text())
 start_lookup = dict(zip(eset["episodes_idx"], eset["start_steps"]))

 cell_objs = [c for c in default_cells(cfg["action_block"])
 if c["name"] in args.cells.split(",")]

 all_traces = []
 for ep in args.episodes:
 if ep not in start_lookup:
 print(f"!! ep {ep} not in eval set; skipping")
 continue
 start = start_lookup[ep]
 for cell in cell_objs:
 try:
 r = run_trace(cfg, cell, model, dataset, processors, transform_dict,
 ep, start)
 all_traces.append(r)
 print(f" ep={ep} cell={cell['name']} success={r['success']} "
 f"n_env={len(r['env_steps'])} n_plans={len(r['plan_events'])} "
 f"wall={r['wall_s']:.1f}s", flush=True)
 except Exception as e:
 import traceback; traceback.print_exc()
 all_traces.append({"ep": ep, "cell": cell["name"], "error": str(e)})

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-trace-{stamp}.json"
 out_path.write_text(json.dumps(all_traces, indent=2, default=str))
 print(f"\nWritten to {out_path}")


if __name__ == "__main__":
 main()
