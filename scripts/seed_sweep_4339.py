"""Quick A5-on-4339 seed-sensitivity sweep.

Each call uses a different CEMSolver/ICEMSolver `seed`. The eval set sampling
seed (--seed argument to run_ablation.py) stays the same so the (episode, start)
pair is identical. Only the solver's internal random generator changes.

Usage:
 python seed_sweep_4339.py --seeds 1 2 3 4 5
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
_LEWM_DIR = Path(__file__).resolve().parents[3] / "le-wm"
sys.path.insert(0, str(_LEWM_DIR))
sys.path.insert(0, str(Path(__file__).parent))

import _threadlimits # noqa: F401 # CPU thread limits, BEFORE numpy/torch/sklearn

import numpy as np, torch, torch.nn.functional as F
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


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--cell", default="A5")
 ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10])
 ap.add_argument("--num_samples", type=int, default=None,
 help="override CEM num_samples (default: cell's value)")
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/diagnostics")
 args = ap.parse_args()

 cfg = TASKS[args.task]
 out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

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

 cell = next(c for c in default_cells(cfg["action_block"]) if c["name"] == args.cell)
 patch_cost_shape(model, cell["cost_shape"])

 if not getattr(ICEMSolver, "_patched_for_action_block", False):
 ICEMSolver.configure = _icem_configure_with_action_block(ICEMSolver.configure)
 ICEMSolver._patched_for_action_block = True

 results = []
 n_samples_used = args.num_samples if args.num_samples else cell["num_samples"]
 for s in args.seeds:
 if cell["solver"] == "cem":
 solver = CEMSolver(
 model=model, batch_size=1, num_samples=n_samples_used,
 var_scale=1.0, n_steps=cell["n_steps"], topk=cell["topk"],
 device="cuda", seed=int(s),
 )
 elif cell["solver"] == "icem":
 solver = ICEMSolver(
 model=model, batch_size=1, num_samples=n_samples_used,
 var_scale=1.0, n_steps=cell["n_steps"], topk=cell["topk"],
 noise_beta=cell["noise_beta"], alpha=cell["alpha"],
 n_elite_keep=cell["n_elite_keep"], return_mean=True,
 device="cuda", seed=int(s),
 )
 else:
 raise ValueError(f"Unsupported solver={cell['solver']!r} on cell {cell['name']!r}")
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
 print(f"\ncell={args.cell} solver={cell['solver']} on {args.task} episode {args.episode} "
 f"(paired start={args.start}): "
 f"{sum(r['success'] for r in results)}/{len(results)} = {rate:.1f}% "
 f"across {len(results)} solver seeds")

 stamp = time.strftime("%Y%m%d-%H%M%S")
 n_tag = f"-n{n_samples_used}"
 out_path = out_dir / (
 f"{args.task}-seed-sweep-{args.cell}-ep{args.episode}-start{args.start}{n_tag}-{stamp}.json"
 )
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "cell": args.cell, "cell_config": cell, "num_samples": n_samples_used,
 "seeds": args.seeds, "results": results,
 "success_count": sum(r["success"] for r in results),
 "success_rate_pct": rate,
 }, indent=2, default=str))
 print(f"Written to {out_path}")


if __name__ == "__main__":
 main()
