"""Coverage test: random-action sim-replay from start of an episode.

Draws N candidate action sequences from N(0, var_scale^2), sim-replays each
from the episode's start state, and reports terminal geodesic distance to the
goal. If many candidates make progress, the action space contains good plans;
the planner's failure must be in ranking. If none, it's a coverage hole.

Run on episode 4339 with N ∈ {300, 1000, 3000} samples to map the coverage
curve.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_LEWM_DIR = Path(__file__).resolve().parents[3] / "le-wm"
sys.path.insert(0, str(_LEWM_DIR))
sys.path.insert(0, str(Path(__file__).parent))

import _threadlimits # noqa: F401 # CPU thread limits, BEFORE numpy/torch/sklearn

import numpy as np
import torch

import stable_pretraining as spt
import stable_worldmodel as swm

from run_ablation import TASKS, build_world
from two_room_geodesic import extract_walls, bfs_distance, euclidean


def reset_env_to_dataset_state(world, dataset, ep_id, start_step, callables):
 """Mirror what world.evaluate_from_dataset does for one env: set state to
 the offline dataset's start step, then return obs/info."""
 chunk = dataset.load_chunk([ep_id], [start_step], [start_step + 1])[0]
 state = np.asarray(chunk["proprio"][0]).reshape(1, -1) # (1, 2)
 goal_pix = chunk["pixels"][0]
 if hasattr(goal_pix, "permute"):
 goal_pix = goal_pix.permute(1, 2, 0).numpy() if goal_pix.shape[0] == 3 else goal_pix.numpy()
 # Use env.unwrapped.call to set state directly
 obs, info = world.envs.reset(options=None)
 # Apply each callable from the eval config
 for c in callables:
 method = c["method"]
 kwargs = {}
 for k, spec in c["args"].items():
 v = chunk[spec["value"]]
 if hasattr(v, "numpy"):
 v = v.numpy()
 kwargs[k] = np.asarray(v).reshape(1, -1)
 world.envs.unwrapped.call(method, **kwargs)
 obs, info = world.envs.reset(options=None)
 return obs, info, state.flatten(), chunk


def run_random_candidates(world, dataset, cfg, ep_id, start_step, N, var_scale,
 seed, plan_steps, action_block):
 """Sample N random Gaussian action sequences, sim-replay each from
 (ep_id, start_step). Return per-candidate terminal proprio + geodesic."""
 rng = np.random.default_rng(seed)
 A_raw = world.envs.action_space.shape[1] # 2 for TwoRoom
 horizon_env = plan_steps * action_block # 5 * 5 = 25 env steps
 actions_all = rng.normal(scale=var_scale, size=(N, horizon_env, A_raw)).astype(np.float32)
 # Clip to action bounds
 low = world.envs.action_space.low[0]; high = world.envs.action_space.high[0]
 actions_all = np.clip(actions_all, low, high)

 # Get goal proprio + start proprio + start pixels (for wall extraction)
 chunk = dataset.load_chunk([ep_id], [start_step], [start_step + 1])[0]
 start_xy = np.asarray(chunk["proprio"][0]).flatten()
 chunk_g = dataset.load_chunk([ep_id], [start_step + cfg["goal_offset_steps"]],
 [start_step + cfg["goal_offset_steps"] + 1])[0]
 goal_xy = np.asarray(chunk_g["proprio"][0]).flatten()
 pix0 = chunk["pixels"][0]
 if hasattr(pix0, "permute") and pix0.shape[0] == 3:
 pix0 = pix0.permute(1, 2, 0).numpy()
 elif hasattr(pix0, "numpy"):
 pix0 = pix0.numpy()
 pix0 = np.asarray(pix0, dtype=np.uint8)
 walls = extract_walls(pix0)
 init_geo = bfs_distance(walls, start_xy, goal_xy)
 init_eucl = euclidean(start_xy, goal_xy)
 print(f" start_xy=({start_xy[0]:.1f},{start_xy[1]:.1f}) goal_xy=({goal_xy[0]:.1f},{goal_xy[1]:.1f})", flush=True)
 print(f" init geo={init_geo:.1f} init eucl={init_eucl:.1f}", flush=True)

 # Sim-replay each candidate
 results = []
 t0 = time.time()
 for i in range(N):
 # Reset to start
 world.envs.unwrapped.call("_set_state", state=start_xy.reshape(1, -1).astype(np.float32))
 obs, info = world.envs.reset(options=None)
 for t in range(horizon_env):
 a = actions_all[i, t].reshape(1, -1)
 obs, _r, _d, _t, info = world.envs.step(a)
 if "proprio" in info:
 term_xy = np.array(info["proprio"]).flatten()[:2]
 else:
 term_xy = np.array(obs).flatten()[:2]
 d_geo = bfs_distance(walls, term_xy, goal_xy)
 d_eucl = euclidean(term_xy, goal_xy)
 results.append({
 "i": i, "term_xy": term_xy.tolist(),
 "term_geo": float(d_geo), "term_eucl": float(d_eucl),
 "delta_geo": float(init_geo - d_geo),
 "delta_eucl": float(init_eucl - d_eucl),
 })
 if (i + 1) % 100 == 0:
 print(f" [{i+1}/{N}] elapsed {time.time()-t0:.1f}s", flush=True)

 return results, init_geo, init_eucl


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--n_samples", type=int, nargs="+", default=[300, 1000, 3000])
 ap.add_argument("--var_scale", type=float, default=1.0)
 ap.add_argument("--seed", type=int, default=42)
 ap.add_argument("--plan_steps", type=int, default=5)
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/diagnostics")
 args = ap.parse_args()

 out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
 cfg = TASKS[args.task]

 dataset = swm.data.HDF5Dataset(
 cfg["dataset_name"], keys_to_cache=cfg["keys_to_cache"],
 cache_dir=swm.data.utils.get_cache_dir(),
 )

 out = []
 for N in args.n_samples:
 print(f"\n=== N={N} samples ===", flush=True)
 world = build_world(cfg, num_envs=1)
 results, init_geo, init_eucl = run_random_candidates(
 world, dataset, cfg, args.episode, args.start,
 N, args.var_scale, args.seed, args.plan_steps, cfg["action_block"],
 )
 deltas = np.array([r["delta_geo"] for r in results])
 positives = (deltas > 0).sum()
 big = (deltas > 20).sum()
 very_big = (deltas > 40).sum()
 best = float(deltas.max())
 print(f" candidates with Δgeo > 0: {positives}/{N} ({100.*positives/N:.1f}%)", flush=True)
 print(f" candidates with Δgeo > 20: {big}/{N} ({100.*big/N:.1f}%)", flush=True)
 print(f" candidates with Δgeo > 40: {very_big}/{N} ({100.*very_big/N:.1f}%)", flush=True)
 print(f" best Δgeo achieved: {best:.1f}", flush=True)
 out.append({
 "n_samples": N, "var_scale": args.var_scale, "seed": args.seed,
 "init_geo": init_geo, "init_eucl": init_eucl,
 "n_pos": int(positives), "n_big_pos": int(big), "n_very_big": int(very_big),
 "best_delta": best,
 "deltas_summary": {
 "mean": float(deltas.mean()), "median": float(np.median(deltas)),
 "max": best, "min": float(deltas.min()),
 },
 })

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-coverage-ep{args.episode}-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "var_scale": args.var_scale, "seed": args.seed,
 "results": out,
 }, indent=2))
 print(f"\nWritten to {out_path}")


if __name__ == "__main__":
 main()
