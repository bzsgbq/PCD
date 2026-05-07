"""Retrieval-only baseline.

Execute the top-K retrieved offline action chunk DIRECTLY in the env,
no iCEM refinement. If retrieval-only ≈ bridge-iCEM, the contribution
is mostly dataset replay. If bridge-iCEM >> retrieval-only, planning
warm-start is the contribution.
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
from stable_worldmodel.policy import AutoCostModel

from run_ablation import TASKS, build_world, img_transform
from dataset_bridge import (
 build_chunk_db, encode_one_pixel_frame, retrieve_bridges,
)


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--mode", choices=["bridge", "start_only", "random"], default="bridge")
 ap.add_argument("--n_bridges", type=int, default=10,
 help="number of candidates to retrieve; we try each in turn (best-of-K) or just the top-1")
 ap.add_argument("--top_k_to_try", type=int, default=10,
 help="execute each of top-K and take best success (best-of-K)")
 ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10],
 help="seeds vary the env random source if any")
 ap.add_argument("--n_db_episodes", type=int, default=200)
 ap.add_argument("--exclude_eval_ep", action="store_true")
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
 tform = img_transform()

 # Build chunk db
 excl = [args.episode] if args.exclude_eval_ep else None
 Z_start, Z_end, A_chunks, ep_ids = build_chunk_db(
 model, dataset, args.n_db_episodes, cfg["action_block"],
 horizon_chunks=10, # match A5 horizon
 tform=tform, exclude_episodes=excl,
 )
 print(f" chunk db: {len(Z_start)} chunks", flush=True)

 # Get start/goal latents
 chunk = dataset.load_chunk([args.episode], [args.start],
 [args.start + cfg["goal_offset_steps"] + 1])[0]
 pix_start = chunk["pixels"][0]
 pix_goal = chunk["pixels"][-1]
 if hasattr(pix_start, "permute") and pix_start.shape[0] == 3:
 pix_start = pix_start.permute(1, 2, 0).numpy()
 pix_goal = pix_goal.permute(1, 2, 0).numpy()
 elif hasattr(pix_start, "numpy"):
 pix_start = pix_start.numpy()
 pix_goal = pix_goal.numpy()
 z_start = encode_one_pixel_frame(model, np.asarray(pix_start, dtype=np.uint8), tform)
 z_goal = encode_one_pixel_frame(model, np.asarray(pix_goal, dtype=np.uint8), tform)

 # Retrieve K candidates
 bridges = retrieve_bridges(z_start, z_goal, Z_start, Z_end, A_chunks,
 K=args.top_k_to_try, mode=args.mode)
 print(f" retrieved {bridges.shape[0]} candidates", flush=True)

 # Execute each candidate in the env (only first chunk = first action_block actions executed
 # before the agent would replan in MPC; but here we do open-loop full horizon).
 # For comparison with bridge-iCEM (which is closed-loop MPC), execute the FULL plan
 # of length horizon * action_block = 50 env steps, replanning within is N/A here.
 horizon_env = bridges.shape[1] * cfg["action_block"] # 10 * 5 = 50
 print(f" executing {bridges.shape[0]} candidates × {horizon_env} env steps each",
 flush=True)

 results = []
 for ci in range(bridges.shape[0]):
 cand = bridges[ci].numpy() # (10, action_block * A_raw)
 actions = cand.reshape(horizon_env, -1) # (50, A_raw)
 try:
 world = build_world(cfg, num_envs=1)
 world.envs.reset(options=None)
 # PushT uses 'state' (5-D), TwoRoom uses 'proprio' (2-D).
 state_key = "state" if "state" in chunk else "proprio"
 world.envs.unwrapped.call("_set_state",
 state=np.asarray(chunk[state_key][0]).flatten().astype(np.float32))
 world.envs.unwrapped.call("_set_goal_state",
 goal_state=np.asarray(chunk[state_key][-1]).flatten().astype(np.float32))
 success = False
 for t in range(horizon_env):
 a = actions[t].reshape(1, -1).astype(np.float32)
 _o, _r, terminated, _t, info = world.envs.step(a)
 if terminated[0]:
 success = True
 break
 results.append({"ci": ci, "success": success, "steps": t + 1})
 print(f" cand {ci}: success={success} (steps={t+1})", flush=True)
 except Exception as e:
 results.append({"ci": ci, "error": str(e)})
 import traceback; traceback.print_exc()

 n_success = sum(1 for r in results if r.get("success", False))
 rate = 100.0 * n_success / max(len(results), 1)
 print(f"\nRetrieval-only ({args.mode}, top-{args.top_k_to_try}) on ep {args.episode}: "
 f"best-of-K = {'yes' if n_success > 0 else 'no'} "
 f"({n_success}/{len(results)} = {rate:.1f}%)")

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-retrieval-only-{args.mode}-ep{args.episode}-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "mode": args.mode, "top_k_to_try": args.top_k_to_try,
 "exclude_eval_ep": args.exclude_eval_ep,
 "results": results, "n_success": n_success,
 "best_of_k_success": n_success > 0,
 "any_success_rate_pct": rate,
 }, indent=2))
 print(f"Written to {out_path}")


if __name__ == "__main__":
 main()
