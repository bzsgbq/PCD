"""Same-candidate oracle reranking via CandidateLogger.

For one (episode, seed) pair under A5, run iCEM end-to-end with the
CandidateLogger active to capture every (candidates, costs) call from
get_cost. We then pull the FINAL ITERATION of plan step 0 and:
 - sim-replay each candidate's first action chunk from the recorded init
 proprio,
 - score under: imagined cost (already known), Q_v2, probe-xy,
 sim-true terminal proprio → BFS Δd_geo,
 - report the table specified.

The eval env is recreated per candidate using `_set_state` so the simulator
is reset exactly.
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
import torch.nn.functional as F
import gymnasium as gym

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy, AutoCostModel
from stable_worldmodel.solver import ICEMSolver

from run_ablation import (
 TASKS, default_cells, build_processors, build_world,
 img_transform, _icem_configure_with_action_block,
)
from train_q_psi import QPsi
from candidate_logger import CandidateLogger
from probe_proxy_eval import fit_probe
from two_room_geodesic import extract_walls, bfs_distance, euclidean


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--cell", default="A5")
 ap.add_argument("--seed", type=int, default=1)
 ap.add_argument("--q_ckpt", default=None,
 help="Q v2 checkpoint to score with (defaults to v1 if None)")
 ap.add_argument("--n_replay", type=int, default=300,
 help="how many of the final-iter candidates to sim-replay")
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

 # Q
 qcfg, q = None, None
 if args.q_ckpt:
 qckpt = torch.load(args.q_ckpt, weights_only=False, map_location="cuda")
 qcfg = qckpt["config"]
 q = QPsi(qcfg["d_latent"], qcfg["max_horizon_chunks"], hidden=qcfg["hidden"]).to("cuda")
 q.load_state_dict(qckpt["state_dict"]); q.eval(); q.requires_grad_(False)
 print(f"Loaded Q from {args.q_ckpt} (best val AUC: {qcfg.get('best_val_auc', 'n/a')})")

 # Probe (cheap, fit fresh)
 tform = img_transform()
 W, b, probe_err = fit_probe(model, dataset, 200, 5, tform)
 print(f"Probe trained: pixel err mean = {probe_err:.2f}")

 cell = next(c for c in default_cells(cfg["action_block"]) if c["name"] == args.cell)
 transform_dict = {"pixels": img_transform(), "goal": img_transform()}
 processors = build_processors(dataset, cfg["keys_to_cache"])

 # Patch criterion to vanilla cumulative latent L2 (the planner CEM uses)
 from run_ablation import patch_cost_shape
 patch_cost_shape(model, "cumulative")

 # Build solver and policy
 solver = ICEMSolver(
 model=model, batch_size=1, num_samples=cell["num_samples"],
 var_scale=1.0, n_steps=cell["n_steps"], topk=cell["topk"],
 noise_beta=cell["noise_beta"], alpha=cell["alpha"],
 n_elite_keep=cell["n_elite_keep"], return_mean=True,
 device="cuda", seed=args.seed,
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

 logger = CandidateLogger(model)
 logger.activate()
 t0 = time.time()
 metrics = world.evaluate_from_dataset(
 dataset, start_steps=[args.start], goal_offset_steps=cfg["goal_offset_steps"],
 eval_budget=cfg["eval_budget"], episodes_idx=[args.episode],
 callables=cfg["callables"], save_video=False, video_path="/tmp",
 )
 logger.deactivate()
 dt = time.time() - t0
 success = bool(metrics["episode_successes"][0])
 print(f"\nClosed-loop A5 seed={args.seed} success={success} wall={dt:.1f}s")
 print(f" total get_cost calls captured: {len(logger.history)}")
 # Each plan step calls get_cost n_steps=30 times; with 10 plan steps → 300 entries
 n_per_plan = cell["n_steps"]
 n_plans = len(logger.history) // n_per_plan
 print(f" inferred {n_plans} plan steps × {n_per_plan} CEM iters")

 # Take the FINAL iter of plan step 0 (the candidates that produced the
 # planner's selected first chunk).
 plan0_final_idx = n_per_plan - 1
 if plan0_final_idx >= len(logger.history):
 print("not enough history captured; aborting")
 return
 plan0_final = logger.history[plan0_final_idx]
 cands = plan0_final["candidates"] # (B=1, S=300, H=10, A_dim=10)
 imag_costs = plan0_final["costs"] # (B=1, S=300)

 # Pull the env state at plan step 0 from the offline dataset.
 chunk = dataset.load_chunk([args.episode], [args.start],
 [args.start + cfg["goal_offset_steps"] + 1])[0]
 start_xy = np.asarray(chunk["proprio"][0]).flatten()
 goal_xy = np.asarray(chunk["proprio"][-1]).flatten()
 pix0 = chunk["pixels"][0]
 if hasattr(pix0, "permute") and pix0.shape[0] == 3:
 pix0 = pix0.permute(1, 2, 0).numpy()
 elif hasattr(pix0, "numpy"):
 pix0 = pix0.numpy()
 pix0 = np.asarray(pix0, dtype=np.uint8)
 walls = extract_walls(pix0)
 init_geo = bfs_distance(walls, start_xy, goal_xy)
 init_eucl = euclidean(start_xy, goal_xy)
 print(f" init geo={init_geo:.1f} init eucl={init_eucl:.1f}")

 # Get goal latent for Q/probe scoring
 goal_pix = chunk["pixels"][-1]
 if hasattr(goal_pix, "permute") and goal_pix.shape[0] == 3:
 goal_pix = goal_pix.permute(1, 2, 0).numpy()
 elif hasattr(goal_pix, "numpy"):
 goal_pix = goal_pix.numpy()
 goal_pix = np.asarray(goal_pix, dtype=np.uint8)
 with torch.inference_mode():
 z_goal = model.encode({"pixels": tform(goal_pix).unsqueeze(0).unsqueeze(0).to("cuda")})["emb"][0, 0]

 # Build a fresh single env for sim replay
 eval_env = build_world(cfg, num_envs=1)

 # Score each candidate
 n = min(args.n_replay, cands.shape[1])
 print(f"\nReplaying {n} candidates from plan-step-0 final iter..")
 rows = []
 t0r = time.time()
 for ci in range(n):
 cand = cands[0, ci] # (H, A_dim) where A_dim = action_block * raw_action_dim = 10
 first_chunk = cand[0].numpy() # (10,)
 first_chunk_actions = first_chunk.reshape(cell["action_block"], -1) # (5, 2)
 try:
 # Reset env to start_xy via _set_state, then step the 5 raw actions
 eval_env.envs.reset(options=None)
 eval_env.envs.unwrapped.call("_set_state",
 state=start_xy.reshape(1, -1).astype(np.float32))
 # Now step
 for a in first_chunk_actions:
 a_b = a.reshape(1, -1).astype(np.float32)
 _o, _r, _d, _t, info = eval_env.envs.step(a_b)
 term_xy = np.array(info["proprio"]).flatten()[:2]
 term_geo = bfs_distance(walls, term_xy, goal_xy)
 term_eucl = euclidean(term_xy, goal_xy)
 row = {
 "ci": ci,
 "imag_cost": float(imag_costs[0, ci]),
 "term_geo": float(term_geo),
 "term_eucl": float(term_eucl),
 "delta_geo": float(init_geo - term_geo),
 }
 # Probe-xy cost (decoded from candidate's IMAGINED terminal latent)
 # We don't have the predictor's rollout for this candidate handy;
 # skip for now (would require rolling the predictor under the cand
 # action sequence and decoding via probe).
 rows.append(row)
 except Exception as e:
 rows.append({"ci": ci, "error": str(e)})
 if (ci + 1) % 50 == 0:
 print(f" {ci+1}/{n} elapsed={time.time()-t0r:.1f}s")

 ok = [r for r in rows if "term_geo" in r]
 if ok:
 deltas = np.array([r["delta_geo"] for r in ok])
 imag = np.array([r["imag_cost"] for r in ok])
 positives = (deltas > 0).sum()
 big = (deltas > 20).sum()
 very_big = (deltas > 40).sum()
 best_true_idx = int(np.argmax(deltas))
 best_imag_idx = int(np.argmin(imag))
 # Where does best-true rank under imagined cost?
 order_imag = np.argsort(imag) # ascending = best first
 rank_of_best_true = int(np.where(order_imag == best_true_idx)[0][0])
 # Δgeo of imag-best
 delta_at_imag_best = float(deltas[best_imag_idx])
 print()
 print(f"== Plan-step-0 candidate scoring (n={len(ok)}) ==")
 print(f" best true Δgeo: {deltas[best_true_idx]:+.2f} px")
 print(f" Δgeo of imagined-best: {delta_at_imag_best:+.2f} px")
 print(f" rank of best-true under imagined L2: {rank_of_best_true} / {len(ok)}")
 print(f" candidates with Δgeo > 0: {positives}/{len(ok)}")
 print(f" candidates with Δgeo > 20: {big}/{len(ok)}")
 print(f" candidates with Δgeo > 40: {very_big}/{len(ok)}")
 print(f" imag-vs-true Spearman: ", end="")
 try:
 from scipy.stats import spearmanr, kendalltau
 rho_s, _ = spearmanr(imag, -deltas) # imag low = good, deltas high = good
 rho_k, _ = kendalltau(imag, -deltas)
 print(f"ρ={rho_s:.3f} Kendall τ={rho_k:.3f}")
 except Exception as e:
 print(f"err: {e}")

 # Save
 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-rerank-v2-ep{args.episode}-seed{args.seed}-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "cell": args.cell, "seed": args.seed,
 "closed_loop_success": success,
 "init_geo": float(init_geo), "init_eucl": float(init_eucl),
 "n_replay": n, "n_ok": len(ok),
 "rows": rows,
 }, indent=2, default=str))
 print(f"\nWritten to {out_path}")


if __name__ == "__main__":
 main()
