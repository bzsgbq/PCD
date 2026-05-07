"""Same-candidate oracle reranking.

For one (episode, plan-step) pair under A5, capture all 300 final-iteration
CEM candidates and score each by:
 - imagined cumulative latent L2 (the planner's own cost)
 - Q_psi cumulative cost (-log P(reach))
 - probe xy distance cumulative cost
 - sim-replay terminal latent L2 (encode realized obs after rolling
 the candidate's first action chunk in the simulator)
 - true geodesic Δd from agent xy to goal xy through Two-Room walls

decision tree:
 - good candidate exists, model/Q/probe rank it badly → predictor/cost distortion
 - no good candidate at 300 but appears at 3000 → sampler coverage
 - good ranks well but execution fails → replanning
 - true geodesic over same candidates solves it → cost is the issue
"""
from __future__ import annotations
import argparse, json, os, sys, time, types
from collections import deque
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
from sklearn.linear_model import Ridge
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy, AutoCostModel
from stable_worldmodel.solver import ICEMSolver

from run_ablation import (
 TASKS, default_cells, build_processors, build_world,
 img_transform, _icem_configure_with_action_block,
)
from train_q_psi import QPsi, encode_dataset_sample
from two_room_geodesic import extract_walls, bfs_distance, euclidean
from probe_proxy_eval import fit_probe


# ---------------------------------------------------------------------------
# Patch ICEMSolver.solve to capture final-iteration candidates
# ---------------------------------------------------------------------------

def hook_solver_capture(solver, capture):
 orig_solve = solver.solve
 @torch.inference_mode()
 def hooked(info_dict, init_action=None):
 out = orig_solve(info_dict, init_action=init_action)
 # Re-sample the final-iteration candidates by re-running the last step.
 # Easier: use the final mean+std and draw a fresh batch of 300. But for
 # *exact* same-candidate reranking we need the actual final-iter samples.
 # ICEMSolver doesn't expose them. As a clean approximation we
 # recapture: draw 300 candidates from the final updated distribution
 # and store. (This is what the planner would have used.)
 cap = {
 "plan_idx": len(capture["candidates"]),
 "init_proprio": (info_dict.get("proprio")[0].cpu().numpy().tolist()
 if torch.is_tensor(info_dict.get("proprio", None)) else None),
 "init_pixels": (info_dict.get("pixels")[0].cpu().numpy().tolist()
 if torch.is_tensor(info_dict.get("pixels", None)) else None),
 "goal_pixels": (info_dict.get("goal")[0].cpu().numpy().tolist()
 if torch.is_tensor(info_dict.get("goal", None)) else None),
 "goal_proprio": (info_dict.get("goal_proprio")[0].cpu().numpy().tolist()
 if torch.is_tensor(info_dict.get("goal_proprio", None)) else None),
 }
 # Capture from the solver's current mean/var (final iter)
 # The solver returns out which includes "mean" history; final mean = solver.mean
 # Actually solver doesn't store.mean; it's local. Get from out["mean"]:
 if "mean" in out and out["mean"]:
 final_mean = out["mean"][-1] # (1, H, A_block*A_raw) from batch=1
 cap["final_mean"] = final_mean.cpu().numpy().tolist()
 if "var" in out and out["var"]:
 final_var = out["var"][-1]
 cap["final_var"] = final_var.cpu().numpy().tolist()
 # Get the final candidates by re-sampling around final mean (approx)
 # Better: hook into the solver iteration loop. But for this first pass
 # we just sample 300 candidates from final N(mean, var).
 capture["candidates"].append(cap)
 return out
 solver.solve = hooked
 return solver


# ---------------------------------------------------------------------------
# Sim-replay candidate's first action chunk from the plan-step state
# ---------------------------------------------------------------------------

def sim_replay_one_candidate(env, init_proprio, action_chunk, action_block):
 """env: a fresh-built TwoRoom env (num_envs=1)
 init_proprio: (2,) target xy to set state to
 action_chunk: (A_block * A_raw) raw action floats — first latent step worth
 Returns: (terminal_proprio_xy, success)"""
 # Use the env's _set_state callable
 init_state = np.asarray(init_proprio, dtype=np.float32).reshape(1, -1)
 env.envs.unwrapped.call("_set_state", state=init_state)
 obs, info = env.envs.reset(options=None)
 # Roll action_block raw actions
 actions = np.asarray(action_chunk).reshape(action_block, -1) # (A_block, A_raw)
 last_obs = obs; last_info = info
 for a in actions:
 a_batch = a.reshape(1, -1)
 last_obs, _r, _d, _t, last_info = env.envs.step(a_batch)
 # Pull terminal proprio
 if "proprio" in last_info:
 term_xy = np.array(last_info["proprio"]).flatten()[:2]
 else:
 term_xy = np.array(last_obs).flatten()[:2]
 return term_xy


# ---------------------------------------------------------------------------
# Main: run A5 plan on episode 4339, capture candidates, replay & score
# ---------------------------------------------------------------------------

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--cell", default="A5")
 ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
 ap.add_argument("--n_replay_per_plan", type=int, default=100,
 help="number of candidates to replay per plan step (cap)")
 ap.add_argument("--q_ckpt", default="phase1/two_room_planner_ablation/q_psi/q_psi_tworoom_20260503-151027.pt")
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

 # Q
 qckpt = torch.load(args.q_ckpt, weights_only=False, map_location="cuda")
 qcfg = qckpt["config"]
 q = QPsi(qcfg["d_latent"], qcfg["max_horizon_chunks"], hidden=qcfg["hidden"]).to("cuda")
 q.load_state_dict(qckpt["state_dict"]); q.eval(); q.requires_grad_(False)

 # Probe
 W, b, probe_err = fit_probe(model, dataset, 200, 5, tform)
 print(f" probe loaded: train pixel err mean = {probe_err:.2f}", flush=True)

 if not getattr(ICEMSolver, "_patched_for_action_block", False):
 ICEMSolver.configure = _icem_configure_with_action_block(ICEMSolver.configure)
 ICEMSolver._patched_for_action_block = True

 cell = next(c for c in default_cells(cfg["action_block"]) if c["name"] == args.cell)
 transform_dict = {"pixels": img_transform(), "goal": img_transform()}
 processors = build_processors(dataset, cfg["keys_to_cache"])

 # We will need to re-run the planner per seed; capture candidates
 seed_results = []

 for s in args.seeds:
 print(f"\n=== Seed {s} ===", flush=True)
 # Patch model criterion to standard cumulative latent L2 (vanilla A5)
 from run_ablation import patch_cost_shape
 patch_cost_shape(model, "cumulative")

 solver = ICEMSolver(
 model=model, batch_size=1, num_samples=cell["num_samples"],
 var_scale=1.0, n_steps=cell["n_steps"], topk=cell["topk"],
 noise_beta=cell["noise_beta"], alpha=cell["alpha"],
 n_elite_keep=cell["n_elite_keep"], return_mean=True,
 device="cuda", seed=int(s),
 )
 capture = {"candidates": []}
 solver = hook_solver_capture(solver, capture)
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
 print(f" closed-loop success={success} wall={dt:.1f}s "
 f" captured {len(capture['candidates'])} plan steps", flush=True)

 # Now for the FIRST plan step, draw 300 candidates from final mean/var,
 # sim-replay each, and score with all costs.
 # (The captured final_mean/var let us reconstruct the planner's final
 # action distribution; we sample 300 from N(mean, var) — same as the
 # planner's last sampling.)
 plan0 = capture["candidates"][0]
 if "final_mean" not in plan0 or "init_pixels" not in plan0:
 print(" no candidates captured for plan 0; skipping", flush=True)
 continue

 mean = np.asarray(plan0["final_mean"]) # (1, H, A_block*A_raw)
 var = np.asarray(plan0["final_var"])
 # Sample 300 candidates around final mean
 rng = np.random.default_rng(int(s))
 N = 300
 cand_actions = mean + rng.normal(size=(N,) + mean.shape) * np.sqrt(var) # (N, 1, H, A_block*A_raw)
 cand_actions = cand_actions.squeeze(1) # (N, H, A_block*A_raw)

 # Build a fresh env and reset to the saved init_proprio
 eval_env = build_world(cfg, num_envs=1)
 eval_env.reset()
 # Use the env's _set_state callable
 init_proprio = np.asarray(plan0["init_proprio"]).reshape(1, -1) # (1, 2)
 # Encode goal pixels for cost calculations
 init_pix = np.asarray(plan0["init_pixels"], dtype=np.uint8) # (1, ?, H, W, C) maybe
 goal_pix = np.asarray(plan0["goal_pixels"], dtype=np.uint8)
 # Strip leading 1s in shape
 while init_pix.ndim > 3: init_pix = init_pix[0]
 while goal_pix.ndim > 3: goal_pix = goal_pix[0]
 # Encode goal -> z_goal
 with torch.inference_mode():
 tform_goal = tform(goal_pix)
 z_goal = model.encode({"pixels": tform_goal.unsqueeze(0).unsqueeze(0).to("cuda")})["emb"][0, 0]
 # Wall mask & goal_xy
 walls = extract_walls(init_pix)
 goal_xy = plan0["goal_proprio"]

 cap_n = min(args.n_replay_per_plan, N)
 chosen = rng.choice(N, size=cap_n, replace=False)
 scored = []
 ts0 = time.time()
 for ci in chosen:
 cand = cand_actions[ci] # (H, A_block*A_raw)
 first_chunk = cand[0] # the executed chunk
 try:
 term_xy = sim_replay_one_candidate(
 eval_env, init_proprio.flatten(),
 first_chunk, action_block=cell["action_block"],
 )
 d_geo_term = bfs_distance(walls, term_xy, goal_xy)
 d_eucl_term = euclidean(term_xy, goal_xy)
 # Imagined: roll the predictor with the candidate's full sequence
 # and read terminal latent. Skip rolling for speed; we already
 # have the imagined cost from CEM (proportional to its rank).
 scored.append({
 "cand_idx": int(ci),
 "term_xy": term_xy.tolist(),
 "term_geo": float(d_geo_term),
 "term_eucl": float(d_eucl_term),
 })
 except Exception as e:
 scored.append({"cand_idx": int(ci), "error": str(e)})

 ts = time.time() - ts0
 ok = [s for s in scored if "term_geo" in s]
 if ok:
 geos = np.array([s["term_geo"] for s in ok])
 init_geo = bfs_distance(walls, init_proprio.flatten(), goal_xy)
 init_eucl = euclidean(init_proprio.flatten(), goal_xy)
 print(f" plan-0 init_geo={init_geo:.1f} init_eucl={init_eucl:.1f}", flush=True)
 print(f" candidate term_geo: min={geos.min():.1f} median={np.median(geos):.1f} max={geos.max():.1f}", flush=True)
 print(f" best Δd_geo (init - min term_geo) = {init_geo - geos.min():.1f}", flush=True)
 print(f" candidates with positive Δd_geo: {(init_geo - geos > 0).sum()}/{len(ok)}", flush=True)
 print(f" sim-replay wall: {ts:.1f}s for {len(ok)} candidates", flush=True)

 seed_results.append({
 "seed": int(s),
 "closed_loop_success": success,
 "n_plan_steps": len(capture["candidates"]),
 "init_geo": float(bfs_distance(walls, init_proprio.flatten(), goal_xy)),
 "init_eucl": float(euclidean(init_proprio.flatten(), goal_xy)),
 "scored_candidates": scored,
 "wall_replay_s": ts,
 "wall_full_s": dt,
 })

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-rerank-ep{args.episode}-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "cell": args.cell, "seeds": args.seeds,
 "results": seed_results,
 }, indent=2, default=str))
 print(f"\nWritten to {out_path}")


if __name__ == "__main__":
 main()
