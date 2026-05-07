"""Privileged contact-cost diagnostic for Push-T.

Train a Ridge probe z -> state(pusher_xy, block_xy) on offline data, then add
 λ * d(decoded_pusher, decoded_block)
to the planner cost. The probe is a privileged-state diagnostic: it uses the
ground-truth simulator state at *training* time, but at planning time it just
decodes the JEPA latent.

Three variants:
 --cost contact_only: cost = λ_c * decoded(pusher-block dist) (forces approach)
 --cost contact_plus: cost = standard latent L2 + λ_c * decoded(pusher-block dist)
 --cost staged: for first N_phase plan steps, contact_only; then contact_plus

If contact_plus solves 14108 → failure is cost blindness (planner needs explicit contact signal).
If still fails → deeper proposal/dynamics issue.
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
from sklearn.linear_model import Ridge

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy, AutoCostModel
from stable_worldmodel.solver import ICEMSolver

from run_ablation import (
 TASKS, default_cells, build_processors, build_world,
 img_transform, _icem_configure_with_action_block, patch_cost_shape,
)
from train_q_psi import encode_dataset_sample


# ---------------------------------------------------------------------------
# Train Ridge probe z -> (pusher_xy, block_xy) on offline data
# ---------------------------------------------------------------------------

@torch.inference_mode()
def fit_state_probe(model, dataset, n_episodes, frame_stride, tform, device="cuda"):
 print(f"== Encoding {n_episodes} eps for state probe ==", flush=True)
 cache = encode_dataset_sample(model, dataset, n_episodes, frame_stride,
 25, 5, tform, device)
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 Z, S = [], []
 for ep, c in cache.items():
 steps = c["steps"]
 chunk = dataset.load_chunk([int(ep)], [0], [int(steps.max() + 1)])[0]
 state_full = chunk["state"]
 if torch.is_tensor(state_full):
 state_full = state_full.numpy()
 # Push-T state[:5] = (agent_x, agent_y, block_x, block_y, block_rot)
 # Take just the 4-D (pusher_xy, block_xy) for our cost
 state_sub = state_full[steps][:, :4]
 Z.append(c["latents"]); S.append(state_sub)
 Z = np.concatenate(Z, 0)
 S = np.concatenate(S, 0)
 print(f" Ridge fit on {Z.shape[0]} samples", flush=True)
 reg = Ridge(alpha=1e-2)
 reg.fit(Z, S)
 pred = reg.predict(Z)
 err = np.sqrt(((pred - S)**2).sum(-1))
 print(f" probe train pixel err: mean={err.mean():.2f}, max={err.max():.2f}", flush=True)
 W = torch.tensor(reg.coef_, dtype=torch.float32, device=device)
 b = torch.tensor(reg.intercept_, dtype=torch.float32, device=device)
 return W, b, float(err.mean())


# ---------------------------------------------------------------------------
# Contact-aware criteria (patched onto JEPA model)
# ---------------------------------------------------------------------------

def make_contact_only_criterion(W, b, lambda_c=1.0):
 """Cost = λ_c * sum_t || decoded_pusher(z_pred,t) - decoded_block(z_pred,t) ||."""
 def crit(self, info_dict):
 pred_emb = info_dict["predicted_emb"] # (B, S, T, D)
 decoded = pred_emb @ W.T + b # (B, S, T, 4)
 pusher = decoded[.., :2]
 block = decoded[.., 2:]
 d = ((pusher - block) ** 2).sum(-1) # (B, S, T)
 cost = lambda_c * d.sum(-1) # cumulative over T -> (B, S)
 return cost
 return crit


def make_contact_plus_criterion(W, b, lambda_c=0.05):
 """Cost = standard cumulative latent L2 + λ_c * cumulative pusher-block distance."""
 def crit(self, info_dict):
 pred_emb = info_dict["predicted_emb"]
 goal_emb = info_dict["goal_emb"]
 # Standard cumulative latent L2 (matches A5)
 g = goal_emb[.., -1, :]
 if g.dim() == pred_emb.dim() - 2:
 g = g.unsqueeze(1) # (B, 1, D) -> (B, S, D) via broadcast
 goal_t = g.unsqueeze(-2).expand_as(pred_emb)
 l2_cost = F.mse_loss(pred_emb, goal_t.detach(), reduction="none").sum(
 dim=tuple(range(2, pred_emb.ndim))
 )
 # Contact term
 decoded = pred_emb @ W.T + b
 pusher = decoded[.., :2]
 block = decoded[.., 2:]
 d = ((pusher - block) ** 2).sum(-1)
 contact_cost = d.sum(-1)
 return l2_cost + lambda_c * contact_cost
 return crit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="pusht")
 ap.add_argument("--episode", type=int, default=14108)
 ap.add_argument("--start", type=int, default=61)
 ap.add_argument("--cell", default="A5")
 ap.add_argument("--cost_variant", choices=["contact_only", "contact_plus"], default="contact_plus")
 ap.add_argument("--lambda_c", type=float, default=0.05)
 ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5])
 ap.add_argument("--num_samples", type=int, default=300)
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
 tform = img_transform()

 if not getattr(ICEMSolver, "_patched_for_action_block", False):
 ICEMSolver.configure = _icem_configure_with_action_block(ICEMSolver.configure)
 ICEMSolver._patched_for_action_block = True

 W, b, probe_err = fit_state_probe(model, dataset, args.n_probe_eps, 5, tform)

 # Patch criterion
 if args.cost_variant == "contact_only":
 model.criterion = types.MethodType(make_contact_only_criterion(W, b, args.lambda_c), model)
 else:
 model.criterion = types.MethodType(make_contact_plus_criterion(W, b, args.lambda_c), model)

 cell = next(c for c in default_cells(cfg["action_block"]) if c["name"] == args.cell)
 transform_dict = {"pixels": img_transform(), "goal": img_transform()}
 processors = build_processors(dataset, cfg["keys_to_cache"])

 results = []
 for s in args.seeds:
 solver = ICEMSolver(
 model=model, batch_size=1, num_samples=args.num_samples,
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
 print(f"\nContact-{args.cost_variant} (λ={args.lambda_c}) on ep {args.episode}: "
 f"{sum(r['success'] for r in results)}/{len(results)} = {rate:.1f}%")

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-contact-{args.cost_variant}-ep{args.episode}-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "cell": args.cell, "cost_variant": args.cost_variant, "lambda_c": args.lambda_c,
 "num_samples": args.num_samples,
 "probe_train_pixel_err_mean": probe_err,
 "results": results, "success_rate_pct": rate,
 }, indent=2))
 print(f"Written to {out_path}")


if __name__ == "__main__":
 main()
