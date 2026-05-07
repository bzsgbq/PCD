"""Dataset-bridge warm-start for iCEM.

Build an offline database of latent-encoded action chunks of length
H_chunks = horizon (= 10 latent steps = 50 env frames in A5 protocol).
At plan time, given (z_current, z_goal), retrieve top-K chunks whose
endpoints minimize d(z_chunk_start, z_t) + d(z_chunk_end, z_g). Insert
into iCEM's first iteration as elite warm-start.

guardrails:
 - retrieval initializes candidates only; iCEM still scores+refines with
 the frozen world model.
 - report retrieval-only execution as a baseline (separate run).
 - random-retrieval ablation (separate run).
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
from torchvision.transforms import v2 as transforms

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
# Build offline chunk database
# ---------------------------------------------------------------------------

@torch.inference_mode()
def build_chunk_db(model, dataset, n_episodes, action_block, horizon_chunks,
 tform, exclude_episodes=None, device="cuda"):
 """For each episode, slice into chunks of length horizon_chunks (in
 latent-step units = action_block env-step units each). For each chunk,
 encode start frame, end frame, and capture the raw action sequence.

 exclude_episodes: optional iterable of episode IDs to filter out
 (e.g., the eval episode we're testing against, to avoid same-ep retrieval).

 Returns:
 Z_start: (N, D) — encoded start latent of each chunk
 Z_end: (N, D) — encoded end latent of each chunk
 A_chunks: (N, horizon_chunks, action_block * A_raw) — actions
 ep_ids: (N,) — episode id each chunk came from
 """
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 ep_indices = np.unique(dataset.get_col_data(col))
 if exclude_episodes is not None:
 excl = set(int(e) for e in exclude_episodes)
 ep_indices = np.array([e for e in ep_indices if int(e) not in excl])
 print(f" excluding {len(excl)} episodes from chunk db", flush=True)
 rng = np.random.default_rng(0)
 chosen = rng.choice(ep_indices, size=min(n_episodes, len(ep_indices)),
 replace=False)
 chosen = np.sort(chosen)

 Z_start, Z_end, A_chunks, EP_IDS = [], [], [], []
 horizon_env = horizon_chunks * action_block
 for i, ep in enumerate(chosen):
 if i % 50 == 0:
 print(f" encoding ep {i+1}/{len(chosen)} (id={ep})", flush=True)
 try:
 mask = dataset.get_col_data(col) == ep
 ep_len = int(mask.sum())
 if ep_len < horizon_env + 1:
 continue
 chunk = dataset.load_chunk([int(ep)], [0], [int(ep_len)])[0]
 pix = chunk["pixels"]
 if pix.shape[1] != 3 and pix.shape[-1] == 3:
 pix = pix.permute(0, 3, 1, 2)
 actions = chunk["action"]
 if not torch.is_tensor(actions):
 actions = torch.tensor(actions)
 actions = actions.float()
 A_raw = actions.shape[-1]

 # Slide chunks; keep stride = action_block to give variety
 for s in range(0, ep_len - horizon_env, action_block):
 e = s + horizon_env
 # Encode start and end frames
 f_s = pix[s]; f_e = pix[e]
 t_s = tform(f_s.permute(1, 2, 0).numpy() if hasattr(f_s, "permute") else np.asarray(f_s))
 t_e = tform(f_e.permute(1, 2, 0).numpy() if hasattr(f_e, "permute") else np.asarray(f_e))
 pair = torch.stack([t_s, t_e]).unsqueeze(0).to(device)
 z = model.encode({"pixels": pair})["emb"][0].cpu().numpy()
 Z_start.append(z[0]); Z_end.append(z[1])
 # Group actions
 A = actions[s:e].view(horizon_chunks, action_block * A_raw).cpu().numpy()
 A_chunks.append(A)
 EP_IDS.append(int(ep))
 except Exception as e:
 print(f" ep {ep} failed: {e}")
 return (np.stack(Z_start), np.stack(Z_end), np.stack(A_chunks), np.array(EP_IDS))


# ---------------------------------------------------------------------------
# Bridge-aware iCEM solver: subclass ICEMSolver, override solve to inject bridges
# ---------------------------------------------------------------------------

class BridgeICEMSolver(ICEMSolver):
 def __init__(self, *args, bridge_actions=None, n_bridges=10, **kwargs):
 """bridge_actions: torch.Tensor of shape (K, horizon, action_dim) or None."""
 super().__init__(*args, **kwargs)
 self.bridge_actions = bridge_actions
 self.n_bridges = n_bridges

 @torch.inference_mode()
 def solve(self, info_dict, init_action=None):
 import time
 # Trampoline to parent's solve, but inject bridges as the initial
 # `prev_topk_candidates`. Easiest: copy parent's solve and modify.
 # We avoid copying the entire body; instead we monkey-patch the
 # initial elite-keep buffer just before iter 0.

 # Actually we need to call get_cost on bridges first to seed
 # prev_topk_candidates. Do it manually:
 if self.bridge_actions is None:
 return super().solve(info_dict, init_action=init_action)

 start_time = time.time()
 outputs = {"costs": [], "mean": [], "var": []}

 mean, var = self.init_action_distrib(init_action)
 mean = mean.to(self.device); var = var.to(self.device)

 # Single batch (we expect num_envs == 1 for diagnostic runs)
 for start_idx in range(0, self.n_envs, self.batch_size):
 end_idx = min(start_idx + self.batch_size, self.n_envs)
 current_bs = end_idx - start_idx
 batch_mean = mean[start_idx:end_idx]
 batch_var = var[start_idx:end_idx]

 expanded_infos = {}
 for k, v in info_dict.items():
 v_batch = v[start_idx:end_idx]
 if torch.is_tensor(v):
 v_batch = v_batch.unsqueeze(1)
 v_batch = v_batch.expand(current_bs, self.num_samples, *v_batch.shape[2:])
 elif isinstance(v, np.ndarray):
 v_batch = np.repeat(v_batch[:, None,..], self.num_samples, axis=1)
 expanded_infos[k] = v_batch

 # Pre-populate prev_topk_candidates with the bridge actions to
 # be carried as elites in iter 0.
 n_b = min(self.n_bridges, self.bridge_actions.shape[0], self.topk)
 br = self.bridge_actions[:n_b].to(self.device).unsqueeze(0).expand(current_bs, n_b, -1, -1)
 # Pad up to topk by repeating the first bridge (or zeros) so the
 # parent loop's `topk_candidates[:, :n_inject]` slice has shape
 n_pad = self.topk - n_b
 if n_pad > 0:
 pad = br[:, :1].expand(current_bs, n_pad, -1, -1)
 prev_topk_candidates = torch.cat([br, pad], dim=1)
 else:
 prev_topk_candidates = br

 batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)

 noise_shape = (current_bs, self.num_samples, self.action_dim, self.horizon)
 freqs = torch.fft.rfftfreq(self.horizon, device=self.device)
 freqs[0] = 1.0
 noise_scale = freqs.pow(-self.noise_beta / 2)
 noise_scale[0] = noise_scale[1]

 for step in range(self.n_steps):
 if self.horizon <= 1:
 noise = torch.randn(noise_shape, generator=self.torch_gen, device=self.device)
 else:
 white = torch.randn(noise_shape, generator=self.torch_gen, device=self.device)
 fft = torch.fft.rfft(white, dim=-1)
 colored = torch.fft.irfft(fft * noise_scale, n=self.horizon, dim=-1)
 std = colored.std(dim=-1, keepdim=True).clamp(min=1e-8)
 noise = colored / std
 noise = noise.transpose(-1, -2)

 candidates = noise * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
 candidates[:, 0] = batch_mean
 if prev_topk_candidates is not None:
 n_inject = min(self.n_elite_keep, prev_topk_candidates.shape[1])
 candidates[:, 1:1 + n_inject] = prev_topk_candidates[:, :n_inject]
 if self._action_low is not None:
 candidates = candidates.clamp(self._action_low, self._action_high)

 current_info = expanded_infos.copy()
 costs = self.model.get_cost(current_info, candidates)

 topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
 topk_candidates = candidates[batch_indices, topk_inds]
 prev_topk_candidates = topk_candidates

 elite_mean = topk_candidates.mean(dim=1)
 elite_var = topk_candidates.std(dim=1)
 batch_mean = self.alpha * batch_mean + (1 - self.alpha) * elite_mean
 batch_var = self.alpha * batch_var + (1 - self.alpha) * elite_var

 final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()
 if self.return_mean:
 mean[start_idx:end_idx] = batch_mean
 else:
 mean[start_idx:end_idx] = topk_candidates[:, 0]
 var[start_idx:end_idx] = batch_var
 outputs["costs"].extend(final_batch_cost)

 outputs["actions"] = mean.detach().cpu()
 outputs["mean"] = [mean.detach().cpu()]
 outputs["var"] = [var.detach().cpu()]

 print(f"BridgeICEM solve time: {time.time() - start_time:.4f}s "
 f"(injected {self.n_bridges} bridges)")
 return outputs


# ---------------------------------------------------------------------------
# Bridge-A5 evaluator
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_one_pixel_frame(model, pixel_uint8, tform, device="cuda"):
 t = tform(pixel_uint8).unsqueeze(0).unsqueeze(0).to(device)
 return model.encode({"pixels": t})["emb"][0, 0].cpu().numpy()


def retrieve_bridges(z_t, z_g, Z_start, Z_end, A_chunks, K=10, mode="bridge"):
 """Return (K, horizon, action_dim) bridge action sequences.
 mode: 'bridge' (sum start+end), 'start_only', 'random'."""
 if mode == "bridge":
 d_s = ((Z_start - z_t) ** 2).sum(-1)
 d_e = ((Z_end - z_g) ** 2).sum(-1)
 score = d_s + d_e
 elif mode == "start_only":
 score = ((Z_start - z_t) ** 2).sum(-1)
 elif mode == "random":
 rng = np.random.default_rng(0)
 idx = rng.choice(len(Z_start), size=K, replace=False)
 return torch.tensor(A_chunks[idx], dtype=torch.float32)
 else:
 raise ValueError(mode)
 idx = np.argsort(score)[:K]
 return torch.tensor(A_chunks[idx], dtype=torch.float32)


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--cell", default="A5")
 ap.add_argument("--mode", choices=["bridge", "start_only", "random"], default="bridge")
 ap.add_argument("--n_bridges", type=int, default=10)
 ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10])
 ap.add_argument("--n_db_episodes", type=int, default=300)
 ap.add_argument("--exclude_eval_ep", action="store_true",
 help="Filter retrieval database to NOT include the eval episode (no-same-episode AMB).")
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

 cell = next(c for c in default_cells(cfg["action_block"]) if c["name"] == args.cell)
 horizon_chunks = cell["horizon"]

 print(f"== Building chunk db ({args.n_db_episodes} eps, horizon={horizon_chunks} chunks)", flush=True)
 excl = [args.episode] if args.exclude_eval_ep else None
 Z_start, Z_end, A_chunks, ep_ids = build_chunk_db(
 model, dataset, args.n_db_episodes, cell["action_block"],
 horizon_chunks, tform, exclude_episodes=excl,
 )
 print(f" chunk db: {len(Z_start)} chunks from {len(np.unique(ep_ids))} episodes,"
 f" action_dim per step = {A_chunks.shape[-1]}", flush=True)
 if args.exclude_eval_ep:
 print(f" exclude_eval_ep=True; ep {args.episode} chunks: "
 f"{int((ep_ids == args.episode).sum())} (should be 0)", flush=True)

 # Encode start and goal latents for the test episode
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

 print(f"\n== Running bridge-{args.mode} A5 on ep {args.episode} ==", flush=True)
 transform_dict = {"pixels": img_transform(), "goal": img_transform()}
 processors = build_processors(dataset, cfg["keys_to_cache"])

 patch_cost_shape(model, "cumulative")

 results = []
 for s in args.seeds:
 # Retrieve fresh bridges per seed if mode is random; else deterministic
 bridges = retrieve_bridges(z_start, z_goal, Z_start, Z_end, A_chunks,
 K=args.n_bridges, mode=args.mode)
 solver = BridgeICEMSolver(
 model=model, batch_size=1, num_samples=cell["num_samples"],
 var_scale=1.0, n_steps=cell["n_steps"], topk=cell["topk"],
 noise_beta=cell["noise_beta"], alpha=cell["alpha"],
 n_elite_keep=cell["n_elite_keep"], return_mean=True,
 device="cuda", seed=int(s),
 bridge_actions=bridges, n_bridges=args.n_bridges,
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
 print(f"\nBridge-{args.mode} A5 on ep {args.episode}: "
 f"{sum(r['success'] for r in results)}/{len(results)} = {rate:.1f}%")

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-bridge-{args.mode}-ep{args.episode}-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task, "episode": args.episode, "start": args.start,
 "cell": args.cell, "mode": args.mode, "n_bridges": args.n_bridges,
 "n_db_episodes": args.n_db_episodes,
 "n_db_chunks": len(Z_start),
 "results": results, "success_rate_pct": rate,
 }, indent=2))
 print(f"Written to {out_path}")


if __name__ == "__main__":
 main()
