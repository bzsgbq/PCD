"""AMB retrieval quality check.

For TwoRoom 4339 retrieval: pull the top-K bridge chunks (with no-same-ep
exclusion) and report:
 - decoded/probe pusher xy distance from chunk-start to current state;
 - decoded/probe pusher xy distance from chunk-end to goal state;
 - same numbers for K random chunks for comparison.

If endpoint-conditioned retrieval picks chunks whose probe-decoded
endpoints are in fact close to (current, goal) state, this confirms the
retrieval geometry is doing useful work despite SIGReg latents being
globally well-mixed.
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
from sklearn.linear_model import Ridge
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel

from run_ablation import TASKS, img_transform
from dataset_bridge import build_chunk_db, encode_one_pixel_frame, retrieve_bridges
from train_q_psi import encode_dataset_sample


@torch.inference_mode()
def fit_xy_probe(model, dataset, n_episodes, frame_stride, tform):
 """Train Ridge probe z -> agent_xy on TwoRoom."""
 cache = encode_dataset_sample(model, dataset, n_episodes, frame_stride,
 25, 5, tform, "cuda")
 Z, P = [], []
 for ep, c in cache.items():
 steps = c["steps"]
 chunk = dataset.load_chunk([int(ep)], [0], [int(steps.max() + 1)])[0]
 proprio_full = chunk["proprio"]
 if torch.is_tensor(proprio_full):
 proprio_full = proprio_full.numpy()
 Z.append(c["latents"]); P.append(proprio_full[steps])
 Z = np.concatenate(Z, 0); P = np.concatenate(P, 0)
 reg = Ridge(alpha=1e-2)
 reg.fit(Z, P)
 err = np.sqrt(((reg.predict(Z) - P) ** 2).sum(-1))
 print(f" probe trained: train pixel err mean={err.mean():.2f}")
 return reg


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--episode", type=int, default=4339)
 ap.add_argument("--start", type=int, default=22)
 ap.add_argument("--K", type=int, default=10)
 ap.add_argument("--n_db_episodes", type=int, default=200)
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/diagnostics")
 args = ap.parse_args()

 out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
 cfg = TASKS["tworoom"]

 dataset = swm.data.HDF5Dataset(
 cfg["dataset_name"], keys_to_cache=cfg["keys_to_cache"],
 cache_dir=swm.data.utils.get_cache_dir(),
 )
 model = AutoCostModel(cfg["ckpt"]).to("cuda").eval()
 model.requires_grad_(False)
 if hasattr(model, "interpolate_pos_encoding"):
 model.interpolate_pos_encoding = True
 tform = img_transform()

 print("== Building chunk db (no-same-ep) ==")
 Z_start, Z_end, A_chunks, ep_ids = build_chunk_db(
 model, dataset, args.n_db_episodes, cfg["action_block"],
 horizon_chunks=10, tform=tform,
 exclude_episodes=[args.episode],
 )
 print(f" {len(Z_start)} chunks from {len(np.unique(ep_ids))} episodes")

 print("== Training probe z → agent_xy ==")
 probe = fit_xy_probe(model, dataset, args.n_db_episodes, 5, tform)

 # Decode all chunk-start and chunk-end latents to xy
 decoded_start_xy = probe.predict(Z_start)
 decoded_end_xy = probe.predict(Z_end)

 # Get current state and goal state from the eval episode
 chunk = dataset.load_chunk([args.episode], [args.start],
 [args.start + cfg["goal_offset_steps"] + 1])[0]
 cur_xy = np.asarray(chunk["proprio"][0]).flatten()
 goal_xy = np.asarray(chunk["proprio"][-1]).flatten()
 print(f" cur_xy={cur_xy.tolist()} goal_xy={goal_xy.tolist()}")

 # Encode start/goal latents
 pix_start = chunk["pixels"][0]
 pix_goal = chunk["pixels"][-1]
 if hasattr(pix_start, "permute") and pix_start.shape[0] == 3:
 pix_start = pix_start.permute(1, 2, 0).numpy()
 pix_goal = pix_goal.permute(1, 2, 0).numpy()
 elif hasattr(pix_start, "numpy"):
 pix_start = pix_start.numpy(); pix_goal = pix_goal.numpy()
 z_cur = encode_one_pixel_frame(model, np.asarray(pix_start, dtype=np.uint8), tform)
 z_goal = encode_one_pixel_frame(model, np.asarray(pix_goal, dtype=np.uint8), tform)

 # Top-K retrieval (latent endpoint nearest)
 score = ((Z_start - z_cur) ** 2).sum(-1) + ((Z_end - z_goal) ** 2).sum(-1)
 topk_idx = np.argsort(score)[:args.K]
 rng = np.random.default_rng(0)
 rand_idx = rng.choice(len(Z_start), size=args.K, replace=False)

 def report(label, idx):
 d_start_xy = np.linalg.norm(decoded_start_xy[idx] - cur_xy[None, :], axis=-1)
 d_end_xy = np.linalg.norm(decoded_end_xy[idx] - goal_xy[None, :], axis=-1)
 sum_d = d_start_xy + d_end_xy
 print(f"\n== {label} ({len(idx)} chunks) ==")
 print(f" decoded(start) xy distance to current_xy: "
 f"min={d_start_xy.min():.1f} median={np.median(d_start_xy):.1f} mean={d_start_xy.mean():.1f}")
 print(f" decoded(end) xy distance to goal_xy: "
 f"min={d_end_xy.min():.1f} median={np.median(d_end_xy):.1f} mean={d_end_xy.mean():.1f}")
 print(f" sum xy distance: "
 f"min={sum_d.min():.1f} median={np.median(sum_d):.1f} mean={sum_d.mean():.1f}")
 return {"d_start": d_start_xy.tolist(), "d_end": d_end_xy.tolist(),
 "min_sum": float(sum_d.min()), "mean_sum": float(sum_d.mean())}

 topk_stats = report("topK (endpoint-nearest)", topk_idx)
 rand_stats = report("random (control)", rand_idx)

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"tworoom-amb-retrieval-quality-ep{args.episode}-{stamp}.json"
 out_path.write_text(json.dumps({
 "episode": args.episode, "start": args.start, "K": args.K,
 "n_db_chunks": len(Z_start),
 "cur_xy": cur_xy.tolist(), "goal_xy": goal_xy.tolist(),
 "topk": topk_stats, "random": rand_stats,
 }, indent=2))
 print(f"\nWritten to {out_path}")


if __name__ == "__main__":
 main()
