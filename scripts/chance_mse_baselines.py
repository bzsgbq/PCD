"""Three chance-MSE baselines for the SIGReg latent space:

 1. Random pair: distance between two encoded latents from random episodes/steps.
 2. Same-trajectory future pair: distance between latent(start) and latent(start+goal_offset)
 from the same trajectory. This is the "chance" that any planner would have to beat.
 3. Expert replay terminal MSE: predicted vs encoded (already in replay_imagination.py).

We report squared L2 distance (matching the JEPA criterion) and a relative version
(squared distance / (latent norm)^2 of the *target* latent).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_LEWM_DIR = Path(__file__).resolve().parents[3] / "le-wm"
if _LEWM_DIR.exists() and str(_LEWM_DIR) not in sys.path:
 sys.path.insert(0, str(_LEWM_DIR))
if str(Path(__file__).parent) not in sys.path:
 sys.path.insert(0, str(Path(__file__).parent))

import _threadlimits # noqa: F401 # CPU thread limits, BEFORE numpy/torch/sklearn

import numpy as np
import torch

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel

from run_ablation import TASKS, img_transform


@torch.inference_mode()
def encode_pixels(model, pixels):
 enc_in = {"pixels": pixels.unsqueeze(0).to("cuda")}
 out = model.encode(enc_in)
 return out["emb"][0] # (T, D)


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--n_random_pairs", type=int, default=200)
 ap.add_argument("--n_same_traj_pairs", type=int, default=200)
 ap.add_argument("--seed", type=int, default=42)
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

 rng = np.random.default_rng(args.seed)
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 ep_indices = np.unique(dataset.get_col_data(col))
 step_idx = dataset.get_col_data("step_idx")
 ep_step = dataset.get_col_data(col)

 # Build {ep -> max step}
 ep_max = {}
 for ep in ep_indices:
 ep_max[int(ep)] = int(np.max(step_idx[ep_step == ep]))

 # ---- 1. Random pair latent squared distance ----
 print(f"\n== Random pair baseline (n={args.n_random_pairs}) ==", flush=True)
 rand_results = []
 for i in range(args.n_random_pairs):
 ep_a = int(rng.choice(ep_indices))
 ep_b = int(rng.choice(ep_indices))
 s_a = int(rng.integers(0, ep_max[ep_a] + 1))
 s_b = int(rng.integers(0, ep_max[ep_b] + 1))
 try:
 ch_a = dataset.load_chunk([ep_a], [s_a], [s_a + 1])[0]
 ch_b = dataset.load_chunk([ep_b], [s_b], [s_b + 1])[0]
 pa = ch_a["pixels"][0]
 pb = ch_b["pixels"][0]
 if pa.ndim == 3 and pa.shape[0] == 3:
 pa = pa.permute(1, 2, 0)
 pb = pb.permute(1, 2, 0)
 t_a = tform(pa.numpy() if hasattr(pa, "numpy") else pa).unsqueeze(0)
 t_b = tform(pb.numpy() if hasattr(pb, "numpy") else pb).unsqueeze(0)
 z_a = encode_pixels(model, t_a)[0]
 z_b = encode_pixels(model, t_b)[0]
 sq_dist = float(((z_a - z_b) ** 2).sum())
 norm_b = float((z_b ** 2).sum())
 rand_results.append({"sq_dist": sq_dist, "norm_b": norm_b,
 "rel": sq_dist / max(norm_b, 1e-8)})
 except Exception as e:
 print(f" pair {i} failed: {e}")
 rand_sq = np.array([r["sq_dist"] for r in rand_results])
 rand_rel = np.array([r["rel"] for r in rand_results])
 print(f" random sq_dist: median={np.median(rand_sq):.2f} mean={rand_sq.mean():.2f}")
 print(f" random rel: median={np.median(rand_rel):.3f} mean={rand_rel.mean():.3f}")

 # ---- 2. Same-trajectory future pair (start vs start+goal_offset) ----
 goal_offset = cfg["goal_offset_steps"]
 print(f"\n== Same-trajectory pair baseline (offset={goal_offset}, n={args.n_same_traj_pairs}) ==",
 flush=True)
 same_results = []
 for i in range(args.n_same_traj_pairs):
 ep = int(rng.choice(ep_indices))
 max_start = ep_max[ep] - goal_offset
 if max_start <= 0:
 continue
 s = int(rng.integers(0, max_start + 1))
 try:
 ch = dataset.load_chunk([ep], [s], [s + goal_offset + 1])[0]
 pix = ch["pixels"]
 p_start = pix[0]
 p_goal = pix[-1]
 if p_start.ndim == 3 and p_start.shape[0] == 3:
 p_start = p_start.permute(1, 2, 0); p_goal = p_goal.permute(1, 2, 0)
 t_a = tform(p_start.numpy() if hasattr(p_start, "numpy") else p_start).unsqueeze(0)
 t_b = tform(p_goal.numpy() if hasattr(p_goal, "numpy") else p_goal).unsqueeze(0)
 z_a = encode_pixels(model, t_a)[0]
 z_b = encode_pixels(model, t_b)[0]
 sq_dist = float(((z_a - z_b) ** 2).sum())
 norm_b = float((z_b ** 2).sum())
 same_results.append({"sq_dist": sq_dist, "norm_b": norm_b,
 "rel": sq_dist / max(norm_b, 1e-8)})
 except Exception as e:
 print(f" pair {i} failed: {e}")
 same_sq = np.array([r["sq_dist"] for r in same_results])
 same_rel = np.array([r["rel"] for r in same_results])
 print(f" same-traj sq_dist: median={np.median(same_sq):.2f} mean={same_sq.mean():.2f}")
 print(f" same-traj rel: median={np.median(same_rel):.3f} mean={same_rel.mean():.3f}")

 # Persist
 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-chance-{stamp}.json"
 out_path.write_text(json.dumps({
 "task": args.task,
 "random_pair": {"n": len(rand_sq), "median_sq": float(np.median(rand_sq)),
 "mean_sq": float(rand_sq.mean()),
 "median_rel": float(np.median(rand_rel)),
 "mean_rel": float(rand_rel.mean())},
 "same_traj_pair": {"n": len(same_sq), "offset": goal_offset,
 "median_sq": float(np.median(same_sq)),
 "mean_sq": float(same_sq.mean()),
 "median_rel": float(np.median(same_rel)),
 "mean_rel": float(same_rel.mean())},
 }, indent=2))
 print(f"\nWritten to {out_path}")


if __name__ == "__main__":
 main()
