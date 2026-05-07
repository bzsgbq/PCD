"""Replay-in-imagination diagnostic.

For each (episode_idx, start_step) pair, we:
 1. Encode every observation in [start, start+goal_offset_steps]
 2. Roll JEPA from the encoded start under the dataset action segment
 3. Compute per-step latent MSE between rollout and encoded truth

This isolates open-loop predictor fidelity on *expert dataset trajectories*.
flagged that this is necessary but not sufficient — random-action and
CEM-proposal fidelity tests are required to claim the predictor is useful for
*planning* (those need env access; later script).

Usage:
 python replay_imagination.py --task tworoom \
 --eval_set phase1/two_room_planner_ablation/results/<run>/eval_set.json
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
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel


TASKS = {
 "tworoom": dict(
 env_name="swm/TwoRoom-v1",
 ckpt="tworoom/lewm",
 dataset_name="tworoom",
 keys_to_cache=["action", "proprio"],
 goal_offset_steps=25,
 history_size=1,
 ),
 "pusht": dict(
 env_name="swm/PushT-v1", ckpt="pusht/lewm", dataset_name="pusht_expert_train",
 keys_to_cache=["action", "proprio", "state"], goal_offset_steps=25, history_size=1,
 ),
 "reacher": dict(
 env_name="swm/ReacherDMControl-v0", ckpt="reacher/lewm",
 dataset_name="dmc/reacher_random",
 keys_to_cache=["action"], goal_offset_steps=25, history_size=1,
 ),
 "cube": dict(
 env_name="swm/OGBCube-v0", ckpt="cube/lewm",
 dataset_name="ogbench/cube_single_expert",
 keys_to_cache=["action"], goal_offset_steps=25, history_size=1,
 ),
}


def img_transform(img_size=224):
 return transforms.Compose([
 transforms.ToImage(),
 transforms.ToDtype(torch.float32, scale=True),
 transforms.Normalize(**spt.data.dataset_stats.ImageNet),
 transforms.Resize(size=img_size),
 ])


@torch.inference_mode()
def replay_episode(model, pixels_seq, action_seq, action_block=5, device="cuda"):
 """pixels_seq: (T, 3, H, W) torch.float32 normalized at every env step.
 action_seq: (T, A_raw) raw env actions.
 action_block: how many env steps the JEPA's action_encoder consumes per
 latent step. JEPA expects actions chunked to (T/action_block, A_raw*action_block).
 We compare predicted latents to the encoded *frame at the end of each chunk*."""
 T_env = pixels_seq.shape[0]
 pixels_seq = pixels_seq.to(device)
 action_seq = action_seq.to(device)

 # JEPA convention: each "chunk" indexes one frame at position i * action_block.
 # We take T_chunks chunks of action_block actions each. With T_env=25,
 # action_block=5, we have chunks at frame indices [0, 5, 10, 15, 20] and
 # T_chunks = 5; these are the teacher latents we compare against.
 n_full_chunks = T_env // action_block # e.g. 25/5=5
 chunk_frame_idx = torch.arange(0, n_full_chunks * action_block, action_block,
 device=device) # [0,5,10,15,20]
 T_chunks = n_full_chunks # 5

 # Truncate actions to exactly T_chunks * action_block raw env actions.
 T_act_used = T_chunks * action_block
 action_seq = action_seq[:T_act_used]

 # Encode every required frame as a teacher target (just the chunk-start frames).
 needed_frames = pixels_seq[chunk_frame_idx] # (T_chunks, C, H, W)
 enc_in = {"pixels": needed_frames.unsqueeze(0)} # (1, T_chunks, C, H, W)
 enc_out = model.encode(enc_in)
 z_true = enc_out["emb"][0] # (T_chunks, D)

 # Group raw actions into action_block-sized chunks: (T_chunks, A_raw*action_block).
 A_raw = action_seq.shape[-1]
 grouped = action_seq.view(T_chunks, action_block * A_raw) # (T_chunks, A_raw*ab)

 # JEPA.rollout signature:
 # info["pixels"]: (B, S, H_hist, C, H, W) — H_hist = history_size in chunks
 # action_sequence: (B, S, T_chunks, A_raw*ab)
 H_hist = 1
 init_pix = needed_frames[:1].unsqueeze(0).unsqueeze(0) # first chunk frame (1, 1, 1, C, H, W)
 actions = grouped.unsqueeze(0).unsqueeze(0) # (1, 1, T_chunks, A_raw*ab)
 info = {"pixels": init_pix.to(device), "action": actions.to(device)}
 info = model.rollout(info, info["action"], history_size=H_hist)
 z_pred = info["predicted_emb"][0, 0] # (T_pred, D)

 # Drop the first entry of z_pred — it's the encoded init, not a prediction.
 # JEPA.rollout starts emb with the init and appends predictions; final length
 # is H_hist + (T_chunks - H_hist + 1) = T_chunks + 1, so z_pred[1:] are
 # predictions for chunks 1.T_chunks.
 z_pred = z_pred[1:] # (T_chunks_pred, D)
 z_true_pred = z_true[1:] # compare predictions to chunks 1.T_chunks
 L = min(z_pred.shape[0], z_true_pred.shape[0])
 z_pred = z_pred[:L]
 z_true_align = z_true_pred[:L]
 per_step_sq = (z_pred - z_true_align).pow(2).sum(-1).cpu().numpy()
 norm_true = z_true_align.pow(2).sum(-1).cpu().numpy()
 return {
 "T_chunks_aligned": int(L),
 "T_env_used": int(T_act_used),
 "per_step_mse": per_step_sq.tolist(),
 "per_step_norm_true": norm_true.tolist(),
 "terminal_mse": float(per_step_sq[-1]),
 "mean_mse": float(per_step_sq.mean()),
 "terminal_relative_mse": float(per_step_sq[-1] / max(norm_true[-1], 1e-8)),
 }


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom", choices=list(TASKS.keys()))
 ap.add_argument("--eval_set", required=True)
 ap.add_argument("--num", type=int, default=16)
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
 tform = img_transform()

 eset = json.loads(Path(args.eval_set).read_text())
 episodes_idx = eset["episodes_idx"][:args.num]
 start_steps = eset["start_steps"][:args.num]

 out = {"task": args.task, "num": args.num, "episodes_idx": episodes_idx,
 "start_steps": start_steps, "results": []}

 for ep, start in zip(episodes_idx, start_steps):
 try:
 chunks = dataset.load_chunk([int(ep)], [int(start)],
 [int(start) + cfg["goal_offset_steps"]])
 ch = chunks[0]
 pix = ch["pixels"] # (T,..)
 if pix.shape[1] != 3 and pix.shape[-1] == 3:
 pix = pix.permute(0, 3, 1, 2)
 pix_t = torch.stack([tform(p.permute(1, 2, 0).numpy()
 if isinstance(p, torch.Tensor) else p)
 for p in pix])
 act = ch["action"].float() if torch.is_tensor(ch["action"]) \
 else torch.tensor(ch["action"], dtype=torch.float32)
 r = replay_episode(model, pix_t, act)
 r["ep"] = int(ep); r["start"] = int(start)
 out["results"].append(r)
 print(f" ep={ep} start={start} : T_chunks={r['T_chunks_aligned']} "
 f"terminal_mse={r['terminal_mse']:.3f} "
 f"mean_mse={r['mean_mse']:.3f} "
 f"rel_terminal={r['terminal_relative_mse']:.4f}")
 except Exception as e:
 import traceback; traceback.print_exc()
 out["results"].append({"ep": int(ep), "start": int(start), "error": str(e)})

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{args.task}-replay-{stamp}.json"
 out_path.write_text(json.dumps(out, indent=2, default=str))
 oks = [r for r in out["results"] if "terminal_mse" in r]
 if oks:
 means = np.array([r["mean_mse"] for r in oks])
 terms = np.array([r["terminal_mse"] for r in oks])
 rels = np.array([r["terminal_relative_mse"] for r in oks])
 print(f"\nWritten to {out_path}")
 print(f" episodes ok: {len(oks)}/{len(out['results'])}")
 print(f" mean step MSE: median={np.median(means):.3f}, mean={means.mean():.3f}")
 print(f" terminal MSE: median={np.median(terms):.3f}, mean={terms.mean():.3f}")
 print(f" terminal relative MSE: median={np.median(rels):.4f}, mean={rels.mean():.4f}")


if __name__ == "__main__":
 main()
