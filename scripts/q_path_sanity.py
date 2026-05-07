"""Q_psi path-wise sanity check.

For a given trajectory of pixel observations, encode each frame to z, then
compute Q_psi(z_t, z_goal, h_remaining) at every t. A correct Q should:
 - increase monotonically toward 1 along the expert path that reaches the goal,
 - stay low along a failed-planner path that doesn't reach the goal,
 - NOT increase for a "shuffled goal" path where the goal latent is randomly
 drawn from a different episode.
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
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel

from run_ablation import TASKS, img_transform
from train_q_psi import QPsi


@torch.inference_mode()
def encode_frames(model, frames_uint8, tform, device="cuda"):
 """frames_uint8: (T, H, W, 3) uint8."""
 pix = torch.stack([tform(f) for f in frames_uint8]) # (T, C, H, W)
 out = model.encode({"pixels": pix.unsqueeze(0).to(device)})
 return out["emb"][0] # (T, D)


@torch.inference_mode()
def q_along_path(q, z_t_seq, z_goal, max_horizon, device="cuda"):
 """Compute Q(z_t, z_goal, h_remaining) for every t. h_remaining =
 max(1, max_horizon - chunk_index). Assumes z_t_seq is sampled at action_block
 stride (so each index is a chunk)."""
 T = z_t_seq.shape[0]
 out = []
 for i in range(T):
 h = max(1.0, float(max_horizon - i))
 h_t = torch.tensor([h], device=device)
 logit = q(z_t_seq[i:i+1], z_goal.unsqueeze(0), h_t).item()
 out.append({"t_chunk": i, "logit": logit, "p_reach": float(torch.sigmoid(torch.tensor(logit)).item())})
 return out


def stride_pixels_from_trace(trace_record, action_block):
 """Return (T_chunks, H, W, 3) uint8 — one frame per latent chunk."""
 steps = trace_record["env_steps"]
 arr = []
 for i in range(0, len(steps), action_block):
 pix = np.asarray(steps[i]["pixels"][0], dtype=np.uint8) # (H, W, 3)
 arr.append(pix)
 return np.stack(arr)


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--q_ckpt", default="phase1/two_room_planner_ablation/q_psi/q_psi_tworoom_20260503-151027.pt")
 ap.add_argument("--trace", required=True)
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/diagnostics")
 args = ap.parse_args()

 out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
 cfg = TASKS[args.task]

 model = AutoCostModel(cfg["ckpt"]).to("cuda").eval()
 model.requires_grad_(False)
 if hasattr(model, "interpolate_pos_encoding"):
 model.interpolate_pos_encoding = True
 tform = img_transform()

 qckpt = torch.load(args.q_ckpt, weights_only=False, map_location="cuda")
 qcfg = qckpt["config"]
 q = QPsi(qcfg["d_latent"], qcfg["max_horizon_chunks"], hidden=qcfg["hidden"]).to("cuda")
 q.load_state_dict(qckpt["state_dict"]); q.eval(); q.requires_grad_(False)
 max_horizon = qcfg["max_horizon_chunks"]

 trace = json.loads(Path(args.trace).read_text())

 # Load dataset to fetch the real goal frame (episode goal_offset_steps
 # after the start). Without this, Q is being asked about reachability
 # to whatever the agent eventually saw — meaningless.
 dataset = swm.data.HDF5Dataset(
 cfg["dataset_name"], keys_to_cache=cfg["keys_to_cache"],
 cache_dir=swm.data.utils.get_cache_dir(),
 )

 results = []
 for entry in trace:
 if "env_steps" not in entry:
 continue
 ep = int(entry["episode_idx"])
 # eval_set start_step (we don't have it directly here; we know it from the canonical eval set).
 # Pull from canonical eval_set.json:
 eset = json.loads(Path(
 "phase1/two_room_planner_ablation/results/tworoom-20260503-121919/eval_set.json"
 ).read_text())
 start_lookup = dict(zip(eset["episodes_idx"], eset["start_steps"]))
 start_step = start_lookup.get(ep, entry.get("start_step", 0))
 goal_step = start_step + cfg["goal_offset_steps"]
 chunk = dataset.load_chunk([ep], [goal_step], [goal_step + 1])[0]
 gp = chunk["pixels"][0]
 if gp.ndim == 3 and gp.shape[0] == 3:
 gp = gp.permute(1, 2, 0)
 gp_np = gp.numpy() if hasattr(gp, "numpy") else np.asarray(gp)
 goal_pix = gp_np.astype(np.uint8)
 z_goal = encode_frames(model, [goal_pix], tform)[0]
 chunk_pix = stride_pixels_from_trace(entry, cfg["action_block"])
 z_t_seq = encode_frames(model, chunk_pix, tform)
 path = q_along_path(q, z_t_seq, z_goal, max_horizon)
 results.append({
 "cell": entry["cell"], "episode_idx": entry["episode_idx"],
 "success": entry["success"],
 "n_chunks": len(path),
 "Q_path": path,
 "Q_logit_first": path[0]["logit"], "Q_logit_last": path[-1]["logit"],
 "Q_p_first": path[0]["p_reach"], "Q_p_last": path[-1]["p_reach"],
 })
 print(f" ep={entry['episode_idx']} cell={entry['cell']:>6} success={entry['success']} "
 f" Q logit start={path[0]['logit']:+.2f} end={path[-1]['logit']:+.2f}"
 f" P_reach start={path[0]['p_reach']:.3f} end={path[-1]['p_reach']:.3f}",
 flush=True)
 # Shuffled-goal control: a random *other-episode* frame as goal.
 # test: Q must NOT increase along this path's progression.
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 all_eps = np.unique(dataset.get_col_data(col))
 rng = np.random.default_rng(0)
 for _ in range(10): # try a few until we get a different episode
 other_ep = int(rng.choice(all_eps))
 if other_ep != ep: break
 rand_step = int(rng.integers(0, 50)) # arbitrary step
 ch_other = dataset.load_chunk([other_ep], [rand_step], [rand_step + 1])[0]
 gp_o = ch_other["pixels"][0]
 if gp_o.ndim == 3 and gp_o.shape[0] == 3:
 gp_o = gp_o.permute(1, 2, 0)
 gp_o_np = gp_o.numpy() if hasattr(gp_o, "numpy") else np.asarray(gp_o)
 z_goal_shuf = encode_frames(model, [gp_o_np.astype(np.uint8)], tform)[0]
 path_shuf = q_along_path(q, z_t_seq, z_goal_shuf, max_horizon)
 results[-1]["Q_path_shuffled_goal"] = path_shuf
 results[-1]["shuffled_goal_ep"] = other_ep
 results[-1]["shuffled_goal_step"] = rand_step
 print(f" shuffled-goal control (ep={other_ep}, step={rand_step}): "
 f"start P={path_shuf[0]['p_reach']:.3f} end P={path_shuf[-1]['p_reach']:.3f}", flush=True)

 # same-trajectory beyond horizon. Goal = SAME episode but
 # at step way past start (e.g. step start_step + 200 if dataset allows),
 # asked at small h. The retrained Q must say P_reach is LOW for h<<200/5.
 try:
 far_step = min(start_step + 200, 250) # well beyond max_horizon_chunks*action_block=50
 ch_far = dataset.load_chunk([ep], [far_step], [far_step + 1])[0]
 gp_f = ch_far["pixels"][0]
 if gp_f.ndim == 3 and gp_f.shape[0] == 3:
 gp_f = gp_f.permute(1, 2, 0)
 gp_f_np = gp_f.numpy() if hasattr(gp_f, "numpy") else np.asarray(gp_f)
 z_goal_far = encode_frames(model, [gp_f_np.astype(np.uint8)], tform)[0]
 path_far = q_along_path(q, z_t_seq, z_goal_far, max_horizon)
 results[-1]["Q_path_far_same_traj"] = path_far
 results[-1]["far_step"] = far_step
 print(f" same-traj-beyond-horizon (ep={ep}, step={far_step}): "
 f"start P={path_far[0]['p_reach']:.3f} end P={path_far[-1]['p_reach']:.3f}",
 flush=True)
 except Exception as e:
 print(f" far-step test failed: {e}", flush=True)

 stamp = time.strftime("%Y%m%d-%H%M%S")
 out_path = out_dir / f"{cfg['env_name'].split('/')[-1]}-q-path-{stamp}.json"
 out_path.write_text(json.dumps(results, indent=2, default=str))
 print(f"\nSaved to {out_path}")


if __name__ == "__main__":
 main()
