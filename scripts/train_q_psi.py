"""Train a minimal Q_psi(z_t, z_g, h) reachability head on frozen LeWM latents.

Following 010 spec:
 - input: z_t, z_g, h (we use 1-hot horizon bucket plus continuous h_norm)
 - output: P(reach <= h)
 - labels: positives = future state from same trajectory at horizon ≤ h.
 negatives = random-trajectory pairs and same-trajectory beyond h.
 - planner cost: -log Q

This script:
 1. Encodes a sample of frames from the offline dataset to a latent cache.
 2. Builds training tuples on top of the cache.
 3. Trains a 2-layer MLP (256-256) with BCE loss.
 4. Saves checkpoint + a small eval log.
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
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel

from run_ablation import TASKS, img_transform


# ---------------------------------------------------------------------------
# Latent cache: encode a sample of trajectory frames into latent space
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_dataset_sample(model, dataset, n_episodes, frame_stride,
 goal_offset_steps, action_block, tform, device="cuda"):
 """Encode a sample of episodes at strided frame indices.
 Returns: dict episode_id -> latent_array (T, D), step_indices (T,).
 """
 col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
 ep_indices = np.unique(dataset.get_col_data(col))
 rng = np.random.default_rng(0)
 chosen = rng.choice(ep_indices, size=min(n_episodes, len(ep_indices)),
 replace=False)
 chosen = np.sort(chosen)

 cache = {}
 for i, ep in enumerate(chosen):
 if i % 50 == 0:
 print(f" encoding ep {i+1}/{len(chosen)} (id={ep})", flush=True)
 try:
 # Find episode length
 mask = dataset.get_col_data(col) == ep
 ep_len = int(mask.sum())
 steps = np.arange(0, ep_len, frame_stride)
 chunk = dataset.load_chunk([int(ep)], [0], [int(ep_len)])[0]
 pix = chunk["pixels"]
 if pix.shape[1] != 3 and pix.shape[-1] == 3:
 pix = pix.permute(0, 3, 1, 2)
 pix_sub = pix[steps] # (T, C, H, W) raw uint8
 t_pix = torch.stack([tform(p.permute(1, 2, 0).numpy()
 if isinstance(p, torch.Tensor) else p)
 for p in pix_sub]) # (T, C, H, W) float
 enc_in = {"pixels": t_pix.unsqueeze(0).to(device)}
 enc_out = model.encode(enc_in)
 latents = enc_out["emb"][0].cpu().numpy() # (T, D)
 cache[int(ep)] = {"steps": steps.astype(int), "latents": latents}
 except Exception as e:
 print(f" ep {ep} failed: {e}")
 return cache


# ---------------------------------------------------------------------------
# Build (z_t, z_g, h_bucket, label) tuples from the latent cache
# ---------------------------------------------------------------------------

def build_pairs(cache, action_block, max_horizon_chunks, n_pos_per_ep, n_neg_per_ep, rng):
 """Construct training tuples.
 horizon is measured in latent-prediction chunks (action_block env steps each).
 """
 pos_records = [] # (z_t, z_g, h_chunks, 1)
 neg_records = [] # (z_t, z_g, h_chunks, 0)
 eps = list(cache.keys())
 for ep in eps:
 c = cache[ep]
 L = len(c["steps"])
 if L < 2:
 continue
 # positives: same-trajectory within max_horizon_chunks
 for _ in range(n_pos_per_ep):
 t_idx = rng.integers(0, L - 1)
 max_dt_chunks = min(max_horizon_chunks,
 (L - 1 - t_idx)) # in our cache stride is action_block, so 1 cache step = 1 chunk
 if max_dt_chunks < 1: continue
 dh = rng.integers(1, max_dt_chunks + 1)
 g_idx = t_idx + dh
 pos_records.append((c["latents"][t_idx], c["latents"][g_idx],
 dh, 1.0))
 # negatives: from random other trajectory at random horizon bucket
 for _ in range(n_neg_per_ep):
 t_idx = rng.integers(0, L)
 other_ep = rng.choice(eps)
 if other_ep == ep:
 continue
 c2 = cache[other_ep]
 g_idx2 = rng.integers(0, len(c2["steps"]))
 dh = rng.integers(1, max_horizon_chunks + 1)
 neg_records.append((c["latents"][t_idx], c2["latents"][g_idx2],
 dh, 0.0))
 return pos_records, neg_records


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class QPsi(nn.Module):
 def __init__(self, d_latent, max_horizon_chunks, hidden=256):
 super().__init__()
 # input: [z_t, z_g, one_hot(h)]
 in_dim = 2 * d_latent + max_horizon_chunks
 self.net = nn.Sequential(
 nn.Linear(in_dim, hidden), nn.GELU(),
 nn.Linear(hidden, hidden), nn.GELU(),
 nn.Linear(hidden, 1),
 )
 self.max_h = max_horizon_chunks

 def forward(self, z_t, z_g, h):
 h = h.long().clamp(1, self.max_h) - 1 # 0.max_h-1
 h_onehot = F.one_hot(h, num_classes=self.max_h).float()
 x = torch.cat([z_t, z_g, h_onehot], dim=-1)
 return self.net(x).squeeze(-1) # logit


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_q(model_q, pos_records, neg_records, n_epochs, batch_size, lr, device):
 pos = torch.tensor(np.stack([r[0] for r in pos_records]), dtype=torch.float32, device=device)
 posg = torch.tensor(np.stack([r[1] for r in pos_records]), dtype=torch.float32, device=device)
 posh = torch.tensor([r[2] for r in pos_records], dtype=torch.float32, device=device)
 posy = torch.ones(len(pos_records), device=device)

 neg = torch.tensor(np.stack([r[0] for r in neg_records]), dtype=torch.float32, device=device)
 negg = torch.tensor(np.stack([r[1] for r in neg_records]), dtype=torch.float32, device=device)
 negh = torch.tensor([r[2] for r in neg_records], dtype=torch.float32, device=device)
 negy = torch.zeros(len(neg_records), device=device)

 Z = torch.cat([pos, neg]); G = torch.cat([posg, negg])
 H = torch.cat([posh, negh]); Y = torch.cat([posy, negy])
 n = len(Z)
 print(f" train: {len(pos_records)} pos + {len(neg_records)} neg = {n} total")

 opt = torch.optim.AdamW(model_q.parameters(), lr=lr)
 for epoch in range(n_epochs):
 perm = torch.randperm(n, device=device)
 epoch_loss = 0.0
 epoch_correct = 0
 for s in range(0, n, batch_size):
 idx = perm[s:s+batch_size]
 logits = model_q(Z[idx], G[idx], H[idx])
 loss = F.binary_cross_entropy_with_logits(logits, Y[idx])
 opt.zero_grad(); loss.backward(); opt.step()
 epoch_loss += loss.item() * len(idx)
 epoch_correct += ((logits > 0).float() == Y[idx]).sum().item()
 print(f" epoch {epoch+1}/{n_epochs}: loss={epoch_loss/n:.4f} "
 f"acc={epoch_correct/n*100:.2f}%", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--n_episodes", type=int, default=300)
 ap.add_argument("--frame_stride", type=int, default=5,
 help="encode every Nth frame (= action_block by default)")
 ap.add_argument("--max_horizon_chunks", type=int, default=10,
 help="max latent-prediction-step horizon (h>=1)")
 ap.add_argument("--n_pos_per_ep", type=int, default=20)
 ap.add_argument("--n_neg_per_ep", type=int, default=20)
 ap.add_argument("--epochs", type=int, default=20)
 ap.add_argument("--batch_size", type=int, default=512)
 ap.add_argument("--lr", type=float, default=1e-3)
 ap.add_argument("--hidden", type=int, default=256)
 ap.add_argument("--out", default="phase1/two_room_planner_ablation/q_psi")
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

 print(f"== Encoding latent cache ({args.n_episodes} eps, stride {args.frame_stride}) ==", flush=True)
 cache = encode_dataset_sample(model, dataset, args.n_episodes,
 args.frame_stride, cfg["goal_offset_steps"],
 cfg["action_block"], tform)
 d_latent = next(iter(cache.values()))["latents"].shape[-1]
 print(f" cache: {len(cache)} eps, d_latent={d_latent}", flush=True)

 print(f"== Building pairs ==", flush=True)
 rng = np.random.default_rng(1)
 pos, neg = build_pairs(cache, cfg["action_block"], args.max_horizon_chunks,
 args.n_pos_per_ep, args.n_neg_per_ep, rng)

 print(f"== Training Q_psi (hidden={args.hidden}, epochs={args.epochs}) ==", flush=True)
 q = QPsi(d_latent, args.max_horizon_chunks, hidden=args.hidden).to("cuda")
 train_q(q, pos, neg, args.epochs, args.batch_size, args.lr, "cuda")

 stamp = time.strftime("%Y%m%d-%H%M%S")
 ckpt_path = out_dir / f"q_psi_{args.task}_{stamp}.pt"
 torch.save({"state_dict": q.state_dict(),
 "config": {"d_latent": d_latent,
 "max_horizon_chunks": args.max_horizon_chunks,
 "hidden": args.hidden,
 "task": args.task,
 "n_pos": len(pos), "n_neg": len(neg),
 "epochs": args.epochs}},
 ckpt_path)
 print(f"\nSaved Q_psi to {ckpt_path}")


if __name__ == "__main__":
 main()
