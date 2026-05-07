"""Q_psi v2 — train with hard negatives.

Negatives:
 - HARD type 1: same-trajectory pair with temporal distance > h.
 - HARD type 2: kNN-nearby latents that are NOT reachable within h
 (cheap proxy: same-trajectory pairs at temporal distance > h with
 extra weight).
 - EASY: random other-trajectory pair (kept as minority).

Validation: SPLIT BY EPISODE (not pair) to avoid same-episode leakage.
Report AUC/AP per horizon bucket.
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
from sklearn.metrics import roc_auc_score, average_precision_score

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel

from run_ablation import TASKS, img_transform
from train_q_psi import QPsi, encode_dataset_sample


# ---------------------------------------------------------------------------
# Build pairs with hard negatives
# ---------------------------------------------------------------------------

def build_pairs_v2(cache, max_horizon_chunks, n_pos_per_ep, n_hard_neg_per_ep,
 n_easy_neg_per_ep, rng):
 """Hard negatives are dominant; easy negatives kept as minority."""
 pos, neg = [], []
 eps = list(cache.keys())
 for ep in eps:
 c = cache[ep]
 L = len(c["steps"])
 if L < 2:
 continue
 # POSITIVES: same-trajectory pair within horizon
 for _ in range(n_pos_per_ep):
 t_idx = rng.integers(0, L - 1)
 max_dt = min(max_horizon_chunks, L - 1 - t_idx)
 if max_dt < 1: continue
 dh = rng.integers(1, max_dt + 1)
 g_idx = t_idx + dh
 pos.append((c["latents"][t_idx], c["latents"][g_idx], dh, 1.0))
 # HARD NEGATIVES type 1: same-trajectory beyond horizon
 for _ in range(n_hard_neg_per_ep):
 t_idx = rng.integers(0, L - 1)
 min_dt = max_horizon_chunks + 1
 if min_dt >= L - t_idx:
 continue # no beyond-horizon pair available
 dt = rng.integers(min_dt, L - t_idx)
 g_idx = t_idx + dt
 # Tell the classifier: "at horizon h, this is NOT reachable"
 # h must be ≤ max_horizon_chunks for the model's input range
 h = rng.integers(1, max_horizon_chunks + 1) # any horizon bucket
 neg.append((c["latents"][t_idx], c["latents"][g_idx], h, 0.0))
 # EASY NEGATIVES: different-trajectory pair (minority)
 for _ in range(n_easy_neg_per_ep):
 other_ep = rng.choice(eps)
 if other_ep == ep: continue
 c2 = cache[other_ep]
 t_idx = rng.integers(0, L)
 g_idx2 = rng.integers(0, len(c2["steps"]))
 h = rng.integers(1, max_horizon_chunks + 1)
 neg.append((c["latents"][t_idx], c2["latents"][g_idx2], h, 0.0))
 return pos, neg


# ---------------------------------------------------------------------------
# Training with episode-disjoint val
# ---------------------------------------------------------------------------

def train_q_with_val(model_q, train_pos, train_neg, val_pos, val_neg, max_h,
 n_epochs, batch_size, lr, device):
 def to_tensors(records):
 if not records: return None, None, None, None
 Z = torch.tensor(np.stack([r[0] for r in records]), dtype=torch.float32, device=device)
 G = torch.tensor(np.stack([r[1] for r in records]), dtype=torch.float32, device=device)
 H = torch.tensor([r[2] for r in records], dtype=torch.float32, device=device)
 Y = torch.tensor([r[3] for r in records], dtype=torch.float32, device=device)
 return Z, G, H, Y

 tr_pos = to_tensors(train_pos); tr_neg = to_tensors(train_neg)
 val_pos_t = to_tensors(val_pos); val_neg_t = to_tensors(val_neg)

 Z = torch.cat([tr_pos[0], tr_neg[0]]); G = torch.cat([tr_pos[1], tr_neg[1]])
 H = torch.cat([tr_pos[2], tr_neg[2]]); Y = torch.cat([tr_pos[3], tr_neg[3]])
 n = len(Z)
 print(f" train: {len(train_pos)} pos + {len(train_neg)} neg = {n}", flush=True)
 print(f" val: {len(val_pos)} pos + {len(val_neg)} neg = {len(val_pos)+len(val_neg)}", flush=True)

 Vz = torch.cat([val_pos_t[0], val_neg_t[0]])
 Vg = torch.cat([val_pos_t[1], val_neg_t[1]])
 Vh = torch.cat([val_pos_t[2], val_neg_t[2]])
 Vy = torch.cat([val_pos_t[3], val_neg_t[3]])

 opt = torch.optim.AdamW(model_q.parameters(), lr=lr, weight_decay=1e-4)
 best_val_auc = 0.0
 best_state = None
 for epoch in range(n_epochs):
 model_q.train()
 perm = torch.randperm(n, device=device)
 epoch_loss = 0.0
 for s in range(0, n, batch_size):
 idx = perm[s:s+batch_size]
 logits = model_q(Z[idx], G[idx], H[idx])
 loss = F.binary_cross_entropy_with_logits(logits, Y[idx])
 opt.zero_grad(); loss.backward(); opt.step()
 epoch_loss += loss.item() * len(idx)

 # Val
 model_q.eval()
 with torch.inference_mode():
 v_logits = model_q(Vz, Vg, Vh).cpu().numpy()
 v_y = Vy.cpu().numpy()
 v_p = 1 / (1 + np.exp(-v_logits))
 try:
 auc = roc_auc_score(v_y, v_p)
 ap = average_precision_score(v_y, v_p)
 except Exception:
 auc = float("nan"); ap = float("nan")
 # Per-horizon AUC
 hbk = []
 for h in range(1, max_h + 1):
 mask = (Vh.cpu().numpy() == h)
 if mask.sum() == 0: continue
 try:
 hauc = roc_auc_score(v_y[mask], v_p[mask]) if (v_y[mask].min() != v_y[mask].max()) else float("nan")
 except Exception:
 hauc = float("nan")
 hbk.append((h, int(mask.sum()), hauc))
 h_summary = ", ".join(f"h={h}:{auc:.2f}" for h, _, auc in hbk)
 print(f" ep {epoch+1}/{n_epochs}: loss={epoch_loss/n:.4f} "
 f"val AUC={auc:.4f} AP={ap:.4f} per-h: {h_summary}", flush=True)
 if auc > best_val_auc:
 best_val_auc = auc
 best_state = {k: v.detach().clone() for k, v in model_q.state_dict().items()}
 if best_state is not None:
 model_q.load_state_dict(best_state)
 print(f" loaded best val AUC = {best_val_auc:.4f}", flush=True)
 return best_val_auc


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--n_episodes", type=int, default=400)
 ap.add_argument("--frame_stride", type=int, default=5)
 ap.add_argument("--max_horizon_chunks", type=int, default=10)
 ap.add_argument("--n_pos_per_ep", type=int, default=30)
 ap.add_argument("--n_hard_neg_per_ep", type=int, default=30)
 ap.add_argument("--n_easy_neg_per_ep", type=int, default=10)
 ap.add_argument("--val_frac", type=float, default=0.2,
 help="fraction of episodes held out for validation")
 ap.add_argument("--epochs", type=int, default=30)
 ap.add_argument("--batch_size", type=int, default=512)
 ap.add_argument("--lr", type=float, default=5e-4)
 ap.add_argument("--hidden", type=int, default=256)
 ap.add_argument("--seed", type=int, default=2)
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

 print(f"== Encoding {args.n_episodes} eps ==", flush=True)
 cache = encode_dataset_sample(model, dataset, args.n_episodes, args.frame_stride,
 cfg["goal_offset_steps"], cfg["action_block"], tform)
 eps = list(cache.keys())
 rng = np.random.default_rng(args.seed)
 rng.shuffle(eps)
 n_val = int(len(eps) * args.val_frac)
 val_eps = set(eps[:n_val])
 train_eps = set(eps[n_val:])
 print(f" train eps: {len(train_eps)}, val eps: {len(val_eps)}")

 train_cache = {ep: cache[ep] for ep in train_eps}
 val_cache = {ep: cache[ep] for ep in val_eps}
 d_latent = next(iter(cache.values()))["latents"].shape[-1]

 print(f"== Building pairs ==", flush=True)
 train_pos, train_neg = build_pairs_v2(
 train_cache, args.max_horizon_chunks,
 args.n_pos_per_ep, args.n_hard_neg_per_ep, args.n_easy_neg_per_ep, rng,
 )
 val_pos, val_neg = build_pairs_v2(
 val_cache, args.max_horizon_chunks,
 args.n_pos_per_ep, args.n_hard_neg_per_ep, args.n_easy_neg_per_ep, rng,
 )

 print(f"== Training Q_psi v2 (hidden={args.hidden}, epochs={args.epochs}) ==", flush=True)
 q = QPsi(d_latent, args.max_horizon_chunks, hidden=args.hidden).to("cuda")
 best_auc = train_q_with_val(q, train_pos, train_neg, val_pos, val_neg,
 args.max_horizon_chunks, args.epochs,
 args.batch_size, args.lr, "cuda")

 stamp = time.strftime("%Y%m%d-%H%M%S")
 ckpt_path = out_dir / f"q_psi_v2_{args.task}_{stamp}.pt"
 torch.save({"state_dict": q.state_dict(),
 "config": {"d_latent": d_latent,
 "max_horizon_chunks": args.max_horizon_chunks,
 "hidden": args.hidden,
 "task": args.task,
 "n_pos_train": len(train_pos), "n_neg_train": len(train_neg),
 "n_pos_val": len(val_pos), "n_neg_val": len(val_neg),
 "epochs": args.epochs, "best_val_auc": best_auc,
 "val_eps": list(val_eps), "train_eps": list(train_eps)}},
 ckpt_path)
 print(f"\nSaved Q_psi v2 to {ckpt_path}")


if __name__ == "__main__":
 main()
