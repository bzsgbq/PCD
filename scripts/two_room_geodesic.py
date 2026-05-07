"""BFS geodesic on Two-Room.

Extract wall+door layout from a rendered Two-Room frame (binarize: dark
pixels = walls, white = free), then BFS from agent xy to goal xy through
free space.

Sanity checks reviewer requested:
 - expert replay geodesic should generally decrease;
 - if start & goal are in different rooms, Euclidean and BFS distances
 should differ materially.

Reports Δd_geo = d_geo(s_0, g) − d_geo(s_T, g). Positive = progress.
"""
from __future__ import annotations
import argparse, json
from collections import deque
from pathlib import Path

import numpy as np


def extract_walls(frame, dark_thresh=30, agent_radius=15, goal_radius=15):
 """frame: (H, W, 3) uint8.
 Returns: (H, W) bool mask, True = wall (impassable)."""
 grey = frame.mean(-1) if frame.ndim == 3 else frame
 walls = grey < dark_thresh
 # Be defensive: dilate by 1 pixel so the agent radius doesn't tunnel through.
 pad = walls.copy()
 pad[1:, :] |= walls[:-1, :]; pad[:-1, :] |= walls[1:, :]
 pad[:, 1:] |= walls[:, :-1]; pad[:, :-1] |= walls[:, 1:]
 return pad


def bfs_distance(walls, start_xy, end_xy):
 """walls: (H, W) bool. start_xy/end_xy in pixel coords (x, y) — note
 that frame[y, x] is the convention. Returns geodesic distance in pixel
 steps (8-connected), or +inf if unreachable."""
 H, W = walls.shape
 sx, sy = int(round(start_xy[0])), int(round(start_xy[1]))
 ex, ey = int(round(end_xy[0])), int(round(end_xy[1]))
 sx = max(0, min(W-1, sx)); sy = max(0, min(H-1, sy))
 ex = max(0, min(W-1, ex)); ey = max(0, min(H-1, ey))
 # If start or goal is inside a wall, snap to nearest free pixel
 if walls[sy, sx]:
 sx, sy = nearest_free(walls, (sx, sy))
 if walls[ey, ex]:
 ex, ey = nearest_free(walls, (ex, ey))
 dist = np.full((H, W), np.inf, dtype=np.float64)
 dist[sy, sx] = 0
 q = deque([(sx, sy)])
 # 8-connected neighbors with sqrt(2) diagonal cost
 NBR = [(-1,-1,1.4142),(-1,0,1),(-1,1,1.4142),
 (0,-1,1), (0,1,1),
 (1,-1,1.4142),(1,0,1),(1,1,1.4142)]
 while q:
 x, y = q.popleft()
 d = dist[y, x]
 if (x, y) == (ex, ey):
 return float(d)
 for dx, dy, w in NBR:
 nx, ny = x + dx, y + dy
 if 0 <= nx < W and 0 <= ny < H and not walls[ny, nx] and dist[ny, nx] > d + w:
 dist[ny, nx] = d + w
 q.append((nx, ny))
 return float(dist[ey, ex])


def nearest_free(walls, xy, max_radius=20):
 H, W = walls.shape
 x0, y0 = xy
 for r in range(1, max_radius + 1):
 for dx in range(-r, r + 1):
 for dy in range(-r, r + 1):
 if abs(dx) != r and abs(dy) != r: continue
 x, y = x0 + dx, y0 + dy
 if 0 <= x < W and 0 <= y < H and not walls[y, x]:
 return x, y
 return x0, y0 # give up


def euclidean(a, b):
 return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


# ---------------------------------------------------------------------------
# Apply to an executed trajectory from a per_step_trace.py JSON
# ---------------------------------------------------------------------------

def trajectory_geodesics(trace_record):
 """Given one entry of a per_step_trace.py output, compute the geodesic
 distance to goal at every env step, plus Δd_geo."""
 steps = trace_record["env_steps"]
 if len(steps) == 0:
 return None
 # Pull first frame's pixels to extract walls. Pixels saved as nested lists.
 pix0 = np.asarray(steps[0]["pixels"][0], dtype=np.uint8) # (H, W, 3)
 walls = extract_walls(pix0)
 goal_xy = steps[0]["goal_state"][0]
 geos = []
 for s in steps:
 agent_xy = s["proprio"][0]
 d_geo = bfs_distance(walls, agent_xy, goal_xy)
 d_eucl = euclidean(agent_xy, goal_xy)
 geos.append({"t": s["t"], "agent_xy": agent_xy, "geo": d_geo, "eucl": d_eucl})
 delta_geo = geos[0]["geo"] - geos[-1]["geo"]
 delta_eucl = geos[0]["eucl"] - geos[-1]["eucl"]
 return {"per_step": geos, "delta_geo": delta_geo, "delta_eucl": delta_eucl,
 "start_geo": geos[0]["geo"], "end_geo": geos[-1]["geo"]}


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--trace", required=True, help="per_step_trace.py JSON")
 ap.add_argument("--out")
 args = ap.parse_args()

 data = json.loads(Path(args.trace).read_text())
 out = []
 for entry in data:
 if "env_steps" not in entry:
 print(f" skipping {entry.get('cell', '?')} ep={entry.get('episode_idx', '?')} (no env_steps)")
 continue
 result = trajectory_geodesics(entry)
 result["cell"] = entry["cell"]
 result["episode_idx"] = entry["episode_idx"]
 result["success"] = entry.get("success")
 out.append(result)
 s = result
 print(f" ep={result['episode_idx']} cell={result['cell']:>6} "
 f"success={result['success']} Δd_geo={s['delta_geo']:+.1f} "
 f"Δd_eucl={s['delta_eucl']:+.1f} "
 f"start_geo={s['start_geo']:.1f} end_geo={s['end_geo']:.1f}")

 if args.out:
 # Strip per_step (verbose) before saving aggregate
 slim = [{k: v for k, v in r.items() if k != "per_step"} for r in out]
 Path(args.out).write_text(json.dumps(slim, indent=2, default=str))
 print(f"\nSaved aggregate to {args.out}")


if __name__ == "__main__":
 main()
