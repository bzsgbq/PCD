"""Integrate Push-T 20-seed stabilizer JSONs into per-cell summaries with Wilson 95% CI.

Handles both naming styles:
 - new (post-edit): pusht-seed-sweep-ep<EP>-n<N>-<stamp>.json
 - old (cells launched before the script edit): pusht-seed-sweep-ep<EP>-<stamp>.json

For old-style files, we infer num_samples from the median wall_s per seed
(N=300 ≈ 80s/seed, N=1000 ≈ 250s/seed on Push-T).
"""
from __future__ import annotations
import json
from math import sqrt
from pathlib import Path
from statistics import median

DIAG = Path("phase1/two_room_planner_ablation/diagnostics")


def wilson(k: int, n: int, z: float = 1.96):
 if n == 0:
 return (0.0, 0.0, 0.0)
 p = k / n
 denom = 1 + z * z / n
 center = (p + z * z / (2 * n)) / denom
 half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
 return p, max(0.0, center - half), min(1.0, center + half)


def infer_n(d: dict) -> int:
 """Infer num_samples from JSON; fall back to wall-time heuristic."""
 if "num_samples" in d:
 return int(d["num_samples"])
 walls = [r["wall_s"] for r in d["results"] if "wall_s" in r]
 if not walls:
 return -1
 m = median(walls)
 # Heuristic, calibrated on observed Push-T runs:
 # N=300 → median ~55s/seed (range ~50-80)
 # N=1000 → median ~130s/seed (range ~125-220)
 # N=3000 → median ~600s/seed
 if m < 100:
 return 300
 if m < 350:
 return 1000
 return 3000


def find_stabilizer_cells(ep: int) -> dict[str, dict]:
 """Return dict label → {k, n, src} for the 20-seed stabilizer cells."""
 out: dict[str, dict] = {}
 seed_files = sorted(DIAG.glob(f"pusht-seed-sweep-ep{ep}-*.json"),
 key=lambda p: p.stat().st_mtime)
 for p in seed_files:
 d = json.loads(p.read_text())
 if d.get("episode") != ep:
 continue
 if len(d["results"]) != 20:
 continue
 n = infer_n(d)
 label = f"vanilla N={n}"
 # If multiple files match the same label (rare), keep the most recent
 if label in out and p.stat().st_mtime <= Path(out[label]["src"]).stat().st_mtime:
 continue
 k = d.get("success_count", sum(1 for r in d["results"] if r.get("success", False)))
 out[label] = {"k": k, "n": 20, "src": str(p)}

 bridge_files = sorted(DIAG.glob(f"pusht-bridge-bridge-ep{ep}-*.json"),
 key=lambda p: p.stat().st_mtime, reverse=True)
 for p in bridge_files:
 d = json.loads(p.read_text())
 if d.get("episode") != ep or len(d["results"]) != 20:
 continue
 k = sum(1 for r in d["results"] if r.get("success", False))
 out["AMB-topK no-same-ep"] = {"k": k, "n": 20, "src": str(p)}
 break

 return out


def fmt_cell(c: dict | None) -> str:
 if c is None:
 return "(pending)"
 p, lo, hi = wilson(c["k"], c["n"])
 return f"{c['k']}/{c['n']} = {100*p:.0f}% [{100*lo:.0f}, {100*hi:.0f}]"


def main():
 summary = {}
 for ep in (8314, 14108):
 cells = find_stabilizer_cells(ep)
 summary[ep] = cells
 print(f"\n== Episode {ep} ==")
 for label in ("vanilla N=300", "vanilla N=1000", "AMB-topK no-same-ep"):
 c = cells.get(label)
 if c is None:
 print(f" {label}: pending")
 else:
 p, lo, hi = wilson(c["k"], c["n"])
 print(f" {label}: {c['k']}/{c['n']} = {100*p:.1f}% Wilson 95% CI [{100*lo:.1f}, {100*hi:.1f}]"
 f" src={Path(c['src']).name}")

 print("\n## Markdown table for §6.1\n")
 print("| Episode | Vanilla N=300 (k/20) | Vanilla N=1000 (k/20) | AMB-topK no-same-ep (k/20) |")
 print("|---------|----------------------:|----------------------:|---------------------------:|")
 for ep in (8314, 14108):
 c = summary[ep]
 cells = [
 fmt_cell(c.get("vanilla N=300")),
 fmt_cell(c.get("vanilla N=1000")),
 fmt_cell(c.get("AMB-topK no-same-ep")),
 ]
 print(f"| {ep} | " + " | ".join(cells) + " |")

 # Save machine-readable summary
 out = {
 str(ep): {
 label: ({"k": c["k"], "n": c["n"], "wilson_ci": list(wilson(c["k"], c["n"])[1:]),
 "src": c["src"]} if c is not None else None)
 for label, c in cells.items()
 }
 for ep, cells in summary.items()
 }
 Path(DIAG / "pusht-stabilizer-summary.json").write_text(json.dumps(out, indent=2))
 print(f"\nSummary JSON: {DIAG / 'pusht-stabilizer-summary.json'}")


if __name__ == "__main__":
 main()
