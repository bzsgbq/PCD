"""Aggregate the 5-seed grid into a single canonical JSON.

Reads the per-seed `results.json` files for tworoom/pusht/cube/reacher × seeds
{0,1,2,3,42} and produces:

 - per `(task, seed)` block: cell -> {k, n, rate_pct, wilson_ci}
 (raw `episode_successes` lists are kept inside the source result
 directories under `results/<task>-<stamp>/results.json`; the canonical
 summary stops at k/n + Wilson CIs and points at `results_dir` for the
 per-episode booleans.)
 - per `task` aggregate block: cell -> {pooled_k, pooled_n=250, pooled_rate_pct,
 pooled_wilson_ci, per_seed[5] = {seed, k, n, rate_pct, wilson_ci}}
 - top-level provenance: result directories, log files, run config (sbs=1,
 num_eval=50, coupled --seed=--solver_seed=s for all rows).

Output:
 diagnostics/grid-path2-summary.json (canonical, overwritten on every run)
 diagnostics/grid-path2-summary.prepathd.bak.json (pre-snapshot;
 created the first time this script overwrites a prior canonical summary
 and never overwritten on subsequent runs, so it remains the genuine
 pre-baseline. Pass `--no-backup` to skip the backup attempt.)
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

DIAG = Path("phase1/two_room_planner_ablation/diagnostics")
RESULTS = Path("phase1/two_room_planner_ablation/results")
LOGS = Path("phase1/two_room_planner_ablation/logs")
SUMMARY = DIAG / "grid-path2-summary.json"
BACKUP = DIAG / "grid-path2-summary.prepathd.bak.json"

CELL_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A0_H10", "A2_H10"]
TASKS = ["tworoom", "pusht", "cube", "reacher"]
SEEDS = [0, 1, 2, 3, 42]


def wilson(k, n, z=1.96):
 if n == 0:
 return [0.0, 0.0]
 p = k / n
 denom = 1 + z * z / n
 center = (p + z * z / (2 * n)) / denom
 half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
 lo = max(0.0, 100 * (center - half))
 hi = min(100.0, 100 * (center + half))
 return [round(lo, 1), round(hi, 1)]


def find_result_dir(task, seed, results_root):
 """Find a result directory matching task config; prefer the one referencing
 the requested seed in its config.json."""
 cands = sorted([d for d in results_root.glob(f"{task}-*") if d.is_dir()])
 matches = []
 for d in cands:
 cfg_path = d / "config.json"
 if not cfg_path.exists():
 continue
 cfg = json.loads(cfg_path.read_text())
 if cfg.get("seed") == seed and cfg.get("solver_seed") == seed:
 matches.append((d, cfg))
 if not matches:
 return None
 # If multiple matches, prefer the most recent (mtime).
 matches.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
 return matches[0]


def find_log_file(task, seed):
 """Find a recent run-log for this (task, seed). Prefer launcher
 log filenames; fall back to legacy filenames including non-gpu-slot patterns."""
 candidates = list(LOGS.glob(f"grid5seed_{task}_seed{seed}_gpu*_slot*.log"))
 candidates += list(LOGS.glob(f"grid5seed_{task}_seed{seed}_gpu*.log"))
 candidates += list(LOGS.glob(f"grid_{task}_seed{seed}_gpu*.log"))
 candidates += list(LOGS.glob(f"grid_{task}_seed{seed}.log"))
 if not candidates:
 return None
 candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
 return candidates[0]


def cell_results(d):
 """Return cell -> {k, n, episode_successes} from results.json."""
 rj = json.loads((d / "results.json").read_text())
 out = {}
 for entry in rj:
 name = entry.get("cell")
 if name is None:
 continue
 succs = entry.get("episode_successes")
 if succs is None:
 # Legacy fallback: derive from success_rate * num_eval if successes list missing.
 sr = entry.get("success_rate")
 nev = entry.get("num_eval", 50)
 if sr is None:
 continue
 k = int(round(sr * nev / 100.0))
 n = nev
 succs = None
 else:
 succs = [bool(s) for s in succs]
 n = len(succs)
 k = sum(succs)
 out[name] = {
 "k": int(k),
 "n": int(n),
 "rate_pct": round(100.0 * k / n, 1) if n else 0.0,
 "wilson_ci": wilson(k, n),
 "episode_successes": succs,
 }
 return out


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--dry-run", action="store_true")
 ap.add_argument("--no-backup", action="store_true")
 args = ap.parse_args()

 out = {
 "grid": "5-seed completion grid",
 "config": "--solver_batch_size 1, --num_eval 50, coupled --seed s --solver_seed s for s in {0,1,2,3,42}, all 4 tasks",
 "wilson_ci_schema": "[lower_pct, upper_pct]",
 "tasks": {},
 }

 missing = []
 for task in TASKS:
 per_task = {
 "seeds": [],
 "per_seed": {},
 "pooled_per_cell": {},
 }

 per_seed_cell_counts = {c: [] for c in CELL_ORDER}
 per_seed_cell_succ = {c: [] for c in CELL_ORDER}

 for seed in SEEDS:
 hit = find_result_dir(task, seed, RESULTS)
 if hit is None:
 missing.append((task, seed))
 continue
 rdir, cfg = hit
 cells = cell_results(rdir)
 log = find_log_file(task, seed)

 per_task["seeds"].append(seed)
 per_task["per_seed"][f"seed{seed}"] = {
 "seed": seed,
 "solver_seed": cfg.get("solver_seed"),
 "num_eval": cfg.get("num_eval"),
 "solver_batch_size": cfg.get("solver_batch_size"),
 "results_dir": str(rdir.relative_to(RESULTS.parent)),
 "log_file": str(log.relative_to(RESULTS.parent)) if log else None,
 "cells": {c: {k: v for k, v in cells[c].items() if k != "episode_successes"}
 for c in CELL_ORDER if c in cells},
 }

 for c in CELL_ORDER:
 if c not in cells:
 continue
 per_seed_cell_counts[c].append((cells[c]["k"], cells[c]["n"]))
 if cells[c].get("episode_successes") is not None:
 per_seed_cell_succ[c].append(cells[c]["episode_successes"])

 # Pool across seeds
 for c in CELL_ORDER:
 ks_ns = per_seed_cell_counts.get(c, [])
 if not ks_ns:
 continue
 pooled_k = sum(k for k, _ in ks_ns)
 pooled_n = sum(n for _, n in ks_ns)
 per_task["pooled_per_cell"][c] = {
 "pooled_k": int(pooled_k),
 "pooled_n": int(pooled_n),
 "pooled_rate_pct": round(100.0 * pooled_k / pooled_n, 1) if pooled_n else 0.0,
 "pooled_wilson_ci": wilson(pooled_k, pooled_n),
 "per_seed_kn": [
 {"seed": s, "k": k, "n": n, "rate_pct": round(100.0 * k / n, 1) if n else 0.0}
 for s, (k, n) in zip(per_task["seeds"], ks_ns)
 ],
 }

 out["tasks"][task] = per_task

 if missing:
 print("WARNING: missing (task, seed) result directories:")
 for ts in missing:
 print(f" - {ts}")
 if not args.dry_run:
 print("Aborting; rerun with --dry-run to inspect partial output.", file=sys.stderr)
 sys.exit(1)

 # Summary print
 print("Pooled per-task per-cell summary (k/n; rate_pct):")
 for task in TASKS:
 pt = out["tasks"].get(task, {})
 if not pt:
 continue
 seeds = pt.get("seeds", [])
 print(f" {task} ({len(seeds)} seeds: {seeds}):")
 for c in CELL_ORDER:
 cell = pt.get("pooled_per_cell", {}).get(c)
 if cell is None:
 continue
 print(f" {c}: {cell['pooled_k']}/{cell['pooled_n']} = {cell['pooled_rate_pct']}% CI {cell['pooled_wilson_ci']}")

 if args.dry_run:
 return

 if SUMMARY.exists() and not args.no_backup and not BACKUP.exists():
 shutil.copy2(SUMMARY, BACKUP)
 print(f"Backed up prior summary -> {BACKUP}")
 elif BACKUP.exists():
 print(f"Pre-backup already exists at {BACKUP}; not overwriting.")

 SUMMARY.write_text(json.dumps(out, indent=2))
 print(f"Wrote canonical summary -> {SUMMARY}")


if __name__ == "__main__":
 main()
