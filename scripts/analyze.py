"""Cross-cell + replay-fidelity analysis for Phase 1 Two-Room ablation.

For each (episode_idx, start_step), report:
 - per-cell success/failure
 - replay-in-imagination terminal_relative_mse from the diagnostic run

Then summarize:
 - per-cell success rates
 - cross-cell flip pattern (always-success / always-fail / RH=1-fragile / etc)
 - correlation between replay MSE and failure rate
"""
from __future__ import annotations
import argparse, json, glob, sys
from pathlib import Path
import numpy as np


def load_run(run_dir):
 run_dir = Path(run_dir)
 eval_set = json.loads((run_dir / "eval_set.json").read_text())
 results = json.loads((run_dir / "results.json").read_text())
 return eval_set, results


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--runs", nargs="+", required=True,
 help="paths to run directories from run_ablation.py")
 ap.add_argument("--replay", default=None,
 help="path to replay diagnostic JSON")
 args = ap.parse_args()

 # Merge per-cell results across runs (assume same eval_set across runs)
 eval_set = None
 cells = {} # name -> episode_successes list
 for r in args.runs:
 es, rs = load_run(r)
 if eval_set is None:
 eval_set = es
 else:
 assert es["episodes_idx"] == eval_set["episodes_idx"], \
 f"eval set mismatch in {r}"
 for entry in rs:
 if "episode_successes" in entry:
 cells[entry["cell"]] = list(entry["episode_successes"])
 elif "error" in entry:
 cells[entry["cell"]] = ["ERR"] * len(eval_set["episodes_idx"])

 n_eps = len(eval_set["episodes_idx"])
 cell_names = sorted(cells.keys())
 print("\n== Per-cell success rate ==")
 for c in cell_names:
 ok = [s for s in cells[c] if s is True]
 n_ok = sum(1 for s in cells[c] if s is True)
 n_err = sum(1 for s in cells[c] if s == "ERR")
 sr = 100.0 * n_ok / max(n_eps - n_err, 1)
 print(f" {c}: {n_ok}/{n_eps - n_err} = {sr:.2f}% ({n_err} errored)")

 print("\n== Per-episode pattern (rows = episodes; cols = cells) ==")
 header = f"{'idx':>3} {'ep':>5} {'start':>5} " + " ".join(f"{c:>4}" for c in cell_names)
 print(header)
 for i in range(n_eps):
 row = f"{i:>3} {eval_set['episodes_idx'][i]:>5} {eval_set['start_steps'][i]:>5} "
 for c in cell_names:
 v = cells[c][i]
 mark = "✓" if v is True else ("✗" if v is False else "·")
 row += f"{mark:>4} "
 print(row)

 # Replay correlation
 if args.replay:
 rdat = json.loads(Path(args.replay).read_text())
 rmap = {r["ep"]: r for r in rdat["results"] if "ep" in r and "terminal_relative_mse" in r}
 ep_ids = eval_set["episodes_idx"]
 rels = np.array([rmap[ep]["terminal_relative_mse"] if ep in rmap else np.nan for ep in ep_ids])
 print(f"\n== Replay MSE per episode ==")
 for i, ep in enumerate(ep_ids):
 row = f"{i:>3} ep={ep:>5} rel_mse={rels[i]:.3f} "
 for c in cell_names:
 v = cells[c][i]
 mark = "✓" if v is True else ("✗" if v is False else "·")
 row += f"{c}:{mark} "
 print(row)

 print(f"\n== Replay MSE vs per-cell failure ==")
 for c in cell_names:
 mask_ok = np.array([cells[c][i] is True for i in range(n_eps)])
 mask_fail = np.array([cells[c][i] is False for i in range(n_eps)])
 if mask_fail.sum() == 0 or mask_ok.sum() == 0:
 continue
 mu_ok = float(np.nanmean(rels[mask_ok]))
 mu_fail = float(np.nanmean(rels[mask_fail]))
 print(f" {c}: fail rel_mse mean = {mu_fail:.3f} vs success rel_mse mean = {mu_ok:.3f}")


if __name__ == "__main__":
 main()
