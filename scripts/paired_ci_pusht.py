"""Paired-difference CIs for the Push-T 20-seed stabilizer cells.

Same methodology as paired_ci.py for TwoRoom: paired bootstrap (B=20000) +
McNemar exact test on the discordant pairs, on seeds 1.20 shared across the
three cells per episode.
"""
from __future__ import annotations
import json
from math import sqrt, comb
from pathlib import Path
import numpy as np

DIAG = Path("phase1/two_room_planner_ablation/diagnostics")

CELLS = {
 8314: {
 "N300": DIAG / "pusht-seed-sweep-ep8314-20260504-180846.json",
 "N1000": DIAG / "pusht-seed-sweep-ep8314-20260504-185414.json",
 "AMB": DIAG / "pusht-bridge-bridge-ep8314-20260504-193759.json",
 },
 14108: {
 "N300": DIAG / "pusht-seed-sweep-ep14108-n300-20260504-200623.json",
 "N1000": DIAG / "pusht-seed-sweep-ep14108-n1000-20260504-210645.json",
 "AMB": DIAG / "pusht-bridge-bridge-ep14108-20260504-214619.json",
 },
}


def load_paired(path: Path) -> dict[int, int]:
 d = json.loads(path.read_text())
 return {int(r["seed"]): 1 if r.get("success") else 0 for r in d["results"]}


def wilson(k: int, n: int, z: float = 1.96):
 if n == 0:
 return 0.0, 0.0, 0.0
 p = k / n
 denom = 1 + z * z / n
 center = (p + z * z / (2 * n)) / denom
 half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
 return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact_p(b: int, c: int) -> float:
 n = b + c
 if n == 0:
 return 1.0
 k = min(b, c)
 p = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
 return min(1.0, p * 2)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, B: int = 20000, alpha: float = 0.05, seed: int = 0):
 rng = np.random.default_rng(seed)
 n = len(a)
 diffs = np.empty(B)
 for i in range(B):
 idx = rng.integers(0, n, size=n)
 diffs[i] = a[idx].mean() - b[idx].mean()
 return float(np.mean(a - b)), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def report(name_a, name_b, A, B):
 seeds = sorted(set(A) & set(B))
 a = np.array([A[s] for s in seeds], dtype=int)
 b = np.array([B[s] for s in seeds], dtype=int)
 n = len(seeds)
 ka, kb = int(a.sum()), int(b.sum())
 pa, la, ha = wilson(ka, n)
 pb, lb, hb = wilson(kb, n)
 delta_mean, dlo, dhi = paired_bootstrap_ci(a, b)
 bb = int(((a == 1) & (b == 0)).sum())
 cc = int(((a == 0) & (b == 1)).sum())
 pval = mcnemar_exact_p(bb, cc)
 print(f" {name_a} vs {name_b} (n_paired={n}): "
 f"{ka}/{n} ({100*pa:.0f}%) vs {kb}/{n} ({100*pb:.0f}%) "
 f"| paired Δ = {100*delta_mean:+.0f} pp [{100*dlo:+.0f}, {100*dhi:+.0f}] "
 f"| McNemar disc {bb}/{cc} p={pval:.3g}")
 return {
 "pair": f"{name_a}_vs_{name_b}", "n_paired": n,
 "marginal": {name_a: {"k": ka, "ci": [la, ha]}, name_b: {"k": kb, "ci": [lb, hb]}},
 "delta_pp": {"mean": 100 * delta_mean, "ci": [100 * dlo, 100 * dhi]},
 "mcnemar": {"a_gt_b": bb, "b_gt_a": cc, "exact_p": pval},
 }


def main():
 out: dict[int, list] = {}
 for ep, paths in CELLS.items():
 if not all(p.exists() for p in paths.values()):
 print(f"\n== Episode {ep} : MISSING file(s); skipping ==")
 continue
 cells = {k: load_paired(p) for k, p in paths.items()}
 print(f"\n== Episode {ep} ==")
 out[ep] = [
 report("AMB", "N300", cells["AMB"], cells["N300"]),
 report("N1000", "N300", cells["N1000"], cells["N300"]),
 report("N1000", "AMB", cells["N1000"], cells["AMB"]),
 ]
 Path(DIAG / "pusht-paired-ci-stabilizer.json").write_text(json.dumps(out, indent=2))
 print(f"\nWritten to {DIAG / 'pusht-paired-ci-stabilizer.json'}")


if __name__ == "__main__":
 main()
