"""Paired-difference CIs for the 100-seed TwoRoom 4339 headline cells.

Reports for each pair (A, B):
 - per-cell Wilson CI for marginal rate
 - paired delta (mean A - mean B) and bootstrap 95% CI
 - McNemar discordant pair counts and exact two-sided p-value
"""
from __future__ import annotations
import json
import numpy as np
from math import sqrt, comb
from pathlib import Path

DIAG = Path("phase1/two_room_planner_ablation/diagnostics")

CELLS = {
 "A5_N300": DIAG / "tworoom-seed-sweep-ep4339-20260504-142931.json",
 "A5_N1000": DIAG / "tworoom-seed-sweep-ep4339-20260504-160559.json",
 "AMB_topK": DIAG / "tworoom-bridge-bridge-ep4339-20260504-143425.json",
}


def load_paired(path: Path) -> dict[int, int]:
 d = json.loads(path.read_text())
 out: dict[int, int] = {}
 for r in d["results"]:
 out[int(r["seed"])] = 1 if r.get("success") else 0
 return out


def wilson(k: int, n: int, z: float = 1.96):
 if n == 0:
 return (0.0, 0.0, 0.0)
 p = k / n
 denom = 1 + z * z / n
 center = (p + z * z / (2 * n)) / denom
 half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
 return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact_p(b: int, c: int) -> float:
 """Two-sided exact McNemar p-value over n_disc = b + c discordant pairs."""
 n = b + c
 if n == 0:
 return 1.0
 k = min(b, c)
 p = 0.0
 for i in range(0, k + 1):
 p += comb(n, i) * (0.5 ** n)
 p *= 2.0
 return min(1.0, p)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, B: int = 20000, alpha: float = 0.05, seed: int = 0):
 rng = np.random.default_rng(seed)
 n = len(a)
 diffs = np.empty(B)
 for i in range(B):
 idx = rng.integers(0, n, size=n)
 diffs[i] = a[idx].mean() - b[idx].mean()
 lo = float(np.quantile(diffs, alpha / 2))
 hi = float(np.quantile(diffs, 1 - alpha / 2))
 return float(np.mean(a - b)), lo, hi


def report_pair(name_a, name_b, A, B):
 seeds = sorted(set(A) & set(B))
 a = np.array([A[s] for s in seeds], dtype=int)
 b = np.array([B[s] for s in seeds], dtype=int)
 n = len(seeds)
 ka, kb = int(a.sum()), int(b.sum())
 pa, la, ha = wilson(ka, n)
 pb, lb, hb = wilson(kb, n)
 delta_mean, dlo, dhi = paired_bootstrap_ci(a, b, B=20000)
 # McNemar discordant counts
 bb = int(((a == 1) & (b == 0)).sum()) # A succeeds, B fails
 cc = int(((a == 0) & (b == 1)).sum()) # A fails, B succeeds
 pval = mcnemar_exact_p(bb, cc)

 print(f"\n== {name_a} vs {name_b} (n_paired={n}) ==")
 print(f" {name_a}: {ka}/{n}={100*pa:.1f}% Wilson [{100*la:.1f}, {100*ha:.1f}]")
 print(f" {name_b}: {kb}/{n}={100*pb:.1f}% Wilson [{100*lb:.1f}, {100*hb:.1f}]")
 print(f" paired Δ ({name_a} − {name_b}) mean = {100*delta_mean:+.1f} pp;"
 f" bootstrap 95% CI [{100*dlo:+.1f}, {100*dhi:+.1f}] pp")
 print(f" McNemar discordant: A>B={bb}, B>A={cc}; exact two-sided p = {pval:.2e}")

 return {
 "pair": f"{name_a}_vs_{name_b}",
 "n_paired": n,
 "marginal": {
 name_a: {"k": ka, "n": n, "rate": pa, "wilson_ci": [la, ha]},
 name_b: {"k": kb, "n": n, "rate": pb, "wilson_ci": [lb, hb]},
 },
 "paired_delta_pp": {
 "mean": 100 * delta_mean,
 "bootstrap_95_ci": [100 * dlo, 100 * dhi],
 "B": 20000,
 },
 "mcnemar": {
 "A_gt_B_disc": bb, "B_gt_A_disc": cc,
 "exact_two_sided_p": pval,
 },
 }


def main():
 cells = {k: load_paired(p) for k, p in CELLS.items()}
 out = []
 out.append(report_pair("AMB_topK", "A5_N300", cells["AMB_topK"], cells["A5_N300"]))
 out.append(report_pair("A5_N1000", "A5_N300", cells["A5_N1000"], cells["A5_N300"]))
 out.append(report_pair("A5_N1000", "AMB_topK", cells["A5_N1000"], cells["AMB_topK"]))

 out_path = DIAG / "tworoom-paired-ci-ep4339.json"
 out_path.write_text(json.dumps(out, indent=2))
 print(f"\nWritten to {out_path}")


if __name__ == "__main__":
 main()
