"""PushT hard cases standalone grouped-bar chart.

X-axis groups: PushT 8314 / PushT 14108.
Bars per group: vanilla N=300, vanilla N=1000, AMB-topK no-same-ep, AMB-random no-same-ep.

Output: paper/figures/fig3c_pusht_hardcases.pdf
"""
from __future__ import annotations
import json
from math import sqrt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

DIAG = Path("phase1/two_room_planner_ablation/diagnostics")
OUT_PDF = Path("paper/figures/fig3c_pusht_hardcases.pdf")
OUT_META = Path("paper/figures/fig3c_pusht_hardcases.metadata.json")


def wilson(k: int, n: int, z: float = 1.96):
 if n == 0:
 return (0.0, 0.0, 0.0)
 p = k / n
 denom = 1 + z * z / n
 center = (p + z * z / (2 * n)) / denom
 half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
 return p, max(0.0, center - half), min(1.0, center + half)


def kn_seed_sweep(path):
 d = json.loads(Path(path).read_text())
 return d.get("success_count", sum(1 for r in d["results"] if r["success"])), len(d["results"])


def kn_bridge(path):
 d = json.loads(Path(path).read_text())
 return sum(1 for r in d["results"] if r.get("success", False)), len(d["results"])


# Use the canonical stabilizer summary numbers (paper-frozen).
ps = json.loads((DIAG / "pusht-stabilizer-summary.json").read_text())

def from_stab(slice_id, key):
 return ps[slice_id][key]["k"], ps[slice_id][key]["n"]


groups = [
 ("PushT 8314", {
 "vanilla N=300": from_stab("8314", "vanilla N=300"),
 "vanilla N=1000": from_stab("8314", "vanilla N=1000"),
 "AMB-topK no-same-ep": from_stab("8314", "AMB-topK no-same-ep"),
 "AMB-random no-same-ep": kn_bridge(DIAG / "pusht-bridge-random-ep8314-20260504-230923.json"),
 }),
 ("PushT 14108", {
 "vanilla N=300": from_stab("14108", "vanilla N=300"),
 "vanilla N=1000": from_stab("14108", "vanilla N=1000"),
 "AMB-topK no-same-ep": from_stab("14108", "AMB-topK no-same-ep"),
 "AMB-random no-same-ep": kn_bridge(DIAG / "pusht-bridge-random-ep14108-20260504-012016.json"),
 }),
]

primitive_order = ["vanilla N=300", "vanilla N=1000",
 "AMB-topK no-same-ep", "AMB-random no-same-ep"]
primitive_color = {
 "vanilla N=300": "#7fb3e6",
 "vanilla N=1000": "#1f4f99",
 "AMB-topK no-same-ep": "#2ca06b",
 "AMB-random no-same-ep": "#9bd0b6",
}

fig, ax = plt.subplots(figsize=(6.6, 2.8))
n_prim = len(primitive_order)
bar_w = 0.18
group_centers = np.arange(len(groups))

for gi, (gname, prims) in enumerate(groups):
 for pi, pname in enumerate(primitive_order):
 k, n = prims[pname]
 p, lo, hi = wilson(k, n)
 rate = 100 * p
 err_lo = max(0.0, 100 * (p - lo))
 err_hi = max(0.0, 100 * (hi - p))
 x = gi + (pi - (n_prim - 1) / 2) * bar_w
 ax.bar(x, rate, width=bar_w * 0.95, color=primitive_color[pname],
 edgecolor="white", label=pname if gi == 0 else None)
 ax.errorbar(x, rate, yerr=[[err_lo], [err_hi]], fmt="none",
 ecolor="#333", elinewidth=0.9, capsize=2.0)
 ax.text(x, rate + max(err_hi, 2) + 1.5, f"{k}/{n}",
 ha="center", va="bottom", fontsize=7.5, color="#222")

ax.set_xticks(group_centers)
ax.set_xticklabels([g[0] for g in groups], fontsize=10)
ax.set_ylim(0, 130)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Closed-loop success (%)", fontsize=10)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
 ax.spines[s].set_visible(False)

ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False,
 ncol=4, handletextpad=0.4, labelspacing=0.3, columnspacing=1.2)

fig.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
print(f"Saved {OUT_PDF}")

meta = {
 "groups": [
 {"slice": g[0],
 "primitives": {p: {"k": v[0], "n": v[1]} for p, v in g[1].items()}}
 for g in groups
 ],
 "rendering": "matplotlib pdf vector, fonttype 42, native ~6.6in wide, no embedded title",
 "generated_by": "fig3c_pusht_hardcases.py",
}
OUT_META.write_text(json.dumps(meta, indent=2))
print(f"Wrote {OUT_META}")
