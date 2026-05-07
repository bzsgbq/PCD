"""TwoRoom 4339 mechanism: standalone forest plot.

Rows (top -> bottom): vanilla A5 N=300, vanilla A5 N=1000, AMB-topK no-same-ep,
retrieval-only no-same-ep, Q v1-cost, Q v2-cost, probe-xy.

Output: paper/figures/fig3a_tworoom_mechanism.pdf
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
import matplotlib.lines as mlines

DIAG = Path("phase1/two_room_planner_ablation/diagnostics")
OUT_PDF = Path("paper/figures/fig3a_tworoom_mechanism.pdf")
OUT_META = Path("paper/figures/fig3a_tworoom_mechanism.metadata.json")


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


def kn_retrieval(path):
 d = json.loads(Path(path).read_text())
 return d.get("n_success"), len(d["results"])


ROWS = [
 ("vanilla A5 N=300", *kn_seed_sweep(DIAG / "tworoom-seed-sweep-ep4339-20260504-142931.json"), "compute"),
 ("vanilla A5 N=1000", *kn_seed_sweep(DIAG / "tworoom-seed-sweep-ep4339-20260504-160559.json"), "compute"),
 ("AMB-topK no-same-ep", *kn_bridge(DIAG / "tworoom-bridge-bridge-ep4339-20260504-143425.json"), "action_prior"),
 ("retrieval-only no-same-ep", *kn_retrieval(DIAG / "tworoom-retrieval-only-bridge-ep4339-20260503-233710.json"), "retrieval"),
 ("Q v1-cost", *kn_seed_sweep(DIAG / "tworoom-q-cost-cumulative-ep4339-20260503-153330.json"), "cost"),
 ("Q v2-cost", *kn_seed_sweep(DIAG / "tworoom-q-cost-cumulative-ep4339-20260503-185451.json"), "cost"),
 ("probe-xy cost", *kn_seed_sweep(DIAG / "tworoom-probe-cumulative-ep4339-20260503-155032.json"), "cost"),
]

GROUP_COLOR = {
 "compute": "#3a7bd5",
 "action_prior": "#2ca06b",
 "retrieval": "#a07d2c",
 "cost": "#d6442d",
}
GROUP_LABEL = {
 "compute": "Compute frontier",
 "action_prior": "Action-prior hook",
 "retrieval": "Retrieval-only",
 "cost": "Cost replacement",
}

fig, ax = plt.subplots(figsize=(6.6, 2.8))
y = np.arange(len(ROWS))[::-1] # top -> bottom

for yi, (label, k, n, group) in zip(y, ROWS):
 p, lo, hi = wilson(k, n)
 rate = 100 * p
 err_lo = max(0.0, 100 * (p - lo))
 err_hi = max(0.0, 100 * (hi - p))
 color = GROUP_COLOR[group]
 ax.errorbar(rate, yi, xerr=[[err_lo], [err_hi]], fmt="o",
 color=color, ecolor=color, elinewidth=1.4,
 markersize=6, capsize=3.0)
 ax.text(min(rate + max(err_hi + 2.5, 4), 102), yi, f" {k}/{n}",
 va="center", fontsize=9, color="#222")

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=9.5)
ax.set_xlim(-3, 113)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Closed-loop success (%)", fontsize=10)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
 ax.spines[s].set_visible(False)

# Legend by group, ordered top -> bottom
legend_handles = [mlines.Line2D([], [], marker="o", color=GROUP_COLOR[g],
 linestyle="None", markersize=6, label=GROUP_LABEL[g])
 for g in ["compute", "action_prior", "retrieval", "cost"]]
ax.legend(handles=legend_handles, fontsize=8.0, loc="lower right",
 frameon=False, handletextpad=0.4, labelspacing=0.3)

fig.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
print(f"Saved {OUT_PDF}")

OUT_META.write_text(json.dumps({
 "rows": [{"label": r[0], "k": r[1], "n": r[2], "group": r[3]} for r in ROWS],
 "groups": GROUP_LABEL,
 "rendering": "matplotlib pdf vector, fonttype 42, native ~6.6in wide, no embedded title",
 "generated_by": "fig3a_tworoom_mechanism.py",
}, indent=2))
print(f"Wrote {OUT_META}")
