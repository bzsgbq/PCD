"""Retrieval-only leakage standalone figure (paired horizontal bars).

Cases:
- TwoRoom 4339: same-episode allowed 8/10 vs no-same-episode 5/10 (30 pp delta).
- PushT 14108: same-episode n/r; no-same-episode 0/10.

Output: paper/figures/fig3b_retrieval_leakage.pdf
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

DIAG = Path("phase1/two_room_planner_ablation/diagnostics")
OUT_PDF = Path("paper/figures/fig3b_retrieval_leakage.pdf")
OUT_META = Path("paper/figures/fig3b_retrieval_leakage.metadata.json")


def kn_retrieval(path):
 d = json.loads(Path(path).read_text())
 return d.get("n_success"), len(d["results"])


# load
tr_with = kn_retrieval(DIAG / "tworoom-retrieval-only-bridge-ep4339-20260503-224837.json") # 8/10
tr_no = kn_retrieval(DIAG / "tworoom-retrieval-only-bridge-ep4339-20260503-233710.json") # 5/10
pt_no = kn_retrieval(DIAG / "pusht-retrieval-only-bridge-ep14108-20260504-012600.json") # 0/10

cases = [
 ("TwoRoom 4339", tr_with, tr_no), # same-ep present
 ("PushT 14108", None, pt_no), # n/r same-ep
]

# horizontal paired bars
fig, ax = plt.subplots(figsize=(6.6, 2.4))
bar_h = 0.34
y = np.arange(len(cases))[::-1]

color_with = "#a07d2c"
color_no = "#5f4a1a"

for yi, (label, with_lk, no_lk) in zip(y, cases):
 if with_lk is not None:
 kk, nn = with_lk
 rate = 100.0 * kk / nn
 ax.barh(yi + bar_h / 2, rate, height=bar_h, color=color_with,
 edgecolor="white", label="same-ep allowed" if yi == y[0] else None)
 ax.text(rate + 1.5, yi + bar_h / 2, f"{kk}/{nn}", va="center", fontsize=9, color="#222")
 else:
 ax.text(1.5, yi + bar_h / 2, "n/r (same-ep not run)",
 va="center", fontsize=8.5, color="#888", style="italic")

 kk, nn = no_lk
 rate = 100.0 * kk / nn
 ax.barh(yi - bar_h / 2, rate, height=bar_h, color=color_no,
 edgecolor="white", label="no-same-ep" if yi == y[0] else None)
 ax.text(max(rate + 1.5, 3), yi - bar_h / 2, f"{kk}/{nn}", va="center", fontsize=9, color="#222")

# annotate the 30 pp leakage on TwoRoom row
yi_tr = y[0]
ax.annotate("", xy=(50, yi_tr - bar_h / 2 + 0.03), xytext=(80, yi_tr + bar_h / 2 - 0.03),
 arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
ax.text(65, yi_tr, "30 pp same-ep\nleakage",
 ha="center", va="center", fontsize=9, color="#222",
 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#333", lw=0.5))

ax.set_yticks(y)
ax.set_yticklabels([c[0] for c in cases], fontsize=9.5)
ax.set_xlim(0, 105)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Retrieval-only best-of-K success (%)", fontsize=10)

# legend
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=color_with, label="same-episode allowed"),
 mpatches.Patch(color=color_no, label="no-same-episode")]
ax.legend(handles=handles, fontsize=9, loc="lower right", frameon=False,
 handletextpad=0.5)

ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
 ax.spines[s].set_visible(False)

fig.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
print(f"Saved {OUT_PDF}")

meta = {
 "cases": [
 {"slice": "TwoRoom 4339",
 "same_ep_allowed_kn": tr_with,
 "no_same_ep_kn": tr_no,
 "delta_pp": 30},
 {"slice": "PushT 14108",
 "same_ep_allowed_kn": None,
 "no_same_ep_kn": pt_no,
 "note": "same-episode variant not run"},
 ],
 "rendering": "matplotlib pdf vector, fonttype 42, native ~6.6in wide, no embedded title",
 "generated_by": "fig3b_retrieval_leakage.py",
}
OUT_META.write_text(json.dumps(meta, indent=2))
print(f"Wrote {OUT_META}")
