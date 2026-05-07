"""Vector heatmap of the four-task planner-perturbation grid (5-seed pool).

Source: phase1/two_room_planner_ablation/diagnostics/grid-path2-summary.json
 (5-seed canonical: tworoom/pusht/cube/reacher × seeds {0,1,2,3,42},
 all rows pooled k/250).
Output: paper/figures/fig2_four_task_sensitivity_heatmap.pdf

No embedded title or figure number; LaTeX caption typesets that.
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

SUMMARY = Path("phase1/two_room_planner_ablation/diagnostics/grid-path2-summary.json")
OUT_PDF = Path("paper/figures/fig2_four_task_sensitivity_heatmap.pdf")
OUT_META = Path("paper/figures/fig2_four_task_sensitivity_heatmap.metadata.json")

CELL_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A0_H10", "A2_H10"]
CELL_LABELS = ["A0", "A1", "A2", "A3", "A4", "A5", "A0_H10", "A2_H10"]

ROW_TASKS = [
 ("tworoom", "Two-Room (5-seed pool)"),
 ("pusht", "Push-T (5-seed pool)"),
 ("cube", "OGBench-Cube (5-seed pool)"),
 ("reacher", "Reacher (5-seed pool)"),
]


def load_grid():
 data = json.loads(SUMMARY.read_text())
 tasks_block = data["tasks"]

 rows = []
 rates = []
 annotations = []

 for task_key, row_label in ROW_TASKS:
 task = tasks_block[task_key]
 per_cell = task["pooled_per_cell"]
 rate_row = []
 annot_row = []
 for c in CELL_ORDER:
 cell = per_cell[c]
 rate_row.append(cell["pooled_rate_pct"])
 annot_row.append(f"{cell['pooled_k']}/{cell['pooled_n']}")
 rows.append(row_label)
 rates.append(rate_row)
 annotations.append(annot_row)

 return rows, np.array(rates, dtype=float), annotations, data


def find_two_largest_a0_drops(mat, rows):
 """Return list of (row_idx, col_idx, delta) for the two largest single-axis
 drops against A0 across the whole heatmap."""
 a0_idx = CELL_ORDER.index("A0")
 drops = []
 for i in range(mat.shape[0]):
 a0 = mat[i, a0_idx]
 for j in range(mat.shape[1]):
 if j == a0_idx:
 continue
 d = mat[i, j] - a0
 drops.append((i, j, d))
 drops.sort(key=lambda x: x[2]) # most negative first
 return drops[:2]


def main():
 rows, mat, annots, raw = load_grid()
 n_rows, n_cols = mat.shape

 fig, ax = plt.subplots(figsize=(6.6, 2.6))

 from matplotlib.colors import LinearSegmentedColormap
 cmap = LinearSegmentedColormap.from_list(
 "muted_rg",
 ["#b65f5f", "#e6c8b5", "#f1eee6", "#bcd2b8", "#4f8765"],
 N=256,
 )

 im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")

 ax.set_xticks(np.arange(n_cols))
 ax.set_xticklabels(CELL_LABELS, fontsize=8.5)
 ax.set_yticks(np.arange(n_rows))
 ax.set_yticklabels(rows, fontsize=8.5)
 ax.set_xlabel("Cell", fontsize=9)

 def luminance(rgb):
 return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

 for i in range(n_rows):
 for j in range(n_cols):
 v = mat[i, j]
 cell_rgba = cmap(v / 100.0)
 tcolor = "black" if luminance(cell_rgba[:3]) > 0.55 else "white"
 ax.text(j, i - 0.16, f"{v:.0f}", ha="center", va="center",
 color=tcolor, fontsize=8.5, weight="bold")
 ax.text(j, i + 0.22, annots[i][j], ha="center", va="center",
 color=tcolor, fontsize=6.5)

 # Highlight the two largest single-axis collapses with dark gray rectangles.
 largest = find_two_largest_a0_drops(mat, rows)
 highlight_meta = []
 for (i, j, d) in largest:
 ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
 fill=False, edgecolor="#1a1a1a", linewidth=1.6))
 highlight_meta.append({
 "row": rows[i],
 "cell": CELL_LABELS[j],
 "rate_pct": float(mat[i, j]),
 "delta_vs_A0_pp": float(round(d, 1)),
 })

 cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
 ticks=[0, 25, 50, 75, 100])
 cbar.set_label("Closed-loop success (%)", fontsize=8.5)
 cbar.ax.tick_params(labelsize=7.5)

 ax.tick_params(axis="x", which="both", bottom=False, top=False)
 ax.tick_params(axis="y", which="both", left=False, right=False)
 for spine in ax.spines.values():
 spine.set_visible(False)

 fig.tight_layout()
 OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
 print(f"Saved {OUT_PDF}")

 OUT_META.write_text(json.dumps({
 "source_summary": str(SUMMARY),
 "rows": rows,
 "cells": CELL_ORDER,
 "values_pct": mat.tolist(),
 "annotations_kn": annots,
 "highlight": {
 "outline_color": "#1a1a1a",
 "two_largest_a0_drops": highlight_meta,
 },
 "rendering": "matplotlib pdf, muted red->neutral->green palette (#b65f5f -> #f1eee6 -> #4f8765), vmin=0 vmax=100; pooled k/n=k/250 across 5 seeds for all rows",
 "generated_by": "fig2_four_task_sensitivity_heatmap.py",
 }, indent=2))
 print(f"Wrote {OUT_META}")


if __name__ == "__main__":
 main()
