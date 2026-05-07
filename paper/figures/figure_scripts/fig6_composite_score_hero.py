"""Fig 6: hero / opener visualizing the composite-score problem.

A single closed-loop number conflates five quality factors and produces
strikingly different answers across tasks. The figure shows: five labeled
factor dials feeding a single composite ``score'' node, which fans out to
four task panels whose pooled-cell ranges visibly disagree.

This is a placeholder schematic; the user can swap in an image-generation-
model render with the same conceptual layout (see
`paper/figures/figure_scripts/fig6_image_generation_prompt.md`).

Output: paper/figures/fig6_composite_score_hero.pdf
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_PDF = Path("paper/figures/fig6_composite_score_hero.pdf")

FACTOR_COLOR = {
 "E": "#6a9a9c", # teal -- encoder sufficiency
 "P": "#6e6aa4", # purple -- predictor rollout fidelity
 "C": "#b08a4a", # ochre -- cost/ranking calibration
 "S": "#c98787", # rose -- candidate-pool coverage
 "R": "#7e8a99", # slate -- dataset-replay channel
}

CLASS_COLOR = {
 "0": "#4f8765",
 "1a": "#3a7bd5",
 "1b": "#d4a23a",
 "2": "#b65f5f",
}


def draw_factor_dial(ax, x, y, w, h, key, label):
 color = FACTOR_COLOR[key]
 box = FancyBboxPatch(
 (x, y), w, h,
 boxstyle="round,pad=0.30,rounding_size=0.35",
 linewidth=1.4, facecolor="white", edgecolor=color,
 )
 ax.add_patch(box)
 # Letter chip
 chip_r = 0.30
 chip_x = x + 0.55
 chip_y = y + h / 2.0
 ax.add_patch(mpatches.Circle((chip_x, chip_y), chip_r,
 facecolor=color, edgecolor="none", zorder=3))
 ax.text(chip_x, chip_y, key, ha="center", va="center",
 color="white", fontsize=11.5, fontweight="bold", zorder=4)
 # Label
 ax.text(chip_x + 0.55, chip_y, label, ha="left", va="center",
 color="#1f2a44", fontsize=9.0)


def draw_task_card(ax, x, y, w, h, name, low, high, low_label, high_label,
 cls_key):
 color = CLASS_COLOR[cls_key]
 box = FancyBboxPatch(
 (x, y), w, h,
 boxstyle="round,pad=0.20,rounding_size=0.35",
 linewidth=1.6, facecolor="white", edgecolor=color,
 )
 ax.add_patch(box)
 # Task name (top of card)
 ax.text(x + w / 2, y + h - 0.30, name, ha="center", va="center",
 color="#0b1745", fontsize=11, fontweight="bold")
 # Bar geometry
 bar_left = x + 0.55
 bar_right = x + w - 0.55
 full_w = bar_right - bar_left
 bar_y = y + 0.45
 bar_h = 0.30
 # Background bar
 ax.add_patch(mpatches.Rectangle(
 (bar_left, bar_y - bar_h / 2), full_w, bar_h,
 facecolor="#eef0f3", edgecolor="none", zorder=2,
 ))
 # Range fill
 ax.add_patch(mpatches.Rectangle(
 (bar_left + low / 100.0 * full_w, bar_y - bar_h / 2),
 (high - low) / 100.0 * full_w, bar_h,
 facecolor=color, edgecolor="none", zorder=3, alpha=0.85,
 ))
 # Endpoint labels above the bar (left + right)
 label_y = bar_y + 0.55
 ax.text(bar_left + 0.05, label_y, f"{low:.1f}% {low_label}",
 ha="left", va="center", color="#5a5f6a", fontsize=7.8)
 ax.text(bar_right - 0.05, label_y, f"{high_label} {high:.1f}%",
 ha="right", va="center", color=color, fontsize=7.8,
 fontweight="bold")
 # Spread chip at bar centre
 ax.text(x + w / 2, bar_y, f"$\\Delta$ {high - low:.1f} pp",
 ha="center", va="center", color="#1f2a44",
 fontsize=8.5, fontweight="bold", zorder=4)


def main():
 fig, ax = plt.subplots(figsize=(11.0, 5.0))
 ax.set_xlim(0, 22); ax.set_ylim(0, 10); ax.set_axis_off()

 # --- Title strip
 ax.text(11.0, 9.55, "One benchmark score, five quality factors, four very different answers",
 ha="center", va="center", color="#0b1745",
 fontsize=14, fontweight="bold")
 ax.text(11.0, 8.95, "A single closed-loop success rate conflates encoder, predictor, cost, coverage, and retrieval.",
 ha="center", va="center", color="#5a5f6a", fontsize=9.5, style="italic")

 # --- Factor dials (left column)
 factors = [
 ("E", "Encoder sufficiency"),
 ("P", "Predictor rollout fidelity"),
 ("C", "Cost / ranking calibration"),
 ("S", "Candidate-pool coverage"),
 ("R", "Dataset-replay channel"),
 ]
 DIAL_W, DIAL_H = 5.8, 1.05
 DIAL_X = 0.5
 DIAL_Y0 = 7.3
 DIAL_DY = 1.25
 for i, (k, lbl) in enumerate(factors):
 y = DIAL_Y0 - i * DIAL_DY
 draw_factor_dial(ax, DIAL_X, y, DIAL_W, DIAL_H, k, lbl)

 # --- Center "single score" node
 SCORE_CX, SCORE_CY = 10.5, 4.85
 SCORE_R = 1.45
 ax.add_patch(mpatches.Circle((SCORE_CX, SCORE_CY), SCORE_R,
 facecolor="#0b1745", edgecolor="#0b1745", zorder=3))
 ax.text(SCORE_CX, SCORE_CY + 0.30, "one", ha="center", va="center",
 color="white", fontsize=10, style="italic")
 ax.text(SCORE_CX, SCORE_CY - 0.30, "score", ha="center", va="center",
 color="white", fontsize=18, fontweight="bold")

 # Arrows from dials to score node
 for i in range(5):
 y = DIAL_Y0 - i * DIAL_DY + DIAL_H / 2
 ax.add_patch(FancyArrowPatch(
 (DIAL_X + DIAL_W + 0.15, y),
 (SCORE_CX - SCORE_R, SCORE_CY),
 arrowstyle="-|>", mutation_scale=10,
 color="#5a5f6a", linewidth=0.9, alpha=0.5, zorder=2,
 ))

 # --- Task cards (right column) with their pooled-cell ranges
 tasks = [
 # name, low, low_label, high, high_label, class_key
 ("Reacher", 22.0, "A0$_{H10}$", 100.0, "A2$_{H10}$", "0"),
 ("Two-Room", 42.4, "A0$_{H10}$", 93.6, "A5", "1a"),
 ("Push-T", 8.4, "A0$_{H10}$", 92.8, "A0", "1b"),
 ("OGBench-Cube", 53.2, "A0$_{H10}$", 73.2, "A5", "2"),
 ]
 CARD_W, CARD_H = 5.8, 1.65
 CARD_X = 15.7
 CARD_Y0 = 7.0
 CARD_DY = 1.78
 for i, t in enumerate(tasks):
 y = CARD_Y0 - i * CARD_DY
 name, low, low_lbl, high, high_lbl, cls_key = t
 draw_task_card(ax, CARD_X, y, CARD_W, CARD_H, name,
 low, high, low_lbl, high_lbl, cls_key)

 # Arrows from score node to task cards
 for i in range(4):
 y = CARD_Y0 - i * CARD_DY + CARD_H / 2
 ax.add_patch(FancyArrowPatch(
 (SCORE_CX + SCORE_R, SCORE_CY),
 (CARD_X - 0.15, y),
 arrowstyle="-|>", mutation_scale=10,
 color="#5a5f6a", linewidth=0.9, alpha=0.6, zorder=2,
 ))

 # Footer
 ax.text(11.0, 0.45,
 "Same released world model, same evaluation episodes — different answers under different planner configurations.",
 ha="center", va="center", color="#5a5f6a",
 fontsize=9.0, style="italic")

 OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.05)
 print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
 main()
