"""Fig 5: four-class taxonomy showcase using real task frames.

Each panel shows one canonical task case with its class-color border, a
class-number badge, real init+goal composite frames, and the cell signature
that anchors the class assignment. Reads pre-extracted composites from
`paper/_archive_fig1_iterations/fig1_case_images/`.

Output: paper/figures/fig5_taxonomy_showcase.pdf
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
from matplotlib.image import imread


CASE_DIR = Path("paper/_archive_fig1_iterations/fig1_case_images")
OUT_PDF = Path("paper/figures/fig5_taxonomy_showcase.pdf")

CLASS_COLORS = {
 "0": "#4f8765", # sage green
 "1a": "#3a7bd5", # slate blue
 "1b": "#d4a23a", # warm amber
 "2": "#b65f5f", # dusty red
}

PANELS = [
 { # top-left
 "cls": "0",
 "image": "reacher_task.png",
 "title": "Reacher",
 "subtitle": "Calibration-only",
 "cells": "A0: 84.8\\% | A2: 99.6\\% | A2$_{H10}$: 100\\%",
 "caption": "Saturates from cumulative-cost cell onwards across\nfive Hydra-coupled seeds; no head-room for further intervention.",
 },
 { # top-right
 "cls": "1a",
 "image": "tworoom_4339.png",
 "title": "Two-Room 4339",
 "subtitle": "Coverage-bound, action-prior amenable",
 "cells": "Vanilla A5 $N{=}300$: 63\\% | AMB-topK: $+22$ pp | $N{=}1000$: $+37$ pp",
 "caption": "Both stronger compute and action-prior intervention\nclose the gap; either signal alone shifts the class.",
 },
 { # bottom-left
 "cls": "1b",
 "image": "pusht_8314.png",
 "title": "Push-T 8314",
 "subtitle": "Coverage-bound, action-prior resistant",
 "cells": "Vanilla $N{=}300$: 36\\% | $N{=}1000$: $+60$ pp | AMB-topK: $\\Delta\\!\\approx\\!0$",
 "caption": "Compute helps strongly; tested action-prior\ninterventions are null at the published budget.",
 },
 { # bottom-right
 "cls": "2",
 "image": "pusht_14108.png",
 "title": "Push-T 14108",
 "subtitle": "Narrow-funnel / planner-frontier",
 "cells": "Vanilla $N{=}300$: 0/5 | AMB cells: 0--1/5 | $N{=}1000$: $+0$ pp",
 "caption": "No tested planner-side intervention reliably\nsolves the case at the published budget.",
 },
]


def panel(ax, p):
 color = CLASS_COLORS[p["cls"]]
 # Outer border (class color)
 for spine in ax.spines.values():
 spine.set_edgecolor(color)
 spine.set_linewidth(2.4)
 ax.set_xticks([]); ax.set_yticks([])
 ax.set_facecolor("#fbfcfd")

 # Layout fractions inside the panel (axes coords)
 BADGE_R = 0.058
 BADGE_CX, BADGE_CY = 0.07, 0.92
 IMG_LEFT, IMG_RIGHT = 0.075, 0.925
 IMG_TOP, IMG_BOTTOM = 0.86, 0.46

 # Class number badge (top-left of panel)
 badge = mpatches.Circle(
 (BADGE_CX, BADGE_CY), BADGE_R,
 transform=ax.transAxes, zorder=4,
 facecolor=color, edgecolor="white", linewidth=1.6,
 )
 ax.add_patch(badge)
 ax.text(BADGE_CX, BADGE_CY, p["cls"], transform=ax.transAxes,
 ha="center", va="center", color="white",
 fontsize=12, fontweight="bold", zorder=5)

 # Init+goal composite image
 img = imread(CASE_DIR / p["image"])
 ax.imshow(
 img,
 extent=(IMG_LEFT, IMG_RIGHT, IMG_BOTTOM, IMG_TOP),
 transform=ax.transAxes, zorder=2, aspect="auto",
 interpolation="lanczos",
 )
 # Thin frame around image
 ax.add_patch(mpatches.Rectangle(
 (IMG_LEFT, IMG_BOTTOM),
 IMG_RIGHT - IMG_LEFT, IMG_TOP - IMG_BOTTOM,
 transform=ax.transAxes, fill=False,
 edgecolor="#cfd3d8", linewidth=0.8, zorder=3,
 ))
 # init / goal subcaption
 ax.text(0.25, IMG_BOTTOM - 0.025, "init", transform=ax.transAxes,
 ha="center", va="top", color="#5a5f6a", fontsize=8.5, style="italic")
 ax.text(0.75, IMG_BOTTOM - 0.025, "goal", transform=ax.transAxes,
 ha="center", va="top", color="#5a5f6a", fontsize=8.5, style="italic")

 # Title
 ax.text(0.5, 0.36, p["title"], transform=ax.transAxes,
 ha="center", va="center", color="#0b1745",
 fontsize=12.5, fontweight="bold")
 # Subtitle
 ax.text(0.5, 0.295, p["subtitle"], transform=ax.transAxes,
 ha="center", va="center", color=color,
 fontsize=10.5, fontweight="bold")
 # Cells
 ax.text(0.5, 0.21, p["cells"], transform=ax.transAxes,
 ha="center", va="center", color="#1f2a44",
 fontsize=9.0, family="monospace")
 # Caption
 ax.text(0.5, 0.10, p["caption"], transform=ax.transAxes,
 ha="center", va="center", color="#5a5f6a",
 fontsize=8.8, style="italic")


def main():
 fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
 for ax, p in zip(axes.flat, PANELS):
 panel(ax, p)
 plt.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.012,
 wspace=0.05, hspace=0.07)
 OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.05)
 print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
 main()
