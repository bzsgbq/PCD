# Planner-Calibrated Diagnostics for Latent World-Model Benchmarks

This repository contains the code, results, and manuscript for an anonymous double-blind submission. The work proposes a per-episode diagnostic protocol that decomposes a latent world model's closed-loop benchmark score into six interpretable strands (default cell, compute frontier, retrieval-only leakage, cost replacement, action-prior hook, expert replay) and assigns failures to four classes (calibration-only, two coverage-bound classes, and a planner-frontier-resistant class). It validates the protocol on the released LeWorldModel checkpoint family across four goal-reaching tasks (Two-Room, Reacher, Push-T, OGBench-Cube).

## Repository layout

```
.
├── README.md                       — this file
├── paper/
│   ├── submission/                 — NeurIPS 2026 LaTeX source
│   │   ├── main.tex
│   │   ├── checklist.tex
│   │   └── neurips_2026.sty / neurips_2026.tex
│   └── figures/                    — figure PDFs/PNGs + figure scripts
│       ├── fig1_planner_calibrated_diagnostics_overview.pdf
│       ├── fig2_four_task_sensitivity_heatmap.pdf
│       ├── fig3a_tworoom_mechanism.pdf
│       ├── fig3b_retrieval_leakage.pdf
│       ├── fig3c_pusht_hardcases.pdf
│       ├── fig5_taxonomy_showcase.png
│       ├── fig6_composite_score_hero.png
│       └── figure_scripts/         — Python scripts that regenerate Fig 2–3
├── scripts/                        — experiment runners
│   ├── run_ablation.py             — main per-cell evaluation driver
│   ├── aggregate_5seed_grid.py     — pools per-seed results into the source-of-record JSON
│   ├── run_grid_5seed_pathd.sh     — multi-GPU launcher for the 5-seed grid
│   ├── run_cube_hardcase_sweep.sh  — Cube hard-case paired solver-seed sweep
│   ├── paired_ci.py / paired_ci_pusht.py
│                                   — paired-bootstrap + McNemar statistics
│   ├── amb_retrieval_quality.py    — probe-decoded chunk-endpoint distances
│   ├── contact_cost_eval.py        — privileged Push-T contact-cost diagnostic
│   ├── train_q_psi.py / train_q_psi_v2.py
│                                   — learned reachability classifier (Q-cost)
│   ├── case_study.py               — per-episode trace driver
│   └── ...                         — supporting modules
└── diagnostics/                    — per-cell JSON results that the paper cites
    ├── grid-path2-summary.json     — canonical 5-seed pooled summary
    ├── tworoom-seed-sweep-ep4339-*.json
    ├── pusht-seed-sweep-ep{8314,14108}-*.json
    ├── cube-hardcase-sweep-summary.json
    └── ... (76 files total)
```

The repository contains:

- the experimental driver code that produced every number in the paper,
- the canonical per-cell result JSONs the paper's tables and figures point to,
- the LaTeX source of the manuscript and the matplotlib scripts that regenerate Figures 2 and 3.

It does **not** contain:

- the LeWorldModel codebase or pretrained checkpoints (third-party; see *Prerequisites* below),
- the offline pixel datasets (third-party; see *Prerequisites*),
- run logs (large; redacted to avoid path leakage).

## Prerequisites

The experiments here are reproductions / ablations of an upstream latent-world-model framework. Install the upstream stack first; the scripts in this repository wrap its evaluation entry-points.

1. **Latent world model stack.** A JEPA-style latent world-model framework with a Cross-Entropy / iCEM planner and a Hydra-based evaluation harness. The driver scripts assume a virtual environment that exposes the upstream `eval.py` and supporting modules under a Python package; we refer to its activation script as `${LEWM_VENV}` below. Follow the upstream project's installation instructions to reach a working `python eval.py task=...` command line.
2. **Offline pixel datasets** for the four tasks (Two-Room, Reacher, Push-T, OGBench-Cube). Download the released benchmark datasets distributed with the upstream framework and place them under a directory we refer to as `${LEWM_DATA}`. The shape used by the scripts here is the standard HDF5 layout with per-episode `pixels`, `actions`, `ep_offset`, `ep_len`.
3. **Released world-model checkpoints** for the four tasks. These are model weights distributed alongside the upstream framework; place them under `${LEWM_DATA}/checkpoints/<task>/` per the upstream layout.
4. **Python environment for the evaluation drivers**: Python ≥ 3.10, plus `numpy`, `pandas`, `matplotlib`, `scipy`, `torch`, `gymnasium`, `h5py`, `hdf5plugin`, `tabulate`. The upstream stack normally already includes these; if your environment differs, install the missing packages with pip.

The scripts use two shell variables exclusively for paths into the upstream stack:

```bash
export LEWM_VENV=/path/to/upstream/lewm/venv         # virtualenv root
export LEWM_DATA=/path/to/upstream/lewm/datasets     # datasets + checkpoints
```

Set both before running anything.

## Reproducing the four-task perturbation grid (Figure 2 / Table 7)

This is the headline 5-seed pooled grid. It runs four tasks × eight planner cells × five Hydra-coupled seeds = 160 cell-runs and produces the canonical `grid-path2-summary.json`.

```bash
# from the repository root
mkdir -p phase1/two_room_planner_ablation/{logs,results,diagnostics}

# kick off the multi-GPU launcher (allocates 10 worker slots across GPUs 0/2/3)
bash scripts/run_grid_5seed_pathd.sh

# wait for completion (~5 hours on 3 modern GPUs)

# aggregate per-seed results into the canonical pooled summary
python scripts/aggregate_5seed_grid.py
# writes: phase1/two_room_planner_ablation/diagnostics/grid-path2-summary.json
```

The launcher is configured for a 4-GPU host with one GPU reserved for other workloads (it skips GPU 1). To run on different hardware:

- adjust the `GPU_LIST` and `SLOTS_PER_GPU` variables at the top of `scripts/run_grid_5seed_pathd.sh`,
- shrink `num_eval` (default 50) inside `scripts/run_ablation.py` for a faster smoke test,
- run individual `(task, seed)` combos directly with:

```bash
python scripts/run_ablation.py \
    --task tworoom \
    --cells A0,A1,A2,A3,A4,A5,A0_H10,A2_H10 \
    --num_eval 50 \
    --seed 0 \
    --solver_seed 0 \
    --solver_batch_size 1
```

## Reproducing the TwoRoom 4339 mechanism (§3.2 + Figure 3a + Table 2)

```bash
# 100-seed paired sweep at the published budget and at N=1000
bash scripts/run_grid_parallel.sh tworoom 4339

# AMB-topK no-same-episode (offline-action-chunk warm start)
python scripts/run_ablation.py \
    --task tworoom \
    --episode 4339 \
    --cells A5_AMB_topK_no_same_ep \
    --num_eval 100 \
    --seed 0 --solver_seed 0

# paired-difference statistics (paired bootstrap + McNemar exact)
python scripts/paired_ci.py
```

Outputs land in `phase1/two_room_planner_ablation/diagnostics/` as `tworoom-seed-sweep-ep4339-*.json`, `tworoom-bridge-bridge-ep4339-*.json`, etc., and feed the paired-stats table.

## Reproducing the Push-T cross-task transfer (§3.3 + Figure 3c + Table 2)

```bash
# 20-seed paired sweep on PushT 8314 / 14108 (vanilla N=300, N=1000, AMB-topK)
bash scripts/run_pusht_stabilizer.sh

# privileged contact-cost diagnostic on PushT 14108
python scripts/contact_cost_eval.py --episode 14108

# paired-difference statistics
python scripts/paired_ci_pusht.py
```

## Reproducing the Cube hard-case sweep (Appendix E + Table 8)

```bash
# 3 episodes × 4 cells × 5 paired solver seeds; 0/60 expected on all
bash scripts/run_cube_hardcase_sweep.sh

# aggregate the per-cell JSONs into cube-hardcase-sweep-summary.json
python scripts/aggregate_cube_hardcase.py
```

## Reproducing the cost-replacement diagnostic (§3.4 + Table 4)

```bash
# train Q-cost (learned reachability classifier) on TwoRoom
python scripts/train_q_psi_v2.py --task tworoom

# evaluate Q-cost / probe-xy / privileged contact cells
python scripts/run_ablation.py \
    --task tworoom --episode 4339 \
    --cells A5_Q_v2_cost,A5_probe_xy_cost \
    --num_eval 10 --seed 0 --solver_seed 0
```

## Reproducing the retrieval-only leakage diagnostic (§3.5 + Figure 3b)

```bash
# best-of-K=10 retrieved chunks, executed open-loop in the simulator
python scripts/run_ablation.py \
    --task tworoom --episode 4339 \
    --cells RETRIEVE_only_K10_no_same_ep,RETRIEVE_only_K10_same_ep \
    --num_eval 10 --seed 0 --solver_seed 0

# probe-decoded chunk-endpoint distance audit
python scripts/amb_retrieval_quality.py --task tworoom --episode 4339
```

## Regenerating the paper figures

```bash
# Figures 2, 3a, 3b, 3c are auto-regenerated from the canonical JSONs
python paper/figures/figure_scripts/fig2_four_task_sensitivity_heatmap.py
python paper/figures/figure_scripts/fig3a_tworoom_mechanism.py
python paper/figures/figure_scripts/fig3b_retrieval_leakage.py
python paper/figures/figure_scripts/fig3c_pusht_hardcases.py
```

Figures 1, 5, and 6 are manually-prepared illustrations included as PDFs / PNGs under `paper/figures/`; their source compositions are documented in their respective figure scripts (Fig 5 / Fig 6 are matplotlib placeholders that can be regenerated with `python paper/figures/figure_scripts/fig5_taxonomy_showcase.py` and `fig6_composite_score_hero.py`).

## Building the manuscript

```bash
cd paper/submission
pdflatex main.tex
pdflatex main.tex   # second pass for cross-references
```

The build expects a working TeX Live distribution with the `natbib`, `booktabs`, `tabularx`, `longtable`, and `microtype` packages.

## Source-of-record manifest

Every numerical claim in the paper points to a specific JSON under `diagnostics/`. Appendix I of the manuscript ("Source-of-record manifest") tabulates the cell → JSON mapping for every cited cell, with $k/n$, Wilson 95% CIs, and the diagnostic-file name. The aggregator (`scripts/aggregate_5seed_grid.py`) walks the per-seed result directories and produces the canonical pooled summary the paper's tables read from.

## Reproducibility notes

- All success-rate comparisons across cells use **paired** evaluation: the same `(episode, seed)` pair is shared between cells, so paired-bootstrap intervals and McNemar's exact test apply.
- The evaluator's CEM / iCEM sample stream depends on the solver seed; the released benchmark batches multiple episodes per environment process and shares a single stream across the batch, which can mask per-episode seed sensitivity. Our runners disable batching (`solver_batch_size=1`) and keep the per-episode CEM sample stream independent.
- Cost-replacement experiments train cost classifiers on encoded latents but score them on predictor-rollout latents during closed-loop control; this distribution mismatch is the cost-calibration warning surfaced in §3.3.

## Ethics and licensing

This repository contains only original code authored for the submission, the original LaTeX manuscript, and per-cell numerical results derived by running the scripts on the released upstream benchmark. The upstream world-model framework, datasets, and pretrained checkpoints are subject to their own licences and are not redistributed here.
