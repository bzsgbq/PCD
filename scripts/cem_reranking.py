"""CEM reranking calibration.

For each chosen (episode_idx, plan_step), reconstruct the simulator state, then:
 1. Sample N=300 action-sequence candidates with the *same* CEM seed and
 proposal distribution that the live planner used at that step.
 2. Score each candidate by:
 (a) imagined cost = JEPA terminal latent distance to goal embedding
 (b) true cost = sim-replayed terminal observation, encoded, distance
 to goal embedding
 3. Report Spearman rho(imagined, true) + top-k accuracy
 (top-1 / top-5 imagined falling in true top-10%).

NOTE: this is a sketch — env per-candidate reset and parallel rollout require
careful integration with `swm.World`. Will be filled in next session.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_LEWM_DIR = Path(__file__).resolve().parents[3] / "le-wm"
if _LEWM_DIR.exists() and str(_LEWM_DIR) not in sys.path:
 sys.path.insert(0, str(_LEWM_DIR))


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--task", default="tworoom")
 ap.add_argument("--eval_set", required=True)
 ap.add_argument("--num_candidates", type=int, default=300)
 args = ap.parse_args()

 raise NotImplementedError(
 "Implementation pending — needs (1) per-candidate env reset to a saved "
 "simulator state, (2) parallel rollout under candidate action sequences, "
 "(3) re-encoding of the resulting terminal observation. Will be implemented "
 "next session."
 )


if __name__ == "__main__":
 main()
