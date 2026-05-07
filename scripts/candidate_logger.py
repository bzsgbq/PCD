"""Observational candidate-logging wrapper around a JEPA-cost-model.

We don't subclass ICEMSolver — instead we wrap the model's `get_cost` to
side-effect-log every (candidates, costs) pair the solver evaluates. The
solver itself runs unchanged (same RNG, same algorithm, same output).

regression check: the wrapper-driven solve must select the same
action as the unwrapped solver under the same seed/config.

Used as a helper module by other scripts that already import _threadlimits;
we add a defensive import here so direct invocation still gets thread limits.
"""
from __future__ import annotations
import sys
from pathlib import Path
if str(Path(__file__).parent) not in sys.path:
 sys.path.insert(0, str(Path(__file__).parent))
import _threadlimits # noqa: F401 # CPU thread limits, BEFORE torch import
import torch


class CandidateLogger:
 """Observational logger. Hold a reference to a model with `get_cost`,
 intercept calls and stash the (candidate, cost) pairs in `self.history`.

 Usage:
 logger = CandidateLogger(model)
 logger.activate() # patches model.get_cost
.. run planner that calls model.get_cost..
 logger.deactivate()
 history = logger.history # list of dicts {candidates, costs, t}
 """

 def __init__(self, model):
 self.model = model
 self._orig_get_cost = None
 self.history = []

 def activate(self):
 if self._orig_get_cost is not None:
 return
 orig = self.model.get_cost
 log = self.history

 def hooked(info_dict, action_candidates):
 cost = orig(info_dict, action_candidates)
 log.append({
 "candidates": action_candidates.detach().cpu(),
 "costs": cost.detach().cpu(),
 })
 return cost

 self._orig_get_cost = orig
 self.model.get_cost = hooked
 return self

 def deactivate(self):
 if self._orig_get_cost is not None:
 self.model.get_cost = self._orig_get_cost
 self._orig_get_cost = None
 return self

 def reset(self):
 self.history = []

 def __enter__(self):
 self.activate()
 return self

 def __exit__(self, exc_type, exc_val, exc_tb):
 self.deactivate()
