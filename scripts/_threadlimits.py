"""Thread/CPU limits for numerical libraries.

Importing this module before torch/numpy/sklearn:
 1. Sets OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/TORCH env vars to 4 (via setdefault)
 so native libraries initialize with bounded thread pools.
 2. Imports torch and calls torch.set_num_threads + set_num_interop_threads,
 reading the count from TORCH_NUM_THREADS (default 4).

Override at launch time:
 OMP_NUM_THREADS=16 TORCH_NUM_THREADS=16 python <script>

**Override nuance:** OMP/MKL/OPENBLAS control native BLAS / OMP
parallelism (numpy, sklearn). PyTorch reads TORCH_NUM_THREADS for its own
intra-op pool — overriding only OMP_NUM_THREADS will *not* change PyTorch's
thread count. To raise both, set both env vars; or in future, a single
LEWM_CPU_THREADS convenience variable could fan out.

Env vars must be set BEFORE Python starts (or before this module is imported)
to affect already-loaded native libraries; importing this module is the
earliest hook our scripts use.
"""
from __future__ import annotations
import os as _os

_THREAD_VARS = (
 "OMP_NUM_THREADS",
 "MKL_NUM_THREADS",
 "OPENBLAS_NUM_THREADS",
 "NUMEXPR_NUM_THREADS",
 "VECLIB_MAXIMUM_THREADS",
 "TORCH_NUM_THREADS",
)
for _v in _THREAD_VARS:
 _os.environ.setdefault(_v, "4")

import torch as _torch # noqa: E402 # AFTER env-var setdefault on purpose

_n = int(_os.environ.get("TORCH_NUM_THREADS", "4"))
_torch.set_num_threads(_n)
try:
 _torch.set_num_interop_threads(1)
except RuntimeError:
 # Can only be called once per process and before any parallel work.
 # Re-imports (or imports after other torch use) hit this branch silently.
 pass
