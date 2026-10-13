"""Compatibility shims that let the optional predictors load under this torch build.

WHAT IS BROKEN, AND WHY THIS IS THE FIX

The GPU torch on this machine is a ROCm build compiled without distributed support:

    torch 2.12.0a0+rocm7.13.0a20260313
    hasattr(torch._C, "_c10d_init")  ->  False
    torch._C._has_distributed()      ->  False
    torch.distributed.is_available() ->  False
    torch.cuda.get_device_name(0)    ->  AMD Radeon(TM) 8060S Graphics   (GPU is fine)

Two optional predictors walk into that gap by accident, and neither of them wants
distributed for anything:

1. structRFM. `transformers.generation.utils` imports `continuous_batching`, whose
   `distributed.py` does `from torch.distributed.tensor.device_mesh import DeviceMesh`.
   That drags in `torch/distributed/tensor/__init__.py` -> `_ops` -> `_dtensor_spec` ->
   `placement_types` -> `_functional_collectives` -> `distributed_c10d` ->
   `from torch._C._distributed_c10d import (...)`, which does not exist. So
   `import transformers` succeeds and `from transformers import AutoModel` raises
   `ModuleNotFoundError: No module named 'torch._C._distributed_c10d'`.

2. RNAbpFlow. Its model is a `pytorch_lightning` module, and
   `pytorch_lightning.utilities.types` -> `torchmetrics` -> `torchmetrics.functional.
   text.bert` -> `from transformers import AutoModel` hits the same wall. So
   RNAbpFlow cannot even import its model class, let alone run it.

Both are incidental imports. `continuous_batching/distributed.py` uses `DeviceMesh` for
one annotation and for code guarded by `dist.is_available()`, which is False. Nothing in
either predictor performs a collective.

WHAT THIS MODULE DOES

`install()` registers two stand-ins, each only when the real thing is genuinely absent,
and each refusing to do anything rather than silently doing the wrong thing:

* `torch.distributed.tensor.device_mesh` with a `DeviceMesh` that raises on
  construction. Without distributed there is no mesh to build, so any code that reaches
  construction is on a path it cannot legitimately be on.

* `torch_scatter`, built on `Tensor.index_add_`. No Windows wheel exists for this torch
  and building the extension needs MSVC, but RNAbpFlow uses exactly two calls:

      utils.py:392  scatter(pos[N,3], batch_indexes[N], dim=0, reduce="mean")
      utils.py:451  scatter_add(cov[N,3,3], batch_indices[N], dim=0)

  Both are `dim=0` with a 1-D index, which `index_add_` computes exactly -- checked
  against an explicit loop in `tests/test_torch_compat.py`, error 0.0. Any other `dim`
  or `reduce` raises `NotImplementedError` instead of guessing, so a future caller
  cannot be quietly given a wrong answer.

WHY A SHIM AND NOT A DIFFERENT ENVIRONMENT. The GPU is the requirement. A CPU-only
interpreter with a distributed-capable torch can host the structRFM encoder, but
RNAbpFlow's upstream sampler hardcodes `torch.device("cuda:N")`, so moving to CPU costs
the feature it is being moved for. Shimming one annotation keeps the ROCm GPU in use and
touches nothing outside this process.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

DEVICE_MESH_MODULE = "torch.distributed.tensor.device_mesh"
TORCH_SCATTER_MODULE = "torch_scatter"


def _distributed_available() -> bool:
    import torch.distributed as dist
    return bool(dist.is_available())


def _make_module(name: str, doc: str, **attrs):
    """Build a stand-in module that survives introspection.

    `__spec__` is not decoration. A module in `sys.modules` with `__spec__ = None` makes
    `importlib.util.find_spec(name)` raise

        ValueError: <name>.__spec__ is None

    and libraries use that call to decide whether an optional dependency is present —
    `torch_geometric` asks about `torch_scatter` exactly that way. Leaving it unset
    would turn "we provided torch_scatter" into a crash somewhere unrelated, which is
    the failure mode this whole mechanism has to avoid. The spec is also what we WANT
    those checks to see: the module is importable, by us.
    """
    import importlib.machinery
    import types

    module = types.ModuleType(name)
    module.__doc__ = doc
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__loader__ = None
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_device_mesh_shim() -> str:
    """Make `from torch.distributed.tensor.device_mesh import DeviceMesh` succeed."""
    import sys

    if _distributed_available():
        return "not needed: torch.distributed is available"
    if DEVICE_MESH_MODULE in sys.modules:
        return "already installed"

    class DeviceMesh:
        """Stand-in. Constructing it is always a bug on this torch build."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "DeviceMesh is a placeholder installed by torusfold.torch_compat: this "
                "torch build has no distributed support, so no device mesh can exist. "
                "The import succeeded only so that modules which merely ANNOTATE with "
                "DeviceMesh can load; constructing one means reaching distributed code "
                "that cannot work here.")

    sys.modules[DEVICE_MESH_MODULE] = _make_module(
        DEVICE_MESH_MODULE,
        "Placeholder installed by torusfold.torch_compat because this torch build was "
        "compiled without distributed support.",
        DeviceMesh=DeviceMesh)
    return "installed (torch.distributed is unavailable)"


def _install_torch_scatter_shim() -> str:
    """Provide `scatter` / `scatter_add` when the compiled extension is absent."""
    import importlib.util
    import sys
    import types

    if TORCH_SCATTER_MODULE in sys.modules:
        return "already installed"
    if importlib.util.find_spec(TORCH_SCATTER_MODULE) is not None:
        return "not needed: torch_scatter is installed"

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        """`out[i] += src[j]` for every `index[j] == i`, along `dim`."""
        if dim not in (0, -src.dim()):
            raise NotImplementedError(
                "torusfold.torch_compat's torch_scatter shim implements dim=0 only; "
                "got dim=%r on a %d-d tensor. Install torch_scatter from "
                "data.pyg.org rather than relying on this shim." % (dim, src.dim()))
        idx = index.to(torch.long)
        if idx.dim() != 1 or idx.numel() != src.shape[0]:
            raise ValueError("index must be 1-D with one entry per row of src")
        size = list(src.shape)
        size[0] = dim_size if dim_size is not None else int(idx.max()) + 1
        if out is None:
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
        out.index_add_(0, idx, src)
        return out

    def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
        if reduce == "sum":
            return scatter_add(src, index, dim, out, dim_size)
        if reduce == "mean":
            total = scatter_add(src, index, dim, None, dim_size)
            idx = index.to(torch.long)
            counts = torch.zeros(total.shape[0], device=src.device, dtype=src.dtype)
            counts.index_add_(0, idx, torch.ones(idx.numel(), device=src.device,
                                                 dtype=src.dtype))
            return total / counts.view([-1] + [1] * (src.dim() - 1)).clamp_min(1)
        raise NotImplementedError(
            "torusfold.torch_compat's torch_scatter shim implements reduce='sum' and "
            "'mean' only; got %r. Install torch_scatter from data.pyg.org rather than "
            "relying on this shim." % (reduce,))

    module = _make_module(
        TORCH_SCATTER_MODULE,
        "Subset of torch_scatter implemented with Tensor.index_add_ by "
        "torusfold.torch_compat. sum and mean at dim=0 only.",
        scatter=scatter, scatter_add=scatter_add)
    sys.modules[TORCH_SCATTER_MODULE] = module
    return "installed (the compiled extension is absent)"


def install() -> Dict[str, str]:
    """Install whichever shims are needed. Idempotent; safe to call anywhere.

    Returns a mapping of what was done, so a caller can report it rather than hide it.
    Both `StructRFMEncoder._load` and the RNAbpFlow runner call this, because both can
    be the first thing in a process to reach the code that needs it.
    """
    return {
        "device_mesh": _install_device_mesh_shim(),
        "torch_scatter": _install_torch_scatter_shim(),
    }


def describe(state: Optional[Dict[str, str]] = None) -> str:
    """One line for a report or a log, naming only what was actually needed."""
    state = install() if state is None else state
    parts = []
    for key, value in state.items():
        if value.startswith("not needed"):
            continue
        parts.append("%s: %s" % (key, value))
    if not parts:
        return "no compatibility shims needed"
    return "compatibility shims active -- " + "; ".join(parts)
