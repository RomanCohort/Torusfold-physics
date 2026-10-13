"""The two shims that let the optional predictors load under this ROCm torch.

WHY THEY EXIST, MEASURED. The GPU torch here reports

    torch 2.12.0a0+rocm7.13.0a20260313
    torch._C._has_distributed()      -> False
    torch.distributed.is_available() -> False
    torch.cuda.get_device_name(0)    -> AMD Radeon(TM) 8060S Graphics

so the GPU works and distributed does not. Two optional predictors import into that gap
by accident:

    torchmetrics.functional.text.bert -> from transformers import AutoModel
      -> transformers.generation.utils -> .continuous_batching
        -> continuous_batching/distributed.py
          -> from torch.distributed.tensor.device_mesh import DeviceMesh
            -> torch/distributed/tensor/__init__.py -> _ops -> _dtensor_spec
              -> placement_types -> _functional_collectives
                -> import torch.distributed.distributed_c10d as c10d
                  -> from torch._C._distributed_c10d import (...)   <-- does not exist

and `pytorch_lightning.utilities.types -> torchmetrics` puts RNAbpFlow's model class on
the same path. The second shim covers the other missing piece: `torch_scatter` has no
Windows wheel for this torch and building it needs MSVC, while RNAbpFlow calls exactly
two of its functions.

These tests are about the shims being EXACT and REFUSING rather than guessing. A shim
that silently computes something else is worse than an ImportError.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torusfold import torch_compat  # noqa: E402


# ── torch_scatter: exactness ─────────────────────────────────────────────────

def _shim():
    """The scatter functions, taken from wherever install() puts them."""
    torch_compat.install()
    return sys.modules["torch_scatter"]


def test_scatter_add_matches_an_explicit_loop():
    ts = _shim()
    torch.manual_seed(0)
    src = torch.randn(9, 3, dtype=torch.float64)
    idx = torch.tensor([0, 2, 2, 1, 0, 2, 1, 1, 0])
    got = ts.scatter_add(src, idx, dim=0)
    want = torch.stack([src[idx == i].sum(0) for i in range(3)])
    assert torch.equal(got, want)


def test_scatter_mean_matches_an_explicit_loop():
    ts = _shim()
    torch.manual_seed(1)
    src = torch.randn(9, 3, dtype=torch.float64)
    idx = torch.tensor([0, 2, 2, 1, 0, 2, 1, 1, 0])
    got = ts.scatter(src, idx, dim=0, reduce="mean")
    want = torch.stack([src[idx == i].mean(0) for i in range(3)])
    assert torch.allclose(got, want, atol=1e-12)


def test_the_3d_case_rnabpflow_actually_uses():
    """`utils.py:451` scatters a [N,3,3] covariance with a 1-D index. That is the call
    whose result is fed to an SVD, so an error here is a silently wrong rotation."""
    ts = _shim()
    torch.manual_seed(2)
    src = torch.randn(6, 3, 3, dtype=torch.float64)
    idx = torch.tensor([0, 1, 1, 0, 1, 0])
    got = ts.scatter_add(src, idx, dim=0)
    want = torch.stack([src[idx == i].sum(0) for i in range(2)])
    assert torch.equal(got, want)


def test_an_empty_group_means_zero_not_nan():
    """`mean` divides by a count; the count for a group with no rows is zero."""
    ts = _shim()
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    idx = torch.tensor([0, 0])
    got = ts.scatter(src, idx, dim=0, reduce="mean", dim_size=3)
    assert got.shape == (3, 2)
    assert torch.allclose(got[0], torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert torch.isfinite(got).all()


def test_the_shim_refuses_a_reduction_it_does_not_implement():
    """`max` and `mul` are real torch_scatter reductions. Guessing at them would be a
    silent wrong answer; installing the real package is the fix."""
    ts = _shim()
    src = torch.randn(4, 2)
    idx = torch.tensor([0, 0, 1, 1])
    for reduce in ("max", "min", "mul"):
        with pytest.raises(NotImplementedError) as exc:
            ts.scatter(src, idx, dim=0, reduce=reduce)
        assert "data.pyg.org" in str(exc.value)


def test_the_shim_refuses_dim_other_than_zero():
    ts = _shim()
    src = torch.randn(2, 4)
    idx = torch.tensor([0, 1, 0, 1])
    with pytest.raises(NotImplementedError) as exc:
        ts.scatter_add(src, idx, dim=1)
    assert "dim=0 only" in str(exc.value)


def test_the_shim_rejects_an_index_of_the_wrong_shape():
    ts = _shim()
    with pytest.raises(ValueError):
        ts.scatter_add(torch.randn(4, 2), torch.tensor([[0, 1], [1, 0]]), dim=0)


def test_gpu_tensors_stay_on_the_gpu():
    """The whole reason for shimming instead of moving to a CPU interpreter."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU in this interpreter")
    ts = _shim()
    src = torch.randn(4, 3, device="cuda")
    idx = torch.tensor([0, 1, 1, 0], device="cuda")
    out = ts.scatter_add(src, idx, dim=0)
    assert out.is_cuda
    assert torch.allclose(out, torch.stack([src[idx == i].sum(0) for i in range(2)]))


# ── the shims must not shadow anything real ──────────────────────────────────

def test_a_real_torch_scatter_is_never_shadowed():
    """If the compiled extension is importable, install() must leave it alone."""
    state = torch_compat.install()
    if importlib.util.find_spec("torch_scatter") is not None:
        # A real package is on the path; whether it was already imported decides the
        # wording, but "installed by us" must never be the answer.
        assert "not needed" in state["torch_scatter"] or state["torch_scatter"] == \
            "already installed"


def test_the_shim_module_survives_find_spec():
    """`torch_geometric` decides whether torch_scatter is present by calling
    `importlib.util.find_spec`, which RAISES on a module whose `__spec__` is None.
    A shim that broke that call would crash code that never asked about us."""
    torch_compat.install()
    spec = importlib.util.find_spec("torch_scatter")
    assert spec is not None
    assert spec.name == "torch_scatter"
    import torch.distributed as dist
    if not dist.is_available():
        spec = importlib.util.find_spec(torch_compat.DEVICE_MESH_MODULE)
        assert spec is not None


def test_install_is_idempotent_and_describes_itself():
    first = torch_compat.install()
    second = torch_compat.install()
    assert set(first) == set(second) == {"device_mesh", "torch_scatter"}
    text = torch_compat.describe(second)
    assert isinstance(text, str) and text


def test_the_device_mesh_shim_is_not_installed_when_distributed_exists():
    """On a torch with distributed, the real DTensor module must be used. This test
    asserts the guard exists rather than the outcome, because the outcome depends on
    the interpreter."""
    import torch.distributed as dist
    state = torch_compat.install()
    if dist.is_available():
        assert state["device_mesh"].startswith("not needed")


# ── the device_mesh placeholder ──────────────────────────────────────────────

def test_the_placeholder_device_mesh_refuses_to_be_constructed():
    """Importing it must work (modules annotate with it) and constructing it must not:
    there is no mesh to build, so reaching construction is a code path that cannot be
    legitimate on this build."""
    import torch.distributed as dist
    if dist.is_available():
        pytest.skip("this interpreter has real distributed support")
    torch_compat.install()
    module = sys.modules[torch_compat.DEVICE_MESH_MODULE]
    with pytest.raises(RuntimeError) as exc:
        module.DeviceMesh()
    assert "placeholder" in str(exc.value).lower()
    assert "torch_compat" in str(exc.value)


def test_the_shims_are_what_makes_the_predictors_importable():
    """End to end, in this interpreter: with the shims, both casualties import.

    Skipped where the packages are not installed at all -- CI has no transformers.
    """
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers is not installed in this interpreter")
    torch_compat.install()
    from transformers import AutoModel, AutoTokenizer       # noqa: F401
    import torchmetrics                                     # noqa: F401
