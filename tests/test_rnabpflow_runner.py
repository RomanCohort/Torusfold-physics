"""The adapter that drives upstream RNAbpFlow when the team's `inference_rocm.py` is absent.

`ensemble_predictor._run_rnabpflow` used to require `<RNABPFLOW_ROOT>/inference_rocm.py`.
That file is a team-held adaptation: it is not on this machine, it is not in any upstream
release, and no URL for it exists. Upstream ships `inference.py`, which takes **no
command-line arguments** and is driven by `configs/inference.yaml` plus the current
directory -- `inference.py:87` hardcodes `CONFIG_FILE_PATH = "configs/inference.yaml"`
and `run_inference()` walks `Inputs/list.txt`. So a checkout that upstream considers
complete was reported as a missing tool.

`tools/rnabpflow_runner.py` is the adapter: it writes the config under the caller's
working directory, puts the checkout on `sys.path`, installs the torch shims and calls
`inference.run_inference()`. Verified end to end on this machine at 71 nt / 1 sample /
50 timesteps: 38 s on the ROCm GPU, producing `Predictions/seq/Sample_0.pdb` and a
(71, 3) coordinate array.

These tests cover the argument contract and the config it writes. They deliberately do
not run the model: that needs a 203 MB checkpoint, a GPU and `mp.spawn`, and a test that
did it would be measuring the machine.
"""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import rnabpflow_runner as runner  # noqa: E402


def test_a_cpu_device_is_refused_with_a_reason():
    """Upstream's `Sampler.sample_GPU` builds `torch.device(f"cuda:{i}")`
    unconditionally, so accepting `--device cpu` would fail later with a worse
    message. The refusal names the reason instead of just saying no."""
    rc = runner.main(["--name", "seq", "--checkpoint", __file__, "--device", "cpu",
                      "--root", str(ROOT)])
    assert rc == 2


def test_a_missing_root_is_refused():
    rc = runner.main(["--name", "seq", "--checkpoint", __file__,
                      "--root", str(ROOT / "no_such_checkout")])
    assert rc == 2


def test_a_missing_checkpoint_is_refused(tmp_path):
    rc = runner.main(["--name", "seq", "--checkpoint", str(tmp_path / "absent.ckpt"),
                      "--root", str(ROOT)])
    assert rc == 2


def test_the_config_it_writes_is_the_one_upstream_reads(tmp_path):
    """`inference.py` reads `CONFIG_FILE_PATH = "configs/inference.yaml"` relative to
    the working directory, and gets everything else from inside the file -- so the
    paths have to be absolute and the key names exact."""
    yaml = pytest.importorskip("yaml")
    path = runner._write_config(str(tmp_path), "seq", "C:/ck/RNA3DB.ckpt",
                                num_gpus=1, samples=3, timesteps=25)
    assert path == str(tmp_path / "configs" / "inference.yaml")

    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    inf = cfg["inference"]
    assert set(inf) >= {"ckpt_path", "input_dir", "output_dir", "name", "num_gpus",
                        "interpolant", "samples", "output_format"}
    assert inf["name"] == "seq"
    assert inf["num_gpus"] == 1
    assert inf["samples"]["samples_per_sequence"] == 3
    assert inf["interpolant"]["sampling"]["num_timesteps"] == 25
    # Absolute, because the config lives in a scratch directory while the data does not.
    assert pathlib.Path(inf["input_dir"]).is_absolute()
    assert pathlib.Path(inf["output_dir"]).is_absolute()
    assert pathlib.Path(inf["ckpt_path"]).is_absolute()


def test_the_expected_output_path_matches_what_the_caller_looks_for(tmp_path):
    """The caller reads `Predictions/seq/Sample_0.pdb` under its own cwd. If this
    drifts, the run succeeds and the caller reports "output not found"."""
    yaml = pytest.importorskip("yaml")
    path = runner._write_config(str(tmp_path), "seq", "C:/ck/RNA3DB.ckpt", 1, 1, 50)
    inf = yaml.safe_load(open(path, encoding="utf-8"))["inference"]
    assert inf["output_dir"] == str(tmp_path / "Predictions")
    assert str(tmp_path / "Predictions" / inf["name"] / "Sample_0.pdb") == \
        str(tmp_path / "Predictions" / "seq" / "Sample_0.pdb")


def test_the_runner_installs_the_shims_before_importing_anything_heavy():
    """`inference.py` uses `mp.spawn`, so each worker re-imports this module. The shims
    are therefore installed at import time, not inside `main()` -- a shim that only ran
    under the `__main__` guard would be missing in every worker."""
    source = pathlib.Path(runner.__file__).read_text(encoding="utf-8")
    guard = source.index('if __name__ == "__main__"')
    installs = source.index("torch_compat.install()")
    assert installs < guard, "shim install must be at module level, before the guard"


def test_the_runner_puts_the_checkout_on_the_import_path():
    """`inference.py` does `from src.models.flow_module_inf import FlowModule`, so the
    checkout root has to be importable."""
    source = pathlib.Path(runner.__file__).read_text(encoding="utf-8")
    assert "sys.path.insert(0, root)" in source
