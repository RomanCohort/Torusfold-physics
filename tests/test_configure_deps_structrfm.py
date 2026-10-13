"""The structRFM row, and the difference between installed and importable.

TWO GAPS, both measured on this machine.

1. A directory named `structRFM` is not a checkpoint.
   `probe: None` for this tool, so the search accepted any non-empty directory whose
   name matched a hint. The first attempt at fetching this checkpoint produced
   exactly that state: 24 MB of 344 MB, curl dying with "schannel: server closed
   abruptly", and a directory that a presence check would have called found. A
   HuggingFace checkpoint is loadable only when `config.json`, `tokenizer.json` and a
   weights file are all there; `_structrfm_verdict` is the rule.

2. `find_spec` said transformers was fine; nothing could import it.
   The report printed "OK transformers 5.9.0" for an interpreter in which
   `from transformers import AutoModel` raises

       ModuleNotFoundError: No module named 'torch._C._distributed_c10d';
       'torch._C' is not a package

   via `transformers.generation.utils` -> `continuous_batching` ->
   `torch.distributed.tensor.device_mesh`. `import transformers` succeeds, so the
   module-import fallback in `check_packages` returned a version and the row looked
   healthy. Presence and importability are different questions and the report now
   answers both.

These are the same defect as the isRNAcirc BLOCKING line: a check that looks at
whether a thing EXISTS rather than whether the caller can USE it.
"""

import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import configure_deps as cd  # noqa: E402


def _checkpoint(tmp_path, *, config=True, tokenizer=True, weights="model.safetensors",
                size=1024):
    """The shape of a HuggingFace checkpoint directory."""
    d = tmp_path / "structRFM"
    d.mkdir(parents=True, exist_ok=True)
    if config:
        (d / "config.json").write_text(json.dumps({"model_type": "bert"}))
    if tokenizer:
        (d / "tokenizer.json").write_text("{}")
    if weights:
        (d / weights).write_bytes(b"\x00" * size)
    return d


# ── gap 1: what counts as a checkpoint ───────────────────────────────────────

def test_a_complete_checkpoint_resolves(tmp_path):
    ok, why = cd._structrfm_verdict(str(_checkpoint(tmp_path)), transformers_ok=True)
    assert ok is True
    assert "model.safetensors" in why


def test_the_other_weights_filename_is_accepted(tmp_path):
    """Checkpoints ship either spelling; which one is not this script's business."""
    ok, why = cd._structrfm_verdict(
        str(_checkpoint(tmp_path, weights="pytorch_model.bin")), transformers_ok=True)
    assert ok is True
    assert "pytorch_model.bin" in why


def test_the_reported_size_is_the_real_file_size(tmp_path):
    """344 MB and 24 MB have to be distinguishable in the report -- that difference
    is the whole reason this check exists."""
    d = _checkpoint(tmp_path, size=2_500_000)
    ok, why = cd._structrfm_verdict(str(d), transformers_ok=True)
    assert ok is True
    assert "3 MB" in why or "2 MB" in why


def test_a_download_that_stopped_early_is_not_a_checkpoint(tmp_path):
    """Exactly what a killed transfer leaves: the small files and no weights."""
    ok, why = cd._structrfm_verdict(
        str(_checkpoint(tmp_path, weights=None)), transformers_ok=True)
    assert ok is False
    assert "without the weights" in why


def test_config_without_a_tokenizer_is_not_loadable(tmp_path):
    ok, why = cd._structrfm_verdict(
        str(_checkpoint(tmp_path, tokenizer=False)), transformers_ok=True)
    assert ok is False
    assert "tokenizer.json" in why


def test_the_search_rejects_a_bare_directory_named_structrfm(tmp_path):
    """The `require` list is what stops the search accepting a name with nothing
    behind it. Without it this returns True with '2 entries'."""
    tool = next(t for t in cd.TOOLS if t["key"] == "structrfm")
    empty = tmp_path / "structRFM"
    empty.mkdir()
    (empty / "README.md").write_text("downloading...")
    ok, why = cd._looks_like_tool(str(empty), tool)
    assert ok is False
    assert "config.json" in why

    good = _checkpoint(tmp_path / "good")
    ok, why = cd._looks_like_tool(str(good), tool)
    assert ok is True


def test_a_present_checkpoint_is_still_resolved_when_the_loader_cannot_import(tmp_path):
    """The checkpoint is not at fault, so it must not be reported as missing -- the
    package row carries that, and this row says so in words."""
    ok, why = cd._structrfm_verdict(str(_checkpoint(tmp_path)), transformers_ok=False)
    assert ok is True
    assert "NOT loadable" in why
    assert "package row" in why


def test_no_claim_is_made_when_the_probe_did_not_run(tmp_path):
    """`--fast` skips the import probes. The row must then say nothing about
    loadability rather than guessing either way."""
    ok, why = cd._structrfm_verdict(str(_checkpoint(tmp_path)), transformers_ok=None)
    assert ok is True
    assert "NOT loadable" not in why


# ── gap 2: installed vs importable ───────────────────────────────────────────

def test_a_failing_probe_is_reported_with_its_reason(tmp_path):
    probes = {"boom": "raise RuntimeError('nope')"}
    out = cd.probe_package_imports(sys.executable, timeout=60, probes=probes)
    assert out["boom"]["ok"] is False
    assert "RuntimeError" in out["boom"]["why"]
    assert "nope" in out["boom"]["why"]


def test_a_succeeding_probe_is_reported_as_ok(tmp_path):
    probes = {"fine": "import json"}
    out = cd.probe_package_imports(sys.executable, timeout=60, probes=probes)
    assert out["fine"]["ok"] is True
    assert out["fine"]["why"] is None


def test_the_probe_tests_what_the_caller_actually_imports():
    """The narrow statement, not the module name. `import transformers` succeeds on
    the machine this was written for while `from transformers import AutoModel`
    does not, so probing the module name would have missed it."""
    assert cd.PACKAGE_IMPORT_PROBES["transformers"].startswith(
        "from transformers import AutoModel")


def test_a_missing_interpreter_yields_no_probes_rather_than_an_exception():
    assert cd.probe_package_imports(r"C:\definitely\not\here\python.exe") == {}
    assert cd.probe_package_imports("") == {}


def test_check_packages_marks_a_found_but_unimportable_package():
    probes = {"transformers": {"ok": False, "why": "ModuleNotFoundError: nope"}}
    pkgs = cd.check_packages(probes)
    row = next(p for p in pkgs if p["module"] == "transformers")
    # The distribution IS installed here; only the import fails. Reporting
    # "MISSING" would send someone to reinstall a package that is already present.
    if row["found"]:
        assert row["usable"] is False
        assert "nope" in row["import_why"]
        assert row["import_probe"] == cd.PACKAGE_IMPORT_PROBES["transformers"]


def test_packages_without_a_probe_are_left_unknown_not_broken():
    pkgs = cd.check_packages({})
    assert all(p["usable"] is None for p in pkgs)


def test_the_summary_separates_not_importable_from_missing():
    """Reinstalling fixes MISSING; it does not fix BROKEN. The two lines are
    different instructions and used to be one."""
    result = {
        "tools": [],
        "binaries": [],
        "interpreters": [],
        "interpreter": None,
        "packages": [
            {"module": "numpy", "label": "numpy", "why": "core", "required": True,
             "found": True, "version": "1.0", "usable": True, "import_why": None,
             "import_probe": None},
            {"module": "transformers", "label": "transformers", "why": "heads",
             "required": False, "found": True, "version": "5.9.0", "usable": False,
             "import_why": "ModuleNotFoundError: no torch._C._distributed_c10d",
             "import_probe": "from transformers import AutoModel, AutoTokenizer"},
        ],
        "import_probes_run": True,
    }
    text = cd.render(result)
    assert "BROKEN" in text
    assert "NOT IMPORTABLE : transformers" in text
    assert "MISSING PACKAGES" not in text


def test_a_skipped_probe_says_so_in_the_report():
    result = {"tools": [], "binaries": [], "interpreters": [], "interpreter": None,
              "packages": [], "import_probes_run": False}
    text = cd.render(result)
    assert "import probes not run" in text


@pytest.mark.parametrize("bad,expected", [
    ("import torch._C._distributed_c10d", "ModuleNotFoundError"),
])
def test_the_known_torch_gap_is_expressible_as_a_probe(bad, expected):
    """The exact statement that fails on a torch built without distributed. Skip if
    this interpreter is one of the ones where it works."""
    out = cd.probe_package_imports(sys.executable, timeout=120, probes={"t": bad})
    if out["t"]["ok"]:
        pytest.skip("this interpreter's torch has distributed compiled in")
    assert expected in out["t"]["why"]
