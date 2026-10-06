# -*- coding: utf-8 -*-
"""The three refinement knobs: the argument wins, then the environment, then the default.

WHY THIS EXISTS. torch_gpu_refine gained refine_mode / refine_steps / refine_temp, and the
measured difference between the two protocols is not cosmetic (findings Parts 24-27): the folding
path returns base frames with a cosine of 0.650 against the deposit's 0.901 and an unstacked product
(0.0 percent of helical steps stacked), while the refinement keeps 0.907 and stacks 50.0 percent,
1.62 A from the deposit. Every call site in this repository omits all three arguments
(docs/refine_mode_call_sites.md), so the protocol an existing call site runs is decided by
TORUSFOLD_REFINE_MODE / _STEPS / _TEMP -- and a protocol that can be switched without the run saying
so is worse than one that cannot be switched at all.

WHAT IS PINNED, without running a refinement (no GPU, no kernels, no 2OIU input, no trajectory):
the precedence in both directions, the loud rejection of a malformed value, the note that records
what the environment changed, and the fact that the entry point really resolves through the helper.
Every call below is a hand-written argument to the resolver, so nothing here starts a sampler.

Needs no torch: the resolver is deliberately separate from the call that runs the protocol, and
measured, importing the module does not put torch in sys.modules -- so this file runs in the
torch-less CI configuration too. The importorskip below is a belt, not the reason it can pass.
"""
import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

mod = pytest.importorskip("torusfold.scheme2.torch_gpu_refine")

# Read from the module so the fixture cannot end up clearing a spelling that no longer exists.
# The documented names themselves are pinned by the first test below.
ENV_VARS = (mod.REFINE_MODE_ENV, mod.REFINE_STEPS_ENV, mod.REFINE_TEMP_ENV)


@pytest.fixture(autouse=True)
def no_refine_environment(monkeypatch):
    """These tests are ABOUT the environment, so the ambient one must not be part of the input."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def resolve(mode=mod.REFINE_MODE_DEFAULT, steps=mod.REFINE_STEPS_DEFAULT,
            temp=mod.REFINE_TEMP_DEFAULT):
    """Call the resolver the way torch_gpu_refine does, with the argument defaults filled in."""
    return mod._resolve_refine_settings(mode, steps, temp)


def test_the_environment_variable_names_are_the_documented_ones():
    """These strings are the interface; a rename is a silent breaking change for every operator."""
    assert mod.REFINE_MODE_ENV == "TORUSFOLD_REFINE_MODE"
    assert mod.REFINE_STEPS_ENV == "TORUSFOLD_REFINE_STEPS"
    assert mod.REFINE_TEMP_ENV == "TORUSFOLD_REFINE_TEMP"


def test_no_environment_means_the_built_in_defaults():
    s = resolve()
    assert (s.mode, s.steps, s.temp) == ("fold", 1000, 300.0)
    assert s.env_note == "", "nothing came from the environment, so there is nothing to announce"


def test_the_built_in_defaults_are_the_ones_the_signature_reports():
    """The resolver's defaults and the entry point's defaults are two spellings of one fact."""
    sig = inspect.signature(mod.torch_gpu_refine)
    assert sig.parameters["refine_mode"].default == mod.REFINE_MODE_DEFAULT
    assert sig.parameters["refine_steps"].default == mod.REFINE_STEPS_DEFAULT
    assert sig.parameters["refine_temp"].default == mod.REFINE_TEMP_DEFAULT


def test_the_environment_switches_the_mode_when_the_argument_is_at_its_default(monkeypatch):
    monkeypatch.setenv("TORUSFOLD_REFINE_MODE", "refine")
    s = resolve()
    assert s.mode == "refine"
    assert "TORUSFOLD_REFINE_MODE=refine" in s.env_note
    assert "was fold" in s.env_note, "the note has to say what it replaced, not only what it set"


def test_the_environment_also_carries_the_two_numbers(monkeypatch):
    monkeypatch.setenv("TORUSFOLD_REFINE_MODE", "refine")
    monkeypatch.setenv("TORUSFOLD_REFINE_STEPS", "4000")
    monkeypatch.setenv("TORUSFOLD_REFINE_TEMP", "350")
    s = resolve()
    assert (s.mode, s.steps, s.temp) == ("refine", 4000, 350.0)
    assert s.steps == 4000 and isinstance(s.steps, int)
    assert isinstance(s.temp, float)


def test_an_explicit_argument_beats_the_environment(monkeypatch):
    """The caller that knows what it wants outranks the operator's default."""
    monkeypatch.setenv("TORUSFOLD_REFINE_MODE", "fold")
    monkeypatch.setenv("TORUSFOLD_REFINE_STEPS", "8000")
    monkeypatch.setenv("TORUSFOLD_REFINE_TEMP", "500")
    s = resolve("refine", 500, 250.0)
    assert (s.mode, s.steps, s.temp) == ("refine", 500, 250.0)
    assert s.env_note == "", "no value was taken from the environment, so nothing may be announced"


def test_only_the_setting_left_at_its_default_is_taken_from_the_environment(monkeypatch):
    """Precedence is per setting, not per call: one argument does not disable the other two."""
    monkeypatch.setenv("TORUSFOLD_REFINE_STEPS", "2000")
    s = resolve(mod.REFINE_MODE_DEFAULT, mod.REFINE_STEPS_DEFAULT, 275.0)
    assert (s.mode, s.steps, s.temp) == ("fold", 2000, 275.0)
    assert "TORUSFOLD_REFINE_STEPS=2000" in s.env_note
    assert "TORUSFOLD_REFINE_TEMP" not in s.env_note


def test_an_environment_value_equal_to_the_default_changes_nothing_and_says_nothing(monkeypatch):
    """A variable set to what the code already does must not produce a line about a change."""
    monkeypatch.setenv("TORUSFOLD_REFINE_MODE", "fold")
    monkeypatch.setenv("TORUSFOLD_REFINE_STEPS", "1000")
    monkeypatch.setenv("TORUSFOLD_REFINE_TEMP", "300.0")
    s = resolve()
    assert (s.mode, s.steps, s.temp) == ("fold", 1000, 300.0)
    assert s.env_note == ""


def test_the_mode_is_accepted_in_any_case(monkeypatch):
    """An operator typing REFINE must not silently get the folding protocol."""
    monkeypatch.setenv("TORUSFOLD_REFINE_MODE", " REFINE ")
    assert resolve().mode == "refine"


def test_the_note_says_that_steps_and_temperature_are_inert_until_the_mode_is_refine(monkeypatch):
    """Both knobs are read only by the refinement branch; a fold run must not imply otherwise."""
    monkeypatch.setenv("TORUSFOLD_REFINE_STEPS", "4000")
    monkeypatch.setenv("TORUSFOLD_REFINE_TEMP", "350")
    s = resolve()
    assert s.mode == "fold" and (s.steps, s.temp) == (4000, 350.0)
    assert "inert while the mode is fold" in s.env_note


@pytest.mark.parametrize("name,value", [
    ("TORUSFOLD_REFINE_MODE", "refinement"),
    ("TORUSFOLD_REFINE_STEPS", "many"),
    ("TORUSFOLD_REFINE_STEPS", "0"),
    ("TORUSFOLD_REFINE_TEMP", "warm"),
    ("TORUSFOLD_REFINE_TEMP", "-5"),
])
def test_a_malformed_value_fails_loudly(monkeypatch, name, value):
    """A typo that left the run folding silently is the failure this mechanism exists to prevent."""
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as excinfo:
        resolve()
    assert name in str(excinfo.value), "the message has to name the variable to fix"


def test_a_malformed_value_fails_even_when_the_argument_wins(monkeypatch):
    """The value is validated whenever it is set: a stale variable is never ignored in silence."""
    monkeypatch.setenv("TORUSFOLD_REFINE_STEPS", "not-a-number")
    with pytest.raises(SystemExit):
        resolve("refine", 500, 250.0)


def test_the_entry_point_resolves_through_the_helper():
    """The helper is only worth testing if the run actually calls it -- a static check, no run."""
    src = inspect.getsource(mod.torch_gpu_refine)
    assert "_resolve_refine_settings(" in src, (
        "torch_gpu_refine must resolve refine_mode/refine_steps/refine_temp through "
        "_resolve_refine_settings, or the environment route reaches nothing")


def test_the_entry_point_prints_the_note_it_gets_back():
    """Resolving is not enough: the run has to SAY that the environment changed its protocol."""
    src = inspect.getsource(mod.torch_gpu_refine)
    assert "_resolve_refine_settings(" in src
    assert "if _refine_env_note:" in src, "the note must be printed, not only computed"
    assert 'print("  [Torch GPU] " + _refine_env_note)' in src, (
        "the resolved note has to reach stdout, which is the only place a run says what it ran")

# ---------------------------------------------------------------------------------------------
# TORUSFOLD_BEAD_SOURCE: where the refinement's INITIAL BASE FRAMES come from (added with the
# environment-only production switch, findings Part 29). Measured on 2OIU: beads read from the
# deposit keep a base-frame cosine of 0.897 and the product stacks 58.3 percent of its helical steps,
# while beads FABRICATED from a P-only input give 0.607 and an unstacked product. The argument wins,
# then the environment; a path that does not exist is refused loudly, because a silent fall-back to
# fabricated beads is the failure this knob exists to prevent.


def _bead_source(value=None):
    import torusfold.scheme2.torch_gpu_refine as T
    return T._resolve_bead_source(value)


def test_bead_source_argument_wins(monkeypatch, tmp_path):
    f = tmp_path / "deposit.pdb"
    f.write_text("END\n", encoding="utf-8")
    monkeypatch.setenv("TORUSFOLD_BEAD_SOURCE", str(tmp_path / "other.pdb"))
    path, note = _bead_source(str(f))
    assert path == str(f)
    assert note == ""


def test_bead_source_environment_is_used_and_reported(monkeypatch, tmp_path):
    f = tmp_path / "deposit.pdb"
    f.write_text("END\n", encoding="utf-8")
    monkeypatch.setenv("TORUSFOLD_BEAD_SOURCE", str(f))
    path, note = _bead_source(None)
    assert path == str(f)
    assert "TORUSFOLD_BEAD_SOURCE" in note


def test_bead_source_absent_is_none(monkeypatch):
    monkeypatch.delenv("TORUSFOLD_BEAD_SOURCE", raising=False)
    path, note = _bead_source(None)
    assert path is None and note == ""


def test_bead_source_missing_file_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("TORUSFOLD_BEAD_SOURCE", str(tmp_path / "nope.pdb"))
    with pytest.raises(SystemExit) as e:
        _bead_source(None)
    assert "TORUSFOLD_BEAD_SOURCE" in str(e.value)


def test_refine_branch_reports_progress_to_on_report():
    """The refine path must call on_report: the folding path hangs it on the REMD instance, which refine
    mode never builds, so without this a live-panel consumer sits on the input coordinates."""
    import torusfold.scheme2.torch_gpu_refine as T
    assert "on_report(" in inspect.getsource(T._refine_langevin)
    assert "on_report=on_report" in inspect.getsource(T.torch_gpu_refine)
