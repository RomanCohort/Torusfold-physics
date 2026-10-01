"""The tabulated potentials: cpu numbers must not move, and the tensors must follow the device.

WHY THIS EXISTS. cg_potentials.make_potential used to REFUSE a non-cpu pos by name:

    "the tabulated potentials live on cpu ... Supporting cuda means moving the table tensors and the
     interpolation together, not just this call, so it is left until something needs it."

The pipeline's CG stage (src/torusfold/scheme2/torch_gpu_refine.py) runs on
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
so "something needs it" is now. force_reference resolves the installed table per call from pos.device
and copies only U; these tests pin both halves -- the cpu numbers (captured before the change, frozen
in GOLDEN) and the device-following, the latter skipped when no CUDA device exists.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import cg_potentials as P          # noqa: E402
import force_reference as FR       # noqa: E402

TABLES = REPO / "results" / "boltzmann_tables_clean.npz"

# Captured on 2026-10-01 BEFORE the device change, from the toy structure below.
GOLDEN = {
    ("bb_bond", "table"): (3.509610409018833e+00, 5.954851533626274e+01),
    ("bb_bond", "table_wall:2000"): (3.509610409018833e+00, 5.954851533626274e+01),
    ("angle", "table"): (2.483220962879895e+01, 0.0),
    ("dihedral", "table"): (3.260350731118078e+01, 0.0),
}


@pytest.fixture(autouse=True)
def restore_installed_tables():
    """These tests install table files, and the INSTALLED record is module state.

    Without this, test_installing_a_table_invalidates_the_device_copies leaves refit_smooth5.npz
    installed and the next test file (test_table_potential_injection.py) compares its wall against
    boltzmann_bonded with a different table -- measured, not hypothetical: 301.92 against 31.97.
    """
    saved = (FR._TABLE, FR._ANGLE_TABLE, FR._BOND_TABLE, dict(FR._DEVICE_TABLES))
    yield
    FR._TABLE, FR._ANGLE_TABLE, FR._BOND_TABLE, dev = saved
    FR._DEVICE_TABLES.clear()
    FR._DEVICE_TABLES.update(dev)


def toy(L=6):
    pos = torch.zeros(1, 3 * L, 3, dtype=torch.float64)
    b = torch.tensor([[0.59 * k, 0.0, 0.0] for k in range(L)], dtype=torch.float64)
    for i in range(L):
        pos[0, 3 * i] = b[i]
        pos[0, 3 * i + 1] = b[i] + torch.tensor([0.12, 0.32, 0.0], dtype=torch.float64)
        pos[0, 3 * i + 2] = pos[0, 3 * i + 1] + torch.tensor([0.07, 0.26, 0.13], dtype=torch.float64)
    return pos


@pytest.mark.parametrize("coord,spec_text", sorted(GOLDEN))
def test_cpu_numbers_are_bit_identical(coord, spec_text):
    P.use_table_file(str(TABLES))
    fn = P.make_potential(coord, P.resolve_spec(spec_text, coord))
    e, f = fn(toy().requires_grad_(True))
    e_sum, f_norm = float(e.detach().sum()), float(f.norm())
    want_e, want_f = GOLDEN[(coord, spec_text)]
    assert e_sum == want_e, f"{coord}/{spec_text}: energy moved from {want_e} to {e_sum}"
    assert f_norm == want_f, f"{coord}/{spec_text}: force moved from {want_f} to {f_norm}"


def test_table_tensors_follow_the_requested_device():
    for kind in ("bb_bond", "angle", "dihedral"):
        assert FR.table_for(kind, "cpu")["U"].device.type == "cpu"
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the cuda half of this test needs one")
    for kind in ("bb_bond", "angle", "dihedral"):
        rec = FR.table_for(kind, "cuda")
        assert rec["U"].device.type == "cuda"
        assert rec["U"].dtype == torch.float64
        # the scalars stay python floats on purpose: they travel with the arithmetic
        assert isinstance(rec["lo"], float) and isinstance(rec["binw"], float)


def test_installing_a_table_invalidates_the_device_copies():
    P.use_table_file(str(TABLES))
    refit = REPO / "results" / "refit_smooth5.npz"
    if not refit.exists():
        pytest.skip("refit_smooth5.npz not present")
    P.use_table_file(str(refit))
    z = np.load(refit)
    assert FR.angle_table()["lo"] == float(z["angle__lo"])
    assert FR._DEVICE_TABLES == {}, "a new install must clear the per-device copies"
