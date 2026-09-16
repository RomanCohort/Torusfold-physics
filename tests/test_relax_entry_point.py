"""The entry relaxation: it descends a strained start, keeps SHAKE's distances, and is OFF.

WHY "OFF" IS THE TEST THAT MATTERS MOST. run_round is the sampler every IBI driver shares, and
round 0 of the 867-chain run was sampled through it. A relaxation that switched itself on would
change the protocol under a run that is already in flight, and nothing in the round jsons would
say so: they record the field's fingerprint, not the entry path. So relax=0 has to evaluate
nothing and touch no coordinate, and that is pinned here by two runs agreeing array by array.

WHY IT EXISTS AT ALL. Twelve chains of the 867 start at 20x the field's energy scale with the
force cap saturated, interpenetrate to 0.02 nm, and melt in 5-15 ps (docs/ibi_loop_and_oxrna_
findings.md, Part 4). The relaxation is steepest descent on the same field and the same cap, with
a normalised direction and a halving line search, and SHAKE after every accepted step.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B          # noqa: E402
import ibi_core as IC                 # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

L = 10


def _chain():
    """A physically ordinary backbone: 0.6 nm P-P steps, P-C4' 0.39, C4'-N 0.335."""
    t = np.arange(L, dtype=float) * 0.6
    P = np.stack([t, 0.4 * np.sin(t * 2.0), 0.4 * np.cos(t * 2.0)], axis=1)
    pos = np.zeros((L, 3, 3), dtype=float)
    pos[:, 0] = P
    pos[:, 1] = P + np.array([0.0, 0.39, 0.0])
    pos[:, 2] = pos[:, 1] + np.array([0.0, 0.0, 0.335])
    return pos


def _strained():
    """The same chain with bead 31 driven onto bead 21 -- a 0.05 nm approach, deep in the wall."""
    pos = _chain()
    pos[7, 1] = pos[5, 0] + np.array([0.05, 0.0, 0.0])
    pos[7, 2] = pos[7, 1] + np.array([0.0, 0.0, 0.335])
    return pos


def _field(pos):
    """(E, max|F|) of a single chain under the shipped field, no pairs, no injection."""
    x = torch.tensor(pos.reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.zeros((0, 2), dtype=torch.long)
    pw = torch.ones(0, dtype=torch.float32)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(x)
    e, f = C.cg_energy_forces(x, ij, pw, cell_list=cl, force_cap=5000.0)
    return float(e.sum()), float(torch.linalg.norm(f.reshape(-1, 3), dim=-1).max())


def _tables(nbins=64):
    tab = {}
    for c in B.COORDS:
        lo, hi = (-1.0, 1.0) if c in ("angle", "dihedral") else (0.0, 2.0)
        binw = (hi - lo) / nbins
        tab[c] = {"lo": lo, "hi": hi, "binw": binw, "U": np.zeros(nbins),
                  "centre": lo + (np.arange(nbins) + 0.5) * binw, "sigma": 0.1 * (hi - lo)}
    return tab


def _round(pos, **kw):
    x = torch.tensor(pos.reshape(1, 3 * L, 3), dtype=torch.float64)
    args = dict(pos=x, vel=torch.zeros_like(x), ij=torch.zeros((0, 2), dtype=torch.long),
                pw=torch.ones(0, dtype=torch.float32), temps=torch.full((1,), 300.0,
                                                                      dtype=torch.float64),
                tab=_tables(), nsteps=40, burn=0, stride=5, blocks=1, friction=1.0,
                force_cap=5000.0, seed=7, nrep=1, progress=False, collect_values=False,
                log=lambda *a, **k: None)
    args.update(kw)
    return IC.run_round(**args)


def test_relax_lowers_a_strained_start_and_keeps_shake():
    con = C.make_intra_constraints(L)
    e0, f0 = _field(_strained())
    x, info = IC.relax_positions(torch.tensor(_strained().reshape(1, 3 * L, 3),
                                             dtype=torch.float64),
                                torch.zeros((0, 2), dtype=torch.long),
                                torch.ones(0, dtype=torch.float32),
                                constraints=con, n_steps=600)
    # info's start numbers are measured AFTER the initial SHAKE projection, so they are not a
    # field evaluation of the raw input (measured on this fixture: 121168 -> 37804 kJ/mol).
    assert info["energy_start"] < e0, "SHAKE projection should have lowered the strained start"
    assert info["max_force_start"] <= f0
    assert info["accepted"] > 0, "the line search rejected every step"
    assert info["energy_end"] < info["energy_start"], (
        f"relaxation did not lower the energy: {info['energy_start']} -> {info['energy_end']}")
    assert info["max_force_end"] <= info["max_force_start"]
    assert torch.isfinite(x).all()
    # SHAKE is applied after every accepted step, so the constrained distances end on their manifold
    assert float(con.residual(x).abs().max()) < 1e-8, "relaxation left the constraint manifold"


def test_relax_is_deterministic():
    con = C.make_intra_constraints(L)
    args = (torch.tensor(_strained().reshape(1, 3 * L, 3), dtype=torch.float64),
            torch.zeros((0, 2), dtype=torch.long), torch.ones(0, dtype=torch.float32))
    a, ia = IC.relax_positions(*args, constraints=con, n_steps=300)
    b, ib = IC.relax_positions(*args, constraints=con, n_steps=300)
    assert ia["energy_end"] == ib["energy_end"]
    assert torch.equal(a, b), "the relaxation consumed randomness; it must not"


def test_relax_zero_is_bit_identical_to_no_relaxation():
    """The default path: nothing evaluated, no coordinate touched, same arrays out."""
    pos = _chain()
    a = _round(pos)
    b = _round(pos, relax=0)
    assert a.relax is None and b.relax is None
    assert torch.equal(a.pos, b.pos), "relax=0 changed the trajectory"
    for c in B.COORDS:
        assert np.array_equal(a.counts[c], b.counts[c]), f"relax=0 changed {c}'s histogram"
    assert a.n_outside == b.n_outside and a.n_total == b.n_total


def test_run_round_records_the_relaxation_when_asked():
    res = _round(_strained(), relax=300)
    assert res.relax is not None and res.relax["accepted"] > 0
    assert res.relax["energy_end"] <= res.relax["energy_start"]
    assert torch.isfinite(res.pos).all()
