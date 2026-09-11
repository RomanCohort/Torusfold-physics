"""One excluded-volume potential, four call sites, and the fifth copy that must not appear.

The term used to be a one-sided linear spring, E = 0.5*K_CLASH*max(0, 0.300-d)^2, written out
four times in torch_cgsim.py: in cg_energy_3bead, in cg_energy, in _clash_f and in
_explicit_forces_clash. Four copies of one law means a retune moves whichever paths read the
global that was changed and silently leaves the rest -- the failure mode this file's neighbours
(test_ff_bonded_targets.py) exist for, and the reason the constants block has no aliases.

What is checked here, in the order the change was asked for:

  1. There is exactly ONE definition of the pair energy and ONE of the force, and no function
     outside the call chain names the stiffness or the range. A fifth copy has to name one of
     the two constants to be a copy, so it is caught; a fifth copy with its own private numbers
     would not be, and that limit is stated rather than papered over.
  2. The form: it diverges as d -> 0, and it is exactly zero with zero slope at sigma, so
     nothing beyond the range is perturbed.
  3. The four computed clash energies and clash forces agree with each other and with the
     closed form, and they all move together when the one function's parameters move.
  4. The parameters are measured, not chosen: sigma is the database's own lower edge and k is
     the Boltzmann inversion of its two lowest populated bins (scripts/
     calibrate_excluded_volume.py is the authority; the numbers are repeated here so a retune
     cannot pass quietly).

Run: python tests/test_clash_single_potential.py
"""
import ast
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C  # noqa: E402

KBT = 2.494

# The database, rsRNASP/Training_set, 191 PDB files / 126 gap-free chains: 213732 non-bonded
# P-P pairs with residue index gap >= 3, minimum 0.3975 nm, and the histogram
# scripts/measure_pp_pair_distribution.py reports.
DB_PAIRS = 213732
DB_PP_MIN = 0.3975
DB_BINS = ((0.36, 0.40, 1), (0.40, 0.45, 6), (0.45, 0.50, 8), (0.50, 0.60, 60))

POTENTIAL_FUNCTIONS = ("_clash_pair_energy", "_clash_pair_dedr", "_clash_pair_energy_force")
# Functions that may name K_CLASH or CLASH_SIGMA: the five field entry points plus the
# cell-list search radius, which is min(cell_size, CLASH_SIGMA) because the 27-cell
# neighbourhood has to span the range.
NAMED_CONSTANT_ALLOWLIST = {
    "cg_energy", "cg_energy_3bead", "cg_energy_forces",
    "cg_forces_explicit_batched", "cg_forces_explicit", "build",
}

SOURCE = Path(C.__file__).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _defs():
    return [n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)]


def _function_nodes():
    return {n.name: n for n in _defs()}


def _walk_hits(node, current, hits):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            _walk_hits(child, child.name, hits)
            continue
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            hits.setdefault(child.func.id, set()).add(current)
        if (isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                and child.id in ("K_CLASH", "CLASH_SIGMA", "CLASH_DIST")):
            hits.setdefault(child.id, set()).add(current)
        _walk_hits(child, current, hits)


def _hits():
    out = {}
    _walk_hits(TREE, "<module>", out)
    return out


# ── 1. one definition, and no fifth copy ────────────────────────────────────────────────────
def test_there_is_exactly_one_definition_of_the_pair_energy_and_the_force():
    defs = _defs()
    for name in POTENTIAL_FUNCTIONS:
        n = sum(1 for d in defs if d.name == name)
        assert n == 1, (
            f"{name} is defined {n} times in torch_cgsim.py; the excluded volume has to have "
            f"one energy and one force so an edit cannot reach only some of the call sites")


def test_only_the_call_chain_calls_the_shared_potential():
    """The four call sites, and nothing else, reach the one pair potential."""
    hits = _hits()
    energy_callers = hits.get("_clash_pair_energy", set())
    force_callers = hits.get("_clash_pair_energy_force", set())
    assert energy_callers == {"cg_energy", "cg_energy_3bead", "_clash_pair_dedr",
                              "_clash_pair_energy_force"}, (
        f"cg_energy and cg_energy_3bead are the two energy-only callers and "
        f"_clash_pair_energy_force is the only force-side caller; got {sorted(energy_callers)}")
    assert force_callers == {"_clash_f"}, (
        f"the force side must be one call from _clash_f (which cg_energy_forces, "
        f"cg_forces_explicit_batched and cg_forces_explicit all go through); "
        f"got {sorted(force_callers)}")


def test_no_function_outside_the_call_chain_names_the_stiffness_or_the_range():
    """A fifth copy has to name K_CLASH or CLASH_SIGMA. This catches that one."""
    hits = _hits()
    named = hits.get("K_CLASH", set()) | hits.get("CLASH_SIGMA", set())
    extra = sorted(named - NAMED_CONSTANT_ALLOWLIST)
    assert not extra, (
        f"{extra} name K_CLASH or CLASH_SIGMA but are not one of the five field entry points "
        f"or the cell-list search radius: another copy of the excluded-volume law is being "
        f"written there. A copy with its own private numbers would not be caught by this test; "
        f"the one-definition test above and the four-path agreement test below are what cover "
        f"the mechanical cases.")


def test_the_old_linear_spring_is_gone():
    for pattern in ("CLASH_DIST -", "(CLASH_DIST", "K_CLASH * over", "over ** 2"):
        assert pattern not in SOURCE, (
            f"{pattern!r} is still in torch_cgsim.py: that is the one-sided linear spring, "
            f"whose force saturates at K_CLASH*0.300 = 150 kJ/mol/nm")


def test_the_retired_range_is_not_read_by_the_module():
    """CLASH_DIST is kept only because tests/test_pair_clash_bsj_criterion.py pins it.

    It is the old range, 0.300 nm, below every distance the database contains. A second range
    that nothing reads is harmless; a second range that something reads is the drift this
    whole change is about, so the module is not allowed to load the name at all.
    """
    loads = [n for n in ast.walk(TREE)
             if isinstance(n, ast.Name) and n.id == "CLASH_DIST" and isinstance(n.ctx, ast.Load)]
    assert loads == [], (
        f"CLASH_DIST is loaded at line(s) {[n.lineno for n in loads]} in torch_cgsim.py; the "
        f"live range is CLASH_SIGMA and the retired name must not come back as a second one")


# ── 2. the form ─────────────────────────────────────────────────────────────────────────────
def _e(r):
    return C._clash_pair_energy(torch.tensor(r, dtype=torch.float64), C.K_CLASH, C.CLASH_SIGMA)


def _dedr(r):
    return C._clash_pair_dedr(torch.tensor(r, dtype=torch.float64), C.K_CLASH, C.CLASH_SIGMA)


def test_the_range_switches_the_term_off_exactly():
    s = C.CLASH_SIGMA
    assert float(_e([s])) == 0.0
    assert float(_e([s + 1e-9, 0.5, 1.0, 5.0]).abs().max()) == 0.0
    assert float(_dedr([s, s + 1e-9, 5.0]).abs().max()) == 0.0
    # and it is zero in float32 too, which is what the O(N^2) caller uses
    e32 = C._clash_pair_energy(torch.tensor([s, 0.5, 3.0], dtype=torch.float32),
                               float(C.K_CLASH), float(C.CLASH_SIGMA))
    assert float(e32.abs().max()) == 0.0


def test_it_goes_to_zero_quadratically_at_the_range():
    """E(sigma-h)/E(sigma-2h) -> 0.25. A linear switch-off would give 0.5."""
    s = C.CLASH_SIGMA
    ratios = [float(_e([s - h])) / float(_e([s - 2 * h])) for h in (1e-3, 1e-4, 1e-5)]
    assert ratios == sorted(ratios), (
        f"the switch-off ratio {ratios} is not increasing toward 0.25 as h falls")
    assert all(0.24 < x < 0.25 for x in ratios), f"ratio out of the quadratic range: {ratios}"
    assert ratios[-1] == pytest.approx(0.25, rel=2e-3), (
        f"E(sigma-h)/E(sigma-2h) = {ratios[-1]:.6f} at h = 1e-5: the slope at sigma is not "
        f"zero, so the range would carry a force step")


def test_it_diverges_as_the_pair_closes():
    r = torch.tensor([0.30, 0.15, 0.075, 0.01, 1e-3, 5e-4], dtype=torch.float64)
    e = C._clash_pair_energy(r, C.K_CLASH, C.CLASH_SIGMA)
    d = C._clash_pair_dedr(r, C.K_CLASH, C.CLASH_SIGMA)
    assert bool((e[1:] > e[:-1]).all()) and bool((d[1:].abs() > d[:-1].abs()).all())
    # E ~ 1/d^2 and |dE/dr| ~ 1/d^3, but only once sigma - d ~ sigma, so the exponents are
    # measured at 1e-3 and 5e-4 nm: halving the distance is a factor 4 and 8 there
    assert float(e[5] / e[4]) == pytest.approx(4.0, rel=1e-2)
    assert float(d[5] / d[4]) == pytest.approx(8.0, rel=1e-2)
    # and they grow without bound: no ceiling, unlike the linear spring's 150 kJ/mol/nm
    assert float(e[3]) > 1.0e6 and float(d[3].abs()) > 1.0e7
    assert float(e[5]) > 1.0e8 and float(d[5].abs()) > 1.0e10


def test_the_force_is_the_gradient_of_the_energy():
    r = torch.linspace(0.02, 0.62, 61, dtype=torch.float64)
    dedr = C._clash_pair_dedr(r, C.K_CLASH, C.CLASH_SIGMA)
    closed = -C.K_CLASH * C.CLASH_SIGMA ** 3 * (C.CLASH_SIGMA - r).clamp(min=0.0) \
        / r.clamp(min=1e-6) ** 3
    rel = float((dedr - closed).abs().max() / closed.abs().max())
    assert rel < 1e-12, (
        f"the autograd derivative is {rel:.3e} away from -k*sigma^3*(sigma-r)/r^3, which is "
        f"the closed form of the one energy in _clash_pair_energy")

    # and the vector force is -dE/dx of that same energy, by autograd on a real pair
    delta = torch.tensor([[0.17, -0.05, 0.03]], dtype=torch.float64)
    pos = torch.stack([delta[0], torch.zeros(3, dtype=torch.float64)]).reshape(1, 2, 3)
    pos = pos.detach().clone().requires_grad_(True)
    dist = torch.linalg.norm(pos[:, 0] - pos[:, 1], dim=-1)
    e = C._clash_pair_energy(dist, C.K_CLASH, C.CLASH_SIGMA)
    e.sum().backward()
    _e_pair, f = C._clash_pair_energy_force(pos[:, 0] - pos[:, 1], dist, C.K_CLASH, C.CLASH_SIGMA)
    f = f.detach()
    # F_i = -dE/dx_i, and the pair's two beads carry equal and opposite force
    assert float((f[0] - (-pos.grad[0, 0].detach())).abs().max()) < 1e-9 * float(f.abs().max())
    assert float((f[0] - pos.grad[0, 1].detach()).abs().max()) < 1e-9 * float(f.abs().max())


# ── 3. the four call sites agree ────────────────────────────────────────────────────────────
def _three_bead_system(L=6):
    """Residues 2 nm apart plus one deliberate contact, so every mask agrees.

    The cell-list paths exclude bead separations <= 2 and cg_energy_3bead excludes residue
    separations <= 1. With residues 2 nm apart the only pair either of them can see is the one
    placed on purpose: C4'(0) 0.20 nm from N(L-1).
    """
    pos = torch.zeros(1, 3 * L, 3, dtype=torch.float64)
    for i in range(L):
        base = torch.tensor([2.0 * i, 0.0, 0.0], dtype=torch.float64)
        pos[0, 3 * i] = base
        pos[0, 3 * i + 1] = base + torch.tensor([0.12, 0.37, 0.0], dtype=torch.float64)
        pos[0, 3 * i + 2] = (pos[0, 3 * i + 1]
                             + torch.tensor([0.07, 0.30, 0.13], dtype=torch.float64))
    # 0.20 nm below P(0) and nothing else inside sigma: C4'(0) is 0.4374 nm away, N(0)
    # 0.7705 nm, and every other bead is at least a residue away
    pos[0, 3 * (L - 1) + 2] = torch.tensor([0.0, 0.0, -0.20], dtype=torch.float64)
    return pos


def _energy_of(path, pos, pairs, weights):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    if path == "cg_energy_3bead":
        return float(C.cg_energy_3bead(pos, pairs, weights)[0])
    if path == "cg_energy_forces":
        return float(C.cg_energy_forces(pos, pairs, weights, cell_list=cl,
                                        force_cap=None)[0])
    if path == "cg_forces_explicit_batched":
        return float(C.cg_forces_explicit_batched(pos, pairs, weights, cell_list=cl)[0])
    if path == "cg_forces_explicit":
        return float(C.cg_forces_explicit(pos, pairs, weights, cell_list=cl)[0])
    raise AssertionError(path)


def _forces_of(path, pos, pairs, weights):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    if path == "cg_energy_forces":
        return C.cg_energy_forces(pos, pairs, weights, cell_list=cl, force_cap=None)[1]
    if path == "cg_forces_explicit_batched":
        return C.cg_forces_explicit_batched(pos, pairs, weights, cell_list=cl)[1]
    if path == "cg_forces_explicit":
        return C.cg_forces_explicit(pos, pairs, weights, cell_list=cl)[1]
    raise AssertionError(path)


def _clash_only(path, pos, pairs, weights, k, sigma):
    """E(k, sigma) - E(0, sigma): every other term cancels, so this is the clash term alone."""
    saved = (C.K_CLASH, C.CLASH_SIGMA)
    try:
        C.K_CLASH, C.CLASH_SIGMA = 0.0, sigma
        e0 = _energy_of(path, pos, pairs, weights)
        C.K_CLASH, C.CLASH_SIGMA = k, sigma
        e1 = _energy_of(path, pos, pairs, weights)
    finally:
        C.K_CLASH, C.CLASH_SIGMA = saved
    return e1 - e0


PATHS = ("cg_energy_3bead", "cg_energy_forces", "cg_forces_explicit_batched",
         "cg_forces_explicit")


def test_all_four_call_sites_compute_the_same_clash_energy():
    """cg_energy_3bead sums the full (N, N) matrix, so it counts every pair twice.

    That double count is pre-existing and is left alone here: it is the caller's summation
    convention, not the pair law, and this path has no production caller (cg_forces_3bead goes
    through cg_energy_forces). It is pinned rather than hidden, so a change to it shows up.
    """
    pos = _three_bead_system()
    no_pairs = torch.zeros((0, 2), dtype=torch.long)
    r = float(torch.linalg.norm(pos[0, 3 * 5 + 2] - pos[0, 0]))
    closed = float(C._clash_pair_energy(torch.tensor([r], dtype=torch.float64),
                                        C.K_CLASH, C.CLASH_SIGMA))
    got = {p: _clash_only(p, pos, no_pairs, None, C.K_CLASH, C.CLASH_SIGMA) for p in PATHS}
    assert closed > 1000.0
    for p, v in got.items():
        factor = 2.0 if p == "cg_energy_3bead" else 1.0
        # abs: the three force paths accumulate the total energy into a float32 total_E, which
        # is ~1.7e6 kJ/mol here, so the difference of the two runs carries ~0.25 kJ/mol of
        # float32 resolution whatever the clash term is
        assert v == pytest.approx(factor * closed, rel=1e-4, abs=0.5), (
            f"{p} returns {v} kJ/mol for a single {r:.2f} nm contact against "
            f"{factor * closed} ({factor:g}x {closed} from the shared potential) -- a second "
            f"copy of the law is in that path")


def test_all_four_call_sites_move_together_when_the_potential_moves():
    pos = _three_bead_system()
    no_pairs = torch.zeros((0, 2), dtype=torch.long)
    base = {p: _clash_only(p, pos, no_pairs, None, C.K_CLASH, C.CLASH_SIGMA) for p in PATHS}
    k2 = {p: _clash_only(p, pos, no_pairs, None, 1.5 * C.K_CLASH, C.CLASH_SIGMA) for p in PATHS}
    s2 = {p: _clash_only(p, pos, no_pairs, None, C.K_CLASH, 1.5 * C.CLASH_SIGMA) for p in PATHS}
    retired = None
    saved = C.CLASH_DIST
    try:
        C.CLASH_DIST = 0.45
        retired = {p: _clash_only(p, pos, no_pairs, None, C.K_CLASH, C.CLASH_SIGMA)
                   for p in PATHS}
    finally:
        C.CLASH_DIST = saved
    for p in PATHS:
        assert k2[p] == pytest.approx(1.5 * base[p], rel=1e-3, abs=0.5), (
            f"{p} did not scale with K_CLASH ({k2[p]} against 1.5 x {base[p]})")
        assert s2[p] != base[p], f"{p} ignored a change to CLASH_SIGMA"
        assert retired[p] == pytest.approx(base[p], rel=1e-12), (
            f"{p} moved when CLASH_DIST changed: the retired range is being read again")


def test_the_force_callers_see_the_same_pair_force():
    """The three force paths must agree with -dE/dx of the shared potential too."""
    pos = _three_bead_system()
    no_pairs = torch.zeros((0, 2), dtype=torch.long)
    r_vec = pos[0, 0] - pos[0, 3 * 5 + 2]
    r = float(torch.linalg.norm(r_vec))
    _e_pair, f_pair = C._clash_pair_energy_force(
        r_vec, torch.tensor([r], dtype=torch.float64), C.K_CLASH, C.CLASH_SIGMA)
    for p in ("cg_energy_forces", "cg_forces_explicit_batched", "cg_forces_explicit"):
        saved = C.K_CLASH
        try:
            C.K_CLASH = 0.0
            f0 = _forces_of(p, pos, no_pairs, None)
            C.K_CLASH = saved
            f1 = _forces_of(p, pos, no_pairs, None)
        finally:
            C.K_CLASH = saved
        f_clash = (f1 - f0)[0]
        assert float(f_clash[0].norm()) == pytest.approx(float(f_pair[0].norm()), rel=1e-4), (
            f"{p} clash force {float(f_clash[0].norm())} against the shared potential's "
            f"{float(f_pair[0].norm())} kJ/mol/nm")
        assert float(f_clash[3 * 5 + 2].norm()) == pytest.approx(float(f_pair[0].norm()),
                                                                 rel=1e-4)
        assert float(f_clash.norm()) == pytest.approx(2.0 ** 0.5 * float(f_pair[0].norm()),
                                                      rel=1e-5), (
            f"{p} put force somewhere other than on the pair's two beads")


def test_the_one_bead_path_uses_the_same_potential():
    """cg_energy has its own neighbour list, but not its own law."""
    L = 7
    pos = torch.zeros(1, L, 3, dtype=torch.float64)
    for i in range(L):
        pos[0, i] = torch.tensor([0.59 * i, 0.0, 0.0], dtype=torch.float64)
    # bead 3 is a bead-index gap of 3, so it is inside the neighbour list's mask
    pos[0, 3] = torch.tensor([0.25, 0.0, 0.0], dtype=torch.float64)
    no_pairs = torch.zeros((0, 2), dtype=torch.long)
    C._clash_nlist.nlist = None          # the list is cached across calls; force a rebuild
    saved = (C.K_CLASH, C.CLASH_SIGMA)
    try:
        C.K_CLASH = 0.0
        e0 = float(C.cg_energy(pos, no_pairs, None))
        C.K_CLASH = saved[0]
        e1 = float(C.cg_energy(pos, no_pairs, None))
    finally:
        C.K_CLASH = saved[0]
    pi, pj = C._clash_nlist.get(pos)
    flat = pos.reshape(1, L, 3)
    d = torch.linalg.norm(flat[:, pi] - flat[:, pj], dim=-1)
    closed = float(C._clash_pair_energy(d, saved[0], saved[1]).sum())
    assert closed > 1.0
    assert e1 - e0 == pytest.approx(closed, rel=1e-6), (
        f"cg_energy returns {e1 - e0} kJ/mol of clash energy against {closed} from the shared "
        f"potential over its own neighbour list")


# ── 4. the parameters are measured ──────────────────────────────────────────────────────────
def test_the_range_is_the_databases_own_lower_edge():
    assert C.CLASH_SIGMA == pytest.approx(DB_PP_MIN, abs=1e-12), (
        f"CLASH_SIGMA is {C.CLASH_SIGMA}; the minimum of the {DB_PAIRS} non-bonded P-P "
        f"distances in the database is {DB_PP_MIN} nm")
    assert C.CLASH_SIGMA > C.CLASH_DIST, (
        "the live range is still at or below the retired 0.300 nm, i.e. below every distance "
        "the database contains, where the term can never fire")


def test_the_stiffness_is_the_boltzmann_inversion_of_the_two_lowest_bins():
    (a0, a1, n0), (b0, b1, n1) = DB_BINS[0], DB_BINS[1]
    v0 = 4.0 / 3.0 * math.pi * (a1 ** 3 - a0 ** 3)
    v1 = 4.0 / 3.0 * math.pi * (b1 ** 3 - b0 ** 3)
    mid = 0.5 * (a0 + a1)
    du = KBT * math.log((n1 / v1) / (n0 / v0))
    shape = 0.5 * (C.CLASH_SIGMA - mid) ** 2 * (C.CLASH_SIGMA / mid) ** 2
    k_criterion = du / shape
    assert k_criterion == pytest.approx(20013.4, rel=1e-4), (
        f"the criterion arithmetic gives {k_criterion:.1f}; the recorded value is 20013.4")
    assert C.K_CLASH == pytest.approx(k_criterion, rel=0.01), (
        f"K_CLASH is {C.K_CLASH} against a criterion value of {k_criterion:.1f} kJ/mol/nm^2 "
        f"(1 bin in [0.36,0.40), 6 in [0.40,0.45), 60/8 further out)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
