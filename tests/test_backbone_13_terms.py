"""The four backbone pairs that had no term: they exist, they are the right four, they do not
wrap onto the closure, and every path's force is the gradient of its own energy.

scripts/audit_intra_residue_pairs.py enumerates every bead pair at a sequence gap of 1, 2 or 3 and
finds that the 3-bead nucleotide (P-C4'-N9/N1, two bonds) plus the excluded volume's |i-j| <= 2
mask left four pairs with no energy at all. An earlier audit HAND-LISTED six classes and so found
only three of the four; the enumeration is what found N9/N1(i)-C4'(i+1). This file is the
regression on the fix.

Isolation uses the same technique test_clash_single_potential.py does -- switch one constant to
zero and difference -- plus the per-term decomposition in scripts/cg_force_terms.py, whose
energies are float64 and therefore exact, unlike the float32 running totals the force paths
accumulate.
"""
import math
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torusfold.scheme2.torch_cgsim as C     # noqa: E402

@pytest.fixture(autouse=True)
def _allow_alternate_fields():
    """This module characterises entry points that are NOT the production field.

    cg_energy / cg_forces_autograd (1 bead per residue), cg_energy_3bead and the two explicit
    paths all raise by default, because they share this module's constant names while computing a
    different potential. Saying so once here is what the opt-in exists for; a reader of this file
    can see which field each test is about.
    """
    saved = C.ALLOW_ALTERNATE_FIELDS
    C.ALLOW_ALTERNATE_FIELDS = True
    yield
    C.ALLOW_ALTERNATE_FIELDS = saved

import torch                                  # noqa: E402
import cg_force_terms as FT                   # noqa: E402

KBT = 2.494     # kJ/mol at 300 K, the kBT the shipped k = kBT/sd^2 table is built on
ATOM = ("P", "C4'", "N9/N1")

# the four terms: module constant -> (decomposition name, atom of the lower bead, atom of the
# higher bead, residue offset, how many there are)
TERMS = {
    "K_INTRA_PN": ("intra P-N9/N1", 0, 2, 0, "L"),
    "K_LINK_CP": ("link C4'-P", 1, 0, 1, "L-1"),
    "K_LINK_NP": ("link N9/N1-P", 2, 0, 1, "L-1"),
    "K_LINK_NC": ("link N9/N1-C4'", 2, 1, 1, "L-1"),
}


def _chain(L=9):
    """A straight chain: P(i) at 0.59 i along x, C4' and N9/N1 offset off-axis.

    Deliberately not closed: residue 0 is 4.72 nm from residue L-1, so a term that wrapped onto
    the closure would contribute a huge, visible number rather than a plausible one.
    """
    pos = torch.zeros(1, 3 * L, 3, dtype=torch.float64)
    for i in range(L):
        base = torch.tensor([0.59 * i, 0.0, 0.0], dtype=torch.float64)
        pos[0, 3 * i] = base
        pos[0, 3 * i + 1] = base + torch.tensor([0.12, 0.32, 0.0], dtype=torch.float64)
        pos[0, 3 * i + 2] = pos[0, 3 * i + 1] + torch.tensor(
            [0.07, 0.26, 0.13], dtype=torch.float64)
    pairs = torch.tensor([[0, 4], [1, 6]], dtype=torch.long)
    w = torch.ones(pairs.shape[0], dtype=torch.float64)
    return pos, pairs, w


def _cl(p):
    c = C.GPUCellList(cell_size=1.5)
    c.build(p)
    return c


def _decomp(pos, pairs, w):
    F, e = FT.term_energies_forces(pos, pairs, w, cell_list=_cl(pos))
    return F, e


def _total(path, pos, pairs, w):
    if path == "cg_energy_3bead":
        return float(C.cg_energy_3bead(pos, pairs, w)[0])
    if path == "cg_energy_forces":
        return float(C.cg_energy_forces(pos, pairs, w, cell_list=_cl(pos), force_cap=None)[0])
    if path == "cg_forces_explicit_batched":
        return float(C.cg_forces_explicit_batched(pos, pairs, w, cell_list=_cl(pos))[0])
    if path == "cg_forces_explicit":
        return float(C.cg_forces_explicit(pos, pairs, w, cell_list=_cl(pos))[0])
    raise KeyError(path)


PATHS = ("cg_energy_3bead", "cg_energy_forces", "cg_forces_explicit_batched",
         "cg_forces_explicit")


def _isolate(path, pos, pairs, w, name):
    """Energy of one term on one path: zero its constant and difference."""
    saved = getattr(C, name)
    try:
        base = _total(path, pos, pairs, w)
        setattr(C, name, 0.0)
        off = _total(path, pos, pairs, w)
    finally:
        setattr(C, name, saved)
    return base - off


def _expected(pos, L, a1, a2, dr, r0):
    """0.5 * k * sum (d - r0)^2 over the class, computed from the geometry directly."""
    n = L if dr == 0 else L - 1
    tot = 0.0
    for i in range(n):
        d = float(torch.linalg.norm(
            pos[0, 3 * i + a1] - pos[0, 3 * (i + dr) + a2]))
        tot += (d - r0) ** 2
    return tot


# ── the four terms exist, on every path ─────────────────────────────────────────────────────
def test_every_force_path_carries_all_four_terms():
    """Each term isolated on each path, against 0.5*k*sum(d-r0)^2 from the geometry.

    The tolerance is loose on purpose: the four force paths accumulate their total energy into a
    float32 tensor that is order 1e5 kJ/mol at this geometry, so a difference of two such totals
    carries ~0.015 kJ/mol of quantisation whatever the term is. The decomposition test below
    checks the same four energies exactly.
    """
    pos, pairs, w = _chain()
    L = 9
    for name, (_dn, a1, a2, dr, _n) in TERMS.items():
        k = getattr(C, name)
        r0 = getattr(C, "BOND_" + name[2:])
        want = 0.5 * k * _expected(pos, L, a1, a2, dr, r0)
        assert want > 1.0, f"{name} would be zero at this geometry, so the test would be vacuous"
        for path in PATHS:
            got = _isolate(path, pos, pairs, w, name)
            assert got == pytest.approx(want, rel=1e-3, abs=0.05), (
                f"{path} {name}: isolated {got} against the closed form {want}")


def test_the_decomposition_reports_the_same_four_energies_exactly():
    """scripts/cg_force_terms.py is what every force-attribution script reads, so a term missing
    from it silently under-reports the field. It was missing all four before this change."""
    pos, pairs, w = _chain()
    L = 9
    _F, e = _decomp(pos, pairs, w)
    for name, (dn, a1, a2, dr, _n) in TERMS.items():
        k = getattr(C, name)
        r0 = getattr(C, "BOND_" + name[2:])
        want = 0.5 * k * _expected(pos, L, a1, a2, dr, r0)
        assert dn in e, f"the decomposition has no {dn!r} term; it has {sorted(e)}"
        assert e[dn] == pytest.approx(want, rel=1e-9, abs=1e-9), (
            f"decomposition {dn} is {e[dn]} against the closed form {want}")


# ── they stop at the closure ────────────────────────────────────────────────────────────────
def test_the_two_link_terms_cover_exactly_L_minus_one_links():
    """The constants are fitted on a linear database, which has no closure link.

    Wrapping them onto residue 0 was tried and reverted: on a straight chain 4.72 nm long it added
    1.49e5 kJ/mol to tests/test_force_gradcheck.py's geometry. It would also put a ~1e5 kJ/mol
    barrier against the compaction K_BSJ exists to drive, and the pipeline starts extended.

    The check is the term count. _expected sums L-1 links for the two classes that cross a link and
    L for the one that does not; a wrapped implementation would sum L and fail here.
    """
    pos, pairs, w = _chain()
    L = 9
    _F, e = _decomp(pos, pairs, w)
    for name, (dn, a1, a2, dr, _n) in TERMS.items():
        k = getattr(C, name)
        r0 = getattr(C, "BOND_" + name[2:])
        internal = 0.5 * k * _expected(pos, L, a1, a2, dr, r0)
        assert e[dn] == pytest.approx(internal, rel=1e-9, abs=1e-9), (
            f"{dn} carries {e[dn]} against {internal} for the internal links only")
        if dr == 1:
            # the same class with the closure link added -- a much larger number on this straight
            # chain, so the check above can tell wrapping from not wrapping
            d_close = float(torch.linalg.norm(pos[0, 3 * (L - 1) + a1] - pos[0, a2]))
            wrapped = internal + 0.5 * k * (d_close - r0) ** 2
            assert wrapped > internal + 1.0, (
                f"{dn}: the closure link contributes {wrapped - internal} on this geometry, too "
                f"little for this test to distinguish the two conventions")


# ── nothing short is left unguarded ─────────────────────────────────────────────────────────
def test_no_pair_at_gap_one_or_two_is_left_unguarded():
    """The complement check, and the reason it is a test rather than a paragraph.

    Every bead pair with |i-j| <= 2 must have a bonded term. The four here are the complete
    complement; the earlier hand-written audit table was not, and missed one of them.
    """
    L = 9
    nb = 3 * L
    guarded = set()
    for i in range(L):
        guarded |= {(3 * i, 3 * i + 1), (3 * i + 1, 3 * i + 2), (3 * i, 3 * i + 2)}
    for i in range(L - 1):
        guarded |= {(3 * i + 1, 3 * i + 3), (3 * i + 2, 3 * i + 3), (3 * i + 2, 3 * i + 4)}
    short = {(a, b) for a in range(nb) for b in range(a + 1, nb) if b - a <= 2}
    left = short - guarded
    assert not left, (
        f"bead pairs with |i-j| <= 2 and no bonded term: {sorted(left)}. The excluded volume "
        f"skips |i-j| <= 2, so nothing in the field assigns these an energy.")


def test_the_three_closure_link_pairs_are_covered_by_the_excluded_volume():
    """The decision not to wrap has to leave the closure guarded by something.

    The closure link connects residue L-1 to residue 0, whose bead indices are 3L-3, 3L-2 and
    3L-1 apart -- all >= 3 for any L >= 2, so the excluded volume does apply to them and they are
    not the same hole at a different index.
    """
    for L in (2, 9, 27, 100):
        gaps = sorted({3 * L - 3, 3 * L - 2, 3 * L - 1})
        assert gaps[0] >= 3, f"L={L}: a closure link has bead gap {gaps[0]}, inside the skipped range"


# ── the constants are the measurement ───────────────────────────────────────────────────────
def test_the_four_constants_are_the_database_measurement():
    """Pinned, so a retune has to arrive with a new measurement.

    scripts/audit_intra_residue_pairs.py, 126 gap-free chains, k = kBT/sd^2:
        P-N9/N1            mean 0.5370  sd 0.0374  ->   1785.9   floor  66.7
        C4'(i)-P(i+1)      mean 0.3800  sd 0.0161  ->   9574.4   floor 154.5
        N9/N1(i)-P(i+1)    mean 0.5457  sd 0.0213  ->   5477.7   floor 116.9
        N9/N1(i)-C4'(i+1)  mean 0.6303  sd 0.0807  ->    383.0   floor  30.9
    """
    pinned = {
        "K_INTRA_PN": (1785.9, 0.5370, 0.0374),
        "K_LINK_CP": (9574.4, 0.3800, 0.0161),
        "K_LINK_NP": (5477.7, 0.5457, 0.0213),
        "K_LINK_NC": (383.0, 0.6303, 0.0807),
    }
    for name, (k, mean, sd) in pinned.items():
        assert getattr(C, name) == pytest.approx(k, rel=1e-4), name
        r0 = getattr(C, "BOND_" + name[2:])
        assert r0 == pytest.approx(mean, abs=1e-9), f"{name} r0 is not the database mean"
        # the pinned sd is the printed rounding of the spread the constant was built from, so
        # recover it from k rather than asserting equality against four decimal places
        assert round(math.sqrt(KBT / k), 4) == pytest.approx(sd, abs=1e-4), (
            f"{name} implies sd = {math.sqrt(KBT / k):.6f}, pinned as {sd}")
        assert math.isfinite(k) and k > 0.0


def test_no_new_force_floor_exceeds_the_p_c4_bond_floor():
    """sqrt(k*kBT) for each new term against 236.3 kJ/mol/nm for P-C4', the stiffest bond."""
    for name in TERMS:
        floor = math.sqrt(getattr(C, name) * KBT)
        assert 0.0 < floor < 236.3, f"{name} floor {floor}"


# ── the force is the gradient of the energy ─────────────────────────────────────────────────
def test_each_of_the_four_terms_is_its_own_gradient():
    """Per-term finite difference through the float64 decomposition.

    The four paths accumulate into float32 totals, so their full-path finite difference is
    quantisation-limited; the decomposition returns float64 per-term energies and is exact. A sign
    error in any one of the four shows up here.
    """
    pos, pairs, w = _chain()
    h = 1e-5
    F, _e = _decomp(pos, pairs, w)
    worst = {}
    for name, (dn, _a1, _a2, _dr, _n) in TERMS.items():
        w_ = 0.0
        for i in range(pos.shape[1]):
            for c in range(3):
                hi = pos.clone(); hi[0, i, c] += h
                lo = pos.clone(); lo[0, i, c] -= h
                fd = -(_decomp(hi, pairs, w)[1][dn] - _decomp(lo, pairs, w)[1][dn]) / (2 * h)
                w_ = max(w_, abs(fd - F[dn][i, c]))
        worst[dn] = w_
    bad = {k: v for k, v in worst.items() if v > 1e-3}
    assert not bad, (
        f"finite-difference force disagrees with the analytic force for {bad}; each of these four "
        f"terms must be the gradient of its own energy (worst per term: {worst})")
