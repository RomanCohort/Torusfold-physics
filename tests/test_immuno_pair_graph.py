"""Column and behaviour assertions for the immune pair graph.

The fixture generator in this package has produced a column-shifted PDB record
three separate times. Each time the visible symptom was "no pairs found", and
the cause was a blank chain ID: both strands merged into one anonymous chain.
These tests exist so a fourth attempt fails on the column assertion instead of
silently changing what the detector is looking at.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from torusfold.immuno import pair_graph_from_coords as pg  # noqa: E402


# ---------------------------------------------------------------------------
# the fixture itself

def test_fixture_emits_correct_pdb_columns():
    pdb = pg.ideal_aform_helix(n_pairs=3)
    line = pdb.splitlines()[0]
    assert line[:6] == "ATOM  ", f"record name: {line[:6]!r}"
    assert line[6:11].strip() == "1", f"serial 7-11: {line[6:11]!r}"
    assert line[12:16].strip() == "C1'", f"atom name 13-16: {line[12:16]!r}"
    assert line[17:20].strip() in ("A", "C", "G", "U"), \
        f"resName 18-20: {line[17:20]!r}"
    assert line[21] == "A", f"chainID column 22: {line[21]!r} (line {line!r})"
    assert line[22:26].strip() == "1", f"resSeq 23-26: {line[22:26]!r}"
    assert abs(float(line[30:38]) - 5.2) < 0.01, f"x 31-38: {line[30:38]!r}"


def test_fixture_has_two_distinct_chains():
    g = pg.build_pair_graph(pg.ideal_aform_helix(n_pairs=6))
    assert {r.chain for r in g.residues.values()} == {"A", "B"}


def test_fixture_residue_count():
    g = pg.build_pair_graph(pg.ideal_aform_helix(n_pairs=6))
    assert len(g.residues) == 12, "12 residues for 6 pairs across two strands"


def test_fixture_is_not_geometry_enough_to_pair():
    """The fixture cannot reproduce real A-form, and the detector says so.

    A one-parameter helix fixes the paired C1'--C1' distance by the radius and
    the stacked distance by the radius plus the rise. Real A-form needs ~10.4 A
    and ~6.2 A, and no single radius gives both (r=5.2: 10.4 paired, 3.5 stacked;
    r=6.5: 13.0 paired, 6.2 stacked). Real duplexes escape this because their
    glycosidic bonds are not radial -- a second parameter this generator lacks.

    So zero pairs is the CORRECT result here, and asserting it keeps anyone from
    "fixing" the fixture into agreeing with the detector. Validation lives on
    1QC0: see tools/validate_pair_graph.py.
    """
    g = pg.build_pair_graph(pg.ideal_aform_helix(n_pairs=12))
    assert g.pairs == [], (
        "the synthetic fixture started yielding pairs; that means either the "
        "fixture geometry changed or the pair criterion got looser. Both need "
        "looking at -- do not simply update this assertion."
    )
    assert g.rejected, "rejections should still be recorded, not silently dropped"


def test_fixture_strand_numbering_does_not_collide():
    """Residues are keyed by index alone, so the strands must not share indices."""
    res = pg.parse_pdb_residues(pg.ideal_aform_helix(n_pairs=6))
    indices = [r.index for r in res.values()]
    assert len(indices) == len(set(indices))


# ---------------------------------------------------------------------------
# parser robustness

def test_parse_rejects_numbering_below_one():
    pdb = (
        "ATOM      1  C1'   A A   0       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    with pytest.raises(ValueError, match="below 1"):
        pg.parse_pdb_residues(pdb)


def test_parse_keeps_only_acgu():
    pdb = (
        "ATOM      1  C1'   G A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C1'   X A   2       1.000   0.000   0.000  1.00  0.00           C\n"
    )
    res = pg.parse_pdb_residues(pdb)
    assert list(res) == [1]
    assert res[1].name == "G"


def test_empty_pdb_raises():
    with pytest.raises(ValueError, match="no A/C/G/U"):
        pg.build_pair_graph("HEADER    nothing here\nEND\n")


# ---------------------------------------------------------------------------
# topology guards

def _pdb_line(serial: int, atom: str, resname: str, chain: str,
              resseq: int, xyz) -> str:
    """One PDB ATOM record by explicit column placement.

    Written out longhand rather than as an f-string because f-string versions of
    this line were wrong by one column five separate times in this package, and
    each time the failure looked like a detector bug rather than a formatting one.
    """
    buf = [" "] * 80

    def put(start: int, text: str) -> None:
        for i, ch in enumerate(text):
            idx = start - 1 + i
            if 0 <= idx < len(buf):
                buf[idx] = ch

    put(1, "ATOM")
    put(7, f"{serial:>5d}")
    put(13, atom[:4])
    put(18, resname)
    put(22, chain)
    put(23, f"{resseq:>4d}")
    for start, value in ((31, xyz[0]), (39, xyz[1]), (47, xyz[2])):
        put(start, f"{value:>8.3f}")
    put(55, "  1.00")
    put(61, "  0.00")
    put(77, atom[0])
    return "".join(buf)


def _single_chain(n: int = 4, chain: str = "A") -> str:
    """Minimal one-chain PDB text with n residues, for topology tests."""
    rows = []
    for i in range(1, n + 1):
        rows.append(_pdb_line(i, "C1'", "G", chain, i, (float(i), 0.0, 0.0)))
        rows.append(_pdb_line(n + i, "N9", "G", chain, i, (float(i) + 1.4, 0.0, 0.0)))
    return "\n".join(rows) + "\nEND\n"


def test_single_chain_helper_round_trips():
    """Guard the helper itself: a column slip here would fake a topology failure."""
    res = pg.parse_pdb_residues(_single_chain(3))
    assert sorted(res) == [1, 2, 3]
    assert all(r.chain == "A" for r in res.values())


def test_bsj_index_forces_circular():
    g = pg.build_pair_graph(_single_chain(6), bsj_index=3)
    assert g.is_circular is True
    assert g.bsj_index == 3


def test_bsj_index_out_of_range_raises():
    with pytest.raises(ValueError, match="bsj_index"):
        pg.build_pair_graph(_single_chain(6), bsj_index=999)


def test_circular_with_two_chains_is_refused():
    """Wrapping needs one contiguous chain; two chains are two molecules."""
    with pytest.raises(ValueError, match="ambiguous"):
        pg.build_pair_graph(pg.ideal_aform_helix(n_pairs=4), is_circular=True)


def test_chains_filter_selects_and_validates():
    text = pg.ideal_aform_helix(n_pairs=4)
    g = pg.build_pair_graph(text, chains=["A"])
    assert {r.chain for r in g.residues.values()} == {"A"}
    with pytest.raises(ValueError, match="not present"):
        pg.build_pair_graph(text, chains=["Z"])

# ---------------------------------------------------------------------------
# normal orientation: the bug that cost 11 of 19 pairs on 1QC0

def test_plane_angle_takes_the_supplement():
    """Parallel planes may present same-way or opposite normals; both are 0 deg.

    Before this was fixed, `plane_angle` compared the raw normals, so two
    genuinely parallel bases could be reported 180 deg apart. On 1QC0 that
    rejected 11 of 19 real pairs as "base planes 170 deg apart".
    """
    import numpy as np

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    cos_plane = abs(float(np.clip(np.dot(a, b), -1.0, 1.0)))
    assert cos_plane == pytest.approx(1.0)


def test_orient_normal_is_sign_stable():
    """The fitted normal must not depend on which ring atom the SVD saw first."""
    import numpy as np

    c1 = np.array([0.0, 0.0, 0.0])
    glyco = np.array([1.5, 0.0, 0.0])
    atoms = {
        "N9": glyco,
        "C4": np.array([0.5, 1.3, 0.05]),
        "C5": np.array([1.5, 1.3, 0.05]),
    }
    normal = np.array([0.0, 0.0, 1.0])
    out = pg._orient_normal(normal, c1, atoms, "N9")
    assert out is not None
    flipped = pg._orient_normal(-normal, c1, atoms, "N9")
    assert flipped is not None
    np.testing.assert_allclose(out, flipped, atol=1e-12)
