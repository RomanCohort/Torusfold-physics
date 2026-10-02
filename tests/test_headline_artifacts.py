# -*- coding: utf-8 -*-
"""The committed artifacts must keep decoding from the shipped viewer, and the numbers
the viewer displays must keep coming back.

WHY A TEST AND NOT A DOC. `artifacts/2013nt/` exists because the demo sequence and the
predicted structure were not in the repository, which is why
`docs/archive/pipeline_audit_2026-09-13.md:1022` records that the full Level 0 path was never
executed end to end. A directory that appears once and then rots is the same failure
with a later date on it. This test is the thing that notices.

It calls `scripts/verify_headline.py`'s own functions rather than re-deriving anything
here, so there is exactly ONE implementation of each check and no second copy to drift.
(The call is in-process on purpose: a subprocess would need piped stdio, which some
confined environments refuse, and a check that cannot run where it is needed is not a
check.)

The expectation is pinned losslessly, not by a count:

  * `rep.failed` empty -- every panel entry marked `reproducible` reproduced, and no IBI
    coordinate failed that was not already known to;
  * `rep.known` equals the set of runs whose worst coordinate is `intra_pc` -- so a NEW
    non-reproducing coordinate fails this test instead of hiding inside a number that
    was allowed to be non-zero.

`intra_pc` is off by 2.2-4.2x in all twelve committed IBI runs and is deliberately not
reproducible from the committed histograms; the note in verify_headline.py carries the
measurement. That is the only reason `allow_known` is used here, and if it is ever
fixed, this test should lose the flag rather than the flag become permanent.

Needs numpy. scipy is optional (one panel entry is skipped without it).
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

verify_headline = pytest.importorskip("verify_headline")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def checked():
    if not (REPO / "artifacts" / "2013nt" / "isrnaclong_final.pdb").exists():
        pytest.skip("artifacts/ is not present in this checkout")
    rep = verify_headline.Report(verbose=False)
    pdb_text, quality = verify_headline.check_artifacts(rep)
    if pdb_text is None:
        pytest.fail("the viewer payload could not be decoded; artifacts cannot be checked")
    verify_headline.check_panel(rep, pdb_text, quality)
    verify_headline.check_ibi(rep, allow_known=True)
    return rep, pdb_text, quality


def test_no_unexpected_failure(checked):
    rep, _, _ = checked
    assert not rep.failed, "checks that used to reproduce no longer do: " + repr(rep.failed)


def test_known_set_is_exactly_intra_pc(checked):
    """Nothing new stopped reproducing. The known set is the IBI runs, all via intra_pc."""
    rep, _, _ = checked
    runs = sorted(d.name for d in (REPO / "results").glob("ibi_*")
                  if (d / "manifest.json").exists())
    assert sorted(rep.known) == runs, (
        "the set of runs failing on a known coordinate changed: "
        f"known={sorted(rep.known)} runs={runs}")


def test_artifacts_decode_from_the_viewer(checked):
    """Bit-for-bit: the committed PDB is what the shipped viewer renders."""
    _, pdb_text, _ = checked
    assert pdb_text.startswith("REMARK")
    assert len([l for l in pdb_text.splitlines() if l.startswith("ATOM")]) == 42831


def test_panel_numbers_that_claim_to_reproduce_do(checked):
    """Every quality.json entry marked `reproducible` must have passed a real check."""
    rep, pdb_text, quality = checked
    claimed = [e["metric"] for e in quality["entries"] if e["verdict"] == "reproducible"]
    assert "pair_satisfaction" not in claimed, \
        "pair_satisfaction does not reproduce (47.2% vs the panel's 100.0%); " \
        "quality.json must not mark it reproducible"
    assert claimed == ["atoms", "sequence_length", "bsj_closure", "bond_rmsd"]
    assert rep.checked >= len(claimed)


def test_extract_script_reproduces_the_committed_bytes():
    """The decode is deterministic: the committed files ARE the viewer's payload."""
    extract = pytest.importorskip("extract_viewer_payload")
    import json
    html = (REPO / "docs" / "circrna_3d_viewer.html").read_text(encoding="utf-8", errors="replace")
    pdb_text, payload_sha = extract.decode_viewer(html)
    prov = json.loads((REPO / "artifacts" / "2013nt" / "provenance.json")
                      .read_text(encoding="utf-8"))
    assert payload_sha == prov["payload_b64_sha256"]
    assert extract.sha256_text(pdb_text) == prov["pdb_sha256"]
    assert pdb_text == (REPO / "artifacts" / "2013nt" / "isrnaclong_final.pdb")\
        .read_text(encoding="utf-8")
    seq = extract.sequence_of(pdb_text)
    assert seq == (REPO / "artifacts" / "2013nt" / "sequence.txt").read_text(encoding="utf-8").strip()
    assert extract.sha256_text(seq + "\n") == prov["sequence_sha256"]
    assert len(seq) == 2013 and set(seq) <= set("ACGU")
