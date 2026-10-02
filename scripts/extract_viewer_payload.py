# -*- coding: utf-8 -*-
"""Recover the delivered 2013 nt structure from the shipped viewer, into artifacts/.

WHY THIS EXISTS. `docs/circrna_3d_viewer.html` is the only Level-0 artifact that
has ever been in the repository, and it carries the whole structure: the page is
self-contained, so the 42,831-atom PDB it renders is inlined as a gzip+base64
payload (`var pdbB64 = "H4sI..."`). The run that produced it is not in this
repository, so the payload is the only surviving copy of the delivered model.

This script inverts that packing. It is the provenance of every file under
`artifacts/2013nt/`: those files are not a re-run, they are the viewer's own
bytes, decoded. Run it again and it must reproduce them byte for byte -- that is
the whole point, so the script fails loudly rather than writing something close.

    python scripts/extract_viewer_payload.py            # verify / rewrite artifacts
    python scripts/extract_viewer_payload.py --check    # verify only, write nothing

Output:
    artifacts/2013nt/isrnaclong_final.pdb   42,831 atoms / 2,013 residues
    artifacts/2013nt/sequence.txt           the sequence read off that structure
    artifacts/2013nt/provenance.json        what was decoded, from what, when

The sequence is READ OUT of the structure rather than taken from anywhere else,
because nothing else has it: `sequence.txt` at the repository root is
git-ignored ("private sequence"), and the audit note at
`docs/archive/pipeline_audit_2026-09-13.md:1022` records that this is why the full Level
0 path was never executed end to end. Recovering it here is what makes the demo
runnable by a third party at all. It is not a new release of anything.
"""
import argparse
import base64
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VIEWER = REPO / "docs" / "circrna_3d_viewer.html"
OUTDIR = REPO / "artifacts" / "2013nt"

# The four residue names a PDB may use for a ribonucleotide, mapped to one letter.
BASE_OF = {"A": "A", "U": "U", "G": "G", "C": "C",
           "RA": "A", "RU": "U", "RG": "G", "RC": "C",
           "DA": "A", "DT": "U", "DG": "G", "DC": "C"}

PAYLOAD_RE = re.compile(r'var\s+pdbB64\s*=\s*"([A-Za-z0-9+/=]+)"')


def decode_viewer(html_text):
    """(pdb_text, sha256_of_payload_b64) from the viewer page."""
    m = PAYLOAD_RE.search(html_text)
    if not m:
        raise SystemExit(f"{VIEWER}: no 'var pdbB64 = \"...\"' payload found")
    b64 = m.group(1)
    raw = base64.b64decode(b64)
    if raw[:2] != b"\x1f\x8b":
        raise SystemExit(f"{VIEWER}: payload is not gzip (magic {raw[:2]!r})")
    pdb = gzip.decompress(raw).decode("utf-8")
    return pdb, hashlib.sha256(b64.encode("ascii")).hexdigest()


def sequence_of(pdb_text):
    """The one-letter sequence, in file order, from the ATOM records."""
    seq, seen = [], set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (line[21], line[22:27])
        if key in seen:
            continue
        seen.add(key)
        name = line[17:20].strip()
        if name not in BASE_OF:
            raise SystemExit(f"residue {key} has name {name!r}, not a ribonucleotide")
        seq.append(BASE_OF[name])
    return "".join(seq)


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the committed artifacts instead of writing them")
    args = ap.parse_args()

    if not VIEWER.exists():
        raise SystemExit(f"{VIEWER} not found")
    pdb_text, payload_sha = decode_viewer(VIEWER.read_text(encoding="utf-8", errors="replace"))

    n_atom = sum(1 for l in pdb_text.splitlines() if l.startswith(("ATOM", "HETATM")))
    seq = sequence_of(pdb_text)
    remark = next((l for l in pdb_text.splitlines() if l.startswith("REMARK")), "")

    print(f"viewer        : {VIEWER.relative_to(REPO)}")
    print(f"payload sha256: {payload_sha}")
    print(f"pdb           : {len(pdb_text)} chars, {n_atom} atoms, {len(seq)} residues")
    print(f"provenance    : {remark.strip()}")
    print(f"sequence      : {seq[:40]}... ({len(seq)} nt)")

    files = {
        OUTDIR / "isrnaclong_final.pdb": pdb_text,
        OUTDIR / "sequence.txt": seq + "\n",
        OUTDIR / "provenance.json": json.dumps({
            "source_file": "docs/circrna_3d_viewer.html",
            "source_payload": "var pdbB64 (gzip + base64)",
            "payload_b64_sha256": payload_sha,
            "pdb_sha256": sha256_text(pdb_text),
            "sequence_sha256": sha256_text(seq + "\n"),
            "n_atoms": n_atom,
            "n_residues": len(seq),
            "pdb_remark_1": remark.strip(),
            "recovered_by": "scripts/extract_viewer_payload.py",
            "note": ("Decoded, not re-run. These are the bytes the shipped viewer renders; "
                     "the run that produced them is not in this repository. "
                     "docs/REPRODUCTION_RESOURCES.md, section 'Provenance of the delivered model'."),
        }, indent=2, ensure_ascii=False) + "\n",
    }

    bad = 0
    for path, text in files.items():
        new = text.encode("utf-8")
        if path.exists():
            old = path.read_bytes()
            if old == new:
                print(f"  ok       {path.relative_to(REPO)}  ({len(new)} B, unchanged)")
                continue
            if args.check:
                print(f"  STALE    {path.relative_to(REPO)}  "
                      f"({len(old)} B on disk vs {len(new)} B decoded)")
                bad += 1
                continue
            print(f"  rewrite  {path.relative_to(REPO)}  ({len(old)} -> {len(new)} B)")
        elif args.check:
            print(f"  MISSING  {path.relative_to(REPO)}")
            bad += 1
            continue
        else:
            print(f"  write    {path.relative_to(REPO)}  ({len(new)} B)")
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(new)

    if args.check and bad:
        raise SystemExit(f"{bad} artifact(s) do not match the viewer payload")
    print("artifacts are the viewer's own bytes" + (" (check only)" if args.check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
