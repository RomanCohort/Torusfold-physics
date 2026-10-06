# -*- coding: utf-8 -*-
"""Run the circular pipeline's CG stage on any circRNA sequence and leave a checkable record.

WHY. The reuse record for this project has two entries and a gap: 2OIU (the one experimentally resolved
circRNA structure, used for every force-field check) and a 2,013 nt demo whose artifact is DECODED, not
re-run -- artifacts/2013nt/provenance.json says "is_a_rerun": false, so a third party cannot reproduce it,
only verify its hashes against the viewer payload it came from. A community-reuse page should carry at least
one record a reader can execute themselves, on a sequence nobody in this team has run before.

WHAT THIS DOES, in the order the real pipeline does it:
  1. read a FASTA (T -> U), predict a secondary structure and a base-pair probability list (ViennaRNA, the
     same call scripts/prep_2oiu_input.py makes) -- or carry on with an empty pair list if ViennaRNA is
     absent, which the CG field accepts (cg_energy_forces is finite with an (0, 2) index tensor);
  2. build a compact circular start with the pipeline's own generator;
  3. refine it with the shipped GPU CG stage under the settings results/README.md recommends
     (TORUSFOLD_CG_TABLES + TORUSFOLD_BASE_STACK + TORUSFOLD_REFINE_MODE=refine + TORUSFOLD_HBOND_REPAIR);
  4. print and write a record: sequence, length, SS, the exact commands, wall time, product size, closure,
     radius of gyration, and the product's SHA-256.

Usage:
    python scripts/demo_circrna_reuse.py <fasta> [name] [steps]
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))


def read_fasta(path):
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    name = lines[0][1:].strip() if lines[0].startswith(">") else Path(path).stem
    seq = "".join(l for l in lines if not l.startswith(">")).upper().replace("T", "U")
    bad = sorted({c for c in seq if c not in "ACGU"})
    if bad:
        raise SystemExit(f"{path}: only ACGU is supported, found {bad}")
    return name, seq


def secondary_structure(seq):
    """(dot-bracket, pairs) from ViennaRNA if it is importable, else no pairs at all."""
    try:
        import ViennaRNA
    except Exception as exc:                                   # noqa: BLE001
        print(f"  ViennaRNA not available here ({exc}); running with an EMPTY pair list")
        return "." * len(seq), []
    fc = ViennaRNA.fold_compound(seq)
    ss = fc.mfe()[0]
    fc.pf()
    bpp = fc.bpp()
    pairs = [[i - 1, j - 1, float(bpp[i][j])]
             for i in range(1, len(seq) + 1) for j in range(i + 1, len(seq) + 1) if bpp[i][j] > 0.1]
    return ss, pairs


def main():
    fasta = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "ct1.fa"
    name, seq = read_fasta(fasta)
    tag = sys.argv[2] if len(sys.argv) > 2 else name
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    L = len(seq)
    out = REPO / "artifacts" / "reuse_demo" / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"circRNA {name}: {L} nt, GC {(seq.count('G') + seq.count('C')) / L:.1%}   source {fasta}")

    t0 = time.time()
    ss, pairs = secondary_structure(seq)
    print(f"  secondary structure: {len(pairs)} pairs with bpp > 0.1 in {time.time() - t0:.1f} s")
    (out / "sequence.txt").write_text(seq + "\n", encoding="utf-8", newline="\n")
    (out / "ss.txt").write_text(ss + "\n", encoding="utf-8", newline="\n")

    from torusfold.scheme2.openmm_gpu_refiner import (                          # noqa: E402
        _generate_compact_coords, _read_p_coords, BOND_P_NEXT)
    # UNIT, and it cost a run: _generate_compact_coords sizes the circle from BOND_P_NEXT, which is 5.90
    # ANGSTROM, so what it returns is Angstrom. This line used to multiply by ten ("nm -> A") and turned a
    # 706 nt circle into one of radius 6,631 A -- past the PDB coordinate columns (31-54, 8.3f), where
    # adjacent fields merge and the loader's whitespace fallback dies on "-1029.8006543.839". A unit that
    # is only wrong by ten is still wrong; the round-trip check below is what makes that visible here
    # rather than three minutes into a refinement.
    start_A = np.asarray(_generate_compact_coords(L, [(int(a), int(b), float(w)) for a, b, w in pairs]),
                         dtype=float)
    span = float(np.abs(start_A).max())
    if span >= 1000.0:
        raise SystemExit(
            "the compact start for %d nt reaches %.0f A, which does not fit the PDB coordinate columns "
            "(8.3f, |x| < 1000 A). The shipped loader reads fixed columns and its whitespace fallback "
            "cannot recover a merged field, so this start format tops out near %d nt at the shipped "
            "%.2f A per step -- a longer chain needs a real starting structure, not a wider field."
            % (L, span, int(2.0 * np.pi * 999.0 / BOND_P_NEXT), BOND_P_NEXT))
    pdb_in = out / "start_p.pdb"
    with pdb_in.open("w", newline="\n") as fh:
        for i in range(L):
            x, y, z = start_A[i]
            fh.write("ATOM  %5d  P   ADE A%4d    %8.3f%8.3f%8.3f  1.00  0.00           P\n"
                     % (i + 1, i + 1, x, y, z))
        fh.write("END\n")
    back = _read_p_coords(str(pdb_in))
    dev = float(np.abs(back - start_A).max()) if back.shape == start_A.shape else float("inf")
    if back.shape != start_A.shape or dev > 1e-3:
        raise SystemExit("the start PDB does not read back as written: %s vs %s, max deviation %.3f A"
                         % (back.shape, start_A.shape, dev))
    print(f"  start: a compact circle of radius {np.linalg.norm(start_A[0]):.1f} A, bond {BOND_P_NEXT} A, "
          f"read back to %.1e A, written to {pdb_in.relative_to(REPO)}" % dev)

    os.environ.setdefault("TORUSFOLD_CG_TABLES", str(REPO / "results" / "production_tables.npz"))
    os.environ.setdefault("TORUSFOLD_BASE_STACK", "16.6:0,1,0.19")
    os.environ.setdefault("TORUSFOLD_REFINE_MODE", "refine")
    os.environ.setdefault("TORUSFOLD_REFINE_STEPS", str(steps))
    os.environ.setdefault("TORUSFOLD_SEED", "20261005")
    os.environ.setdefault("TORUSFOLD_HBOND_REPAIR", "1")
    from torusfold.scheme2.torch_gpu_refine import torch_gpu_refine                # noqa: E402
    t0 = time.time()
    pdb_out, energy, diag = torch_gpu_refine(str(pdb_in), str(out), seq, ss, name=tag,
                                             verbose=True, refine_mode="fold", bead_source_pdb=None)
    wall = time.time() - t0

    from openmm.app import PDBFile                                                # noqa: E402
    pdb = PDBFile(str(pdb_out))
    coords = np.array([[p.x, p.y, p.z] for p in pdb.positions])                   # nm
    p_only = coords[[a.index for r in pdb.topology.residues() for a in r.atoms()
                     if a.name.strip() == "P"]]
    rms_closure = float(np.linalg.norm(p_only[0] - p_only[-1]))
    rg = float(np.sqrt(((p_only - p_only.mean(0)) ** 2).sum(1).mean()))
    sha = hashlib.sha256(Path(pdb_out).read_bytes()).hexdigest()
    record = {
        "circrna": name, "sequence_length": L, "source_fasta": str(fasta),
        "secondary_structure": ss, "n_pairs": len(pairs),
        "environment": {k: os.environ[k] for k in
                        ("TORUSFOLD_CG_TABLES", "TORUSFOLD_BASE_STACK", "TORUSFOLD_REFINE_MODE",
                         "TORUSFOLD_REFINE_STEPS", "TORUSFOLD_SEED", "TORUSFOLD_HBOND_REPAIR")
                        if k in os.environ},
        "wall_seconds": wall, "product_atoms": len(pdb.positions),
        "closure_nm": rms_closure, "radius_of_gyration_nm": rg,
        "product": str(Path(pdb_out).relative_to(REPO)), "product_sha256": sha,
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
    print(f"\nrecord: {L} nt -> {len(pdb.positions)} atoms in {wall:.0f} s")
    print(f"  closure |P0 - Plast| {rms_closure:.3f} nm   Rg {rg:.2f} nm   sha256 {sha[:16]}...")
    print(f"  written to {(out / 'record.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
