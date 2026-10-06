"""Can a per-residue ROLL optimisation repair the products' base pairing?

The diagnosis: in every product, only 1 of the 12 reference pairs has all its key contacts within 3.6 A, and
the distances are 3.7-13 A -- the bases are not merely twisted, they are in the wrong place relative to their
partners. Nothing in the CG field targets pairing geometry for a reconstruction, so this asks the cheapest
possible repair question first: the model's three beads per residue are a RIGID unit, and rotating that unit
about the local backbone axis is one degree of freedom per residue which Part 16 measured to be soft (12.4
kJ/mol for 30 degrees). If a sweep over that one angle per residue can bring the pairs into contact, the
repair is cheap and belongs in the all-atom stage; if it cannot, the contacts need the base frame to be the
model's own, which is the architectural question.

This is a POST-HOC repair of a product, not the field acting: it is reported as such.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import measure_base_stacking as M

KEY = M.WC_TRIPLES


def key_distances(residues, seq, i, j):
    trip = KEY.get((seq[i], seq[j])) or KEY.get((seq[j], seq[i]))
    if trip is None:
        return []
    flip = (seq[i], seq[j]) not in KEY
    ai, aj = residues[i]["atoms"], residues[j]["atoms"]
    out = []
    for a, b in trip:
        na, nb = (b, a) if flip else (a, b)
        if na not in ai or nb not in aj:
            continue
        out.append(float(np.linalg.norm(np.asarray(ai[na], float) - np.asarray(aj[nb], float))))
    return out


def pair_score(residues, seq, pairs, i):
    """Soft penalty for residue i's own pairs: 0 once every key contact is inside 3.4 A."""
    s = 0.0
    for (a, b) in pairs:
        if i not in (a, b):
            continue
        for d in key_distances(residues, seq, a, b):
            s += max(0.0, d - 3.4) ** 2
    return s


def rotate_residue(residues, i, axis, angle, pivot):
    c, s = np.cos(angle), np.sin(angle)
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)
    return {nm: (R @ (np.asarray(x, float) - pivot)) + pivot for nm, x in residues[i]["atoms"].items()}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/plan_c/bead_product/2oiu_refine.pdb"
    seq, residues, _p, _ch = M.parse_pdb(path)
    seq_c, res_c, _pc, _chc = M.parse_pdb("artifacts/2oiu/2OIU.pdb")
    planes_c = [M.plane(r) for r in res_c]
    pairs = M.wc_pairs(seq_c, res_c, planes_c)
    L = len(seq)
    P = [np.asarray(residues[i]["atoms"]["P"], float) for i in range(L)]

    def contacts(tag):
        n = 0
        for (a, b) in pairs:
            ds = key_distances(residues, seq, a, b)
            if ds and max(ds) <= 3.6:
                n += 1
        print("   %-28s %d/%d pairs satisfy the criterion" % (tag, n, len(pairs)), flush=True)
        return n

    print("== %s (post-hoc roll optimisation, 1 DOF per residue)" % Path(path).name, flush=True)
    contacts("before")
    for p in range(3):
        moved = 0
        for i in range(L):
            axis = P[(i + 1) % L] - P[(i - 1) % L] if L > 2 else np.array([0.0, 0.0, 1.0])
            if np.linalg.norm(axis) < 1e-9:
                continue
            base = pair_score(residues, seq, pairs, i)
            if base <= 0.0:
                continue
            best, best_ang = base, 0.0
            keep = {nm: np.asarray(x, float).copy() for nm, x in residues[i]["atoms"].items()}
            for deg in range(10, 360, 10):
                residues[i]["atoms"] = rotate_residue(residues, i, axis, np.radians(deg), P[i])
                s = pair_score(residues, seq, pairs, i)
                if s < best - 1e-9:
                    best, best_ang = s, deg
            residues[i]["atoms"] = rotate_residue(residues, i, axis, np.radians(best_ang), P[i]) \
                if best_ang else keep
            if best_ang:
                moved += 1
        print("   pass %d: %d residues rotated" % (p + 1, moved), flush=True)
        contacts("after pass %d" % (p + 1))
    planes = [M.plane(r) for r in residues]
    rows, sk = M.measure(seq_c, residues, pairs, planes)
    M.summarize(rows, len(pairs), sk, "product AFTER the roll repair")
    M.summarize(*M.measure(seq_c, M.parse_pdb(path)[1], pairs,
                           [M.plane(r) for r in M.parse_pdb(path)[1]])[:2],
                "product BEFORE", "(reference row)")


if __name__ == "__main__":
    main()
