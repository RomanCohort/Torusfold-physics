"""Place the bases so the Watson-Crick edges actually face each other: the genuine placement problem.

WHY THIS AND NOT THE ROLL. Part 28 tested the cheapest available repair -- rotating the whole RIGID residue
about the local backbone axis, one degree of freedom per residue -- and it took the key contacts from 1/12 to
1/12 while destroying the stacking (50.0 -> 0.0 percent). This is a different degree of freedom and a real one
of the molecule: the base rotates about the GLYCOSIDIC bond (C1' -> N9/N1), which is the chi torsion, and
leaves the sugar and the phosphate where the reconstruction put them.

WHAT IS OPTIMISED, per base, over chi alone:
    E = w_pair * SUM over that base's pairs of the key-contact excesses, squared
      + w_keep * (1 - n . n_original)          # do not move the base plane unless pairing needs it
The first term is the pairing geometry nothing in the chain of tools currently targets; the second is what the
rejected roll repair was missing, and it is the reason this one can fix pairing without losing stacking.

The sweep is coarse (5 degrees) and Gauss-Seidel over residues, a few passes: 71 bases times 72 angles is
cheap, and a product is a one-off structure rather than a trajectory.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import measure_base_stacking as M

KEY = M.WC_TRIPLES
# The base atoms of each residue: the ring plus its substituents. Everything else is sugar/phosphate and does
# not move under chi.
BASE_ATOMS = {
    "A": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4", "N6"),
    "G": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4", "N6", "O6", "N2"),
    "C": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "U": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
}
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}


def rot_about(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def chi_rotate(atoms, base, axis_point, axis, angle):
    """Rotate only the base atoms about the glycosidic axis (C1' -> N9/N1), which is chi."""
    R = rot_about(axis, angle)
    out = dict(atoms)
    for nm in BASE_ATOMS[base]:
        if nm in atoms:
            out[nm] = (R @ (np.asarray(atoms[nm], float) - axis_point)) + axis_point
    return out


def key_penalty(residues, seq, pairs, i, w_pair, w_keep, normals, normals0):
    pen = 0.0
    for (a, b) in pairs:
        if i not in (a, b):
            continue
        trip = KEY.get((seq[a], seq[b])) or KEY.get((seq[b], seq[a]))
        if trip is None:
            continue
        flip = (seq[a], seq[b]) not in KEY
        aa, ab = residues[a]["atoms"], residues[b]["atoms"]
        for x, y in trip:
            na, nb = (y, x) if flip else (x, y)
            if na not in aa or nb not in ab:
                continue
            d = float(np.linalg.norm(np.asarray(aa[na], float) - np.asarray(ab[nb], float)))
            pen += w_pair * max(0.0, d - 3.2) ** 2
    if w_keep:
        n = normals[i]
        pen += w_keep * (1.0 - float(np.dot(n, normals0[i])))
    return pen


def plane_normal(residue, base):
    pts = np.array([residue["atoms"][nm] for nm in BASE_ATOMS[base] if nm in residue["atoms"]], dtype=float)
    if len(pts) < 4:
        return np.array([0.0, 0.0, 1.0])
    c = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - c)
    n = vt[2]
    return n / np.linalg.norm(n)


def main():
    product = sys.argv[1] if len(sys.argv) > 1 else "results/plan_c/bead_product/2oiu_envswitch.pdb"
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    w_pair = 1.0
    w_keep = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    seq, residues, _p, _ch = M.parse_pdb(product)
    seq_c, res_c, _pc, _chc = M.parse_pdb("artifacts/2oiu/2OIU.pdb")
    ref_pairs = M.wc_pairs(seq_c, res_c, [M.plane(r) for r in res_c])
    L = len(seq)
    normals0 = [plane_normal(residues[i], seq[i]) for i in range(L)]

    def report(tag):
        n = 0
        for (a, b) in ref_pairs:
            trip = KEY.get((seq[a], seq[b])) or KEY.get((seq[b], seq[a]))
            flip = (seq[a], seq[b]) not in KEY
            ds = []
            for x, y in trip:
                na, nb = (y, x) if flip else (x, y)
                ds.append(float(np.linalg.norm(np.asarray(residues[a]["atoms"][na], float)
                                               - np.asarray(residues[b]["atoms"][nb], float))))
            if ds and max(ds) <= 3.6:
                n += 1
        planes = [M.plane(r) for r in residues]
        rows, sk = M.measure(seq_c, residues, ref_pairs, planes)
        stacked = 100.0 * np.mean([r["stacked"] for r in rows]) if rows else float("nan")
        rise = np.mean([r["rise"] for r in rows]) if rows else float("nan")
        print("   %-22s contacts %2d/%d   stacked %5.1f%%   rise %5.2f A"
              % (tag, n, len(ref_pairs), stacked, rise), flush=True)
        return n, stacked

    print("== %s  (chi placement, w_keep=%.2g)" % (Path(product).name, w_keep), flush=True)
    report("before")
    for p in range(passes):
        changed = 0
        for i in range(L):
            base = seq[i]
            atoms0 = residues[i]["atoms"]
            gly = GLY[base]
            if "C1'" not in atoms0 or gly not in atoms0:
                continue
            axis_point = np.asarray(atoms0["C1'"], float)
            axis = np.asarray(atoms0[gly], float) - axis_point
            if np.linalg.norm(axis) < 1e-9:
                continue
            normals = [plane_normal(residues[k], seq[k]) for k in range(L)]
            best = key_penalty(residues, seq, ref_pairs, i, w_pair, w_keep, normals, normals0)
            best_ang = 0.0
            for deg in range(5, 360, 5):
                residues[i]["atoms"] = chi_rotate(atoms0, base, axis_point, axis, np.radians(deg))
                normals = [plane_normal(residues[k], seq[k]) for k in range(L)]
                s = key_penalty(residues, seq, ref_pairs, i, w_pair, w_keep, normals, normals0)
                if s < best - 1e-9:
                    best, best_ang = s, deg
            residues[i]["atoms"] = chi_rotate(atoms0, base, axis_point, axis, np.radians(best_ang)) \
                if best_ang else atoms0
            changed += 1 if best_ang else 0
        print("   pass %d: %d bases rotated" % (p + 1, changed), flush=True)
        report("after pass %d" % (p + 1))


if __name__ == "__main__":
    main()
