"""Accuracy of aform_from_template.reconstruct_all_atom, with 1EHZ chain A as truth.

Only the true P trace is supplied; the reconstruction is superposed onto the truth using
P atoms alone. Templates come from 1EHZ, so shape is perfect by construction and all
remaining error is the placement rule. Self-inclusion caveat: measures the rule, not
generalisation. Loads through truth_1ehz.py, which includes the HETATM modified residues.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
import torusfold.scheme2.aform_from_template as A
from torusfold.scheme2.aform_from_template import reconstruct_all_atom

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order)
seq = "".join(base_of)
print(f"1EHZ truth: {meta['n_residues']} residues, {meta['n_atoms']} atoms, "
      f"contiguous={meta['contiguous']}, paired={meta['n_paired']}")
pairs = [(a, b) for a, b in partner.items() if a < b]


def frame(i, sign=None, window=None):
    b = ps[i + 1] - ps[i]
    b = b / np.linalg.norm(b)
    j = partner.get(i)
    if j is not None:
        d = ps[j] - ps[i]
        r = d - np.dot(d, b) * b
        rn = np.linalg.norm(r)
        if rn > 1e-6:
            return b, r / rn
    w = min(max(1, int(A._FALLBACK_WINDOW if window is None else window)), L)
    sg = A._FALLBACK_SIGN if sign is None else sign
    c = ps[[(i + k) % L for k in range(-w, w + 1)]].mean(axis=0)
    r = (ps[i] - c) * sg
    r = r - np.dot(r, b) * b
    rn = np.linalg.norm(r)
    return b, (r / rn if rn > 1e-6 else np.array([0.0, 0.0, 1.0]))


print()
print("measured anchor offsets (along b, along r, along n), Angstrom")
for nm in ("C1'", "C4'"):
    rows = []
    for i in range(L - 1):
        if partner.get(i) is None or base_of[i] not in "ACGU" or nm not in res[order[i]]:
            continue
        b, r = frame(i)
        v = res[order[i]][nm] - ps[i]
        rows.append([float(np.dot(v, b)), float(np.dot(v, r)), float(np.dot(v, np.cross(b, r)))])
    a = np.array(rows)
    print(f"  {nm:4s} n={len(a):3d}  {a[:,0].mean():6.2f}+/-{a[:,0].std():5.2f}  "
          f"{a[:,1].mean():6.2f}+/-{a[:,1].std():5.2f}  {a[:,2].mean():6.2f}+/-{a[:,2].std():5.2f}")
print("  module now:", A._ANCHOR_OFFSETS)


def rmsd_by_group(label):
    s = reconstruct_all_atom(ps, seq, pairs=pairs)
    m = []
    for i, k in enumerate(order):
        a = s.residue_atom_index[i]
        for nm, serial in a.items():
            if nm in res[k]:
                m.append((i, np.asarray(s.atoms[serial].xyz, float), res[k][nm]))
    rP = np.array([s.atoms[s.residue_atom_index[i]["P"]].xyz for i in range(L)], dtype=float)
    rc, tc = rP.mean(0), ps.mean(0)
    H = (rP - rc).T @ (ps - tc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ap = lambda c: (c - rc) @ R.T + tc
    g = {"all": [], "paired": [], "unpaired": []}
    for i, rr, tt in m:
        e = float(np.linalg.norm(ap(rr) - tt))
        g["all"].append(e)
        g["paired" if i in partner else "unpaired"].append(e)
    print(f"  {label:30s} overall {np.mean(g['all']):6.3f}  paired {np.mean(g['paired']):6.3f}  "
          f"unpaired {np.mean(g['unpaired']):6.3f}")
    return g


print()
print("accuracy (per-atom RMSD after P-superposition)")
rmsd_by_group("as shipped")
keep = (A._FALLBACK_SIGN, A._FALLBACK_WINDOW)
print()
print("fallback sweep (sign x window); only the unpaired group moves")
for sign in (-1.0, +1.0):
    for win in (2, 4, 8, 16, L):
        A._FALLBACK_SIGN, A._FALLBACK_WINDOW = sign, win
        rmsd_by_group(f"sign {sign:+.0f} window {'L' if win >= L else win}")
A._FALLBACK_SIGN, A._FALLBACK_WINDOW = keep

print()
print("upper bound (per-residue optimal orientation)")
s = reconstruct_all_atom(ps, seq, pairs=pairs)
g2 = []
for i in range(L):
    a = s.residue_atom_index[i]
    names = [nm for nm in a if nm in res[order[i]]]
    rc2 = np.array([np.asarray(s.atoms[a[nm]].xyz, float) for nm in names])
    tc2 = np.array([res[order[i]][nm] for nm in names])
    c1, c2 = rc2.mean(0), tc2.mean(0)
    H2 = (rc2 - c1).T @ (tc2 - c2)
    U2, _, Vt2 = np.linalg.svd(H2)
    d2 = np.sign(np.linalg.det(Vt2.T @ U2.T))
    R2 = Vt2.T @ np.diag([1.0, 1.0, d2]) @ U2.T
    for nm, rr, tt in zip(names, rc2, tc2):
        g2.append(float(np.linalg.norm((rr - c1) @ R2.T + c2 - tt)))
print(f"  {np.mean(g2):.3f} A")
