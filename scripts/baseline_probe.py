"""Clean-set baseline: reproduce the ORIGINAL placement rule (legacy on-axis anchors,
global-centroid radial roll, no pairing) so the before/after is measured on the same
76-residue truth set."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
import torusfold.scheme2.aform_from_template as A
from torusfold.scheme2.aform_from_template import reconstruct_all_atom

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order); seq = "".join(base_of)

def measure(label, pairs):
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
        g["all"].append(e); g["paired" if i in partner else "unpaired"].append(e)
    print(f"  {label:34s} overall {np.mean(g['all']):6.3f}  paired {np.mean(g['paired']):6.3f}  unpaired {np.mean(g['unpaired']):6.3f}")

keep_off = dict(A._ANCHOR_OFFSETS)
keep_sig, keep_win = A._FALLBACK_SIGN, A._FALLBACK_WINDOW

A._ANCHOR_OFFSETS = {"C1'": (5.5, 1.5, 0.0), "C4'": (4.2, 0.0, 0.0)}
A._FALLBACK_SIGN, A._FALLBACK_WINDOW = 1.0, L
print("clean-set comparison")
measure("original rule (no pairs, radial, legacy)", None)
A._ANCHOR_OFFSETS = keep_off
measure("+ partner roll only", [(a, b) for a, b in partner.items() if a < b])
A._FALLBACK_SIGN, A._FALLBACK_WINDOW = keep_sig, keep_win
measure("+ 3-D frame, corrected offsets, sign/win", [(a, b) for a, b in partner.items() if a < b])
