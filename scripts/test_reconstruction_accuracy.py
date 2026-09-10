"""Accuracy of aform_from_template.reconstruct_all_atom with 1EHZ chain A as truth."""
import sys, collections, numpy as np
sys.path.insert(0, r"D:\torusfold-hybrid\src")
import torusfold.scheme2.aform_from_template as A
from torusfold.scheme2.aform_from_template import reconstruct_all_atom

PDB = r"D:\Torusfold-Templates\_tools\1ehz.pdb"
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
WCP = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}

order, res = [], collections.OrderedDict()
for line in open(PDB):
    if not line.startswith("ATOM") or line[16] not in (" ", "A"):
        continue
    key = (line[21], line[22:27].strip(), line[17:20].strip())
    try:
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    except ValueError:
        continue
    if key not in res:
        res[key] = {}; order.append(key)
    res[key].setdefault(line[12:16].strip(), xyz)

runs, cur = [], []
for k in order:
    if k[2] in "ACGU" and "P" in res[k]:
        cur.append(k)
    else:
        if len(cur) > len(runs):
            runs = cur
        cur = []
if len(cur) > len(runs):
    runs = cur
run = runs
seq = "".join(k[2] for k in run)
ps = np.array([res[k]["P"] for k in run])
L = len(run)

wc = []
for a in range(L):
    for b in range(a + 3, L):
        if (seq[a], seq[b]) not in WCP:
            continue
        if 9.0 <= np.linalg.norm(res[run[a]]["C1'"] - res[run[b]]["C1'"]) <= 11.5:
            wc.append((a, b))
print(f"run: {L} residues ({run[0][1]}..{run[-1][1]}), geometry-derived WC pairs: {len(wc)}")


def evaluate(pairs, label):
    s = reconstruct_all_atom(ps, seq, pairs=pairs) if pairs is not None else reconstruct_all_atom(ps, seq)
    m = []
    for i, k in enumerate(run):
        a = s.residue_atom_index[i]
        for nm, serial in a.items():
            if nm in res[k]:
                m.append((i, nm, np.asarray(s.atoms[serial].xyz, float), res[k][nm]))
    rP = np.array([x[2] for x in m if x[1] == "P"])
    tP = np.array([x[3] for x in m if x[1] == "P"])
    rc, tc = rP.mean(0), tP.mean(0)
    H = (rP - rc).T @ (tP - tc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ap = lambda c: (c - rc) @ R.T + tc
    g, ang = collections.defaultdict(list), collections.defaultdict(list)
    for i, nm, rr, tt in m:
        g[nm].append(float(np.linalg.norm(ap(rr) - tt)))
        if nm in ("C1'", "C4'") or nm == GLY[seq[i]]:
            v1, v2 = rr - ps[i], tt - ps[i]
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            ang[nm].append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    allr = np.array([v for vs in g.values() for v in vs])
    print()
    print(f"=== {label} ===")
    print(f"overall per-atom RMSD after P-superposition: {allr.mean():.3f} +/- {allr.std():.3f} A")
    for nm in ["P", "C1'", "C4'", "O3'", "N9", "N1", "O2'", "C5'", "O5'"]:
        if nm in g:
            a = f"{np.mean(ang[nm]):6.1f} +/- {np.std(ang[nm]):4.1f}" if nm in ang else ""
            print(f"  {nm:6s} {len(g[nm]):4d} {np.mean(g[nm]):8.3f}   {a:>22s}")


evaluate(None, "A. measured anchors + radial roll")
evaluate(wc, "B. measured anchors + partner-directed roll")

keep = (A._C1_ALONG, A._C1_PERP, A._C4_ALONG, A._C4_PERP)
A._C1_ALONG, A._C1_PERP = 5.5, 1.5
A._C4_ALONG, A._C4_PERP = 4.2, 0.0
evaluate(wc, "C. ABLATION: legacy anchors + partner-directed roll")
A._C1_ALONG, A._C1_PERP, A._C4_ALONG, A._C4_PERP = keep

print()
print("=== E. upper bound: each residue superposed optimally on its own truth ===")
s = reconstruct_all_atom(ps, seq, pairs=wc)
g2 = collections.defaultdict(list)
for i, k in enumerate(run):
    a = s.residue_atom_index[i]
    names = [nm for nm in a if nm in res[k]]
    rc2 = np.array([np.asarray(s.atoms[a[nm]].xyz, float) for nm in names])
    tc2 = np.array([res[k][nm] for nm in names])
    c1, c2 = rc2.mean(0), tc2.mean(0)
    H2 = (rc2 - c1).T @ (tc2 - c2)
    U2, _, Vt2 = np.linalg.svd(H2)
    d2 = np.sign(np.linalg.det(Vt2.T @ U2.T))
    R2 = Vt2.T @ np.diag([1.0, 1.0, d2]) @ U2.T
    for nm, rr, tt in zip(names, rc2, tc2):
        g2[nm].append(float(np.linalg.norm((rr - c1) @ R2.T + c2 - tt)))
allg = np.array([v for vs in g2.values() for v in vs])
print(f"per-atom RMSD with perfect per-residue orientation: {allg.mean():.3f} A")
print("  (non-zero only because 1EHZ residues differ slightly from the templates' shapes)")
