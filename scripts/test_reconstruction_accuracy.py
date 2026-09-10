"""Accuracy of aform_from_template.reconstruct_all_atom with 1EHZ chain A as truth.

Only the true P trace is supplied; the reconstruction is superposed onto the truth using
P atoms alone. Templates come from 1EHZ, so residue shape is perfect by construction and
all remaining error is the placement rule. Self-inclusion caveat: this measures the
placement rule, not generalisation.
"""
import sys, collections, numpy as np
sys.path.insert(0, r"D:\torusfold-hybrid\src")
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
        if (seq[a], seq[b]) in WCP and 9.0 <= np.linalg.norm(res[run[a]]["C1'"] - res[run[b]]["C1'"]) <= 11.5:
            wc.append((a, b))
paired = {i for pr in wc for i in pr}
print(f"run: {L} residues, {len(wc)} WC pairs -> {len(paired)} paired, {L - len(paired)} unpaired")

s = reconstruct_all_atom(ps, seq, pairs=wc)
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

g, ang, bypart = collections.defaultdict(list), collections.defaultdict(list), {"paired": [], "unpaired": []}
for i, nm, rr, tt in m:
    e = float(np.linalg.norm(ap(rr) - tt))
    g[nm].append(e)
    bypart["paired" if i in paired else "unpaired"].append(e)
    if nm in ("C1'", "C4'") or nm == GLY[seq[i]]:
        v1, v2 = rr - ps[i], tt - ps[i]
        c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        ang[nm].append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))

allr = np.array([v for vs in g.values() for v in vs])
print()
print(f"overall per-atom RMSD after P-superposition: {allr.mean():.3f} +/- {allr.std():.3f} A")
for show, disp in (("P", "P"), ("C1'", "C1'"), ("C4'", "C4'"), ("O3'", "O3'"), ("N9", "N9"), ("N1", "N1"), ("O2'", "O2'"), ("C5'", "C5'"), ("O5'", "O5'")):
    if show in g:
        a2 = f"{np.mean(ang[show]):6.1f} +/- {np.std(ang[show]):4.1f}" if show in ang else ""
        print(f"  {disp:6s} {len(g[show]):4d} {np.mean(g[show]):8.3f}   {a2:>22s}")
print()
for k, v in bypart.items():
    print(f"  {k:9s} n={len(v):4d}  RMSD {np.mean(v):6.3f} +/- {np.std(v):5.3f} A")
