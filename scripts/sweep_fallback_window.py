import sys, collections, numpy as np
sys.path.insert(0, r"D:\torusfold-hybrid\src")
import torusfold.scheme2.aform_from_template as A
from torusfold.scheme2.aform_from_template import reconstruct_all_atom

PDB = r"D:\Torusfold-Templates\_tools\1ehz.pdb"
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

def run_one(win, sign):
    A._FALLBACK_WINDOW = win; A._FALLBACK_SIGN = sign
    s = reconstruct_all_atom(ps, seq, pairs=wc)
    m = []
    for i, k in enumerate(run):
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
        g["all"].append(e); g["paired" if i in paired else "unpaired"].append(e)
    return g

print(f"{'window':>8s} {'sign':>5s} {'overall':>16s} {'paired':>16s} {'unpaired':>16s}")
for sign in (+1.0, -1.0):
    for win in (4, 8, 12, 1000):
        g = run_one(win, sign)
        lab = "L" if win >= L else str(win)
        print(f"{lab:>8s} {sign:>+5.1f} {np.mean(g['all']):9.3f}+/-{np.std(g['all']):5.2f} "
              f"{np.mean(g['paired']):9.3f}+/-{np.std(g['paired']):5.2f} "
              f"{np.mean(g['unpaired']):9.3f}+/-{np.std(g['unpaired']):5.2f}")
