"""Measure 1EHZ anchor offsets in a full 3-D local frame.

Frame (the rule that will also be used to place them):
    b   = P[i] -> P[i+1], normalized
    r   = direction to the base-pair partner, orthogonalized against b, normalized
    n   = b x r
Offsets are the components of (X - P[i]) along (b, r, n), per base.
"""
import collections, numpy as np

PDB = r"D:\Torusfold-Templates\_tools\1ehz.pdb"
WCP = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}

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
L = len(run)
ps = np.array([res[k]["P"] for k in run])

partner = {}
for a in range(L):
    for b in range(a + 3, L):
        if (seq[a], seq[b]) not in WCP:
            continue
        if 9.0 <= np.linalg.norm(res[run[a]]["C1'"] - res[run[b]]["C1'"]) <= 11.5:
            partner.setdefault(a, b); partner.setdefault(b, a)
print(f"run {L} residues, paired {len(partner)}")

rows = collections.defaultdict(list)
for i in range(L - 1):
    j = partner.get(i)
    if j is None:
        continue
    b = ps[i + 1] - ps[i]; b = b / np.linalg.norm(b)
    d = ps[j] - ps[i]
    r = d - np.dot(d, b) * b
    rn = np.linalg.norm(r)
    if rn < 1e-6:
        continue
    r = r / rn
    n = np.cross(b, r)
    for nm in ("C1'", "C4'", "O3'", GLY[seq[i]]):
        v = res[run[i]][nm] - ps[i]
        rows[(seq[i], nm)].append([float(np.dot(v, b)), float(np.dot(v, r)), float(np.dot(v, n))])

print()
print("components (along b, along r, along n), Angstrom")
print(f"{'':10s} {'n':>4s} {'par':>16s} {'r-dir':>16s} {'n-dir':>16s}   |net|")
for nm in ("C1'", "C4'", "O3'", "N9", "N1"):
    for base in ("A", "U", "G", "C"):
        v = rows.get((base, nm))
        if not v:
            continue
        a = np.array(v)
        net = np.linalg.norm(a, axis=1)
        print(f"{base + ' ' + nm:10s} {len(v):4d} {a[:,0].mean():7.2f}+/-{a[:,0].std():5.2f} "
              f"{a[:,1].mean():7.2f}+/-{a[:,1].std():5.2f} {a[:,2].mean():7.2f}+/-{a[:,2].std():5.2f}   {net.mean():5.2f}")
    allv = [x for (bb, nn), vv in rows.items() if nn == nm for x in vv]
    if allv:
        a = np.array(allv)
        net = np.linalg.norm(a, axis=1)
        print(f"{'-- all ' + nm:10s} {len(allv):4d} {a[:,0].mean():7.2f}+/-{a[:,0].std():5.2f} "
              f"{a[:,1].mean():7.2f}+/-{a[:,1].std():5.2f} {a[:,2].mean():7.2f}+/-{a[:,2].std():5.2f}   {net.mean():5.2f}")
        print()
