"""What happens if the two mistargeted constants are recalibrated?

The constants are re-derived here from the data rather than typed in, so there is one
source. For a restraint of the form 0.5*k*(q - q0)^2, or one that restrains cos(q), the
value of q0 that minimises the mean squared deviation is the MEAN of the observed q (for
the cosine form, the mean of the observed COSINES -- not the cosine of the mean angle).

Beads are read straight from the deposited coordinates of rsRNASP's training set. No
reconstruction is involved, so nothing here depends on our own template code.

Three settings are compared on the same structures:
  A  as shipped              STACK_R0 0.505 nm, DIH_PPPP 180 deg
  B  stacking + dihedral     both re-derived
  C  B + the P-P-P angle     all three re-derived

What is measured:
  * the raw force magnitude at the native geometry, and how much of it the 200 kJ/mol/nm
    cap in cg_energy_forces absorbs
  * whether the field still prefers the native structure over perturbed decoys -- the
    obvious risk of "fixing" a force field is flattening it

Run: python scripts/recalibrate_ff_targets.py [n_structures]
"""
import collections
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C
from truth_1ehz import WCP

DATA = Path(r"D:\torusfold-cgdata\rsRNASP\Training_set")
N_STRUCT = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MIN_L, MAX_L = 20, 120
SIGMAS = (0.3, 0.6, 1.0)          # Angstrom, bead-position perturbation
N_DECOY_PER_SIGMA = 2
SEED = 20260213


def chains_of(pdb):
    ch = collections.OrderedDict()
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname not in ("A", "C", "G", "U"):
            continue
        aname = line[12:16].strip()
        gly = "N9" if rname in ("A", "G") else "N1"
        if aname not in ("P", "C4'", "C1'", gly):
            continue
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        rec = ch.setdefault(line[21], collections.OrderedDict())
        e = rec.setdefault((line[22:27].strip(), rname), {})
        e.setdefault(aname, xyz)
    out = []
    for rec in ch.values():
        lst = []
        for (_resid, rname), at in rec.items():
            gly = "N9" if rname in ("A", "G") else "N1"
            if not all(a in at for a in ("P", "C4'", "C1'", gly)):
                continue
            lst.append({"res": rname, "P": at["P"], "C4": at["C4'"],
                        "N": at[gly], "C1": at["C1'"]})
        if len(lst) > MIN_L and len(lst) <= MAX_L:
            out.append(lst)
    return out


def wc_pairs(lst):
    out = []
    for a in range(len(lst)):
        for b in range(a + 3, len(lst)):
            if (lst[a]["res"], lst[b]["res"]) not in WCP:
                continue
            if 9.0 <= np.linalg.norm(lst[a]["C1"] - lst[b]["C1"]) <= 11.5:
                out.append((a, b))
    return out


def cos_angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return (v1 * v2).sum(-1) / np.clip(n, 1e-9, None)


def cos_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = np.cross(b0, b1)
    n1 = np.cross(b1, b2)
    n0 /= np.clip(np.linalg.norm(n0, axis=-1, keepdims=True), 1e-9, None)
    n1 /= np.clip(np.linalg.norm(n1, axis=-1, keepdims=True), 1e-9, None)
    return (n0 * n1).sum(-1)


# ---- load -----------------------------------------------------------------
structs = []
for f in sorted(DATA.glob("*.pdb")):
    if len(structs) >= N_STRUCT:
        break
    for lst in chains_of(f):
        pr = wc_pairs(lst)
        if len(pr) < 4:
            continue
        beads = np.stack([[r["P"], r["C4"], r["N"]] for r in lst]) / 10.0     # A -> nm
        structs.append({"name": f.stem, "beads": beads, "pairs": pr, "L": len(lst)})
        if len(structs) >= N_STRUCT:
            break

print(f"{len(structs)} native structures from rsRNASP/Training_set, beads taken directly")
print(f"from the deposited coordinates; lengths {min(s['L'] for s in structs)}"
      f"-{max(s['L'] for s in structs)} residues")

# Fit and test must not be the same structures, or the recalibration is scored on the data
# it was fitted to and the improvement is circular. The first half derives the constants;
# every number reported below is measured on the second half.
FIT = structs[:max(1, len(structs) // 2)]
TEST = structs[max(1, len(structs) // 2):]
print(f"fitted on {len(FIT)} structures, tested on {len(TEST)} held-out structures")
print()

# ---- derive the constants from the fit half -------------------------------
st_all, dih_all, ang_all = [], [], []
for s in FIT:
    P = s["beads"][:, 0, :]
    if len(P) > 2:
        st_all.append(np.linalg.norm(P[2:] - P[:-2], axis=-1))
    if len(P) > 4:
        dih_all.append(cos_dihedral(P[:-3], P[1:-2], P[2:-1], P[3:]))
    if len(P) > 3:
        ang_all.append(cos_angle(P[:-2], P[1:-1], P[2:]))
st_all = np.concatenate(st_all)
dih_all = np.concatenate(dih_all)
ang_all = np.concatenate(ang_all)

def mode_of(x, lo, hi, n=40):
    """The peak of the histogram, which is where the observable actually sits.

    The MEAN minimises the mean squared restraint energy, which is the right target only
    if the distribution is symmetric. check_bonded_distributions.py showed it is not: the
    pseudo-torsion is strongly skewed, so its mean falls between states rather than on
    one, and a restraint to the mean pins the structure somewhere it never visits. Both
    are reported so the difference is visible.
    """
    h, e = np.histogram(x, bins=np.linspace(lo, hi, n + 1))
    i = h.argmax()
    return 0.5 * (e[i] + e[i + 1])


STACK_MEAN = float(st_all.mean())
STACK_NEW = mode_of(st_all, 0.4, 2.0, 32)
DIH_MEAN = float(dih_all.mean())
DIH_COS_NEW = mode_of(dih_all, -1.0, 1.0)
ANG_MEAN = float(ang_all.mean())
ANG_COS_NEW = mode_of(ang_all, -1.0, 1.0)
print(f"re-derived from the {len(FIT)} fit structures")
print(f"  STACK_R0   {C.STACK_R0:.3f} -> mode {STACK_NEW:.3f} (mean {STACK_MEAN:.3f}) nm")
print(f"  DIH cos    {math.cos(C.DIH_PPPP):+.3f} -> mode {DIH_COS_NEW:+.3f} "
      f"(mean {DIH_MEAN:+.3f})")
print(f"  ANGLE cos  {math.cos(C.ANGLE_PPP):+.3f} -> mode {ANG_COS_NEW:+.3f} "
      f"(mean {ANG_MEAN:+.3f})")
print()
print("the mode is where the observable actually sits. The mean minimises the mean squared")
print("restraint energy but for a skewed distribution it falls between states, so a")
print("restraint to it pins the structure somewhere it never visits. Both are shown below.")
print()

ORIG = (C.STACK_R0, math.cos(C.DIH_PPPP), math.cos(C.ANGLE_PPP))
VARIANTS = [("A as shipped", ORIG),
            ("B mode", (STACK_NEW, DIH_COS_NEW, ORIG[2])),
            ("B' mean", (STACK_MEAN, DIH_MEAN, ORIG[2])),
            ("C mode+ang", (STACK_NEW, DIH_COS_NEW, ANG_COS_NEW))]


def apply(cfg):
    C.STACK_R0 = cfg[0]
    C._R0_STACK = cfg[0]
    C.DIH_PPPP = math.acos(max(min(cfg[1], 1.0), -1.0))
    C.ANGLE_PPP = math.acos(max(min(cfg[2], 1.0), -1.0))


def evaluate(beads, pairs):
    """(energy, capped force array kJ/mol/nm)."""
    L = len(beads)
    pos = torch.tensor(beads.reshape(1, 3 * L, 3), dtype=torch.float64)
    pi = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pj = torch.tensor([b for _, b in pairs], dtype=torch.long)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    e, f = C.cg_energy_forces(pos, torch.stack([pi, pj], 1),
                              torch.ones(len(pairs), dtype=torch.float64), cell_list=cl)
    return float(e.reshape(-1)[0]), f.reshape(3 * L, 3).numpy()


rng = np.random.default_rng(SEED)
results = {}
for label, cfg in VARIANTS:
    apply(cfg)
    fmed, fsat, ntot, ranks, winner_e = [], 0, 0, [], []
    for s in TEST:
        e0, f0 = evaluate(s["beads"], s["pairs"])
        m = np.linalg.norm(f0, axis=1)
        fmed.append(np.median(m))
        fsat += int((np.abs(m - 200.0) < 1e-6).sum())
        ntot += len(m)
        cand = [e0]
        for sig in SIGMAS:
            for _ in range(N_DECOY_PER_SIGMA):
                pert = s["beads"] + rng.normal(0, sig / 10.0, s["beads"].shape)   # A -> nm
                cand.append(evaluate(pert, s["pairs"])[0])
        cand = np.array(cand)
        ranks.append(int((cand < e0).sum()) + 1)          # 1 = native is best
        winner_e.append(cand.min() - e0)
    results[label] = dict(fmed=float(np.mean(fmed)), fsat=fsat, ntot=ntot,
                          ranks=np.array(ranks), margin=float(np.mean(winner_e)))

apply(ORIG)

print(f"{'setting':14s} {'median |F|':>12s} {'on cap':>10s} {'native rank':>12s} "
      f"{'mean(decoy-native)':>19s}")
print("-" * 74)
for label, _ in VARIANTS:
    r = results[label]
    print(f"{label:14s} {r['fmed']:12.1f} {r['fsat']:5d}/{r['ntot']:<4d} "
          f"{r['ranks'].mean():12.2f} {r['margin']:19.1f}")
print()
print("every number below is on the HELD-OUT structures, not the ones the constants were")
print("fitted to.")
print("median |F| and 'on cap' are kJ/mol/nm at the native geometry, after the cap in")
print("cg_energy_forces. 'native rank' averages 1.0 if the native is always the lowest of")
print(f"the {1 + len(SIGMAS)*N_DECOY_PER_SIGMA} candidates; higher is worse.")
print("'mean(decoy-native)' is how much lower the best decoy scores; negative means a decoy")
print("beat the native.")

# ---- which term still carries the force, before and after ------------------
# The variant table above says the recalibration does not change the magnitude much.
# This says which term is left holding it.
import cg_force_terms

print()
print("per-term mean |F| at the native geometry, averaged over the structures (kJ/mol/nm)")
print()
table = {}
for label, cfg in [VARIANTS[0], VARIANTS[1]]:
    apply(cfg)
    acc = collections.defaultdict(list)
    for s in TEST:
        L = s["L"]
        pos = torch.tensor(s["beads"].reshape(1, 3 * L, 3), dtype=torch.float64)
        ij = torch.tensor(s["pairs"], dtype=torch.long)
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        terms, _ = cg_force_terms.term_energies_forces(
            pos, ij, torch.ones(len(s["pairs"]), dtype=torch.float64), cell_list=cl)
        for k, F in terms.items():
            acc[k].append(float(np.linalg.norm(F, axis=1).mean()))
    table[label] = {k: float(np.mean(v)) for k, v in acc.items()}
apply(ORIG)

names = sorted(table[VARIANTS[0][0]], key=lambda k: -table[VARIANTS[0][0]][k])
print(f"{'term':20s} {'A as shipped':>13s} {'B stack+dih':>13s} {'change':>10s}")
print("-" * 60)
for k in names:
    a = table[VARIANTS[0][0]][k]
    b = table[VARIANTS[1][0]][k]
    print(f"{k:20s} {a:13.2f} {b:13.2f} {b - a:+10.2f}")
print("-" * 60)
sa = sum(table[VARIANTS[0][0]].values())
sb = sum(table[VARIANTS[1][0]].values())
print(f"{'sum of terms':20s} {sa:13.2f} {sb:13.2f} {sb - sa:+10.2f}")
