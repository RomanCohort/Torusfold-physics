"""How much does it cost to rotate one base frame? The stacking term's real competition.

The scan says a three-well base-level term at 10 kJ/mol moves the base-level marginals by ~10-20 percent of
what is needed. That is not a tuning question if reorienting a base costs hundreds of kJ/mol, so the cost
is measured directly: rotate ONE residue's rigid (P, C4', N) unit about the local backbone axis, and
evaluate the four cross-residue springs that connect it to its neighbours, with the constants the field
actually uses.
"""
import sys, math
import numpy as np
import torch
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import boltzmann_bonded as B
from torusfold.scheme2 import torch_cgsim as C

pool = [s for s in B.load_structures(limit=5000) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s = sorted(pool, key=lambda d: d["name"])[1]          # 1L2X
p = np.asarray(s["pos"], float)                        # (L, 3, 3) nm
L = p.shape[0]
K = dict(intra_pn=C.K_INTRA_PN, link_cp=C.K_LINK_CP, link_np=C.K_LINK_NP, link_nc=C.K_LINK_NC)
T = dict(intra_pn=C.BOND_INTRA_PN, link_cp=C.BOND_LINK_CP, link_np=C.BOND_LINK_NP, link_nc=C.BOND_LINK_NC)


def spring(pA, iA, pB, jB, k, t):
    d = float(np.linalg.norm(pA[iA] - pB[jB]))
    return 0.5 * k * (d - t) ** 2


def cost(p):
    e = 0.0
    for i in range(L):
        e += spring(p, i, p, i, K["intra_pn"], T["intra_pn"])
    for i in range(L - 1):
        e += spring(p[i], 1, p[i + 1], 0, K["link_cp"], T["link_cp"])   # C4'(i)-P(i+1)
        e += spring(p[i], 2, p[i + 1], 0, K["link_np"], T["link_np"])   # N(i)-P(i+1)
        e += spring(p[i], 2, p[i + 1], 1, K["link_nc"], T["link_nc"])   # N(i)-C4'(i+1)
    return e


e0 = cost(p)
print("chain %s L=%d, link network energy %.1f kJ/mol" % (s["name"], L, e0))
print("\nRotating ONE residue's rigid unit about the local P-P axis (keeping it rigid):")
print("%6s %12s %14s %16s" % ("angle", "cost kBT", "cost kJ/mol", "per-base term eps"))
for deg in (5, 10, 20, 30, 45):
    q = p.copy()
    i = L // 2
    ax = p[i + 1, 0] - p[i - 1, 0]
    ax = ax / np.linalg.norm(ax)
    th = math.radians(deg)
    Kmat = np.array([[math.cos(th) + ax[0]**2*(1-math.cos(th)), ax[0]*ax[1]*(1-math.cos(th)) - ax[2]*math.sin(th), ax[0]*ax[2]*(1-math.cos(th)) + ax[1]*math.sin(th)],
                     [ax[1]*ax[0]*(1-math.cos(th)) + ax[2]*math.sin(th), math.cos(th) + ax[1]**2*(1-math.cos(th)), ax[1]*ax[2]*(1-math.cos(th)) - ax[0]*math.sin(th)],
                     [ax[2]*ax[0]*(1-math.cos(th)) - ax[1]*math.sin(th), ax[2]*ax[1]*(1-math.cos(th)) + ax[0]*math.sin(th), math.cos(th) + ax[2]**2*(1-math.cos(th))]])
    for k in range(3):
        q[i, k] = p[i, 0] + Kmat @ (p[i, k] - p[i, 0])
    de = cost(q) - e0
    print("%5d deg %12.1f %14.1f %16.1f" % (deg, de / 2.494, de, de))
print("\n(For scale: kBT = 2.494 kJ/mol; the stacking term's strength in the scan was 0, 4 and 10 kJ/mol"
      "\n PER PAIR, i.e. at most ~1 kBT each.)")
