"""Diagnostic: are high-penalty dihedral residues on long-range pair paths?

Hypothesis: circRNA topology forces backbone distortion to satisfy both
BSJ closure and long-range pairing. Residues with worst dihedral deviations
should cluster on the connection paths between far-end pair partners.
"""
import sys, os
sys.path.insert(0, "C:/Users/颜子壹/TorusFold-scheme2-rl/src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 1. Load data ──
OUT = "C:/Users/颜子壹/TorusFold-scheme2-rl/output_2013nt"

# P-only coords (CG)
from openmm.app import PDBFile
pdb = PDBFile(os.path.join(OUT, "isrnaclong_final.pdb"))
L_res = list(pdb.topology.residues())
L = len(L_res)

p_coords = np.zeros((L, 3), dtype=np.float64)
seq_chars = []
for i, res in enumerate(L_res):
    rn = res.name.upper().strip()
    if rn == "T": rn = "U"
    seq_chars.append(rn)
    for atom in res.atoms():
        if atom.name.strip() == "P":
            pos = pdb.positions[atom.index]
            p_coords[i] = [pos.x, pos.y, pos.z]
seq = "".join(seq_chars)

# bpp pairs
bpp = np.load(os.path.join(OUT, "pair_heatmap.npy"))
pairs = []
for i in range(L):
    for j in range(i + 1, L):
        if bpp[i][j] > 0.1:
            pairs.append((i, j, float(bpp[i][j])))

# Far-end pairs (|i-j| > 50)
far_pairs = [(i, j, w) for i, j, w in pairs if abs(i - j) > 50]
near_pairs = [(i, j, w) for i, j, w in pairs if abs(i - j) <= 50]
print(f"L={L}, pairs={len(pairs)}, far={len(far_pairs)}, near={len(near_pairs)}")

# ── 2. Compute backbone dihedrals from P coordinates ──
# P-P-P-P dihedral = backbone helical twist
# A-form ideal: ~33 degrees
def compute_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle between 4 points (degrees)."""
    v1 = p1 - p0
    v2 = p2 - p1
    v3 = p3 - p2
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    n1 /= np.linalg.norm(n1) + 1e-8
    n2 /= np.linalg.norm(n2) + 1e-8
    cos_d = np.clip(np.dot(n1, n2), -1, 1)
    sin_d = np.dot(np.cross(n1, n2), v2 / (np.linalg.norm(v2) + 1e-8))
    return np.degrees(np.arctan2(sin_d, cos_d))

dihedrals = np.zeros(L)
for i in range(L):
    i0 = (i - 1) % L
    i1 = i
    i2 = (i + 1) % L
    i3 = (i + 2) % L
    dihedrals[i] = compute_dihedral(p_coords[i0], p_coords[i1], p_coords[i2], p_coords[i3])

# Deviation from A-form (33 degrees)
ideal_dih = 33.0
dih_deviation = np.abs(dihedrals - ideal_dih)
# Handle wrap-around: dihedral is periodic
dih_deviation = np.minimum(dih_deviation, 360 - dih_deviation)

print(f"Dihedral stats: mean={np.mean(dihedrals):.1f} deg, std={np.std(dihedrals):.1f}")
print(f"Deviation from A-form: mean={np.mean(dih_deviation):.1f}, max={np.max(dih_deviation):.1f}")

# ── 3. For each residue, compute shortest path distance to any far-end pair ──
# "Is this residue on a long-range pair connection path?"
# Build a simple path: for each far pair (i,j), mark all residues between i and j
# along the shorter arc of the circle

def circular_distance(i, j, L):
    """Shortest distance along circular sequence."""
    d = abs(i - j)
    return min(d, L - d)

def circular_path_residues(i, j, L):
    """Return residues on the shorter arc between i and j."""
    d = abs(i - j)
    if d > L - d:
        # shorter path goes the other way
        start, end = j, i
        if start > end:
            return list(range(start, L)) + list(range(0, end + 1))
        else:
            return list(range(start, end + 1))
    else:
        start, end = min(i, j), max(i, j)
        return list(range(start, end + 1))

# For each residue, find minimum circular distance to any far-end pair endpoint
min_dist_to_far = np.full(L, L // 2, dtype=float)
n_far_pairs_touching = np.zeros(L, dtype=int)

for i, j, w in far_pairs:
    for res_idx in range(L):
        d = circular_distance(res_idx, i, L)
        d2 = circular_distance(res_idx, j, L)
        min_d = min(d, d2)
        if min_d < min_dist_to_far[res_idx]:
            min_dist_to_far[res_idx] = min_d
        # Count how many far pairs have this residue on their path
        path = set(circular_path_residues(i, j, L))
        if res_idx in path:
            n_far_pairs_touching[res_idx] += 1

# Normalize: how "central" is this residue to far-end pairing
# 0 = far from any far pair, 1 = on many far pair paths
far_involvement = n_far_pairs_touching / max(n_far_pairs_touching.max(), 1)

# ── 4. Correlation analysis ──
from scipy import stats
corr, pval = stats.spearmanr(dih_deviation, far_involvement)
print(f"\nSpearman correlation (dihedral deviation vs far-pair involvement):")
print(f"  rho = {corr:.3f}, p = {pval:.2e}")

# Also check: are top-20 worst dihedral residues on far-pair paths?
top20_worst = np.argsort(dih_deviation)[-20:]
top20_on_path = np.sum(n_far_pairs_touching[top20_worst] > 0)
print(f"  Top-20 worst dihedral residues on far-pair paths: {top20_on_path}/20")

# ── 5. Plot ──
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1, 1]})

# Panel 1: Dihedral deviation + far-pair involvement
ax1 = axes[0]
color1 = "#2563eb"
color2 = "#dc2626"
ax1.bar(np.arange(L), dih_deviation, color=color1, alpha=0.6, width=1.0, label="|Dihedral - 33 deg|")
ax1b = ax1.twinx()
ax1b.plot(np.arange(L), far_involvement, color=color2, linewidth=0.8, alpha=0.8, label="Far-pair involvement")
ax1b.fill_between(np.arange(L), far_involvement, alpha=0.1, color=color2)
ax1.set_ylabel("Dihedral deviation (deg)", color=color1, fontsize=10)
ax1b.set_ylabel("Far-pair involvement", color=color2, fontsize=10)
ax1.set_title(f"Dihedral Deviation vs Long-Range Pairing (L={L}, rho={corr:.3f}, p={pval:.1e})",
              fontsize=12, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9)
ax1b.legend(loc="upper right", fontsize=9)

# Mark far-pair endpoints
for i, j, w in far_pairs:
    ax1.axvline(i, color="green", alpha=0.1, linewidth=0.5)
    ax1.axvline(j, color="green", alpha=0.1, linewidth=0.5)

# Panel 2: Dihedral angle itself
ax2 = axes[1]
ax2.plot(np.arange(L), dihedrals, color="#059669", linewidth=0.6)
ax2.axhline(33, color="red", linestyle="--", linewidth=1, alpha=0.7, label="A-form (33 deg)")
ax2.set_ylabel("Dihedral (deg)", fontsize=10)
ax2.legend(fontsize=9)
ax2.set_ylim(-180, 180)

# Panel 3: Scatter - dihedral deviation vs far-pair involvement
ax3 = axes[2]
ax3.scatter(far_involvement, dih_deviation, s=3, alpha=0.3, c="#6366f1")
# Highlight top-20 worst
ax3.scatter(far_involvement[top20_worst], dih_deviation[top20_worst],
            s=30, facecolors="none", edgecolors="red", linewidths=1.5, label="Top-20 worst")
ax3.set_xlabel("Far-pair involvement (fraction of far pairs passing through)", fontsize=10)
ax3.set_ylabel("|Dihedral - 33| (deg)", fontsize=10)
ax3.legend(fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUT, "fig_torsion_vs_pairs.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
