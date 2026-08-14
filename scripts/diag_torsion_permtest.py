"""Permutation test: are top-20 worst dihedral residues on far-pair paths?"""
import sys, os
sys.path.insert(0, "C:/Users/颜子壹/TorusFold-scheme2-rl/src")

import numpy as np
from openmm.app import PDBFile

OUT = "C:/Users/颜子壹/TorusFold-scheme2-rl/output_2013nt"

# ── Load P coordinates ──
pdb = PDBFile(os.path.join(OUT, "isrnaclong_final.pdb"))
L_res = list(pdb.topology.residues())
L = len(L_res)
p_coords = np.zeros((L, 3), dtype=np.float64)
for i, res in enumerate(L_res):
    for atom in res.atoms():
        if atom.name.strip() == "P":
            pos = pdb.positions[atom.index]
            p_coords[i] = [pos.x, pos.y, pos.z]

# ── Compute dihedrals ──
def compute_dihedral(p0, p1, p2, p3):
    v1, v2, v3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    n1 /= np.linalg.norm(n1) + 1e-8
    n2 /= np.linalg.norm(n2) + 1e-8
    cos_d = np.clip(np.dot(n1, n2), -1, 1)
    sin_d = np.dot(np.cross(n1, n2), v2 / (np.linalg.norm(v2) + 1e-8))
    return np.degrees(np.arctan2(sin_d, cos_d))

dihedrals = np.array([
    compute_dihedral(p_coords[(i-1)%L], p_coords[i], p_coords[(i+1)%L], p_coords[(i+2)%L])
    for i in range(L)
])
dih_dev = np.abs(dihedrals - 33.0)
dih_dev = np.minimum(dih_dev, 360 - dih_dev)

# ── Build far-pair path mask ──
bpp = np.load(os.path.join(OUT, "pair_heatmap.npy"))
far_pairs = [(i, j) for i in range(L) for j in range(i+51, L) if bpp[i][j] > 0.1]

on_path = np.zeros(L, dtype=bool)
for i, j in far_pairs:
    # Mark the shorter arc
    d = abs(i - j)
    if d <= L - d:
        on_path[i:j+1] = True
    else:
        on_path[j:] = True
        on_path[:i+1] = True

n_on_path = on_path.sum()
n_far_pairs = len(far_pairs)
print(f"L={L}, far_pairs={n_far_pairs}, residues_on_any_path={n_on_path}/{L}")

# ── Top-20 worst dihedrals ──
top20 = np.argsort(dih_dev)[-20:]
observed_on_path = on_path[top20].sum()
print(f"Observed: {observed_on_path}/20 top-20 worst on far-pair paths")

# ── Permutation test ──
np.random.seed(42)
N_PERM = 100000
exceed_count = 0
for _ in range(N_PERM):
    perm_idx = np.random.choice(L, 20, replace=False)
    n = on_path[perm_idx].sum()
    if n >= observed_on_path:
        exceed_count += 1

p_value = exceed_count / N_PERM
print(f"Permutation test (N={N_PERM}):")
print(f"  P(on_path >= {observed_on_path}/20) = {p_value:.6f}")
if p_value < 0.001:
    print(f"  *** Significant at p < 0.001 ***")
elif p_value < 0.01:
    print(f"  ** Significant at p < 0.01 **")
elif p_value < 0.05:
    print(f"  * Significant at p < 0.05 *")
else:
    print(f"  Not significant (p >= 0.05)")

# ── Also test top-10 ──
top10 = np.argsort(dih_dev)[-10:]
obs10 = on_path[top10].sum()
exc10 = 0
for _ in range(N_PERM):
    perm_idx = np.random.choice(L, 10, replace=False)
    if on_path[perm_idx].sum() >= obs10:
        exc10 += 1
p10 = exc10 / N_PERM
print(f"\nTop-10: {obs10}/10 on path, permutation p = {p10:.6f}")

# ── Correlation ──
from scipy import stats
rho, pval = stats.spearmanr(dih_dev, on_path.astype(float))
print(f"\nSpearman (dihedral_dev vs on_path): rho={rho:.4f}, p={pval:.4e}")

# ── Energy analysis: are high-dihedral-residue regions higher energy? ──
# Check if dihedral deviation correlates with local P-P distance (strain)
local_strain = np.zeros(L)
for i in range(L):
    neighbors = [(i-2)%L, (i-1)%L, (i+1)%L, (i+2)%L]
    dists = [np.linalg.norm(p_coords[i] - p_coords[n]) for n in neighbors]
    local_strain[i] = np.std(dists)

rho2, pval2 = stats.spearmanr(dih_dev, local_strain)
print(f"Spearman (dihedral_dev vs local_strain): rho={rho2:.4f}, p={pval2:.4e}")
