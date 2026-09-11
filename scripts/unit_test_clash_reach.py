"""Is the excluded volume actually delivering its force, and can the pair list see the pair?
"""
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

SIG = C.CLASH_SIGMA
K = C.K_CLASH
print(f"CLASH_SIGMA {SIG}  K_CLASH {K}")

print()
print("1. _clash_pair_energy_force, the law _clash_f calls")
print(f"{'r (nm)':>8s} {'E (kJ/mol)':>14s} {'|F| (kJ/mol/nm)':>18s}")
for r in (0.0863, 0.15, 0.25, 0.35, 0.3975, 0.40):
    d = torch.tensor([[[r, 0.0, 0.0]]], dtype=torch.float64)
    dist = torch.tensor([[r]], dtype=torch.float64)
    e, f = C._clash_pair_energy_force(d, dist, K, SIG)
    print(f"{r:8.4f} {float(e[0,0]):14.2f} "
          f"{float(torch.linalg.norm(f[0,0])):18.2f}")

print()
print("2. does GPUCellList + _clash_f find a |i-j|>=3 pair at 0.0863 nm?")
for sep_beads in (9, 27):
    NB = 48
    pos = torch.zeros(1, NB, 3, dtype=torch.float64)
    for b in range(NB):
        pos[0, b, 0] = 50.0 + 3.0 * b          # everything far apart
    j = sep_beads
    pos[0, j] = pos[0, 0] + torch.tensor([0.0863, 0.0, 0.0], dtype=torch.float64)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    npairs = len(cl.neighbor_pairs)
    if npairs:
        hit = ((cl.neighbor_pairs == 0).any(axis=1) & (cl.neighbor_pairs == j).any(axis=1))
        found = bool(hit.any())
    else:
        found = False
    e, F = C._clash_f(pos, cl, K, SIG)
    print(f"   bead 0 vs bead {j} at 0.0863 nm: pair list has {npairs} pairs, "
          f"contains (0,{j}) = {found}, E = {float(e[0]):.2f} kJ/mol, "
          f"|F on bead 0| = {float(torch.linalg.norm(F[0,0])):.2f} kJ/mol/nm")

print()
print("3. and through cg_energy_forces, on a real chain, is the walled pair's contribution there?")
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B   # noqa: E402
s = [x for x in B.load_structures(limit=50) if len(x["pos"]) >= 20][0]
L = len(s["pos"]); NB = 3 * L
ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
xs = torch.tensor(s["pos"].reshape(1, NB, 3), dtype=torch.float64)
cl = C.GPUCellList(cell_size=1.5); cl.build(xs)
print(f"   {s['name']}: {NB} beads, cell list holds {len(cl.neighbor_pairs)} pairs "
      f"(|i-j|>=3 only)")
d = torch.cdist(xs[0], xs[0]) + torch.eye(NB) * 1e6
i, j = divmod(int(d.argmin()), NB)
print(f"   closest pair at the start: beads ({i},{j}) d = {float(d[i,j]):.4f} nm, |i-j| = {abs(i-j)}")
print(f"   is it in the cell list: {bool(((cl.neighbor_pairs == i).any(axis=1) & (cl.neighbor_pairs == j).any(axis=1)).any())}")
