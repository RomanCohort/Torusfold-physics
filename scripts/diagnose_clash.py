"""Why does the clash term never fire, and is the batch-0 mask a real bug?

3x and 3aa both report zero clash energy and zero clash force on native structures, and 3ab
listed that as unexplained: either nothing is inside the 0.30 nm threshold, or the term is
broken.

Two separate questions.

(a) How close do beads actually get? Needs measuring before anything can be concluded.

(b) The mask in _clash_f is built from dist[0] -- batch element 0 only -- and then applied to
every replica. torch_cgsim.py:1379 carries the comment "judge using only the first batch",
so it is deliberate or at least known, but the pipeline runs 64 replicas and a pair that
clashes in replica 3 and not in replica 0 would get no force at all.

Run: python scripts/diagnose_clash.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

structs = [s for s in B.load_structures(limit=200) if len(s["pairs"]) >= 4][:24]
print(f"{len(structs)} native structures, {sum(len(s['pos']) * 3 for s in structs)} beads")
print()

# ---- (a) how close do beads get --------------------------------------------
mins = []
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    pi, pj, delta, dist = cl.get_pair_info(pos)
    if len(dist[0]):
        mins.append(float(dist[0].min()))
mins = np.array(mins)
print(f"(a) minimum bead-bead distance per structure, over the cell list's pair set")
print(f"    min {mins.min():.3f} nm, median {np.median(mins):.3f} nm, max {mins.max():.3f} nm")
print(f"    the clash cutoff CLASH_DIST is {C.CLASH_DIST} nm")
print(f"    structures with a pair under the cutoff: {int((mins < C.CLASH_DIST).sum())} / {len(mins)}")
print()

# ---- (b) does a clash in replica 1 register --------------------------------
L = 12
pos = torch.zeros(2, 3 * L, 3, dtype=torch.float64)
for i in range(L):
    pos[:, 3 * i, 0] = 1.0 * i
    pos[:, 3 * i + 1] = pos[:, 3 * i] + torch.tensor([0.2, 0.3, 0.1], dtype=torch.float64)
    pos[:, 3 * i + 2] = pos[:, 3 * i + 1] + torch.tensor([0.1, 0.2, 0.25], dtype=torch.float64)
pos = pos + torch.randn(2, 3 * L, 3, generator=torch.Generator().manual_seed(3),
                        dtype=torch.float64) * 0.5
pairs = torch.zeros((0, 2), dtype=torch.long)


def clash_energy(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    e, _f = C._clash_f(p, cl, C.K_CLASH, C.CLASH_SIGMA)
    return e.detach().numpy()


base = clash_energy(pos)
print(f"(b) a two-replica system")
print(f"    clash energy, both replicas as built        {base}")
# force a close approach in replica 1 only
pos2 = pos.clone()
pos2[1, 0] = pos2[1, 1] + torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64)
after = clash_energy(pos2)
print(f"    clash energy after closing a gap in replica 1 only   {after}")
print(f"    replica 1 moved by {float(torch.linalg.norm(pos2[1]-pos[1]).max()):.4f} nm")
pos3 = pos.clone()
pos3[0, 0] = pos3[0, 1] + torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64)
third = clash_energy(pos3)
print(f"    clash energy after closing a gap in replica 0 only   {third}")
print()
print(f"    direct distance between beads 0 and 1, replica 0: "
      f"{float(torch.linalg.norm(pos2[0,0]-pos2[0,1])):.4f} nm")
print(f"    direct distance between beads 0 and 1, replica 1: "
      f"{float(torch.linalg.norm(pos2[1,0]-pos2[1,1])):.4f} nm")
