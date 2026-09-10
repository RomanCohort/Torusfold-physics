"""Which term of the CG force field carries the force at a near-native geometry?

The decomposition lives in cg_force_terms.py so that recalibrate_ff_targets.py can reuse
it. This script is the driver: it builds the 1EHZ start geometry and prints the table.

Background. scripts/relax_with_and_without_cgrnasp.py reported a median per-bead force of
2000 in the units it printed. That figure came from multiplying cg_energy_forces' output by
10 on the way out of nm; the factor belongs the other way (1 nm = 10 A, so kJ/mol/nm ->
kJ/mol/A is a division). The real number is therefore 200 kJ/mol/nm = 20 kJ/mol/A, and it
is EXACTLY the hard cap that cg_energy_forces applies at its last line. A median sitting on
the cap means the cap is not a safety net, it is the effective force law.

That matters beyond this script, because the production path cg_forces_explicit_batched has
no such cap at all, so whatever the raw magnitude is, that is what the integrator sees.

Run: python scripts/diagnose_cg_force_terms.py [--recalibrated]
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
from cg_force_terms import term_energies_forces
from torusfold.scheme2.aform_from_template import real_cg_beads
import torusfold.scheme2.torch_cgsim as C

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order)
seq = "".join(base_of)
pairs = [(a, b) for a, b in partner.items() if a < b]

beads_A = real_cg_beads(ps, seq, pairs=pairs)          # (L, 3, 3) Angstrom
pos = torch.tensor(beads_A.reshape(1, 3 * L, 3) / 10.0, dtype=torch.float64)   # nm
pi_t = torch.tensor([a for a, _ in pairs], dtype=torch.long)
pj_t = torch.tensor([b for _, b in pairs], dtype=torch.long)
pw = torch.ones(len(pairs), dtype=torch.float64)
pairs_ij = torch.stack([pi_t, pj_t], 1)

cl = C.GPUCellList(cell_size=1.5)
cl.build(pos)

if "--recalibrated" in sys.argv:
    C.STACK_R0 = 1.125
    C.DIH_PPPP = math.acos(0.975)
    print("recalibrated: STACK_R0 1.125 nm, DIH cos +0.975")
else:
    print("as shipped")

terms, energies = term_energies_forces(pos, pairs_ij, pw, cell_list=cl)
raw = sum(terms.values())
e_lib, f_lib_capped = C.cg_energy_forces(pos, pairs_ij, pw, cell_list=cl)
lib_cap_np = f_lib_capped.reshape(3 * L, 3).numpy()
try:
    e_batch, f_batch = C.cg_forces_explicit_batched(pos, pairs_ij, pw, cell_list=cl)
    batch_np = f_batch.reshape(3 * L, 3).numpy()
except Exception as exc:
    print(f"cg_forces_explicit_batched not callable as assumed: {type(exc).__name__}: {exc}")
    batch_np = None


def stat(F):
    m = np.linalg.norm(F, axis=1)
    return m.mean(), np.percentile(m, 95), m.max(), int((m > 200.0).sum())


print(f"1EHZ, L = {L}, {len(pairs)} WC pairs, near-native geometry")
print("unit: kJ/mol/nm (the library's internal unit). 200 kJ/mol/nm = 20 kJ/mol/A.")
print()
print(f"{'term':20s} {'mean':>9s} {'p95':>9s} {'max':>10s} {'#beads>200':>11s}")
print("-" * 62)
for name, F in sorted(terms.items(), key=lambda kv: -stat(kv[1])[1]):
    mn, p95, mx, n = stat(F)
    print(f"{name:20s} {mn:9.2f} {p95:9.2f} {mx:10.2f} {n:11d}")
print("-" * 62)
for label, F in [("SUM of terms above", raw), ("library capped", lib_cap_np)]:
    mn, p95, mx, n = stat(F)
    print(f"{label:20s} {mn:9.2f} {p95:9.2f} {mx:10.2f} {n:11d}")
if batch_np is not None:
    mn, p95, mx, n = stat(batch_np)
    print(f"{'library UNCAPPED':20s} {mn:9.2f} {p95:9.2f} {mx:10.2f} {n:11d}"
          f"   <- production path (line 1074)")
print()
fmag = np.linalg.norm(lib_cap_np, axis=1)
print(f"beads sitting exactly on the 200 kJ/mol/nm cap: {(np.abs(fmag-200)<1e-6).sum()} / {3*L}")
print()

e_mine = sum(energies.values())
print("energy cross-check (kJ/mol) -- a decomposition is only believable if it reproduces")
print("the library's energy")
print(f"  sum of my 13 terms                 {e_mine:14.4f}")
print(f"  cg_energy_forces total_E           {float(e_lib.reshape(-1)[0]):14.4f}")
if batch_np is not None:
    print(f"  cg_forces_explicit_batched total_E {float(e_batch.reshape(-1)[0]):14.4f}")
print("  (cg_energy_forces adds GB/SA + Manning on top of the 13 terms; the batch path")
print("   implements its own set, so exact agreement is not expected -- the point is")
print("   whether the two library paths agree with EACH OTHER.)")
print()

print("geometry driving each term")
print(f"  P(0)-P({L-1}) distance            "
      f"{float(torch.linalg.norm(pos[0,3*0]-pos[0,3*(L-1)])):6.3f} nm"
      f"   (bsj closure target {C.BOND_P_NEXT} nm; 1EHZ is a tRNA, not a circle)")
nn = torch.linalg.norm(pos[0, 3 * pi_t + 2] - pos[0, 3 * pj_t + 2], dim=-1).numpy()
pp = torch.linalg.norm(pos[0, 3 * pi_t] - pos[0, 3 * pj_t], dim=-1).numpy()
print(f"  WC pair N-N distance    {nn.min():6.3f} .. {nn.mean():.3f} .. {nn.max():6.3f} nm"
      f"   (pair/bpp target {C.PAIR_NN} nm)")
print(f"  WC pair P-P distance    {pp.min():6.3f} .. {pp.mean():.3f} .. {pp.max():6.3f} nm"
      f"   (pair guide target {C.PAIR_NN} nm)")
st = torch.linalg.norm(pos[0, 3 * torch.arange(L - 2)] - pos[0, 3 * torch.arange(2, L)],
                       dim=-1).numpy()
print(f"  P(i)-P(i+2) distance    {st.min():6.3f} .. {st.mean():.3f} .. {st.max():6.3f} nm"
      f"   (stacking target {C.STACK_R0:.3f} nm)")
pi_, pj_, delta_, dist_ = cl.get_pair_info(pos)
print(f"  cell-list pairs under the {C.CLASH_DIST} nm clash cutoff: "
      f"{int((dist_[0] < C.CLASH_DIST).sum())} / {len(dist_[0])}")
