"""Which term of the CG force field blows up at a near-native geometry?

Background. scripts/relax_with_and_without_cgrnasp.py reported a median per-bead force of
2000 in the units it printed. That figure came from multiplying cg_energy_forces' output by
10 on the way out of nm; the factor belongs the other way (1 nm = 10 A, so kJ/mol/nm ->
kJ/mol/A is a division). The real number is therefore 200 kJ/mol/nm = 20 kJ/mol/A, and it
is EXACTLY the hard cap that cg_energy_forces applies at its last line. A median sitting on
the cap means the cap is not a safety net, it is the effective force law: essentially every
bead is saturated.

That matters beyond this script, because the production path cg_forces_explicit_batched has
no such cap at all, so whatever the raw magnitude is, that is what the integrator sees.

This script decomposes the raw force term by term at the start geometry, using the library's
own helper functions so the terms are the library's, not a reimplementation. All forces are
reported in kJ/mol/nm (the library's internal unit) and kJ/mol/A.

Run: python scripts/diagnose_cg_force_terms.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
from torusfold.scheme2.aform_from_template import real_cg_beads
import torusfold.scheme2.torch_cgsim as C

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order)
seq = "".join(base_of)
pairs = [(a, b) for a, b in partner.items() if a < b]

beads_A = real_cg_beads(ps, seq, pairs=pairs)          # (L, 3, 3) Angstrom
pos = torch.tensor(beads_A.reshape(1, 3 * L, 3) / 10.0, dtype=torch.float64)   # nm
P = lambda i: 3 * i + 0
C4 = lambda i: 3 * i + 1
NN = lambda i: 3 * i + 2

pi_t = torch.tensor([a for a, _ in pairs], dtype=torch.long)
pj_t = torch.tensor([b for _, b in pairs], dtype=torch.long)
pw = torch.ones(len(pairs), dtype=torch.float64)

terms = {}
energies = {}


def add(name, F, e=None):
    terms[name] = F.reshape(3 * L, 3).numpy()
    if e is not None:
        energies[name] = float(e.reshape(-1)[0])


# 1. backbone P-P bonds
idx = torch.arange(L - 1)
e_, f = C._bond_f(pos, P(idx), P(idx + 1), C.K_BB, C.BOND_P_NEXT)
add("bb bond P-P", f, e_)

# 2/3. intra-bead
r = torch.arange(L)
e1_, f1 = C._bond_f(pos, P(r), C4(r), C.K_INTRA, C.BOND_P_C4)
e2_, f2 = C._bond_f(pos, C4(r), NN(r), C.K_INTRA, C.BOND_C4_N)
add("intra P-C4'", f1, e1_)
add("intra C4'-N", f2, e2_)

# 4. BSJ closure bond  (P0 - P(L-1))
d = pos[:, P(0)] - pos[:, P(L - 1)]
r_bsj = torch.linalg.norm(d, dim=-1, keepdim=True)
fb = (-C.K_BSJ * (r_bsj - C.BOND_P_NEXT) * d / r_bsj).squeeze(-1)
Fb = torch.zeros_like(pos)
Fb[:, P(0)] += fb
Fb[:, P(L - 1)] -= fb
add("bsj closure", Fb, (0.5*C.K_BSJ*(r_bsj.squeeze(-1)-C.BOND_P_NEXT)**2))

# 5/6. angles, dihedrals
ea_, fa = C._angle_f(pos, C.K_ANGLE, math.cos(C.ANGLE_PPP))
ed_, fd = C._dihedral_f(pos, C.K_DIH, math.cos(C.DIH_PPPP))
add("angle P-P-P", fa, ea_)
add("dihedral P-P-P-P", fd, ed_)

# 7. WC pairing on N beads
if len(pairs):
    delta = pos[:, NN(pi_t)] - pos[:, NN(pj_t)]
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
    ke = C.K_PAIR * torch.ones(1, dtype=torch.float64)[None, :, None]
    fp = -ke * (dist - C.PAIR_NN) * delta / dist
    Fp = torch.zeros_like(pos)
    Fp[:, NN(pi_t)] += fp.squeeze(-1)
    Fp[:, NN(pj_t)] -= fp.squeeze(-1)
    add("wc pair N-N", Fp, (0.5*C.K_PAIR*(dist-C.PAIR_NN)**2).sum(-1).sum(-1))

# 8. stacking on P_i - P_{i+2}
if L > 2:
    st = torch.arange(L - 2)
    ds = pos[:, P(st)] - pos[:, P(st + 2)]
    rs = torch.linalg.norm(ds, dim=-1, keepdim=True)
    fs = -C.K_STACK * (rs - C.STACK_R0) * ds / rs
    Fs = torch.zeros_like(pos)
    Fs[:, P(st)] += fs.squeeze(-1)
    Fs[:, P(st + 2)] -= fs.squeeze(-1)
    add("stacking P-P", Fs, (0.5*C.K_STACK*(rs-C.STACK_R0)**2).sum(-1).sum(-1))

# 9. clash
cl = C.GPUCellList(cell_size=1.5)
cl.build(pos)
ec_, fc = C._clash_f(pos, cl, C.K_CLASH, C.CLASH_DIST)
add("clash", fc, ec_)

# 10. pair guide (P-P of WC pairs, sigmoid)
if len(pairs):
    dg = pos[:, P(pi_t)] - pos[:, P(pj_t)]
    rg = torch.linalg.norm(dg, dim=-1, keepdim=True)
    _, sig = C._sigmoid_f(rg, C.PAIR_NN, C.K_PAIR_GUIDE, 0.2)
    fg = -C.K_PAIR_GUIDE / 0.2 * sig * dg / rg
    Fg = torch.zeros_like(pos)
    Fg[:, P(pi_t)] += fg.squeeze(-1)
    Fg[:, P(pj_t)] -= fg.squeeze(-1)
    add("pair guide", Fg, C._sigmoid_f(rg, C.PAIR_NN, C.K_PAIR_GUIDE, 0.2)[0].sum(-1))

# 11. BSJ guide (P0 - P(L-1), sigmoid)
dbg = pos[:, P(0)] - pos[:, P(L - 1)]
rbg = torch.linalg.norm(dbg, dim=-1, keepdim=True)
_, sigbg = C._sigmoid_f(rbg, C.PAIR_NN, C.K_BSJ_GUIDE, 0.2)
fbg = -C.K_BSJ_GUIDE / 0.2 * sigbg * dbg / rbg
Fbg = torch.zeros_like(pos)
Fbg[:, P(0)] += fbg.squeeze(-1)
Fbg[:, P(L - 1)] -= fbg.squeeze(-1)
add("bsj guide", Fbg, C._sigmoid_f(rbg, C.PAIR_NN, C.K_BSJ_GUIDE, 0.2)[0].squeeze(-1))

# 12. BSJ contact
Fc = torch.zeros_like(pos)
if L > 16:
    for off in range(min(8, L // 2)):
        i1, i2 = off, L - 1 - off
        if i1 < i2:
            dc = pos[:, P(i1)] - pos[:, P(i2)]
            rc = torch.linalg.norm(dc, dim=-1, keepdim=True)
            wc = torch.exp(-0.1 * (rc / C.PAIR_NN))
            fcx = -C.K_BSJ_CONTACT * 0.1 / C.PAIR_NN * wc * dc / (rc * rc)
            Fc[:, P(i1)] += fcx.squeeze(-1)
            Fc[:, P(i2)] -= fcx.squeeze(-1)
add("bsj contact", Fc, torch.tensor(0.0))

# 13. BPP
if len(pairs):
    db = pos[:, NN(pi_t)] - pos[:, NN(pj_t)]
    rb = torch.linalg.norm(db, dim=-1, keepdim=True)
    xb = (C.PAIR_NN - rb) / 0.3
    sb = torch.sigmoid(xb)
    fbpp = -C.K_BPP / 0.3 * sb * db / rb
    Fbpp = torch.zeros_like(pos)
    Fbpp[:, NN(pi_t)] += fbpp.squeeze(-1)
    Fbpp[:, NN(pj_t)] -= fbpp.squeeze(-1)
    add("bpp", Fbpp, (-C.K_BPP*C._stable_softplus(xb)).sum(-1).sum(-1))

# --- library totals -------------------------------------------------------
e_lib, f_lib_capped = C.cg_energy_forces(pos, torch.stack([pi_t, pj_t], 1), pw, cell_list=cl)
try:
    e_batch, f_batch = C.cg_forces_explicit_batched(
        pos, torch.stack([pi_t, pj_t], 1), pw, cell_list=cl)
    have_batch = True
except Exception as exc:                                    # signature differences
    print(f"cg_forces_explicit_batched not callable as assumed: {type(exc).__name__}: {exc}")
    have_batch = False

raw = sum(terms.values())
raw_np = raw
lib_cap_np = f_lib_capped.reshape(3 * L, 3).numpy()
batch_np = f_batch.reshape(3 * L, 3).numpy() if have_batch else None

# Most terms are exactly zero on most beads, so a median is 0 by construction and useless
# for ranking them. Report the mean, the 95th percentile, and how many beads are over the
# cap of the capped path.
def stat(F):
    m = np.linalg.norm(F, axis=1)
    return m.mean(), np.percentile(m, 95), m.max(), int((m > 200.0).sum())


print(f"1EHZ, L = {L}, {len(pairs)} WC pairs, start geometry (P RMSD 0.480 A vs crystal)")
print("unit: kJ/mol/nm (the library's internal unit). 200 kJ/mol/nm = 20 kJ/mol/A.")
print()
print(f"{'term':20s} {'mean':>9s} {'p95':>9s} {'max':>10s} {'#beads>200':>11s}")
print("-" * 62)
rows = sorted(terms.items(), key=lambda kv: -stat(kv[1])[1])
for name, F in rows:
    mn, p95, mx, n = stat(F)
    print(f"{name:20s} {mn:9.2f} {p95:9.2f} {mx:10.2f} {n:11d}")
print("-" * 62)
for label, F in [("SUM of terms above", raw_np), ("library capped", lib_cap_np)]:
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

# --- energy cross-check ---------------------------------------------------
# A decomposition is only believable if it reproduces the library's energy.
e_mine = sum(energies.values())
print("energy cross-check (kJ/mol)")
print(f"  sum of my 13 terms                 {e_mine:14.4f}")
print(f"  cg_energy_forces total_E           {float(e_lib.reshape(-1)[0]):14.4f}")
if have_batch:
    print(f"  cg_forces_explicit_batched total_E {float(e_batch.reshape(-1)[0]):14.4f}")
    print("  (cg_energy_forces adds GB/SA + Manning on top of the 13 terms; the batch path")
    print("   implements its own set, so exact agreement is not expected -- the point is")
    print("   whether the two library paths agree with EACH OTHER.)")
print()

# --- geometry that drives the big terms -----------------------------------
def dist(a, b):
    return float(torch.linalg.norm(pos[0, a] - pos[0, b]))


print("geometry driving each term")
print(f"  P(0)-P({L-1}) distance            {dist(P(0), P(L-1)):6.3f} nm"
      f"   (bsj closure target {C.BOND_P_NEXT} nm; 1EHZ is a tRNA, not a circle)")
if len(pairs):
    nn = torch.linalg.norm(pos[0, NN(pi_t)] - pos[0, NN(pj_t)], axis=-1).numpy()
    pp = torch.linalg.norm(pos[0, P(pi_t)] - pos[0, P(pj_t)], axis=-1).numpy()
    print(f"  WC pair N-N distance    {nn.min():6.3f} .. {nn.mean():.3f} .. {nn.max():6.3f} nm"
          f"   (pair/bpp target {C.PAIR_NN} nm)")
    print(f"  WC pair P-P distance    {pp.min():6.3f} .. {pp.mean():.3f} .. {pp.max():6.3f} nm"
          f"   (pair guide target {C.PAIR_NN} nm)")
bb = torch.linalg.norm(pos[0, P(torch.arange(L-1))] - pos[0, P(torch.arange(1, L))], axis=-1).numpy()
print(f"  backbone P-P bonds      {bb.min():6.3f} .. {bb.mean():.3f} .. {bb.max():6.3f} nm"
      f"   (target {C.BOND_P_NEXT} nm)")
pi_, pj_, delta_, dist_ = cl.get_pair_info(pos)
print(f"  cell-list pairs under the {C.CLASH_DIST} nm clash cutoff: "
      f"{int((dist_[0] < C.CLASH_DIST).sum())} / {len(dist_[0])}")
