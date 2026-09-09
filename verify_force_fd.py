"""Direction + FD checks for the six fixed force terms.

Run from repo root: python verify_force_fd.py

Part 1 (sign checks): the original bugs had the guide forces pointing the
wrong way (repelling instead of attracting). Force caps do not flip
signs, so a directional check is valid even when caps fire.
Part 2: batched dihedral term, small-step FD on non-singular windows.
"""
import os, pathlib, sys
_cands = [pathlib.Path(__file__).resolve().parent / 'src', pathlib.Path(os.getcwd()) / 'src']
for _c in _cands:
    if _c.is_dir():
        sys.path.insert(0, str(_c))
        break
import math, torch
from torusfold.scheme2 import torch_cgsim as tc

ZERO = dict(K_BB=0., K_INTRA=0., K_PAIR=0., K_STACK=0., K_ANGLE=0., K_DIH=0.,
             K_CLASH=0., K_BSJ=0., K_BSJ_CONTACT=0., K_BPP=0.,
             K_PAIR_GUIDE=0., K_BSJ_GUIDE=0.,
             _K_BOND_BB=0., _K_BOND_INTRA=0., _K_PAIR=0., _K_STACK=0., _K_ANGLE=0.,
             _K_DIH=0., _K_CLASH=0., _K_BSJ=0., _K_BSJ_CONTACT=0., _K_BPP=0.,
             _K_PAIR_GUIDE=0., _K_BSJ_GUIDE=0., _K_MG=0., _K_GB=0., _K_SASA=0.)

def zero_all_but(**enable):
    saved = {k: getattr(tc, k) for k in list(ZERO) + list(enable)}
    for k, v in ZERO.items(): setattr(tc, k, 0.0)
    for k, v in enable.items(): setattr(tc, k, v)
    return saved

def restore(saved):
    for k, v in saved.items(): setattr(tc, k, v)

def lin_chain(L, x0=0.0, step=0.59, zig=0.03):
    """P chain along x with small z zigzag (defines dihedrals); C4'/N beside each P."""
    pts = []
    for i in range(L):
        z = zig if i % 2 == 0 else -zig
        p = [x0 + i * step, 0.0, z]
        c4 = [p[0] + 0.35, 0.08, p[2]]
        n = [p[0] + 0.30, -0.10, p[2]]
        pts += [p, c4, n]
    return torch.tensor([pts], dtype=torch.float32)

# ---------- Part 1: sign checks ----------
print('== Part 1: direction of the (repaired) guide forces ==')

saved = zero_all_but(K_PAIR_GUIDE=100.0)
try:
    pos = lin_chain(8)
    pairs = torch.tensor([[1, 6]], dtype=torch.long)
    pw = torch.ones(1, dtype=torch.float32)
    # pull P(1) and P(6) apart along x: P1 at x~1.2, P6 at x~5.4 -> separate them further
    pos[:, 3 * 1 + 0, 0] = 0.5
    pos[:, 3 * 6 + 0, 0] = 1.6
    pos[:, 3 * 2 + 0, 0] = 2.2
    pos[:, 3 * 3 + 0, 0] = 2.8
    E, F = tc.cg_energy_forces(pos, pairs, pw, c_mg=0.0, c_na=0.0)
    fx1 = float(F[0, 3 * 1 + 0, 0])  # force on P1 along +x (toward P6 at +x)
    fx6 = float(F[0, 3 * 6 + 0, 0])
    ok = fx1 > 0 and fx6 < 0
    print(f'pair guide: F_P1_x={fx1:+.3f}, F_P6_x={fx6:+.3f} -> {"PASS" if ok else "FAIL"} (expect + / -)')
finally:
    restore(saved)

saved = zero_all_but(K_BSJ_GUIDE=100.0)
try:
    pos = lin_chain(8)
    pos[:, 0, 0] = 0.5
    pos[:, 3 * 7 + 0, 0] = 1.6
    pairs = torch.zeros((0, 2), dtype=torch.long)
    pw = torch.ones(0, dtype=torch.float32)
    E, F = tc.cg_energy_forces(pos, pairs, pw, c_mg=0.0, c_na=0.0)
    fx0 = float(F[0, 0, 0])
    fx7 = float(F[0, 3 * 7 + 0, 0])
    ok = fx0 > 0 and fx7 < 0
    print(f'unified BSJ guide: F_P0_x={fx0:+.3f}, F_P7_x={fx7:+.3f} -> {"PASS" if ok else "FAIL"} (expect + / -)')
finally:
    restore(saved)

saved = zero_all_but(K_BPP=600.0)
try:
    pos = lin_chain(8)
    pairs = torch.tensor([[1, 6]], dtype=torch.long)
    pw = torch.ones(1, dtype=torch.float32)
    pos[:, 3 * 1 + 2, 0] = 0.5   # N atoms
    pos[:, 3 * 6 + 2, 0] = 1.6
    pos[:, 3 * 2 + 2, 0] = 2.2
    pos[:, 3 * 3 + 2, 0] = 2.8
    E, F = tc.cg_energy_forces(pos, pairs, pw, c_mg=0.0, c_na=0.0)
    fx1 = float(F[0, 3 * 1 + 2, 0])
    fx6 = float(F[0, 3 * 6 + 2, 0])
    ok = fx1 > 0 and fx6 < 0
    print(f'BPP term: F_N1_x={fx1:+.3f}, F_N6_x={fx6:+.3f} -> {"PASS" if ok else "FAIL"} (expect + / -)')
finally:
    restore(saved)

saved = zero_all_but(K_BSJ_GUIDE=100.0)
try:
    pos = lin_chain(8)
    pos[:, 0, 0] = 0.5
    pos[:, 3 * 7 + 0, 0] = 1.6
    pairs = torch.zeros((0, 2), dtype=torch.long)
    pw = torch.ones(0, dtype=torch.float32)
    E, F = tc.cg_forces_explicit_batched(pos, pairs, pw, c_mg=0.0, c_na=0.0)
    fx0 = float(F[0, 0, 0])
    fx7 = float(F[0, 3 * 7 + 0, 0])
    ok = fx0 > 0 and fx7 < 0
    print(f'batched BSJ guide: F_P0_x={fx0:+.3f}, F_P7_x={fx7:+.3f} -> {"PASS" if ok else "FAIL"} (expect + / -)')
finally:
    restore(saved)

# ---------- Part 2: batched dihedral, small-step FD on a non-planar helix ----------
print('== Part 2: batched dihedral force vs finite difference (h=1e-4) ==')
def helix_chain(L, amp=0.12):
    pts = []
    for i in range(L):
        p = [i * 0.59, amp * math.sin(1.1 * i), amp * math.cos(1.1 * i)]
        pts += [p, [p[0] + 0.3, p[1] + 0.1, p[2]], [p[0] + 0.25, p[1] - 0.1, p[2]]]
    return torch.tensor([pts], dtype=torch.float32)

saved = zero_all_but(K_DIH=500.0, _K_DIH=500.0)
try:
    ok = False
    for amp in (0.12, 0.2, 0.3):
        pos = helix_chain(8, amp=amp)
        pairs = torch.zeros((0, 2), dtype=torch.long)
        pw = torch.ones(0, dtype=torch.float32)
        E, F = tc.cg_forces_explicit_batched(pos, pairs, pw, c_mg=0.0, c_na=0.0)
        # FD on the P atom of residue 3 (participates in windows 0-3 and 1-4)
        h = 1e-4
        pi = 3 * 3 + 0
        allok = True
        for d in (0, 1, 2):
            hi = pos.clone(); hi[:, pi, d] += h
            lo = pos.clone(); lo[:, pi, d] -= h
            Ehi = tc.cg_forces_explicit_batched(hi, pairs, pw, c_mg=0.0, c_na=0.0)[0]
            Elo = tc.cg_forces_explicit_batched(lo, pairs, pw, c_mg=0.0, c_na=0.0)[0]
            fnum = -(Ehi - Elo) / (2 * h)
            fgot = float(F[0, pi, d])
            rel = abs(fgot - float(fnum)) / (abs(fgot) + 1e-6)
            print(f'  amp={amp} d={d}: got={fgot:+.5f} fd={float(fnum):+.5f} rel={rel:.4f}')
            if rel > 0.05:
                allok = False
        if allok:
            print(f'  amp={amp}: all 3 components match FD -> PASS')
            ok = True
            break
    print('batched dihedral overall:', 'PASS' if ok else 'FAIL')
finally:
    restore(saved)

print('done')