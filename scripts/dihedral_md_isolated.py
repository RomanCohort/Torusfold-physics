"""V1: is the dihedral 1-D sigma 0.351 (not 0.588), checked in a real MD that does NOT
assume the flat-phi measure?

scripts/dihedral_measure_1d.py got sigma(cos phi) = 0.351 from two routes that share one
assumption: that a single torsion's equilibrium measure is flat in phi (dphi). That measure
is what MAKES the result 0.351 instead of the sqrt(kBT/k) = 0.588 upper bound, so a skeptical
reader is entitled to ask for a route that does not assume it. This is that route.

We integrate a real 4-atom (P0-P1-P2-P3) chain with the repository's own BAOAB Langevin
integrator (batch_langevin_step) and its own force helpers. The geometry is held so that phi
is the ONLY soft coordinate, but the holding is done with STIFF harmonic restraints, not by
freezing or zeroing:

    _bond_f  on the three P-P bonds,   k = 2e4 kJ/mol/nm^2   (r0 = 0.590 nm)
    _angle_f on the two P-P-P angles,  k = 2e3                (cos 120 deg)
    _dihedral_f on the one torsion,    k = K_DIH = 7.2        (cos 12.84 deg, the shipped term)

The shipped K_BB / K_ANGLE are NOT used and NOT set to zero -- the geometry cannot fall apart,
which is the failure mode a "zero the rest" control would have. The bond and angle sigmas are
reported and are ~1 percent of their targets, so the only coordinate that actually fluctuates
is the torsion.

Why this is a clean test of the measure, not of the restraints: the dihedral energy
0.5*K*(cos phi - c)^2 depends only on phi, and the bond/angle restraints depend only on
lengths and angles, so the Boltzmann factor factorises and the equilibrium marginal of phi is
exp(-U_dih/kBT) times the measure induced on phi by the Cartesian embedding of a genuine
torsion. Whatever that induced measure is, the sampled sigma(cos phi) reports it directly:
0.351 confirms flat-phi (the measure argument), 0.588 would refute it.

Run: python scripts/dihedral_md_isolated.py [n_steps] [burn] [stride] [seed]
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
BURN = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 20
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 20260218
FRICTION = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
MASS = float(sys.argv[6]) if len(sys.argv) > 6 else 110.0

torch.manual_seed(SEED)

KBT = B.KBT
K_DIH = C.K_DIH
C_DIH = float(np.cos(C.DIH_PPPP))
K_BOND_HOLD = 2e4          # hold the three bonds (stiff, NOT the shipped K_BB)
K_ANGLE_HOLD = 2e3         # hold the two angles
BOND0 = 0.590
THETA = np.radians(120.0)  # non-collinear so the torsion is a genuine free rotor
C_ANGLE = float(np.cos(THETA))

# Build P0-P1-P2-P3 with P1-P2 along +x, so the torsion phi is rotation about x.
b = BOND0
p1 = torch.tensor([0.0, 0.0, 0.0])
p2 = torch.tensor([b, 0.0, 0.0])
p0 = torch.tensor([b * np.cos(THETA), b * np.sin(THETA), 0.0])
phi0 = np.radians(12.84)
p3 = torch.tensor([b + b * np.cos(np.pi - THETA),
                   b * np.sin(np.pi - THETA) * np.cos(phi0),
                   b * np.sin(np.pi - THETA) * np.sin(phi0)])
# The pipeline layout is (B, 3L, 3): L residues, three beads each. We keep only the four P
# atoms live (indices 0,3,6,9); the C4'/N beads are inert ghosts, untouched by the bond,
# angle and dihedral helpers (all of which read P(i)=3i+0), so they carry no force and do not
# couple back to the torsion.
_p = torch.stack([p0, p1, p2, p3])                      # (4, 3)
pos = torch.zeros(1, 12, 3, dtype=torch.float64)
pos[:, 0::3, :] = _p.unsqueeze(0)
pos[:, 1::3, :] = _p.unsqueeze(0) + 0.1
pos[:, 2::3, :] = _p.unsqueeze(0) + 0.2
vel = torch.zeros_like(pos)
temps = torch.full((1,), 300.0, dtype=torch.float64)

P = lambda i: 3 * i + 0
idx_b = torch.arange(3)          # bonds P(i)-P(i+1), i=0,1,2


def forces_at(p):
    e1, f1 = C._bond_f(p, P(idx_b), P(idx_b + 1), K_BOND_HOLD, BOND0)
    e2, f2 = C._angle_f(p, K_ANGLE_HOLD, C_ANGLE)
    e3, f3 = C._dihedral_f(p, K_DIH, C_DIH)
    return f1 + f2 + f3


def qcos(p):
    """cos of the single torsion, the same normal-based definition the pipeline uses."""
    return float(B._cos_dihedral(p[:, P(0)], p[:, P(1)], p[:, P(2)], p[:, P(3)])[0])


print(f"isolated-torsion MD: 4 P atoms, BAOAB Langevin, dt=0.002 ps, friction={FRICTION}, mass={MASS} amu", flush=True)
print(f"  hold: 3 bonds k={K_BOND_HOLD:g} at r0={BOND0}; 2 angles k={K_ANGLE_HOLD:g} at "
      f"cos({np.degrees(THETA):.0f} deg)={C_ANGLE:.3f}", flush=True)
print(f"  dihedral: K_DIH={K_DIH}, target cos={C_DIH:.4f} (phi0={np.degrees(np.arccos(C_DIH)):.2f} deg)", flush=True)
print(f"  initial cos(phi) = {qcos(pos):.4f}", flush=True)
print(f"  KBT = {KBT:.4f}; reference sigma = {float(np.load(REPO / 'results' / 'boltzmann_tables_clean.npz')['dihedral__sigma']):.4f}", flush=True)
print(flush=True)

samples = []
bond_sq = 0.0
angle_cos_sq = 0.0
n = 0
# disjoint blocks of the sampling window, to make convergence visible rather than assumed
NB = 5
blk = [[] for _ in range(NB)]
for step in range(NSTEPS):
    with torch.no_grad():
        f = forces_at(pos)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps,
                                         dt_ps=0.002, mass_amu=MASS,
                                         friction=FRICTION, force_fn=forces_at)
    if step >= BURN and step % STRIDE == 0:
        samples.append(qcos(pos))
        bi = min(((step - BURN) * NB) // max(NSTEPS - BURN, 1), NB - 1)
        blk[bi].append(samples[-1])
        # bond/angle sigma to prove the geometry is held
        r0 = torch.norm(pos[:, P(1)] - pos[:, P(0)], dim=-1)[0]
        r1 = torch.norm(pos[:, P(2)] - pos[:, P(1)], dim=-1)[0]
        r2 = torch.norm(pos[:, P(3)] - pos[:, P(2)], dim=-1)[0]
        bond_sq += (float(r0) ** 2 + float(r1) ** 2 + float(r2) ** 2)
        ca = B._cos_angle(pos[:, P(0)], pos[:, P(1)], pos[:, P(2)])[0]
        cb = B._cos_angle(pos[:, P(1)], pos[:, P(2)], pos[:, P(3)])[0]
        angle_cos_sq += float(ca) ** 2 + float(cb) ** 2
        n += 1

s = np.array(samples)
print(f"sampled {len(s)} frames (steps {BURN}..{NSTEPS} stride {STRIDE})")
print(f"  mean cos(phi)  = {s.mean():.4f}")
print(f"  sigma cos(phi) = {s.std(ddof=0):.4f}")
print(f"  bond rms       = {np.sqrt(bond_sq / (3 * n)):.5f} nm   (target {BOND0}, held)")
print(f"  angle cos rms  = {np.sqrt(angle_cos_sq / (2 * n)):.5f} (target {C_ANGLE:.3f}, held)")
print()
print("per-block mean / sigma of cos(phi), disjoint equal-time blocks:")
print(f"{'block':>5s} {'mean':>9s} {'sigma':>8s}")
for bi, b in enumerate(blk):
    bb = np.array(b)
    if len(bb):
        print(f"{bi:5d} {bb.mean():9.4f} {bb.std(ddof=0):8.4f}")
    else:
        print(f"{bi:5d} {'--':>9s} {'--':>8s}")
print("  a flat block series = converged; a monotone drift = still relaxing.")
print()
print("interpretation:")
print(f"  sqrt(kBT/k) upper bound        : {np.sqrt(KBT / K_DIH):.4f}")
print(f"  quadrature prediction (flat phi): 0.3507")
print(f"  this real MD                    : {s.std(ddof=0):.4f}")
print("  if the converged MD lands near 0.351, the measure argument is confirmed empirically;")
print("  if near 0.588, the flat-phi assumption was wrong and the mechanism needs rethinking.")
