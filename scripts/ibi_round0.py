"""IBI round 0: does the shipped full field reproduce the reference bonded marginals?

scripts/sample_bonded_chain.py removed the nonbonded terms, and the chain unfolded -- correctly,
since a bonded-only chain has no reason to stay folded. But that made the acceptance test fail
for a reason that has nothing to do with the potentials, and it left the actual question
unanswered.

The variance decomposition says only 2.6 to 10.5 percent of the pooled sigma is between
structures; the rest is residue-to-residue spread inside each chain. So the pooled reference
is close to a single chain's own distribution, and a single chain sampled with the FULL field
is a fair stand-in. That is the run this script does.

What it measures is the IBI residual, before any update:

    dU(q) = kBT * ln( P_sim(q) / P_ref(q) )

If the shipped field already matches the reference, dU is flat and IBI has nothing to do. If it
is not flat, this is the correction IBI would apply on its first round, and its size says
whether the rest of the loop is worth running.

The field is used exactly as shipped, including the 200 kJ/mol/nm force cap, because that is
what the pipeline runs.

Run: python scripts/ibi_round0.py [n_rep] [n_steps] [struct_idx]
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

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FRICTION = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 25
NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
SEED = 20260218

z = np.load(NPZ)
TAB = {}
for name in B.COORDS:
    TAB[name] = {"centre": z[f"{name}__centre"], "U": z[f"{name}__U"],
                 "binw": float(z[f"{name}__binw"]), "sigma": float(z[f"{name}__sigma"]),
                 "lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"])}

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[IDX]
L = len(s0["pos"])
print(f"structure {s0['name']}  L={L}  pairs={len(s0['pairs'])}")
print(f"{NREP} replicas, {NSTEPS} steps of 0.002 ps = {NSTEPS * 0.002:.1f} ps per replica")
print(f"full field as shipped, 300 K, mass 110 Da, friction {FRICTION}/ps, "
      f"sampling every {STRIDE} steps")
# Provenance. Three earlier runs of this experiment were invalidated by a force-field defect
# found after they started -- a thermostat at 0.4 T, an effective mass 100x too large, and a
# K_INTRA 52x too soft -- and none of them recorded which field they had actually run against,
# so each result had to be judged by its numbers alone. Print the field's fingerprint instead.
print("field: " + "  ".join(
    f"{n}={getattr(C, n)}" for n in
    ("K_BB", "K_INTRA_PC", "K_INTRA_CN", "K_PAIR", "K_ANGLE", "K_DIH", "K_BPP",
     "K_STACK", "K_CLASH", "K_BSJ", "K_BSJ_GUIDE")))
print("second B half-kick: non-symplectic fallback (no force_fn passed). At gamma "
      f"{FRICTION} and dt 0.002 the pump is dt*omega^2/(4*gamma) = "
      f"{0.002 * 3.015 ** 2 / (4 * FRICTION):.4f} of the drag per step, so the stationary "
      "state is perturbed at that level and this is not the source of any large effect.")
print()

pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)

torch.manual_seed(SEED)
burn = NSTEPS // 5
# hoisted so the progress line can print the coupling ratio while the run is going
K_SHIPPED = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA, "intra_cn": C.K_INTRA,
             "angle": C.K_ANGLE, "dihedral": C.K_DIH, "stack": C.K_STACK}
counts = {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
acc = {c: [0.0, 0.0, 0] for c in B.COORDS}     # sum, sumsq, n
# Clash watch. The analytical claim about the intra-bead bonds assumes the repulsion never
# fires, and C4'-N sits at 0.335 nm against a 0.300 nm cutoff, so that is not free.
clash_min = []
clash_below = 0
import time
t0 = time.time()
for step in range(NSTEPS):
    with torch.no_grad():
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps,
                                         dt_ps=0.002, mass_amu=110.0,
                                         friction=FRICTION)
    if step == 0:
        print(f"first step {time.time() - t0:.3f} s")
    if step >= burn and step % STRIDE == 0:
        with torch.no_grad():
            for c in B.COORDS:
                q = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
                acc[c][0] += q.sum()
                acc[c][1] += (q ** 2).sum()
                acc[c][2] += q.size
                t = TAB[c]
                k = np.round((q - t["centre"][0]) / t["binw"]).astype(np.int64)
                ok = (k >= 0) & (k < len(t["U"]))
                counts[c] += np.bincount(k[ok], minlength=len(t["U"]))
            beads = pos.reshape(NREP, -1, 3)
            dd = torch.cdist(beads, beads)
            dd = dd + torch.eye(dd.shape[-1], device=dd.device) * 10.0
            clash_min.append(float(dd.min()))
            clash_below += int((dd < C.CLASH_DIST).sum())
    if (step + 1) % max(NSTEPS // 10, 1) == 0:
        el = time.time() - t0
        parts = []
        for c in B.COORDS:
            s1, s2, n = acc[c]
            if n:
                mm = s1 / n
                ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
                k = K_SHIPPED[c]
                s1d = np.sqrt(B.KBT / k) if k > 0 else np.nan
                parts.append(f"{c[:5]} {ss / s1d:5.2f}" if s1d == s1d
                             else f"{c[:5]}  n/a")
            else:
                parts.append(f"{c[:5]}   --")
        print(f"  {step+1:>7d} {el:6.0f}s  " + "  ".join(parts))

el = time.time() - t0
print(f"done in {el:.0f} s, {NSTEPS / el:.1f} steps/s")
print()
print(f"clash watch: cutoff {C.CLASH_DIST:.3f} nm, closest bead pair ever {min(clash_min):.4f} nm, "
      f"pairs below the cutoff over the run {clash_below}")
print()

# reference density from the stored table, on the same bins
def ref_p(c):
    U = TAB[c]["U"]
    p = np.exp(-(U - U.min()) / B.KBT)
    return p / p.sum()

# The single-coordinate prediction. Whatever the coupling does, a harmonic restraint of
# stiffness k on one coordinate has variance kBT/k on its own. If the shipped k was chosen as
# kBT/sigma_ref^2, then sigma_1D equals sigma_ref by construction and any gap between the
# sampled sigma and sigma_1D is the coupling, which is the only thing IBI can address.
print(f"{'coordinate':10s} {'ref sig':>8s} {'1-D sig':>8s} {'sim sig':>8s} "
      f"{'sim/ref':>8s} {'sim/1D':>7s} {'dU min':>8s} {'dU max':>8s} {'|dU|>1kBT':>10s}")
print("-" * 86)
rows = {}
for c in B.COORDS:
    t = TAB[c]
    s1, s2, n = acc[c]
    m = s1 / n
    sig = float(np.sqrt(max(s2 / n - m * m, 0.0)))
    ps = (counts[c] + 1.0) / (counts[c].sum() + len(counts[c]))
    pr = ref_p(c)
    dU = B.KBT * np.log(ps / pr)
    ref_m = float((t["centre"] * pr).sum())
    frac = float((np.abs(dU) > B.KBT).mean())
    k = K_SHIPPED[c]
    s1d = float(np.sqrt(B.KBT / k)) if k > 0 else float("nan")
    rows[c] = (m, sig, dU)
    print(f"{c:10s} {t['sigma']:8.4f} {s1d:8.4f} {sig:8.4f} "
          f"{sig / t['sigma']:8.3f} {sig / s1d:7.3f} {dU.min():8.2f} {dU.max():8.2f} "
          f"{frac:10.3f}")
print()
print("sim/1D is the coupling correction, and it is the whole content of the IBI step:")
print("where it is 1.0 the coupled chain already reproduces the single-coordinate result.")
print("sim/ref mixes that with the choice of k itself, which is a separate question.")
print()
print("=== the correction at the reference mode, and its curvature ===")
print(f"{'coordinate':10s} {'mode q':>9s} {'dU(mode)':>10s} {'dU at +1 sig':>13s} "
      f"{'dU at -1 sig':>13s}")
print("-" * 60)
for c in B.COORDS:
    t = TAB[c]
    m, sig, dU = rows[c]
    i0 = int(np.argmin(t["U"]))
    idx = lambda q: int(np.clip(np.round((q - t["centre"][0]) / t["binw"]), 0, len(dU) - 1))
    q0 = float(t["centre"][i0])
    print(f"{c:10s} {q0:9.4f} {dU[i0]:10.2f} {dU[idx(q0 + t['sigma'])]:13.2f} "
          f"{dU[idx(q0 - t['sigma'])]:13.2f}")
