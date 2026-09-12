"""IBI round-0 residual and fold drift on the ONE genuinely circular reference, 2OIU.

ibi_round0.py measures whether the shipped full field reproduces the reference bonded
marginals, but it takes its structure from a pool filtered to
len(pairs)>=8 and 24<=len(pos)<=34. 2OIU is L=71 and is not in that pool, so the production
configuration -- the three BSJ terms ON -- has never been run against a structure on which they
are legitimate. This is that run.

Why 2OIU and why "file order". scripts/measure_bsj_on_circular_reference.py established that in
its file order residue 1 is already covalently bonded to residue 71 (O3'(71)-P(1) = 1.598 A), and
|P(0)-P(L-1)| = 0.5915 nm sits on the BSJ closure target BOND_P_NEXT = 0.590 nm. The production
target is a circular RNA, so the BSJ terms are physically legitimate here and must stay ON --
unlike the linear pool, where they fire on a several-nm end-to-end distance and read 91.55% of the
energy. This script runs the FULL field as shipped (BSJ ON, force_cap read from the signature)
exactly as ibi_round0.py does, on 2OIU.

Protocol is ibi_round0.py's, so the result is comparable to the linear-pool rows:
    8 replicas, dt 0.002 ps, mass 110 Da, friction 0.1 /ps, stride 25, burn 20000 (40 ps),
    sampling window 40-200 ps, --blocks=8, symplectic tail kick.

It reports:
  - the six bonded coordinates' sim/ref and the joint J = mean |ln(sim/ref)|, whole window;
  - the --blocks=8 stationarity table (disjoint equal-time blocks, each with its own J);
  - per-block Rg and P-bead RMSD vs the crystal start, so "does the sampler return a structure
    within RMSD of the crystal" (NOTES.md's open validation item) is answered, not assumed.

Caveats that must travel with any number printed here (see the report, not just this docstring):
  2OIU is a SINGLE chain (L=71), so every conclusion is "on the one circular reference", not a
  statistic. The reference sigma is pooled over 126 deposited chains, so single-chain-vs-pooled
  mismatch is UNRESOLVED here (another agent is on it). Both are stated, not hidden.

Run: python scripts/ibi_round0_circular.py [n_rep] [n_steps] [struct_idx] [friction] [stride] [burn]
                                            [--blocks=N]

struct_idx >= 0 selects a pool chain (the same filter ibi_round0.py uses) for the drift
self-check; struct_idx = -1 (default) selects 2OIU by name. The positional layout is
ibi_round0.py's own, so the self-check can invoke both scripts with the SAME argv and diff.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402


def _opt_int(name, default):
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            return int(a[len(pre):])
    return default


NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else -1
FRICTION = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 25
_burn_arg = int(sys.argv[6]) if len(sys.argv) > 6 else 0
burn = _burn_arg if _burn_arg > 0 else max(NSTEPS // 5, 1)
NB = max(_opt_int("blocks", 4), 1)   # ibi_round0.py's default, so a no-flag self-check diffs clean
SEED = 20260218

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
z = np.load(NPZ)
TAB = {}
for name in B.COORDS:
    TAB[name] = {"centre": z[f"{name}__centre"], "U": z[f"{name}__U"],
                 "binw": float(z[f"{name}__binw"]), "sigma": float(z[f"{name}__sigma"]),
                 "lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"])}

# 2OIU is not in ibi_round0's pool, so load it by name -- no change to any tracked loader.
# The positional struct_idx is argv[3], exactly as ibi_round0.py has it, so the self-check can
# invoke both scripts with the SAME argv and diff the output: a pool struct_idx (>=0) selects
# from the same filter ibi_round0 uses; IDX=-1 (default) selects 2OIU by name, the circular
# reference the pool is missing.
structs = B.load_structures(limit=400)
pool = [s for s in structs if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
if IDX >= 0:
    if IDX >= len(pool):
        raise SystemExit(f"struct_idx {IDX} out of range; the pool has {len(pool)} chains")
    s0 = pool[IDX]
    _note = f"pool[{IDX}] = {s0['name']} (linear, the drift self-check arm)"
else:
    s0 = next((s for s in structs if s["name"] == "2OIU"), None)
    if s0 is None:
        raise SystemExit("2OIU not found in the loader")
    _note = "2OIU (the one circular reference)"
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
P0_ref = s0["pos"].reshape(3 * L, 3)[0::3].copy()   # crystal P beads (L, 3), RMSD anchor

print(f"structure {s0['name']}  L={L}  pairs={len(ij)}  ({_note})")
print(f"{NREP} replicas, {NSTEPS} steps of 0.002 ps = {NSTEPS * 0.002:.1f} ps per replica")
print(f"full field as shipped, BSJ terms ON, 300 K, mass 110 Da, friction {FRICTION}/ps, "
      f"sampling every {STRIDE} steps")
print(f"burn = {burn} steps = {burn * 0.002:.1f} ps; sampling window "
      f"{burn * 0.002:.1f}-{NSTEPS * 0.002:.1f} ps; {NB} blocks")

# Provenance fingerprint, the same list ibi_round0 prints, so the two runs are comparable.
_FINGERPRINT = ("K_BB", "K_INTRA_PC", "K_INTRA_CN", "K_INTRA_PN",
                "K_LINK_CP", "K_LINK_NP", "K_LINK_NC",
                "K_PAIR", "K_ANGLE", "K_DIH", "K_BPP", "K_STACK",
                "K_CLASH", "CLASH_SIGMA", "K_BSJ", "K_BSJ_GUIDE")
_missing = [n for n in _FINGERPRINT if not hasattr(C, n)]
if _missing:
    raise RuntimeError(f"fingerprint names {_missing}, not defined in the module")
print("field: " + "  ".join(f"{n}={getattr(C, n)}" for n in _FINGERPRINT))
import inspect as _inspect
_cap = _inspect.signature(C.cg_energy_forces).parameters["force_cap"].default
print(f"force_cap={_cap}  K_BSJ_CONTACT={C.K_BSJ_CONTACT}  mass=110.0 Da  dt=0.002 ps  "
      f"friction={FRICTION}/ps")
_gs = C._sigmoid_f(torch.tensor([0.5, 3.0]), 1.0, 1.0, 0.2)[0]
print(f"guide shape: E(0.5 nm)={float(_gs[0]):.3f}  E(3.0 nm)={float(_gs[1]):.3f}  ("
      + ("long-range: zero below r0, bounded pull above" if float(_gs[0]) > 0
         else "SHORT-RANGE REWARD -- the pre-6e7a44b form") + ")")
print()

pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)


def rg(p):
    """Radius of gyration per replica. p: (B, 3L, 3) nm -> (B,)."""
    c = p.mean(dim=1, keepdim=True)
    return torch.linalg.norm(p - c, dim=-1).pow(2).mean(dim=1).sqrt()


def p_rmsd(p):
    """P-bead RMSD of each replica to the crystal P beads, after Kabsch alignment.

    p: (B, 3L, 3) -> (B,) RMSD in nm. The crystal anchor P0_ref is (L, 3).
    """
    P = p[:, 0::3, :].detach().numpy()       # (B, L, 3)
    ref = P0_ref                             # (L, 3)
    out = np.zeros(P.shape[0])
    for k in range(P.shape[0]):
        a = P[k] - P[k].mean(0)
        b = ref - ref.mean(0)
        H = a.T @ b
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        out[k] = float(np.sqrt(((a @ R - b) ** 2).sum(-1).mean()))
    return out


# static reading at the start, to tie this run to measure_bsj_on_circular_reference.py
_rg0 = float(rg(pos)[0])
_d0 = pos[:, 0] - pos[:, 3 * (L - 1)]
_r0 = C._safe_norm(_d0, dim=-1, keepdim=True, eps=1e-6)
_fbsj0 = (C.K_BSJ * (_r0 - C.BOND_P_NEXT)).abs().max()
print(f"native start: Rg = {_rg0:.3f} nm,  |P(0)-P(L-1)| = "
      f"{float(_r0[0, 0]):.4f} nm,  |bsj closure force| = {float(_fbsj0):.2f} kJ/mol/nm")
print()

torch.manual_seed(SEED)

b_counts = {b: {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
            for b in range(NB)}
b_acc = {b: {c: [0.0, 0.0, 0] for c in B.COORDS} for b in range(NB)}
b_frames = [0] * NB
b_rg = [[] for _ in range(NB)]
b_rmsd = [[] for _ in range(NB)]


def _forces_at(p):
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2)[1]


def _summed():
    ct = {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
    ac = {c: [0.0, 0.0, 0] for c in B.COORDS}
    for b in range(NB):
        for c in B.COORDS:
            ct[c] += b_counts[b][c]
            a = b_acc[b][c]
            ac[c][0] += a[0]
            ac[c][1] += a[1]
            ac[c][2] += a[2]
    return ct, ac


clash_min = []
clash_below_live = 0
t0 = time.time()
for step in range(NSTEPS):
    with torch.no_grad():
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps,
                                         dt_ps=0.002, mass_amu=110.0,
                                         friction=FRICTION, force_fn=_forces_at)
    if step == 0:
        print(f"first step {time.time() - t0:.3f} s")
    if step >= burn and step % STRIDE == 0:
        blk = min(((step - burn) * NB) // max(NSTEPS - burn, 1), NB - 1)
        b_frames[blk] += 1
        with torch.no_grad():
            for c in B.COORDS:
                q = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
                a = b_acc[blk][c]
                a[0] += q.sum()
                a[1] += (q ** 2).sum()
                a[2] += q.size
                t = TAB[c]
                k = np.round((q - t["centre"][0]) / t["binw"]).astype(np.int64)
                ok = (k >= 0) & (k < len(t["U"]))
                b_counts[blk][c] += np.bincount(k[ok], minlength=len(t["U"]))
            b_rg[blk].extend(rg(pos).tolist())
            b_rmsd[blk].extend(p_rmsd(pos).tolist())
            beads = pos.reshape(NREP, -1, 3)
            dd = torch.cdist(beads, beads)
            dd = dd + torch.eye(dd.shape[-1], device=dd.device) * 10.0
            clash_min.append(float(dd.min()))
            clash_below_live += int((dd < C.CLASH_SIGMA).sum())
    if (step + 1) % max(NSTEPS // 10, 1) == 0:
        el = time.time() - t0
        parts, _joint = [], []
        _ct, _ac = _summed()
        for c in B.COORDS:
            s1, s2, n = _ac[c]
            if n:
                mm = s1 / n
                ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
                rs = TAB[c]["sigma"]
                r = ss / rs if rs > 0 else float("nan")
                parts.append(f"{c[:5]} {r:5.3f}")
                if r == r and r > 0:
                    _joint.append(abs(float(np.log(r))))
            else:
                parts.append(f"{c[:5]}   --")
        _j = float(np.mean(_joint)) if _joint else float("nan")
        print(f"  {step + 1:>7d} {el:6.0f}s  " + "  ".join(parts) + f"   J {_j:.4f}")

el = time.time() - t0
print(f"done in {el:.0f} s, {NSTEPS / el:.1f} steps/s")
print()
print(f"clash watch: live range {C.CLASH_SIGMA:.4f} nm; closest bead pair ever "
      f"{min(clash_min):.4f} nm; pair instances below the live range {clash_below_live}")
print()


def ref_p(c):
    U = TAB[c]["U"]
    p = np.exp(-(U - U.min()) / B.KBT)
    return p / p.sum()


print(f"{'coordinate':10s} {'ref sig':>8s} {'sim sig':>8s} {'sim/ref':>8s} "
      f"{'dU min':>8s} {'dU max':>8s} {'|dU|>1kBT':>10s}")
print("-" * 62)
counts, acc = _summed()
rows = {}
for c in B.COORDS:
    t = TAB[c]
    s1, s2, n = acc[c]
    m = s1 / n
    sig = float(np.sqrt(max(s2 / n - m * m, 0.0)))
    ps = (counts[c] + 1.0) / (counts[c].sum() + len(counts[c]))
    pr = ref_p(c)
    dU = B.KBT * np.log(ps / pr)
    frac = float((np.abs(dU) > B.KBT).mean())
    rows[c] = (m, sig, dU)
    print(f"{c:10s} {t['sigma']:8.4f} {sig:8.4f} {sig / t['sigma']:8.3f} "
          f"{dU.min():8.2f} {dU.max():8.2f} {frac:10.3f}")
print()

# Joint residual (whole window), to sit next to the linear-pool rows.
_joint = [abs(float(np.log(rows[c][1] / TAB[c]["sigma"]))) for c in B.COORDS]
print(f"JOINT J (whole window, {burn * 0.002:.0f}-{NSTEPS * 0.002:.0f} ps): "
      f"{float(np.mean(_joint)):.4f}")
print()

if NB > 1:
    print(f"=== stationarity: {NB} disjoint blocks of the sampling window ===")
    print(f"{'block':>5s} {'window (ps)':>15s} {'frames':>7s} {'J':>7s}  "
          + "  ".join(f"{c[:5]:>5s}" for c in B.COORDS))
    print("-" * (37 + 7 * len(B.COORDS)))
    _Jb = []
    for b in range(NB):
        lo = burn + (NSTEPS - burn) * b // NB
        hi = burn + (NSTEPS - burn) * (b + 1) // NB
        parts, js = [], []
        for c in B.COORDS:
            s1, s2, n = b_acc[b][c]
            if n:
                mm = s1 / n
                ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
                r = ss / TAB[c]["sigma"] if TAB[c]["sigma"] > 0 else float("nan")
                parts.append(f"{r:5.3f}")
                if r == r and r > 0:
                    js.append(abs(float(np.log(r))))
            else:
                parts.append("   --")
        j = float(np.mean(js)) if js else float("nan")
        _Jb.append(j)
        print(f"{b + 1:5d} {lo * 0.002:6.1f}-{hi * 0.002:6.1f} {b_frames[b]:7d} {j:7.4f}  "
              + "  ".join(parts))
    _lo, _hi = min(_Jb), max(_Jb)
    _rel = (_hi - _lo) / _lo if _lo > 0 else float("nan")
    print(f"  block J spread: {_lo:.4f} to {_hi:.4f}  ({_rel * 100:.1f} percent of the lowest)")

# Rg and RMSD drift per block.
print()
print(f"{'block':>5s} {'window (ps)':>15s} {'Rg (nm)':>9s} {'dRg vs native':>13s} "
      f"{'P-RMSD (nm)':>12s}")
print("-" * 54)
for b in range(NB):
    lo = burn + (NSTEPS - burn) * b // NB
    hi = burn + (NSTEPS - burn) * (b + 1) // NB
    if not b_rg[b]:
        print(f"{b + 1:5d} {lo * 0.002:6.1f}-{hi * 0.002:6.1f}   (no samples)")
        continue
    mr = float(np.mean(b_rg[b]))
    mrmsd = float(np.mean(b_rmsd[b]))
    print(f"{b + 1:5d} {lo * 0.002:6.1f}-{hi * 0.002:6.1f} {mr:9.3f} {mr - _rg0:+13.3f} "
          f"{mrmsd:12.3f}")
print()
print(f"Rg: block 1 {np.mean(b_rg[0]):.3f} -> block {NB} {np.mean(b_rg[-1]):.3f} nm")
print(f"P-RMSD vs crystal: block 1 {np.mean(b_rmsd[0]):.3f} -> block {NB} "
      f"{np.mean(b_rmsd[-1]):.3f} nm")
