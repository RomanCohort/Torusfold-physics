"""IBI round-0 residual measured on the PRODUCTION replica-exchange sampler (2D REST2 x T-REMD).

What this measures, and why it exists
--------------------------------------
The IBI residual dU(q) = kBT * ln(P_sim(q) / P_ref(q)) is only a valid input to an IBI update
when P_sim is the EQUILIBRIUM marginal. The calibration fixture (scripts/ibi_round0.py) samples
8 independent replicas at a single 300 K -- no replica exchange -- and its block spread (23.1%
on idx0, results/E1_stationarity.log) says that fixture never reaches equilibrium: the residual
it reports is a mixture over a still-drifting window, not a Boltzmann ratio.

This script re-measures the SAME six bonded coordinates with the SAME full field, but sampling
the cold (300 K, lambda=1.0) rung of the production 2D REST2 x T-REMD grid (the
BatchedREMD2D design: all replicas in one tensor, temperature axis + lambda axis), and asks the
single gate question: does replica exchange collapse the block-to-block drift?

Sampling decision (the one that must be right)
----------------------------------------------
REMD walks replicas across the temperature ladder. The marginal we want is the TARGET
temperature's -- the distribution of whichever replica currently occupies the 300 K rung, NOT a
pool over all replicas (that would mix different temperatures into one histogram). Concretely:
slot 0 of the grid is (T=300, lambda=1.0) by construction (temps[0]=t_lo, lambdas[0]=1.0), and
after every exchange round the coordinate sitting in slot 0 IS "the replica currently occupying
the cold rung". So each sample is pos[0], taken at every stride after burn.

The lambda axis: lambda=1.0 is the UNSCALED (full) field, byte-identical to the fixture's field,
so the two marginals are comparable on the same Hamiltonian. lambda<1 rungs scale only the WC
pair term (K_STACK=0 and the bonded terms bb/intra/angle/dihedral are not lambda-scaled), so they
help the cold rung by REST2 mixing but are not themselves sampled. Sampling (T=300, lambda=1.0)
is therefore the single correct comparison point against ibi_round0.

Burn
----
The fixture proved burn=NSTEPS//5 (3.2 ps) far too short. Under exchange the cold rung must ALSO
wait for replicas to round-trip the ladder before it stops carrying the starting configuration.
With exchange_interval steps between swap rounds and per-edge acceptance a, one climb of the
n_t-rung ladder costs ~(n_t-1)*exchange_interval/a steps; a round trip is twice that. burn is set
to one round trip (a is assumed 0.4, the measured value is printed so the assumption is
checkable), so the window starts once the cold rung has seen at least one pass of genuinely
exchanged configurations.

Main judgement
--------------
Identical to ibi_round0: --blocks disjoint equal-time blocks of the sampling window, each with
its own joint J = mean |ln(sim/ref)| over the six coordinates. If the spread collapses below the
fixture's 23.1%, exchange fixed equilibration and IBI can start. If it stays ~20%, the slow mode
is not on the temperature/lambda axes and exchange does not help.

Run: python scripts/ibi_remd_residual.py NSTEPS [idx] [--n_t=6] [--t_hi=1000]
     [--lambdas=1.0,0.82,0.67,0.55] [--exchange=1000] [--friction=0.1] [--burn=0] [--blocks=8]
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B                 # noqa: E402
import torusfold.scheme2.torch_cgsim as C    # noqa: E402


def _opt(name, default):
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            return a[len(pre):]
    return default


def _opt_int(name, default):
    v = _opt(name, None)
    return int(v) if v is not None else default


def _opt_float(name, default):
    v = _opt(name, None)
    return float(v) if v is not None else default


_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
NSTEPS = int(_positional[0]) if _positional else 40000
IDX = int(_positional[1]) if len(_positional) > 1 else 0

N_T = _opt_int("n_t", 6)
T_LO = _opt_float("t_lo", 300.0)
T_HI = _opt_float("t_hi", 550.0)
LAMBDAS = [float(x) for x in _opt("lambdas", "1.0,0.82,0.67,0.55").split(",")]
EXCH = _opt_int("exchange", 500)
FRICTION = _opt_float("friction", 0.1)
STRIDE = _opt_int("stride", 25)
NB = max(_opt_int("blocks", 8), 1)
SEED = _opt_int("seed", 20260218)
BURN_ARG = _opt_int("burn", 0)

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
z = np.load(NPZ)
TAB = {}
for name in B.COORDS:
    TAB[name] = {"centre": z[f"{name}__centre"], "U": z[f"{name}__U"],
                 "binw": float(z[f"{name}__binw"]), "sigma": float(z[f"{name}__sigma"]),
                 "lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"])}

pool = [s for s in B.load_structures(limit=400)
        if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[IDX]
L = len(s0["pos"])

# ── grid, identical to BatchedREMD2D by construction ──
temps = np.geomspace(T_LO, T_HI, N_T).tolist()          # n_t temperature rungs
lambdas = list(LAMBDAS)                                  # n_lam lambda rungs
N_LAM = len(lambdas)
N_REP = N_T * N_LAM
temps_grid = np.repeat(np.asarray(temps), N_LAM)         # (n_rep,)
lams_grid = np.tile(np.asarray(lambdas), N_T)            # (n_rep,)
beta = 1.0 / (C.KB_KJ * np.asarray(temps))               # (n_t,)
# slot 0 = (temps[0]=300 K, lambdas[0]=1.0) = the target rung sampled below.

print(f"structure {s0['name']}  L={L}  pairs={len(s0['pairs'])}")
print(f"REMD grid: {N_T} temperatures (geomspace {T_LO}-{T_HI} K) x {N_LAM} lambdas "
      f"{lambdas} = {N_REP} replicas, exchange every {EXCH} steps")
print(f"{NSTEPS} steps of 0.002 ps = {NSTEPS * 0.002:.1f} ps, "
      f"sampling the cold rung (T={temps[0]:.1f} K, lambda={lambdas[0]}) every {STRIDE} steps")
print(f"full field as shipped, mass 110 Da, friction {FRICTION}/ps, float32, symplectic BAOAB")

# Provenance fingerprint, same list as ibi_round0 so the two runs are auditable against the
# same constants.
_FINGERPRINT = ("K_BB", "K_INTRA_PC", "K_INTRA_CN", "K_INTRA_PN",
                "K_LINK_CP", "K_LINK_NP", "K_LINK_NC",
                "K_PAIR", "K_ANGLE", "K_DIH", "K_BPP", "K_STACK",
                "K_CLASH", "CLASH_SIGMA", "K_BSJ", "K_BSJ_GUIDE")
_missing = [n for n in _FINGERPRINT if not hasattr(C, n)]
if _missing:
    raise RuntimeError(f"fingerprint names missing from module: {_missing}")
print("field: " + "  ".join(f"{n}={getattr(C, n)}" for n in _FINGERPRINT))
import inspect as _inspect
_cap = _inspect.signature(C.cg_energy_forces).parameters["force_cap"].default
print(f"force_cap={_cap}  mass=110.0 Da  dt=0.002 ps  friction={FRICTION}/ps")
print(f"note: rest2_remd_2d.py (2D REST2xREMD production sampler) defaults n_t=6, t_hi=550, "
      f"lambdas=(1.0,0.82,0.67,0.55), exchange=1000; the GPU BatchedREMD2D path in isrnaclong is "
      f"constructed n_t=8, 8 lambdas, t_hi=1000, exchange=1000. This run uses the 6x4 grid and "
      f"exchange={EXCH}; see the report for the choice and the discrepancies.")

# burn: one round trip of the ladder. Climb = (n_t-1) accepted T-edges; at per-edge acceptance a
# that is (n_t-1)/a exchange rounds of `exchange` steps each; round trip is twice that.
_ASSUMED_A = 0.4
burn = BURN_ARG if BURN_ARG > 0 else int(2 * (N_T - 1) * EXCH / _ASSUMED_A)
burn = min(burn, max(NSTEPS // 2, 1))
print(f"burn = {burn} steps = {burn * 0.002:.1f} ps "
      f"(one ladder round trip at assumed {_ASSUMED_A:.0%} acceptance; "
      f"measured acceptance printed below). sampling window {burn * 0.002:.1f}-"
      f"{NSTEPS * 0.002:.1f} ps")
print()

K_SHIPPED = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA_PC, "intra_cn": C.K_INTRA_CN,
             "angle": C.K_ANGLE, "dihedral": C.K_DIH, "stack": C.K_STACK}

# ── initial state: all replicas at the native structure, like ibi_round0 ──
pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(N_REP, 1, 1)
pos = pos.to(torch.float32)
vel = torch.zeros_like(pos)
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
temps_t = torch.tensor(temps_grid, dtype=torch.float32)
lams_t = torch.tensor(lams_grid, dtype=torch.float32)

torch.manual_seed(SEED)
np.random.seed(SEED)

# per-block accumulators, same layout as ibi_round0
b_counts = {b: {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
            for b in range(NB)}
b_acc = {b: {c: [0.0, 0.0, 0] for c in B.COORDS} for b in range(NB)}
b_frames = [0] * NB

# exchange acceptance bookkeeping, same as BatchedREMD2D.run
accT = [0] * (N_T - 1); attT = [0] * (N_T - 1)
accL = [0] * (N_LAM - 1); attL = [0] * (N_LAM - 1)


def _energy_split(pos_in, indices=None):
    """Return (own-lambda energy, lambda=1 solute energy) per replica.

    solute = E(lambda=1) - E(lambda=0) is the REST2 reference solute term; the GB/SA, BPP,
    bonded and clash terms do not depend on lambda so they cancel, leaving the WC pair term.
    """
    if indices is None:
        indices = torch.arange(pos_in.shape[0], device=pos_in.device)
        own_lams = lams_t
    else:
        indices = torch.as_tensor(indices, dtype=torch.long, device=pos_in.device)
        own_lams = lams_t[indices]
    p = pos_in[indices].detach()
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    with torch.no_grad():
        en_ref, _ = C.cg_energy_forces(p, ij, pw, lam=1.0, cell_list=cl)
        en_base, _ = C.cg_energy_forces(p, ij, pw, lam=0.0, cell_list=cl)
        en_own, _ = C.cg_energy_forces(p, ij, pw, lams=own_lams, cell_list=cl)
    return en_own.detach(), (en_ref - en_base).detach()


def _refresh_slot_energies(slots, energies, solute_np):
    """Re-evaluate energies at the swapped slots (coordinates moved into new Hamiltonian slots)."""
    sl = np.asarray(sorted(set(int(x) for x in slots)), dtype=np.int64)
    e_slot, s_slot = _energy_split(pos, sl)
    energies[sl] = e_slot.cpu().numpy()
    solute_np[sl] = s_slot.cpu().numpy()


def _summed():
    ct = {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
    ac = {c: [0.0, 0.0, 0] for c in B.COORDS}
    for b in range(NB):
        for c in B.COORDS:
            ct[c] += b_counts[b][c]
            a = b_acc[b][c]
            ac[c][0] += a[0]; ac[c][1] += a[1]; ac[c][2] += a[2]
    return ct, ac


def _forces_at(p):
    """Fresh forces at the post-update coordinates (symplectic tail kick)."""
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, lams=lams_t, cell_list=cl2)[1]


t0 = time.time()
n_reports = max(1, NSTEPS // EXCH)
for rep in range(n_reports):
    for _ in range(EXCH):
        with torch.no_grad():
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            _e, f = C.cg_energy_forces(pos, ij, pw, lams=lams_t, cell_list=cl)
            pos, vel = C.batch_langevin_step(
                pos, vel, f, temps_t, dt_ps=0.002, mass_amu=110.0, friction=FRICTION,
                force_fn=_forces_at)
        step = rep * EXCH + _
        if step >= burn and step % STRIDE == 0:
            blk = min(((step - burn) * NB) // max(NSTEPS - burn, 1), NB - 1)
            b_frames[blk] += 1
            with torch.no_grad():
                for c in B.COORDS:
                    q = B.coords_of(pos[:1], c).reshape(-1).numpy().astype(np.float64)
                    a = b_acc[blk][c]
                    a[0] += q.sum(); a[1] += (q ** 2).sum(); a[2] += q.size
                    t = TAB[c]
                    k = np.round((q - t["centre"][0]) / t["binw"]).astype(np.int64)
                    ok = (k >= 0) & (k < len(t["U"]))
                    b_counts[blk][c] += np.bincount(k[ok], minlength=len(t["U"]))

    # ── exchange round (at the end of every `exchange` steps), same protocol as run() ──
    with torch.no_grad():
        e_full, e_solute = _energy_split(pos)
    energies = e_full.cpu().numpy()
    solute_np = e_solute.cpu().numpy()

    # temperature axis: every vertical edge in each lambda column
    for cj in range(N_LAM):
        col = [ri * N_LAM + cj for ri in range(N_T)]
        for k in range(N_T - 1):
            a, b = col[k], col[k + 1]
            attT[k] += 1
            log_alpha = C.exchange_log_alpha_temperature(
                beta[k], beta[k + 1], energies[a], energies[b])
            if C.metropolis_accept_log_alpha(float(log_alpha)):
                accT[k] += 1
                tmp_pos = pos[a].clone()
                pos[a] = pos[b].clone()
                pos[b] = tmp_pos
                tmp_vel = vel[a].clone()
                vel[a] = vel[b].clone() * math.sqrt(temps[k] / temps[k + 1])
                vel[b] = tmp_vel * math.sqrt(temps[k + 1] / temps[k])
                _refresh_slot_energies((a, b), energies, solute_np)

    # lambda axis: alternating-parity horizontal edges in each temperature row
    l_edge_start = rep & 1
    for ri in range(N_T):
        row = [ri * N_LAM + cj for cj in range(N_LAM)]
        for k in range(l_edge_start, N_LAM - 1, 2):
            a, b = row[k], row[k + 1]
            attL[k] += 1
            log_alpha = C.exchange_log_alpha_lambda(
                beta[ri], lambdas[k], lambdas[k + 1], solute_np[a], solute_np[b])
            if C.metropolis_accept_log_alpha(float(log_alpha)):
                accL[k] += 1
                tmp_pos = pos[a].clone()
                pos[a] = pos[b].clone()
                pos[b] = tmp_pos
                tmp_vel = vel[a].clone()
                vel[a] = vel[b].clone()
                vel[b] = tmp_vel
                _refresh_slot_energies((a, b), energies, solute_np)

    if (rep + 1) % max(n_reports // 10, 1) == 0 or rep == n_reports - 1:
        el = time.time() - t0
        aT = [f"{a}/{t}" for a, t in zip(accT, attT)]
        aL = [f"{a}/{t}" for a, t in zip(accL, attL)]
        parts = []
        _ct, _ac = _summed()
        for c in B.COORDS:
            s1, s2, n = _ac[c]
            if n:
                mm = s1 / n
                ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
                r = ss / TAB[c]["sigma"]
                parts.append(f"{c[:5]} {r:5.3f}")
        print(f"  step {step + 1:>7d} {el:6.0f}s  " + "  ".join(parts)
              + f"   T-acc [{','.join(aT)}]  L-acc [{','.join(aL)}]")

el = time.time() - t0
print(f"done in {el:.0f} s, {NSTEPS / el:.1f} steps/s")
print()


def ref_p(c):
    U = TAB[c]["U"]
    p = np.exp(-(U - U.min()) / B.KBT)
    return p / p.sum()


print(f"{'coordinate':10s} {'ref sig':>8s} {'1-D sig':>8s} {'sim sig':>8s} "
      f"{'sim/ref':>8s} {'sim/1D':>7s}")
print("-" * 60)
rows = {}
counts, acc = _summed()
for c in B.COORDS:
    t = TAB[c]
    s1, s2, n = acc[c]
    m = s1 / n
    sig = float(np.sqrt(max(s2 / n - m * m, 0.0)))
    k = K_SHIPPED[c]
    s1d = float(np.sqrt(B.KBT / k)) if k > 0 else float("nan")
    rows[c] = (m, sig)
    print(f"{c:10s} {t['sigma']:8.4f} {s1d:8.4f} {sig:8.4f} "
          f"{sig / t['sigma']:8.3f} {sig / s1d:7.3f}")
print()

# ── exchange acceptance (the NOTES.md criterion) ──
_tot_acc = sum(accT) + sum(accL)
_tot_att = sum(attT) + sum(attL)
print("=== exchange acceptance ===")
print(f"T-axis per edge : " + "  ".join(
    f"{a / max(t, 1):.2f}" for a, t in zip(accT, attT)))
print(f"L-axis per edge : " + "  ".join(
    f"{a / max(t, 1):.2f}" for a, t in zip(accL, attL)))
print(f"overall         : {_tot_acc / max(_tot_att, 1):.3f}  ({_tot_acc}/{_tot_att})")
if _tot_att and (_tot_acc / _tot_att < 0.10 or _tot_acc / _tot_att > 0.80):
    print("  WARNING: acceptance outside 10-80% is a configuration problem; fix before trusting"
          " the marginal.")
print()

# ── stationarity: disjoint blocks, the main judgement ──
if NB > 1:
    print(f"=== stationarity: {NB} disjoint blocks of the sampling window (cold rung) ===")
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
    print("  fixture (no exchange, same structure/field/window) block spread was 23.1 percent"
          " (results/E1_stationarity.log).")
    print()
