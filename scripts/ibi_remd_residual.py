"""IBI round-0 residual measured on the PRODUCTION replica-exchange sampler (2D REST2 x T-REMD).

What this measures, and why it exists
--------------------------------------
The IBI residual dU(q) = kBT * ln(P_sim(q) / P_ref(q)) is only a valid input to an IBI update
when P_sim is the EQUILIBRIUM marginal. The calibration fixture (scripts/ibi_round0.py) samples
8 independent replicas at a single 300 K -- no replica exchange -- and its block spread (23.1%
on idx0, results/E1_stationarity.log) says that fixture never reaches equilibrium. A first REMD
run (24 replicas, 200 ps) got block spread 33.9% (results/remd_idx0_100k.log): exchange alone
did not collapse the drift. This script extends that run to the PRODUCTION sampling budget
(660 ps / replica, docs/REPRODUCTION_RESOURCES.md) and reports the block-J SEQUENCE so we can see
whether it flattens, which is the actual gate.

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

Field override
--------------
--set=K_BSJ=0,K_BSJ_GUIDE=0,K_BSJ_CONTACT=0 turns off the circular-closure trio. That is the
CORRECT configuration for the linear test chain (README: the BSJ trio is 91.55% of the energy on
a linear reference and should be zeroed for a linear run); the gate question "does the sampler
reach stationarity" is then not confounded by a force that should not be there. The no-exchange
BSJ-off reference is results/F_noBSJ_idx0.log (J=0.0995, block spread 29.9%).

Checkpoint / resume (the long runs here get killed)
---------------------------------------------------
Every --chunk steps the full state (24 replica coords + velocities in nm, the per-block
histogram/moment accumulators, the RNG states, the exchange counters) is dumped to --ckpt, and
on startup a checkpoint whose config key matches and whose step is below NSTEPS is resumed. So a
kill costs at most one chunk, and re-running the SAME command line continues from where it died.

Run: python scripts/ibi_remd_residual.py NSTEPS [idx] [--n_t=6] [--t_hi=550]
     [--lambdas=1.0,0.82,0.67,0.55] [--exchange=500] [--friction=0.1] [--burn=20000]
     [--blocks=24] [--chunk=20000] [--seed=20260218] [--set=K_BSJ=0,K_BSJ_GUIDE=0,K_BSJ_CONTACT=0]
     [--ckpt=results/remd_ckpt_idx0.npz]
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


# ── patch any --set=K=V,K=V before anything reads the constants ──
_set_str = _opt("set", "")
_set = {}
for a in sys.argv[1:]:
    if a.startswith("--set="):
        for kv in a[len("--set="):].split(","):
            kv = kv.strip()
            if not kv:
                continue
            name, _, val = kv.partition("=")
            _set[name.strip()] = float(val)
for _n, _v in _set.items():
    if not hasattr(C, _n):
        raise SystemExit(f"no such constant in torch_cgsim: {_n!r}; refusing to run, because a "
                         f"silently ignored name would produce a run of the unpatched field")
    setattr(C, _n, _v)

_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
NSTEPS = int(_positional[0]) if _positional else 330000
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
CHUNK = _opt_int("chunk", 20000)
CKPT = _opt("ckpt", "results/remd_ckpt_idx0.npz")

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
temps = np.geomspace(T_LO, T_HI, N_T).tolist()
lambdas = list(LAMBDAS)
N_LAM = len(lambdas)
N_REP = N_T * N_LAM
temps_grid = np.repeat(np.asarray(temps), N_LAM)
lams_grid = np.tile(np.asarray(lambdas), N_T)
beta = 1.0 / (C.KB_KJ * np.asarray(temps))
# slot 0 = (temps[0]=300 K, lambdas[0]=1.0) = the target rung sampled below.

# burn: default 40 ps, aligned with the fixture / reference windows so the windows are comparable.
burn = _opt_int("burn", 20000)
burn = min(burn, max(NSTEPS - 1, 1))

# ── provenance ──
if _set:
    print("PATCHED FIELD: " + ", ".join(f"{k}={v:g}" for k, v in _set.items()))
    print("Every constant below is read after this patch, so the fingerprint reflects it.")
print(f"structure {s0['name']}  L={L}  pairs={len(s0['pairs'])}")
print(f"REMD grid: {N_T} temperatures (geomspace {T_LO}-{T_HI} K) x {N_LAM} lambdas "
      f"{lambdas} = {N_REP} replicas, exchange every {EXCH} steps")
print(f"{NSTEPS} steps of 0.002 ps = {NSTEPS * 0.002:.1f} ps, "
      f"sampling the cold rung (T={temps[0]:.1f} K, lambda={lambdas[0]}) every {STRIDE} steps")
print(f"full field (after any --set patch), mass 110 Da, friction {FRICTION}/ps, "
      f"float32, symplectic BAOAB, chunk {CHUNK} steps")
_FINGERPRINT = ("K_BB", "K_INTRA_PC", "K_INTRA_CN", "K_INTRA_PN",
                "K_LINK_CP", "K_LINK_NP", "K_LINK_NC",
                "K_PAIR", "K_ANGLE", "K_DIH", "K_BPP", "K_STACK",
                "K_CLASH", "CLASH_SIGMA", "K_BSJ", "K_BSJ_GUIDE", "K_BSJ_CONTACT")
_missing = [n for n in _FINGERPRINT if not hasattr(C, n)]
if _missing:
    raise RuntimeError(f"fingerprint names missing from module: {_missing}")
print("field: " + "  ".join(f"{n}={getattr(C, n)}" for n in _FINGERPRINT))
import inspect as _inspect
_cap = _inspect.signature(C.cg_energy_forces).parameters["force_cap"].default
print(f"force_cap={_cap}  mass=110.0 Da  dt=0.002 ps  friction={FRICTION}/ps")
print(f"note: rest2_remd_2d.py (2D REST2xREMD production sampler) defaults n_t=6, t_hi=550, "
      f"lambdas=(1.0,0.82,0.67,0.55), exchange=1000; the GPU BatchedREMD2D path in isrnaclong is "
      f"constructed n_t=8, 8 lambdas, t_hi=1000, exchange=1000. This run uses the {N_T}x{N_LAM} "
      f"grid and exchange={EXCH}.")
print(f"burn = {burn} steps = {burn * 0.002:.1f} ps; sampling window "
      f"{burn * 0.002:.1f}-{NSTEPS * 0.002:.1f} ps; {NB} disjoint blocks")
print()

K_SHIPPED = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA_PC, "intra_cn": C.K_INTRA_CN,
             "angle": C.K_ANGLE, "dihedral": C.K_DIH, "stack": C.K_STACK}


def _config_key():
    fp = tuple(f"{n}={getattr(C, n)}" for n in _FINGERPRINT)
    return repr((IDX, s0["name"], L, N_T, T_LO, T_HI, tuple(lambdas), EXCH, FRICTION,
                 STRIDE, NB, burn, SEED, NSTEPS, _set_str, fp))


# ── per-block accumulators, same layout as ibi_round0 ──
def _fresh_blocks():
    b_counts = {b: {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
                for b in range(NB)}
    b_acc = {b: {c: [0.0, 0.0, 0] for c in B.COORDS} for b in range(NB)}
    frames = [0] * NB
    return b_counts, b_acc, frames


def _summed():
    ct = {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
    ac = {c: [0.0, 0.0, 0] for c in B.COORDS}
    for b in range(NB):
        for c in B.COORDS:
            ct[c] += b_counts[b][c]
            a = b_acc[b][c]
            ac[c][0] += a[0]; ac[c][1] += a[1]; ac[c][2] += a[2]
    return ct, ac


def _save_ckpt(path, step, pos, vel):
    d = {"step": step, "pos": pos.detach().cpu().numpy(),
         "vel": vel.detach().cpu().numpy(),
         "frames": np.asarray(b_frames),
         "accT": np.asarray(accT), "attT": np.asarray(attT),
         "accL": np.asarray(accL), "attL": np.asarray(attL),
         "torch_rng": torch.get_rng_state().numpy()}
    st = np.random.get_state()
    d["np_kind"] = st[0]; d["np_state"] = st[1]; d["np_pos"] = st[2]
    d["np_has_gauss"] = st[3]; d["np_cached"] = st[4]
    d["config_key"] = _config_key()
    for c in B.COORDS:
        d[f"count_{c}"] = np.stack([b_counts[b][c] for b in range(NB)])
        d[f"acc_{c}"] = np.stack([b_acc[b][c] for b in range(NB)])
    np.savez(path, **d)


def _load_ckpt(path):
    if not (REPO / path).exists():
        return None
    z = np.load(REPO / path, allow_pickle=False)
    if str(z["config_key"]) != _config_key():
        raise SystemExit(
            f"checkpoint {path} config does not match this invocation; refusing to resume onto "
            f"a different field/grid. Delete the checkpoint or rerun the exact command.")
    step = int(z["step"])
    pos = torch.from_numpy(z["pos"].astype(np.float32))
    vel = torch.from_numpy(z["vel"].astype(np.float32))
    for b in range(NB):
        b_frames[b] = int(z["frames"][b])
        for c in B.COORDS:
            b_counts[b][c] = z[f"count_{c}"][b].astype(np.int64)
            b_acc[b][c] = [float(z[f"acc_{c}"][b][0]), float(z[f"acc_{c}"][b][1]),
                           int(z[f"acc_{c}"][b][2])]
    accT[:] = z["accT"].tolist(); attT[:] = z["attT"].tolist()
    accL[:] = z["accL"].tolist(); attL[:] = z["attL"].tolist()
    torch.set_rng_state(torch.from_numpy(z["torch_rng"]))
    np.random.set_state((str(z["np_kind"]), z["np_state"].astype(np.uint32),
                         int(z["np_pos"]), float(z["np_has_gauss"]), float(z["np_cached"])))
    return step, pos, vel


# ── initial / resumed state ──
b_counts, b_acc, b_frames = _fresh_blocks()
accT = [0] * (N_T - 1); attT = [0] * (N_T - 1)
accL = [0] * (N_LAM - 1); attL = [0] * (N_LAM - 1)

resumed = _load_ckpt(CKPT)
if resumed is not None:
    start_step, pos, vel = resumed
    print(f"resumed from {CKPT} at step {start_step} "
          f"({start_step * 0.002:.1f} ps of {NSTEPS * 0.002:.1f} ps)")
else:
    start_step = 0
    pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(N_REP, 1, 1)
    pos = pos.to(torch.float32)
    vel = torch.zeros_like(pos)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
temps_t = torch.tensor(temps_grid, dtype=torch.float32)
lams_t = torch.tensor(lams_grid, dtype=torch.float32)


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
    sl = np.asarray(sorted(set(int(x) for x in slots)), dtype=np.int64)
    e_slot, s_slot = _energy_split(pos, sl)
    energies[sl] = e_slot.cpu().numpy()
    solute_np[sl] = s_slot.cpu().numpy()


def _forces_at(p):
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, lams=lams_t, cell_list=cl2)[1]


t0 = time.time()
step = start_step
while step < NSTEPS:
    chunk_end = min(step + CHUNK, NSTEPS)
    while step < chunk_end:
        with torch.no_grad():
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            _e, f = C.cg_energy_forces(pos, ij, pw, lams=lams_t, cell_list=cl)
            pos, vel = C.batch_langevin_step(
                pos, vel, f, temps_t, dt_ps=0.002, mass_amu=110.0, friction=FRICTION,
                force_fn=_forces_at)
        step += 1
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
        if step % EXCH == 0:
            # ── exchange round (production protocol), parity derived from absolute step ──
            with torch.no_grad():
                e_full, e_solute = _energy_split(pos)
            energies = e_full.cpu().numpy()
            solute_np = e_solute.cpu().numpy()
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
            l_edge_start = (step // EXCH - 1) & 1
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

    # chunk end: checkpoint + progress line
    el = time.time() - t0
    _tot_acc = sum(accT) + sum(accL)
    _tot_att = sum(attT) + sum(attL)
    _ovr = _tot_acc / max(_tot_att, 1)
    parts = []
    _ct, _ac = _summed()
    _js = []
    for c in B.COORDS:
        s1, s2, n = _ac[c]
        if n:
            mm = s1 / n
            ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
            r = ss / TAB[c]["sigma"]
            parts.append(f"{c[:5]} {r:5.3f}")
            if r > 0:
                _js.append(abs(float(np.log(r))))
        else:
            parts.append(f"{c[:5]}   --")
    _j = float(np.mean(_js)) if _js else float("nan")
    print(f"  step {step:>7d} ({step * 0.002:6.1f} ps) {el:6.0f}s {step / max(el, 1):.1f} st/s  "
          + "  ".join(parts) + f"   J {_j:.4f}  acc {_ovr:.3f}")
    try:
        _save_ckpt(CKPT, step, pos, vel)
    except Exception as exc:
        print(f"  [checkpoint failed: {exc}]")

el = time.time() - t0
_done = NSTEPS - start_step
print(f"done in {el:.0f} s, {_done / el:.1f} steps/s (wall, this invocation)")
print()


def ref_p(c):
    U = TAB[c]["U"]
    p = np.exp(-(U - U.min()) / B.KBT)
    return p / p.sum()


print(f"{'coordinate':10s} {'ref sig':>8s} {'1-D sig':>8s} {'sim sig':>8s} "
      f"{'sim/ref':>8s} {'sim/1D':>7s}")
print("-" * 60)
counts, acc = _summed()
for c in B.COORDS:
    t = TAB[c]
    s1, s2, n = acc[c]
    m = s1 / n
    sig = float(np.sqrt(max(s2 / n - m * m, 0.0)))
    k = K_SHIPPED[c]
    s1d = float(np.sqrt(B.KBT / k)) if k > 0 else float("nan")
    print(f"{c:10s} {t['sigma']:8.4f} {s1d:8.4f} {sig:8.4f} "
          f"{sig / t['sigma']:8.3f} {sig / s1d:7.3f}")
print()

_tot_acc = sum(accT) + sum(accL)
_tot_att = sum(attT) + sum(attL)
print("=== exchange acceptance ===")
print(f"T-axis per edge : " + "  ".join(f"{a / max(t, 1):.2f}" for a, t in zip(accT, attT)))
print(f"L-axis per edge : " + "  ".join(f"{a / max(t, 1):.2f}" for a, t in zip(accL, attL)))
print(f"overall         : {_tot_acc / max(_tot_att, 1):.3f}  ({_tot_acc}/{_tot_att})")
if _tot_att and (_tot_acc / _tot_att < 0.10 or _tot_acc / _tot_att > 0.80):
    print("  WARNING: acceptance outside 10-80% is a configuration problem; fix before trusting"
          " the marginal.")
print()

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
    third = max(1, NB // 3)
    _last = _Jb[-third:]
    _first = _Jb[:NB - third]
    _last_range = (max(_last) - min(_last)) / min(_last) if min(_last) > 0 else float("nan")
    print(f"  block J spread (whole window): {_lo:.4f} to {_hi:.4f}  ({_rel * 100:.1f}%)")
    print(f"  last {third}/{NB} blocks: {min(_last):.4f} to {max(_last):.4f}  "
          f"(range {_last_range * 100:.1f}%); first {NB - third} blocks: "
          f"{min(_first):.4f} to {max(_first):.4f}")
    print("  reference no-exchange fixture block spread: 23.1% (BSJ on, E1_stationarity.log), "
          "29.9% (BSJ off, F_noBSJ_idx0.log).")
    print()
