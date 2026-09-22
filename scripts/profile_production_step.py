"""What one PRODUCTION step costs on the biggest chain, and how much of it is the diagnostic.

The isolated pieces (scripts/profile_giant_chain.py, 2026-09-22) say 8FMW_24 (L=2929, 8787 beads)
costs 521 ms per cg_energy_forces call and 679 ms per closest_bead_pair call, both 1 torch thread.
That predicts 5.9 h for the 32,500 steps of a round. Production takes 9.2-18.2 h for the same work.

The gap has exactly three candidate sources and this script separates them:
  1. the integrator adds work per step that the isolated field call does not have -- SHAKE, the
     Langevin thermostat, the position update, and a cell-list rebuild;
  2. the entry relaxation (5000 requested steps, ~630 accepted/rejected force evaluations on this
     chain) is charged to the same task timer;
  3. the per-frame clash diagnostic is called on every SAMPLED frame, and at this size it costs
     more than the force field it watches.

run_round is timed with and without the clash call (the module-level name is patched, which is why
the patch works: run_round calls the module global), so (3) comes out as a difference rather than
an estimate. Read-only with respect to the production run: no heartbeat, no task file, no table.
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))
os.environ.setdefault("IBI_LOOP_OUT", str(REPO / "results" / "ibi_relax"))
os.environ.setdefault("IBI_LOOP_REF", str(REPO / "results" / "refit_smooth5.npz"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402

torch.set_num_threads(1)

import boltzmann_bonded as B            # noqa: E402
import ibi_core as IC                   # noqa: E402
import ibi_loop as IL                   # noqa: E402
import torusfold.scheme2.torch_cgsim as C  # noqa: E402

ROUND_NPZ = REPO / "results" / "ibi_relax" / "tables_r4.npz"
NSTEPS = 150


def build(chain, nrep=1):
    L = len(chain["pos"])
    tab = IC.load_tables(str(ROUND_NPZ))
    pos = torch.tensor(np.asarray(chain["pos"], dtype=np.float64).reshape(1, 3 * L, 3)).repeat(nrep, 1, 1)
    vel = torch.zeros_like(pos)
    temps = torch.full((nrep,), 300.0, dtype=torch.float64)
    ij = torch.tensor(list(chain["pairs"]), dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(chain["pairs"]), dtype=torch.float32)
    con = C.make_intra_constraints(L)
    return L, tab, pos, vel, temps, ij, pw, con


def run(pos, vel, temps, ij, pw, con, tab, pot_kw, relax=0, nsteps=NSTEPS):
    t0 = time.perf_counter()
    res = IC.run_round(pos=pos, vel=vel, ij=ij, pw=pw, temps=temps, tab=tab,
                       nsteps=nsteps, burn=0, stride=5, blocks=8, friction=1.0,
                       force_cap=5000.0, pot_kw=pot_kw, seed=11, nrep=pos.shape[0],
                       progress=False, constraints=con, relax=relax,
                       log=lambda *a, **k: None)
    return time.perf_counter() - t0, res


def main():
    pool = [s for s in B.load_structures(limit=5000) if len(s["pairs"]) >= 8][:867]
    chain = max(pool, key=lambda s: len(s["pos"]))
    L, tab, pos, vel, temps, ij, pw, con = build(chain)
    pots, pot_kw = IL._build_potentials(ROUND_NPZ)
    print(f"chain {chain['name']} L={L} beads={3*L}, table {ROUND_NPZ.name}, "
          f"{NSTEPS} steps, burn 0, stride 5, 1 torch thread")

    # (2) the entry relaxation, on its own timer, before the sampler is called.
    p2 = pos.clone()
    t0 = time.perf_counter()
    _, info = IC.relax_positions(p2, ij, pw, pot_kw=pot_kw, constraints=con, n_steps=5000,
                                 force_cap=5000.0, log=None)
    t_relax = time.perf_counter() - t0
    print(f"relax: {t_relax:8.1f} s  ({info['accepted']} accepted / {info['rejected']} rejected, "
          f"{info['evals']} evals) -> {t_relax/3600:.2f} h of a task")

    # (1)+(3) the sampler, as production runs it, then with the clash diagnostic patched out.
    t_full, _ = run(pos.clone(), vel.clone(), temps, ij, pw, con, tab, pot_kw)
    per_full = t_full / NSTEPS
    real = IC.closest_bead_pair
    IC.closest_bead_pair = lambda p, chunk=512: (float("nan"), 0, 0)
    try:
        t_noclash, _ = run(pos.clone(), vel.clone(), temps, ij, pw, con, tab, pot_kw)
    finally:
        IC.closest_bead_pair = real
    per_nc = t_noclash / NSTEPS
    print(f"run_round as production: {per_full*1e3:8.1f} ms/step -> {per_full*32500/3600:5.2f} h "
          f"for 32500 steps")
    print(f"       clash patched out: {per_nc*1e3:8.1f} ms/step -> {per_nc*32500/3600:5.2f} h")
    print(f"       clash share      : {(per_full-per_nc)/per_full*100:8.1f} % of a step, "
          f"{(per_full-per_nc)*32500/3600:5.2f} h of a round's task")
    print(f"       isolated field   : {521.26:8.1f} ms/call -> the integrator+list overhead is "
          f"{per_nc*1e3 - 521.26 - 679.30/5:6.1f} ms/step on top of it")


if __name__ == "__main__":
    main()
