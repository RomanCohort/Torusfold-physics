"""The closed IBI loop: sample under the round-r table, update, repeat.

scripts/ibi_bonded.py is the update half and scripts/ibi_round0.py is one round of the sampling
half; neither drives the other. Its module docstring names this file's absence ("A driver for it
is four calls") and this is that driver.

CHAINS RUN IN PARALLEL PROCESSES. They are independent simulations, and one process at a time
with torch's default thread count leaves the machine almost idle: measured on a 32-logical-CPU
box, a single process gets 0.041 M bead-steps/s while the same work split 16 ways gets ~16x the
wall clock for the same total work. The per-worker thread count is what makes that work out, and
it is 2 rather than 32 because the field is a few hundred TINY ops per call -- intra-op thread
synchronisation costs more than the work it splits:

    one cg_energy_forces call, 1L2X L=27, B=16:  1 thread 4.589 ms   4  4.684   16  8.142   32  9.594
    full step, M bead-steps/s, 1 thread:  B=16 0.073   B=64 0.107   B=256 0.127   B=1024 0.129

so `workers x threads` should equal the logical CPU count, with threads=2.

The GPU is NOT used and that is a measurement, not an omission: bead-steps/s is flat from B=64
to B=16384 at 0.51 GiB of 99.7 GiB, and the CPU stays ahead at every point, while the same box
does a 4096^3 matmul 35x faster on the GPU. The field is kernel-launch bound; a bigger batch
makes each kernel bigger without making fewer of them.

Three things about the loop itself are load-bearing:

1. P_REF COMES FROM ROUND 0 AND IS NEVER RECOMPUTED. Deriving it from the table being updated
   makes the correction identically zero, and the run then reports a fixed point it reached by
   construction.

2. THE SAMPLER MUST SAMPLE THE TABLE IT IS UPDATING. cg_potentials.use_table_file installs a
   table that force_reference's potentials CLOSE OVER, so the order is
   use_table_file -> make_potential -> sample, and it must be repeated INSIDE each worker
   process: the closure does not survive pickling.

3. THE BURN IS 40 PS, AND THAT IS A MEASUREMENT. Two chains run from their deposited coordinates
   with burn=0 and the window cut into ten 4 ps blocks gave sigma per block for bb_bond:

       1NTA L=21  0.0757 0.0758 0.0673 0.0644 0.0606 0.0567 0.0528 0.0541 0.0523 0.0515
       3NPQ L=39  0.0751 0.0827 0.0755 0.0695 0.0653 0.0604 0.0583 0.0581 0.0555 0.0514

   Monotone across the whole 40 ps with no plateau, block 1 41-46 percent above the last two,
   and the dihedral 46-83 percent above and still falling. So the transient is longer than
   40 ps and the burn cannot be shortened; sampling inside it biases the update toward a
   transient. Note also where sigma is heading -- 0.051 against a reference of 0.0470 -- so the
   chains relax toward a state that does not match the reference.

What is NOT updated, and why:
  * intra_pc, intra_cn -- rigid constraints (rigid_bonds.py). sigma_sim is exactly 0.0, so they
    have no marginal to invert; run_round excludes them from binning and from J.
  * stack -- B.COORDS carries it and cg_potentials.COORDS does not, so its table cannot be
    injected. It is copied forward unchanged and its J is still reported.

RESUMING (--start-round=N). These are 21-hour sampling rounds on the 867-chain pool, and a
machine that goes down in round 2 used to cost all of it: nothing was on disk that a later
process could start from.

    python scripts/ibi_loop.py 867 4 1 32500 5 --start-round=1

Two things a resumed process has to get right, and both are deliberate here:

  * p_ref STILL COMES FROM REF_NPZ. What changes round to round is the table the sampler runs
    under; the target is fixed. So a resume loads tables_r{N}.npz as the table to simulate and
    REF_NPZ as the target, and refuses by name when the two do not share bins (lo/hi/binw/
    sigma). p_ref is compared against the histogram bin by bin, so a table on a different
    support would either raise inside plan_update or -- worse -- line up and compare the wrong
    bins, which is the failure mode this whole file is written against.
  * THE DIVERGENCE HISTORY IS REBUILT FROM round0..N-1.json. plan_update refuses a round whose
    correction has risen for PATIENCE consecutive rounds and is GROWTH times its value at the
    start of that window. A resumed process with an empty history cannot see a rise it is in the
    middle of: it would need two more rounds before the rule has anything to compare against,
    and every round in between is one a continuous run would have refused.

Run: python scripts/ibi_loop.py <n_structures> <n_rounds> [nrep] [nsteps] [stride]
                                  [--start-round=N]
     python scripts/ibi_loop.py 7 4 16 32500 25
     python scripts/ibi_loop.py 867 4 1 32500 5 --start-round=1
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B              # noqa: E402
import cg_potentials as P                 # noqa: E402
import ibi_bonded as I                    # noqa: E402
import ibi_core as IC                     # noqa: E402
import torusfold.scheme2.torch_cgsim as C  # noqa: E402

# Which table defines p_ref. IBI_LOOP_REF points it at a refit; the reference is the loop's
# TARGET, so the run prints the absolute path rather than a file name.
REF_NPZ = Path(os.environ.get(
    "IBI_LOOP_REF", str(REPO / "results" / "boltzmann_tables_clean.npz")))
OUT_ROOT = Path(os.environ.get("IBI_LOOP_OUT", str(REPO / "results" / "ibi_loop")))

# The three coordinates cg_potentials can inject.
UPDATED = ("bb_bond", "angle", "dihedral")
CARRIED = ("stack",)

# 40 ps at dt = 0.002 ps; see the docstring for why it is not shorter.
DEFAULT_BURN_STEPS = 20000
# Fraction of each round's correction to apply, PER COORDINATE.
#
# A single value would be the wrong shape. Measured over six rounds on the 7-chain pool with
# everything else fixed (friction 1.0, table smoothed, wall 2000), max|dU| in kJ/mol:
#
#     round      0     1     2     3     4     5
#     bb_bond  7.94  2.78  2.32  2.12  1.90  1.88    monotone, converging
#     dihedral 5.52  1.22  0.56  0.27  0.30  0.29    converged (tol is 0.249)
#     angle    4.85  5.56  5.08  5.33  3.74  6.28    OSCILLATING, no trend
#
# Two of the three converge at gain 1.0 and one does not, so damping is applied where it is
# needed rather than to the loop as a whole -- halving bb_bond and dihedral would only slow two
# coordinates that are already reaching their fixed point.
#
# IBI_LOOP_GAIN sets the default for every coordinate; IBI_LOOP_GAIN_<COORD> overrides one.
GAIN = float(os.environ.get("IBI_LOOP_GAIN", 1.0))
GAIN_BY_COORD = {c: float(os.environ.get(f"IBI_LOOP_GAIN_{c.upper()}", GAIN)) for c in UPDATED}
# "table" alone has no restoring force outside its support (both edge slopes of the shipped
# bb_bond table are exactly 0.0), so the wall is what bounds an excursion.
#
# 200 IS NOT ENOUGH, AND THE ASYMMETRY SAYS WHY. Measured on 1L2X at friction 1.0, 20 ps, the
# excursion is one-sided: bb_bond reaches 0.9916 nm against an upper support edge of 0.8095,
# i.e. 0.182 nm out, while the lower edge is only grazed by 0.0465. The compressed side is held
# by the table's own lower tail plus K_CLASH and the excluded volume; the stretched side is held
# by nothing but this wall, and at k=200 an excursion of 0.18 nm costs 0.5*200*0.18^2 = 3.2
# kJ/mol = 1.3 kBT. The harmonic it replaced charges 465 kJ/mol for the same displacement.
#
# Raising it works, but ONLY once the heating is fixed -- at 7597 K a stiff wall was worse than
# a soft one, because a wall firing on an already-pumped system injects more. At friction 1.0:
#
#     wall k     T settles    bb_bond outside support    bb_bond sd
#      200         569 K             1.20%                 0.0777
#      800         611 K             1.20%                 0.0817
#     2000         573 K             0.48%                 0.0735
#
# The temperature is unchanged and sd falls only 5 percent, so 2000 trims the escaping tail
# rather than distorting the distribution. 800 and 200 are statistically indistinguishable.
WALL_K = float(os.environ.get("IBI_LOOP_WALL", 2000.0))

# Entry relaxation: descent steps on the injected Hamiltonian before the first Langevin step.
# 0 -- the default -- is the historical protocol, straight to 300 K from the deposited
# coordinates, which is what round 0 of the 867-chain run sampled under. See
# ibi_core.relax_positions for what the twelve melting chains look like without it.
#
# READ PER WORKER, NOT PASSED THROUGH A TASK TUPLE, and that is deliberate. A spawn worker
# imports this file again at every round, so an edit here reaches a run that is already in
# flight while its parent keeps the old code in memory. What makes that safe is the task tuple's
# SHAPE: adding an element to it would make the workers unpack something the parent never built
# and kill every round after the edit, while reading the environment is invisible to them.
RELAX_STEPS = int(os.environ.get("IBI_LOOP_RELAX", 0))

# 1 THREAD PER WORKER, AND ONE WORKER PER CPU. Measured aggregate throughput over the machine,
# 32 logical CPUs, L=29, 16 replicas, the real step (2026-09-14):
#
#     workers x threads    each M bead-steps/s    SUM M bead-steps/s
#        1 x 32                  0.0294                 0.0294
#        4 x  8                  0.0274                 0.1098
#       16 x  2                  0.0224                 0.3581
#       32 x  1                  0.0184                 0.5895   <- best
#
# The per-process rate FALLS as workers rise and threads fall (0.0294 -> 0.0184) while the total
# RISES 20x. That is the opposite of the intra-process result, where extra threads make one
# process slower; the two are not in conflict, because the cost there is intra-op
# synchronisation and here it is that N workers each doing their own chain simply do N times the
# work. So the machine is best used as MANY SINGLE-THREADED PROCESSES, not one wide one.
_N_THREADS = int(os.environ.get("IBI_LOOP_THREADS", 1))
_N_WORKERS = int(os.environ.get("IBI_LOOP_WORKERS", max(1, os.cpu_count() or 2)))


def _opt(i, default):
    return type(default)(sys.argv[i]) if len(sys.argv) > i else default


def _flag_int(name, default):
    """`--name=N` from the command line, else the default. ibi_update.py's _opt, typed."""
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            raw = a[len(pre):]
            if not raw.lstrip("-").isdigit():
                raise SystemExit(f"--{name}= wants an integer, got {raw!r}")
            return int(raw)
    return default


# --------------------------------------------------------------------------- worker
def _build_potentials(round_npz):
    """use_table_file FIRST, then make_potential: the potential closes over the current table.

    Called inside every worker rather than in the parent, because the closure cannot be pickled
    and a worker that inherited only the table's PATH would be one use_table_file() call away
    from sampling the wrong potential.
    """
    P.use_table_file(str(round_npz))
    pots = []
    for coord in UPDATED:
        spec = P.resolve_spec(f"table_wall:{WALL_K:g}", coord) if coord == "bb_bond" \
            else P.resolve_spec("table", coord)
        pots.append((coord, spec, P.make_potential(coord, spec)))
    return pots, P.potential_kwargs(pots)


def _sample_one(task):
    """One chain, one round, in its own process. Returns counts, not samples."""
    (round_npz, pos_np, pairs, nrep, nsteps, burn, stride, blocks, friction, seed, threads) = task
    torch.set_num_threads(threads)
    L = pos_np.shape[0]
    _pots, pot_kw = _build_potentials(round_npz)
    tab = IC.load_tables(str(round_npz))
    pos = torch.tensor(pos_np.reshape(1, 3 * L, 3), dtype=torch.float64).repeat(nrep, 1, 1)
    vel = torch.zeros_like(pos)
    temps = torch.full((nrep,), 300.0, dtype=torch.float64)
    ij = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(pairs), dtype=torch.float32)
    con = C.make_intra_constraints(L)
    # THE ENTRY STATE, recorded before anything moves. Round 0 carried no such record, so finding
    # the twelve melting chains afterwards took a re-derivation from the per-structure
    # out-of-support counts; one field evaluation per chain per round buys the distribution of
    # E and max|F| under THIS round's injected potential, which is the number a gate would use and
    # the number that says how many starts are already sitting on the force cap.
    #
    # It is a new KEY in the returned dict and not a new element in the task tuple: the parent
    # passes every key it does not recognize straight into the round json, while the tuple's shape
    # is fixed by whichever code built it -- see RELAX_STEPS above.
    with torch.no_grad():
        _cl0 = C.GPUCellList(cell_size=1.5)
        _cl0.build(pos)
        _e0, _f0 = C.cg_energy_forces(pos, ij, pw, cell_list=_cl0, force_cap=5000.0, **pot_kw)
        _fmax0 = float(torch.linalg.norm(_f0.reshape(-1, 3), dim=-1).max())
    entry = {"energy_0": float(_e0.mean()), "max_force_0": _fmax0,
             "at_cap_0": bool(_fmax0 >= 5000.0)}
    t0 = time.time()
    res = IC.run_round(pos=pos, vel=vel, ij=ij, pw=pw, temps=temps, tab=tab,
                       nsteps=nsteps, burn=burn, stride=stride, blocks=blocks, friction=friction,
                       force_cap=5000.0, pot_kw=pot_kw, seed=seed, nrep=nrep, progress=False,
                       constraints=con, relax=RELAX_STEPS, log=lambda *a, **k: None)
    _wv, _wj = IC.simref(res.acc, tab, skip=res.skip)
    _u, _o = IC.j_denominator(res.acc, tab, skip=res.skip)
    return {
        "entry": entry,
        "relax": res.relax,
        "counts": {c: np.asarray(res.counts[c], dtype=np.int64) for c in B.COORDS},
        "n_outside": {c: int(res.n_outside[c]) for c in B.COORDS},
        "n_total": {c: int(res.n_total[c]) for c in B.COORDS},
        "joint_J": None if _wj != _wj else float(_wj),
        "j_coords": [_u, _o],
        "residues": L,
        "seconds": time.time() - t0,
    }


# ----------------------------------------------------------------------------- resume
def history_from_rounds(out_root, start_round, coords=UPDATED):
    """Previous rounds' applied max|dU| per coordinate, oldest first, out of the round jsons.

    This is plan_update's divergence history. Dropping it on a resume is not a smaller version
    of the same check, it is a different one: the rule needs PATIENCE+1 values before it can
    fire at all, so a resumed run would wave through exactly the first rounds after a rise.
    """
    hist = {c: [] for c in B.COORDS}
    for r in range(start_round):
        path = out_root / f"round{r}.json"
        if not path.exists():
            raise SystemExit(
                f"round {r} of this run has no {path}, so the divergence history a resume needs "
                f"cannot be rebuilt. Running anyway would check the first rounds after the "
                f"resume against an empty history, which is a weaker test than a continuous run "
                f"applies -- refuse rather than quietly change the guard.")
        prev = json.loads(path.read_text(encoding="utf-8"))
        for c in coords:
            hist[c].append(float(prev["updates"][c]["max_abs_dU"]))
        print(f"  history from round{r}.json: "
              + " ".join(f"{c}={hist[c][-1]:.4f}" for c in coords))
    return hist


def check_bins_agree(tables, ref_tables, what, coords=UPDATED):
    """Refuse a resumed table that is not on the reference's bins.

    lo/hi/binw decide which bin a sample lands in and sigma is the denominator every sim/ref is
    reported in; write_round_file copies all four from the reference, so a mismatch means this
    table came from a different reference file. Comparing bins that are not the same bins does
    not raise -- it produces a correction that looks like every other correction.
    """
    for c in coords:
        for key in ("lo", "hi", "binw", "sigma"):
            a, b = float(tables[c][key]), float(ref_tables[c][key])
            if abs(a - b) > 1e-9 * max(1.0, abs(b)):
                raise SystemExit(
                    f"{what}: {c}__{key} is {a!r}, the reference's is {b!r}. The update compares "
                    f"this round's histogram against p_ref bin by bin, so a table on different "
                    f"bins is either a mis-fit reference or a silent misalignment.")
        if len(tables[c]["U"]) != len(ref_tables[c]["U"]):
            raise SystemExit(
                f"{what}: {c} has {len(tables[c]['U'])} bins, the reference has "
                f"{len(ref_tables[c]['U'])}")


# ----------------------------------------------------------------------------- main
def main():
    n_struct = _opt(1, 7)
    n_rounds = _opt(2, 4)
    # --start-round=N is a resume: it samples under tables_r{N}.npz instead of the reference and
    # rebuilds the divergence history from the round jsons. See the module docstring.
    start_round = _flag_int("start-round", int(os.environ.get("IBI_LOOP_START_ROUND", 0)))
    if not 0 <= start_round < n_rounds:
        raise SystemExit(
            f"start_round={start_round} is outside the rounds this run would do "
            f"(0..{n_rounds - 1}). A resume starts AT a round, so continuing a finished run "
            f"means raising the round count as well, e.g. `867 6 1 32500 5 --start-round=4`.")
    nrep = _opt(3, 16)
    nsteps = _opt(4, 32500)
    stride = _opt(5, 25)
    # FRICTION 1.0, NOT 0.1, AND IT DOES NOT CHANGE THE PHYSICS. The Langevin stationary
    # distribution exp(-U/kBT) is independent of gamma; gamma sets only how fast it is reached.
    # So for equilibrium marginals -- which is all IBI measures -- 1.0 is as valid as 0.1 and it
    # dissipates what 0.1 cannot.
    #
    # What it has to dissipate, measured 2026-09-14 on 1L2X, 16 replicas, 20 ps in 8 blocks,
    # table injected, kinetic temperature at the end of each arm:
    #
    #     injected table            friction   T settles at   bb_bond outside support
    #     none (shipped field)        0.1         525 K            0.0%
    #     record table,  120 bins     0.1         577 K            0.2%
    #     refit_full_1000, 1000 bins  0.1        7597 K  RISING   11.1%
    #     refit_full_1000, 1000 bins  1.0         569 K            0.2%
    #
    # The 1000-bin table is a PIECEWISE-LINEAR potential, so its force is discontinuous at every
    # bin edge, and a symplectic integrator pumps energy on a discontinuous force. Going from 120
    # bins to 1000 multiplied the jump by 8 (it is dU/binw) and the number of edges crossed per
    # step by 8 as well. Halving dt does NOT help -- the jumps are in space, so smaller steps
    # cross the same edges -- and a stiffer wall makes it worse. That temperature is what turned
    # this loop's first run from J 0.43 into J 3.7: sigma scales as sqrt(T), so at 7597 K the
    # sampled bb_bond is ~5x too wide, 12 percent of it lands outside the reference support, the
    # update is refused on support_drift, the table freezes, and the other coordinates' updates
    # then push the bond the rest of the way out.
    #
    # It also explains part of every sim/ref > 1 this project has recorded: the earlier rounds all
    # ran at friction 0.1, i.e. ~577 K on the 120-bin table, which widens sigma by sqrt(577/300)
    # = 1.39 on its own.
    friction = float(os.environ.get("IBI_LOOP_FRICTION", 1.0))
    seed = 20260914

    allow_short = "--allow-short-burn" in sys.argv
    burn = max(DEFAULT_BURN_STEPS, nsteps // 5)
    if burn >= nsteps:
        if not allow_short:
            raise SystemExit(
                f"nsteps={nsteps} is at or below the {burn}-step burn floor. The transient runs "
                f"past 40 ps (see the docstring), and sampling inside it fits the table to a "
                f"transient. Raise nsteps above {burn + 5000}, or pass --allow-short-burn to "
                f"smoke-test the wiring -- the run is then stamped short_burn and its tables are "
                f"not results.")
        burn = max(1, nsteps // 5)
        print(f"!! SHORT BURN: burn {burn} of {nsteps} steps, wire-testing only.\n")

    # WHICH CHAINS. "small" is the 24-34 residue band every earlier round used; "all" is every
    # chain the loader extracts, which on _cgdata/combined is 867 of them, 21 to 2929 residues.
    # The band was a speed choice while the field was being validated, and it is the wrong pool
    # for a reference: the fitted tables come from all of them, so sampling only the short ones
    # compares a 7-chain ensemble against a database whose median chain is 72 residues and whose
    # longest is 2929. Use "all" for a round whose result is meant to be the table.
    POOL = os.environ.get("IBI_LOOP_POOL", "small")
    if POOL == "all":
        # No pair filter. The reference tables are fitted over every chain the loader
        # extracts, and 106 of the 867 have no Watson-Crick pair at all -- they are ribosome
        # fragments, single strands with nothing complementary inside the fragment. Requiring a
        # pair would sample 761 of the 867 and compare that ensemble against a reference built
        # from all of them. An empty pair list is legal: measured, cg_energy_forces returns
        # finite energies and forces with an (0, 2) index tensor.
        pool = B.load_structures(limit=5000)
    else:
        pool = [s for s in B.load_structures(limit=5000) if len(s["pairs"]) >= 8
                and 24 <= len(s["pos"]) <= 34]
    if len(pool) < n_struct:
        raise SystemExit(f"asked for {n_struct} structures, the pool has {len(pool)}")
    structs = pool[:n_struct]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"closed IBI loop: {n_struct} structures x {n_rounds} rounds"
          + (f", resuming at round {start_round}" if start_round else ""))
    print(f"  reference {REF_NPZ}  (p_ref from here, in every round, never recomputed)")
    print(f"  {_N_WORKERS} worker processes x {_N_THREADS} torch threads "
          f"= {_N_WORKERS * _N_THREADS} of {os.cpu_count()} logical CPUs")
    print(f"  {nrep} replicas, {nsteps} steps = {nsteps * 0.002:.0f} ps, burn {burn} = "
          f"{burn * 0.002:.0f} ps, window {(nsteps - burn) * 0.002:.0f} ps at stride {stride}")
    print(f"  friction {friction}/ps, 300 K, constraints ON, wall_k {WALL_K:g}, gain " + " ".join(f"{c}={GAIN_BY_COORD[c]:g}" for c in UPDATED)
          + (f", relax {RELAX_STEPS} steps" if RELAX_STEPS else ""))
    # Truncated: the all-chains pool is 867 names, which buries the rest of the header. The full
    # list is recoverable from the round json's per_structure entries.
    _ls = [len(s["pos"]) for s in structs]
    _names = ", ".join(f"{s['name']}(L={len(s['pos'])})" for s in structs[:6])
    print(f"  {len(structs)} structures, L {min(_ls)}-{max(_ls)}: {_names}"
          + (f", ... +{len(structs) - 6} more" if len(structs) > 6 else ""))
    print(f"  updated {UPDATED}; carried {CARRIED}; excluded {tuple(B.CONSTRAINED)} (rigid)\n")

    ref_tables = I.load_clean_tables(REF_NPZ)
    # The target is the reference file's, in a resume exactly as in a fresh run: what changed
    # during the rounds already done is the table that was sampled, not what it is aiming at.
    p_ref = {c: I.probability_from_table(ref_tables[c]) for c in B.COORDS}
    if start_round:
        resume_npz = OUT_ROOT / f"tables_r{start_round}.npz"
        if not resume_npz.exists():
            raise SystemExit(
                f"--start-round={start_round} wants the table that round {start_round} was to "
                f"sample, {resume_npz}, and it is not there. That file is written at the end of "
                f"round {start_round - 1}; without it there is nothing to resume from, and a "
                f"fresh run is the honest option.")
        tables = I.load_clean_tables(resume_npz)
        check_bins_agree(tables, ref_tables, str(resume_npz))
        print(f"  resume: round {start_round} samples under {resume_npz.name}, "
              f"target {REF_NPZ.name}")
    else:
        tables = ref_tables
    hist_by_coord = history_from_rounds(OUT_ROOT, start_round)

    def write_round_file(path, tabs):
        """A file use_table_file and ibi_core.load_tables can BOTH read.

        sigma and centre come from the REFERENCE and are copied, not recomputed: they describe
        the target, not the iteration.
        """
        payload = {}
        for c, t in tabs.items():
            payload[f"{c}__lo"] = float(t["lo"])
            payload[f"{c}__hi"] = float(t["hi"])
            payload[f"{c}__binw"] = float(t["binw"])
            payload[f"{c}__U"] = np.asarray(t["U"], dtype=float)
            payload[f"{c}__centre"] = np.asarray(t["centre"], dtype=float)
            payload[f"{c}__sigma"] = float(t["sigma"])
        np.savez(path, **payload)

    round_log = []
    for rnd in range(start_round, n_rounds):
        t_round = time.time()
        round_npz = OUT_ROOT / f"tables_r{rnd}.npz"
        # On a resumed round this rewrites the file it just loaded. That is deliberate: what the
        # workers sample is the file on disk, so the two cannot drift apart.
        write_round_file(round_npz, tables)
        print(f"=== round {rnd} ===")
        print(f"  sampling under {round_npz.name}")

        # Split each chain's replicas across several workers. With a pool of 7 chains and 32
        # CPUs, one worker per chain leaves 25 cores idle no matter how the threads are set, and
        # the split has to be over the WORK, not over the threads. Replicas are independent
        # trajectories, so slicing them and summing the histograms is exact -- it changes the
        # sampling, not the estimator. The chunks land on the SAME chain, so the batch within one
        # worker is just smaller; that is why per-worker throughput drops and the sum does not.
        per_chain_chunks = max(1, _N_WORKERS // max(1, len(structs)))
        tasks, task_owner = [], []
        for i, s in enumerate(structs):
            base, extra = divmod(nrep, per_chain_chunks)
            for k in range(per_chain_chunks):
                rep = base + (1 if k < extra else 0)
                if rep <= 0:
                    continue
                tasks.append((str(round_npz), np.asarray(s["pos"], dtype=np.float64),
                              list(s["pairs"]), rep, nsteps, burn, stride, 8, friction,
                              seed + 1000 * k + i, _N_THREADS))
                task_owner.append(i)
        print(f"  {len(tasks)} tasks over {len(structs)} chains "
              f"({per_chain_chunks} replica-chunks each) on {_N_WORKERS} workers")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(_N_WORKERS, len(tasks))) as pool_procs:
            results = pool_procs.map(_sample_one, tasks)

        # Per-chain line: every chunk of a chain reported its own J, so the chain's line names the
        # spread across its chunks rather than pretending they are one number.
        for i, s in enumerate(structs):
            mine = [r for r, o in zip(results, task_owner) if o == i]
            if not mine:
                continue
            js = [r["joint_J"] for r in mine if r["joint_J"] is not None]
            jtxt = "nan" if not js else (f"{min(js):.4f}-{max(js):.4f}" if len(js) > 1
                                         else f"{js[0]:.4f}")
            n = sum(r["n_total"]["bb_bond"] for r in mine)
            print(f"    {s['name']:9s} L={len(s['pos']):4d}  J {jtxt}  n={n}  "
                  f"({len(mine)} chunks, max {max(r['seconds'] for r in mine):.0f} s)")

        updates = {}
        for c in UPDATED:
            counts = sum((r["counts"][c] for r in results), np.zeros(len(tables[c]["U"]), np.int64))
            n_out = sum(r["n_outside"][c] for r in results)
            n_tot = sum(r["n_total"][c] for r in results)
            hist = I.SimHistogram(counts=counts, n=n_tot, n_outside=n_out,
                                  lo=float(tables[c]["lo"]), hi=float(tables[c]["hi"]),
                                  nbins=len(tables[c]["U"]))
            # history is PER COORDINATE: the divergence guard compares this round's correction
            # against the previous rounds' for the SAME coordinate.
            #
            # GAIN < 1 BECAUSE THE COORDINATES ARE UPDATED TOGETHER. Each correction is a 1-D
            # marginal inversion, but the three coordinates are coupled, so a full step moves
            # each one into a distribution the others just changed. Measured at friction 1.0 and
            # gain 1.0, angle's correction GREW every round -- 6.21, 8.72, 9.65, 10.46 kJ/mol --
            # the signature the divergence guard exists to catch, and joint J rose with it
            # (0.19 -> 0.42) while bb_bond's out-of-support fraction rose 0.86 -> 5.53 percent.
            # Damping the step is the standard remedy for a coupled fixed-point iteration.
            #
            # SMOOTH BECAUSE THE CORRECTION IS A HISTOGRAM RATIO. kBT*ln(P_sim/P_ref) carries the
            # same Poisson noise the reference table does, and the table it is added to is
            # piecewise linear, so an unsmoothed correction re-injects exactly the random force
            # field that heats the sampler. See boltzmann_bonded.SMOOTH_WIDTH: the bin convention
            # there is a WINDOW WIDTH while smooth_correction's is +-bins, hence the conversion.
            res = I.plan_update(tables[c], hist, p_ref[c], history=hist_by_coord[c], gain=GAIN_BY_COORD[c],
                                smooth_bins=(B.SMOOTH_WIDTH - 1) // 2)
            entry = {"ok": bool(res.ok), "converged": bool(res.converged),
                     "max_abs_dU": float(res.max_abs_dU), "n_samples": int(n_tot),
                     "n_outside": int(n_out)}
            hist_by_coord[c].append(float(res.max_abs_dU))
            try:
                new_table = res.require_table()
                entry["status"] = "applied"
            except I.IBIRefusal as exc:
                new_table = None
                entry["status"] = "refused"
                entry["reason"] = str(exc)
            updates[c] = entry
            print(f"  update {c:9s} {entry['status']:8s} max|dU|={entry['max_abs_dU']:.4f} "
                  f"kBT={entry['max_abs_dU'] / B.KBT:.4f}  n={entry['n_samples']} "
                  f"outside={entry['n_outside']}"
                  + ("" if entry["status"] == "applied" else f"\n      {entry.get('reason','')}"))
            if new_table is not None:
                tables[c] = dict(tables[c], U=np.asarray(new_table["U"], dtype=float))
        for c in CARRIED:
            print(f"  carried {c:9s} unchanged (no injection path for it)")

        round_log.append({"round": rnd, "seconds": time.time() - t_round,
                          "per_structure": [{k: v for k, v in r.items() if k != "counts"}
                                            for r in results],
                          "updates": updates, "short_burn": bool(burn < DEFAULT_BURN_STEPS)})
        (OUT_ROOT / f"round{rnd}.json").write_text(
            json.dumps(round_log[-1], indent=2, default=float), encoding="utf-8", newline="\n")
        write_round_file(OUT_ROOT / f"tables_r{rnd + 1}.npz", tables)
        print(f"  wrote tables_r{rnd + 1}.npz  ({time.time() - t_round:.0f} s for this round)\n")

    print(f"done: {n_rounds} rounds in {sum(r['seconds'] for r in round_log):.0f} s")
    print("Round files and per-round json are in " + str(OUT_ROOT))


if __name__ == "__main__":
    main()
