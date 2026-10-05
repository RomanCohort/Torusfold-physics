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
  * stack -- B.scored_coords() carries it and cg_potentials.COORDS does not, so its table cannot be
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

EVERY TASK'S RESULT IS WRITTEN TO DISK AS IT ARRIVES, under OUT_ROOT/tasks_r<N>/<idx>.npz,
with a heartbeat file beside it. Two reasons, both measured: four workers of round 0 died on
2026-09-16 (VCRUNTIME140.dll and ucrtbase.dll 0xc0000409), the Pool respawned them at the same
seconds so the run LOOKED healthy, and the tasks they held were lost forever while pool.map
waited for results that could never arrive -- 29.7 h in, with every finished chain's histogram in
the parent's memory. Now a dead worker is visible (nothing is beating), its tasks are re-run in a
fresh pool, and a round that is killed resumes from the results already on disk instead of
starting over.

Run: python scripts/ibi_loop.py <n_structures> <n_rounds> [nrep] [nsteps] [stride]
                                  [--start-round=N]
     python scripts/ibi_loop.py 7 4 16 32500 25
     python scripts/ibi_loop.py 867 4 1 32500 5 --start-round=1
"""
import json
import multiprocessing as mp
import os
import sys
import threading
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
import plan_c_basis as PB                 # noqa: E402
import torusfold.scheme2.torch_cgsim as C  # noqa: E402

# Which table defines p_ref. IBI_LOOP_REF points it at a refit; the reference is the loop's
# TARGET, so the run prints the absolute path rather than a file name.
REF_NPZ = Path(os.environ.get(
    "IBI_LOOP_REF", str(REPO / "results" / "boltzmann_tables_clean.npz")))
OUT_ROOT = Path(os.environ.get("IBI_LOOP_OUT", str(REPO / "results" / "ibi_loop")))

# The three coordinates cg_potentials can inject.
UPDATED = ("bb_bond", "angle", "dihedral")
# CARRIED: measured every round and scored in the joint J, but with no table injected as a potential.
# `stack` has been here since it was found to be an algebraic function of the bond and the angle.
#
# The base-level coordinates joined it on 2026-10-05, and for the same reason in a different place: the
# model's ACCESS to them is the bead-to-plane map, and their dynamics come from base_stacking's term
# (IBI_LOOP_BASE_STACK), not from a 1-D table. What the loop adds by scoring them is the thing it has
# never had -- a scoreboard that includes the coordinates stacking is actually about, so a run can say
# whether the base level is at its target as well as whether the trace is.
CARRIED = ("stack", "base_dist", "base_rise", "base_cos")

# STACK IS NOT A COORDINATE THE LOOP CAN CONTROL, and that is a property of the MODEL, not a policy
# choice. src/torusfold/scheme2/torch_cgsim.py (K_STACK = 0.0, around line 203) records it, and the
# parts that matter here are:
#
#   1. the stack coordinate is the P(i)-P(i+2) distance, an EXACT DERIVED quantity:
#      |P(i)-P(i+2)|^2 = |b_i|^2 + |b_{i+1}|^2 - 2|b_i||b_{i+1}|cos_a, checked against both sides by
#      scripts/assess_stacking_redundancy.py on 1278 windows to 6.7e-16 nm^2 (R^2 = 1.000000). The two
#      bond lengths are held by K_BB and cos_a by K_ANGLE, so the term is a THIRD SPRING ON A DERIVED
#      QUANTITY;
#   2. ablating it changes nothing physical (funnel rank/gap stay 1.000 / 0.0 while the bonded
#      subset's force-cap saturation falls from 4.02 to 0.58 per cent);
#   3. the model cannot express stacking at all: the beads are P, C4-prime and one N9/N1 point -- no
#      plane, no normal, no rise, no twist. "A term named for stacking that restrains a
#      backbone-derived distance was not doing that job."
#
# So stack's ratio in sim_ref_table is a CONSISTENCY CHECK, not a target: it is bb_bond's and angle's
# error propagated through that identity. Evidence: on round 3 angle's ratio is 1.228 and stack's 1.189
# while bb_bond's is 1.004, i.e. stack follows ANGLE; and over nine full-pool rounds its table-implied
# sigma was constant at 0.12683 while its sampled sigma moved 1.175 -> 1.084 only because the other
# coordinates moved. Any J that averages it carries a constant |ln(sim/ref)| of about 0.17 that no
# update can remove, which is why the mean below leaves it out.
#
# DO NOT add a stack_potential kwarg to "make stack controllable": that would hang a fourth spring on a
# derived quantity. Handling stacking means adding degrees of freedom (planes, normals, more beads per
# residue), which is a different model and not a fitting problem.
UNCONTROLLED = ("stack",)

# WHICH COORDINATES J AVERAGES OVER: the ones the loop is actually converging. Stack is outside by
# construction (it is not in UPDATED); a frozen coordinate is outside by choice.
_FREEZE_RAW = os.environ.get("IBI_LOOP_FREEZE", "")
FROZEN = tuple(x.strip() for x in _FREEZE_RAW.split(",") if x.strip())
_BAD_FREEZE = [c for c in FROZEN if c not in UPDATED]
if _BAD_FREEZE:
    raise SystemExit(f"IBI_LOOP_FREEZE names {_BAD_FREEZE}, which are not updated coordinates; "
                     f"the legal names are {list(UPDATED)}")
CONTROLLED = tuple(c for c in UPDATED if c not in FROZEN)

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
# THE SUPPORT GATE IS A KNOB NOW (2026-10-04), and the record says what refusing it costs.
#
# Arm A refused bb_bond BOTH rounds -- 1.02 then 1.08 percent of observations outside
# [0.373417, 0.809522] against a 1 percent gate -- and bb_bond is the one coordinate whose sampled
# width tracks its own table almost 1:1: measured over the nine campaign rounds, sigma_sampled against
# sigma_implied has slope 1.111 (R2 0.94), against 0.311 for the angle and 0.019 for the stack (which
# has no table at all). Its sampled sigma ends at 0.05405 against a 0.05402 target -- it is the one
# coordinate the loop actually finished, and arm A spent two rounds not touching it.
#
# The gate exists so the moments are not taken on a truncated view (ibi_bonded.plan_update), and the
# size of that truncation is bounded and small here: f*d with f = 0.0102 and the measured excursion
# d = 0.18 nm (this file's wall note) is a mean shift of 0.0018 nm, 3.4 percent of the target sigma --
# the same order as the round-to-round noise in a 3.3e8-sample histogram. So the gate can be raised
# rather than removed, and the fraction is now REPORTED for every coordinate whether or not it refuses.
#
# Default unchanged (0.01): every table in results/ holds this value, and tests/test_ibi_driver_rules
# pins the default path against golden digests. A launcher that wants the working knob to work sets
# IBI_LOOP_SUPPORT_GATE=0.03.
SUPPORT_GATE = float(os.environ.get("IBI_LOOP_SUPPORT_GATE", I.DEFAULT_MAX_OUTSIDE_FRAC))

# BASE-LEVEL STACKING, opt-in, 0 = off and the field is unchanged (2026-10-05). Strengths and shapes are
# measured in findings Parts 15-17: the sum form (independent penalties, broad orientation factor) has
# 111.5 kJ/mol/nm of force at an unstacked geometry against 7.4 for the product-shaped reward, and it is the
# one that creates the rise's one-sidedness; the default weights discount the base-base distance because at
# equal weights it over-constrained its own width (sampled sd 0.122 against the target's 0.198).
BASE_STACK_EPS = float(os.environ.get("IBI_LOOP_BASE_STACK", "0"))
BASE_STACK_FORM = os.environ.get("IBI_LOOP_BASE_STACK_FORM", "sum")
BASE_STACK_W = tuple(float(x) for x in
                     os.environ.get("IBI_LOOP_BASE_STACK_W", "0.3,1,1").split(","))
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

# WHICH UPDATE OPERATOR. "table" is the per-bin marginal inversion every round so far has used;
# "moments" is the relative-entropy step on a low-order correction (Plan B', docs/
# plan_b_coupled_update.md). Default is "table" so that a run already in flight keeps its protocol
# when this file changes under it -- the same reason RELAX_STEPS is read per worker.
OPERATOR = os.environ.get("IBI_LOOP_OPERATOR", "table")
CORRECTION_K = int(os.environ.get("IBI_LOOP_CORRECTION_K", "8"))

# PER-COORDINATE RULES (the two-lever arm, 2026-10-01; docs/archive/plan_c_basis_family.md section 5). An
# empty string means "the operator above, for every coordinate" -- today's behaviour -- and that
# default path is bit-identical: tests/test_ibi_driver_rules.py pins it against golden digests
# captured before this block existed.
#
#   IBI_LOOP_RULE_DIHEDRAL=bspline16   the moment operator with a B-spline design (this many functions,
#     with IBI_LOOP_RULE_BSPLINE_M) and an EIGENVALUE-relative ridge, instead of Chebyshev K=8 with the
#     trace-relative one. The target stays p_ref: same rule, a basis that can carry the dihedral's edge
#     mass. Measured on seven chains: a Chebyshev refit delivers 0.378 less implied edge mass than its
#     target holds and its step cycles (corr -0.94, amplitude growing), while every B-spline arm is
#     within 0.012 of the target and converges.
#   IBI_LOOP_RULE_ANGLE=selfconsistent   the angle is REFITTED to the ensemble the field itself
#     produced, through plan_c_basis.fit on the Chebyshev K=8 design. It is a REPLACEMENT, not an
#     increment: under a self-consistent target the increment kBT*ln(p_sim/p_ref) is identically zero,
#     and a loop that used it would report a fixed point it reached by construction. It does not consume
#     p_ref at all, so the "p_ref comes from the reference and is never recomputed" invariant -- which
#     is a statement about what the INCREMENT is measured against -- is untouched.
#   IBI_LOOP_RULE_RIDGE                relative ridge for the bspline16 rule (default 1e-3).
RULE_BY_COORD = {
    "angle": os.environ.get("IBI_LOOP_RULE_ANGLE", "").strip().lower(),
    "dihedral": os.environ.get("IBI_LOOP_RULE_DIHEDRAL", "").strip().lower(),
}
RULE_BSPLINE_M = int(os.environ.get("IBI_LOOP_RULE_BSPLINE_M", "16"))
RULE_RIDGE_REL = float(os.environ.get("IBI_LOOP_RULE_RIDGE", "1e-3"))
# The self-consistent refit uses the C2s pipeline's own ridge (plan_c_loop.RIDGE_REL, trace-relative),
# so the full-pool angle arm is the same rule as the seven-chain C2s8/CA16 arms and not a new one.
RULE_C2S_RIDGE_REL = float(os.environ.get("IBI_LOOP_RULE_C2S_RIDGE", "1e-1"))
RULE_C2S_SUPPORT_FRAC = float(os.environ.get("IBI_LOOP_RULE_C2S_SUPPORT", "1e-3"))
RULE_C2S_TAPER_DECADES = float(os.environ.get("IBI_LOOP_RULE_C2S_TAPER", "2.0"))

# --- task checkpointing, and the watchdog that reads it --------------------------------------
#
# WHY THIS EXISTS, measured. On 2026-09-16 four pool workers of this loop died (four AppCrash
# reports, VCRUNTIME140.dll at 23:04:18 x3 and ucrtbase.dll 0xc0000409 at 23:08:32) and the Pool
# respawned four replacements AT THOSE SAME SECONDS -- which is why the run looked healthy. The
# tasks the dead workers were holding went with them: multiprocessing.Pool does NOT replay a task
# whose worker died, and pool.map blocks on a result that will never arrive. Round 0 sat there
# until it was killed, 29.7 h in, with every completed chain's histogram held in the parent's
# memory where nothing could reach it.
#
# So: each task's result is written to disk the moment it arrives (done_dir/<idx>.npz), a round
# RESUMES by skipping the indices already on disk, and each worker touches a heartbeat file every
# HEARTBEAT_S so the parent can tell "still integrating" from "dead". A worker that dies is not
# invisible any more: the parent notices that nothing anywhere is beating, and re-runs the
# outstanding tasks in a fresh pool.
HEARTBEAT_S = float(os.environ.get("IBI_LOOP_HEARTBEAT", 15.0))
# How often the parent looks at the heartbeats while it waits for results.
POLL_S = float(os.environ.get("IBI_LOOP_POLL", 60.0))
# A task that HAS beaten and then gone silent for this long is dead: its worker crashed holding
# it, and Pool's replacement is idle because the task was already handed out. A task that has
# never beaten is not evidence of anything -- with 867 tasks and 33 workers, most are still in the
# queue -- which is why the predicate reads mtimes instead of counting missing files.
STALE_S = float(os.environ.get("IBI_LOOP_STALE", 420.0))
# A chain that crashes its worker every attempt is a finding, not something to loop on forever.
MAX_ATTEMPTS = int(os.environ.get("IBI_LOOP_ATTEMPTS", 4))
# TEST HOOK, and the only reason it is in a production path: IBI_LOOP_CRASH_ONCE=<idx[,idx]> makes
# that task's worker die with no traceback the first time it runs -- exactly how the four workers
# of round 0 died -- so the DEAD WORKER path can be exercised instead of hoped for. The marker file
# it leaves is what makes it fire once, so the retry then succeeds. Never set outside a test.
_CRASH_ONCE = os.environ.get("IBI_LOOP_CRASH_ONCE", "")

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


def keep_awake():
    """Ask Windows not to suspend while the loop runs. Returns True when it took effect.

    WHY. The 2026-09-16 round spent 29.7 h of wall clock on what is about 21 h of work, because
    the machine kept entering Modern Standby -- twelve "exiting Modern Standby" events between
    13:56 and 17:40 alone. A suspend does not kill the sampling (the processes freeze and resume,
    and the watchdog now knows the difference), but the round takes half again as long, and an
    idle machine is indistinguishable from a dead run to anybody watching.

    ES_CONTINUOUS | ES_SYSTEM_REQUIRED holds the system awake until this process exits or clears
    it, and it is opt-in (IBI_LOOP_KEEP_AWAKE=1): it changes how the machine behaves for hours, so
    it is the caller's decision and not a default.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        return False


# --------------------------------------------------------------------- task checkpoints
def task_npz(done_dir, idx):
    """Where task `idx`'s result lives. Its existence IS the resume condition."""
    return Path(done_dir) / f"{idx}.npz"


def longest_first(tasks, length_of):
    """The dispatch order: the longest chain goes first.

    WHY. A round's wall clock is the longest chain's own cost PLUS whatever the schedule adds to
    it, and the loader's order is alphabetical. On round 0 of the 867-chain pool the largest chain
    (8FMW_24, 2929 residues) was task 536 of 867, so it started near the end and the last stretch
    of the round ran on ONE core while thirty-one sat idle -- measured on 2026-09-19: 19.9
    CPU-seconds per 20 s of wall, 3 percent of the box. LPT, longest processing time first, is the
    standard answer: the long chains enter the first wave, so the tail overlaps with them instead
    of trailing them.

    `length_of(idx)` returns the chain's size. The task INDEX is deliberately not touched by the
    sort: it names the checkpoint file, and a dispatch order that renumbered the tasks would make
    a resume read one chain's result as another's -- silently, because the files would still be
    valid, just for the wrong chain. So this returns a new order and leaves the indices alone.
    """
    return sorted(tasks, key=lambda t: -length_of(t[0]))


def save_task_result(done_dir, idx, r):
    """Write one chain-round's counts and scalars, and its entry/relax metadata beside them."""
    counts = {f"counts__{c}": np.asarray(r["counts"][c], dtype=np.int64) for c in B.scored_coords()}
    scalars = {}
    for c in B.scored_coords():
        scalars[f"n_outside__{c}"] = int(r["n_outside"][c])
        scalars[f"n_total__{c}"] = int(r["n_total"][c])
    np.savez(task_npz(done_dir, idx),
             joint_J=np.float64(np.nan if r["joint_J"] is None else r["joint_J"]),
             joint_J_all=np.float64(np.nan if r.get("joint_J_all") is None else r["joint_J_all"]),
             joint_J_base=np.float64(np.nan if r.get("joint_J_base") is None else r["joint_J_base"]),
             joint_J_table=np.float64(np.nan if r.get("joint_J_table") is None
                                      else r["joint_J_table"]),
             sim_ref_table=np.asarray([np.nan if v is None else v
                                       for v in (r.get("sim_ref_table") or [np.nan] * len(B.scored_coords()))],
                                      dtype=float),
             residues=int(r["residues"]), seconds=float(r["seconds"]),
             j_coords=np.asarray(r["j_coords"], dtype=np.int64), **counts, **scalars)
    task_npz(done_dir, idx).with_suffix(".json").write_text(
        json.dumps({"entry": r.get("entry"), "relax": r.get("relax")}, indent=1, default=float),
        encoding="utf-8")


def load_task_result(done_dir, idx):
    """The exact dict _sample_one returned, as far as the update half reads it."""
    z = np.load(task_npz(done_dir, idx))
    meta_path = task_npz(done_dir, idx).with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    j = float(z["joint_J"])
    # Both keys are optional on read: tasks written before 2026-09-21 do not carry them, and a
    # replay of those rounds must not invent a number (None, not nan, so the round json says
    # "not measured" rather than "measured as nan").
    # OPTIONAL ON READ: a task file written before 2026-10-01 has no joint_J_all, and replaying it
    # must not fail -- the campaign's own checkpoints have to stay loadable.
    ja = float(z["joint_J_all"]) if "joint_J_all" in z.files else float("nan")
    jt = float(z["joint_J_table"]) if "joint_J_table" in z.files else float("nan")
    srt = (list(np.asarray(z["sim_ref_table"], dtype=float)) if "sim_ref_table" in z.files
           else [float("nan")] * len(B.scored_coords()))
    return {"counts": {c: z[f"counts__{c}"] for c in B.scored_coords()},
            "joint_J_all": None if ja != ja else ja,
            "joint_J_base": (float(z["joint_J_base"]) if "joint_J_base" in z.files else None),
            "joint_J_table": None if jt != jt else jt,
            "sim_ref_table": [None if v != v else float(v) for v in srt],
            "n_outside": {c: int(z[f"n_outside__{c}"]) for c in B.scored_coords()},
            "n_total": {c: int(z[f"n_total__{c}"]) for c in B.scored_coords()},
            "joint_J": None if j != j else j,
            "j_coords": [int(v) for v in z["j_coords"]],
            "residues": int(z["residues"]), "seconds": float(z["seconds"]),
            "entry": meta.get("entry"), "relax": meta.get("relax")}


def dead_tasks(outstanding, stale_s, now=None, since=None):
    """The outstanding tasks that started beating and then stopped. Returns their tuple list.

    A task with NO heartbeat file is not returned: it has not been handed to a worker yet, and
    with 867 tasks over 33 workers that is most of them at any moment. A task that beat and then
    went quiet is the signature of a worker that crashed while holding it -- the Pool respawns the
    worker, and the task it was holding is never re-issued by anybody.

    `since` is the moment this attempt started, and a beat older than it is not evidence: on a
    RESUME the disk is full of heartbeats from the attempt that died, and without this the first
    check of the new attempt declares all of them dead. Measured on 2026-09-17: every restart
    burned a pool cycle on that false verdict and printed "DEAD WORKER: 25 task(s)" before a
    single step had been taken.
    """
    now = time.time() if now is None else now
    out = []
    for t in outstanding:
        try:
            mtime = Path(t[1]).stat().st_mtime
        except OSError:
            continue
        if since is not None and mtime < since:
            continue
        if now - mtime > stale_s:
            out.append(t)
    return out


def touch_heartbeats(outstanding, now=None):
    """Call after a wall-clock jump: silence across a suspend is not a crash.

    A machine that sleeps freezes the parent too, so on wake every heartbeat looks hours old. The
    honest repair is to move them all to now and let the next window speak, rather than to kill
    and re-run a pool whose workers were never dead.
    """
    now = time.time() if now is None else now
    for t in outstanding:
        p = Path(t[1])
        if p.exists():
            try:
                os.utime(p, (now, now))
            except OSError:
                pass


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
    kw = P.potential_kwargs(pots)
    # A BASE-LEVEL STACKING TERM, opt-in, off by default (2026-10-05).
    #
    # The trace coordinates above cannot express stacking: measured over 20 crystal fragments and the
    # field's own sampler, the crystals' base-base distance is 0.570 +- 0.198 nm and their rise along the
    # mean base-plane normal is 0.328 +- 0.188 nm with a 5th percentile at +0.06 -- one-sided, because one
    # base lies OVER its neighbour -- while the sampler gives 0.750 nm and a rise whose 5th percentile is
    # -0.33 nm, i.e. no preference at all (findings Part 15). A term built from those measurements and
    # shaped as INDEPENDENT penalties creates the asymmetry: the rise's 5th percentile moves to +0.02 nm at
    # 20 kJ/mol per pair (Part 17). The same form's max|F| at an unstacked geometry is 111.5 against 7.4 for
    # a product-shaped reward at the same nominal strength, which is why this is the shape that ships here.
    #
    # WHY IT BELONGS IN THE LOOP rather than in a scan: with the term ON, the trace tables are being fitted
    # against an ensemble that includes it -- which is the only honest way to keep both, since the trace
    # tables in results/ were fitted with no base-level term present and a stacking term pulls the trace
    # away from its own fitted marginals (measured: joint J 0.139 -> 0.215 on two chains at 20 kJ/mol).
    #
    # IBI_LOOP_BASE_STACK=0 (default) leaves the field bit-identical to before this block existed.
    if BASE_STACK_EPS > 0.0:
        from torusfold.scheme2.base_stacking import make_base_stack_potential
        kw = dict(kw, base_stack_potential=make_base_stack_potential(
            eps=BASE_STACK_EPS, form=BASE_STACK_FORM,
            w_d=BASE_STACK_W[0], w_r=BASE_STACK_W[1], w_t=BASE_STACK_W[2]))
    return pots, kw


def process_memory_mb():
    """(working set, commit) of THIS process in MB, or (nan, nan) if the API is not there.

    The heartbeat carries it because the round that died on 2026-09-16 died of memory exhaustion,
    and nothing recorded which process grew, on which chain, or how fast: it had to be
    reconstructed afterwards from Windows' Resource-Exhaustion events.
    """
    if os.name != "nt":
        return float("nan"), float("nan")
    try:
        import ctypes
        import ctypes.wintypes as wt
        global _PMC, _K32
        try:
            _PMC
        except NameError:
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
            _K32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
            _K32.K32GetProcessMemoryInfo.restype = wt.BOOL
        c = _PMC()
        c.cb = ctypes.sizeof(c)
        if not _K32.K32GetProcessMemoryInfo(_K32.GetCurrentProcess(), ctypes.byref(c),
                                            ctypes.sizeof(c)):
            return float("nan"), float("nan")
        return c.WorkingSetSize / 1e6, c.PagefileUsage / 1e6
    except Exception:
        return float("nan"), float("nan")


def _beat_write(path):
    """Timestamp, pid and memory.

    The pid maps a growing process to the chain in the file name beside it; the memory says
    whether it is growing at all. Without both, a memory death is invisible until Windows says so.
    """
    ws, commit = process_memory_mb()
    Path(path).write_text(f"{time.time():.0f} pid={os.getpid()} ws_mb={ws:.0f} "
                          f"commit_mb={commit:.0f}\n", encoding="utf-8")


def _sample_one(task):
    """One chain, one round, in its own process. Returns (task index, counts -- not samples).

    The index comes back because the PARENT writes the result to disk the moment it arrives. A
    result that only exists in the parent's memory is a result a crash throws away, and that is
    exactly what happened to round 0 on 2026-09-16: 29.7 h of sampling, four workers dead, every
    completed histogram unreachable.
    """
    (idx, hb_path, round_npz, pos_np, pairs, nrep, nsteps, burn, stride, blocks, friction, seed,
     threads) = task
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

    # A beat every HEARTBEAT_S of WALL time, not every N steps: "no beat for four minutes" has to
    # mean the same thing for a 21-residue chain and a 2929-residue one, and in steps it does not.
    # The parent reads these files to tell a slow task from a dead one.
    stop_hb = threading.Event()

    def _beat():
        while not stop_hb.wait(HEARTBEAT_S):
            try:
                _beat_write(hb_path)
            except OSError:
                pass

    try:
        _beat_write(hb_path)
    except OSError:
        pass
    threading.Thread(target=_beat, daemon=True).start()

    if _CRASH_ONCE and str(idx) in _CRASH_ONCE.split(","):
        marker = Path(hb_path).with_suffix(".crashed")
        if not marker.exists():
            marker.write_text("first attempt died here\n", encoding="utf-8")
            os._exit(1)      # no traceback, no cleanup: the way a real worker death looks

    t0 = time.time()
    try:
        res = IC.run_round(pos=pos, vel=vel, ij=ij, pw=pw, temps=temps, tab=tab,
                           nsteps=nsteps, burn=burn, stride=stride, blocks=blocks,
                           friction=friction, force_cap=5000.0, pot_kw=pot_kw, seed=seed,
                           nrep=nrep, progress=False, constraints=con, relax=RELAX_STEPS,
                           log=lambda *a, **k: None)
    finally:
        stop_hb.set()
    # TWO Js. joint_J is the mean over the CONTROLLED coordinates -- what the loop is converging;
    # joint_J_all is the old four-coordinate mean, kept so this arm can be read against rounds 0-8 of
    # the campaign. See UNCONTROLLED above for why stack is not in the first one.
    _wv, _wj = IC.simref(res.acc, tab, skip=res.skip, only=CONTROLLED)
    _av, _aj = IC.simref(res.acc, tab, skip=res.skip)
    # A THIRD J, for the base level, when the run scores it (IBI_SCORE_BASE=1). joint_J stays the mean over
    # the CONTROLLED trace coordinates, so every earlier run's J remains comparable; this one is the number
    # the base-level work is about -- whether the coordinates stacking is actually made of are at their
    # crystal targets, which the loop could not say before 2026-10-05.
    if B.scored_coords() != B.COORDS:
        _bv, _bj = IC.simref(res.acc, tab, skip=res.skip, only=B.BASE_COORDS)
    else:
        _bv, _bj = [], float("nan")
    _u, _o = IC.j_denominator(res.acc, tab, skip=res.skip)
    # THE SAME RESIDUAL AGAINST THE TABLE'S OWN DISTRIBUTION, reported beside the old one and read
    # by nothing in the update path (ibi_core.implied_sigma, Part 5 of the IBI findings). Both are
    # carried because the stored sigma is outlier-inflated for bb_bond and stack, so the old number
    # has a floor for those and the new one does not; keeping both is what makes rounds before and
    # after this change comparable instead of silently re-based.
    _tv, _tj = IC.simref_table(res.acc, tab, skip=res.skip)
    return idx, {
        "entry": entry,
        "relax": res.relax,
        "sim_ref_table": [None if v != v else float(v) for v in _tv],
        "joint_J_table": None if _tj != _tj else float(_tj),
        "counts": {c: np.asarray(res.counts[c], dtype=np.int64) for c in B.scored_coords()},
        "n_outside": {c: int(res.n_outside[c]) for c in B.scored_coords()},
        "n_total": {c: int(res.n_total[c]) for c in B.scored_coords()},
        "joint_J": None if _wj != _wj else float(_wj),
        "joint_J_all": None if _aj != _aj else float(_aj),
        "joint_J_base": None if _bj != _bj else float(_bj),
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
    hist = {c: [] for c in B.scored_coords()}
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
# --------------------------------------------------------------------------- the update, per coordinate
def update_one_coord(coord, table, counts, n_tot, n_out, p_ref, hist, hist_norm, hist_by_coord,
                     rule=""):
    """One coordinate's update for one round: (UpdateResult, the rule that ran).

    THE DEFAULT PATH IS THE CODE THAT PRODUCED EVERY TABLE IN results/ibi_relax, moved here verbatim so
    it can be tested without running a round. tests/test_ibi_driver_rules.py pins it against golden
    digests captured before this function existed (plan_update 5a65353e88b5acd9, moment_correction
    1ece619997c290b5 on a deterministic synthetic input), and pins that calling with no rule is the same
    call the loop used to make inline.

    The two rules the two-lever arm needs, and what each one is NOT:

      bspline16       the moment operator with a B-spline design and an eigenvalue-relative ridge. Same
                      operator, same target (p_ref), a basis that can carry the dihedral's edge mass.
      selfconsistent  a REPLACEMENT refit of the ensemble the field itself produced, through
                      plan_c_basis.fit on the Chebyshev K=8 design with the C2s pipeline's ridge. It
                      does not touch p_ref: an increment against a self-consistent target is
                      identically zero, which is why this is a fit and not a correction.
    """
    rule = (rule or "").strip().lower()
    if rule == "bspline16":
        A = PB.design_bspline(table["centre"], float(table["lo"]), float(table["hi"]),
                              int(RULE_BSPLINE_M))
        res = I.moment_correction(table, counts, n_tot, n_out, p_ref, K=int(RULE_BSPLINE_M),
                                  gain=GAIN_BY_COORD[coord], design=A, max_outside_frac=SUPPORT_GATE,
                                  ridge_rel=RULE_RIDGE_REL, ridge_form="eig")
        _norm = float(res.diagnostics.get("moment_norm", float("nan")))
        if _norm == _norm:
            _div, _msg = I.divergence_check(hist_norm[coord], _norm)
            if _div:
                res = I.UpdateResult(table=None, dU=np.zeros(len(table["U"])),
                                     diagnostics=dict(res.diagnostics, message=_msg),
                                     status=I.STATUS_REFUSED, reason=I.REFUSE_DIVERGENCE)
            hist_norm[coord].append(_norm)
        return res, rule
    if rule == "selfconsistent":
        p_ens = I.probability_from_counts(np.asarray(counts, dtype=float), pseudo=I.DEFAULT_PSEUDO)
        target = -B.KBT * np.log(np.clip(p_ens, 1e-300, None))
        target = target - target.min()
        A, _x = I._chebyshev_design(table["centre"], float(table["lo"]), float(table["hi"]),
                                    int(CORRECTION_K))
        U_new, diag = PB.fit(A, table, counts, U_target=target, ridge_rel=RULE_C2S_RIDGE_REL,
                             ridge_form="trace", gain=GAIN_BY_COORD[coord],
                             support_frac=RULE_C2S_SUPPORT_FRAC,
                             taper_decades=RULE_C2S_TAPER_DECADES)
        U_new = np.asarray(U_new, dtype=float)
        dU = U_new - np.asarray(table["U"], dtype=float)
        diag = dict(diag, method="selfconsistent_refit", coord=coord,
                    max_abs_dU=float(np.abs(dU).max()))
        _norm2 = float(diag.get("mass_weighted_std_dU", float("nan")))
        if _norm2 == _norm2:
            _div2, _msg2 = I.divergence_check(hist_norm[coord], _norm2)
            if _div2:
                return I.UpdateResult(table=None, dU=np.zeros(len(table["U"])),
                                      diagnostics=dict(diag, message=_msg2),
                                      status=I.STATUS_REFUSED,
                                      reason=I.REFUSE_DIVERGENCE), rule
            hist_norm[coord].append(_norm2)
        return I.UpdateResult(table=dict(table, U=U_new), dU=dU, diagnostics=diag,
                              status=I.STATUS_OK), rule
    if OPERATOR == "moments":
        # Plan B' (docs/archive/plan_b_coupled_update.md): the table keeps its shape and the correction is
        # low-order -- d_k = eta_k (<T_k>_sim - <T_k>_ref), the relative entropy's gradient. The
        # divergence guard is the SAME rule plan_update applies, fed the moment norm instead of
        # max|dU|, so a coupled step that starts growing is caught by the same instrument rather than
        # by a second opinion.
        res = I.moment_correction(table, counts, n_tot, n_out, p_ref, K=CORRECTION_K,
                                  gain=GAIN_BY_COORD[coord], max_outside_frac=SUPPORT_GATE)
        _norm = float(res.diagnostics.get("moment_norm", float("nan")))
        if _norm == _norm:
            _div, _msg = I.divergence_check(hist_norm[coord], _norm)
            if _div:
                res = I.UpdateResult(table=None, dU=np.zeros(len(table["U"])),
                                     diagnostics=dict(res.diagnostics, message=_msg),
                                     status=I.STATUS_REFUSED, reason=I.REFUSE_DIVERGENCE)
            hist_norm[coord].append(_norm)
        return res, OPERATOR
    res = I.plan_update(table, hist, p_ref, history=hist_by_coord[coord],
                        gain=GAIN_BY_COORD[coord], max_outside_frac=SUPPORT_GATE,
                        smooth_bins=(B.SMOOTH_WIDTH - 1) // 2)
    # AN UNRECOGNISED RULE RUNS THE OPERATOR, AND REPORTS THE OPERATOR. Echoing the unknown string
    # would put a rule name in the round json that no code path implements -- measured by
    # test_unknown_rule_falls_back_to_the_default_path, which caught exactly that.
    return res, OPERATOR


def marginal_report(table, U_after, counts, p_ref, edge_frac=0.05):
    """One coordinate's marginals, in the currencies the loop has to be read in.

    WHY THIS IS IN THE ROUND JSON. Two of the loop's three controlled coordinates turned out not to be
    controllable by their own table, and neither side's numbers say so on their own:

      * the DIHEDRAL's sampled outer-5-percent mass sat at 0.318-0.361 for four campaign rounds while
        its table's own implied edge moved 0.170 -> 0.152 -- the table moved, the ensemble did not;
      * the ANGLE's table walked its implied sigma 0.3218 -> 0.1307 while the sampled sigma followed
        only 0.4159 -> 0.3493. The least-squares slope over nine rounds is 0.311 (R2 0.70): a table
        change reaches the ensemble at about a third of its size, and the pooled width cannot go below
        that line's intercept, 0.3253, which is 1 percent ABOVE the 0.3218 target. The angle's
        remaining discrepancy is the model's floor, not the table's.

    So every round now records, per coordinate: the sampled sigma and edge mass (what the pool did), the
    target's own sigma and edge mass (what the reference holds), the table's implied sigma and edge mass
    before and after the update (what the fit asked for), and the ratios between them. sigma_implied is
    exp(-U/kBT) of the table ALONE; it is not the distribution the chain samples, which is exactly why
    the two are quoted side by side rather than one standing in for the other.

    Verified against the record: on arm A's round 0 this returns implied_sigma_after = 0.68491 (angle)
    and 0.51862 (dihedral), the two numbers ibi_armA_verdict.py reads back out of tables_r1.
    """
    def _sigma(p, centre):
        m = float((p * centre).sum())
        return float(np.sqrt(max((p * (centre - m) ** 2).sum(), 0.0)))

    def _edge(p):
        k = max(1, int(len(p) * edge_frac))
        return float(p[:k].sum() + p[-k:].sum())

    def _implied(U):
        U = np.asarray(U, dtype=float)
        p = np.exp(-(U - U.min()) / B.KBT)
        return p / p.sum()

    centre = np.asarray(table["centre"], dtype=float)
    c = np.asarray(counts, dtype=float)
    tot = float(c.sum())
    p_sim = c / tot if tot > 0 else np.zeros_like(c)
    p_ref = np.asarray(p_ref, dtype=float)
    p_before = _implied(table["U"])
    s_sim, s_tgt, s_before = _sigma(p_sim, centre), _sigma(p_ref, centre), _sigma(p_before, centre)
    e_sim, e_tgt = _edge(p_sim), _edge(p_ref)
    out = {"sampled_sigma": s_sim, "sampled_edge": e_sim,
           "target_sigma": s_tgt, "target_edge": e_tgt,
           "implied_sigma_before": s_before, "implied_edge_before": _edge(p_before),
           "sim_over_implied_before": (s_sim / s_before) if s_before else float("nan"),
           "sim_over_target": (s_sim / s_tgt) if s_tgt else float("nan"),
           "edge_gap_sampled": e_sim - e_tgt,
           "implied_sigma_after": None, "implied_edge_after": None,
           "implied_sigma_ratio": None, "sim_over_implied_after": None,
           "edge_gap_table_after": None}
    if U_after is not None:
        p_after = _implied(U_after)
        s_after, e_after = _sigma(p_after, centre), _edge(p_after)
        out["implied_sigma_after"] = s_after
        out["implied_edge_after"] = e_after
        out["implied_sigma_ratio"] = (s_after / s_before) if s_before else float("nan")
        out["sim_over_implied_after"] = (s_sim / s_after) if s_after else float("nan")
        out["edge_gap_table_after"] = e_after - e_tgt
    return out


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
    _rules = {c: RULE_BY_COORD.get(c, "") for c in UPDATED if RULE_BY_COORD.get(c, "")}
    print(f"  friction {friction}/ps, 300 K, constraints ON, wall_k {WALL_K:g}, gain " + " ".join(f"{c}={GAIN_BY_COORD[c]:g}" for c in UPDATED)
          + f", support gate {SUPPORT_GATE:g}"
          + (f", relax {RELAX_STEPS} steps" if RELAX_STEPS else "")
          + (f", operator {OPERATOR}" + (f" (K={CORRECTION_K})" if OPERATOR == "moments" else ""))
        + (", per-coordinate rules " + " ".join(f"{c}={r}" for c, r in _rules.items())
           if _rules else "")
        # The base-level term belongs in the header for the same reason the frozen list does: a run whose
        # field is different from every previous run has to say so in its own output, not only in its
        # launcher.
        + (f", base-level stacking eps {BASE_STACK_EPS:g} kJ/mol ({BASE_STACK_FORM}, "
           f"w {','.join(f'{x:g}' for x in BASE_STACK_W)})" if BASE_STACK_EPS > 0.0 else "")
        # The frozen list belongs in the header: a run with IBI_LOOP_FREEZE set is judged on the
        # coordinates it did NOT touch, and the log is the only place that says so. (The launcher
        # records it too; the header is what a reader of the output file sees.)
        + (", frozen " + ",".join(FROZEN) if FROZEN else ""))
    # Truncated: the all-chains pool is 867 names, which buries the rest of the header. The full
    # list is recoverable from the round json's per_structure entries.
    _ls = [len(s["pos"]) for s in structs]
    _names = ", ".join(f"{s['name']}(L={len(s['pos'])})" for s in structs[:6])
    print(f"  {len(structs)} structures, L {min(_ls)}-{max(_ls)}: {_names}"
          + (f", ... +{len(structs) - 6} more" if len(structs) > 6 else ""))
    print(f"  updated {UPDATED}; carried {CARRIED}; excluded {tuple(B.CONSTRAINED)} (rigid)")
    if os.environ.get("IBI_LOOP_KEEP_AWAKE", "") not in ("", "0"):
        print("  keep-awake: " + ("held (ES_SYSTEM_REQUIRED)" if keep_awake() else "NOT held"))
    print()

    ref_tables = I.load_clean_tables(REF_NPZ, coords=B.scored_coords())
    # The target is the reference file's, in a resume exactly as in a fresh run: what changed
    # during the rounds already done is the table that was sampled, not what it is aiming at.
    p_ref = {c: I.probability_from_table(ref_tables[c]) for c in B.scored_coords()}
    if start_round:
        resume_npz = OUT_ROOT / f"tables_r{start_round}.npz"
        if not resume_npz.exists():
            raise SystemExit(
                f"--start-round={start_round} wants the table that round {start_round} was to "
                f"sample, {resume_npz}, and it is not there. That file is written at the end of "
                f"round {start_round - 1}; without it there is nothing to resume from, and a "
                f"fresh run is the honest option.")
        tables = I.load_clean_tables(resume_npz, coords=B.scored_coords())
        check_bins_agree(tables, ref_tables, str(resume_npz))
        print(f"  resume: round {start_round} samples under {resume_npz.name}, "
              f"target {REF_NPZ.name}")
    else:
        tables = ref_tables
    hist_by_coord = history_from_rounds(OUT_ROOT, start_round)
    # The moment operator's divergence guard needs its own history: it is fed |d<T>|max rather than
    # max|dU|, and mixing the two units in one list would compare nothing to nothing.
    hist_norm = {c: [] for c in B.scored_coords()}

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
        done_dir = OUT_ROOT / f"tasks_r{rnd}"
        done_dir.mkdir(parents=True, exist_ok=True)
        tasks, task_owner = [], []
        for i, s in enumerate(structs):
            base, extra = divmod(nrep, per_chain_chunks)
            for k in range(per_chain_chunks):
                rep = base + (1 if k < extra else 0)
                if rep <= 0:
                    continue
                idx = len(tasks)
                tasks.append((idx, str(done_dir / f"{idx}.hb"), str(round_npz),
                              np.asarray(s["pos"], dtype=np.float64), list(s["pairs"]),
                              rep, nsteps, burn, stride, 8, friction, seed + 1000 * k + i,
                              _N_THREADS))
                task_owner.append(i)

        # RESUME INSIDE A ROUND: a task whose result is already on disk is not re-run, so a round
        # killed at hour 20 costs the chains that were in flight and nothing else.
        remaining = [t for t in tasks if not task_npz(done_dir, t[0]).exists()]
        remaining = longest_first(remaining, lambda i: len(structs[task_owner[i]]["pos"]))
        print(f"  {len(tasks)} tasks over {len(structs)} chains "
              f"({per_chain_chunks} replica-chunks each) on {_N_WORKERS} workers"
              + (f"; {len(tasks) - len(remaining)} already on disk, {len(remaining)} to run"
                 if len(remaining) != len(tasks) else ""))

        # A POOL WHOSE WORKER DIED HOLDS ENTRIES THAT WILL NEVER ARRIVE, so the wait is bounded by
        # the heartbeats rather than by patience: every outstanding task silent for QUIET_S means
        # the workers are gone, and the outstanding ones are re-run in a fresh pool. Re-running is
        # exact -- a task is a pure function of the table, the chain and the seed.
        attempt = 0
        while remaining and attempt < MAX_ATTEMPTS:
            attempt += 1
            if attempt > 1:
                print(f"  attempt {attempt}: {len(remaining)} task(s) to run")
            # Clear the heartbeats this attempt is about to write: left alone, the files from the
            # attempt that died are all older than STALE_S and the first check reads them as
            # hundreds of dead workers. attempt_started is the other half of that guard, for files
            # an unlink could not reach.
            attempt_started = time.time()
            for t in remaining:
                try:
                    Path(t[1]).unlink()
                except OSError:
                    pass
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(_N_WORKERS, len(remaining))) as pool_procs:
                it = pool_procs.imap_unordered(_sample_one, remaining)
                last_check = time.time()
                while remaining:
                    try:
                        idx, r = it.next(timeout=POLL_S)
                    except mp.TimeoutError:
                        now = time.time()
                        if now - last_check > 3 * POLL_S:
                            print(f"  note: {now - last_check:.0f} s passed without a check -- "
                                  f"the machine was suspended. Heartbeats refreshed; this is not a "
                                  f"crash.")
                            touch_heartbeats(remaining, now)
                        last_check = now
                        dead = dead_tasks(remaining, STALE_S, now, since=attempt_started)
                        if dead:
                            # NAME THE CHAINS, not just the task indices: the only way to tell a
                            # random machine-level death from a chain that kills its worker every
                            # time is to see whether the same names come back.
                            who = []
                            for t in dead[:10]:
                                ci = task_owner[t[0]] if t[0] < len(task_owner) else None
                                who.append(str(t[0]) if ci is None else
                                           f"{t[0]}({structs[ci]['name']} "
                                           f"L={len(structs[ci]['pos'])})")
                            print(f"  DEAD WORKER: {len(dead)} task(s) stopped beating "
                                  f"{STALE_S:.0f} s ago "
                                  f"({', '.join(who)}"
                                  + (f", +{len(dead) - 10} more" if len(dead) > 10 else "")
                                  + f") -- this is what multiprocessing.Pool does not tell "
                                  f"anybody: it respawns the worker and the task it held is never "
                                  f"re-issued. Restarting the {len(remaining)} outstanding "
                                  f"task(s); the ones already on disk are untouched.")
                            break
                        continue
                    save_task_result(done_dir, idx, r)
                    remaining = [t for t in remaining if t[0] != idx]
        if remaining:
            raise SystemExit(
                f"{len(remaining)} task(s) did not complete in {MAX_ATTEMPTS} attempts: "
                + ", ".join(str(t[0]) for t in remaining[:20])
                + (f" (+{len(remaining) - 20} more)" if len(remaining) > 20 else "")
                + ". Nothing is on disk for them, the round cannot finish, and a chain that "
                  "crashes its worker on every attempt is a finding -- look at the chain, not at "
                  "the retry count.")
        results = [load_task_result(done_dir, i) for i in range(len(tasks))]

        # Per-chain line: every chunk of a chain reported its own J, so the chain's line names the
        # spread across its chunks rather than pretending they are one number.
        for i, s in enumerate(structs):
            mine = [r for r, o in zip(results, task_owner) if o == i]
            if not mine:
                continue
            js = [r["joint_J"] for r in mine if r["joint_J"] is not None]
            jtxt = "nan" if not js else (f"{min(js):.4f}-{max(js):.4f}" if len(js) > 1
                                         else f"{js[0]:.4f}")
            # J4 is the OLD four-coordinate mean, printed beside J so a line can be read against the
            # campaign's records without a second lookup.
            j4 = [r.get("joint_J_all") for r in mine if r.get("joint_J_all") is not None]
            j4txt = "nan" if not j4 else (f"{min(j4):.4f}-{max(j4):.4f}" if len(j4) > 1
                                          else f"{j4[0]:.4f}")
            n = sum(r["n_total"]["bb_bond"] for r in mine)
            print(f"    {s['name']:9s} L={len(s['pos']):4d}  J {jtxt}  J4 {j4txt}  n={n}  "
                  f"({len(mine)} chunks, max {max(r['seconds'] for r in mine):.0f} s)")

        updates = {}
        for c in UPDATED:
            counts = sum((r["counts"][c] for r in results), np.zeros(len(tables[c]["U"]), np.int64))
            n_out = sum(r["n_outside"][c] for r in results)
            n_tot = sum(r["n_total"][c] for r in results)
            if c in FROZEN:
                # FROZEN (IBI_LOOP_FREEZE): the table is carried unchanged, the potential is STILL
                # built from it so the sampler keeps running under the frozen field, and the coordinate
                # keeps being measured and reported -- the record has to show what the untouched
                # reading was. Measured 2026-10-01 on the full pool: the angle's SAMPLED marginal is
                # already close to the reference (sigma 9 per cent off, edge mass within 0.01) while
                # every update pushes the TABLE away from the sampler -- the table operator plateaus
                # after about 20 per cent, and the moment operator drove implied sigma 0.3218 -> 0.1307
                # (-59 per cent) against a sampler that followed only -16 per cent. So "do not touch
                # it" is a control arm with evidence behind it, and no guard bookkeeping applies: a
                # coordinate that never updates has no correction history to diverge.
                _ci = list(B.scored_coords()).index(c)
                _rt = [r["sim_ref_table"][_ci] for r in results
                       if r.get("sim_ref_table") and r["sim_ref_table"][_ci] is not None]
                _meas = float(np.mean(_rt)) if _rt else float("nan")
                updates[c] = {"status": "frozen", "rule": "frozen", "reason": "IBI_LOOP_FREEZE",
                              "max_abs_dU": 0.0, "n_samples": int(n_tot), "n_outside": int(n_out),
                              "measured_sim_ref_table": None if _meas != _meas else _meas,
                              # A frozen coordinate is still MEASURED: the acceptance criteria are on
                              # the sampled marginal now, and a table that is carried unchanged still
                              # has to be read against what the pool did under it.
                              "marginals": marginal_report(tables[c], None, counts, p_ref[c])}
                print(f"  update {c:9s} frozen   table carried; measured sim/ref_table "
                      f"{_meas:.4f}  n={n_tot}" if _meas == _meas
                      else f"  update {c:9s} frozen   table carried; no measured ratio")
                continue
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
            # THE DISPATCH IS PER COORDINATE NOW (2026-10-01, the two-lever arm). With no rule set this
            # is one call into update_one_coord, whose default branch is the same moment/plan_update
            # dispatch that produced every table in results/ibi_relax; the golden digests in
            # tests/test_ibi_driver_rules.py pin that the default path did not move.
            res, rule_used = update_one_coord(c, tables[c], counts, n_tot, n_out, p_ref[c], hist,
                                              hist_norm, hist_by_coord,
                                              rule=RULE_BY_COORD.get(c, ""))
            entry = {"ok": bool(res.ok), "converged": bool(res.converged),
                     "max_abs_dU": float(res.max_abs_dU), "n_samples": int(n_tot),
                     "n_outside": int(n_out), "rule": str(rule_used)}
            hist_by_coord[c].append(float(res.max_abs_dU))
            try:
                new_table = res.require_table()
                entry["status"] = "applied"
            except I.IBIRefusal as exc:
                new_table = None
                entry["status"] = "refused"
                entry["reason"] = str(exc)
            entry["marginals"] = marginal_report(
                tables[c], (new_table["U"] if new_table is not None else None), counts, p_ref[c])
            updates[c] = entry
            print(f"  update {c:9s} {entry['status']:8s} max|dU|={entry['max_abs_dU']:.4f} "
                  f"kBT={entry['max_abs_dU'] / B.KBT:.4f}  n={entry['n_samples']} "
                  f"outside={entry['n_outside']}"
                  + ("" if entry["status"] == "applied" else f"\n      {entry.get('reason','')}"))
            if "moment_norm" in res.diagnostics:
                # Any moment-based rule prints this: the production moments operator, and the
                # B-spline variant the two-lever arm runs on the dihedral.
                print(f"      moments  |d<T>|max={res.diagnostics['moment_norm']:.4f}  "
                      f"estimated dS = -{res.diagnostics['rel_entropy_drop_kbt']:.4f} kBT  "
                      f"K={res.diagnostics.get('K', CORRECTION_K)}  rule={rule_used}")
            elif "mass_weighted_std_dU" in res.diagnostics:
                print(f"      refit    mass-weighted std={res.diagnostics['mass_weighted_std_dU']:.4f} "
                      f"kJ/mol  ridge={res.diagnostics.get('ridge_rel')}  rule={rule_used}")
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
