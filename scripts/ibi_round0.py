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

The field is used exactly as shipped, including its force cap (force_cap's default, read from the
signature below rather than written here), because that is what the pipeline runs.

Run: python scripts/ibi_round0.py [n_rep] [n_steps] [struct_idx] [friction] [stride] [burn]

The BURN argument is separate from NSTEPS on purpose. It used to be NSTEPS // 5, so asking for a
longer run moved the sampling window as well as lengthening it, and a residual that fell could
not be told apart from a window that had merely slid past a transient. Pass it explicitly to keep
the window fixed while the run grows. 0 or omitted means NSTEPS // 5, the old behaviour.

The progress line reports each coordinate's sim/ref and the joint mean |ln(sim/ref)| over the
window SO FAR, so one long run shows whether the estimate has stopped moving. Before this it
printed sim/1D, which is the coupling correction and not the acceptance number.

BUT a cumulative number cannot establish stationarity on its own: it is one average over
everything seen so far, so a good early stretch hides a bad later one. The final report splits
the sampling window into --blocks=N disjoint equal-TIME blocks and gives each its own J. Read
the spread of those, not the movement of the cumulative line.
"""
import inspect
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


def _opt_int(name, default):
    """--name=N out of argv, leaving the positional arguments untouched."""
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            return int(a[len(pre):])
    return default


def _opt(name, default=""):
    """--name=VALUE out of argv as a string, leaving the positional arguments untouched."""
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            return a[len(pre):]
    return default


# ── optional substitutes for the two backbone angular terms ──
# With neither flag this script is bit-identical to what it was before they existed: both
# potentials stay None and cg_energy_forces takes its own default branch. That is checked two
# ways -- tests/test_table_potential_injection.py for the plumbing, and the whole-run regression
# against results/newfield_8x8000.log for the trajectory.
#
# With one passed, the sampled distribution is a DIFFERENT one, and nothing in the constant
# fingerprint below can show it: _FINGERPRINT names constants, while a potential replaces the
# SHAPE of a term and leaves every constant alone. That is the same hole this script's docstring
# records as having invalidated three earlier runs, so the resolved spec is echoed into the log
# rather than left to be inferred from the numbers it produces.
import cg_potentials as P          # noqa: E402
import ibi_core as IC              # noqa: E402

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
SEED = 20260218

# --table=FILE is the table this round SIMULATES: round 0 uses the shipped/reference tables, and
# round N passes what ibi_update.py wrote for round N-1. It is NOT the reference the update is
# checked against -- that one is fixed, and changing it would make every update a no-op by
# construction.
#
# Resolved BEFORE the potentials are built, and pushed into them. The table specs in
# force_reference read a file once and cache the result, so building a potential first would
# freeze round 0's table inside it: the round would then SAMPLE under the old potential while
# BINNING against the new table, and report a converged update for the wrong reason.
_TABLE_PATH = Path(_opt("table", str(NPZ)))
P.use_table_file(_TABLE_PATH)
TAB = IC.load_tables(_TABLE_PATH)

_POTS = []          # [(coord, spec, potential)]
for _coord in ("angle", "dihedral"):
    _text = _opt(_coord, "")
    if _text:
        _spec = P.resolve_spec(_text, _coord)
        _POTS.append((_coord, _spec, P.make_potential(_coord, _spec)))
_POT_KW = {f"{_c}_potential": _pot for _c, _s, _pot in _POTS}

# --cap=auto|none|NUMBER. "auto" reads force_cap's own default off the signature, which is the
# shipped behaviour and the default here. force_cap rescales the SUMMED force vector, so a term
# whose honest force exceeds it stops being -dE/dx wherever the cap fires -- which is a property
# of the cap, not of the term, and is why "none" exists.
_cap_text = _opt("cap", "auto")
if _cap_text == "auto":
    CAP = inspect.signature(C.cg_energy_forces).parameters["force_cap"].default
elif _cap_text == "none":
    CAP = None
else:
    CAP = float(_cap_text)

# --write=DIR dumps one npz per coordinate in the form ibi_bonded.plan_update consumes, so a
# round can be updated without a human reading the numbers back out of the log. Without it this
# script prints and writes nothing, exactly as before.
_WRITE = _opt("write", "")

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
# The list has to name EVERY constant whose value changes the sampled distribution. The four
# backbone terms were added after this script was written, and without them here a run under the
# old field and a run under the new one print the same fingerprint -- which is the provenance hole
# the paragraph above says invalidated three earlier runs.
_FINGERPRINT = ("K_BB", "K_INTRA_PC", "K_INTRA_CN", "K_INTRA_PN",
                "K_LINK_CP", "K_LINK_NP", "K_LINK_NC",
                "K_PAIR", "K_ANGLE", "K_DIH", "K_BPP", "K_STACK",
                "K_CLASH", "CLASH_SIGMA", "K_BSJ", "K_BSJ_GUIDE")
_missing = [n for n in _FINGERPRINT if not hasattr(C, n)]
if _missing:
    raise RuntimeError(f"the fingerprint names {_missing}, which this module does not define; a "
                       f"renamed constant would silently drop out of the provenance record")
print("field: " + "  ".join(f"{n}={getattr(C, n)}" for n in _FINGERPRINT))
import inspect as _inspect
_cap = _inspect.signature(C.cg_energy_forces).parameters["force_cap"].default
print(f"force_cap={_cap}  mass=110.0 Da  dt=0.002 ps  friction={FRICTION}/ps")
# The SHAPE of the guide is not a constant, so the fingerprint above cannot see it: a run under
# the short-range-reward form and a run under the long-range form print the SAME fingerprint.
# That is the same provenance hole this script's own paragraph says invalidated three earlier
# runs, so probe the shape directly instead of trusting a list of numbers.
#
# The discriminator is the SIGN of E, and it has to be. The first version of this probe tested
# `E(0.5) < E(3.0)` on the reasoning that the long-range form rises with distance. It does --
# but so does the short-range reward, because -softplus((r0-d)/w) also rises with d, from -k to
# 0. So that test was true for BOTH shapes and the probe could never print its warning. Measured
# with scripts/e3_old_field.py, which patches _sigmoid_f back to the pre-6e7a44b form: the old
# shape gives E(0.5) = -2.579, E(3.0) = -0.000, and the old probe called that "long-range".
# What actually separates them is the overall sign, which is k-independent:
#     long-range form:  E = +k*softplus((dist-r0)/width)  ->  E > 0 everywhere
#     short-range form: E = -k*softplus((r0-dist)/width)  ->  E < 0 everywhere
_gs = C._sigmoid_f(torch.tensor([0.5, 3.0]), 1.0, 1.0, 0.2)[0]
_gnear, _gfar = float(_gs[0]), float(_gs[1])
print(f"guide shape: E(0.5 nm)={_gnear:.3f}  E(3.0 nm)={_gfar:.3f}  ("
      + ("long-range: zero below r0, bounded pull above" if _gnear > 0
         else "SHORT-RANGE REWARD -- the pre-6e7a44b form") + ")")
# The second B half-kick takes force_fn, so this run is symplectic. It has to be: the
# non-symplectic fallback pumps energy at dt*omega^2/(4*gamma) of the drag per step, and after
# K_INTRA was split into its two measured values the stiffest coordinate is C4'-N at
# 36399.2 kJ/mol/nm^2 over a 55 amu reduced mass, omega = sqrt(36399.2/55) = 25.73 /ps. At
# gamma = 0.1 and dt = 0.002 that ratio is dt*omega^2/(4*gamma) = 3.31 -- the pump is more than
# three times the drag, so the fallback would heat the intra-bead bonds rather than merely
# perturb them. The extra force evaluation doubles the cost and is worth it here.
print(f"second B half-kick: symplectic (force_fn recomputes at the post-update coordinates). "
      f"Without it the stiffest coordinate would pump at "
      f"{0.002 * (36399.2 / 55.0) / (4 * FRICTION):.2f}x the drag per step.")
print()

pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)

# The potentials actually in force, and their honest force on the STARTING geometry. Printed
# here rather than beside the constant fingerprint because it needs pos, and the point is to see
# BEFORE the run whether force_cap will clip what is being injected: the cap rescales the summed
# force, so a clipped term is not -dE/dx wherever it fires. docs/dihedral_table_decision.md
# records the numbers this is guarding -- the table's honest max is 15736 against a cap of 5000.
if _POTS:
    print("potentials: " + "  ".join(f"{_c}={P.describe(_s)}" for _c, _s, _ in _POTS)
          + f"  cap={'none' if CAP is None else f'{CAP:g}'}")
    print("            NOTE: the constant fingerprint above does NOT reflect this -- it names "
          "constants,")
    print("            and a potential replaces the SHAPE of a term while leaving every "
          "constant alone.")
    with torch.no_grad():
        for _c, _s, _pot in _POTS:
            _e, _f = _pot(pos)
            _fm = float(_f.abs().max())
            if CAP is None:
                _verdict = "no cap in force"
            elif _fm <= CAP:
                _verdict = f"under cap {CAP:g}"
            else:
                _verdict = (f"OVER cap {CAP:g} by {_fm / CAP:.2f}x -- the cap will clip it, and "
                            f"the clipped term is not -dE/dx; --cap=none shows the true one")
            print(f"  injected {_c:8s} |F|max {_fm:9.1f} on the start geometry   ({_verdict})")
    print()
pw = torch.ones(len(ij), dtype=torch.float32)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)

# The noise sequence is seeded inside run_round, which is called after pos is built -- the seed
# and the starting coordinates have to stay paired exactly as they were, or the trajectory moves.
# The sampling window, decoupled from the run length. See the docstring.
_burn_arg = int(sys.argv[6]) if len(sys.argv) > 6 else 0
burn = _burn_arg if _burn_arg > 0 else max(NSTEPS // 5, 1)
print(f"burn = {burn} steps = {burn * 0.002:.1f} ps; sampling window "
      f"{burn * 0.002:.1f}-{NSTEPS * 0.002:.1f} ps")
print("  progress: each column is sim/ref; J is the joint mean |ln(sim/ref)| over the window so far")
# hoisted so the progress line can print the coupling ratio while the run is going
K_SHIPPED = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA_PC, "intra_cn": C.K_INTRA_CN,
             "angle": C.K_ANGLE, "dihedral": C.K_DIH, "stack": C.K_STACK}
# Stationarity needs DISJOINT blocks, not a cumulative average. A cumulative J over [burn, t]
# cannot separate "the window is settling" from "the early part of the window happened to look
# good": it is one number computed over everything seen so far. That is exactly how the
# 40-200 ps run read 0.0911 and then climbed to 0.0968 -- the second half was worse and the
# cumulative average hid it behind the first. So the accumulators are per block and every
# whole-window number below is the sum over blocks.
NB = max(_opt_int("blocks", 4), 1)
# The loop itself lives in ibi_core. It is not duplicated here: docs/statistical_potentials_as_
# forces.md:2754 records that this repository has already drifted three times from duplicated
# logic and requires the sampler loop not to exist in two copies. Everything the report below
# needs is on the returned object.
_res = IC.run_round(pos=pos, vel=vel, ij=ij, pw=pw, temps=temps, tab=TAB,
                    nsteps=NSTEPS, burn=burn, stride=STRIDE, blocks=NB,
                    friction=FRICTION, force_cap=CAP, pot_kw=_POT_KW, seed=SEED, nrep=NREP)
# Bound to the names the report below already uses, so the report is untouched by the extraction.
# That is the whole discipline of this change: it may MOVE code, it may not restate it -- and the
# gate is that this script reproduces its own historical stdout bit for bit.
counts, acc = _res.counts, _res.acc
b_acc, b_frames = _res.b_acc, _res.b_frames
clash_min = _res.clash_min
clash_below, clash_below_live = _res.clash_below, _res.clash_below_live


print(f"done in {_res.seconds:.0f} s, {_res.steps_per_s:.1f} steps/s")
if _WRITE:
    _bJ = []
    for _b in range(NB):
        _bv, _bj = IC.simref(_res.b_acc[_b], TAB)
        _bJ.append(None if _bj != _bj else round(float(_bj), 6))
    _wv, _wj = IC.simref(_res.acc, TAB)
    IC.write_round(_WRITE, _res, TAB, meta={
        "cmdline": " ".join(sys.argv),
        "table_file": str(_TABLE_PATH),
        "structure": s0["name"], "L": L,
        "nrep": NREP, "nsteps": NSTEPS, "burn": burn, "stride": STRIDE, "blocks": NB,
        "friction": FRICTION, "seed": SEED, "force_cap": CAP, "dt_ps": 0.002, "mass_amu": 110.0,
        "potentials": {_c: P.describe(_s) for _c, _s, _ in _POTS},
        "fingerprint": {n: getattr(C, n) for n in _FINGERPRINT},
        "joint_J": None if _wj != _wj else round(float(_wj), 6),
        "block_J": _bJ,
        "per_coordinate": {c: {"n": int(_res.n_total[c]), "n_outside": int(_res.n_outside[c]),
                               "sim_ref": (None if v != v else round(float(v), 6))}
                           for c, v in zip(B.COORDS, _wv)},
    })
    print(f"wrote per-coordinate histograms to {_WRITE}/")
print()
print(f"clash watch: live range {C.CLASH_SIGMA:.4f} nm, retired cutoff {C.CLASH_DIST:.3f} nm; "
      f"closest bead pair ever {min(clash_min):.4f} nm")
print(f"  pair instances below the live range : {clash_below_live}")
print(f"  pair instances below the old 0.300 : {clash_below}  "
      f"(the number an earlier version of this script reported as if it were the live one)")
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
# counts/acc are bound from the run_round result above; the report does not recompute them.
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
# ── stationarity: disjoint blocks of the same sampling window, each with its own J ──
# This is the instrument the cumulative line cannot replace. If the blocks agree, the window is
# a sample of one distribution and a cumulative J over it means something. If they do not, the
# run has not reached the stationary distribution and no residual taken over it is an
# equilibrium number -- which is what the 40-200 ps run showed when its cumulative J dipped to
# 0.0911 and then climbed back to 0.0968.
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
    print("  Rule of thumb: within a few percent the blocks are consistent and the window is a"
          " sample of one distribution. Tens of percent means the run is still drifting and any"
          " residual over it is a mixture, not an equilibrium value. The per-coordinate columns"
          " say WHICH coordinate is moving.")
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
