"""The IBI sampling loop: ONE copy, shared by every run that needs it.

docs/statistical_potentials_as_forces.md:2754 records that this repository "has already drifted
three times from duplicated logic" and states the requirement directly: 采样环不能有两份 -- the
sampler loop must not exist in two copies. Now that cg_energy_forces can take the angle and
dihedral terms as injected potentials, the same loop has to serve two kinds of run:

    the shipped field      ibi_round0.py, the stationarity gate
    an injected field      a round driver, with tabulated or fitted angular terms

Writing the second loop next to the first is exactly the drift that note forbids, so the loop
lives here and both scripts are callers. The extraction is gated on ibi_round0.py reproducing
its own historical stdout bit for bit (modulo the timing fields), because that script produced
numbers this repository's conclusions rest on.

What is deliberately NOT here:

  * argument parsing, the constant fingerprint, the guide-shape probe -- per script
  * potential construction and the force-cap policy  -- per script
  * the terminal report's tables and their wording   -- per script

Those are what still distinguish the callers, and moving them in would turn a shared loop into a
framework with a configuration flag for every difference.

The loop is what it always was: N independent replicas at one temperature, symplectic BAOAB with
the force recomputed at the post-update coordinates, sampling every STRIDE steps after BURN into
--blocks disjoint equal-TIME blocks. Stationarity is read off the block spread, never off the
cumulative line -- see run_round's note.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B                    # noqa: E402
import torusfold.scheme2.torch_cgsim as C       # noqa: E402

DT_PS = 0.002
MASS_AMU = 110.0


def load_tables(npz_path, skip=()):
    """The stored tables in boltzmann_bonded's format, minus any coordinate in `skip`.

    Note this does NOT call B.prepare: the loop bins against centre/U/binw/sigma and never
    evaluates a table as a potential, so the interpolation cache would be dead weight here. A
    caller that injects a table potential builds it through cg_potentials, which goes to
    force_reference for the interpolation instead.

    skip exists so a constrained round tolerates a table file that does not carry the
    constrained coordinates at all. Requiring them would force every writer to keep two dead
    arrays alive for coordinates nothing samples; requiring them only where they are used is
    the same check, applied to the set that actually reaches the binning loop.
    """
    z = np.load(npz_path)
    keep = [n for n in B.COORDS if n not in set(skip)]
    need = [f"{n}__{k}" for n in keep
            for k in ("centre", "U", "binw", "sigma", "lo", "hi")]
    missing = [k for k in need if k not in z.files]
    if missing:
        raise KeyError(
            f"{npz_path} is missing {len(missing)} key(s), e.g. {missing[0]!r}. A table written "
            f"by ibi_update.py carries plan_update's TABLE_KEYS, which omit sigma on purpose -- "
            f"sigma belongs to the reference, and the driver is expected to carry it across. "
            f"Seeing this means the driver did not.")
    return {name: {"centre": z[f"{name}__centre"], "U": z[f"{name}__U"],
                   "binw": float(z[f"{name}__binw"]), "sigma": float(z[f"{name}__sigma"]),
                   "lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"])}
            for name in keep}


def summed(b_counts, b_acc, nb):
    """Whole-window counts and moments, as the sum over the blocks."""
    ct = {c: np.zeros_like(b_counts[0][c]) for c in B.COORDS}
    ac = {c: [0.0, 0.0, 0] for c in B.COORDS}
    for b in range(nb):
        for c in B.COORDS:
            ct[c] += b_counts[b][c]
            a = b_acc[b][c]
            ac[c][0] += a[0]
            ac[c][1] += a[1]
            ac[c][2] += a[2]
    return ct, ac


def sim_ref_ratio(acc, coord, tab, samples=None):
    """sigma_sim / sigma_ref for one coordinate, or nan if it cannot be scored.

    The single place that decides what "scorable" means, so simref and j_denominator can never
    disagree about the denominator -- a divergence between those two would be invisible, since
    both would still return a number.

    nan means one of three things, and they are NOT equivalent: no samples at all; a reference
    sigma of zero; or a sigma_sim of zero. The last one is the interesting case and it is why
    this returns a ratio rather than a verdict -- a coordinate held exactly rigid has
    sigma_sim == 0.0 to the last bit, and that is a property of the MODEL, not of the round.
    """
    s1, s2, n = acc[coord]
    if samples is not None:
        n = samples
    if not n:
        return float("nan")
    mm = s1 / n
    ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
    sig = tab[coord]["sigma"]
    return ss / sig if sig > 0 else float("nan")


def simref(acc, tab, skip=()):
    """(per-coordinate sim/ref values, joint J) from a moments accumulator.

    J is the mean over the coordinates of |ln(sim/ref)|. Returned in B.COORDS order, with nan for
    anything skipped or unscorable.

    THE DENOMINATOR IS NOT len(B.COORDS), AND THAT IS THE POINT. Three things shrink it: a
    coordinate named in `skip`; a coordinate with no samples; and -- the one that bites -- a
    coordinate whose sigma_sim is exactly zero, which is counted out by the `r > 0` guard rather
    than scored. A rigid constraint produces exactly that: measured on a six-residue round, a
    constrained P-C4' has sigma_sim == 0.0 over 48 observations, because SHAKE puts it in the
    same place every frame.

    So a constrained round scored WITHOUT an explicit skip reports a J averaged over four
    coordinates while the run's own configuration says six, and nothing in the output says so.
    The J is not inflated -- it is silently computed over fewer things, which reads as an
    improvement. That is a worse failure than a bad number, because a bad number gets
    investigated. run_round therefore prints the denominator next to J and reports it as
    res.j_coords; see j_denominator.
    """
    skip = set(skip)
    vals = [float("nan") if c in skip else sim_ref_ratio(acc, c, tab) for c in B.COORDS]
    js = [abs(float(np.log(r))) for r in vals if r == r and r > 0]
    return vals, (float(np.mean(js)) if js else float("nan"))


def j_denominator(acc, tab, skip=()):
    """(how many coordinates J actually averaged over, how many were offered). Prints "4/6".

    A joint J is a mean, and a mean over four things is not a mean over six. Without this the one
    number a reader compares between rounds silently changes meaning when a coordinate drops out
    -- and dropping out is exactly what a constrained coordinate does, with no other symptom.

    It counts with the SAME predicate simref scores with (sim_ref_ratio + r > 0), so "4/6" always
    describes the J that was printed next to it. Two independent counts of the same thing is how
    a denominator comes to disagree with its own numerator.
    """
    skip = set(skip)
    offered = [c for c in B.COORDS if c not in skip]
    scored = [c for c in offered
              if (lambda r: r == r and r > 0)(sim_ref_ratio(acc, c, tab))]
    return len(scored), len(offered)


def closest_bead_pair(pos, chunk=512):
    """(closest bead-bead distance, pairs below CLASH_DIST, pairs below CLASH_SIGMA), chunked.

    WHY THIS FUNCTION EXISTS, measured. The obvious implementation -- torch.cdist(beads, beads),
    then a copy that adds the diagonal -- materialises an N x N float64 matrix PER SAMPLED FRAME.
    A 2929-residue chain is 8787 beads, so that is 617 MB, twice, 2500 times per chain. On the
    round of 2026-09-16 a single worker was holding 25 GB of commit because of it, the machine hit
    its commit limit (llama-server.exe held 68.5 GB of the same limit, and Windows popped "virtual
    memory insufficient"), three workers died with access violations inside VCRUNTIME140 at
    23:04:12, multiprocessing.Pool replaced them without a word, and the round hung at 98 percent
    for 29.7 h. Killed the run: commit free went 1.2 GB -> 29.7 GB.

    The work is the same O(N^2); what changes is that nothing bigger than chunk x N is ever alive.
    A chunk of 512 over 8787 beads is 36 MB against 1234 MB, and the min and the two counts are
    accumulated exactly as before. Replicas are still independent systems -- the loop runs per
    replica, so a batch does not acquire inter-replica contacts it never had.

    One deliberate difference: a coordinate with no distinct pair at all returns inf, where the
    N x N version returned the 10.0 it had just written onto its own diagonal. That 10.0 was the
    artefact of the hack that excluded self-pairs, not a measurement, and a one-bead chain is the
    only way to see it.
    """
    n_rep = pos.shape[0]
    best, below, below_live = float("inf"), 0, 0
    for r in range(n_rep):
        b = pos[r]
        n = b.shape[0]
        for i in range(0, n, chunk):
            d = torch.cdist(b[i:i + chunk], b)                  # (chunk, n), never (n, n)
            rows = torch.arange(i, min(i + chunk, n), device=d.device)
            d[torch.arange(d.shape[0], device=d.device), rows] = float("inf")
            best = min(best, float(d.min()))
            below += int((d < C.CLASH_DIST).sum())
            below_live += int((d < C.CLASH_SIGMA).sum())
    return best, below, below_live


def relax_positions(pos, ij, pw, pot_kw=None, constraints=None, n_steps=1500, step_nm=0.005,
                    force_cap=5000.0, tol=50.0, cell_size=1.5, report=250, log=None):
    """Descend the injected Hamiltonian before the sampler's first step. Returns (pos, info).

    WHY THIS EXISTS. run_round goes straight to 300 K Langevin from whatever coordinates it is
    handed, and on the 867-chain pool that is fine for 855 chains and catastrophic for twelve.
    Measured with the round-0 table injected, 65 ps, burn 0 (scripts/diagnose_chain_meltdown.py):

        chain       E at deposit   max|F|        closest approach   bb_bond outside at 65 ps
        9JHD_3         186919      5000 (cap)        0.026 nm              100.0%
        8D8K_31        233206      5000 (cap)        0.021 nm              100.0%
        7QVP_7         207501      5000 (cap)        0.023 nm               95.8%
        controls     7623-46623   4021-5000         0.152-0.249 nm          0.0%

    The twelve are not a bond that is a little too wide; they interpenetrate to 0.02 nm inside the
    clash wall, where the cap turns the restoring force into a constant, and they melt in 5-15 ps.
    Their joint residual is 1.31-2.05 against a pool median of 0.17, and they carried 99.48 percent
    of the bb_bond excursion that refused round 0's update (docs/ibi_loop_and_oxrna_findings.md,
    Part 4). That is a property of the entry point, not of the potential.

    WHAT IT DOES. Steepest descent on the SAME field and the SAME cap the sampler will integrate --
    not on the uncapped potential check_minimizer_method.py reports on -- with the direction
    NORMALISED, because a capped force's magnitude carries no information while its direction
    still points downhill. The step is accepted only when the energy falls and halved when it does
    not, which is the line search diagnose_wall_penetration.py already uses. SHAKE is applied
    after every accepted step, so the constrained distances stay on their manifold; unlike
    torch_gpu_refine's Adam there is no momentum to carry a step through the projection.

    Deterministic: no RNG is touched, so it does not shift the dynamics a seed produces. info
    carries E and max|F| at both ends, the accept/reject counts, the number of field evaluations,
    and the cap flags -- a caller that would rather DROP a chain than relax it needs the start
    numbers, and a caller that keeps it needs to see max|F| leave the cap.

    ONE TRAP, stated because it is easy to read as a bug: the start is projected onto the
    constraint manifold BEFORE it is measured, so info["energy_start"] is not a field evaluation
    of the array the caller passed. On a strained fixture that is not a small difference --
    121167.8 kJ/mol in, 37804.2 out, because the clash that carried the energy is partly the
    constrained distances being off their targets. Compare end to start WITHIN info, not against
    your own evaluation of the raw input.
    """
    pot_kw = pot_kw or {}
    x = torch.as_tensor(pos, dtype=torch.float64).clone()
    if constraints is not None:
        x = constraints.shake(x)

    def ef(p):
        cl = C.GPUCellList(cell_size=cell_size)
        cl.build(p)
        _e, _f = C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=force_cap, **pot_kw)
        return (float(_e.sum()), _f,
                float(torch.linalg.norm(_f.reshape(-1, 3), dim=-1).max()))

    info = {"steps": int(n_steps), "accepted": 0, "rejected": 0, "evals": 0,
            "energy_start": float("nan"), "energy_end": float("nan"),
            "max_force_start": float("nan"), "max_force_end": float("nan"),
            "hit_cap": False, "left_cap": False}

    with torch.no_grad():
        e, f, fmax = ef(x)
        info["evals"] = 1
        info.update(energy_start=e, max_force_start=fmax,
                    hit_cap=bool(force_cap is not None and fmax >= force_cap))
        s = float(step_nm)
        for _ in range(int(n_steps)):
            if not (fmax > tol):
                break
            trial = x + (s / fmax) * f
            if constraints is not None:
                trial = constraints.shake(trial)
            e_t, f_t, fmax_t = ef(trial)
            info["evals"] += 1
            if e_t < e:
                x, e, f, fmax = trial, e_t, f_t, fmax_t
                info["accepted"] += 1
                s = min(s * 1.2, float(step_nm))
                if report and log and info["accepted"] % report == 0:
                    log(f"  relax {info['accepted']:6d} accepted  E {e:12.1f} kJ/mol  "
                        f"max|F| {fmax:9.2f}  step {s:.2e} nm")
            else:
                info["rejected"] += 1
                s *= 0.5
                if s < 1e-9:
                    break
    info["energy_end"], info["max_force_end"] = e, fmax
    info["left_cap"] = bool(force_cap is not None and fmax < force_cap)
    return x.detach(), info


class RoundResult:
    """What run_round produces. Reporting lives in the caller, so this is plain state."""

    def __init__(self):
        self.b_counts = None      # [block][coord] -> int64 histogram
        self.b_acc = None         # [block][coord] -> [sum, sum of squares, count]
        self.b_frames = None      # [block] -> frames
        self.counts = None        # whole window, sum over blocks
        self.acc = None
        self.clash_min = []       # closest bead pair, per sampled frame
        self.clash_below = 0      # pair instances below the retired 0.300 cutoff
        self.clash_below_live = 0 # pair instances below CLASH_SIGMA
        # plan_update's SimHistogram wants n (every observation offered, in support or not) and
        # n_outside (how many fell outside [lo, hi]) next to the in-support counts. Counted here
        # rather than reconstructed by the caller, because only the binning loop knows them.
        self.n_total = {c: 0 for c in B.COORDS}
        self.n_outside = {c: 0 for c in B.COORDS}
        self.skip = ()            # the coordinates excluded from J, by name
        self.j_coords = (0, 0)    # (averaged over, offered), as j_denominator returns
        self.values = None        # {coord: (frames, M)} raw q, only when collect_values
        self.positions = None     # [(B, 3L, 3)] sampled frames, only when collect_positions
        self.pos = None           # final coordinates
        self.vel = None
        self.seconds = 0.0
        self.steps_per_s = 0.0
        self.relax = None         # relax_positions' diagnostics, only when relax > 0


def run_round(*, pos, vel, ij, pw, temps, tab, nsteps, burn, stride, blocks, friction,
              force_cap, pot_kw=None, seed=None, collect_values=False, nrep=None,
              collect_positions=False,
              progress=True, log=print, constraints=None, skip=None, relax=0):
    """Sample, binning into `blocks` disjoint equal-time blocks. Returns a RoundResult.

    The accumulators are per block on purpose. A cumulative J over [burn, t] cannot separate
    "the window is settling" from "the early part of the window happened to look good": it is
    one number over everything seen so far. That is how the 40-200 ps run read 0.0911 and then
    climbed to 0.0968 -- the second half was worse and the average hid it behind the first.

    force_cap is passed through to cg_energy_forces unchanged, including None. pot_kw is the
    injection dict ({"angle_potential": .., "dihedral_potential": .., "bond_potential": ..}),
    empty for the shipped field. Both are the caller's decision and neither is defaulted here.

    constraints: the distance set the integrator holds rigid. None (the default) builds the
    standard intra-residue P-C4' / C4'-N set, False disables it, an object uses that one. The
    default is "build it" because cg_energy_forces HAS NO P-C4' OR C4'-N TERM ANY MORE -- a round
    that ran unconstrained would sample a field in which those two distances are held only by
    whatever pulls on the beads, and C4'(L-1) by nothing at all.

    skip: coordinates excluded from the binning loop and from the joint J. None (the default)
    means B.CONSTRAINED whenever constraints are active and () otherwise. Pass () explicitly to
    score the constrained coordinates anyway, which is a thing to do once, deliberately, to see
    what the exclusion is worth.

    relax: descent steps on the injected Hamiltonian before the first Langevin step, 0 by default.
    Zero is the historical behaviour with nothing evaluated and no coordinate touched -- that
    matters, because round 0 of the 867-chain run was sampled under it and a path that quietly
    moved a bead would make the two incomparable. A nonzero value calls relax_positions and leaves
    its diagnostics on res.relax; see there for what a deposited start can be, and what it costs.
    """
    pot_kw = pot_kw or {}
    nb = max(blocks, 1)
    if nrep is None:
        nrep = pos.shape[0]

    if constraints is None:
        con = C.make_intra_constraints(pos.shape[1] // 3)
    elif constraints is False:
        con = None
    else:
        con = constraints
    if skip is None:
        skip = tuple(B.CONSTRAINED) if con is not None else ()
    else:
        skip = tuple(skip)

    res = RoundResult()
    res.b_counts = {b: {c: np.zeros(len(tab[c]["U"]), dtype=np.int64) for c in B.COORDS}
                    for b in range(nb)}
    res.b_acc = {b: {c: [0.0, 0.0, 0] for c in B.COORDS} for b in range(nb)}
    res.b_frames = [0] * nb
    res.skip = skip
    if collect_values:
        res.values = {c: [] for c in B.COORDS}
    if collect_positions:
        res.positions = []

    if seed is not None:
        torch.manual_seed(seed)

    _skip_set = set(skip)

    if relax:
        # Before t0 and before the seed is used for anything: the descent is deterministic, so it
        # belongs to neither the stochastics nor the timing this run records.
        pos, res.relax = relax_positions(pos, ij, pw, pot_kw=pot_kw, constraints=con,
                                         n_steps=relax, force_cap=force_cap, log=log)
        if log:
            log(f"  relaxed {res.relax['accepted']} accepted / "
                f"{res.relax['rejected']} rejected of {res.relax['steps']} steps: "
                f"E {res.relax['energy_start']:.1f} -> {res.relax['energy_end']:.1f} kJ/mol, "
                f"max|F| {res.relax['max_force_start']:.1f} -> "
                f"{res.relax['max_force_end']:.1f}"
                + ("  (left the cap)" if res.relax["hit_cap"] and res.relax["left_cap"] else ""))

    def _forces_at(p):
        """Fresh forces at the post-update coordinates, for the symplectic tail kick.

        The cell list has to be rebuilt here rather than reused: it is built from positions.
        """
        cl2 = C.GPUCellList(cell_size=1.5)
        cl2.build(p)
        return C.cg_energy_forces(p, ij, pw, cell_list=cl2, force_cap=force_cap, **pot_kw)[1]

    t0 = time.time()
    for step in range(nsteps):
        with torch.no_grad():
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl, force_cap=force_cap, **pot_kw)
            pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=DT_PS,
                                             mass_amu=MASS_AMU, friction=friction,
                                             force_fn=_forces_at, constraints=con)
        if step == 0:
            log(f"first step {time.time() - t0:.3f} s")
        if step >= burn and step % stride == 0:
            # which disjoint block this sample falls in; blocks are equal in TIME, not in count
            blk = min(((step - burn) * nb) // max(nsteps - burn, 1), nb - 1)
            res.b_frames[blk] += 1
            with torch.no_grad():
                for c in B.COORDS:
                    if c in _skip_set:
                        continue
                    q = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
                    if res.values is not None:
                        res.values[c].append(q.copy())
                    a = res.b_acc[blk][c]
                    a[0] += q.sum()
                    a[1] += (q ** 2).sum()
                    a[2] += q.size
                    t = tab[c]
                    k = np.round((q - t["centre"][0]) / t["binw"]).astype(np.int64)
                    ok = (k >= 0) & (k < len(t["U"]))
                    res.b_counts[blk][c] += np.bincount(k[ok], minlength=len(t["U"]))
                    res.n_total[c] += int(q.size)
                    res.n_outside[c] += int((~ok).sum())
                _cm, _cb, _cbl = closest_bead_pair(pos.reshape(nrep, -1, 3))
                res.clash_min.append(_cm)
                res.clash_below += _cb
                res.clash_below_live += _cbl
                if res.positions is not None:
                    # ONE FRAME'S GEOMETRY, kept so a caller can ask a question the histograms
                    # cannot answer: did this chain stay near the geometry it was started from?
                    # Size is frames x 3L x 3 x 8 B -- 17 MB for the largest chain at stride 25
                    # and a 2000-step run, which is why it is off by default and why no round of
                    # the production loop pays for it.
                    res.positions.append(pos.detach().clone())
        if progress and (step + 1) % max(nsteps // 10, 1) == 0:
            el = time.time() - t0
            _ct, _ac = summed(res.b_counts, res.b_acc, nb)
            vals, j = simref(_ac, tab, skip)
            _used, _off = j_denominator(_ac, tab, skip)
            parts = [f"{c[:5]} {v:5.3f}" if v == v else f"{c[:5]}   --" for c, v in zip(B.COORDS, vals)]
            log(f"  {step+1:>7d} {el:6.0f}s  " + "  ".join(parts) +
                f"   J {j:.4f} over {_used}/{_off}")

    res.seconds = time.time() - t0
    res.steps_per_s = nsteps / res.seconds if res.seconds else float("nan")
    res.pos, res.vel = pos, vel
    res.counts, res.acc = summed(res.b_counts, res.b_acc, nb)
    res.j_coords = j_denominator(res.acc, tab, skip)
    if res.values is not None:
        res.values = {c: (np.asarray(v, dtype=np.float64) if v
                          else np.zeros((0, 1), dtype=np.float64))
                      for c, v in res.values.items()}
    if res.positions is not None:
        res.positions = (torch.stack(res.positions) if res.positions
                         else torch.zeros((0, pos.shape[0], pos.shape[1], 3)))
    return res


def write_round(outdir, res, tab, meta=None):
    """Write one round's histograms in the form ibi_bonded.plan_update consumes.

    This is the function that closes the "a human reads the log" gap. ibi_round0.py computes
    dU = kBT*ln(P_sim/P_ref) and stops there -- it writes nothing -- so the update half has no
    caller and the numbers have to be retyped by eye. Everything plan_update's SimHistogram
    wants lands here, one file per coordinate, alongside the table that was simulated and the
    raw samples so a round can be re-binned or re-measured later without resampling.

    counts/n/n_outside are the three fields plan_update checks; n and n_outside are NOT
    derivable from counts, which is why they are counted in the binning loop rather than
    reconstructed. values is the raw q; for a run of 8 x 4000 frames x 80 windows it is about
    2.6e6 float64 per coordinate, which is worth carrying to keep a round re-analysable.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    nb = len(res.b_frames)
    for c in B.COORDS:
        vals = res.values[c] if res.values is not None else np.zeros((0, 1), dtype=np.float64)
        np.savez(out / f"{c}.npz",
                 counts=res.counts[c],
                 n=np.int64(res.n_total[c]),
                 n_outside=np.int64(res.n_outside[c]),
                 lo=float(tab[c]["lo"]), hi=float(tab[c]["hi"]), binw=float(tab[c]["binw"]),
                 nbins=np.int64(len(tab[c]["U"])),
                 U=tab[c]["U"], centre=tab[c]["centre"], sigma=float(tab[c]["sigma"]),
                 block_counts=np.asarray([res.b_counts[b][c] for b in range(nb)], dtype=np.int64),
                 block_frames=np.asarray(res.b_frames, dtype=np.int64),
                 values=vals)
    if meta is not None:
        (out / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
