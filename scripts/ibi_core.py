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


def load_tables(npz_path):
    """The six stored tables in boltzmann_bonded's format.

    Note this does NOT call B.prepare: the loop bins against centre/U/binw/sigma and never
    evaluates a table as a potential, so the interpolation cache would be dead weight here. A
    caller that injects a table potential builds it through cg_potentials, which goes to
    force_reference for the interpolation instead.
    """
    z = np.load(npz_path)
    need = [f"{n}__{k}" for n in B.COORDS
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
            for name in B.COORDS}


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


def simref(acc, tab):
    """(six sim/ref values, joint J) from a moments accumulator.

    J is the mean over the coordinates of |ln(sim/ref)|; a coordinate with no samples is skipped
    rather than counted as zero. Returned in B.COORDS order.
    """
    vals, js = [], []
    for c in B.COORDS:
        s1, s2, n = acc[c]
        if not n:
            vals.append(float("nan"))
            continue
        mm = s1 / n
        ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
        sig = tab[c]["sigma"]
        r = ss / sig if sig > 0 else float("nan")
        vals.append(r)
        if r == r and r > 0:
            js.append(abs(float(np.log(r))))
    return vals, (float(np.mean(js)) if js else float("nan"))


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
        self.values = None        # {coord: (frames, M)} raw q, only when collect_values
        self.pos = None           # final coordinates
        self.vel = None
        self.seconds = 0.0
        self.steps_per_s = 0.0


def run_round(*, pos, vel, ij, pw, temps, tab, nsteps, burn, stride, blocks, friction,
              force_cap, pot_kw=None, seed=None, collect_values=False, nrep=None,
              progress=True, log=print):
    """Sample, binning into `blocks` disjoint equal-time blocks. Returns a RoundResult.

    The accumulators are per block on purpose. A cumulative J over [burn, t] cannot separate
    "the window is settling" from "the early part of the window happened to look good": it is
    one number over everything seen so far. That is how the 40-200 ps run read 0.0911 and then
    climbed to 0.0968 -- the second half was worse and the average hid it behind the first.

    force_cap is passed through to cg_energy_forces unchanged, including None. pot_kw is the
    injection dict ({"angle_potential": .., "dihedral_potential": ..}), empty for the shipped
    field. Both are the caller's decision and neither is defaulted here.
    """
    pot_kw = pot_kw or {}
    nb = max(blocks, 1)
    if nrep is None:
        nrep = pos.shape[0]

    res = RoundResult()
    res.b_counts = {b: {c: np.zeros(len(tab[c]["U"]), dtype=np.int64) for c in B.COORDS}
                    for b in range(nb)}
    res.b_acc = {b: {c: [0.0, 0.0, 0] for c in B.COORDS} for b in range(nb)}
    res.b_frames = [0] * nb
    if collect_values:
        res.values = {c: [] for c in B.COORDS}

    if seed is not None:
        torch.manual_seed(seed)

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
                                             force_fn=_forces_at)
        if step == 0:
            log(f"first step {time.time() - t0:.3f} s")
        if step >= burn and step % stride == 0:
            # which disjoint block this sample falls in; blocks are equal in TIME, not in count
            blk = min(((step - burn) * nb) // max(nsteps - burn, 1), nb - 1)
            res.b_frames[blk] += 1
            with torch.no_grad():
                for c in B.COORDS:
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
                beads = pos.reshape(nrep, -1, 3)
                dd = torch.cdist(beads, beads)
                dd = dd + torch.eye(dd.shape[-1], device=dd.device) * 10.0
                res.clash_min.append(float(dd.min()))
                res.clash_below += int((dd < C.CLASH_DIST).sum())
                res.clash_below_live += int((dd < C.CLASH_SIGMA).sum())
        if progress and (step + 1) % max(nsteps // 10, 1) == 0:
            el = time.time() - t0
            _ct, _ac = summed(res.b_counts, res.b_acc, nb)
            vals, j = simref(_ac, tab)
            parts = [f"{c[:5]} {v:5.3f}" if v == v else f"{c[:5]}   --" for c, v in zip(B.COORDS, vals)]
            log(f"  {step+1:>7d} {el:6.0f}s  " + "  ".join(parts) + f"   J {j:.4f}")

    res.seconds = time.time() - t0
    res.steps_per_s = nsteps / res.seconds if res.seconds else float("nan")
    res.pos, res.vel = pos, vel
    res.counts, res.acc = summed(res.b_counts, res.b_acc, nb)
    if res.values is not None:
        res.values = {c: (np.asarray(v, dtype=np.float64) if v
                          else np.zeros((0, 1), dtype=np.float64))
                      for c, v in res.values.items()}
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
