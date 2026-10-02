"""The same-field floor: two independent trajectories under ONE field, measured.

WHY THIS EXISTS. Every floor in docs/archive/plan_c_c2_stabilization.md section 4.2 was bracketed from pairs
whose FIELDS also moved -- the arms that had all but stopped (<= 1 kJ/mol injected step) -- so those
numbers bound the sampling noise instead of measuring it. This samples one field twice, with
different seeds, and the distance between the resulting ensembles is sampling noise and nothing else.

THE FIELD IS THE ONE C2s8's LAST ROUND ENDED ON (results/plan_c/fields/c2stab/C2s8_r4.npz), because
that is the field section 4.3's verdict has to be compared against: C2s8's last round-to-round
distances are 0.114 / 0.102 / 0.816 (ln_mean, bb_bond / angle / dihedral) and the question is
whether those are noise or movement. Same pool (the seven chains), same protocol (relax 5000, burn
1000, window 5000, stride 5, 8 replicas), two seeds.

Budget: 7 chains x 8 replicas x 5000 steps x 2, at most 6 worker processes. Measured cost 614 s of
wall clock on 6 workers (1.0 core-hours).

IT ALSO STORES THE TWO THINGS THE POOLED ENSEMBLES THREW AWAY (to-run item 2): per-chain and
per-block counts, in plan_c_loop.store_ensembles' own key convention so the offline instrument reads
this file with the accessors it already has.

--reuse re-analyses a stored run instead of sampling, which is the point of storing the
decomposition: measured 2026-09-24, the first analysis pass died on a chain whose angle histogram
was empty (every observation outside the support), and the counts on disk made that a two-second
re-analysis instead of another eleven minutes of sampling. Empty chains are now reported and skipped.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))
for _n, _d in (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("KMP_BLOCKTIME", "0")):
    os.environ.setdefault(_n, _d)
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402
torch.set_num_threads(1)

import boltzmann_bonded as B                            # noqa: E402
import ibi_core as IC                                   # noqa: E402
import plan_c_instrument as PI                          # noqa: E402
import plan_c_loop as L                                 # noqa: E402
import torusfold.scheme2.torch_cgsim as C               # noqa: E402

OUT = REPO / "results" / "plan_c"
DEFAULT_FIELD = OUT / "fields" / "c2stab" / "C2s8_r4.npz"
BLOCKS = 8
# THE SAMPLED COORDINATES, not B.COORDS. b.COORDS also lists intra_pc and intra_cn, which are SHAKE
# constraints and carry no samples at all (n_total = 0 in every round of every arm), so iterating
# over it asks the instrument to compare two empty histograms and it refuses -- measured 2026-09-24,
# that is what stopped the first analysis pass after one row. stack is sampled but carried, and the
# C2s8 verdict this floor is compared against is about these three.
COORDS = ("bb_bond", "angle", "dihedral")
# C2s8's last round-to-round distances (ln_mean, docs/archive/plan_c_c2_stabilization.md 4.3), the numbers
# this floor exists to be compared with.
C2S8_LAST = {"bb_bond": 0.114, "angle": 0.102, "dihedral": 0.816}


def _sample(task):
    """One (chain, trajectory) pair. Returns its whole histogram, per chain and per block."""
    chain_i, traj_i, name, pos_np, pairs, field_npz, seed, nrep, nsteps, burn, stride = task
    Lc = pos_np.shape[0]
    _pots, pot_kw = L._build_potentials(Path(field_npz))
    tab = IC.load_tables(str(field_npz))
    pos = torch.tensor(np.asarray(pos_np, dtype=np.float64).reshape(1, 3 * Lc, 3)).repeat(nrep, 1, 1)
    ij = torch.tensor(list(pairs), dtype=torch.long).reshape(-1, 2)
    con = C.make_intra_constraints(Lc)
    t0 = time.time()
    res = IC.run_round(pos=pos, vel=torch.zeros_like(pos), ij=ij,
                       pw=torch.ones(len(ij), dtype=torch.float32),
                       temps=torch.full((nrep,), 300.0, dtype=torch.float64), tab=tab,
                       nsteps=nsteps, burn=burn, stride=stride, blocks=BLOCKS,
                       friction=L.FRICTION, force_cap=L.FORCE_CAP, pot_kw=pot_kw, seed=seed,
                       nrep=nrep, progress=False, constraints=con, relax=5000,
                       log=lambda *a, **k: None)
    return {"chain": int(chain_i), "traj": int(traj_i), "name": name, "L": int(Lc),
            "seconds": time.time() - t0,
            "counts": {c: np.asarray(res.counts[c], dtype=np.int64) for c in B.COORDS},
            "blocks": {c: np.asarray([res.b_counts[b][c] for b in range(BLOCKS)], dtype=np.int64)
                       for c in B.COORDS},
            "n_total": {c: int(res.n_total[c]) for c in B.COORDS},
            "n_outside": {c: int(res.n_outside[c]) for c in B.COORDS}}


def _stored_keys(traj, ch, c):
    """Both conventions, newest first: plan_c_loop.store_ensembles' then this script's first pass."""
    return (f"t{traj}__chain{ch}__{c}", f"t{traj}_chain{ch}__{c}")


def _stored_block_key(z, traj, ch, c):
    """This chain's blocks, from the per-chain array key or from the first pass's per-chain key."""
    arr = f"t{traj}__chain_blocks__{c}"
    if arr in z.files:
        # A repeated re-save used to add a phantom chain axis on top of the real one; normalise to
        # (blocks, bins) rather than trusting the file's rank.
        v = np.asarray(z[arr], dtype=np.int64)
        while v.ndim > 3:
            v = v[:, 0]
        return v[ch]
    old = f"t{traj}_chain{ch}__blocks__{c}"
    return np.asarray(z[old], dtype=np.int64) if old in z.files else None


def load_stored(path, names):
    """Rebuild the per-(trajectory, chain) dicts from a stored npz, in the loop's key convention."""
    z = np.load(path)
    ens = {}
    for traj in (0, 1):
        for ch, name in enumerate(names):
            key = next((k for k in _stored_keys(traj, ch, "bb_bond") if k in z.files), None)
            if key is None:
                raise SystemExit(f"{path} has no per-chain counts for t{traj} chain{ch}")
            row = {"chain": ch, "traj": traj, "name": name, "L": None, "seconds": float("nan"),
                   "counts": {}, "blocks": {}, "n_total": {}, "n_outside": {}}
            for c in COORDS:
                row["counts"][c] = np.asarray(
                    z[next(k for k in _stored_keys(traj, ch, c) if k in z.files)], dtype=np.int64)
                bk = _stored_block_key(z, traj, ch, c)
                if bk is not None:
                    row["blocks"][c] = bk
            ens[(traj, ch)] = row
    return ens


def resave(path, names, ens):
    """Rewrite the npz in the loop's convention, so the instrument's accessors read it."""
    store = {}
    for traj in (0, 1):
        for c in COORDS:
            blocks = [ens[(traj, ch)]["blocks"][c] for ch in range(len(names))
                      if ens[(traj, ch)]["blocks"]]
            if len(blocks) == len(names):
                block = np.stack([np.asarray(b, dtype=np.int64) for b in blocks])
                nbins = int(np.asarray(ens[(traj, 0)]["counts"][c]).size)
                if block.shape != (len(names), BLOCKS, nbins):
                    # A repeated re-save of a re-saved file nested a phantom chain axis here once
                    # and the jackknife then read a three-dimensional histogram. Refuse rather than
                    # write it back.
                    print(f"   refusing to re-save {c}: block array has shape {block.shape}, "
                          f"expected ({len(names)}, {BLOCKS}, {nbins})")
                    continue
                if all(np.array_equal(block[0], block[i]) for i in range(1, len(names))):
                    # SEVEN IDENTICAL PER-CHAIN BLOCK ARRAYS IS THE SIGNATURE OF THE DUPLICATION,
                    # not a coincidence: the run this file came from had a per-chain spread of
                    # 0.04 in ln_mean. Drop them instead of persisting the damage.
                    print(f"   refusing to re-save {c}: all {len(names)} per-chain block arrays "
                          f"are identical (duplication, not data)")
                    continue
                store[f"t{traj}__chain_blocks__{c}"] = block
                store[f"t{traj}__blocks__{c}"] = block.sum(axis=0)
            for ch in range(len(names)):
                store[f"t{traj}__chain{ch}__{c}"] = ens[(traj, ch)]["counts"][c]
    np.savez(path, **store)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--field", default=str(DEFAULT_FIELD))
    ap.add_argument("--seed-a", type=int, default=20260924)
    ap.add_argument("--seed-b", type=int, default=20260925)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--nrep", type=int, default=8)
    ap.add_argument("--nsteps", type=int, default=5000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default=str(OUT / "same_field_floor"))
    ap.add_argument("--reuse", action="store_true",
                    help="re-analyse --out.npz instead of sampling; the decomposition is why it can")
    args = ap.parse_args(argv)
    if args.workers > 6:
        raise SystemExit("--workers is capped at 6: the production round owns the rest of the box")

    field = Path(args.field)
    if not field.exists():
        raise SystemExit(f"field {field} does not exist")
    tabs = L.load_field(field)
    print(f"field {field.name}: coordinates {sorted(tabs)}")
    pool, hold = L.build_pool(7, 4)
    names = [s["name"] for s in pool]
    centre = {c: tabs[c]["centre"] for c in B.COORDS}

    if args.reuse:
        ens = load_stored(f"{args.out}.npz", names)
        resave(f"{args.out}.npz", names, ens)
        print(f"reused {args.out}.npz (no sampling); re-saved in the loop's key convention")
    else:
        tasks = []
        for traj_i, base in enumerate((args.seed_a, args.seed_b)):
            for i, s in enumerate(pool):
                tasks.append((i, traj_i, s["name"], np.asarray(s["pos"], dtype=np.float64),
                              s["pairs"], str(field), base + i, args.nrep, args.nsteps, args.burn,
                              args.stride))
        print(f"sampling {len(tasks)} tasks ({len(pool)} chains x 2 trajectories) on "
              f"{args.workers} workers; field {field.name}, seeds {args.seed_a}/{args.seed_b}")
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as p:
            rows = p.map(_sample, tasks)
        print(f"done in {time.time() - t0:.0f} s wall")
        ens = {(r["traj"], r["chain"]): r for r in rows}
        resave(f"{args.out}.npz", names, ens)

    json_payload = {"field": str(field), "seeds": [args.seed_a, args.seed_b],
                    "protocol": {"nrep": args.nrep, "nsteps": args.nsteps, "burn": args.burn,
                                 "stride": args.stride, "blocks": BLOCKS, "relax": 5000,
                                 "pool": names},
                    "per_chain": {}, "pooled": {}, "empty": {}}

    print("\n== SAME-FIELD FLOOR: two independent trajectories, one field (ln_mean / TV)")
    print(f"{'coord':10s} {'pooled ln_mean':>14s} {'pooled TV':>10s} "
          f"{'per-chain ln_mean min/med/max':>32s} {'per-chain TV med':>17s} {'chains':>7s}")
    for c in COORDS:
        a = sum(ens[(0, ch)]["counts"][c] for ch in range(len(pool)))
        b = sum(ens[(1, ch)]["counts"][c] for ch in range(len(pool)))
        pooled = PI.metrics(a, b, centre[c], reps=200)
        # A CHAIN WHOSE WHOLE HISTOGRAM IS EMPTY IS REPORTED, NOT CRASHED ON. Measured 2026-09-24:
        # the first pass died here, on an angle whose every observation fell outside the table's
        # support -- which is itself the finding, and it must not look like a broken script.
        empty, per = [], []
        for ch in range(len(pool)):
            ca, cb = ens[(0, ch)]["counts"][c], ens[(1, ch)]["counts"][c]
            if ca.sum() <= 0 or cb.sum() <= 0:
                empty.append({"chain": ch, "name": names[ch],
                              "n_a": int(ca.sum()), "n_b": int(cb.sum()),
                              "n_outside_a": ens[(0, ch)]["n_outside"].get(c),
                              "n_outside_b": ens[(1, ch)]["n_outside"].get(c)})
                continue
            per.append(PI.metrics(ca, cb, centre[c], reps=0))
        lm = np.array([d["ln_mean"] for d in per]) if per else np.array([np.nan])
        tv = np.array([d["tv"] for d in per]) if per else np.array([np.nan])
        json_payload["pooled"][c] = {k: pooled[k] for k in ("ln_mean", "tv", "n_a", "n_b",
                                                            "dq50_sig", "poisson_p95")}
        json_payload["per_chain"][c] = {
            "ln_mean": [float(v) for v in lm], "tv": [float(v) for v in tv],
            "ln_mean_min": float(np.nanmin(lm)), "ln_mean_median": float(np.nanmedian(lm)),
            "ln_mean_max": float(np.nanmax(lm)), "tv_median": float(np.nanmedian(tv)),
            "n_chains_used": len(per), "names": names}
        json_payload["empty"][c] = empty
        print(f"{c:10s} {pooled['ln_mean']:14.4f} {pooled['tv']:10.4f} "
              f"{np.nanmin(lm):10.4f} / {np.nanmedian(lm):.4f} / {np.nanmax(lm):.4f} "
              f"{np.nanmedian(tv):17.4f} {len(per):7d}")
        for e in empty:
            print(f"           empty: chain {e['chain']} {e['name']} (in-support "
                  f"{e['n_a']} vs {e['n_b']}, outside {e['n_outside_a']} vs {e['n_outside_b']}) "
                  f"-- every observation fell outside the support")

    print("\n== the comparison the floor exists for: C2s8's last step against it")
    print(f"{'coord':10s} {'C2s8 3->4':>10s} {'same-field floor':>17s} {'ratio':>7s} "
          f"{'is the step above the floor?':>30s}")
    for c in COORDS:
        fl = json_payload["pooled"][c]["ln_mean"]
        above = "yes" if C2S8_LAST[c] > fl else "no -- it IS the floor"
        print(f"{c:10s} {C2S8_LAST[c]:10.4f} {fl:17.4f} {C2S8_LAST[c]/fl:7.2f}x {above:>30s}")

    print("\n== does the pooled value scale like independent chains? (the sqrt(n) check)")
    for c in COORDS:
        p = json_payload["per_chain"][c]
        n = p["n_chains_used"]
        if not n:
            continue
        pred = json_payload["pooled"][c]["ln_mean"] * np.sqrt(n)
        print(f"{c:10s} per-chain median {p['ln_mean_median']:.4f} against pooled x sqrt({n}) "
              f"= {pred:.4f} (ratio {p['ln_mean_median']/pred:.2f}); per-chain spread "
              f"{p['ln_mean_max']-p['ln_mean_min']:.4f}")

    print("\n== block jackknife at full window size (the other thing the decomposition buys)")
    z = np.load(f"{args.out}.npz")
    for c in COORDS:
        if f"t0__blocks__{c}" not in z.files or f"t1__blocks__{c}" not in z.files:
            print(f"{c:10s} not available: this run's block decomposition did not survive the "
                  f"analysis script's re-save (see the doc); the floor above does not use it")
            continue
        ba = np.asarray(z[f"t0__blocks__{c}"], dtype=float)
        bb = np.asarray(z[f"t1__blocks__{c}"], dtype=float)
        jl, jt = [], []
        for b in range(ba.shape[0]):
            d = PI.metrics(np.delete(ba, b, axis=0).sum(axis=0),
                           np.delete(bb, b, axis=0).sum(axis=0), centre[c], reps=0)
            jl.append(d["ln_mean"])
            jt.append(d["tv"])
        JL, JT = PI.jackknife(jl), PI.jackknife(jt)
        json_payload.setdefault("jackknife", {})[c] = {"ln_mean": JL, "tv": JT}
        print(f"{c:10s} ln_mean {JL['mean']:.4f} +- {JL['std_jackknife']:.4f}   "
              f"TV {JT['mean']:.4f} +- {JT['std_jackknife']:.4f}   (n={JL['n']} blocks)")

    Path(f"{args.out}.json").write_text(json.dumps(json_payload, indent=1, default=float),
                                        encoding="utf-8")
    print(f"\nwrote {args.out}.npz and {args.out}.json")


if __name__ == "__main__":
    main()
