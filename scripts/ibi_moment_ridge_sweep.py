"""How much of the moment operator's step is information, and how much is the ridge.

WHY THIS EXISTS. Plan C's parametric arm was stabilised on 2026-09-23 by a ridge sweep on a
seven-chain self-consistent refit, and the shape of that sweep is the reason to run this one: raising
the ridge from 1e-3 to 1e-1 moved max|dU| from 60.6 to 23.2 kJ/mol while the ensemble-weighted step
barely moved (1.7 -> 2.6 kJ/mol of mass-weighted std). In other words max|dU| was reporting tail
oscillation, not a force the ensemble felt.

The production operator carries its own ridge -- ibi_bonded.DEFAULT_RIDGE_REL, hard-coded at
1e-3 * trace(Cov)/n since the toy test that showed a 1e-8 ridge turning a matching simulation's
rounding-level moment difference into ~1 kJ/mol -- and it has never been swept. This sweeps it
against the real 867-chain histograms, on the fields the loop actually used, with NO sampling: every
number comes from the task files already on disk.

WHAT IT REPORTS, per coordinate and per ridge, and why each one is there:

  max|dU|        the operator's own headline, in kJ/mol and kBT -- a statement about the TAILS
  bulk max|dU|   the same maximum restricted to bins that hold at least 1 percent of the peak
                 density: the step where the ensemble actually lives
  rms step       sqrt(sum_i p_i (dU_i - <dU>_p)^2), the force perturbation the sampler feels, with
                 p_i the simulation's own bin probabilities
  tail share     the fraction of that variance contributed by bins below 1 percent of the peak --
                 the number that says whether the headline is about the field or about its tails
  |d<T>|max      the moment norm the step is about to cancel (the operator's own objective measure)

Usage: python scripts/ibi_moment_ridge_sweep.py [round ...]   (default: the latest round on disk)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "ibi_relax"
ARCHIVE = OUT / "table_operator_archive"
REF_NPZ = REPO / "results" / "refit_smooth5.npz"
DEST = REPO / "results" / "ibi_moment_ridge"
COORDS = ("bb_bond", "angle", "dihedral")
RIDGES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
KS = (4, 8, 16)
KBT = 2.494
BULK_FRAC = 0.01     # "the bulk" = bins above 1 percent of the peak probability


def load_histograms(round_n):
    """Pooled counts, n_total and n_outside per coordinate, summed over every task on disk."""
    d = OUT / f"tasks_r{round_n}"
    files = sorted(d.glob("*.npz"))
    if not files:
        raise SystemExit(f"no task files under {d}")
    counts, n_tot, n_out = {}, {}, {}
    for f in files:
        with np.load(f) as z:
            for c in ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack"):
                counts[c] = counts.get(c, 0) + z[f"counts__{c}"]
                n_tot[c] = n_tot.get(c, 0) + int(z[f"n_total__{c}"])
                n_out[c] = n_out.get(c, 0) + int(z[f"n_outside__{c}"])
    return files, counts, n_tot, n_out


def metrics(dU, counts, diagnostics):
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    mean = float((p * dU).sum())
    var = float((p * (dU - mean) ** 2).sum())
    peak = p.max()
    bulk = p >= BULK_FRAC * peak
    return {
        "max_abs_dU": float(np.abs(dU).max()),
        "max_abs_dU_kBT": float(np.abs(dU).max()) / KBT,
        "bulk_max_abs_dU": float(np.abs(dU[bulk]).max()) if bulk.any() else float("nan"),
        "rms_step": float(np.sqrt(max(var, 0.0))),
        "tail_share": float((p[~bulk] * (dU[~bulk] - mean) ** 2).sum() / var) if var > 0 else float("nan"),
        "moment_norm": float(diagnostics.get("moment_norm", float("nan"))),
        "rel_entropy_drop_kbt": float(diagnostics.get("rel_entropy_drop_kbt", float("nan"))),
    }


def main():
    rounds = [int(a) for a in sys.argv[1:]]
    if not rounds:
        rounds = [max(int(p.stem[5:]) for p in OUT.glob("round*.json"))]
    import os
    os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))
    os.environ.setdefault("IBI_LOOP_REF", str(REF_NPZ))
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import boltzmann_bonded as B
    import ibi_bonded as I
    import ibi_core as IC

    p_ref = {c: I.probability_from_table(t) for c, t in I.load_clean_tables(REF_NPZ).items()}
    out = {}
    for rnd in rounds:
        files, counts, n_tot, n_out = load_histograms(rnd)
        # BOTH FIELDS FOR THIS ROUND: the one the loop actually sampled under (archived before the
        # replay overwrote it) and the moment trajectory's own, which is what the running rounds use.
        sources = {"sampled": ARCHIVE / f"tables_r{rnd}.npz", "live": OUT / f"tables_r{rnd}.npz"}
        print(f"=== round {rnd}: {len(files)} task files ===")
        for tag, path in sources.items():
            if not path.exists():
                print(f"  {tag}: {path.name} missing, skipped")
                continue
            tables = IC.load_tables(str(path))
            print(f"  field [{tag}] {path}")
            for c in COORDS:
                rows = []
                for K in KS:
                    for ridge in RIDGES:
                        try:
                            res = I.moment_correction(tables[c], counts[c], n_tot[c], n_out[c],
                                                      p_ref[c], K=K, gain=1.0, ridge_rel=ridge)
                            m = metrics(np.asarray(res.dU, dtype=float), counts[c], res.diagnostics)
                            m.update({"K": K, "ridge_rel": ridge, "status": res.status,
                                      "reason": res.reason})
                        except Exception as exc:                      # singular solve at ridge 0
                            m = {"K": K, "ridge_rel": ridge, "status": "raised",
                                 "reason": f"{type(exc).__name__}: {exc}"}
                        rows.append(m)
                out.setdefault(f"r{rnd}_{tag}", {})[c] = rows
                print(f"    {c}")
                print("      K  ridge     max|dU|  bulk max   rms step  tail share  |d<T>|max  dS(kBT)  status")
                for m in rows:
                    # "ok" and "converged" both mean the operator produced a step; the driver is
                    # what renames them to "applied", so a sweep of the operator itself sees the
                    # internal names (ibi_bonded.STATUS_*).
                    if "max_abs_dU" in m:
                        print(f"      {m['K']:>2} {m['ridge_rel']:<9.0e} {m['max_abs_dU']:9.3f} "
                              f"{m['bulk_max_abs_dU']:9.3f} {m['rms_step']:10.3f} "
                              f"{m['tail_share']:11.3f} {m['moment_norm']:10.4f} "
                              f"{m['rel_entropy_drop_kbt']:8.4f}  {m['status']}")
                    else:
                        print(f"      {m['K']:>2} {m['ridge_rel']:<9.0e} "
                              f"{'-':>9} {'-':>9} {'-':>10} {'-':>11} {'-':>10} {'-':>8}  "
                              f"{m['status']}: {str(m['reason'])[:44]}")
                def pick(K, r):
                    for m in rows:
                        if m["K"] == K and m["ridge_rel"] == r:
                            return m
                    return None
                a, b = pick(8, 1e-3), pick(8, 1e-1)
                if a and b and "max_abs_dU" in a and "max_abs_dU" in b:
                    print(f"      -> K=8: ridge 1e-3 -> 1e-1 moves max|dU| {a['max_abs_dU']:.3f} -> "
                          f"{b['max_abs_dU']:.3f} kJ/mol, rms step {a['rms_step']:.3f} -> "
                          f"{b['rms_step']:.3f}, tail share {a['tail_share']:.3f} -> {b['tail_share']:.3f}")
        print()
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / f"moment_ridge_sweep_r{'-'.join(str(r) for r in rounds)}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
