"""Compose the production table file: bb_bond and dihedral from the campaign, angle from the refit.

WHY THIS MIX, and it is a measured choice rather than a preference. Over the 9-round campaign at full
pool (867 chains):

  * bb_bond CONVERGED. Its correction fell to 0.35-0.60 kJ/mol and against the table's own sigma its
    sim/ref sat at 1.004 by round 3.
  * dihedral CONVERGED too (correction 0.38 -> 0.27 kJ/mol, sim/ref 1.015 by round 3), and its
    edge-mass deficit turned out to be a BASIS problem, fixed by a B-spline basis rather than by more
    rounds. The campaign's dihedral table is still the best available one.
  * angle did NOT converge under EITHER operator: the marginal inversion rang (3.00 / 3.94 / 3.02 /
    4.27 / 4.31 / 3.72 kJ/mol over rounds 0-5, no decay) and the moment operator drove the table's
    implied sigma from 0.3218 down to 0.1307 while the SAMPLED sigma followed only a third of the way
    (0.4159 -> 0.3493). The measured transmission over the nine rounds is 0.311 (R2 0.70,
    scripts/ibi_transmission_scan.py) and the same line's intercept -- the pooled width at an
    infinitely narrow table -- is 0.3253, ABOVE the 0.3218 target: the angle was at the model's floor
    by round 8 (|ln(sampled/target)| = 0.082, inside the 10 percent the sampled-side verdict asks for,
    scripts/ibi_verdict.py), and what the campaign kept buying with more rounds was table drift, not
    geometry. So the production field keeps the PRE-CAMPAIGN refit table for it, which is also the
    target the campaign was inverting against, and it is kept for the reason that matters: the table
    stops moving, and the sampled marginal it produces is already inside tolerance.
  * stack is not injected at all: it is an exact function of bb_bond and the P-P-P angle, the sampler
    has no stack potential, and its spring in torch_cgsim is zero for the same reason.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import ibi_core as IC                     # noqa: E402  (the campaign record moved: see below)

# THE CAMPAIGN RECORD MOVED, and this path used to be hard-coded to it. results/ibi_relax is gone --
# a housekeeping pass parked the 0.44 GB of per-chain histograms out of results/ on 2026-10-01 and
# only part of it came back under _strays. ibi_core.campaign_root() is the one place that knows the
# candidates and prefers the complete copy, so this builder runs again from a fresh checkout.
CAMPAIGN = IC.campaign_root() / "tables_r9.npz"
REFIT = REPO / "results" / "refit_smooth5.npz"
DEST = REPO / "results" / "production_tables.npz"
KBT = 2.494


def record(path, coord):
    with np.load(path) as z:
        return {"lo": float(z[f"{coord}__lo"]), "binw": float(z[f"{coord}__binw"]),
                "U": np.asarray(z[f"{coord}__U"], dtype=float)}


def implied_sigma(rec):
    U = rec["U"]
    p = np.exp(-(U - U.min()) / KBT)
    p = p / p.sum()
    centre = rec["lo"] + rec["binw"] * (np.arange(len(U)) + 0.5)
    mean = float((p * centre).sum())
    return float(np.sqrt(max((p * (centre - mean) ** 2).sum(), 0.0)))


def main():
    out = {}
    print("coord     source      implied_sigma        lo      binw")
    for coord, source in (("bb_bond", "campaign"), ("dihedral", "campaign"), ("angle", "refit")):
        rec = record(CAMPAIGN if source == "campaign" else REFIT, coord)
        other = record(REFIT if source == "campaign" else CAMPAIGN, coord)
        for field in ("lo", "binw"):
            assert abs(rec[field] - other[field]) < 1e-12, (
                coord + ": " + field + " differs between the two source files, so they are not on "
                "the same grid and the mix would be meaningless")
        if len(rec["U"]) != len(other["U"]):
            raise SystemExit(coord + ": bin counts differ")
        print(f"{coord:9s} {source:10s} {implied_sigma(rec):14.5f} {rec['lo']:9.5f} {rec['binw']:9.5f}")
        out[coord + "__U"] = rec["U"]
        out[coord + "__lo"] = rec["lo"]
        out[coord + "__binw"] = rec["binw"]
    np.savez(DEST, **out)
    print("")
    print("wrote " + str(DEST.relative_to(REPO)))
    print("  pre-campaign (refit) implied sigmas: " + ", ".join(
        c + "=" + format(implied_sigma(record(REFIT, c)), ".5f")
        for c in ("bb_bond", "angle", "dihedral")))


if __name__ == "__main__":
    main()
