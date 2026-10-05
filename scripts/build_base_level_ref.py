"""The reference tables for the three BASE-LEVEL coordinates, measured from the crystal database.

WHY. boltzmann_bonded.COORDS gained base_dist, base_rise and base_cos on 2026-10-05 so that the loop can
score them, and the loop needs a p_ref for every coordinate it scores. The definitions live in coords_of
(they are functions of the three beads through the rigid template map), so this script measures the TARGET
by calling the very same function on crystal beads -- there is no second implementation to drift.

The output is a MERGED reference file: every entry of the existing table file is copied verbatim and the
three new ones are appended, so a loop arm can point IBI_LOOP_REF at it and get all nine coordinates.

Grids are fixed and wide on purpose: a coordinate's support has to cover what the SAMPLER does, not only
what the crystals do, or the out-of-support fraction trips the gate and the round reports a truncated view.
Measured ranges (findings Parts 15-18): sampled base_dist 0.3-1.6 nm, base_rise -0.7 to +1.0 nm, base_cos
in [-1, 1].
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B               # noqa: E402
import ibi_core as IC                      # noqa: E402
import measure_base_stacking as M          # noqa: E402

REF = REPO / "results" / "refit_smooth5.npz"
OUT = REPO / "results" / "refit_smooth5_with_base.npz"
ONLY = REPO / "results" / "base_level_ref.npz"
KBT = 2.494
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
GRID = {"base_dist": (0.20, 1.80), "base_rise": (-1.20, 1.20), "base_cos": (-1.0, 1.0)}
NBINS = 1000
SMOOTH_SIGMA_BINS = 3.0


def smooth(p, sigma=SMOOTH_SIGMA_BINS):
    if sigma <= 0:
        return p
    half = int(4 * sigma)
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([p[half:0:-1], p, p[-2:-half - 2:-1]]) if half > 0 else p
    return np.convolve(pad, k, mode="same")[half:half + len(p)]


def main():
    n_frag = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    vals = {k: [] for k in GRID}
    files = sorted((REPO / "_cgdata" / "combined").glob("*.pdb"))
    done = 0
    for f in files:
        if done >= n_frag:
            break
        seq, residues, _pP, _ch = M.parse_pdb(f)
        if seq is None or len(seq) < 8 or len(seq) > 300:
            continue
        try:
            beads = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]]
                              for r in residues])
        except KeyError:
            continue
        if not np.all(np.isfinite(beads)):
            continue
        pos = torch.tensor((beads / 10.0).reshape(1, -1, 3), dtype=torch.float64)   # A -> nm
        with torch.no_grad():
            for k in GRID:
                vals[k].append(B.coords_of(pos, k).reshape(-1).numpy())
        done += 1
    print("fragments measured: %d" % done, flush=True)

    tables = IC.load_tables(str(REF))
    new = {}
    for k, (lo, hi) in GRID.items():
        v = np.concatenate(vals[k])
        v = v[np.isfinite(v)]
        hist, _edges = np.histogram(v, bins=NBINS, range=(lo, hi))
        p = smooth(hist.astype(float))
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        U = -KBT * np.log(p)
        U = U - U.min()
        centre = lo + (hi - lo) / NBINS * (np.arange(NBINS) + 0.5)
        mode = float(centre[int(np.argmin(U))])
        new[k] = {"lo": float(lo), "hi": float(hi), "binw": float((hi - lo) / NBINS),
                  "nbins": NBINS, "centre": centre, "U": U, "sigma": float(np.std(v)),
                  "target": mode}
        print("  %-10s n=%7d  mean %8.4f  sd %7.4f  mode %8.4f  min(U) at bin %d"
              % (k, v.size, v.mean(), v.std(), mode, int(np.argmin(U))), flush=True)

    merged = dict(tables)
    merged.update(new)
    payload = {}
    for c, t in merged.items():
        payload[f"{c}__U"] = np.asarray(t["U"], dtype=float)
        payload[f"{c}__lo"] = float(t["lo"])
        payload[f"{c}__hi"] = float(t["hi"])
        payload[f"{c}__binw"] = float(t["binw"])
        payload[f"{c}__sigma"] = float(t.get("sigma", 0.0))
        if "centre" in t:
            payload[f"{c}__centre"] = np.asarray(t["centre"], dtype=float)
    np.savez(OUT, **payload)
    np.savez(ONLY, **{f"{c}__{f}": v for c, t in new.items()
                      for f, v in (("U", t["U"]), ("lo", [t["lo"]]), ("hi", [t["hi"]]),
                                   ("binw", [t["binw"]]), ("sigma", [t["sigma"]]))})
    print("\nwrote %s (%d coordinates: %s)" % (OUT.name, len(merged), ", ".join(sorted(merged))),
          flush=True)
    check = IC.load_tables(str(OUT))
    print("reload check: %d coordinates, base entries present: %s"
          % (len(check), all(k in check for k in GRID)), flush=True)


if __name__ == "__main__":
    main()
