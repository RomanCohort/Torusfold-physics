"""Why do the reconstructions keep so few WC contacts? Measure the key-atom distances themselves.

The criterion is all key contacts within 3.6 A (A N1-U N3 / A N6-U O4, G N1-C N3 / G O6-C N4 / G N2-C O2).
Products keep 0-3 of 12 pairs, which says something is wrong but not WHAT. This prints every key distance
for every crystal pair, in the deposit and in each product, so the failure mode is visible: too far, too
close, or present but pointing the wrong way.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import measure_base_stacking as M

KEY = {("A", "U"): (("N1", "N3"), ("N6", "O4")),
       ("G", "C"): (("N1", "N3"), ("O6", "N4"), ("N2", "O2"))}


def key_dists(path, tag, seq_ref=None, pairs_ref=None):
    seq, residues, _p, _ch = M.parse_pdb(path)
    if seq_ref is None:
        planes = [M.plane(r) for r in residues]
        pairs = M.wc_pairs(seq, residues, planes)
        seq_ref, pairs_ref = seq, pairs
    print("== %s (%d residues, %d pairs from the reference)" % (tag, len(seq), len(pairs_ref)))
    out = []
    for (i, j) in pairs_ref:
        trip = KEY.get((seq_ref[i], seq_ref[j])) or KEY.get((seq_ref[j], seq_ref[i]))
        if trip is None:
            continue
        flip = (seq_ref[i], seq_ref[j]) not in KEY
        ai, aj = residues[i]["atoms"], residues[j]["atoms"]
        ds = []
        for a, b in trip:
            na, nb = (b, a) if flip else (a, b)
            if na not in ai or nb not in aj:
                ds.append(float("nan"))
                continue
            ds.append(float(np.linalg.norm(np.asarray(ai[na], float) - np.asarray(aj[nb], float))))
        out.append((i, j, ds))
        print("   %2d-%2d %s  %s   worst %.2f A  %s"
              % (i, j, seq_ref[i] + seq_ref[j],
                 " ".join("%s-%s %5.2f" % (t[0] if not flip else t[1], t[1] if not flip else t[0], d)
                          for t, d in zip(trip, ds)),
                 np.nanmax(ds), "OK" if np.nanmax(ds) <= 3.6 else ""))
    ok = sum(1 for _i, _j, ds in out if np.nanmax(ds) <= 3.6)
    print("   -> %d/%d pairs satisfy the criterion\n" % (ok, len(out)))
    return seq_ref, pairs_ref


seq, pairs = key_dists("artifacts/2oiu/2OIU.pdb", "DEPOSIT")
key_dists("results/plan_c/bead_product/2oiu_refine.pdb", "PRODUCT: refine mode + bead frame", seq, pairs)
key_dists("results/plan_c/bead_product/2oiu_pipeline.pdb", "PRODUCT: folding mode + bead frame", seq, pairs)
key_dists("results/plan_c/bead_product/2oiu_product_ptrace.pdb", "PRODUCT: P-trace reconstruction", seq, pairs)
