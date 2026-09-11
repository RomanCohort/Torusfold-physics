"""Why is the pair-weight channel zero for most true pairs? Two candidates, told apart by counting.

compute_rcm_score returns confidence = crossing_total / (crossing + within_up + within_down),
summed over k in [5, 7, 9, 11, 13] with max_mismatch = 0. A confidence of exactly 0 therefore
means crossing_total == 0, which happens either because

  (A) the flanks really contain no exact reverse-complement kmer pair -- a property of the
      input, or
  (B) rcm_crossing's cumulative-sum prefilter drops matches that the base-by-base validation
      would have accepted -- a property of the code.

The prefilter is score = |Re(s1_i + s2_j)| + |Im(s1_i + s2_j)| <= max_mismatch, where s1 and s2
are kmer sums of the complex base encoding. With max_mismatch = 0 it demands the two kmer sums
cancel exactly, which is necessary for a reverse complement but is claimed to be sufficient too
only for k=1. This counts the same thing twice, exactly, and compares.

Brute force is a Counter intersection: s1's kmers against the reverse complement of s2's kmers.

Run: python scripts/diagnose_pair_weight.py [n_pairs]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
from torusfold.scheme2.rcm import (   # noqa: E402
    _COMPLEMENT, compute_rcm_score, rcm_crossing, rcm_within,
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
K = [5, 7, 9, 11, 13]


def revcomp(s):
    return "".join(_COMPLEMENT.get(c, "N") for c in reversed(s))


def brute_crossing(s1, s2, k):
    """Exact count of (i, j) with s1[i:i+k] == reverse_complement(s2[j:j+k])."""
    if len(s1) < k or len(s2) < k:
        return 0
    c1 = Counter(s1[i:i + k] for i in range(len(s1) - k + 1))
    c2 = Counter(revcomp(s2)[j:j + k] for j in range(len(s2) - k + 1))
    return sum(n * c2.get(m, 0) for m, n in c1.items())


def brute_within(s, k):
    if len(s) < 2 * k:
        return 0
    c = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    rc = Counter(revcomp(s)[i:i + k] for i in range(len(s) - k + 1))
    total = 0
    for m, n in c.items():
        total += n * rc.get(m, 0)
    return total


rows = []
for f in sorted((REPO.parent / "torusfold-cgdata" / "rsRNASP" / "Training_set").glob("*.pdb")):
    if len(rows) >= N:
        break
    for beads, pairs, names in B._chain_residues(str(f), with_names=True):
        seq = "".join(names)
        L = len(seq)
        flank = min(200, L // 4)
        for i, j in pairs:
            up = seq[max(0, i - flank):i]
            dn = seq[j:min(L, j + flank)]
            if len(up) < 5 or len(dn) < 5:
                continue
            r = compute_rcm_score(up, dn)
            b_cross = sum(brute_crossing(up, dn, k) for k in K)
            b_up = sum(brute_within(up, k) for k in K)
            b_dn = sum(brute_within(dn, k) for k in K)
            rows.append((flank, len(up), len(dn), r["crossing_total"], b_cross,
                         r["within_up_total"] + r["within_down_total"], b_up + b_dn,
                         r["confidence"]))
            if len(rows) >= N:
                break
        if len(rows) >= N:
            break

a = np.array(rows, dtype=float)
print(f"{len(a)} true pairs from the deposited structures")
print(f"flank window length: min {a[:,0].min():.0f}  median {np.median(a[:,0]):.0f}  "
      f"max {a[:,0].max():.0f}")
print()
print("=== does the prefilter lose matches the validation would accept? ===")
print(f"{'k':>3s} {'code crossing':>15s} {'brute crossing':>16s} {'lost':>8s}")
print("-" * 46)
tot_code = tot_brute = 0
for k in K:
    c = b = 0
    for f in sorted((REPO.parent / "torusfold-cgdata" / "rsRNASP" / "Training_set").glob("*.pdb")):
        for beads, pairs, names in B._chain_residues(str(f), with_names=True):
            seq = "".join(names)
            L = len(seq); flank = min(200, L // 4)
            for i, j in pairs:
                up = seq[max(0, i - flank):i]; dn = seq[j:min(L, j + flank)]
                if len(up) < 5 or len(dn) < 5:
                    continue
                c += rcm_crossing(up, dn, k, 0)[0]
                b += brute_crossing(up, dn, k)
    tot_code += c; tot_brute += b
    print(f"{k:3d} {c:15d} {b:16d} {b - c:8d}")
print(f"{'all':>3s} {tot_code:15d} {tot_brute:16d} {tot_brute - tot_code:8d}")
print()

print("=== what makes confidence zero ===")
zc = int((a[:, 3] == 0).sum())
print(f"pairs with code crossing_total == 0 : {zc}/{len(a)} = {100 * zc / len(a):.1f} percent")
print(f"pairs with BRUTE crossing == 0      : "
      f"{int((a[:,4] == 0).sum())}/{len(a)} = "
      f"{100 * (a[:,4] == 0).mean():.1f} percent")
print()
print(f"{'':22s} {'median':>10s} {'mean':>10s} {'p95':>10s} {'max':>10s}")
print("-" * 66)
for col, name in ((3, "code crossing"), (4, "brute crossing"),
                  (5, "code within"), (6, "brute within"), (7, "confidence")):
    v = a[:, col]
    print(f"{name:22s} {np.median(v):10.1f} {v.mean():10.1f} "
          f"{np.percentile(v, 95):10.1f} {v.max():10.1f}")
print()
print("=== if the denominator were only the crossing term ===")
denom = a[:, 4] + a[:, 6]
with np.errstate(invalid="ignore", divide="ignore"):
    alt = np.where(denom > 0, a[:, 4] / denom, 0.0)
print(f"brute-only confidence: median {np.median(alt):.3f}, "
      f"mean {alt.mean():.3f}, fraction zero {100 * (alt == 0).mean():.1f} percent")
print()
print("=== flank length vs crossing ===")
print(f"{'window':>8s} {'pairs':>7s} {'brute cross median':>20s} {'zero frac':>10s}")
print("-" * 50)
for lo, hi in ((5, 9), (10, 14), (15, 24), (25, 49), (50, 200)):
    m = (a[:, 0] >= lo) & (a[:, 0] <= hi)
    if m.sum():
        print(f"{lo:4d}-{hi:<3d} {int(m.sum()):7d} {np.median(a[m, 4]):20.1f} "
              f"{100 * (a[m, 4] == 0).mean():9.1f}%")
