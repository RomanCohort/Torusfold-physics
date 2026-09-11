"""Can this database supply a thermal width for the base-pair restraint, or only a lower bound?

tests/test_pair_clash_bsj_criterion.py says K_PAIR's lower bound is kBT/sigma_NN^2 = 204.8 and that
it is a BOUND because "each entry is one conformation, so the pooled spread mixes sequence and
conformer variation into the thermal width". That is an argument. This is the measurement.

One-way decomposition of the pooled spread of a coordinate over deposited chains:

    SS_total = SS_between + SS_within

with MS_within the mean squared deviation of a coordinate from its own chain's mean, and
MS_between the spread of chain means. If nearly all of the pooled variance is BETWEEN chains, then
it is structure-to-structure and sequence variation; if nearly all is WITHIN chains, then it is
residue-to-residue variation inside one molecule -- either way NOT thermal, because each deposited
file is one conformation. A thermal width would need several conformers of one sequence, which this
database does not have.

Run for the base-pair N-N distance, which is what K_PAIR restrains, and for the backbone P-P bond
as the control, since refit_tables_clean.py already reports 2.6 to 10.5 percent between structures
there.

Run: python scripts/decompose_pair_spread.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B   # noqa: E402

KBT = B.KBT if hasattr(B, "KBT") else 2.494


def one_way(groups):
    """(MS_within, MS_between, n_eff) for a list of per-chain arrays."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(g.size for g in groups)
    if k < 2 or n <= k:
        return float("nan"), float("nan"), 0
    grand = np.concatenate(groups).mean()
    ss_within = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ms_within = ss_within / (n - k)
    ms_between = ss_between / (k - 1)
    n_eff = (n - sum(g.size ** 2 for g in groups) / n) / (k - 1)
    return ms_within, ms_between, n_eff


def report(tag, groups, pooled_vals):
    v = np.concatenate(groups)
    ms_w, ms_b, n_eff = one_way(groups)
    sd_within = float(np.sqrt(ms_w))
    var_between = max((ms_b - ms_w) / n_eff, 0.0) if n_eff else float("nan")
    sd_between = float(np.sqrt(var_between))
    sd_pooled = float(v.std())
    share = (sd_between ** 2) / (sd_pooled ** 2) if sd_pooled else float("nan")
    print(f"=== {tag} ===")
    print(f"  observations {v.size} over {len(groups)} chains "
          f"(per chain min {min(len(g) for g in groups)}, median "
          f"{int(np.median([len(g) for g in groups]))}, max {max(len(g) for g in groups)})")
    print(f"  pooled  mean {v.mean():.4f}  sd {sd_pooled:.4f}")
    print(f"  within-chain sd (MS_within^0.5)        {sd_within:.4f}")
    print(f"  between-chain sd (variance components) {sd_between:.4f}   n_eff {n_eff:.2f}")
    print(f"  share of the pooled VARIANCE that is between chains: {100 * share:.2f}%")
    print(f"  k = kBT/sd^2   pooled {KBT / sd_pooled ** 2:9.1f}   "
          f"within-chain {KBT / sd_within ** 2:9.1f}   "
          f"between-only {KBT / var_between if var_between > 0 else float('inf'):9.1f}")
    print(f"  force floor sqrt(k*kBT)  pooled {np.sqrt(KBT / sd_pooled ** 2 * KBT):8.1f}   "
          f"within-chain {np.sqrt(KBT / sd_within ** 2 * KBT):8.1f}")
    print()
    return sd_pooled, sd_within, sd_between


structs = B.load_structures()
pair_groups = []
bond_groups = []
for s in structs:
    pos = s["pos"]                      # (L, 3, 3): P, C4', N9/N1
    if pos.shape[0] < 4:
        continue
    if s["pairs"]:
        n = pos[:, 2, :]
        pair_groups.append(np.array([float(np.linalg.norm(n[i] - n[j])) for i, j in s["pairs"]]))
    p = pos[:, 0, :]
    bond_groups.append(np.array([float(np.linalg.norm(p[i] - p[i + 1]))
                                 for i in range(pos.shape[0] - 1)]))

pp = report("base-pair N-N distance (what K_PAIR restrains)", pair_groups, None)
bb = report("backbone P(i)-P(i+1) bond (the control)", bond_groups, None)
print("shipped K_PAIR = 600.0, PAIR_NN = 1.0 nm")
print()
print("Read the 'share between chains' line. Both coordinates share the same limit: a deposited")
print("file is ONE conformation, so whichever component dominates, neither is a thermal width. The")
print("pooled sd is therefore an upper bound on the thermal spread, and kBT/sd^2 a LOWER bound on")
print("k -- exactly what the criterion test already claims, now measured rather than argued.")
print()
print("The control is the point: refit_tables_clean.py reports 2.6 to 10.5 percent between")
print("structures for the bonded coordinates. If the base-pair coordinate comes out in the same")
print("range, the two are limited the same way and there is no way to raise K_PAIR's lower bound")
print("from this data. If it comes out much HIGHER, then the base-pair spread is dominated by")
print("structure and sequence variation, the lower bound is correspondingly weaker, and 600 has")
print("more room to be justified -- but still no measurement that picks it.")
