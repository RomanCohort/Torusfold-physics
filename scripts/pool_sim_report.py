"""Pool the saved per-chain bonded moments and compare like-for-like against pooled references.

pool_sim_bonded.py samples each chain and saves the per-coordinate (sum, sum-of-squares, count)
moments. This script consumes those npz files and answers the question the single-chain run left
open: does scoring ONE simulated chain against the 126-chain pooled reference (the ibi_round0.py
comparison) change once the simulation side is itself pooled over several chains and the
reference side uses the same pooling口径?

Three reference denominators are put side by side, because they are different quantities and the
whole point is to see how much the conclusion moves between them:

    R_table  the npz __sigma the sampler scores against (refit_tables_clean.py, 96 chains,
             lengths 20-120). This is the field's own calibration target: k = kBT/sigma^2.
    R_126    the concatenated sd over all 126 deposited chains -- the "126-chain pool" the
             question literally names. decompose_bonded_spread.py's var_pooled.
    R_7      the concatenated sd over the 7 chains that pass the pool filter and actually get
             simulated (len(pairs) >= 8, 24 <= L <= 34). This is the same population the sim is
             drawn from, so S_pool / R_7 is the like-for-like comparison.

A length-binned reference table is also printed, because several coordinates have per-chain
sample counts that scale with chain length, and the 7 simulated chains are all short (24-34)
against a database spanning 20-120.

Run: python scripts/pool_sim_report.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"


def pooled_sd(structs):
    """(sd, n) of the concatenation of one coordinate's observations over all given chains."""
    out = {}
    vals = {c: [] for c in B.COORDS}
    for s in structs:
        pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
        for c in B.COORDS:
            vals[c].append(B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64))
    for c in B.COORDS:
        v = np.concatenate(vals[c])
        out[c] = (float(v.std()), int(v.size))
    return out


def moments_from_npz(path):
    z = np.load(path)
    acc = {}
    for c in B.COORDS:
        acc[c] = (float(z[f"{c}__sum"]), float(z[f"{c}__sumsq"]), int(z[f"{c}__n"]))
    meta = {k[len("meta__"):]: z[k].tolist() for k in z.files if k.startswith("meta__")}
    return acc, meta


def sd_from_acc(acc):
    s1, s2, n = acc
    m = s1 / n
    return float(np.sqrt(max(s2 / n - m * m, 0.0)))


# ---- reference side ----------------------------------------------------------------------
z = np.load(NPZ)
R_table = {c: float(z[f"{c}__sigma"]) for c in B.COORDS}

all_structs = B.load_structures()
pool7 = [s for s in all_structs if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
R_126 = {c: pooled_sd(all_structs)[c][0] for c in B.COORDS}
R_7 = {c: pooled_sd(pool7)[c][0] for c in B.COORDS}

# length bins: the simulated chains are 24-34; the database spans 20-120
def lens(structs):
    return [len(s["pos"]) for s in structs]


short = [s for s in all_structs if len(s["pos"]) <= 40]
long_ = [s for s in all_structs if len(s["pos"]) > 40]
R_short = {c: pooled_sd(short)[c][0] for c in B.COORDS}
R_long = {c: pooled_sd(long_)[c][0] for c in B.COORDS}

print(f"{len(all_structs)} chains in database; {len(pool7)} pass the sim filter "
      f"(L {min(lens(pool7))}-{max(lens(pool7))}); {len(short)} short (<=40), {len(long_)} long (>40)")
print()

# ---- simulation side ----------------------------------------------------------------------
files = sorted((REPO / "results").glob("pooled_sim_*.npz"))


def _tag_of(f):
    stem = f.stem
    body = stem[len("pooled_sim_"):]
    return body[:body.rfind("_idx")]


tags = sorted({_tag_of(f) for f in files}) if files else []
print(f"found {len(files)} sample files; tags: {tags}")
print()

for tag in tags:
    tag_files = [f for f in files if _tag_of(f) == tag]
    idxs = sorted(int(f.stem.split("idx")[-1]) for f in tag_files)
    # per-chain moments, keyed by idx
    chains = {}
    for f in tag_files:
        idx = int(f.stem.split("idx")[-1])
        acc, meta = moments_from_npz(f)
        chains[idx] = (acc, meta)
    if not chains:
        continue
    n_chains = len(chains)
    print(f"{'=' * 90}\nconfig [{tag}]: {n_chains} chains (idx {idxs})\n{'=' * 90}")

    # pooled sim moments: sum over chains of sum/sumsq/n
    S_single = {c: [] for c in B.COORDS}
    S_pool = {}
    S_within = {}
    for c in B.COORDS:
        s1 = sum(chains[i][0][c][0] for i in chains)
        s2 = sum(chains[i][0][c][1] for i in chains)
        n = sum(chains[i][0][c][2] for i in chains)
        S_pool[c] = sd_from_acc((s1, s2, n))
        # within-chain: pooled per-chain variance = sum_i (sumsq_i - sum_i^2/n_i) / sum_i n_i
        sw2 = sum(chains[i][0][c][1] - chains[i][0][c][0] ** 2 / chains[i][0][c][2]
                  for i in chains)
        S_within[c] = float(np.sqrt(max(sw2 / n, 0.0)))
        for i in sorted(chains):
            S_single[c].append((i, sd_from_acc(chains[i][0][c])))

    # per-chain sim/ref against the table (the E2 comparison), for the record
    print(f"per-chain sim/ref vs R_table (the ibi_round0/E2 metric):")
    hdr = "  idx " + " ".join(f"{i:>7d}" for i in sorted(chains))
    print(hdr)
    for c in B.COORDS:
        row = "  ".join(f"{sd / R_table[c]:7.3f}" for _, sd in S_single[c])
        print(f"  {c:10s} {row}")

    print()
    print(f"{'coordinate':10s} {'R_table':>8s} {'R_126':>8s} {'R_7':>8s} {'S_single*':>9s} "
          f"{'S_within':>9s} {'S_pool':>8s} {'within%':>8s}")
    print("-" * 76)
    for c in B.COORDS:
        ss = np.mean([sd for _, sd in S_single[c]])
        within_pct = 100 * (S_within[c] ** 2) / (S_pool[c] ** 2)
        print(f"{c:10s} {R_table[c]:8.4f} {R_126[c]:8.4f} {R_7[c]:8.4f} {ss:9.4f} "
              f"{S_within[c]:9.4f} {S_pool[c]:8.4f} {within_pct:7.1f}%")

    print()
    print(f"sim/ref under each pairing (S_pool / R_*):")
    print(f"{'coordinate':10s} {'/R_table':>9s} {'/R_126':>9s} {'/R_7':>9s} "
          f"{'mean S_i/R_table':>17s} {'pool vs single':>15s} {'pool vs R7':>13s}")
    print("-" * 88)
    for c in B.COORDS:
        ss_mean = np.mean([sd / R_table[c] for _, sd in S_single[c]])
        pt = S_pool[c] / R_table[c]
        p7 = S_pool[c] / R_7[c]
        print(f"{c:10s} {pt:9.3f} {S_pool[c] / R_126[c]:9.3f} {p7:9.3f} "
              f"{ss_mean:17.3f} {pt - ss_mean:+15.3f} {p7 - pt:+13.3f}")
    print()
    print("  'within%' is the share of the pooled sim variance that is within-chain; 100% would")
    print("  mean pooling the chains changed nothing (between-chain sim spread = 0).")
    print("  'pool vs single' = S_pool/R_table - mean(S_i/R_table): how much pooling the sim")
    print("  alone moves the number. 'pool vs R7' = S_pool/R_7 - S_pool/R_table: how much switching")
    print("  the reference from the whole database to the same 7 structures moves it.")
    print()

# ---- length-binned reference -----------------------------------------------------------------
print("=" * 90)
print("reference sd by chain length: the simulated chains are all short, the table targets all")
print("=" * 90)
print(f"{'coordinate':10s} {'R_table':>8s} {'short<=40':>10s} {'long>40':>10s} "
      f"{'short/long':>11s} {'R_7 (24-34)':>12s}")
print("-" * 70)
for c in B.COORDS:
    print(f"{c:10s} {R_table[c]:8.4f} {R_short[c]:10.4f} {R_long[c]:10.4f} "
          f"{R_short[c] / R_long[c]:11.3f} {R_7[c]:12.4f}")
print()
print("short/long != 1 means the coordinate's native spread depends on chain length. Where it is")
print("far from 1, a field calibrated on the whole database cannot be expected to reproduce a")
print("short-chain simulation to sim/ref = 1 against its own short-chain native population.")
