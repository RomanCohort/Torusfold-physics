"""How much room does the DivideFold tier of pair_w actually have?

isrnaclong.py builds pair_w from method agreement in a fixed order: hard (1.0) first, then MFE
softs (0.8), then PF mediums (their own probability p), and only then DivideFold pairs (0.6), each
tier claiming only pairs no earlier tier claimed. DivideFold is not available on this machine, so
the 0.6 tier cannot be measured by running it. What CAN be measured is the ROOM it has: the share
of the database's true pairs that the MFE and PF tiers miss entirely. Those are the only pairs the
0.6 tier could ever claim.

If that share is near zero, the tier is arithmetically unable to matter and the missing measurement
stops being a gap. If it is large, the tier is a real part of the channel and its absence from the
measurement in tests/../measure_pair_weight_quality.py has to be stated wherever the number is used.

Same ViennaRNA settings and tier boundaries as the pipeline: md.circ = 1, P > 0.9 high,
0.5 < P <= 0.9 medium.

Run: python scripts/measure_dividefold_tier_room.py [n_files]
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B   # noqa: E402

N_FILES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
import RNA                     # noqa: E402

chains = []
for f in sorted(B.DATA.glob("*.pdb"))[:N_FILES]:
    for rec in B._chain_residues(str(f), with_names=True):
        if rec[1]:
            chains.append((f.stem, rec[2], rec[1]))
print(f"{len(chains)} chains with at least one accepted WC pair, "
      f"{sum(len(p) for _, _, p in chains)} pairs")

tiers = {"hard (1.0)": 0, "mfe soft (0.8)": 0, "pf medium (p)": 0, "NONE -> the 0.6 tier": 0}
total = 0
for name, names, pairs in chains:
    seq = "".join(names)
    md = RNA.md()
    md.circ = 1
    fc = RNA.fold_compound(seq, md)
    fc.pf()
    plist = fc.plist_from_probs(0.01)
    pf_high = {(ep.i - 1, ep.j - 1) for ep in plist if ep.p > 0.9}
    pf_mid = {(ep.i - 1, ep.j - 1) for ep in plist if 0.5 < ep.p <= 0.9}
    ss, _ = fc.mfe()
    stack, mfe = [], set()
    for k, c in enumerate(ss):
        if c == "(":
            stack.append(k)
        elif c == ")" and stack:
            mfe.add((stack.pop(), k))
    votes = {}
    for src in (pf_high, mfe):
        for p in src:
            votes[p] = votes.get(p, 0) + 1
    hard = {p for p, v in votes.items() if v >= 2} | set(pf_high)
    seen = set(hard)
    soft = set()
    for p in mfe:
        if p not in seen:
            soft.add(p)
            seen.add(p)
    mid = {p for p in pf_mid if p not in seen}

    for p in pairs:
        pp = (min(p), max(p))
        total += 1
        if pp in hard:
            tiers["hard (1.0)"] += 1
        elif pp in soft:
            tiers["mfe soft (0.8)"] += 1
        elif pp in mid:
            tiers["pf medium (p)"] += 1
        else:
            tiers["NONE -> the 0.6 tier"] += 1

print()
print(f"{'tier':24s} {'true pairs':>11s} {'share':>9s}")
print("-" * 46)
for k, v in tiers.items():
    print(f"{k:24s} {v:11d} {100.0 * v / total:8.2f}%")
print()
room = tiers["NONE -> the 0.6 tier"]
print(f"The 0.6 tier's room is {room} of {total} true pairs ({100.0 * room / total:.2f}%). Only those")
print("pairs could ever receive 0.6, and they receive it only if DivideFold predicts them.")
print()
if room / total < 0.02:
    print("That is small enough that the tier cannot move the channel much, so its absence from the")
    print("measured AUC is a footnote rather than a hole.")
else:
    print("That is NOT small: the tier could claim a real share of the true pairs, so every number")
    print("quoted from scripts/measure_pair_weight_quality.py has to say that the 0.6 tier is")
    print("missing from it and that the missing piece is this large.")
