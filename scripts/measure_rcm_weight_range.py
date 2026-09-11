"""What range does the pipeline's pair_w actually take?

isrnaclong.py:773 replaces every pair weight with compute_rcm_score(...)['confidence'],
discarding the 1.0 / 0.8 / 0.6 / BPP values that were built up over lines 738-760. And
rcm.py:221 defines that confidence as

    crossing / (crossing + within_up + within_down)

so a pair surrounded by internal repeats scores near zero and its restraint would be scaled
to near zero with it. Every measurement in this arc used pair_w = ones, so the range has to
be known before any of it can be called representative.

This calls the real function on real RNA sequences -- the chains from the training set,
split at a range of positions to imitate the BSJ flanks -- and on shuffled controls, since
a shuffled sequence destroys the crossing signal the score depends on.

Run: python scripts/measure_rcm_weight_range.py [n_seqs]
"""
import collections
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from torusfold.scheme2.rcm import compute_rcm_score

import _cgdata
DATA = _cgdata.rsrnasp()
N_SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 40
FLANK = 200
SEED = 20260220


def sequences(pdb, limit=1):
    ch = collections.OrderedDict()
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname not in ("A", "C", "G", "U"):
            continue
        rec = ch.setdefault(line[21], collections.OrderedDict())
        rec[line[22:27].strip()] = rname
    return ["".join(v.values()) for v in ch.values() if len(v) >= 80][:limit]


seqs = []
for f in sorted(DATA.glob("*.pdb")):
    if len(seqs) >= N_SEQ:
        break
    seqs.extend(sequences(f, limit=1))
print(f"{len(seqs)} sequences, lengths {min(map(len, seqs))}-{max(map(len, seqs))}")
print()

rng = random.Random(SEED)


def weights(seq, shuf=False):
    s = list(seq)
    if shuf:
        rng.shuffle(s)
        s = "".join(s)
    else:
        s = seq
    n = len(s)
    out = []
    for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
        cut = int(n * frac)
        up = s[max(0, cut - FLANK):cut]
        down = s[cut:min(n, cut + FLANK)]
        if len(up) < 20 or len(down) < 20:
            continue
        out.append(compute_rcm_score(up, down)["confidence"])
    return out


real, shuf = [], []
for s in seqs:
    real.extend(weights(s, False))
    shuf.extend(weights(s, True))
real = np.array(real)
shuf = np.array(shuf)

print(f"{'':16s} {'n':>6s} {'mean':>8s} {'sd':>8s} {'min':>8s} {'p25':>8s} "
      f"{'median':>8s} {'p75':>8s} {'max':>8s}")
print("-" * 80)
for name, v in (("real sequence", real), ("shuffled", shuf)):
    print(f"{name:16s} {len(v):6d} {v.mean():8.3f} {v.std():8.3f} {v.min():8.3f} "
          f"{np.percentile(v,25):8.3f} {np.median(v):8.3f} {np.percentile(v,75):8.3f} "
          f"{v.max():8.3f}")
print()
print("the pipeline multiplies the WC pair term and the bpp term by this value.")
for thr in (0.9, 0.5, 0.2, 0.1, 0.05):
    print(f"  fraction of real weights below {thr:.2f}: {(real < thr).mean()*100:5.1f}%")
print()
print("if a large share sit near zero, those pairs carry almost no restraint, and a test")
print("run with pair_w = ones is not a conservative approximation of the pipeline.")
