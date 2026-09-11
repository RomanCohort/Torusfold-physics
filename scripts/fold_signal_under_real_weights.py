"""Does the fold signal survive the pipeline's actual pair weights?

3u measured that the WC pair term and bpp carry the entire fold signal, 24/24 and 23/24, and
3v that under register-shift decoys only bpp does, 111/111. Both used pair_w = ones.

measure_rcm_weight_range.py then showed the pipeline does not use ones: isrnaclong.py:773
replaces every weight with compute_rcm_score confidence, and over real sequences that gives
a median of 0.25 with about 40 percent of values at or near zero. Since both pair terms are
scaled by that weight, the pipeline runs with most of its pairing restraint switched off.

This reruns the fold-signal test under three weight regimes on the same structures and the
same decoys, so the only thing that changes is the weight.

Run: python scripts/fold_signal_under_real_weights.py [n_structs]
"""
import collections
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import cg_force_terms as FT
import torusfold.scheme2.torch_cgsim as C
from torusfold.scheme2.rcm import compute_rcm_score

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
MIN_PAIRS = 6
SEED = 20260221
CACHE = Path(__file__).resolve().parent.parent / "results" / "rcm_weights.npz"
import _cgdata
DATA = _cgdata.rsrnasp()
FLANK = 200


def rcm_weights():
    """Empirical weight sample, one value per cut position per sequence."""
    if CACHE.exists():
        return np.load(CACHE)["w"]
    ch_all = []
    for f in sorted(DATA.glob("*.pdb")):
        ch = collections.OrderedDict()
        for line in open(f):
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            if line[17:20].strip() not in ("A", "C", "G", "U"):
                continue
            ch.setdefault(line[21], collections.OrderedDict())[line[22:27].strip()] = \
                line[17:20].strip()
        for v in ch.values():
            if len(v) >= 80:
                ch_all.append("".join(v.values()))
        if len(ch_all) >= 40:
            break
    out = []
    for s in ch_all:
        n = len(s)
        for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
            cut = int(n * frac)
            up, down = s[max(0, cut - FLANK):cut], s[cut:min(n, cut + FLANK)]
            if len(up) >= 20 and len(down) >= 20:
                out.append(compute_rcm_score(up, down)["confidence"])
    w = np.asarray(out, dtype=np.float64)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, w=w)
    return w


W = rcm_weights()
print(f"empirical RCM weights: n={len(W)}, median {np.median(W):.3f}, "
      f"{(W < 0.05).mean()*100:.1f} percent below 0.05")
print()

structs = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= MIN_PAIRS][:N]
rng = np.random.default_rng(SEED)
regimes = {
    "ones (what 3u/3v used)": None,
    "all at the median 0.25": np.full(9999, 0.25),
    "drawn from the real sample": W,
}
print(f"{len(structs)} structures with at least {MIN_PAIRS} pairs")
print()


def derange(pairs, gen):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    for _ in range(200):
        perm = gen.permutation(len(b))
        if all(perm[i] != i for i in range(len(b))):
            break
    out = []
    for i, j in enumerate(perm):
        lo, hi = (a[i], b[j]) if a[i] < b[j] else (b[j], a[i])
        out.append((lo, hi))
    return out


def energies(s, pairs, w):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    pw = torch.tensor(w, dtype=torch.float64) if w is not None else None
    _, e = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    return e


print(f"{'regime':30s} {'pair delta':>12s} {'pair wins':>11s} {'bpp delta':>12s} "
      f"{'bpp wins':>10s} {'tog. wins':>10s}")
print("-" * 92)
for label, sample in regimes.items():
    gen = np.random.default_rng(SEED + 1)
    pv, bv, wins, n = [], [], 0, 0
    for s in structs:
        npr = len(s["pairs"])
        if sample is None:
            w_nat = np.ones(npr)
        else:
            w_nat = np.asarray(sample)[gen.integers(0, len(sample), npr)]
        d = derange(s["pairs"], gen)
        w_dec = w_nat[:len(d)]
        e_nat = energies(s, s["pairs"], w_nat)
        e_dec = energies(s, d, w_dec)
        pv.append(e_dec["wc pair N-N"] - e_nat["wc pair N-N"])
        bv.append(e_dec["bpp"] - e_nat["bpp"])
        tot = (e_dec["wc pair N-N"] + e_dec["bpp"]) - (e_nat["wc pair N-N"] + e_nat["bpp"])
        wins += int(tot > 0)
        n += 1
    pv, bv = np.array(pv), np.array(bv)
    print(f"{label:30s} {pv.mean():12.1f} {int((pv>0).sum()):>7d}/{n:<4d} "
          f"{bv.mean():12.1f} {int((bv>0).sum()):>6d}/{n:<4d} {wins:>6d}/{n:<4d}")
print()
print("the decoy is the crude random re-pairing of 3u, kept because it is the one regime")
print("where the pair term was shown to respond at all. Both pair terms are scaled by the")
print("weight, so a regime with small weights weakens whatever signal they carry.")
