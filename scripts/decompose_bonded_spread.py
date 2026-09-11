"""Can a single chain ever reproduce the pooled reference width? The ceiling, per coordinate.

The reference marginals are pooled over 126 deposited chains. A simulation here is ONE chain. So
even a field that reproduced its own chain's distribution exactly would show sim/ref != 1 wherever
the pooled spread has a between-chain part. This script measures that part for each bonded
coordinate and turns it into the number the IBI residual has to be read against:

    ceiling = sqrt(MS_within / var_pooled)

MS_within is the variance an observation has around its OWN chain's mean. If the pooled spread is
all within-chain, the ceiling is 1 and sim/ref = 1 is reachable. If a large share is between-chain
-- structure and sequence variation no single trajectory can produce -- the ceiling is below 1, and
a sim/ref sitting AT the ceiling is not a defect of the field.

This matters now. The converged-window run (section 3ba) gives dihedral sim/ref = 0.708, which is
59 percent of the joint residual. K_PAIR already went through this question once, in
decompose_pair_spread.py, and the answer was 'it cannot be measured' rather than 'the bound is
weak'. The bonded coordinates have never been asked.

Note what angle and dihedral ARE: coords_of returns the COSINE of the angle and of the dihedral,
not the angle itself. The sigma compared below is the sigma of the cosine.

Run: python scripts/decompose_bonded_spread.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'src'))
import boltzmann_bonded as B   # noqa: E402

KBT = B.KBT if hasattr(B, 'KBT') else 2.494
NPZ = REPO / 'results' / 'boltzmann_tables_clean.npz'


def one_way(groups):
    """(MS_within, MS_between, n_eff) for a list of per-chain arrays."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(g.size for g in groups)
    if k < 2 or n <= k:
        return float('nan'), float('nan'), 0
    grand = np.concatenate(groups).mean()
    ss_within = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ms_within = ss_within / (n - k)
    ms_between = ss_between / (k - 1)
    n_eff = (n - sum(g.size ** 2 for g in groups) / n) / (k - 1)
    return ms_within, ms_between, n_eff


structs = B.load_structures()
groups = {c: [] for c in B.COORDS}
for s in structs:
    pos = torch.tensor(s['pos'].reshape(1, -1, 3), dtype=torch.float64)
    for c in B.COORDS:
        v = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
        if v.size:
            groups[c].append(v)

try:
    z = np.load(NPZ)
    tab_sigma = {c: float(z[c + '__sigma']) for c in B.COORDS}
except Exception:
    tab_sigma = {}

# sim/ref measured on the converged window, section 3ba. QUOTED, not recomputed here; the point of
# printing it is to put each coordinate next to its own ceiling in one place.
MEASURED = {'bb_bond': 1.121, 'intra_pc': 1.010, 'intra_cn': 1.000,
            'angle': 0.902, 'dihedral': 0.708, 'stack': 0.992}

print(f'{len(structs)} structures loaded, {len(B.COORDS)} bonded coordinates')
print()
hdr = ('coordinate', 'obs', 'chains', 'pooled sd', 'within sd', 'between %', 'CEILING',
       'table sd', 'measured', 'vs ceiling')
print(f'{hdr[0]:<11s} {hdr[1]:>7s} {hdr[2]:>7s} {hdr[3]:>10s} {hdr[4]:>10s} '
      f'{hdr[5]:>10s} {hdr[6]:>8s} {hdr[7]:>9s} {hdr[8]:>9s} {hdr[9]:>11s}')
print('-' * 110)
for c in B.COORDS:
    gs = groups[c]
    v = np.concatenate(gs)
    ms_w, ms_b, n_eff = one_way(gs)
    sd_pooled = float(v.std())
    sd_within = float(np.sqrt(ms_w))
    var_between = max((ms_b - ms_w) / n_eff, 0.0) if n_eff else float('nan')
    share = (var_between / sd_pooled ** 2) if sd_pooled else float('nan')
    ceil = float(np.sqrt(ms_w / sd_pooled ** 2)) if sd_pooled else float('nan')
    m = MEASURED[c]
    if abs(m - ceil) < 0.03:
        verdict = 'AT CEILING'
    elif m > ceil:
        verdict = 'above'
    else:
        verdict = 'below'
    print(f'{c:<11s} {v.size:7d} {len(gs):7d} {sd_pooled:10.4f} {sd_within:10.4f} '
          f'{100 * share:9.2f}% {ceil:8.3f} {tab_sigma.get(c, float("nan")):9.4f} '
          f'{m:9.3f} {verdict:>11s}')
print()
print('Read the CEILING column first. It is the largest sim/ref a single chain can produce even if',
      'it samples its own distribution perfectly, so a measured value AT the ceiling means the',
      'field is not the thing to fix -- the comparison against a pooled reference is. A value',
      'BELOW the ceiling is real headroom and is what IBI can act on.')
print()
print('MEASURED is quoted from section 3ba of docs/statistical_potentials_as_forces.md, not',
      'recomputed here. If that section moves, this line moves with it.')