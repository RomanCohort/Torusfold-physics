"""Structure of a cgRNASP potential table, and how differentiable it is.

The C implementation evaluates it as potential[i][j][(int)(distance/0.3)] - a lookup by
integer bin. If the table is piecewise constant, the gradient with respect to coordinates
is zero almost everywhere, which is the whole problem for using it as a force.
"""
import collections
from pathlib import Path

import numpy as np

D = Path(r"D:\torusfold-cgdata\cgRNASP\cgRNASP\data")
FILES = {
    "0-1 (sep 1)": ("0-1_short-ranged.potential", 17),
    "1-2 (sep 2)": ("1-2_short-ranged.potential", 30),
    "2-4 (sep 3-4)": ("2-4_short-ranged.potential", 43),
    "long (sep>=5)": ("long-ranged.potential", 80),
}

for label, (fn, iv) in FILES.items():
    blocks = collections.OrderedDict()
    for line in open(D / fn):
        p = line.split()
        if len(p) < 4:
            continue
        i, j, b, e = int(p[0]), int(p[1]), int(p[2]), float(p[3])
        blocks.setdefault((i, j), {})[b] = e
    is_ = sorted({k[0] for k in blocks})
    js = sorted({k[1] for k in blocks})
    sizes = collections.Counter(len(v) for v in blocks.values())
    flat = tot = 0
    uniq = []
    for k, v in blocks.items():
        bs = sorted(v)
        arr = [v[b] for b in bs]
        for a, b in zip(arr, arr[1:]):
            tot += 1
            if a == b:
                flat += 1
        uniq.append(len(set(arr)) / len(arr))
    print(f"== {label}   file {fn}   intervals={iv}")
    print(f"   blocks (type pairs) : {len(blocks)}    i in {is_[0]}..{is_[-1]}   j in {js[0]}..{js[-1]}")
    print(f"   bins per block      : {dict(sizes)}")
    print(f"   consecutive-bin steps that are exactly flat : {flat}/{tot} = {flat / tot:.1%}")
    print(f"   distinct values per block, mean share       : {np.mean(uniq):.1%}")
    print()
