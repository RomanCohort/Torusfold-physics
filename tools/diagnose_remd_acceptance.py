"""Why is the REMD temperature-swap acceptance exactly zero?

What is already known:
  * torch_gpu_refine.py:780-784 builds BatchedREMD2D with t_lo=300.0, t_hi=1000.0
    hard-coded, and n_t = remd_n_replicas // n_lam.
  * The log printed T-acc=['0/8','0/24',...] -- denominators stepping by 8, so
    n_t = 8 temperature rungs, and every one of 8 attempted swaps was rejected
    at each exchange interval. Not a rounding-down of a small rate: 0 acceptances.
  * The Metropolis criterion in rest2_remd_2d.py is delta = (beta_a - beta_b) *
    (E_a - E_b), which is textbook, and the file records a fix for a previous
    sign/logic error there.

So either the ladder is too coarse for the criterion to ever pass, or the
energies being compared are not what the criterion assumes. This script answers
which by (a) reading the ladder the code actually builds, (b) looking for saved
per-replica energies, and (c) computing what acceptance a ladder with these
spacings can support.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
SRC = REPO / "src" / "torusfold" / "scheme2"
RUN = REPO / "results" / "immuno_full"

KB_KJ = 0.008314462618


def beta(t: float) -> float:
    return 1.0 / (KB_KJ * t)


# ---------------------------------------------------------------------------
print("=" * 78)
print("1. the ladder the code builds")
print("=" * 78)

text = (SRC / "rest2_remd_2d.py").read_text(encoding="utf-8").splitlines()
for i, l in enumerate(text, 1):
    if "temps" in l and ("linspace" in l or "=" in l) and 60 < i < 140:
        print(f"  rest2_remd_2d.py:{i}: {l.strip()[:110]}")
print()
text2 = (SRC / "torch_cgsim.py").read_text(encoding="utf-8").splitlines()
for i, l in enumerate(text2, 1):
    if "temp" in l.lower() and ("linspace" in l or "t_lo" in l or "t_hi" in l):
        print(f"  torch_cgsim.py:{i}: {l.strip()[:110]}")

# Reproduce the ladder both plausible ways and show the spacing.
for n_t in (8, 16):
    lin = np.linspace(300.0, 1000.0, n_t)
    geo = 300.0 * (1000.0 / 300.0) ** (np.arange(n_t) / max(1, n_t - 1))
    print()
    print(f"  n_t={n_t}")
    print(f"    linspace : {np.round(lin, 1).tolist()}")
    print(f"    geometric: {np.round(geo, 1).tolist()}")
    d = np.diff(lin) / lin[:-1]
    print(f"    linspace neighbour spacing: {np.round(d * 100, 1).tolist()} %")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("2. saved diagnostics from the run")
print("=" * 78)
found = False
for p in sorted(RUN.rglob("*.json")):
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    s = json.dumps(j)
    if "acceptance" in s or "accT" in s or "temperatures" in s:
        print(f"  {p.relative_to(RUN)}")
        for k in j:
            print(f"    {k}: {str(j[k])[:120]}")
        found = True
if not found:
    print("  no saved acceptance/temperature diagnostics in the run directory")
    print("  (they exist only in memory, in the diagnostics dict)")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("3. what acceptance can this ladder support?")
print("=" * 78)
print("""
  The Metropolis exponent is d = (beta_a - beta_b) * (E_a - E_b). A swap is
  accepted when d >= 0, or with probability exp(d) otherwise. For a rough
  estimate of the pass rate we need the spread of E between neighbouring rungs.

  The run's own energy readout gives a scale. From the log, E_min across the
  10-block relaxation fell from 5456 to ~1900, and the REMD finished at E=683
  against a rounded 2000 at round 1. Take a conservative per-rung energy
  difference of a few hundred kJ/mol and compute the exponent.
""")
for dE in (100.0, 300.0, 500.0, 1000.0):
    for n_t in (8, 16, 32):
        temps = np.linspace(300.0, 1000.0, n_t)
        db = abs(beta(temps[1]) - beta(temps[0]))
        exponent = -db * dE
        print(f"  dE={dE:6.0f} kJ/mol  n_t={n_t:3d}  |dbeta|={db:.6f}  "
              f"exponent={exponent:8.3f}  acceptance={math.exp(exponent):.2e}")
    print()
