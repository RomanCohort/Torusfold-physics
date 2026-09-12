"""K_PAIR sweep: does lowering it pull the joint IBI residual back?

The bracket on K_PAIR is [204.8, 2173.9] and nothing inside it is measured
(docs/statistical_potentials_as_forces.md). A sweep is how the inside gets a number, and the
number it creates is "which value minimises the joint residual against the reference bonded
marginals".

This does not re-implement the sampler. It runs scripts/ibi_round0.py through runpy with
torusfold.scheme2.torch_cgsim.K_PAIR patched, which is safe because every use site reads the
module global at call time (lines 735, 853, 1222, 1609, 1999) rather than binding it as a
default argument. The patched value appears in ibi_round0's fingerprint line, so each log
records the field it actually ran against.

Seeds matter here and the single seed was the first version's mistake. A 16 ps run of ONE
27-residue chain has 2-4 percent of seed-to-seed spread in the joint metric, which is the
same size as the effect a K_PAIR change produces -- so a single-seed sweep cannot tell a
0.29 from a 0.30, and reporting one would be reporting noise. Each k is therefore run over
several seeds and reported as mean +- spread. ibi_round0 hardcodes its seed, so the bootstrap
below replaces torch.manual_seed before the module runs.

The joint metric is mean over the six bonded coordinates of |ln(sim/ref)|. ibi_round0 prints
the per-coordinate sim/ref but not the mean, so it is recomputed here from that table.

Run: python scripts/scan_k_pair.py [k ...]
     SCAN_SEEDS="20260218 7 12345" python scripts/scan_k_pair.py 600 340
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")
ROW = re.compile(r"^(" + "|".join(COORDS) + r")\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")

BOOT = """
import runpy, sys, torch
_real = torch.manual_seed
torch.manual_seed = lambda s, _r=_real, _o=%(seed)r: _r(_o)
sys.path.insert(0, r'%(src)s')
import torusfold.scheme2.torch_cgsim as C
C.K_PAIR = %(k)r
sys.argv = ['ibi_round0.py', '%(nrep)s', '%(nsteps)s', '0', '0.1', '25', '%(burn)s',
            '--blocks=%(blocks)s']
runpy.run_path(r'%(script)s', run_name='__main__')
"""

# "  block J spread: 0.0908 to 0.1117  (23.1 percent of the lowest)"
SPREAD = re.compile(r"block J spread:\s*([\d.]+) to ([\d.]+)\s*\(([\d.]+) percent")


def joint_from(log):
    """mean |ln(sim/ref)| over the six rows, straight out of the printed table."""
    ratios = {}
    for line in log.splitlines():
        m = ROW.match(line)
        if m:
            ratios[m.group(1)] = float(m.group(5))
    missing = [c for c in COORDS if c not in ratios]
    if missing:
        raise RuntimeError(f"no sim/ref row for {missing}; the table format moved")
    return sum(abs(math.log(v)) for v in ratios.values()) / len(ratios), ratios


def spread_from(log):
    """The block J spread, which is the ruler the difference has to beat (handoff section 7):
    a K_PAIR effect smaller than the block-to-block spread of a single arm is not a measurement.
    """
    m = SPREAD.search(log)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def run_one(k, seed, out, nrep="8", nsteps="8000", burn="0", blocks="8"):
    boot = BOOT % {"seed": seed, "src": REPO / "src", "k": k,
                   "nrep": nrep, "nsteps": nsteps, "burn": burn, "blocks": blocks,
                   "script": REPO / "scripts" / "ibi_round0.py"}
    log = subprocess.run([sys.executable, "-c", boot],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    tag = f"k{int(k)}_s{seed}_N{nsteps}_b{burn}_B{blocks}"
    (out / f"scan_{tag}.log").write_text(log.stdout + log.stderr, encoding="utf-8")
    if log.returncode != 0:
        print(f"  K_PAIR={k} seed={seed} FAILED, see results/scan_{tag}.log")
        print(log.stderr[-1500:])
        return None, None, None
    j, ratios = joint_from(log.stdout)
    return j, ratios, spread_from(log.stdout)


def main():
    ks = [float(a) for a in sys.argv[1:]] or [600.0, 470.0, 400.0, 340.0]
    seeds = [int(s) for s in os.environ.get("SCAN_SEEDS", "20260218").split()]
    nsteps = os.environ.get("SCAN_NSTEPS", "8000")
    burn = os.environ.get("SCAN_BURN", "0")
    blocks = os.environ.get("SCAN_BLOCKS", "8")
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    print(f"protocol: {nsteps} steps, burn {burn}"
          f"{' (NSTEPS//5)' if burn == '0' else ''}, "
          f"window {'default' if burn == '0' else str(int(burn) * 0.002) + ' ps'} onward, "
          f"{blocks} blocks")

    per_k = {}
    for k in ks:
        vals, spreads = [], []
        for seed in seeds:
            print(f"\n{'=' * 72}\nK_PAIR = {k}   seed = {seed}\n{'=' * 72}", flush=True)
            j, ratios, sp = run_one(k, seed, out, nsteps=nsteps, burn=burn, blocks=blocks)
            if j is None:
                continue
            vals.append(j)
            print(f"  {'  '.join(f'{c}={ratios[c]:.3f}' for c in COORDS)}")
            print(f"  joint mean|ln(sim/ref)| = {j:.4f}"
                  + (f"   block spread {sp[2]:.1f}% ({sp[0]:.4f} to {sp[1]:.4f})" if sp else ""))
            if sp:
                spreads.append(sp)
        if vals:
            per_k[k] = (vals, spreads)

    print(f"\n{'=' * 72}\nsummary\n{'=' * 72}")
    print(f"{'K_PAIR':>8s} {'n':>3s} {'mean joint':>11s} {'min':>8s} {'max':>8s} "
          f"{'seed spread':>12s} {'block spread':>13s} {'vs 600':>9s}")
    base = per_k.get(600.0, (None,))[0]
    base_mean = sum(base) / len(base) if base else None
    for k in ks:
        if k not in per_k:
            continue
        vals, spreads = per_k[k]
        mean = sum(vals) / len(vals)
        delta = f"{(base_mean - mean) / base_mean * 100:+.2f}%" if base_mean else "   --"
        bs = f"{max(s[2] for s in spreads):.1f}%" if spreads else "  --"
        print(f"{k:8.0f} {len(vals):3d} {mean:11.4f} {min(vals):8.4f} {max(vals):8.4f} "
              f"{max(vals) - min(vals):12.4f} {bs:>13s} {delta:>9s}")
    if base and len(base) > 1:
        print(f"\nseed-to-seed spread at the baseline K_PAIR alone: "
              f"{max(base) - min(base):.4f} ({max(base) / min(base) - 1:.1%} of the smaller value).")
    print()
    print("Handoff section 7: the K_PAIR difference has to beat the BLOCK spread, not just the seed")
    print("spread. A single-arm block spread of tens of percent means the window is a mixture, and")
    print("no difference measured on it is an equilibrium difference -- run E1/E1b first.")


if __name__ == "__main__":
    main()
