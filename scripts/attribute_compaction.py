"""Which term compacts the chain? Ablate one group at a time and watch Rg.

Measured fact this starts from (scripts/check_fold_drift.py, 1L2X, 8 replicas):

    native Rg 1.229 nm  ->  1.061 nm by 80 ps  ->  1.055 nm at 400 ps

and the deposited pool this structure is compared against has Rg mean 1.550 +- 0.255 nm (7
chains, the same filter ibi_round0 uses; 1L2X is the most compact of them). So the simulated
chain sits about a third below the ensemble its marginals are scored against, and it does it
within the first 80 ps. The angular marginals are narrow as a consequence -- angle 0.67-0.90 and
dihedral 0.59-0.71 of reference across all five structures E2 ran.

The strongest candidate is the BSJ group, and it is not a subtle one. README, "Known limits":

    Three terms are 91.55 percent of the energy on a linear reference (bsj closure 68.17,
    bsj contact 23.46). They act on P(0)-P(L-1), which only exists in a circular molecule. A run
    on a linear chain should set all three to zero and say so in its provenance line.

The structures in this pool are LINEAR deposited chains -- the database has no covalently closed
chain at all, which is also why K_BSJ was set by transferability rather than measurement. So
`bsj closure` restrains |P(0)-P(L-1)| toward 0.590 nm while the linear chain's ends start 3.9 nm
apart on average, with K_BSJ = 1122.4. That is a several-thousand kJ/mol/nm force pulling the two
ends of an open chain together, and section 3az recorded the largest single force in the field,
5249.57, on exactly this term.

The others are ablated for completeness, not because they are equally likely: the WC pair spring
and the excluded volume both act on the interior, and the bpp term is the only sequence-dependent
one.

This does NOT re-implement the sampler. It runs scripts/check_fold_drift.py through runpy with
constants patched, the same way scan_k_pair.py patches K_PAIR, and parses the Rg column it
already prints. Each arm gets its own log, so every number can be traced to a run.

Run: python scripts/attribute_compaction.py [n_steps] [idx]
"""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NSTEPS = sys.argv[1] if len(sys.argv) > 1 else "40000"
IDX = sys.argv[2] if len(sys.argv) > 2 else "0"
BURN = "500"          # 1 ps, so the first block still sees the pre-collapse state
BLOCKS = "8"
NREP = "8"

# The first block starts at 1 ps, and by 3.2 ps Rg is already down to 1.10 nm, so a later burn
# would throw away the collapse itself and leave nothing to attribute.
ARMS = [
    ("full (as shipped)", {}),
    ("no BSJ trio", {"K_BSJ": 0.0, "K_BSJ_GUIDE": 0.0, "K_BSJ_CONTACT": 0.0}),
    ("no K_PAIR", {"K_PAIR": 0.0}),
    ("no excluded volume", {"K_CLASH": 0.0}),
    ("no K_BPP", {"K_BPP": 0.0}),
]

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)-\s*([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")
NATIVE = re.compile(r"native start: Rg = ([\d.]+) nm")

BOOT = """
import runpy, sys
sys.path.insert(0, r'%(src)s')
import torusfold.scheme2.torch_cgsim as C
for _n, _v in %(patch)r.items():
    if not hasattr(C, _n):
        raise SystemExit('no such constant: ' + _n)
    setattr(C, _n, _v)
sys.argv = ['check_fold_drift.py', '%(nrep)s', '%(nsteps)s', '%(idx)s', '0.1', '25', '%(burn)s',
            '--blocks=%(blocks)s']
runpy.run_path(r'%(script)s', run_name='__main__')
"""


def run_arm(label, patch, out):
    boot = BOOT % {"src": REPO / "src", "patch": patch, "nrep": NREP, "nsteps": NSTEPS,
                   "idx": IDX, "burn": BURN, "blocks": BLOCKS,
                   "script": REPO / "scripts" / "check_fold_drift.py"}
    p = subprocess.run([sys.executable, "-c", boot],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tag = label.replace(" ", "_").replace("(", "").replace(")", "")
    (out / f"compaction_{tag}.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        print(f"  {label}: FAILED, see results/compaction_{tag}.log")
        print(p.stderr[-1200:])
        return None
    nat = NATIVE.search(p.stdout)
    rows = [m.groups() for m in (ROW.match(l) for l in p.stdout.splitlines()) if m]
    if not rows:
        print(f"  {label}: no table rows parsed")
        return None
    return {"native": float(nat.group(1)) if nat else None,
            "rg": [float(r[3]) for r in rows],
            "angle": [float(r[8]) for r in rows],
            "stack": [float(r[9]) for r in rows],
            "window": [(float(r[1]), float(r[2])) for r in rows]}


out = REPO / "results"
out.mkdir(exist_ok=True)
results = {}


def _one(item):
    label, patch = item
    return label, run_arm(label, patch, out)


# The arms are independent runs of the same sampler, so they go out together. Sequentially this
# is five full trajectories back to back, and the whole point is to compare them.
with ThreadPoolExecutor(max_workers=len(ARMS)) as ex:
    for label, r in ex.map(_one, ARMS):
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
        if r is None:
            continue
        results[label] = r
        print(f"  native Rg {r['native']:.3f}   Rg: " + " ".join(f"{v:.3f}" for v in r["rg"]))
        print(f"  angle: " + " ".join(f"{v:.3f}" for v in r["angle"])
              + "    stack: " + " ".join(f"{v:.3f}" for v in r["stack"]))

if len(results) < 2:
    sys.exit(1)
print(f"\n{'=' * 72}\nsummary: Rg per block ({NSTEPS} steps = {int(NSTEPS) * 0.002:.0f} ps)\n{'=' * 72}")
w = next(iter(results.values()))["window"]
print(f"{'arm':>22s} {'native':>8s} {'Rg b1':>8s} {'Rg last':>8s} {'drift':>8s} "
      f"{'angle last':>11s} {'stack last':>11s}")
print("-" * 84)
for label, r in results.items():
    print(f"{label:>22s} {r['native']:8.3f} {r['rg'][0]:8.3f} {r['rg'][-1]:8.3f} "
          f"{r['rg'][-1] - r['rg'][0]:+8.3f} {r['angle'][-1]:11.3f} {r['stack'][-1]:11.3f}")
print(f"\nblocks span {w[0][0]:.1f}-{w[-1][1]:.1f} ps; 'Rg b1' is the first block, not t=0")
print("The arm whose Rg stays near the native value is the term doing the compacting.")
