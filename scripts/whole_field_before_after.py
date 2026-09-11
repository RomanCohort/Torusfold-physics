"""The whole-field before and after, which 3x admitted was never measured.

3v reported beads over the 200 kJ/mol/nm cap falling from 26.26 to 3.25 percent, but that was
measured on the sum of the six BONDED terms only -- measure_bonded_force_distribution.py
excludes bpp, the pair guide, the BSJ terms, the clash term and the GB/SA block. 3x then
measured the CURRENT whole field and found bpp alone puts 22.31 percent of beads over the cap,
which means the subset number said very little about the field.

This loads torch_cgsim.py from 70daa97, the last commit before the force field changed, and
compares the whole field against the working tree on the same structures, the same geometry
and the same pair weights. The cap is inside both versions, so the comparable statistic is
how many beads sit ON it -- that counts the beads whose raw force exceeded 200 either way.

Run: python scripts/whole_field_before_after.py [n_structs]
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as NEW

REF = "70daa97"
REL = "src/torusfold/scheme2/torch_cgsim.py"
CAP = 200.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CACHE = REPO / "results" / "rcm_weights.npz"


def load_from_git(ref, name):
    src = subprocess.run(["git", "show", f"{ref}:{REL}"], cwd=REPO, capture_output=True,
                         text=True, encoding="utf-8", check=True).stdout
    tmp = Path(tempfile.gettempdir()) / f"{name}.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OLD = load_from_git(REF, "old_torch_cgsim_ba")
structs = [s for s in B.load_structures(limit=300) if len(s["pairs"]) >= 4][:N]
W = np.load(CACHE)["w"] if CACHE.exists() else np.full(100, 0.35)
print(f"before: {REF}:{REL}")
print(f"after:  working tree")
print(f"{len(structs)} structures")
print()
print(f"{'':12s} {'K_ANGLE':>9s} {'K_DIH':>8s} {'STACK_R0':>10s} {'cos(DIH)':>10s}")
print("-" * 54)
for label, m in (("before", OLD), ("after", NEW)):
    print(f"{label:12s} {m.K_ANGLE:9.1f} {m.K_DIH:8.1f} {m.STACK_R0:10.3f} "
          f"{np.cos(m.DIH_PPPP):+10.3f}")
print()


def measure(mod, s, w):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    cl = mod.GPUCellList(cell_size=1.5)
    cl.build(pos)
    pw = torch.tensor(w, dtype=torch.float32)
    e, f = mod.cg_energy_forces(pos, ij, pw, cell_list=cl)
    return float(e.reshape(-1)[0]), f.reshape(3 * L, 3).numpy()


gen = np.random.default_rng(11)
for wlabel, sample in (("pair_w = ones", None),
                       ("pair_w from the measured RCM shape", W)):
    print("=" * 74)
    print(wlabel)
    print(f"  {'':10s} {'mean |F|':>10s} {'p95 |F|':>10s} {'max |F|':>10s} "
          f"{'on cap':>9s} {'fraction':>10s}")
    print("  " + "-" * 62)
    for label, mod in (("before", OLD), ("after", NEW)):
        mags, capped, ntot = [], 0, 0
        for s in structs:
            npr = len(s["pairs"])
            w = np.ones(npr) if sample is None else \
                np.asarray(sample)[gen.integers(0, len(sample), npr)]
            _e, f = measure(mod, s, w)
            m = np.linalg.norm(f, axis=1)
            mags.append(m)
            capped += int((np.abs(m - CAP) < 1e-6).sum())
            ntot += len(m)
        m = np.concatenate(mags)
        print(f"  {label:10s} {m.mean():10.2f} {np.percentile(m,95):10.2f} {m.max():10.2f} "
              f"{capped:9d} {capped/ntot*100:9.2f}%")
    print()

print("'on cap' counts beads sitting exactly on 200 kJ/mol/nm, i.e. whose raw force exceeded")
print("it. The cap is inside both versions, so this is the comparable statistic.")
