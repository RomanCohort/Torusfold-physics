"""Is the 20 A disagreement between two scripts a setup difference, or trajectory chaos?

Same protocol (1000 steps, 300 K, no relaxation, 2OIU from its deposit), only the seed changes. If the
final Kabsch RMSD spans a wide range across seeds, the two scripts were never in disagreement -- they drew
different samples, and a single-draw product comparison is meaningless. If it is tight, something in the two
setups really does differ.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))
import cg_potentials as CP, ibi_core as IC, measure_base_stacking as M
from torusfold.scheme2 import torch_cgsim as C, base_frames as BF

CRYSTAL = REPO / "artifacts" / "2oiu" / "2OIU.pdb"
SPEC = REPO / "results" / "plan_c" / "_2oiu_input.json"
PROD = REPO / "results" / "production_tables.npz"
REF = REPO / "results" / "refit_smooth5_with_base.npz"
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
EPS, W = 16.6, (0.0, 1.0, 0.19)

seq, residues, p_crystal_A, ch = M.parse_pdb(CRYSTAL)
L = len(seq)
beads_A = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]] for r in residues])
spec_pairs = [(int(i), int(j)) for i, j, _w in json.loads(SPEC.read_text(encoding="utf-8"))["pairs"]]
tab = IC.load_tables(str(REF))
with np.load(PROD) as z:
    for c in ("bb_bond", "angle", "dihedral"):
        tab[c] = dict(tab[c], U=np.asarray(z[f"{c}__U"], dtype=float))
_pots, pot_kw = CP.build_potential_kwargs(str(PROD), wall_k=2000.0,
                                          coords=("bb_bond", "angle", "dihedral"),
                                          base_stack={"eps": EPS, "form": "sum",
                                                      "w_d": W[0], "w_r": W[1], "w_t": W[2]})
pos0 = torch.tensor((beads_A / 10.0).reshape(1, 3 * L, 3), dtype=torch.float64)
ij = torch.tensor(spec_pairs, dtype=torch.long).reshape(-1, 2)
rms = []
for seed in (20261005, 1, 2, 3, 4, 5, 6, 7):
    res = IC.run_round(pos=pos0.clone(), vel=torch.zeros_like(pos0), ij=ij,
                       pw=torch.ones(len(ij), dtype=torch.float32),
                       temps=torch.full((1,), 300.0, dtype=torch.float64), tab=tab,
                       nsteps=1000, burn=0, stride=25, blocks=4, friction=1.0, force_cap=5000.0,
                       pot_kw=pot_kw, seed=seed, nrep=1, progress=False,
                       constraints=C.make_intra_constraints(L), relax=0,
                       collect_positions=True, log=lambda *a, **k: None)
    P = res.positions.numpy()[:, 0].reshape(-1, 3 * L, 3)[:, 0::3, :] * 10.0
    r = M.kabsch_rmsd(P[-1], p_crystal_A)
    rms.append(r)
    print("seed %-9s final RMSD vs deposit %6.2f A   (this process: %d threads)"
          % (seed, r, torch.get_num_threads()), flush=True)
a = np.asarray(rms)
print("\n%d seeds: min %.2f  median %.2f  max %.2f  sd %.2f A" % (a.size, a.min(), np.median(a), a.max(), a.std()))
print("spread max/min = %.1fx" % (a.max() / max(a.min(), 1e-9)))
