"""What the PRODUCT looks like: the shipped P-trace reconstruction against the sampled bead frame.

One CG sampling run of 2OIU under the production field (with the calibrated base-level term ON), then two
all-atom structures built from the SAME state -- the P trace it ended on, and the three beads it actually
moved. Both are written to PDB and measured with the stacking instrument on their own ring atoms, so the
comparison is between products rather than intermediates.

WHY THE LOOP'S SAMPLER RATHER THAN THE REFINER: torch_gpu_refine's CG stage is a pre-fold + REMD +
relaxation chain that took 237 s on this machine when last measured and did not finish in 20 minutes under
the load of a shared working tree; ibi_core.run_round is the same field, the same tables and the same
constraints, samples one chain in about a hundred seconds, and is what every measurement in findings
Parts 15-22 already used -- so a comparison run on i
t is comparable to them.

The protocol is deliberately short -- this is a reconstruction comparison, not a convergence claim.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))

import boltzmann_bonded as B                                              # noqa: E402
import cg_potentials as CP                                                # noqa: E402
import ibi_core as IC                                                     # noqa: E402
import measure_base_stacking as M                                          # noqa: E402
from torusfold.scheme2 import torch_cgsim as C                            # noqa: E402
from torusfold.scheme2 import base_frames as BF                            # noqa: E402
from torusfold.scheme2.aform_from_template import (                        # noqa: E402
    reconstruct_all_atom, reconstruct_all_atom_from_beads, write_allatom_pdb)

CRYSTAL = REPO / "artifacts" / "2oiu" / "2OIU.pdb"
SPEC = REPO / "results" / "plan_c" / "_2oiu_input.json"
PROD = REPO / "results" / "production_tables.npz"
WORK = REPO / "results" / "plan_c" / "bead_product"
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
EPS, W = 16.6, (0.0, 1.0, 0.19)      # the fourth calibration (findings Part 22)


def metrics(tag, seq, residues, pairs):
    planes = [M.plane(r) for r in residues]
    detected = M.wc_pairs(seq, residues, planes)
    rows, skipped = M.measure(seq, residues, pairs, planes)
    M.summarize(rows, len(pairs), skipped, tag,
                "(%d/%d WC contacts kept)" % (len(detected), len(pairs)))


def bead_stats(tag, nb_A, nrm, n):
    d, rise, cos = [], [], []
    nb_A = np.asarray(nb_A, dtype=float).reshape(n, 3, 3)
    nrm = np.asarray(nrm, dtype=float).reshape(n, 3)
    for i in range(n - 1):
        a = nrm[i].ravel().copy()
        b = nrm[i + 1].ravel().copy()
        if float(np.dot(a, b)) < 0:
            b = -b
        m = a + b
        nm = float(np.linalg.norm(m))
        m = m / nm if nm > 1e-9 else a
        # nb_A is (L, 3, 3): the three beads per residue, so the BASE bead is index 2 (N9 or N1).
        dc = (nb_A[i + 1, 2] - nb_A[i, 2]) / 10.0
        d.append(float(np.linalg.norm(dc)))
        rise.append(float(np.dot(dc, m)))
        cos.append(float(np.dot(a, b)))
    print("  %-24s base_dist %6.4f  base_rise %+7.4f (p5 %+7.4f)  base_cos %6.4f"
          % (tag, np.mean(d), np.mean(rise), np.percentile(rise, 5), np.mean(cos)), flush=True)


def main():
    seq, residues, p_crystal, ch = M.parse_pdb(CRYSTAL)
    planes = [M.plane(r) for r in residues]
    pairs = M.wc_pairs(seq, residues, planes)
    beads_crystal = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]]
                              for r in residues])                       # Angstrom
    L = len(seq)
    print("crystal %s: L=%d, %d WC pairs" % (CRYSTAL.name, L, len(pairs)), flush=True)
    metrics("crystal (reference)", seq, residues, pairs)
    bead_stats("crystal", beads_crystal, BF.normals_np(beads_crystal, BF.pooled_coef(1.0)), L)

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec_pairs = [(int(i), int(j)) for i, j, _w in spec["pairs"]]
    WORK.mkdir(parents=True, exist_ok=True)

    tab = IC.load_tables(str(REPO / "results" / "refit_smooth5_with_base.npz"))
    with np.load(PROD) as z:
        for c in ("bb_bond", "angle", "dihedral"):
            tab[c] = dict(tab[c], U=np.asarray(z[f"{c}__U"], dtype=float))
    _pots, pot_kw = CP.build_potential_kwargs(str(PROD), wall_k=2000.0,
                                              coords=("bb_bond", "angle", "dihedral"),
                                              base_stack={"eps": EPS, "form": "sum",
                                                          "w_d": W[0], "w_r": W[1], "w_t": W[2]})
    # argv: [nsteps] [temperature] [relax] -- the protocol axes measured by
    # scripts/measure_refinement_protocol.py, so the product comparison can be run on the protocol that
    # actually keeps the trace where it started (which is what the reconstruction's payoff depends on).
    nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    relax = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    pos0 = torch.tensor((beads_crystal / 10.0).reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(spec_pairs, dtype=torch.long).reshape(-1, 2)
    t0 = time.time()
    res = IC.run_round(pos=pos0, vel=torch.zeros_like(pos0), ij=ij,
                       pw=torch.ones(len(ij), dtype=torch.float32),
                       temps=torch.full((1,), temp, dtype=torch.float64), tab=tab,
                       nsteps=nsteps, burn=0, stride=25, blocks=4, friction=1.0, force_cap=5000.0,
                       pot_kw=pot_kw, seed=20261005, nrep=1, progress=False,
                       constraints=C.make_intra_constraints(L), relax=relax,
                       collect_positions=True, log=lambda *a, **k: None)
    frames = res.positions.numpy()[:, 0].reshape(-1, 3 * L, 3)          # (F, 3L, 3) nm
    # A -> the crystal trace is in Angstrom, so the P trace has to be too. Measured the hard way: without
    # the * 10 the script printed "trace vs crystal 21.53 A" for a state that had moved 1.6 A, and an
    # 8-seed sweep of the same protocol (1.33-1.65 A, sd 0.12) is what exposed it.
    p_final = frames[-1][0::3] * 10.0
    print("\nCG sampling: %d frames in %.0f s | trace vs crystal %.2f A (Kabsch)"
          % (len(frames), time.time() - t0, M.kabsch_rmsd(p_final, p_crystal)), flush=True)

    beads_A = frames[-1].reshape(L, 3, 3) * 10.0                        # Angstrom
    bead_stats("sampled, term ON", beads_A, BF.normals_np(beads_A, BF.pooled_coef(1.0)), L)

    print("", flush=True)
    st_a = reconstruct_all_atom(p_final, seq, pairs=[(i, j) for (i, j) in pairs])
    pa = WORK / "2oiu_product_ptrace.pdb"
    write_allatom_pdb(st_a, str(pa))
    metrics("PRODUCT A: P trace (shipped path)", seq, M.structure_to_residues(st_a, seq), pairs)

    st_b = reconstruct_all_atom_from_beads(frames[-1].reshape(L, 3, 3), seq)
    pb = WORK / "2oiu_product_beadframe.pdb"
    write_allatom_pdb(st_b, str(pb))
    metrics("PRODUCT B: sampled bead frame", seq, M.structure_to_residues(st_b, seq), pairs)

    print("\nwrote %s and %s" % (pa.name, pb.name), flush=True)
    print("(one draw, %d steps at %.0f K with %d relaxation steps, production tables WITH the term at"
          " eps=%.1f w=%s -- a reconstruction comparison, not a convergence claim)"
          % (nsteps, temp, relax, EPS, W), flush=True)


if __name__ == "__main__":
    main()
