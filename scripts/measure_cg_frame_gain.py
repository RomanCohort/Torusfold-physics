"""The payoff of reconstructing bases from the SAMPLED frame, measured on a real CG run.

WHAT THIS COMPARES, on one and the same CG state:
  (1) what the pipeline ships today -- reconstruct_all_atom on the final P trace, where the base roll
      is guessed from the base-pair partner or a radial fallback;
  (2) reconstruct_all_atom_from_beads on the beads the sampler actually moved, i.e. the same template
      fitted onto the sampled P / C4' / N frame.

The beads are obtained honestly: torch_gpu_refine now takes cg_bead_sink, which hands out the last
3-bead state (it always existed -- REMD carries it across rounds -- but the interface sliced it to
P and threw the rest away). The protocol here is deliberately SHORT: this is a reconstruction
comparison, not a convergence claim, so 4 replicas x 5000 steps with the pre-fold and the physical
relaxation on is enough to produce a real sampled base frame on a real 2OIU trace.

The instrument is scripts/measure_base_stacking.py: helical steps from the crystal's own WC pairs,
stacked = rise 2.5-4.0 A and normal angle <= 30 degrees, and the WC contacts kept are counted
separately because the template path was measured to keep almost none of them.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))

import measure_base_stacking as M                      # noqa: E402
from torusfold.scheme2.torch_gpu_refine import torch_gpu_refine            # noqa: E402
from torusfold.scheme2.aform_from_template import (                        # noqa: E402
    reconstruct_all_atom, reconstruct_all_atom_from_beads)

CRYSTAL = REPO / "artifacts" / "2oiu" / "2OIU.pdb"
CG_IN = REPO / "results" / "plan_c" / "ab_2oiu" / "crystal_p.pdb"
SPEC = REPO / "results" / "plan_c" / "_2oiu_input.json"
WORK = REPO / "results" / "plan_c" / "cg_frame_probe"


def measure_structure(st, seq, pairs, tag):
    rows, skipped = M.measure(seq, st_residues(st, seq), pairs, [M.plane(r) for r in st_residues(st, seq)])
    return rows, skipped


def st_residues(st, seq):
    return M.structure_to_residues(st, seq)


def main():
    seq, residues, p_crystal, ch = M.parse_pdb(CRYSTAL)
    planes = [M.plane(r) for r in residues]
    pairs = M.wc_pairs(seq, residues, planes)
    print("crystal %s: L=%d, %d WC pairs" % (CRYSTAL.name, len(seq), len(pairs)), flush=True)
    M.summarize(M.measure(seq, residues, pairs, planes)[0], len(pairs), 0, "crystal (reference)")

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    WORK.mkdir(parents=True, exist_ok=True)
    sink = []
    t0 = time.time()
    pdb_out, energy, diag = torch_gpu_refine(
        str(CG_IN), str(WORK), seq, spec["ss"], name="2oiu_cgframe",
        nstep=2000, use_remd=True, remd_n_replicas=4, remd_n_steps=5000,
        use_multistage_remd=False, use_potential_refine=True, verbose=True,
        skip_cg_to_allatom=True, cg_bead_sink=sink)
    dt = time.time() - t0
    print("\nCG run: %.1f s, E=%.6g, sink entries %d" % (dt, float(energy), len(sink)), flush=True)

    p_final = np.asarray(sink[0]["p"], dtype=float) if sink else None
    beads = sink[0]["beads"] if sink else None
    if p_final is None or beads is None:
        print("no bead frame was handed out: nothing to compare (the sink reported %s)"
              % ("empty" if not sink else "beads=None"))
        return
    rms = M.kabsch_rmsd(p_final, p_crystal)
    print("final P trace vs the crystal trace: %.3f A (Kabsch), %d beads, %s"
          % (rms, beads.shape[0], "finite" if np.all(np.isfinite(beads)) else "NOT finite"), flush=True)

    # (1) the shipped path
    rows, skipped, det = M.measure_reconstruction(seq, p_final, pairs, "heuristic")
    M.summarize(rows, len(pairs), skipped, "P-trace template (shipped)",
                "(%d/%d WC contacts kept)" % (det, len(pairs)))
    # (2) the sampled frame
    st = reconstruct_all_atom_from_beads(beads, seq)
    residues2 = st_residues(st, seq)
    planes2 = [M.plane(r) for r in residues2]
    det2 = len(M.wc_pairs(seq, residues2, planes2))
    rows2, skipped2 = M.measure(seq, residues2, pairs, planes2)
    M.summarize(rows2, len(pairs), skipped2, "SAMPLED bead frame (new)",
                "(%d/%d WC contacts kept)" % (det2, len(pairs)))

    # and a sanity number: has the bead frame kept the same P trace it was sampled with?
    be_p = beads[:, 0, :]
    print("\nbead P vs the P trace they belong to: %.4f A RMSD (should be ~0; it is the same state)"
          % float(np.sqrt(((be_p - p_final) ** 2).sum(1).mean())), flush=True)
    print("(protocol: 4 replicas x 5000 steps, one draw, no convergence claim -- this measures the"
          " RECONSTRUCTION of one real CG state, not the field.)", flush=True)


if __name__ == "__main__":
    main()
