"""A/B on 2OIU: the pipeline's CG refinement under the FITTED field against the analytic one.

The switch added on 2026-10-01 (TORUSFOLD_CG_TABLES) makes this possible for the first time, and the
ROCm torch build on this machine makes it possible ON THE GPU. Both arms run the same shipped refiner
(torch_gpu_refine) from the same crystal P trace, the same secondary structure and the same base-pair
list; the only difference is which field the CG energy calls take.

What is compared, and what is not: the two arms' energies belong to different fields and are NOT
comparable -- each is reported for its own record. The comparison is geometric: the P-trace
coordinates (bb_bond, angle, dihedral, stack -- all four are functions of the P trace alone) of each
refined structure against the target the tables were fitted to (refit_smooth5), plus the Kabsch RMSD
between the two products. REMD velocities are not seeded, so each arm is one draw; that is stated in
the output rather than controlled for.
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

import boltzmann_bonded as B                      # noqa: E402
import ibi_bonded as I                            # noqa: E402
from torusfold.scheme2.torch_gpu_refine import torch_gpu_refine   # noqa: E402
from openmm.app import PDBFile                    # noqa: E402

IN = REPO / "results" / "plan_c" / "_2oiu_input.json"
TABLE = REPO / "results" / "production_tables.npz"
COORDS = ("bb_bond", "angle", "dihedral", "stack")


def p_coords_of(pdb_path, L):
    pdb = PDBFile(str(pdb_path))
    out = []
    for res in pdb.topology.residues():
        for atom in res.atoms():
            if atom.name.strip() == "P":
                q = pdb.positions[atom.index]
                out.append([q.x, q.y, q.z])
    return np.asarray(out, dtype=float)[:L]


def trace_stats(p_nm):
    L = p_nm.shape[0]
    t = torch.tensor(np.repeat(p_nm, 3, axis=0), dtype=torch.float64).reshape(1, 3 * L, 3)
    return {c: B.coords_of(t, c).reshape(-1).numpy() for c in COORDS}


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    A, Bm = a - ca, b - cb
    U, _, Vt = np.linalg.svd(Bm.T @ A)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return float(np.sqrt(((Bm @ R - A) ** 2).sum(1).mean()))


def main():
    spec = json.loads(IN.read_text(encoding="utf-8"))
    seq, ss, pairs = spec["seq"], spec["ss"], spec["pairs"]
    L = len(seq)
    crystal = np.asarray(spec["p_coords_nm"], dtype=float)
    work = REPO / "results" / "plan_c" / "ab_2oiu"
    work.mkdir(parents=True, exist_ok=True)
    p_pdb = work / "crystal_p.pdb"
    with p_pdb.open("w", encoding="utf-8") as fh:
        fh.write("HEADER 2OIU crystal P coords\n")
        for i in range(L):
            x, y, z = crystal[i] * 10.0
            fh.write("ATOM  %5d  P   RA A%4d    %8.3f%8.3f%8.3f\n" % (i + 1, i + 1, x, y, z))
        fh.write("END\n")
    print("2OIU %d nt, %d pairs, BSJ %.3f nm | torch %s cuda %s"
          % (L, len(pairs), float(np.linalg.norm(crystal[0] - crystal[-1])),
             torch.__version__, torch.cuda.is_available()), flush=True)

    targets = I.load_clean_tables(str(REPO / "results" / "refit_smooth5.npz"))
    results = {}
    for arm, tables in (("tables", True), ("analytic", False)):
        if tables:
            os.environ["TORUSFOLD_CG_TABLES"] = str(TABLE)
        else:
            os.environ.pop("TORUSFOLD_CG_TABLES", None)
        print("\n=== arm %s (%s) ===" % (arm, "fitted tables" if tables else "analytic field"), flush=True)
        t0 = time.time()
        pdb_out, energy, diag = torch_gpu_refine(
            str(p_pdb), str(work / arm), seq, ss, name="2oiu_" + arm,
            nstep=20000, use_remd=True, remd_n_replicas=8, remd_n_steps=20000,
            use_multistage_remd=True, use_potential_refine=True, verbose=False,
            skip_cg_to_allatom=True)
        dt = time.time() - t0
        p = p_coords_of(pdb_out, L)
        results[arm] = {"pdb": str(pdb_out), "energy": float(energy), "seconds": dt, "p": p}
        print("  %.1f s, energy %.6g, BSJ %.3f nm" % (dt, float(energy),
                                                      float(np.linalg.norm(p[0] - p[-1]))), flush=True)
        print("  diag keys: " + ", ".join(sorted(diag.keys())[:8]), flush=True)

    print("\nP-trace statistics of the two products against the fit target (refit_smooth5)", flush=True)
    print("  coord      target_sd   analytic(sd)  tables(sd)   |ln(sd/target)| an / tab", flush=True)
    for c in COORDS:
        tgt = float(targets[c]["sigma"])
        st = {a: trace_stats(results[a]["p"])[c] for a in results}
        an, tb = float(st["analytic"].std()), float(st["tables"].std())
        print("  %-9s %10.4f %13.4f %12.4f %18.3f / %.3f"
              % (c, tgt, an, tb, abs(np.log(an / tgt)), abs(np.log(tb / tgt))), flush=True)
    rmsd = kabsch(results["analytic"]["p"], results["tables"]["p"])
    print("\nP-only Kabsch RMSD between the two refined structures: %.3f A" % (rmsd * 10.0), flush=True)
    print("(one draw per arm: REMD velocities are not seeded)", flush=True)


if __name__ == "__main__":
    main()
