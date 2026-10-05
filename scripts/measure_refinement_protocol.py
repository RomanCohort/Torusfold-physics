"""Which refinement protocol keeps a chain near the geometry it was deposited in?

WHY. Part 23 measured that the sampled-frame reconstruction is a real gain on an A-form-like trace
(43.9 -> 75.7 percent stacked on ideal input) and a LOSS on a drifted one, because its rigid template fit
turns a distorted bead triangle into a base-plane error. The 2OIU run it used ended 21.5 A from the deposit,
so the reconstruction's payoff depends entirely on a question nobody had measured: what protocol keeps the
trace where it started?

THE SHAPE OF THE DRIFT IS THE POINT, so every frame is kept and the RMSD is reported along the trajectory,
not only at the end. A startup kick (the deposit has clashes under the CG field, so it relaxes hard in the
first picosecond and then sits) saturates; diffusion grows as sqrt(t) and never stops. The fix is different
in the two cases -- a relaxation before sampling for the first, a restraint or a colder thermostat for the
second -- and the curve says which one this is.

AXES: relaxation before sampling (0 = straight to the thermostat, 5000 = the retention instrument's descent),
temperature (the field's tables were fitted at 300 K, so a colder run is off-manifold and is being asked only
to stay put), and length. One 2OIU chain, 5000 steps at stride 25 unless stated.
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

import cg_potentials as CP                                                # noqa: E402
import ibi_core as IC                                                     # noqa: E402
import measure_base_stacking as M                                          # noqa: E402
from torusfold.scheme2 import torch_cgsim as C                            # noqa: E402
from torusfold.scheme2 import base_frames as BF                            # noqa: E402

CRYSTAL = REPO / "artifacts" / "2oiu" / "2OIU.pdb"
SPEC = REPO / "results" / "plan_c" / "_2oiu_input.json"
PROD = REPO / "results" / "production_tables.npz"
REF = REPO / "results" / "refit_smooth5_with_base.npz"
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
EPS, W = 16.6, (0.0, 1.0, 0.19)


def bead_stats(nb_A, n):
    d, rise, cos = [], [], []
    nrm = BF.normals_np(nb_A, BF.pooled_coef(1.0))
    for i in range(n - 1):
        a, b = nrm[i].copy(), nrm[i + 1].copy()
        if float(np.dot(a, b)) < 0:
            b = -b
        m = a + b
        nm = float(np.linalg.norm(m))
        m = m / nm if nm > 1e-9 else a
        dc = (nb_A[i + 1, 2] - nb_A[i, 2]) / 10.0
        d.append(float(np.linalg.norm(dc)))
        rise.append(float(np.dot(dc, m)))
        cos.append(float(np.dot(a, b)))
    return np.mean(d), np.mean(rise), float(np.percentile(rise, 5)), np.mean(cos)


def one(tag, seq, beads_A, p_crystal_A, spec_pairs, tab, pot_kw, T, nsteps, relax, stride=25):
    L = len(seq)
    pos0 = torch.tensor((beads_A / 10.0).reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(spec_pairs, dtype=torch.long).reshape(-1, 2)
    t0 = time.time()
    res = IC.run_round(pos=pos0, vel=torch.zeros_like(pos0), ij=ij,
                       pw=torch.ones(len(ij), dtype=torch.float32),
                       temps=torch.full((1,), float(T), dtype=torch.float64), tab=tab,
                       nsteps=nsteps, burn=0, stride=stride, blocks=4, friction=1.0,
                       force_cap=5000.0, pot_kw=pot_kw, seed=20261005, nrep=1, progress=False,
                       constraints=C.make_intra_constraints(L), relax=relax,
                       collect_positions=True, log=lambda *a, **k: None)
    frames = res.positions.numpy()[:, 0].reshape(-1, 3 * L, 3)
    P = frames[:, 0::3, :] * 10.0                     # nm -> A, P only
    rms = [M.kabsch_rmsd(P[k], p_crystal_A) for k in range(len(P))]
    d, rise, p5, cos = bead_stats(frames[-1].reshape(L, 3, 3) * 10.0, L)
    q = [rms[min(len(rms) - 1, int(len(rms) * f))] for f in (0.0, 0.25, 0.5, 0.75)]
    print("  %-34s T=%3.0f n=%5d relax=%4d | RMSD 0%%/25%%/50%%/75%%/end %5.2f %5.2f %5.2f %5.2f %5.2f A"
          " | beads d %.3f rise %+.3f (p5 %+.3f) cos %.3f | %.0f s"
          % (tag, T, nsteps, relax, q[0], q[1], q[2], q[3], rms[-1], d, rise, p5, cos,
             time.time() - t0), flush=True)
    return {"tag": tag, "T": T, "nsteps": nsteps, "relax": relax,
            "rmsd_frames": rms, "rmsd_end": rms[-1],
            "base_dist": d, "base_rise": rise, "base_rise_p5": p5, "base_cos": cos}


def main():
    seq, residues, p_crystal_A, ch = M.parse_pdb(CRYSTAL)
    L = len(seq)
    beads_A = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]]
                        for r in residues])
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec_pairs = [(int(i), int(j)) for i, j, _w in spec["pairs"]]

    tab = IC.load_tables(str(REF))
    with np.load(PROD) as z:
        for c in ("bb_bond", "angle", "dihedral"):
            tab[c] = dict(tab[c], U=np.asarray(z[f"{c}__U"], dtype=float))
    _pots, pot_kw = CP.build_potential_kwargs(str(PROD), wall_k=2000.0,
                                              coords=("bb_bond", "angle", "dihedral"),
                                              base_stack={"eps": EPS, "form": "sum",
                                                          "w_d": W[0], "w_r": W[1], "w_t": W[2]})
    print("2OIU, L=%d, %d pairs | reference base-level: " % (L, len(spec_pairs))
          + "d %.4f rise %+.4f (p5 %+.4f) cos %.4f"
          % bead_stats(beads_A, L), flush=True)
    out = []
    # relax=1000 is in the list because Part 23's 2OIU run used it (the product script's default) and
    # ended 21.5 A from the deposit, while relax=0 and relax=5000 here end at 1.6 and 3.4 A. If a PARTIAL
    # relaxation is the worst case, that is the actionable finding: a descent that stops halfway leaves
    # strain that the sampler then spends the trajectory releasing.
    for tag, T, n, relax in (("relax 1000 then 300 K, 5000", 300.0, 5000, 1000),
                             ("relax 2000 then 300 K, 5000", 300.0, 5000, 2000),
                             ("relax 5000 then 300 K, 5000 (repeat)", 300.0, 5000, 5000)):
        out.append(one(tag, seq, beads_A, p_crystal_A, spec_pairs, tab, pot_kw, T, n, relax))
    (REPO / "results" / "plan_c" / "_refinement_protocols.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote results/plan_c/_refinement_protocols.json", flush=True)


if __name__ == "__main__":
    main()
