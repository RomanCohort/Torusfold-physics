"""Two scoreboards for the same thing: what the TERM measures, and what the PRODUCT is.

WHY. Part 24 found that the field can score itself as stacked while the all-atom product is not, and traced
it to a difference of question rather than a bug: base_frames (and therefore the term, and therefore every
base-level number in Parts 15-24) takes the base normal as a fixed LINEAR combination of the three bead
vectors -- exact for the template's geometry, a linearisation away from it -- while
reconstruct_all_atom_from_beads fits the whole rigid residue onto those three points, where a distorted
triangle makes the best-fit rotation a shape-mismatch artefact. Near the deposited geometry the two agree
(Part 14 measured 43.9 -> 75.7 percent with the crystal's own rigid units); at 1.58 A of drift they did not.

This script reports both numbers for the SAME sampled frames, so the size of the gap is a measurement instead
of a caveat:

    linear scoreboard   base_dist / base_rise / base_cos through the rigid template MAP (what the term uses)
    product scoreboard  the same quantities measured on the ring atoms of reconstruct_all_atom_from_beads

and the crystal reference for both. Units are Angstrom throughout, so the two columns are comparable.

Protocol: one 2OIU chain, 1000 steps at 300 K with no pre-relaxation -- the protocol Part 24's sweep found
keeps the trace closest (1.58 A, sd 0.12 over eight seeds) -- term on at the calibrated setting.
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
from torusfold.scheme2.aform_from_template import reconstruct_all_atom_from_beads   # noqa: E402

CRYSTAL = REPO / "artifacts" / "2oiu" / "2OIU.pdb"
SPEC = REPO / "results" / "plan_c" / "_2oiu_input.json"
PROD = REPO / "results" / "production_tables.npz"
REF = REPO / "results" / "refit_smooth5_with_base.npz"
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
EPS, W = 16.6, (0.0, 1.0, 0.19)


def linear_rows(beads_A, n):
    """base_dist / base_rise / base_cos through the map the term uses, in Angstrom."""
    nrm = BF.normals_np(beads_A, BF.pooled_coef(1.0))
    d, rise, cos = [], [], []
    for i in range(n - 1):
        a, b = nrm[i].copy(), nrm[i + 1].copy()
        if float(np.dot(a, b)) < 0:
            b = -b
        m = a + b
        nm = float(np.linalg.norm(m))
        m = m / nm if nm > 1e-9 else a
        dc = beads_A[i + 1, 2] - beads_A[i, 2]
        d.append(float(np.linalg.norm(dc)))
        rise.append(float(np.dot(dc, m)))
        cos.append(float(np.dot(a, b)))
    return np.array(d), np.array(rise), np.array(cos)


def product_rows(beads_A, seq, pairs):
    """The same three quantities measured on the ring atoms of the RECONSTRUCTION.

    beads_A is in ANGSTROM -- reconstruct_all_atom_from_beads' unit. Feeding it nm (which this function did
    first) scales the anchors by ten against a template that is in Angstrom, so the rigid fit lands somewhere
    arbitrary: on the CRYSTAL the N9/N1 separation read 1.65 A against the beads' own 5.31.
    """
    L = len(seq)
    beads_A = np.asarray(beads_A, dtype=float).reshape(L, 3, 3)
    st = reconstruct_all_atom_from_beads(beads_A, seq)
    residues = M.structure_to_residues(st, seq)
    # SIGN CONVENTION, the same one base_frames uses for the map: an SVD plane normal comes out with an
    # arbitrary sign, so without flipping it to point along +e3 = (C4'-P) x (N-P) the mean rise over pairs
    # is a sum of mixed signs and reads ~0 -- measured on the CRYSTAL, where the two definitions must agree:
    # the map's mean rise was +3.186 A and the ring-atom one read +0.086 +- 0.997 before this line existed.
    planes = []
    for i, r in enumerate(residues):
        c, n, v = M.plane(r)
        e1 = beads_A[i, 1] - beads_A[i, 0]
        e2 = beads_A[i, 2] - beads_A[i, 0]
        if float(np.dot(n, np.cross(e1, e2))) < 0:
            n = -np.asarray(n, dtype=float)
        planes.append((c, np.asarray(n, dtype=float), v))
    d, rise, cos = [], [], []
    # THE SAME POINTS as the map measures, or the comparison is apples to oranges: the map's base_dist and
    # base_rise are |N(i+1) - N(i)| and its projection, built on the N9/N1 BEADS. Measuring the product on
    # the ring CENTROIDS instead (the obvious thing, and what this script did first) gives 2.62 against
    # 5.31 A on the CRYSTAL, where the two definitions must agree -- a factor of two, from the centroid
    # sitting in the middle of the ring while N9/N1 sits at its edge.
    for i in range(L - 1):
        if planes[i] is None or planes[i + 1] is None:
            continue
        _ci, ni, _vi = planes[i]
        _cj, nj, _vj = planes[i + 1]
        a, b = np.asarray(ni, float), np.asarray(nj, float)
        if float(np.dot(a, b)) < 0:
            b = -b
        m = a + b
        nm = float(np.linalg.norm(m))
        m = m / nm if nm > 1e-9 else a
        gi = "N9" if seq[i] in "AG" else "N1"
        gj = "N9" if seq[i + 1] in "AG" else "N1"
        if gi not in residues[i]["atoms"] or gj not in residues[i + 1]["atoms"]:
            continue
        dc = np.asarray(residues[i + 1]["atoms"][gj], float) - np.asarray(residues[i]["atoms"][gi], float)
        d.append(float(np.linalg.norm(dc)))
        rise.append(float(np.dot(dc, m)))
        cos.append(float(np.dot(a, b)))
    rows, _sk = M.measure(seq, residues, pairs, planes)
    return np.array(d), np.array(rise), np.array(cos), rows


def summarize(tag, d, rise, cos, extra=""):
    print("  %-26s base_dist %6.3f +- %-5.3f  base_rise %+6.3f +- %-5.3f (p5 %+6.3f)  base_cos %6.3f %s"
          % (tag, d.mean(), d.std(), rise.mean(), rise.std(), np.percentile(rise, 5), cos.mean(), extra),
          flush=True)


def main():
    seq, residues, p_crystal_A, ch = M.parse_pdb(CRYSTAL)
    L = len(seq)
    planes = [M.plane(r) for r in residues]
    pairs = M.wc_pairs(seq, residues, planes)
    beads_A = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]]
                        for r in residues])
    spec_pairs = [(int(i), int(j)) for i, j, _w in json.loads(SPEC.read_text(encoding="utf-8"))["pairs"]]
    print("2OIU L=%d, %d WC pairs | crystal reference:" % (L, len(pairs)), flush=True)
    summarize("crystal (linear map)", *linear_rows(beads_A, L))
    summarize("crystal (ring atoms)", *product_rows(beads_A, seq, pairs)[:3])

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
    t0 = time.time()
    res = IC.run_round(pos=pos0, vel=torch.zeros_like(pos0), ij=ij,
                       pw=torch.ones(len(ij), dtype=torch.float32),
                       temps=torch.full((1,), 300.0, dtype=torch.float64), tab=tab,
                       nsteps=1000, burn=0, stride=25, blocks=4, friction=1.0, force_cap=5000.0,
                       pot_kw=pot_kw, seed=20261005, nrep=1, progress=False,
                       constraints=C.make_intra_constraints(L), relax=0,
                       collect_positions=True, log=lambda *a, **k: None)
    frames = res.positions.numpy()[:, 0].reshape(-1, 3 * L, 3)
    print("\n%d frames in %.0f s" % (len(frames), time.time() - t0), flush=True)

    take = list(range(0, len(frames), max(1, len(frames) // 12)))[:12]
    lin_d, lin_r, lin_c = [], [], []
    prd_d, prd_r, prd_c = [], [], []
    stacked = []
    for k in take:
        f = frames[k]
        a, b, c = linear_rows(f.reshape(L, 3, 3) * 10.0, L)
        lin_d.append(a.mean()); lin_r.append(b.mean()); lin_c.append(c.mean())
        pd, pr, pc, rows = product_rows(f.reshape(L, 3, 3) * 10.0, seq, pairs)   # nm -> A
        prd_d.append(pd.mean()); prd_r.append(pr.mean()); prd_c.append(pc.mean())
        stacked.append(100.0 * np.mean([r["stacked"] for r in rows]) if rows else float("nan"))
    print("", flush=True)
    summarize("SAMPLED  linear map", np.array(lin_d), np.array(lin_r), np.array(lin_c),
              "(%d frames)" % len(take))
    summarize("SAMPLED  ring atoms", np.array(prd_d), np.array(prd_r), np.array(prd_c),
              "(stacked %.1f%% of helical steps)" % np.nanmean(stacked))
    lr, pr = np.array(lin_r), np.array(prd_r)
    print("\nper-frame correlation of the two rise definitions: r = %.3f over %d frames"
          % (float(np.corrcoef(lr, pr)[0, 1]), len(lr)), flush=True)
    print("  linear rise %s" % np.round(lr, 2).tolist(), flush=True)
    print("  ring   rise %s" % np.round(pr, 2).tolist(), flush=True)
    print("  stacked %%  %s" % np.round(np.array(stacked), 1).tolist(), flush=True)
    (REPO / "results" / "plan_c" / "_product_scoreboard.json").write_text(json.dumps(
        {"linear_rise": lr.tolist(), "product_rise": pr.tolist(), "stacked": stacked,
         "linear_dist": lin_d, "product_dist": prd_d}, indent=2), encoding="utf-8")
    print("\nwrote results/plan_c/_product_scoreboard.json", flush=True)


if __name__ == "__main__":
    main()
