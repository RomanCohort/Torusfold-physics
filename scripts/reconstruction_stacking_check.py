"""A': does the all-atom level carry the stacking the CG model cannot express?

WHY THIS EXISTS. The CG model's four scored coordinates are all functions of the P trace alone
(boltzmann_bonded.coords_of): bb_bond = |P(i)-P(i+1)|, angle = the P-P-P pseudo-angle, dihedral =
the P-P-P-P pseudo-dihedral, and stack = |P(i)-P(i+2)|, which the cosine law makes an EXACT function
of the first two. So "stack" carries no base-plane information at all -- and the model says as much
in torch_cgsim (K_STACK = 0.0: the P_i-P_{i+2} spring is redundant to 8.5e-16 with bond+angle, and a
3-bead residue has no plane, no normal, no rise and no twist).

That leaves a question the IBI loop cannot answer, because it never sees all atoms: if stacking is
handed to the all-atom level (aform_from_template reconstruction + amber14-OL3 refinement, which this
repo already runs -- README level 5), is it actually THERE in the product? This measures it directly,
from CG P traces off the pool, with no sampling of the CG model:

  1. reconstruct 1EHZ all-atom from the deposited P trace
  2. measure the base stacking geometry (centroid distances, inter-plane angles, rise)
  3. refine under amber14-OL3 + OBC1
  4. measure again -- and then repeat with a PERTURBED P trace, to ask the decision-relevant version:
     does refinement RESTORE stacking when the CG arrangement is wrong, or does it only polish the
     arrangement the reconstruction handed it?
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

PURINE = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")
PYRIMIDINE = ("N1", "C2", "O2", "N3", "C4", "C5", "C6")


def base_atoms(name):
    return PURINE if name in "AG" else PYRIMIDINE


def planes(coords, structure):
    """(centroid, normal) per residue, from the base ring atoms; None where unusable."""
    out = []
    for i, letter in enumerate(structure.sequence):
        idx = structure.residue_atom_index[i]
        names = [n for n in base_atoms(letter) if n in idx]
        if len(names) < 3:
            out.append(None)
            continue
        pts = np.array([coords[idx[n]] for n in names], dtype=float)
        c = pts.mean(axis=0)
        u, s, vt = np.linalg.svd(pts - c)
        out.append((c, vt[2]))
    return out


def stacking_metrics(coords, structure):
    pl = planes(coords, structure)
    cent, angs, rises, stacked = [], [], [], 0
    n_pairs = 0
    for i in range(len(structure.sequence) - 1):
        a, b = pl[i], pl[i + 1]
        if a is None or b is None:
            continue
        d = float(np.linalg.norm(b[0] - a[0]))
        cosang = abs(float(np.dot(a[1], b[1])))
        ang = float(np.degrees(np.arccos(min(1.0, cosang))))
        # rise: centroid separation along the mean plane normal
        nmean = a[1] + b[1]
        nmean = nmean / max(1e-9, np.linalg.norm(nmean))
        rise = abs(float(np.dot(b[0] - a[0], nmean)))
        cent.append(d)
        angs.append(ang)
        rises.append(rise)
        n_pairs += 1
        if d < 6.0 and ang < 45.0:
            stacked += 1
    if not cent:
        return None
    return {"n_pairs": n_pairs,
            "centroid_med": float(np.median(cent)),
            "interplane_med": float(np.median(angs)),
            "rise_med": float(np.median(rises)),
            "stacked_frac": stacked / n_pairs}


def main():
    import boltzmann_bonded as B
    import torch  # noqa: F401  (loaded by the loader anyway)
    # reconstruct_all_atom lives in TWO modules; the one that takes base pairs (so a paired residue's
    # base is anchored toward its partner) is aform_from_template's, not allatom_reconstruct's.
    from torusfold.scheme2.aform_from_template import reconstruct_all_atom
    from torusfold.scheme2.allatom_reconstruct import get_atom_xyzs

    KMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    pool = [s for s in B.load_structures(limit=5000, with_names=True) if len(s["pairs"]) >= 8]
    pool = [s for s in pool if 25 <= len(s["pos"]) <= 90][:3]
    print(f"chains: " + ", ".join(f"{s['name']}(L={len(s['pos'])}, {len(s['pairs'])} pairs)"
                                  for s in pool), flush=True)
    for s in pool:
        pos = np.asarray(s["pos"], dtype=float)          # (L, 3, 3) nm, beads P/C4'/N9-N1
        seq = "".join(s["names"])
        p_ang = pos[:, 0, :] * 10.0                       # P trace, Angstrom
        print(f"\n=== {s['name']} L={len(seq)}  seq={seq[:40]}{'...' if len(seq) > 40 else ''}",
              flush=True)
        for tag, p_in in (("deposited", p_ang), ("perturbed", p_ang + np.random.default_rng(7).normal(
                0, 1.5, size=p_ang.shape))):
            try:
                st = reconstruct_all_atom(p_in, seq, pairs=s["pairs"])
                # TERMINAL NAMES. amber_refine's topology builder is written for CIRCULAR chains
                # (circRNA): every residue is internal, so the internal A/G/C/U templates match and
                # the 5'/3' ends are bonded to each other. A linear pool chain needs the real
                # terminal templates instead, and the reconstruction's first residue already has
                # P+OP1+OP2 while its last has O3', which is exactly what amber14's A5/A3 want.
                # Without this the force field answers "No template found for residue 0 (G): the set
                # of atoms is similar to G, but has 1 O atom too many".
                st.atoms[st.residue_atom_spans[0][0]].res_name += "5"
                st.atoms[st.residue_atom_spans[-1][0]].res_name += "3"
                coords = get_atom_xyzs(st)
                m0 = stacking_metrics(coords, st)
                print(f"  [{tag}] reconstructed: {len(st.atoms)} atoms, stacking {m0}", flush=True)
            except Exception as exc:
                print(f"  [{tag}] reconstruction FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            try:
                # from the MODULE, not the package: torusfold.scheme2 re-exports the FUNCTION
                # under this name, so "import amber_refine as AR" binds a function and AR.amber_refine
                # does not exist.
                from torusfold.scheme2.amber_refine import amber_refine as AR_fn
                out = AR_fn(st, [tuple(p) for p in s["pairs"]], platform_name="auto",
                            max_iterations=3000)
                rc = np.asarray(out[0], dtype=float)
                info = out[3] if len(out) > 3 else {}
                m1 = stacking_metrics(rc, st)
                print(f"  [{tag}] refined:      stacking {m1}  info={dict(list(info.items())[:4])}",
                      flush=True)
            except Exception as exc:
                print(f"  [{tag}] refine FAILED: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
