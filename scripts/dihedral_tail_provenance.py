"""Where do the dihedral intermediate tail and trans peak come from, structure by structure?

The reference dihedral distribution (results/boltzmann_tables_clean.npz) is bimodal: a cis
peak at cos(phi) ~ +1 (63% of mass), a trans peak at ~ -1 (6.5%, 2.25 kBT above cis), and a
broad intermediate tail of ~30% in cos in (-0.8, +0.8). Two stories fit those numbers:

  (a) database pollution: a few low-quality / non-A-form structures put most of the mass in
      the intermediate tail and/or the trans peak, so the reference should be cleaned;
  (b) real conformational heterogeneity: the tail and trans peak are spread evenly across the
      chains, so the field must express them.

This script decides between them from the RAW PDB coordinates (not the table): it loads every
chain through boltzmann_bonded.load_structures, computes the P-P-P-P pseudo-torsion cosine
with the same normal-based definition the pipeline uses, and attributes each observation to
cis (q > 0.8), trans (q < -0.9) or intermediate (-0.8 <= q <= 0.8). It then reports, for the
intermediate and trans groups, how concentrated the mass is (top contributors and a
Herfindahl index), and lists the top structures.

It does NOT report the experimental method / resolution: the shipped PDB files have been
stripped to bare ATOM/HETATM records (no HEADER, no REMARK 2/3), and no metadata file ships
with the archive (the README only cites the rsRNASP paper, Tan et al.). That provenance has
to come from an external source and is out of scope here.

Run: python scripts/dihedral_tail_provenance.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402


def dihedral_cos(beads):
    """All P(i)-P(i+1)-P(i+2)-P(i+3) cosines in one chain, same definition as the pipeline.

    beads is (L, 3, 3) = (residues, {P, C4', N}, xyz); the torsion uses the P atom
    (index 0) of four consecutive residues."""
    P = torch.tensor(beads[:, 0, :], dtype=torch.float64)   # (L, 3)
    L = P.shape[0]
    if L < 4:
        return np.array([])
    a = torch.arange(L - 3)
    return B._cos_dihedral(
        P[a], P[a + 1], P[a + 2], P[a + 3]
    ).numpy()


def main():
    structs = B.load_structures()   # every gap-free chain in rsRNASP/Training_set
    n_chain = len(structs)
    total = 0
    per_struct = []
    for s in structs:
        q = dihedral_cos(s["pos"])
        if len(q) == 0:
            continue
        cis = int((q > 0.8).sum())
        trans = int((q < -0.9).sum())
        mid = int(((q >= -0.8) & (q <= 0.8)).sum())
        per_struct.append(dict(name=s["name"], n=len(q), cis=cis, trans=trans, mid=mid))
        total += len(q)

    cis = sum(s["cis"] for s in per_struct)
    trans = sum(s["trans"] for s in per_struct)
    mid = sum(s["mid"] for s in per_struct)
    n_chains_with_obs = len(per_struct)
    print(f"chains with >=1 dihedral: {n_chains_with_obs}   total dihedral obs: {total}")
    print(f"cis (q>0.8): {cis} ({cis/total:.3f})   trans (q<-0.9): {trans} ({trans/total:.3f})   "
          f"intermediate (-0.8..0.8): {mid} ({mid/total:.3f})")
    print()

    def concentration(key, label):
        vals = np.array([s[key] for s in per_struct], dtype=float)
        nstruct_nonzero = int((vals > 0).sum())
        # Herfindahl over the structures that carry any of this mass
        w = vals / vals.sum()
        H = float((w ** 2).sum())
        order = np.argsort(-vals)
        print(f"=== {label}: {int(vals.sum())} obs across {nstruct_nonzero} structures ===")
        print(f"    Herfindahl index = {H:.4f}  (1/{nstruct_nonzero:.1f} = {1/nstruct_nonzero:.4f} "
              f"if perfectly uniform; 1.0 if one structure)")
        print(f"    top 8 structures and their share of this group:")
        for i in order[:8]:
            print(f"      {per_struct[i]['name']:6s}  {int(vals[i]):4d} obs  "
                  f"{vals[i]/vals.sum():.3f} of group  "
                  f"(of its own {per_struct[i]['n']} dihedrals: {vals[i]/per_struct[i]['n']:.3f})")
        top5 = vals[order[:5]].sum() / vals.sum()
        print(f"    top-5 structures carry {top5:.3f} of the group's mass")
        print()

    concentration("mid", "intermediate tail q in (-0.8, +0.8)")
    concentration("trans", "trans peak q < -0.9")
    concentration("cis", "cis peak q > 0.8  (control: should be near-uniform if the set is homogeneous)")

    print("reading:")
    print("  a high Herfindahl (near 1) with a few structures carrying most of the group =")
    print("  pollution by outliers; a Herfindahl near 1/N = genuine, evenly-spread heterogeneity.")
    print()
    print("method/resolution provenance: NOT available from the local archive. The PDB files")
    print("are stripped to ATOM/HETATM (no HEADER, no REMARK 2/3) and the only metadata is the")
    print("rsRNASP README citing Tan et al. (Wuhan Univ). Resolutions must come from RCSB / the")
    print("rsRNASP paper supplement, not from this repository.")


if __name__ == "__main__":
    main()
