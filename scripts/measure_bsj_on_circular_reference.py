"""Measure the BSJ closure term on a genuinely circular reference, not a linear chain.

The production target is a circular RNA: P(0) and P(L-1) are covalently joined, so the three
BSJ terms (K_BSJ / K_BSJ_GUIDE / K_BSJ_CONTACT) are physically legitimate and must stay ON. Every
validation run so far has used LINEAR deposited chains, whose two ends sit several nanometres
apart, so `bsj closure` fires on a violation that would not exist in the real construct and reads
as 68-91 percent of the energy (README "Known limits", section 3aa/3au).

Section 3ab tried to fix the test geometry by cyclically renumbering a linear chain and failed:
renumbering leaves the two ends still ~5 nm apart and fabricates a cross-molecule "bond", so
`bb bond` blew up from 30.4 to 2513.7. That is not a circle.

This script uses the ONE genuinely circular structure in the database, 2OIU (the only
experimentally resolved circRNA, per README). In its file order residue 1 is already bonded to
residue 71 -- the loader's own internal P-P bonds average 0.5864 nm and the wrap bond
|P(L-1)-P(0)| is ~0.5915 nm, i.e. the two ends ARE the phosphodiester neighbours, no renumbering
needed. On that geometry the BSJ closure coordinate sits at its target BOND_P_NEXT = 0.590 nm,
so the closure force should be near zero -- not the several-thousand kJ/mol/nm a linear chain
produces.

What is measured, per structure:
  - |P(0)-P(L-1)| (the closure coordinate, must be ~0.590 nm for a real circle);
  - the BSJ closure energy 0.5*K_BSJ*(r-BOND_P_NEXT)^2 and its endpoint force K_BSJ*|r-BOND_P_NEXT|,
    read from the SAME inline formula cg_energy_forces uses (torch_cgsim lines 1271-1276), not from
    cg_force_terms' duplicate, which is documented to drift;
  - the BSJ guide and BSJ contact forces (the other two BSJ terms), same inline formulas;
  - the full production field cg_energy_forces (all 17 terms, BSJ ON, force_cap=5000) so the
    closure reading can be put next to the whole-field force on the two junction beads.

The linear 7-chain pool (len(pairs)>=8 and 24<=len(pos)<=34, the ibi_round0 filter) is printed
alongside as the control. 2OIU has L=71 and 42 pairs, so it is NOT a member of that pool; it is
the circular reference the pool is missing.

Run: python scripts/measure_bsj_on_circular_reference.py [out.log]
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B                       # noqa: E402
import torusfold.scheme2.torch_cgsim as C           # noqa: E402


def closure_geometry(pos):
    """|P(0)-P(L-1)| in nm for a (3L,3) bead array."""
    return float(np.linalg.norm(pos[0, 0] - pos[-1, 0]))


def bs_terms(pos):
    """BSJ closure/guide/contact energy+force, from cg_energy_forces' own inline formulas.

    pos: (3L, 3) numpy, nm. Returns a dict of (energy kJ/mol, force_per_endpoint kJ/mol/nm).
    """
    L = len(pos)
    t = torch.tensor(pos.reshape(1, 3 * L, 3), dtype=torch.float64)
    P = lambda i: 3 * i
    out = {}

    # BSJ closure (torch_cgsim lines 1271-1276)
    d = t[:, P(0)] - t[:, P(L - 1)]
    r = C._safe_norm(d, dim=-1, keepdim=True, eps=1e-6)
    e = (0.5 * C.K_BSJ * (r.squeeze(-1) - C.BOND_P_NEXT) ** 2).sum()
    f = (C.K_BSJ * (r - C.BOND_P_NEXT)).abs().max()
    out["bsj closure"] = (float(e), float(f))

    # BSJ guide (torch_cgsim lines 1329-1335)
    e_bg, sig_bg = C._sigmoid_f(r, C.PAIR_NN, C.K_BSJ_GUIDE, 0.2)
    f_bg = (C.K_BSJ_GUIDE / 0.2 * sig_bg).abs().max()
    out["bsj guide"] = (float(e_bg.sum()), float(f_bg))

    # BSJ contact (torch_cgsim lines 1337-1352); only when L > 16
    e_c = 0.0
    f_c = 0.0
    if L > 16:
        for off in range(min(8, L // 2)):
            i1, i2 = off, L - 1 - off
            if i1 < i2:
                dc = t[:, P(i1)] - t[:, P(i2)]
                rc = C._safe_norm(dc, dim=-1, keepdim=True, eps=1e-6)
                wc = torch.exp(-0.1 * (rc / C.PAIR_NN))
                e_c += float((C.K_BSJ_CONTACT * wc).sum())
                f_c = max(f_c, float((C.K_BSJ_CONTACT * 0.1 / C.PAIR_NN * wc).abs().max()))
    out["bsj contact"] = (e_c, f_c)
    return out


def full_field(pos, pairs):
    """Production field: cg_energy_forces with BSJ ON, force_cap=5000.

    Returns (max per-bead force norm kJ/mol/nm, force norm on junction P beads, total energy).
    """
    L = len(pos)
    t = torch.tensor(pos.reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float64)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(t)
    E, F = C.cg_energy_forces(t, ij, pw, cell_list=cl)
    Fn = torch.linalg.norm(F.reshape(3 * L, 3), dim=-1)
    junction = max(float(Fn[0]), float(Fn[3 * (L - 1)]))
    return float(Fn.max()), junction, float(E)


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / \
        "measure_bsj_on_circular_reference.log"
    out_path = out_path if out_path.is_absolute() else REPO / "results" / out_path
    out_path.parent.mkdir(exist_ok=True)

    structs = B.load_structures()
    cir = next((s for s in structs if s["name"] == "2OIU"), None)
    pool = [s for s in structs if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]

    lines = []
    def emit(msg=""):
        print(msg)
        lines.append(msg)

    emit("BSJ closure term on a genuinely circular reference (2OIU) vs the linear pool")
    emit(f"production constants: K_BSJ={C.K_BSJ}  BOND_P_NEXT={C.BOND_P_NEXT}  "
         f"K_BSJ_GUIDE={C.K_BSJ_GUIDE}  K_BSJ_CONTACT={C.K_BSJ_CONTACT}  "
         f"PAIR_NN={C.PAIR_NN}  force_cap=5000.0")
    emit(f"linear pool (len(pairs)>=8, 24<=len(pos)<=34): {len(pool)} chains")
    emit()

    hdr = (f"{'structure':12s} {'L':>3s} {'pairs':>5s} {'|P0-PL-1|':>10s} "
           f"{'bsjE':>8s} {'bsjF':>9s} {'guideF':>8s} {'contF':>8s} "
           f"{'maxF':>9s} {'juncF':>9s}")
    emit(hdr)
    emit("-" * len(hdr))

    def row(name, pos, pairs):
        e2e = closure_geometry(pos)
        bt = bs_terms(pos)
        maxf, juncf, _ = full_field(pos, pairs)
        emit(f"{name:12s} {len(pos):3d} {len(pairs):5d} {e2e:10.4f} "
             f"{bt['bsj closure'][0]:8.2f} {bt['bsj closure'][1]:9.2f} "
             f"{bt['bsj guide'][1]:8.2f} {bt['bsj contact'][1]:8.2f} "
             f"{maxf:9.2f} {juncf:9.2f}")

    # circular reference first
    if cir is None:
        emit("2OIU not found in the loader")
    else:
        emit("CIRCULAR reference (2OIU, file order already closed):")
        row("2OIU", cir["pos"], cir["pairs"])
    emit()
    emit("LINEAR control (the pool every validation run used):")
    for s in pool:
        row(s["name"], s["pos"], s["pairs"])

    emit()
    emit("Reading: bsjE = 0.5*K_BSJ*(|P0-PL-1| - 0.590)^2 (kJ/mol); bsjF = K_BSJ*|...| on each")
    emit("junction bead (kJ/mol/nm); guideF / contF the two other BSJ terms; maxF = largest")
    emit("per-bead force norm of the FULL production field; juncF = full-field force norm on the")
    emit("two junction P beads. For a real circle bsjE ~ 0 and bsjF << 5000; on a linear chain")
    emit("bsjF is thousands and dominates the junction beads.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
