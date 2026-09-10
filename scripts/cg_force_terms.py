"""Per-term decomposition of the CG force field at a given geometry.

Extracted so that diagnose_cg_force_terms.py and recalibrate_ff_targets.py share one
implementation -- the terms are the library's own helpers, not a reimplementation, and
there should be exactly one copy of the wiring.

The library's constants are read at call time from torusfold.scheme2.torch_cgsim, so a
caller that rebinds them sees the change here too.
"""
import math

import numpy as np
import torch

import torusfold.scheme2.torch_cgsim as C


def term_energies_forces(pos_nm, pairs_ij, pair_w=None, cell_list=None):
    """Return (terms, energies): per-term force arrays (3L,3) and per-term scalars.

    pos_nm: (1, 3L, 3) torch tensor in nm. Mirrors cg_energy_forces lines 695-794,
    which is the 13 terms that make up the field; GB/SA and Manning are not included
    because they are computed there by autograd on a detached copy.
    """
    B, N, _ = pos_nm.shape
    L = N // 3
    eps = 1e-6
    P = lambda i: 3 * i + 0
    C4 = lambda i: 3 * i + 1
    NN = lambda i: 3 * i + 2

    pi = pairs_ij[:, 0].long() if pairs_ij.numel() else None
    pj = pairs_ij[:, 1].long() if pairs_ij.numel() else None
    dev = pos_nm.device

    terms, energies = {}, {}

    def add(name, F, e):
        terms[name] = F.reshape(3 * L, 3).detach().cpu().numpy()
        energies[name] = float(e.reshape(-1)[0])

    # 1. backbone P-P bonds
    idx = torch.arange(L - 1, device=dev)
    e, f = C._bond_f(pos_nm, P(idx), P(idx + 1), C.K_BB, C.BOND_P_NEXT)
    add("bb bond P-P", f, e)

    # 2/3. intra-bead
    r = torch.arange(L, device=dev)
    e1, f1 = C._bond_f(pos_nm, P(r), C4(r), C.K_INTRA, C.BOND_P_C4)
    e2, f2 = C._bond_f(pos_nm, C4(r), NN(r), C.K_INTRA, C.BOND_C4_N)
    add("intra P-C4'", f1, e1)
    add("intra C4'-N", f2, e2)

    # 4. BSJ closure
    d = pos_nm[:, P(0)] - pos_nm[:, P(L - 1)]
    rb = C._safe_norm(d, dim=-1, keepdim=True, eps=eps)
    add("bsj closure", _pair_force(pos_nm, P(0), P(L - 1),
                                   -C.K_BSJ * (rb - C.BOND_P_NEXT) * d / rb),
        (0.5 * C.K_BSJ * (rb.squeeze(-1) - C.BOND_P_NEXT) ** 2).sum(dim=-1))

    # 5/6. angles and dihedrals
    ea, fa = C._angle_f(pos_nm, C.K_ANGLE, math.cos(C.ANGLE_PPP))
    ed, fd = C._dihedral_f(pos_nm, C.K_DIH, math.cos(C.DIH_PPPP))
    add("angle P-P-P", fa, ea)
    add("dihedral P-P-P-P", fd, ed)

    if pi is not None and len(pi):
        w = pair_w[:len(pi)].to(dev).float() if pair_w is not None else \
            torch.ones(len(pi), device=dev)

        # 7. WC pairing on the base beads
        delta = pos_nm[:, NN(pi)] - pos_nm[:, NN(pj)]
        dist = C._safe_norm(delta, dim=-1, keepdim=True, eps=eps)
        k_e = C.K_PAIR * w[None, :, None]
        add("wc pair N-N",
            _scatter(pos_nm, NN(pi), NN(pj), -k_e * (dist - C.PAIR_NN) * delta / dist),
            (0.5 * k_e * (dist - C.PAIR_NN) ** 2).sum(dim=-1).sum(dim=-1))

        # 10. pair guide
        dg = pos_nm[:, P(pi)] - pos_nm[:, P(pj)]
        rg = C._safe_norm(dg, dim=-1, keepdim=True, eps=eps)
        _, sig = C._sigmoid_f(rg, C.PAIR_NN, C.K_PAIR_GUIDE, 0.2)
        add("pair guide",
            _scatter(pos_nm, P(pi), P(pj),
                     -C.K_PAIR_GUIDE / 0.2 * sig * dg / rg),
            C._sigmoid_f(rg, C.PAIR_NN, C.K_PAIR_GUIDE, 0.2)[0].sum(dim=-1))

        # 13. BPP
        db = pos_nm[:, NN(pi)] - pos_nm[:, NN(pj)]
        rbp = C._safe_norm(db, dim=-1, keepdim=True, eps=eps)
        xb = (C.PAIR_NN - rbp) / 0.3
        sb = torch.sigmoid(xb)
        add("bpp",
            _scatter(pos_nm, NN(pi), NN(pj), -C.K_BPP / 0.3 * sb * db / rbp),
            (-C.K_BPP * C._stable_softplus(xb)).sum(dim=-1).sum(dim=-1))

    # 8. stacking
    if L > 2:
        st = torch.arange(L - 2, device=dev)
        ds = pos_nm[:, P(st)] - pos_nm[:, P(st + 2)]
        rs = C._safe_norm(ds, dim=-1, keepdim=True, eps=eps)
        F = torch.zeros_like(pos_nm)
        fmag = -C.K_STACK * (rs - C.STACK_R0) * ds / rs
        F[:, P(st)] += fmag.squeeze(-1)
        F[:, P(st + 2)] -= fmag.squeeze(-1)
        add("stacking P-P", F, (0.5 * C.K_STACK * (rs - C.STACK_R0) ** 2).sum(-1).sum(-1))

    # 9. clash
    if cell_list is not None:
        ec, fc = C._clash_f(pos_nm, cell_list, C.K_CLASH, C.CLASH_DIST)
        add("clash", fc, ec)

    # 11. BSJ guide
    dbg = pos_nm[:, P(0)] - pos_nm[:, P(L - 1)]
    rbg = C._safe_norm(dbg, dim=-1, keepdim=True, eps=eps)
    _, sbg = C._sigmoid_f(rbg, C.PAIR_NN, C.K_BSJ_GUIDE, 0.2)
    add("bsj guide", _pair_force(pos_nm, P(0), P(L - 1),
                                 -C.K_BSJ_GUIDE / 0.2 * sbg * dbg / rbg),
        C._sigmoid_f(rbg, C.PAIR_NN, C.K_BSJ_GUIDE, 0.2)[0].squeeze(-1))

    # 12. BSJ contact
    if L > 16:
        Fc = torch.zeros_like(pos_nm)
        for off in range(min(8, L // 2)):
            i1, i2 = off, L - 1 - off
            if i1 < i2:
                dc = pos_nm[:, P(i1)] - pos_nm[:, P(i2)]
                rc = C._safe_norm(dc, dim=-1, keepdim=True, eps=eps)
                wc = torch.exp(-0.1 * (rc / C.PAIR_NN))
                fc = -C.K_BSJ_CONTACT * 0.1 / C.PAIR_NN * wc * dc / (rc * rc)
                Fc[:, P(i1)] += fc.squeeze(-1)
                Fc[:, P(i2)] -= fc.squeeze(-1)
        add("bsj contact", Fc, torch.zeros(1, dtype=pos_nm.dtype))

    return terms, energies


def _scatter(pos_nm, ii, jj, f):
    F = torch.zeros_like(pos_nm)
    F[:, ii] += f.squeeze(-1)
    F[:, jj] -= f.squeeze(-1)
    return F


def _pair_force(pos_nm, i, j, f):
    F = torch.zeros_like(pos_nm)
    F[:, i] += f.squeeze(-1)
    F[:, j] -= f.squeeze(-1)
    return F
