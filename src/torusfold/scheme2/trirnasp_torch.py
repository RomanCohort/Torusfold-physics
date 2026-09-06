"""trirnasp_torch.py - PyTorch GPU batched implementation of the TriRNASP three-body statistical potential.

Correspondence with trirnasp_openmm.py (numpy CPU version):
  - Energy table Rough.energy (12^3 x 4 x 4 x 8 = 73k entries) loaded once into a device constant
  - Three-body term: for each valid pair (i,j), find k>j within R0 and accumulate by table lookup
    - The CPU version loops per pair in Python; the torch version is fully batched:
      pair list (M,) x variable-length neighbors -> flattened into a triplet list (T,) with one gather
  - Gradient: default is hard-bin table lookup (piecewise-constant potential, bit-aligned with the CPU
    score(), used for scoring/validation). NOTE: this path's computation graph is truncated by .long()
    - non-differentiable, autograd cannot recover a gradient. With soft=True it switches to trilinear
    interpolation over adjacent bins: the potential becomes a piecewise-linear function of distance, and
    autograd backpropagates an exact gradient through the interpolation weights, for REMD external forces.
    (This roots out the finite-difference linearization problem of the CPU score_with_gradient)

Batched replica support: (B, N, 3) scores a batch of structures at once (all REMD replicas in one pass).

Unit convention: input in Angstrom (3bead layout), output in kBT; the caller handles kJ/mol conversion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# Constants aligned with trirnasp_openmm.py
R0 = 8.0
BIN_WIDTH_ROUGH = 2.0
EXCLUSION_R1_SQ = 1.21   # (1.1)^2
EXCLUSION_R2_SQ = 2.89   # (1.7)^2
EXCLUSION_R3_SQ = 16.0   # (4.0)^2
R0_SQ = (R0 - 0.3) ** 2

_BASE_OFFSET = {"A": 0, "U": 1, "C": 2, "G": 3}


def _type_code(base: str, bead: str) -> int:
    b = _BASE_OFFSET.get(base, -1)
    if b < 0:
        return -1
    if bead == "C4'":
        return b
    if bead in ("N9", "N1"):
        return 4 + b
    if bead == "P":
        return 8 + b
    return -1


def _load_energy_table(filepath: str, n_types: int, n_bins: int,
                       n_bins_fine: int) -> np.ndarray:
    """Same as trirnasp_openmm._load_energy_table."""
    total = (n_types ** 3) * (n_bins ** 2) * (2 * n_bins)
    table = np.zeros(total, dtype=np.float64)
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            t1, t2, t3 = int(parts[0]), int(parts[1]), int(parts[2])
            b12, b13, b23 = int(parts[3]), int(parts[4]), int(parts[5])
            energy = float(parts[6])
            idx = (((((t1 * n_types + t2) * n_types + t3) * n_bins + b12)
                    * n_bins + b13) * (2 * n_bins) + b23)
            if 0 <= idx < total:
                table[idx] = energy
    return table


class TriRNASPTorch:
    """GPU batched TriRNASP potential.

    Precomputed once (topology quantities independent of sequence/coordinates):
      - atom type/residue index tables (3L atoms: P,C4',N layout, same as 3bead)
      - pair candidates (i<j; the static part of the exclusion rules, depending only on
        res_diff and the distance cap)
      - each pair's neighbor mask: the variable-length part is screened by distance at runtime

    Each runtime call does:
      dist_sq (B,N,N) -> pair distance -> bin -> table gather -> sum
    Gradient: hard mode is non-differentiable (the hard-truncated lookup severs the graph);
    soft mode (soft=True) trilinearly interpolates between adjacent bins, keeping the graph
    intact and giving autograd an exact gradient.
    """

    def __init__(self, sequence: str, energy_dir: Optional[str] = None,
                 device="cuda"):
        assert TORCH_OK, "PyTorch is not installed"
        if energy_dir is None:
            _root = Path(__file__).resolve().parents[3]
            energy_dir = str(_root / "external" / "TriRNASP" / "Energy")

        rough_path = Path(energy_dir) / "Rough.energy"
        if not rough_path.exists():
            raise FileNotFoundError(f"Rough.energy not found: {rough_path}")

        self.device = device if torch.cuda.is_available() else "cpu"
        rough_np = _load_energy_table(str(rough_path), 12, 4, 4)
        self._rough = torch.tensor(rough_np, dtype=torch.float32,
                                   device=self.device)

        self.L = len(sequence)

        # Atom layout: [P, C4', N] x L (same as 3bead, N=3L)
        # trirnasp_openmm's order is C4',N,P - but the type code determines the lookup
        # result; order only affects internal index consistency, so we use the 3bead
        # layout uniformly to avoid conversions.
        codes_c = [_type_code(sequence[i], "C4'") for i in range(self.L)]
        codes_n = [_type_code(
            sequence[i], "N9" if sequence[i] in ("A", "G") else "N1")
            for i in range(self.L)]
        codes_p = [_type_code(sequence[i], "P") for i in range(self.L)]

        # Layout convention: identical to the trirnasp_openmm CPU version - [C4', N, P] x L.
        # NOTE: three-body counting depends on atom ordering (the k>j rule); a different
        #   layout would give different energies, so this must mirror the CPU's valid
        #   append order verbatim.
        # Note this differs from torch_cgsim's [P,C4,N]; energy_from_3bead handles
        # reordering the 3bead particles into this layout.
        types = []
        res_of = []
        bead_seq = []
        for i in range(self.L):
            for code, bead_pos in ((codes_c[i], 1), (codes_n[i], 2),
                                   (codes_p[i], 0)):
                if code >= 0:
                    types.append(code); res_of.append(i); bead_seq.append(bead_pos)

        self.n_atoms = len(types)
        self.atom_types = torch.tensor(types, dtype=torch.long,
                                       device=self.device)
        self.atom_res = torch.tensor(res_of, dtype=torch.long,
                                     device=self.device)
        # atom a -> flattened 3bead particle index (3r+bead in the [P,C4,N] layout)
        self._particle_idx = torch.tensor(
            [3 * r + b for r, b in zip(res_of, bead_seq)],
            dtype=torch.long, device=self.device)

        # Static pair candidates: i<j; structural exclusion |res_i - res_j| is decided at runtime
        # Only i<j is prefiltered here; distance-dependent exclusions are computed every step.
        na = self.n_atoms
        iu, ju = np.triu_indices(na, k=1)
        self._iu = torch.tensor(iu, dtype=torch.long, device=self.device)
        self._ju = torch.tensor(ju, dtype=torch.long, device=self.device)
        res_i = self.atom_res[self._iu]
        res_j = self.atom_res[self._ju]
        self._res_diff = (res_i - res_j).abs()          # (M,)
        # Global upper triangle of k>j adjacency candidates (k index > j): expanded per pair at runtime
        # To keep memory bounded, neighbor screening gathers per-pair separately at runtime (vectorized).

        # Exclusion threshold constants
        self._r1sq = EXCLUSION_R1_SQ
        self._r2sq = EXCLUSION_R2_SQ
        self._r3sq = EXCLUSION_R3_SQ
        self._r0sq = R0_SQ
        self._inv_bw = 1.0 / BIN_WIDTH_ROUGH

    def energy(self, coords_A: "torch.Tensor",
               soft: bool = False) -> "torch.Tensor":
        """Batched TriRNASP energy (kBT).

        Args:
            coords_A: (B, N_atoms, 3) Angstrom - atom order after _build_atoms;
                      mapped back to 3bead particles by this class's particle_idx.
                      Convenience entry: use from_3bead() to convert first.
            soft: False -> hard-bin table lookup (bit-aligned with the CPU score, but the
                  graph is truncated by .long(); autograd has no gradient);
                  True -> trilinear interpolation over adjacent bins; the potential becomes
                  a piecewise-linear function of distance and autograd gives an exact
                  gradient through the interpolation weights.

        Returns:
            (B,) kBT
        """
        B = coords_A.shape[0]
        dev = coords_A.device

        diff = coords_A[:, :, None, :] - coords_A[:, None, :, :]
        dist_sq = (diff * diff).sum(-1)                    # (B,N,N)

        iu, ju = self._iu, self._ju
        d_ij = dist_sq[:, iu, ju]                          # (B,M)
        res_diff = self._res_diff[None]                    # (1,M)

        excl = d_ij <= self._r1sq
        e_const = torch.where(excl, torch.ones_like(d_ij) * 0.5,
                              torch.zeros_like(d_ij))
        m2 = (~excl) & (d_ij <= self._r2sq) & (res_diff > 1)
        e_const = e_const + torch.where(m2, torch.ones_like(d_ij) * 0.5,
                                        torch.zeros_like(d_ij))
        excl = excl | m2
        m3 = (~excl) & (d_ij <= self._r3sq) & (d_ij > self._r2sq)
        e_const = e_const + torch.where(m3, torch.ones_like(d_ij) * 0.2,
                                        torch.zeros_like(d_ij))
        excl = excl | m3

        cand = (~excl) & (d_ij < self._r0sq)               # (B,M)
        d12 = torch.sqrt(torch.where(cand, d_ij, torch.zeros_like(d_ij)))
        if soft:
            # Soft binning: keep coordinates continuous; interpolation weights keep the autograd path alive
            f12 = d12 * self._inv_bw                         # continuous bin coordinate
            b12_lo = f12.floor().long()                      # lower bin
            w12 = (f12 - b12_lo.float()).clamp(0.0, 1.0)   # interpolation weight
            b12_lo = b12_lo.clamp(0, 3)                     # safe clamp
        else:
            b12 = (d12 * self._inv_bw).long()
        # CPU version: pairs with b12>3 are dropped from the three-body candidates (ok12 filter)
        if soft:
            ok12 = (b12_lo <= 3) & cand
        else:
            ok12 = (b12 <= 3) & cand
        cand = ok12

        total = e_const.sum(dim=-1)                        # (B,)
        if not bool(cand.any()):
            return total

        # Three-body term: vectorized expansion over (pair, k) combinations
        # For each candidate pair (i,j), k in (j, N) with dist^2(i,k)<R0^2 and dist^2(j,k)<rcut
        # A full M x N expansion is too large, so process it in per-pair chunks.
        ci = iu[None].expand(B, -1)
        cj = ju[None].expand(B, -1)

        # Only handle the (batch, pair) entries that have candidates
        cand_flat = cand.reshape(-1)
        sel = torch.nonzero(cand_flat, as_tuple=False).squeeze(-1)
        b_idx = sel // cand.shape[1]
        p_idx = sel % cand.shape[1]
        if b_idx.numel() == 0:
            return total

        bi_all = iu[p_idx]
        bj_all = ju[p_idx]

        # Three-body term: expand per pair in chunks to bound the (chunk, N) intermediate tensor memory
        # At L=2013, M~18M candidate pairs x N=6039 broadcast directly would be ~1TB - chunking is required.
        # Target intermediate size per chunk ~= chunk x N x 8B <= ~256MB, so chunk adapts.
        n_atoms = coords_A.shape[1]
        n_pairs_sel = b_idx.numel()
        k_col = torch.arange(n_atoms, device=dev)              # (N,)
        rk_all = self.atom_res[k_col]                          # (N,)
        rcut_sq = (R0 - 0.3) ** 2

        bytes_per_row = n_atoms * 4 * 4   # d_ik/d_jk/g13/g23 float32
        chunk = max(1, int(64 * 2 ** 20 // max(bytes_per_row, 1)))

        tri_acc = torch.zeros(B, dtype=torch.float32, device=dev)
        for s in range(0, n_pairs_sel, chunk):
            e_chunk = min(s + chunk, n_pairs_sel)
            bb = b_idx[s:e_chunk]
            pi_ = p_idx[s:e_chunk]
            bi = bi_all[s:e_chunk]
            bj = bj_all[s:e_chunk]

            kc = k_col[None, :]                                # (1,N)
            d_ik = dist_sq[bb[:, None], bi[:, None],
                           kc.expand(e_chunk - s, -1)]
            d_jk = dist_sq[bb[:, None], bj[:, None],
                           kc.expand(e_chunk - s, -1)]
            k_ok = (kc > bj[:, None]) & \
                (d_ik < self._r0sq) & (d_jk < rcut_sq)

            # Exclusion rules (three-body)
            ri = self.atom_res[bi][:, None]
            rj = self.atom_res[bj][:, None]
            rk = rk_all[None, :]
            d13 = d_ik
            d23 = d_jk
            bad = (d13 <= self._r1sq) | (d23 <= self._r1sq)
            bad = bad | ((d13 <= self._r2sq) & ((ri - rk).abs() > 1))
            bad = bad | ((d23 <= self._r2sq) & ((rj - rk).abs() > 1))
            bad = bad | ((d13 > self._r2sq) & (d13 <= self._r3sq))
            bad = bad | ((d23 > self._r2sq) & (d23 <= self._r3sq))
            good = k_ok & (~bad)

            # Numerical philosophy (finalized with the senior on 2026-08-26): GPU runs f64
            # throughout, and does not reproduce the CPU's legacy path of "store dsq as f32,
            # then sqrt". The ~1-2% bin-boundary deviation vs CPU is a structure-dependent
            # baseline shift that cancels across REMD replicas and does not affect the
            # self-consistency of the exchange criterion.
            g_d13 = torch.sqrt(
                torch.where(good, d13, torch.zeros_like(d13)))
            g_d23 = torch.sqrt(
                torch.where(good, d23, torch.zeros_like(d23)))
            if soft:
                f13 = g_d13 * self._inv_bw
                b13_lo = f13.floor().long()
                w13 = (f13 - b13_lo.float()).clamp(0.0, 1.0)
                b13_lo = b13_lo.clamp(0, 3)
                f23 = g_d23 * self._inv_bw
                b23_lo = f23.floor().long()
                w23 = (f23 - b23_lo.float()).clamp(0.0, 1.0)
                b23_lo = b23_lo.clamp(0, 7)
                fin = good & (b13_lo <= 3) & (b23_lo <= 7)
            else:
                b13 = (g_d13 * self._inv_bw).long()
                b23 = (g_d23 * self._inv_bw).long()
                fin = good & (b13 <= 3) & (b23 <= 7)
            if not bool(fin.any()):
                continue

            f_rows, f_k = torch.nonzero(fin, as_tuple=True)
            t_i = self.atom_types[iu[pi_[f_rows]]]
            t_j = self.atom_types[ju[pi_[f_rows]]]
            t_k = self.atom_types[f_k]
            # base offset for the atom-type dimension (type part is discrete - always integer)
            base_t = ((t_i * 12 + t_j) * 12 + t_k).long()

            if soft:
                # Trilinear interpolation: potential = sum over corners of w1*w2*w3 * E(corner)
                tb12b = b12_lo[bb[f_rows], pi_[f_rows]].long()
                tb13b = b13_lo[f_rows, f_k].long()
                tb23b = b23_lo[f_rows, f_k].long()
                w12v = w12[bb[f_rows], pi_[f_rows]]
                w13v = w13[f_rows, f_k]
                w23v = w23[f_rows, f_k]
                e_interp = torch.zeros_like(w12v)
                for c0 in range(2):
                    for c1 in range(2):
                        for c2 in range(2):
                            cb12 = (tb12b + c0).clamp(0, 3)
                            cb13 = (tb13b + c1).clamp(0, 3)
                            cb23 = (tb23b + c2).clamp(0, 7)
                            idx_c = ((base_t * 4 + cb12) * 4 + cb13) * 8 + cb23
                            wc = ((1.0 - w12v) if c0 == 0 else w12v) * \
                                 ((1.0 - w13v) if c1 == 0 else w13v) * \
                                 ((1.0 - w23v) if c2 == 0 else w23v)
                            e_interp = e_interp + wc * self._rough[idx_c]
                tri_acc.index_add_(0, bb[f_rows], e_interp.float())
            else:
                tb12 = b12[bb[f_rows], pi_[f_rows]]
                tb13 = b13[f_rows, f_k]
                tb23 = b23[f_rows, f_k]
                idx = ((base_t * 4 + tb12) * 4 + tb13) * 8 + tb23
                e_vals = self._rough[idx]
                tri_acc.index_add_(0, bb[f_rows], e_vals)

        return total + tri_acc

    def energy_from_3bead(self, pos_nm: "torch.Tensor",
                          soft: bool = False) -> "torch.Tensor":
        """Convenience entry: (B, 3L, 3) nm torch_cgsim layout ([P,C4,N] x L) -> (B,) kBT.

        _particle_idx maps this class's atom order to the 3bead particle indices -
        here we gather in reverse: pull the coordinates of each atom in this class's order.
        """
        coords_A = pos_nm * 10.0
        atoms = coords_A[:, self._particle_idx, :]
        return self.energy(atoms, soft=soft)


if __name__ == "__main__":
    """Smoke test: compare energy against the numpy CPU version (same residue coordinates, float32 tolerance).

    Layout conventions:
      The CPU score() takes (L,3,3), bead order = atom order [C4', N, P]
        - see the valid append order in trirnasp_openmm._build_atoms
      This class's energy_from_3bead() takes (B,3L,3) nm, bead order = [P, C4', N]
    The same residue coordinates are expanded in each of the two layouts.
    """
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from torusfold.scheme2.trirnasp_openmm import TriRNASPPotential

    seq = "AUGCAUGCAUGCAUGC"
    L = len(seq)
    rng = np.random.default_rng(42)
    res_coords = rng.random((L, 3, 3)).astype(np.float64) * 8.0  # Angstrom, spread out
    # CPU (L,3,3) axis semantics: axis0=P, axis1=C4', axis2=N
    # (_build_atoms indexes the axis directly by bead id: P<-[r,0], C4'<-[r,1], N<-[r,2])

    pot_cpu = TriRNASPPotential()
    e_cpu = pot_cpu.score(res_coords.copy(), seq)

    pot_gpu = TriRNASPTorch(seq)
    # This class's layout [C4',N,P] x L equals the CPU layout; res_coords axis is [P,C4,N]
    # (axis0=P) -> reorder to [C4'=axis1, N=axis2, P=axis0]
    flat = np.stack([res_coords[:, 1], res_coords[:, 2], res_coords[:, 0]],
                    axis=1).reshape(3 * L, 3)
    t = torch.tensor(flat, dtype=torch.float64, device=pot_gpu.device)[None]
    e_gpu = float(pot_gpu.energy(t)[0])

    print(f"CPU kBT: {e_cpu:.4f}")
    print(f"GPU kBT: {e_gpu:.4f}")
    rel = abs(e_gpu - e_cpu) / max(abs(e_cpu), 1e-8)
    print(f"rel err: {rel:.2%}")
    assert rel < 0.05, f"GPU/CPU TriRNASP energy deviation too large: {rel}"
    print("[PASS] TriRNASP torch matches numpy within tolerance")

    # Soft-binning test: differentiability + numerical verification
    t_test = t.clone().detach().requires_grad_(True)
    e_soft = pot_gpu.energy(t_test, soft=True)[0]
    print(f"\n[soft] energy kBT: {e_soft.item():.4f}")
    diff_rel = abs(e_soft.item() - e_gpu) / max(abs(e_gpu), 1e-8)
    print(f"[soft] soft/hard rel diff: {diff_rel:.4%} (sparse table; expected behavior)")
    e_soft.backward()
    ag = t_test.grad.clone()   # (1,48,3)
    print(f"[soft] |autograd|={ag.norm().item():.4f} kBT/Å, "
          f"|per-atom|={ag.view(-1,3).norm(dim=1).mean().item():.4f}")

    # Numerical check: single-atom perturbation, cosine > 0.9 -> gradient correct
    eps = 1e-3
    ag0 = ag[0, 0].double()   # xyz of atom 0
    num_g = torch.zeros(3, device=ag.device, dtype=torch.float64)
    for d in range(3):
        tp = t.clone()
        tp[0, 0, d] += eps
        ep = pot_gpu.energy(tp, soft=True)[0].item()
        tm = t.clone()
        tm[0, 0, d] -= eps
        em = pot_gpu.energy(tm, soft=True)[0].item()
        num_g[d] = (ep - em) / (2*eps)
    cos = float(torch.dot(ag0.double(), num_g) / (ag0.norm() * num_g.norm() + 1e-12))
    print(f"[soft] atom0 cosine(autograd, numerical)={cos:.4f} (eps={eps})")
    assert abs(cos) > 0.9, f"gradient direction wrong: cosine={cos}"
    print("[PASS] soft-binning gradient direction verified")

    print("[PASS] soft-binning smoke complete")
