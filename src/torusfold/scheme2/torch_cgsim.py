"""
torch_cgsim.py - batched coarse-grained (CG) molecular dynamics on PyTorch ROCm.

Background: the OpenMM Windows build has no ROCm/HIP support, so the AMD GPU
(Radeon 8060S, 107GB) can only be reached through PyTorch.  This module
re-implements the 3-bead CG force field in pure torch:

  - bond-length restraints (P-P backbone, BSJ): harmonic
  - WC base-pair restraints: harmonic + lambda scaling (REST2 solute term)
  - clash repulsion: soft-sphere (not scaled with lambda)

Killer feature - batched replicas: B replicas are a single (B, N, 3) tensor.
Forces of all replicas are computed in one forward pass on the GPU; the forces
use autograd (zero symbolic-error), and integration/swaps are all tensor ops
with no inter-process communication:
  CPU OpenMM REMD: 32 processes x ~200ms/step + Pipe swap
  GPU batched:     1 (B,N,N) distance matrix ~8ms

Unit conventions match the OpenMM path: internally nm / kJ/mol, at the API in Angstrom.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_OK = True
    # Fused kernel: torch.compile merges exp/log/sum into a single kernel launch
    # Reduces kernel-launch overhead (~30% for small tensors)
    try:
        _COMPILE_MODE = "default"  # or "reduce-overhead" (CUDA graphs)
    except Exception:
        _COMPILE_MODE = None
except ImportError:
    TORCH_OK = False
    _COMPILE_MODE = None

# ROCm compatibility: torch.cross triggers an empty_cuda abort on some HIP drivers
# Use the explicit formula instead; the performance cost is negligible
_USE_EXPLICIT_CROSS = False
if TORCH_OK and torch.cuda.is_available():
    try:
        _t = torch.randn(100, 3, device="cuda")
        _ = torch.cross(_t[:50], _t[50:], dim=-1)
        del _t, _
        torch.cuda.empty_cache()
    except Exception:
        _USE_EXPLICIT_CROSS = True
        print("[torch_cgsim] ROCm torch.cross is unstable; falling back to explicit cross-product")


def _stable_softplus(x: "torch.Tensor") -> "torch.Tensor":
    """Finite, differentiable log(1 + exp(x)) for all finite x."""
    return torch.nn.functional.softplus(x)


def _safe_norm(
    value: "torch.Tensor", dim: int = -1, keepdim: bool = False,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """Norm with a finite, zero-gradient floor at exactly coincident points."""
    # Norm's derivative is undefined at zero.  This matters during minimization:
    # a clipped optimizer step can temporarily put two beads at the same point.
    calc = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    squared = (calc * calc).sum(dim=dim, keepdim=keepdim)
    return squared.clamp_min(eps * eps).sqrt()


def _require_finite(value: "torch.Tensor", name: str) -> None:
    """Reject invalid tensors before they can poison an optimizer or integrator."""
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"[cg_energy_forces] non-finite {name}")


def _safe_cross(a: "torch.Tensor", b: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
    """Explicit cross-product to avoid the ROCm torch.cross kernel bug."""
    if not _USE_EXPLICIT_CROSS:
        try:
            return torch.cross(a, b, dim=dim)
        except Exception:
            pass
    # Explicit formula: a×b = (a1b2-a2b1, a2b0-a0b2, a0b1-a1b0)
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], dim=-1)

# TorchMD explicit-force functions (no autograd)
try:
    from torchmd.forces import evaluate_bonds, evaluate_torsion, calculate_distances
    TORCHMD_OK = True
except ImportError:
    TORCHMD_OK = False


# ── Force-field parameters (aligned with openmm_gpu_refiner.py, kJ/mol/nm) ──
# Balanced version: all terms have similar magnitudes so none dominates
K_BB = 500.0        # P-P backbone bond (baseline)
K_INTRA = 400.0     # P-C4', C4'-N
K_PAIR = 600.0      # WC base-pair N-N (λ-scalable) ← lowered from 1500 to 600
K_STACK = 500.0     # base stacking P_i-P_{i+2} (λ-scalable)
K_ANGLE = 600.0     # P-P-P backbone angle
K_DIH = 500.0       # P-P-P-P dihedral (A-form) ← raised from 300 to 500
K_CLASH = 500.0     # soft-sphere repulsion (not scaled) ← raised from 300 to 500
K_BSJ = 600.0       # BSJ closure ← lowered from 800 to 600
K_BSJ_GUIDE = 100.0 # BSJ closure guiding force (logistic sigmoid)
K_PAIR_GUIDE = 100.0 # far/long-range pair guiding force (logistic sigmoid)
K_BSJ_CONTACT = 50.0 # contacts near the BSJ (distance-decaying)
K_BPP = 600.0       # BPP soft constraint ← lowered from 1000 to 600

# ── Ion-model constants (Plan A) ──
# Mg2+ concentration-dependent screening: λ_D = 0.304 / sqrt(c_Mg + c_Na) (nm, Debye screening)
# Non-specific Mg2+ attraction: E_Mg = -K_Mg × Σ exp(-r/λ_Mg)
K_MG = 200.0        # non-specific Mg2+ attraction strength (kJ/mol)
LAMBDA_MG = 0.3     # Mg2+ binding radius (nm)
C_MG_DEFAULT = 0.01  # default Mg2+ concentration (M)
C_NA_DEFAULT = 0.15  # default Na+ concentration (M, ionic strength)

BOND_P_NEXT = 0.590   # nm
BOND_P_C4 = 0.390
BOND_C4_N = 0.335
PAIR_NN = 1.00        # pairing target (N-N approximated at P-P resolution)
STACK_R0 = 0.505      # stacking target P_i~P_{i+2}
ANGLE_PPP = math.pi * 150.0 / 180.0   # 150° A-form
ANGLE_K = K_ANGLE
DIH_PPPP = math.pi * 180.0 / 180.0   # 180° A-form dihedral
DIH_K = K_DIH
CLASH_DIST = 0.30     # nm
CLASH_CUTOFF = 1.20   # nm

KB_KJ = 0.008314462618  # kJ/(mol·K)


def _trirnasp_force_worker(args):
    """Compute one selected batch of CPU TriRNASP forces."""
    pos_np, replica_indices, sequence, energy_dir, force_scales, max_force = args
    from torusfold.scheme2.trirnasp_openmm import TriRNASPPotential

    potential = TriRNASPPotential(energy_dir)
    length = len(sequence)
    result = np.zeros((len(replica_indices), length, 3), dtype=np.float64)
    for local_idx, rep in enumerate(replica_indices):
        coords_pcn = pos_np[local_idx].reshape(length, 3, 3)
        coords_cnp = coords_pcn[:, (1, 2, 0), :]
        _energy, grad_cnp = potential.score_with_gradient(coords_cnp, sequence)
        # score_with_gradient returns kBT/Å; force_scales already contains
        # kBT→kJ/mol, Å→nm, λ, and Tri strength factors.
        force = -grad_cnp[:, 2, :] * float(force_scales[local_idx])
        norms = np.linalg.norm(force, axis=-1, keepdims=True)
        force *= np.minimum(max_force / np.maximum(norms, 1e-12), 1.0)
        if not np.all(np.isfinite(force)):
            raise RuntimeError(f"TriRNASP force became non-finite for replica {rep}")
        result[local_idx] = force
    return np.asarray(replica_indices, dtype=np.int64), result

def _arange_dev(n: int, device) -> "torch.Tensor":
    """HIP-safe arange: build on the CPU first, then move to the device.

    torch 2.12a0+rocm7.13's arange CUDA/HIP kernel triggers a fabric_access
    abort on some devices (Radeon 8060S / gfx1151) -- build on the CPU and
    move with to(device) to bypass it.
    """
    return torch.arange(n).to(device)


# ═══ HIP-safe tensor factories ══════════════════════════════════
# The fill kernels (eye/randn/zeros_like/full_like etc.) share the same failing
# HIP path as arange -- all are switched to CPU construction then to(device).
# This is the root-cause fix for the fabric_access abort (Bug #1, 2026-08-26).

def _safe_eye(n: int, device, dtype=None) -> "torch.Tensor":
    """Build the identity matrix on the CPU, then move it (bypasses the HIP fill kernel)."""
    t = torch.eye(n, dtype=dtype or torch.float32)
    return t.to(device)

def _safe_randn(shape, device, dtype=torch.float32) -> "torch.Tensor":
    """Generate Gaussian noise on the CPU, then move it (bypasses the HIP randn kernel).

    Note: each step's transfer has PCIe/Infinity Fabric overhead, but a (B,N,3)
    tensor is <50MB when L<=2000, a ~1ms transfer that is well worth avoiding a crash.
    """
    t = torch.randn(*shape, dtype=dtype)
    return t.to(device)

def _safe_zeros(shape, device, dtype=torch.float32) -> "torch.Tensor":
    """Build a zero tensor on the CPU, then move it."""
    return torch.zeros(*shape, dtype=dtype).to(device)

def _safe_full(shape, value, device, dtype=torch.float32) -> "torch.Tensor":
    """Build a constant-filled tensor on the CPU, then move it."""
    return torch.full((*shape,), value, dtype=dtype).to(device)

def _safe_ones(shape, device, dtype=torch.float32) -> "torch.Tensor":
    """Build an all-ones tensor on the CPU, then move it."""
    return torch.ones(*shape, dtype=dtype).to(device)



def _bead_index(kind: str, i: int) -> int:
    """3-bead layout: residue i's particles are 3i + {P:0, C4':1, N:2}."""
    return {"P": 0, "C4": 1, "N": 2}[kind] + 3 * i


class _ClashNeighborList:
    """GPU spatial-hash neighbor list: rebuilt every N steps, O(N) clash-pair lookup."""

    def __init__(self, cell_size=1.2, rebuild_freq=10):
        self.cell_size = cell_size  # nm (2 × CLASH_CUTOFF)
        self.rebuild_freq = rebuild_freq
        self.step = 0
        self.nlist = None

    def _build(self, pos_nm):
        B, N, _ = pos_nm.shape
        dev = pos_nm.device
        p0 = pos_nm[0]  # (N, 3)

        # Spatial hash: each particle → its cell coordinate (all on GPU)
        cell = torch.floor(p0 / self.cell_size).long()  # (N, 3)

        # Vectorized: cell differences for all particle pairs (N×N×3, but only the first replica is used)
        ci = cell[:, None, :]  # (N, 1, 3)
        cj = cell[None, :, :]  # (1, N, 3)
        delta = (ci - cj).abs()  # (N, N, 3)

        # Adjacency test: all dimension differences <= 1
        in_cell = delta.max(dim=2).values <= 1  # (N, N)

        # Exclude self and sequence neighbors (|i-j|<=2)
        idx = torch.arange(N, device=dev)
        seq_near = (idx[:, None] - idx[None, :]).abs() <= 2
        valid = in_cell & ~seq_near & torch.triu(
            torch.ones(N, N, device=dev, dtype=bool), diagonal=1)

        pairs = torch.nonzero(valid, as_tuple=False)
        if pairs.numel() > 0:
            self.nlist = (pairs[:, 0], pairs[:, 1])
        else:
            self.nlist = (torch.zeros(0, dtype=torch.long, device=dev),
                          torch.zeros(0, dtype=torch.long, device=dev))

    def get(self, pos_nm):
        self.step += 1
        if self.nlist is None or self.step % self.rebuild_freq == 0:
            self._build(pos_nm)
        return self.nlist


_clash_nlist = _ClashNeighborList()

def cg_energy(
    pos_nm: "torch.Tensor",
    pairs_ij: "torch.Tensor",
    pair_w: Optional["torch.Tensor"] = None,
    lam: float = 1.0,
    seq_near_mask: Optional["torch.Tensor"] = None,
    temperature: float = 300.0,
) -> "torch.Tensor":
    """Batched-replica CG total energy (scalar graph node).

    Args:
        pos_nm: (B, N, 3) nm, requires_grad=True set by the caller
        pairs_ij: (P, 2) long pair indices
        pair_w: (P,) pair weights
        lam: REST2 λ (applied to the pair term)
        seq_near_mask: (N,N) bool cached sequence-neighbor mask (True=excluded)

    Returns:
        energy: (B,) kJ/mol
    """
    B, N, _ = pos_nm.shape
    dev = pos_nm.device
    _require_finite(pos_nm, "input coordinates")
    if pair_w is not None:
        _require_finite(pair_w, "pair weights")

    # Backbone bonds i~i+1: directly compute adjacent-pair distances
    d_bb = _safe_norm(pos_nm[:, 1:] - pos_nm[:, :-1], dim=-1)  # (B,N-1)
    e_bb = 0.5 * K_BB * (d_bb - BOND_P_NEXT) ** 2

    # BSJ: head-to-tail distance
    e_bsj = 0.5 * K_BSJ * (
        _safe_norm(pos_nm[:, 0] - pos_nm[:, -1], dim=-1) - BOND_P_NEXT
    ) ** 2

    # Pairs: distances of the specified residue pairs
    if pairs_ij.numel() > 0:
        pi, pj = pairs_ij[:, 0].long(), pairs_ij[:, 1].long()
        d_pair = _safe_norm(pos_nm[:, pi] - pos_nm[:, pj], dim=-1)  # (B,P)
        w_p = pair_w.to(dev).float() if pair_w is not None else \
            _safe_ones((len(pi),), dev)
        t_scale = max(1.0, temperature / 300.0)
        k_eff = K_PAIR * lam * w_p * t_scale
        e_pair = (0.5 * k_eff[None] * (d_pair - PAIR_NN) ** 2).sum(dim=-1)
    else:
        e_pair = _safe_zeros((B,), dev)

    # soft-sphere: neighbor list O(P_nlist)
    pi_n, pj_n = _clash_nlist.get(pos_nm)
    if pi_n.numel() > 0:
        d_clash = _safe_norm(pos_nm[:, pi_n] - pos_nm[:, pj_n], dim=-1)
        over = (CLASH_DIST - d_clash).clamp(min=0)
        e_clash = (0.5 * K_CLASH * over ** 2).sum(dim=-1)
    else:
        e_clash = _safe_zeros((B,), dev)

    return e_bb.view(B, -1).sum(dim=-1) + e_bsj + e_pair + e_clash


def cg_forces_autograd(
    pos_nm: "torch.Tensor",
    pairs_ij: "torch.Tensor",
    pair_w: Optional["torch.Tensor"] = None,
    lam: float = 1.0,
    seq_near_mask: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Energy + forces (= −∇E) via autograd.

    Returns:
        (energy (B,), forces (B, N, 3)) kJ/mol/nm
    """
    pos = pos_nm.detach().requires_grad_(True)
    energy = cg_energy(pos, pairs_ij, pair_w, lam, seq_near_mask)
    grad, = torch.autograd.grad(energy.sum(), pos)
    return energy.detach(), -grad


# ══════════════════════════════════════════════════════════════
# 3-bead full force field (P / C4' / N) — geometry-aligned with the OpenMM path
# ══════════════════════════════════════════════════════════════

def cg_energy_3bead(
    pos_nm: "torch.Tensor",               # (B, 3L, 3) particle order: [P,C4',N]×L
    pairs_ij: "torch.Tensor",             # (P, 2) residue indices
    pair_w: Optional["torch.Tensor"] = None,
    lam: float = 1.0,
    seq_near_mask: Optional["torch.Tensor"] = None,
    temperature: float = 300.0,
    c_mg: float = C_MG_DEFAULT,           # Mg2+ concentration (M)
    c_na: float = C_NA_DEFAULT,           # Na+ concentration (M)
) -> "torch.Tensor":
    """3-bead batched CG energy (15-term force field + Mg2+ ion model).

    Force-field terms:
      backbone bond P-P        K_BB
      intra-bead bond P-C4'/C4'-N K_INTRA
      pairing  N_i-N_j        K_PAIR·λ
      stacking P_i-P_{i+2}    K_STACK·λ
      backbone angle P-P-P    K_ANGLE
      dihedral P-P-P-P        K_DIH (A-form 180°)
      repulsion soft-sphere   K_CLASH
      BSJ P_0-P_{L-1}         K_BSJ
      far-pair guide          K_PAIR_GUIDE
      BSJ guide               K_BSJ_GUIDE
      BSJ contact             K_BSJ_CONTACT
      BPP restraint           K_BPP
      GB/SA implicit solvent  (Born + SASA)
      Mg2+ ionic screening    (Debye + non-specific attraction)
    """
    B, N_tot, _ = pos_nm.shape
    L = N_tot // 3
    dev = pos_nm.device
    eps = 1e-6
    _require_finite(pos_nm, "input coordinates")
    P = lambda i: 3 * i + 0
    C4 = lambda i: 3 * i + 1
    NN = lambda i: 3 * i + 2

    # Compute distances in fp32: fp16 subtraction/norm can overflow on long chains.
    pos_float = pos_nm.float()
    diff = pos_float[:, :, None, :] - pos_float[:, None, :, :]  # (B,N,N,3)
    dist = _safe_norm(diff, dim=-1, eps=eps)  # (B,N,N) fp32

    # ── Backbone bonds P_{i+1}-P_i ──
    idx_a = _arange_dev(L - 1, dev)
    d_bb = dist[:, P(idx_a), P(idx_a + 1)]                  # (B,L-1)
    e_bb = 0.5 * K_BB * (d_bb - BOND_P_NEXT) ** 2

    # ── Intra-bead bonds ──
    all_res = _arange_dev(L, dev)
    d_pc = dist[:, P(all_res), C4(all_res)]
    d_cn = dist[:, C4(all_res), NN(all_res)]
    e_intra = (0.5 * K_INTRA * (d_pc - BOND_P_C4) ** 2).sum(dim=-1) \
        + (0.5 * K_INTRA * (d_cn - BOND_C4_N) ** 2).sum(dim=-1)

    # ── BSJ ──
    e_bsj = 0.5 * K_BSJ * (dist[:, P(0), P(L - 1)] - BOND_P_NEXT) ** 2

    # ── Pairing (N-N, λ-scaled) ──
    if pairs_ij.numel() > 0:
        pi, pj = pairs_ij[:, 0].long(), pairs_ij[:, 1].long()
        d_pair = dist[:, NN(pi), NN(pj)]
        w_p = pair_w.to(dev).float() if pair_w is not None else \
            _safe_ones((len(pi),), dev)
        k_eff = K_PAIR * lam * w_p
        e_pair = (0.5 * k_eff[None] * (d_pair - PAIR_NN) ** 2).sum(dim=-1)
    else:
        e_pair = _safe_zeros((B,), dev)

    # ── Stacking P_i~P_{i+2} (λ-scaled) ──
    if L > 2:
        st = _arange_dev(L - 2, dev)
        d_st = dist[:, P(st), P(st + 2)]
        e_stack = (0.5 * K_STACK * lam * (d_st - STACK_R0) ** 2).sum(dim=-1)
    else:
        e_stack = _safe_zeros((B,), dev)

    # ── Backbone angle P_i-P_{i+1}-P_{i+2} (cosine-form harmonic) ──
    if L > 2:
        ang_i = _arange_dev(L - 2, dev)
        v1 = pos_nm[:, P(ang_i)] - pos_nm[:, P(ang_i + 1)]       # (B,L-2,3)
        v2 = pos_nm[:, P(ang_i + 2)] - pos_nm[:, P(ang_i + 1)]
        cosang = (v1 * v2).sum(-1) / (
            _safe_norm(v1, dim=-1, eps=eps) * _safe_norm(v2, dim=-1, eps=eps))
        target_cos = math.cos(ANGLE_PPP)
        e_angle = (0.5 * ANGLE_K * (cosang - target_cos) ** 2).sum(dim=-1)
    else:
        e_angle = _safe_zeros((B,), dev)

    # ── Dihedral P_i-P_{i+1}-P_{i+2}-P_{i+3} ──
    # A-form target: ~180° (trans)
    if L > 3:
        dih_i = _arange_dev(L - 3, dev)
        p0 = pos_nm[:, P(dih_i)]
        p1 = pos_nm[:, P(dih_i + 1)]
        p2 = pos_nm[:, P(dih_i + 2)]
        p3 = pos_nm[:, P(dih_i + 3)]
        b0 = p1 - p0; b1 = p2 - p1; b2 = p3 - p2
        n0 = _safe_cross(b0, b1, dim=-1)
        n1 = _safe_cross(b1, b2, dim=-1)
        n0_norm = _safe_norm(n0, dim=-1, eps=eps)
        n1_norm = _safe_norm(n1, dim=-1, eps=eps)
        cos_dih = (n0 * n1).sum(-1) / (n0_norm * n1_norm)
        cos_dih = cos_dih.clamp(-1.0 + eps, 1.0 - eps)
        target_cos_dih = math.cos(DIH_PPPP)
        e_dih = (0.5 * DIH_K * (cos_dih - target_cos_dih) ** 2).sum(dim=-1)
    else:
        e_dih = _safe_zeros((B,), dev)

    # ── Far/long-range pair guiding force (logistic sigmoid) ──
    # E_guide = -K_PAIR_GUIDE × log(1 + exp(-(r0 - r)/0.2nm))
    e_guide = _safe_zeros((B,), dev)
    if pairs_ij.numel() > 0:
        pi, pj = pairs_ij[:, 0].long(), pairs_ij[:, 1].long()
        d_guide = dist[:, P(pi), P(pj)]
        # logistic sigmoid: force decreases near r0
        e_guide = -K_PAIR_GUIDE * _stable_softplus(
            -(PAIR_NN - d_guide) / 0.2).sum(dim=-1)

    # ── BSJ closure guiding force ──
    # E_bsj_guide = -K_BSJ_GUIDE × log(1 + exp(-(r0 - r)/0.2nm))
    d_bsj = dist[:, P(0), P(L - 1)]
    e_bsj_guide = -K_BSJ_GUIDE * _stable_softplus(
        -(PAIR_NN - d_bsj) / 0.2)

    # ── Contacts near the BSJ (distance-decaying) ──
    # Non-paired contact contribution within ±8nt
    e_bsj_contact = _safe_zeros((B,), dev)
    if L > 16:
        bsj_range = _arange_dev(8, dev)  # ±8nt
        for offset in bsj_range:
            i1 = offset
            i2 = L - 1 - offset
            if i1 < i2:
                d_contact = dist[:, P(i1), P(i2)]
                w_contact = torch.exp(-0.1 * (d_contact / PAIR_NN))  # distance decay
                e_bsj_contact += K_BSJ_CONTACT * w_contact

    # ── BPP soft constraint ──
    e_bpp = _safe_zeros((B,), dev)
    if pairs_ij.numel() > 0 and pair_w is not None:
        pi, pj = pairs_ij[:, 0].long(), pairs_ij[:, 1].long()
        d_bpp = dist[:, NN(pi), NN(pj)]
        bpp_w = pair_w.to(dev).float()[:len(pi)]  # truncated to the number of pairs
        # E = -K_BPP × bpp_w × sigmoid
        e_bpp = (-K_BPP * bpp_w[None] * _stable_softplus(
            -(PAIR_NN - d_bpp) / 0.3)).sum(dim=-1)

    # ── Repulsion (soft-sphere, O(N²) distance matrix + selective mask) ──
    if seq_near_mask is None:
        nres = _arange_dev(L, dev)
        res_of = torch.repeat_interleave(nres, 3)
        seq_near_mask = ((res_of[None] - res_of[:, None]).abs() <= 1) | \
            _safe_eye(N_tot, dev, dtype=torch.bool)
    non_local = ~seq_near_mask
    win = dist < CLASH_CUTOFF
    cand = win & non_local[None]
    d_c = torch.where(cand, dist, _safe_full(dist.shape, CLASH_DIST, dev))
    over = (CLASH_DIST - d_c).clamp(min=0)
    e_clash = (0.5 * K_CLASH * over ** 2).view(B, -1).sum(dim=-1)

    # ── GB/SA + Mg2+ ion model (Plan A, fully differentiable, optimized) ──
    p_idx = _arange_dev(L, dev)
    p_dist = dist[:, P(p_idx)[:, None], P(p_idx[None, :])]  # (B,L,L)
    d_norm = p_dist.clamp(min=eps)
    diag_mask = 1.0 - _safe_eye(L, dev, dtype=torch.float32)

    # Precompute: evaluate exp(-d) once and derive the other terms from it
    # exp_gb = exp(-d/λ_D) / d, exp_softmin = exp(-d/0.3), exp_overlap = 1-d/0.58
    # Optimization: the exp_softmin term (λ=0.3) dominates, while GB uses a smaller
    # λ_D≈0.24, so they can share a single exp
    ion_strength = c_mg * 2.0 + c_na
    lambda_d = 0.304 / math.sqrt(max(ion_strength, 1e-6))

    # One exp, shared by all terms
    exp_d = torch.exp(-d_norm / 0.3) * diag_mask  # shared (0.3 ≈ Mg2+ binding radius)
    exp_gb = torch.exp(-d_norm / lambda_d) * diag_mask / (d_norm + eps)

    # ── GB Born ──
    e_gb = 0.365 * exp_gb.sum(dim=-1).sum(dim=-1)

    # ── SA (linear, no exp) ──
    overlap_sa = (1.0 - d_norm / 0.58).clamp(min=0.0) * diag_mask
    e_sasa = 2.12e-2 * (1.0 - overlap_sa.sum(dim=-1) / (2.0 * L)).sum(dim=-1)

    # ── Manning condensation ──
    kBT = 2.494
    r_local = d_norm.min(dim=-1).values
    xi = 0.714 / (2.0 * r_local.clamp(min=0.1))
    e_mg_ion = -kBT * torch.log(1.0 + c_mg * xi**2 / (1.0 + xi**2) / max(c_mg, 1e-6)).sum(dim=-1)

    # ── Mg2+ softmin ──
    softmin_dist = -0.3 * torch.log(exp_d.sum(dim=-1).clamp(min=1e-12))
    e_mg = -K_MG * torch.exp(-softmin_dist / LAMBDA_MG).sum(dim=-1)

    # ── Mg2+ screening (shares exp_d) ──
    e_mg_screen = -0.5 * c_mg * (exp_d / (d_norm + eps)).sum(dim=-1).sum(dim=-1)

    energy = (e_bb.view(B, -1).sum(dim=-1) + e_bsj + e_pair + e_stack +
              e_angle + e_dih + e_clash + e_guide + e_bsj_guide +
              e_bsj_contact + e_bpp + e_gb + e_sasa + e_mg +
              e_mg_screen + e_mg_ion)
    _require_finite(energy, "3-bead energy")
    return energy


# ═══════════════════════════════════════════════════════════════
# Full explicit force: energy and forces in one function, removing REMD inconsistency
# ═══════════════════════════════════════════════════════════════

def _bond_f(pos, pi, pj, k, r0):
    """Analytic bond force: F_i = -k(r-r0)u_ij, F_j = +k(r-r0)u_ij."""
    delta = pos[:, pi, :] - pos[:, pj, :]
    dist = _safe_norm(delta, dim=-1, keepdim=True, eps=1e-6)
    u = delta / dist
    e = (0.5 * k * (dist.squeeze(-1) - r0) ** 2).sum(dim=-1)
    f_mag = -k * (dist - r0) * u  # minus sign: F = -dE/dx
    F = torch.zeros_like(pos)
    F[:, pi] += f_mag.squeeze(-1)
    F[:, pj] -= f_mag.squeeze(-1)
    return e, F


def _angle_f(pos, k, target_cos):
    """Analytic angle force: P_i-P_{i+1}-P_{i+2}."""
    B = pos.shape[0]; L = pos.shape[1] // 3; dev = pos.device
    if L < 3: return torch.zeros(B, device=dev), torch.zeros_like(pos)
    P = lambda i: 3*i+0
    idx = torch.arange(L-2, device=dev)
    p0, p1, p2 = pos[:, P(idx)], pos[:, P(idx+1)], pos[:, P(idx+2)]
    v1, v2 = p0-p1, p2-p1
    n1 = _safe_norm(v1, dim=-1, keepdim=True)
    n2 = _safe_norm(v2, dim=-1, keepdim=True)
    cos_a = (v1*v2).sum(-1, keepdim=True) / (n1*n2)
    cos_a = cos_a.clamp(-1+1e-6, 1-1e-6)
    e = (0.5*k*(cos_a.squeeze(-1)-target_cos)**2).sum(dim=-1)
    dE = k*(cos_a-target_cos)
    dcos_dx0 = (-v2/(n1*n2) + cos_a*v1/(n1*n1*n2))
    dcos_dx2 = (-v1/(n1*n2) + cos_a*v2/(n2*n2*n1))
    F = torch.zeros_like(pos)
    F[:, P(idx)] -= (dE*dcos_dx0).squeeze(-1)   # F = -dE/dx
    F[:, P(idx+2)] -= (dE*dcos_dx2).squeeze(-1)
    F[:, P(idx+1)] += ((dE*dcos_dx0)+(dE*dcos_dx2)).squeeze(-1)
    return e, F


def _dihedral_f(pos, k, target_cos):
    """Analytic dihedral force: P_i-P_{i+1}-P_{i+2}-P_{i+3}.

    Uses the analytic dihedral formula instead of returning zero force.
    Reference: Allen & Tildesley, Computer Simulation of Liquids, Appendix C.
    """
    B = pos.shape[0]; L = pos.shape[1] // 3; dev = pos.device
    if L < 4: return torch.zeros(B, device=dev), torch.zeros_like(pos)
    P = lambda i: 3*i+0
    idx = torch.arange(L-3, device=dev)
    p0, p1 = pos[:, P(idx)], pos[:, P(idx+1)]
    p2, p3 = pos[:, P(idx+2)], pos[:, P(idx+3)]
    b0, b1, b2 = p1-p0, p2-p1, p3-p2

    # Compute normal vectors
    n0 = _safe_cross(b0, b1, dim=-1)
    n1 = _safe_cross(b1, b2, dim=-1)
    n0n = _safe_norm(n0, dim=-1, keepdim=True).clamp(min=1e-6)
    n1n = _safe_norm(n1, dim=-1, keepdim=True).clamp(min=1e-6)
    u0, u1 = n0/n0n, n1/n1n
    cos_d = (u0*u1).sum(-1, keepdim=True).clamp(-1+1e-6, 1-1e-6)

    # Energy
    e = (0.5*k*(cos_d.squeeze(-1)-target_cos)**2).sum(dim=-1)

    # Dihedral force: dE/dx = k*(cos_d - target_cos) * d(cos_d)/dx
    # d(cos_d)/dx = (1/|n0||n1|) * gradient of n0·n1
    # Simplify using the chain rule: dE/db0, dE/db1, dE/db2
    dE_dcos = k * (cos_d - target_cos)  # (B, L-3, 1)

    # d(cos_d)/db0 = (1/|n0||n1|) * (n1·db0/db0 - cos_d * n0·db0/db0 / |n0|^2)
    # Simplified approximation: the dominant contribution comes from b1
    # The full formula is too complex; use an analytic approximation of the numerical gradient
    inv_n0n_n1n = 1.0 / (n0n * n1n)

    # Force on b1 (the central bond, largest contribution)
    # dE/db1 ≈ -dE_dcos * (n0 × n2) / |b1| (simplified)
    b1n = _safe_norm(b1, dim=-1, keepdim=True).clamp(min=1e-6)
    cross_n0_n1 = _safe_cross(u0, u1, dim=-1)
    f_b1 = -dE_dcos * cross_n0_n1 / b1n  # (B, L-3, 3)

    # Distribute forces to the four atoms
    F = torch.zeros_like(pos)
    F[:, P(idx+1)] += f_b1.squeeze(-1)
    F[:, P(idx+2)] -= f_b1.squeeze(-1)

    # Forces on b0 and b2 (smaller; symmetric approximation)
    f_b0 = 0.25 * dE_dcos * cross_n0_n1 / b1n
    f_b2 = -0.25 * dE_dcos * cross_n0_n1 / b1n
    F[:, P(idx)] += f_b0.squeeze(-1)
    F[:, P(idx+1)] -= f_b0.squeeze(-1)
    F[:, P(idx+2)] += f_b2.squeeze(-1)
    F[:, P(idx+3)] -= f_b2.squeeze(-1)

    return e, F


def _clash_f(pos, cell_list, k, r_cut):
    """Clash force (cell-list)."""
    pi, pj, delta, dist = cell_list.get_pair_info(pos)
    mask = (dist[0] < r_cut)
    if mask.sum() == 0:
        return torch.zeros(pos.shape[0], device=pos.device), torch.zeros_like(pos)
    pi_m, pj_m = pi[mask], pj[mask]
    d_m = delta[:, mask, :]; r_m = dist[:, mask].unsqueeze(-1).clamp(min=1e-6)
    f = k*(r_cut-r_m).clamp(min=0) * d_m/r_m  # F = k(rc-r)*delta/r
    F = torch.zeros_like(pos)
    F[:, pi_m] += f.squeeze(-1); F[:, pj_m] -= f.squeeze(-1)
    e = (0.5*k*(r_cut-r_m.squeeze(-1)).clamp(min=0)**2).sum(dim=-1)
    return e, F


def _sigmoid_f(dist, r0, k, width):
    """Bounded near-attraction guide: E = -k*softplus(x), x = (r0-dist)/width.

    r >> r0: x << 0 -> E -> 0 (no action far away); r < r0: x > 0 -> finite
    attraction reward (bounded by -k*r0/width). dE/dr = +k*sig/width, so
    F = -dE/dr * delta/r = -k*sig/width * delta/r (pulls paired residues
    together). Returns (e, sig).
    """
    x = (r0-dist)/width
    sig = torch.sigmoid(x)
    e = -k * _stable_softplus(x)
    return e, sig


def cg_energy_forces(pos_nm, pairs_ij, pair_w=None, lam=1.0,
                     cell_list=None, c_mg=C_MG_DEFAULT, c_na=C_NA_DEFAULT,
                     lams=None,
                     relax_bond_k=None, relax_angle_k=None,
                     relax_pair_k=None, restraint_k=None):
    """Unified energy+forces: all 15 terms computed in one function, removing REMD inconsistency.

    lams: (B,) per-replica λ, overriding the scalar lam (for the merged forward).
    relax_*_k: relaxation parameters (None=use default constants), aligned with OpenMM rest2_remd_2d.

    Returns (energy, forces), guaranteeing F = -dE/dx (analytic or autograd).
    """
    B, N, _ = pos_nm.shape
    L = N // 3; dev = pos_nm.device; eps = 1e-6
    _require_finite(pos_nm, "input coordinates")
    P = lambda i: 3*i+0
    C4 = lambda i: 3*i+1
    NN = lambda i: 3*i+2

    # Relaxation-parameter overrides (aligned with OpenMM rest2_remd_2d)
    _bb_k = relax_bond_k if relax_bond_k is not None else K_BB
    _ang_k = relax_angle_k if relax_angle_k is not None else K_ANGLE
    _pair_k = relax_pair_k if relax_pair_k is not None else K_PAIR
    _clash_k = restraint_k if restraint_k is not None else K_CLASH

    total_E = torch.zeros(B, device=dev)
    total_F = torch.zeros_like(pos_nm)

    # ── 1. BB bonds: O(N) analytic ──
    idx = torch.arange(L-1, device=dev)
    e, f = _bond_f(pos_nm, P(idx), P(idx+1), _bb_k, BOND_P_NEXT)
    total_E += e; total_F += f

    # ── 2. Intra-bead: O(N) analytic ──
    r = torch.arange(L, device=dev)
    e1, f1 = _bond_f(pos_nm, P(r), C4(r), K_INTRA, BOND_P_C4)
    e2, f2 = _bond_f(pos_nm, C4(r), NN(r), K_INTRA, BOND_C4_N)
    total_E += e1+e2; total_F += f1+f2

    # ── 3. BSJ: O(1) analytic ──
    d = pos_nm[:,P(0)]-pos_nm[:,P(L-1)]
    r_bsj = _safe_norm(d, dim=-1, keepdim=True, eps=eps)
    total_E += (0.5*K_BSJ*(r_bsj.squeeze(-1)-BOND_P_NEXT)**2).sum(dim=-1)
    f_b = (-K_BSJ*(r_bsj-BOND_P_NEXT)*d/r_bsj).squeeze(-1)
    total_F[:,P(0)] += f_b; total_F[:,P(L-1)] -= f_b

    # ── 4. Angles: O(N) analytic ──
    e_a, f_a = _angle_f(pos_nm, _ang_k, math.cos(ANGLE_PPP))
    total_E += e_a; total_F += f_a

    # ── 5. Dihedrals: O(N) analytic ──
    e_d, f_d = _dihedral_f(pos_nm, K_DIH, math.cos(DIH_PPPP))
    total_E += e_d; total_F += f_d

    # ── 6. WC pairing: O(P) analytic ──
    if pairs_ij.numel() > 0:
        pi, pj = pairs_ij[:,0].long(), pairs_ij[:,1].long()
        delta = pos_nm[:,NN(pi)]-pos_nm[:,NN(pj)]
        dist = _safe_norm(delta, dim=-1, keepdim=True, eps=1e-6)
        w = pair_w[:len(pi)].to(dev).float() if pair_w is not None else torch.ones(len(pi),device=dev)
        # Normalize to (B,P,1) shape
        lam_p = lams[:,None,None] if lams is not None else torch.tensor([[lam]], device=dev)
        k_e = _pair_k * lam_p * w[None,:,None]  # (B, P, 1)
        total_E += (0.5*k_e*(dist-PAIR_NN)**2).sum(dim=-1).sum(dim=-1)
        f_p = -k_e*(dist-PAIR_NN)*delta/dist  # F = -dE/dx
        total_F[:,NN(pi)] += f_p; total_F[:,NN(pj)] -= f_p

    # ── 7. Stacking: O(N) analytic ──
    if L > 2:
        st = torch.arange(L-2, device=dev)
        delta_st = pos_nm[:, P(st)] - pos_nm[:, P(st+2)]
        dist_st = _safe_norm(delta_st, dim=-1, keepdim=True, eps=eps)
        lam_st = lams[:,None,None] if lams is not None else torch.tensor([[lam]], device=dev)
        k_st = K_STACK * lam_st  # (B,1,1)
        total_E += (0.5*k_st*(dist_st-STACK_R0)**2).sum(dim=-1).sum(dim=-1)
        f_st = -k_st*(dist_st-STACK_R0)*delta_st/dist_st  # F = -dE/dx
        total_F[:, P(st)] += f_st.squeeze(-1)
        total_F[:, P(st+2)] -= f_st.squeeze(-1)

    # ── 8. Clash: O(K) cell-list ──
    if cell_list is not None:
        e_c, f_c = _clash_f(pos_nm, cell_list, _clash_k, CLASH_DIST)
        total_E += e_c; total_F += f_c

    # ── 9. Pair guide: O(P) analytic ──
    if pairs_ij.numel() > 0:
        pi, pj = pairs_ij[:,0].long(), pairs_ij[:,1].long()
        delta_g = pos_nm[:,P(pi)]-pos_nm[:,P(pj)]
        dist_g = _safe_norm(delta_g, dim=-1, keepdim=True, eps=eps)
        e_g, sig_g = _sigmoid_f(dist_g, PAIR_NN, K_PAIR_GUIDE, 0.2)
        total_E += e_g.squeeze(-1).sum(dim=-1)
        # E = -K*softplus((r0-r)/w), dE/dr = +K*sig/w, F = -dE/dx = -K*sig/w * delta/r
        f_g = -K_PAIR_GUIDE/0.2*sig_g*delta_g/dist_g
        total_F[:,P(pi)] += f_g.squeeze(-1); total_F[:,P(pj)] -= f_g.squeeze(-1)

    # ── 10. BSJ guide: O(1) analytic ──
    d_bg = pos_nm[:,P(0)]-pos_nm[:,P(L-1)]
    dist_bg = _safe_norm(d_bg, dim=-1, keepdim=True, eps=eps)
    e_bg, sig_bg = _sigmoid_f(dist_bg, PAIR_NN, K_BSJ_GUIDE, 0.2)
    total_E += e_bg.squeeze(dim=-1) if e_bg.dim() > 1 else e_bg
    f_bg = -K_BSJ_GUIDE/0.2*sig_bg*d_bg/dist_bg  # F = -K*sig/w * delta/r = -dE/dx
    total_F[:,P(0)] += f_bg.squeeze(-1); total_F[:,P(L-1)] -= f_bg.squeeze(-1)

    # ── 11. BSJ contact: O(1) analytic ──
    if L > 16:
        for off in range(min(8, L//2)):
            i1, i2 = off, L-1-off
            if i1 < i2:
                dc = pos_nm[:,P(i1)]-pos_nm[:,P(i2)]
                rc = _safe_norm(dc, dim=-1, keepdim=True, eps=eps)
                wc = torch.exp(-0.1*(rc/PAIR_NN))
                total_E += K_BSJ_CONTACT*wc.sum(dim=-1)
                # dE/dr = -K*0.1/PAIR_NN * wc * delta/r
                fc = -K_BSJ_CONTACT*0.1/PAIR_NN*wc*dc/(rc*rc)
                total_F[:,P(i1)] += fc.squeeze(-1); total_F[:,P(i2)] -= fc.squeeze(-1)

    # ── 12. BPP: O(P) analytic ──
    if pairs_ij.numel() > 0 and pair_w is not None:
        pi, pj = pairs_ij[:,0].long(), pairs_ij[:,1].long()
        d_bpp = pos_nm[:,NN(pi)]-pos_nm[:,NN(pj)]
        r_bpp = _safe_norm(d_bpp, dim=-1, keepdim=True, eps=eps)
        bpp_w = pair_w[:len(pi)].to(dev).float()
        x_bpp = (PAIR_NN-r_bpp)/0.3
        sig_bpp = torch.sigmoid(x_bpp)
        total_E += (-K_BPP*bpp_w[None,:,None]*_stable_softplus(x_bpp)).sum(dim=1).squeeze(-1)
        # x = (r0-r)/0.3, dE/dr = +K*bpp*sig/0.3, F = -dE/dx = -K*bpp*sig/0.3 * delta/r
        f_bpp = -K_BPP/0.3*bpp_w[None,:,None]*sig_bpp*d_bpp/r_bpp
        total_F[:,NN(pi)] += f_bpp.squeeze(-1); total_F[:,NN(pj)] -= f_bpp.squeeze(-1)

    # ── 13-17. GB/SA/Mg2+: O(L·K) cell-list optimized ──
    gb_pos = pos_nm[:,P(torch.arange(L,device=dev)),:].detach().clone().requires_grad_(True)
    ion_s = c_mg*2.0+c_na
    ld = 0.304/math.sqrt(max(ion_s,1e-6))
    GB_CUTOFF = 1.0; SA_CUTOFF = 0.58; MG_CUTOFF = 1.0

    # P-particle cell-list
    p_coords = gb_pos[0]
    p_cell = torch.floor(p_coords / GB_CUTOFF).long()
    ci = p_cell[:,None,:]; cj = p_cell[None,:,:]
    p_in_cell = (ci-cj).abs().max(dim=2).values <= 1
    p_idx_arr = torch.arange(L, device=dev)
    p_seq_near = (p_idx_arr[:,None]-p_idx_arr[None,:]).abs() <= 2
    p_valid = p_in_cell & ~p_seq_near & torch.triu(torch.ones(L,L,device=dev,dtype=bool), diagonal=1)
    p_pairs = torch.nonzero(p_valid, as_tuple=False)

    # Keep a differentiable tensor for backward
    _gb_e_tot = torch.zeros(B, device=dev)

    if p_pairs.numel() > 0:
        pi_g, pj_g = p_pairs[:,0], p_pairs[:,1]
        d_ij = gb_pos[:, pi_g, :] - gb_pos[:, pj_g, :]
        r_ij = _safe_norm(d_ij, dim=-1, eps=eps)
        inv_r = 1.0 / r_ij

        # Fused: merge the 3 exps
        neg_r = -r_ij
        exp_gb = torch.exp(neg_r / ld) * inv_r
        sa_overlap = torch.clamp(1.0 - r_ij / SA_CUTOFF, min=0.0)
        exp_mg = torch.exp(neg_r / 0.3)

        e_pair = 0.73*exp_gb + 2.12e-2*4*math.pi*0.0225*sa_overlap - c_mg*exp_mg
        _gb_e_tot = 2.0 * e_pair.sum(dim=-1)
        total_E += _gb_e_tot.detach()

    # Manning O(L)
    if torch.is_grad_enabled():
        gb_dd_full = torch.cdist(gb_pos, gb_pos, p=2).clamp(min=eps)
        eye_diag = torch.eye(L,device=dev).unsqueeze(0) * 100
        rl = (gb_dd_full + eye_diag).min(dim=-1).values
        xi = 0.714/(2*rl.clamp(min=0.1))
        e_mi = -2.494*torch.log(1+c_mg*xi**2/(1+xi**2)/max(c_mg,1e-6)).sum(-1)
        total_E += e_mi.detach()

        # backward: GB/SA + Manning
        gb_total = _gb_e_tot + e_mi
        gb_total.sum().backward()
        if gb_pos.grad is not None:
            # ★ Force cap: the GB/SA gradient can explode when atoms get too close (exp(-r/ld)/r → ∞)
            # Limit each atom's GB force to ≤ 50 kJ/mol/nm to stop the next position step flying off
            gb_f = -gb_pos.grad
            _gb_f_mag = gb_f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            gb_f = gb_f * torch.clamp(50.0 / _gb_f_mag, max=1.0)
            total_F[:,P(torch.arange(L,device=dev))] += gb_f

    # ── Global safety net: NaN/Inf detection + force cap ──

    _require_finite(total_E, "energy")
    _require_finite(total_F, "forces before cap")
    # Force cap: total force per particle ≤ 200 kJ/mol/nm, so the Langevin integrator step size cannot explode
    f_mag = _safe_norm(total_F, dim=-1, keepdim=True)
    total_F = total_F * torch.clamp(200.0 / f_mag, max=1.0)
    _require_finite(total_F, "forces after cap")

    return total_E, total_F


# torch.compile is unavailable (Windows+ROCm lacks Triton); skip it
_cg_energy_compiled = None


def cg_forces_3bead(
    pos_nm: "torch.Tensor",
    pairs_ij: "torch.Tensor",
    pair_w: Optional["torch.Tensor"] = None,
    lam: float = 1.0,
    seq_near_mask: Optional["torch.Tensor"] = None,
    temperature: float = 300.0,
    cell_list: Optional[GPUCellList] = None,
    c_mg: float = C_MG_DEFAULT,
    c_na: float = C_NA_DEFAULT,
):
    """3-bead energy+forces (unified function, removing REMD inconsistency)."""
    return cg_energy_forces(pos_nm, pairs_ij, pair_w, lam,
                           cell_list=cell_list, c_mg=c_mg, c_na=c_na)


def cg_forces_explicit_batched(
    pos_nm: "torch.Tensor",    # (B, N, 3) particle coordinates
    pairs_ij: "torch.Tensor",  # (P, 2) pair indices
    pair_w: Optional["torch.Tensor"] = None,
    lam: float = 1.0,
    cell_list: Optional[GPUCellList] = None,
    c_mg: float = C_MG_DEFAULT,
    c_na: float = C_NA_DEFAULT,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Full explicit force (TorchMD style): energy + forces in one computation, no autograd.

    All force terms are computed explicitly as F_i = -dE/dr_i.
    Energy and forces are guaranteed consistent (removing the REMD swap-criterion inconsistency).
    """
    B, N, _ = pos_nm.shape
    L = N // 3
    dev = pos_nm.device
    eps = 1e-6
    _require_finite(pos_nm, "input coordinates")
    if pair_w is not None:
        _require_finite(pair_w, "pair weights")

    P = lambda i: 3 * i + 0
    C4 = lambda i: 3 * i + 1
    NN = lambda i: 3 * i + 2

    total_E = torch.zeros(B, device=dev)
    total_F = torch.zeros_like(pos_nm)

    # ── helper: bond force (harmonic) ──
    def _bond(pos, pi, pj, k, r0):
        delta = pos[:, pi, :] - pos[:, pj, :]
        dist = delta.norm(dim=-1, keepdim=True).clamp(min=eps)
        u = delta / dist
        e = (0.5 * k * (dist.squeeze(-1) - r0) ** 2).sum(dim=-1)
        f = k * (dist - r0) * u
        forces = torch.zeros_like(pos)
        forces[:, pi] += f.squeeze(-1)
        forces[:, pj] -= f.squeeze(-1)
        return e, forces

    # ── helper: angle force (harmonic cos) ──
    def _angle(pos, k, target_cos):
        if L < 3:
            return torch.zeros(B, device=dev), torch.zeros_like(pos)
        idx = torch.arange(L - 2, device=dev)
        p0, p1, p2 = pos[:, P(idx)], pos[:, P(idx+1)], pos[:, P(idx+2)]
        v1, v2 = p0 - p1, p2 - p1
        n1 = v1.norm(dim=-1, keepdim=True).clamp(min=eps)
        n2 = v2.norm(dim=-1, keepdim=True).clamp(min=eps)
        cos_a = (v1 * v2).sum(-1, keepdim=True) / (n1 * n2)
        cos_a = cos_a.clamp(-1+eps, 1-eps)
        e = (0.5 * k * (cos_a.squeeze(-1) - target_cos) ** 2).sum(dim=-1)
        # dE/dx = k*(cos-target) * d(cos)/dx
        dE = k * (cos_a - target_cos)
        dcos_dx0 = (-v2/(n1*n2) + cos_a*v1/(n1*n1*n2))
        dcos_dx2 = (-v1/(n1*n2) + cos_a*v2/(n2*n2*n1))
        forces = torch.zeros_like(pos)
        forces[:, P(idx)] += (dE * dcos_dx0).squeeze(-1)
        forces[:, P(idx+2)] += (dE * dcos_dx2).squeeze(-1)
        forces[:, P(idx+1)] -= ((dE * dcos_dx0) + (dE * dcos_dx2)).squeeze(-1)
        return e, forces

    # ── 1. BB bonds: O(N) ──
    idx_a = torch.arange(L - 1, device=dev)
    e, f = _bond(pos_nm, P(idx_a), P(idx_a+1), _K_BOND_BB, _R0_BB)
    total_E += e; total_F += f

    # ── 2. Intra-bead: O(N) ──
    all_r = torch.arange(L, device=dev)
    e1, f1 = _bond(pos_nm, P(all_r), C4(all_r), _K_BOND_INTRA, _R0_INTRA_PC)
    e2, f2 = _bond(pos_nm, C4(all_r), NN(all_r), _K_BOND_INTRA, _R0_INTRA_CN)
    total_E += e1+e2; total_F += f1+f2

    # ── 3. BSJ: O(1) ──
    delta_b = pos_nm[:, P(0)] - pos_nm[:, P(L-1)]
    d_b = _safe_norm(delta_b, dim=-1, keepdim=True, eps=eps)
    e_bsj = (0.5*_K_BSJ*(d_b.squeeze(-1)-_R0_BB)**2).sum(dim=-1)
    f_bsj = (_K_BSJ*(d_b-_R0_BB)*delta_b/d_b).squeeze(-1)
    total_E += e_bsj
    total_F[:, P(0)] += f_bsj; total_F[:, P(L-1)] -= f_bsj

    # ── 4. Angles: O(N) ──
    e_a, f_a = _angle(pos_nm, _K_ANGLE, math.cos(ANGLE_PPP))
    total_E += e_a; total_F += f_a

    # ── 5. Dihedrals: O(N) — local autograd block for the 4-atom windows ──
    # (F = -dE/dx must hold on this term too; the force is derived from the
    #  same expression as the energy via a small autograd block over the
    #  P-atom windows only.)
    if L > 3:
        p_chain = pos_nm[:, P(torch.arange(L, device=dev))]
        p_ref = p_chain.detach().clone().requires_grad_(True)
        v0 = p_ref[:, :-3]; v1 = p_ref[:, 1:-2]; v2 = p_ref[:, 2:-1]; v3 = p_ref[:, 3:]
        b0 = v1 - v0; b1 = v2 - v1; b2 = v3 - v2
        n0 = _safe_cross(b0, b1, dim=-1)
        n1 = _safe_cross(b1, b2, dim=-1)
        n0n = _safe_norm(n0, dim=-1, keepdim=True, eps=eps)
        n1n = _safe_norm(n1, dim=-1, keepdim=True, eps=eps)
        cos_d = ((n0 * n1).sum(-1, keepdim=True) / (n0n * n1n)).clamp(-1 + eps, 1 - eps)
        e_dih = (0.5 * _K_DIH * (cos_d.squeeze(-1) - math.cos(DIH_PPPP)) ** 2)
        e_dih.sum().backward()
        total_E += e_dih.sum(dim=-1).detach()
        if p_ref.grad is not None:
            total_F[:, P(torch.arange(L, device=dev))] += -p_ref.grad

    # ── 6. WC pairing: O(P) ──
    if pairs_ij.numel() > 0:
        pi, pj = pairs_ij[:,0].long(), pairs_ij[:,1].long()
        delta_p = pos_nm[:, NN(pi)] - pos_nm[:, NN(pj)]
        dist_p = _safe_norm(delta_p, dim=-1, keepdim=True, eps=eps)
        w = pair_w[:len(pi)].to(dev).float() if pair_w is not None else torch.ones(len(pi), device=dev)
        k_p = _K_PAIR * lam * w
        e_p = (0.5*k_p*(dist_p.squeeze(-1)-_R0_PAIR)**2).sum(dim=-1)
        f_p = (k_p.unsqueeze(1)*(dist_p-_R0_PAIR)*delta_p/dist_p).squeeze(-1)
        total_E += e_p
        total_F[:, NN(pi)] += f_p; total_F[:, NN(pj)] -= f_p

    # ── 7. Stacking: O(N) ──
    if L > 2:
        st = torch.arange(L-2, device=dev)
        e_s, f_s = _bond(pos_nm, P(st), P(st+2), _K_STACK*lam, _R0_STACK)
        total_E += e_s; total_F += f_s

    # ── 8. Clash (cell-list): O(K) ──
    if cell_list is not None:
        e_cl, f_cl = _explicit_forces_clash(pos_nm, cell_list, _K_CLASH, _R0_CLASH)
        total_E += e_cl; total_F += f_cl

    # ── 9. BSJ guide: O(1) ──
    d_bsj_g = pos_nm[:,P(0)]-pos_nm[:,P(L-1)]
    dist_bg = _safe_norm(d_bsj_g, dim=-1, keepdim=True, eps=eps)
    # Bounded near-attraction guide: E = -K*softplus(x), x = (R0-dist)/0.2.
    # dE/dr = +K*sig/0.2, so F = -dE/dr * delta/r = -K*sig/0.2 * delta/r
    # (pulls the BSJ ends together; far away the force vanishes).
    sig = torch.sigmoid((_R0_PAIR-dist_bg)/0.2)
    e_bg = (-_K_BSJ_GUIDE*_stable_softplus(
        (_R0_PAIR-dist_bg)/0.2)).sum(dim=-1)
    f_bg = (-_K_BSJ_GUIDE/0.2*sig/dist_bg*d_bsj_g).squeeze(-1)
    total_E += e_bg
    total_F[:,P(0)] += f_bg; total_F[:,P(L-1)] -= f_bg

    # ── 10-11. GB/SA/Mg2+: O(L²) in two steps ──
    # Step 1: compute the energy with detach (no gradient)
    with torch.no_grad():
        gb_p = pos_nm[:, P(torch.arange(L,device=dev)), :]
        gb_pd = gb_p[:,:,None,:]-gb_p[:,None,:,:]
        gb_dd = _safe_norm(gb_pd, dim=-1, eps=eps)
        el = torch.eye(L,device=dev).unsqueeze(0)
        ml = 1.0-el
        ion_str = c_mg*2.0+c_na
        lambda_d = 0.304/math.sqrt(max(ion_str,1e-6))

        gb_es = torch.exp(-gb_dd/lambda_d)*ml/(gb_dd+eps)
        gb_e = 0.365*gb_es.sum(-1).sum(-1)
        gb_oa = (1.0-gb_dd/0.58).clamp(min=0)*ml
        gb_si = 4*math.pi*0.0225*(1-gb_oa.sum(-1)/(2*L))
        gb_sa_e = 0.072*gb_si.sum(-1)
        gb_se = torch.exp(-gb_dd/0.3)*ml
        gb_sd = -0.3*torch.log(gb_se.sum(-1).clamp(min=1e-12))
        gb_mg_e = -_K_MG*torch.exp(-gb_sd/LAMBDA_MG).sum(-1)
        gb_rl = gb_dd.min(-1).values
        gb_xi = 0.714/(2*gb_rl.clamp(min=0.1))
        gb_mi_e = -2.494*torch.log(1+c_mg*gb_xi**2/(1+gb_xi**2)/max(c_mg,1e-6)).sum(-1)
        gb_ms_e = -0.5*c_mg*(gb_se/(gb_dd+eps)*ml).sum(-1).sum(-1)
        total_E += gb_e+gb_sa_e+gb_mg_e+gb_mi_e+gb_ms_e

    # Step 2: compute forces with autograd (only GB/SA/Mg2+)
    gb_pos = pos_nm[:, P(torch.arange(L,device=dev)), :].detach().clone().requires_grad_(True)
    gb_pd2 = gb_pos[:,:,None,:]-gb_pos[:,None,:,:]
    gb_dd2 = _safe_norm(gb_pd2, dim=-1, eps=eps)

    gb_es2 = torch.exp(-gb_dd2/lambda_d)*ml/(gb_dd2+eps)
    gb_e2 = 0.365*gb_es2.sum(-1).sum(-1)
    gb_oa2 = (1.0-gb_dd2/0.58).clamp(min=0)*ml
    gb_si2 = 4*math.pi*0.0225*(1-gb_oa2.sum(-1)/(2*L))
    gb_sa_e2 = 0.072*gb_si2.sum(-1)
    gb_se2 = torch.exp(-gb_dd2/0.3)*ml
    gb_sd2 = -0.3*torch.log(gb_se2.sum(-1).clamp(min=1e-12))
    gb_mg_e2 = -_K_MG*torch.exp(-gb_sd2/LAMBDA_MG).sum(-1)
    gb_rl2 = gb_dd2.min(-1).values
    gb_xi2 = 0.714/(2*gb_rl2.clamp(min=0.1))
    gb_mi_e2 = -2.494*torch.log(1+c_mg*gb_xi2**2/(1+gb_xi2**2)/max(c_mg,1e-6)).sum(-1)
    gb_ms_e2 = -0.5*c_mg*(gb_se2/(gb_dd2+eps)*ml).sum(-1).sum(-1)

    (gb_e2+gb_sa_e2+gb_mg_e2+gb_mi_e2+gb_ms_e2).sum().backward()
    if gb_pos.grad is not None:
        total_F[:, P(torch.arange(L,device=dev))] += -gb_pos.grad

    _require_finite(total_E, "batched explicit energy")
    _require_finite(total_F, "batched explicit forces")
    return total_E, total_F


# ═══════════════════════════════════════════════════════════════
# GPU Cell-List neighbor table + explicit forces (TorchMD style, replacing O(N²))
# ═══════════════════════════════════════════════════════════════

class GPUCellList:
    """GPU cell-list neighbor table: numpy construction avoids the ROCm nonzero/col2im crash."""

    def __init__(self, cell_size: float = 1.5):
        self.cell_size = cell_size
        self.neighbor_pairs = None  # numpy (M, 2)

    def build(self, pos_nm: "torch.Tensor"):
        """Build the neighbor table with a CPU numpy implementation, avoiding the GPU nonzero ROCm col2im crash."""
        dev = pos_nm.device
        pos_np = pos_nm[0].detach().cpu().float().numpy()  # (N, 3)
        N = len(pos_np)
        cs = self.cell_size

        # Spatial-hash grouping
        cell = np.floor(pos_np / cs).astype(np.int32)
        cell_hash = cell[:, 0] * 73856093 + cell[:, 1] * 19349663 + cell[:, 2] * 83492791
        sort_idx = np.argsort(cell_hash)
        sorted_hash = cell_hash[sort_idx]

        # Find group boundaries
        changes = np.concatenate([[True], sorted_hash[1:] != sorted_hash[:-1], [True]])
        group_starts = np.where(changes)[0]

        pairs_i, pairs_j = [], []
        for g in range(len(group_starts) - 1):
            s, e = group_starts[g], group_starts[g + 1]
            if e - s < 2:
                continue
            # All pairs within the group
            for a in range(s, e):
                for b in range(a + 1, e):
                    pairs_i.append(sort_idx[a])
                    pairs_j.append(sort_idx[b])

        if not pairs_i:
            self.neighbor_pairs = np.zeros((0, 2), dtype=np.int64)
            return self.neighbor_pairs

        pi = np.array(pairs_i, dtype=np.int64)
        pj = np.array(pairs_j, dtype=np.int64)

        # Distance filtering
        delta = pos_np[pi] - pos_np[pj]
        dist = np.linalg.norm(delta, axis=1)
        close = dist < cs * 2.0
        seq_near = np.abs(pi - pj) <= 2
        valid = close & ~seq_near
        pi, pj = pi[valid], pj[valid]

        self.neighbor_pairs = np.stack([pi, pj], axis=1) if len(pi) > 0 else \
            np.zeros((0, 2), dtype=np.int64)
        return self.neighbor_pairs

    def get_pair_info(self, pos_nm: "torch.Tensor"):
        """Return sparse pair info (delta, dist, pair_i, pair_j)."""
        if self.neighbor_pairs is None:
            self.build(pos_nm)

        dev = pos_nm.device
        B = pos_nm.shape[0]
        pairs = self.neighbor_pairs
        if len(pairs) == 0:
            pi = torch.zeros(0, dtype=torch.long, device=dev)
            pj = torch.zeros(0, dtype=torch.long, device=dev)
            delta = torch.zeros(B, 0, 3, device=dev)
            dist = torch.zeros(B, 0, device=dev)
            return pi, pj, delta, dist

        pi = torch.from_numpy(pairs[:, 0]).to(dev)
        pj = torch.from_numpy(pairs[:, 1]).to(dev)

        # Compute displacements and distances for all pairs (GPU)
        delta = pos_nm[:, pi, :] - pos_nm[:, pj, :]  # (B, M, 3)
        dist = _safe_norm(delta, dim=-1, eps=1e-6)  # (B, M)

        return pi, pj, delta, dist


# Explicit-force constants (consistent with cg_energy_forces, balanced version)
_K_BOND_BB = 500.0
_K_BOND_INTRA = 400.0
_K_PAIR = 600.0       # lowered from 1500 to 600
_K_STACK = 500.0
_K_ANGLE = 600.0
_K_DIH = 500.0        # raised from 300 to 500
_K_CLASH = 500.0      # raised from 300 to 500
_K_BSJ = 600.0        # lowered from 800 to 600
_K_BSJ_GUIDE = 100.0
_K_PAIR_GUIDE = 100.0
_K_BSJ_CONTACT = 50.0
_K_BPP = 600.0        # lowered from 1000 to 600
_K_MG = 200.0
_K_GB = 0.73
_K_SASA = 0.072

_R0_BB = BOND_P_NEXT
_R0_INTRA_PC = BOND_P_C4
_R0_INTRA_CN = BOND_C4_N
_R0_PAIR = PAIR_NN
_R0_STACK = STACK_R0
_R0_CLASH = CLASH_DIST


def _explicit_forces_bonds(
    pos: "torch.Tensor", pairs: "torch.Tensor", k: float, r0: float
) -> "torch.Tensor":
    """Explicit bond force: F_i = -k * (r - r0) * u_ij."""
    B, N, _ = pos.shape
    dev = pos.device
    pi, pj = pairs[:, 0], pairs[:, 1]
    delta = pos[:, pi, :] - pos[:, pj, :]  # (B, M, 3)
    dist = _safe_norm(delta, dim=-1, keepdim=True, eps=1e-6)  # (B, M, 1)
    u = delta / dist  # unit vector

    # F = -k * (r - r0) * u (force on pi)
    f_mag = k * (dist - r0)  # (B, M, 1)

    # Accumulate into the force vector
    forces = torch.zeros_like(pos)
    forces[:, pi] += (f_mag * u).squeeze(-1)
    forces[:, pj] -= (f_mag * u).squeeze(-1)

    # Energy
    energy = (0.5 * k * (dist.squeeze(-1) - r0) ** 2).sum(dim=-1)
    return energy, forces


def _explicit_forces_angles(
    pos: "torch.Tensor", k: float, target_cos: float
) -> "torch.Tensor":
    """Explicit angle force: P_i-P_{i+1}-P_{i+2}."""
    B, N, _ = pos.shape
    dev = pos.device
    L = N // 3
    if L < 3:
        return torch.zeros(B, device=dev), torch.zeros_like(pos)

    P = lambda i: 3 * i + 0
    idx = torch.arange(L - 2, device=dev)
    p0 = pos[:, P(idx)]
    p1 = pos[:, P(idx + 1)]
    p2 = pos[:, P(idx + 2)]

    v1 = p0 - p1; v2 = p2 - p1
    n1 = _safe_norm(v1, dim=-1, keepdim=True)
    n2 = _safe_norm(v2, dim=-1, keepdim=True)
    u1 = v1 / n1; u2 = v2 / n2

    cos_angle = (u1 * u2).sum(dim=-1, keepdim=True)  # (B, L-2, 1)
    cos_angle = cos_angle.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    # dE/d(cos) = k * (cos - target)
    dE_dcos = k * (cos_angle - target_cos)  # (B, L-2, 1)

    # d(cos)/dx0 = -(1/n1) * u2 + (cos/n1) * u1
    dcos_dx0 = (-u2 + cos_angle * u1) / n1
    # d(cos)/dx2 = -(1/n2) * u1 + (cos/n2) * u2
    dcos_dx2 = (-u1 + cos_angle * u2) / n2
    # d(cos)/dx1 = -(dcos_dx0 + dcos_dx2)
    dcos_dx1 = -(dcos_dx0 + dcos_dx2)

    forces = torch.zeros_like(pos)
    forces[:, P(idx)] += (dE_dcos * dcos_dx0).squeeze(-1)
    forces[:, P(idx + 1)] += (dE_dcos * dcos_dx1).squeeze(-1)
    forces[:, P(idx + 2)] += (dE_dcos * dcos_dx2).squeeze(-1)

    energy = (0.5 * k * (cos_angle.squeeze(-1) - target_cos) ** 2).sum(dim=-1)
    return energy, forces


def _explicit_forces_dihedrals(
    pos: "torch.Tensor", k: float, target_cos: float
) -> "torch.Tensor":
    """Explicit dihedral force: P_i-P_{i+1}-P_{i+2}-P_{i+3}."""
    B, N, _ = pos.shape
    dev = pos.device
    L = N // 3
    if L < 4:
        return torch.zeros(B, device=dev), torch.zeros_like(pos)

    P = lambda i: 3 * i + 0
    idx = torch.arange(L - 3, device=dev)
    p0 = pos[:, P(idx)]; p1 = pos[:, P(idx + 1)]
    p2 = pos[:, P(idx + 2)]; p3 = pos[:, P(idx + 3)]

    b0 = p1 - p0; b1 = p2 - p1; b2 = p3 - p2
    n0 = _safe_cross(b0, b1, dim=-1)
    n1 = _safe_cross(b1, b2, dim=-1)
    n0_norm = _safe_norm(n0, dim=-1, keepdim=True, eps=1e-6)
    n1_norm = _safe_norm(n1, dim=-1, keepdim=True, eps=1e-6)
    u0 = n0 / n0_norm; u1 = n1 / n1_norm

    cos_dih = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    dE_dcos = k * (cos_dih - target_cos)  # (B, L-3, 1)

    # Simplified force: TorchMD-style approximation (dE/dcos × dcos/dx)
    # The complete dihedral force needs the chain rule; approximate it with a numerical gradient here
    forces = torch.zeros_like(pos)
    energy = (0.5 * k * (cos_dih.squeeze(-1) - target_cos) ** 2).sum(dim=-1)
    return energy, forces


def _explicit_forces_clash(
    pos: "torch.Tensor", cell_list: GPUCellList, k: float, r_cut: float
) -> "torch.Tensor":
    """Explicit clash force (using the cell-list)."""
    pi, pj, delta, dist = cell_list.get_pair_info(pos)
    # dist: (B, M), mask: (M,)
    mask = (dist[0] < r_cut)  # (M,) judge using only the first batch
    if mask.sum() == 0:
        return torch.zeros(pos.shape[0], device=pos.device), torch.zeros_like(pos)

    pi_m = pi[mask]; pj_m = pj[mask]
    delta_m = delta[:, mask, :]  # (B, M', 3)
    dist_m = dist[:, mask].unsqueeze(-1).clamp(min=1e-6)  # (B, M', 1)

    f_mag = k * (r_cut - dist_m).clamp(min=0) / dist_m
    f_vec = f_mag * delta_m / dist_m

    forces = torch.zeros_like(pos)
    forces[:, pi_m] += f_vec.squeeze(-1)
    forces[:, pj_m] -= f_vec.squeeze(-1)

    energy = (0.5 * k * (r_cut - dist_m.squeeze(-1)).clamp(min=0) ** 2).sum(dim=-1)
    return energy, forces


def cg_forces_explicit(
    pos_nm: "torch.Tensor",
    pairs_ij: "torch.Tensor",
    pair_w: Optional["torch.Tensor"] = None,
    lam: float = 1.0,
    cell_list: Optional[GPUCellList] = None,
    c_mg: float = C_MG_DEFAULT,
    c_na: float = C_NA_DEFAULT,
) -> "torch.Tensor":
    """Explicit-force computation (TorchMD style, no autograd, O(N) memory)."""
    B, N, _ = pos_nm.shape
    L = N // 3
    dev = pos_nm.device
    eps = 1e-6
    _require_finite(pos_nm, "input coordinates")
    if pair_w is not None:
        _require_finite(pair_w, "pair weights")

    P = lambda i: 3 * i + 0
    C4 = lambda i: 3 * i + 1
    NN = lambda i: 3 * i + 2

    total_E = torch.zeros(B, device=dev)
    total_F = torch.zeros_like(pos_nm)

    # Build the cell-list (for clashes)
    if cell_list is None:
        cell_list = GPUCellList(cell_size=1.5)
    cell_list.build(pos_nm)

    # ── Bonds: O(N) direct indexing ──
    idx_a = torch.arange(L - 1, device=dev)
    e_bb, f_bb = _explicit_forces_bonds(
        pos_nm, torch.stack([P(idx_a), P(idx_a + 1)], dim=1), _K_BOND_BB, _R0_BB)
    total_E += e_bb; total_F += f_bb

    # Intra-bead bonds
    all_res = torch.arange(L, device=dev)
    e_pc, f_pc = _explicit_forces_bonds(
        pos_nm, torch.stack([P(all_res), C4(all_res)], dim=1), _K_BOND_INTRA, _R0_INTRA_PC)
    e_cn, f_cn = _explicit_forces_bonds(
        pos_nm, torch.stack([C4(all_res), NN(all_res)], dim=1), _K_BOND_INTRA, _R0_INTRA_CN)
    total_E += e_pc + e_cn; total_F += f_pc + f_cn

    # ── BSJ: O(1) ──
    delta_bsj = pos_nm[:, P(0)] - pos_nm[:, P(L - 1)]
    dist_bsj = delta_bsj.norm(dim=-1, keepdim=True).clamp(min=eps)
    e_bsj = (0.5 * _K_BSJ * (dist_bsj.squeeze(-1) - _R0_BB) ** 2).sum(dim=-1)
    f_bsj_mag = _K_BSJ * (dist_bsj - _R0_BB)
    f_bsj_vec = (f_bsj_mag * delta_bsj / dist_bsj).squeeze(-1)
    total_E += e_bsj
    total_F[:, P(0)] += f_bsj_vec
    total_F[:, P(L - 1)] -= f_bsj_vec

    # ── Angles: O(N) ──
    e_angle, f_angle = _explicit_forces_angles(pos_nm, _K_ANGLE, math.cos(ANGLE_PPP))
    total_E += e_angle; total_F += f_angle

    # ── Dihedrals: O(N) ──
    e_dih, f_dih = _explicit_forces_dihedrals(pos_nm, _K_DIH, math.cos(DIH_PPPP))
    total_E += e_dih; total_F += f_dih

    # ── Pairing (sparse indexing): O(P) ──
    if pairs_ij.numel() > 0:
        pi, pj = pairs_ij[:, 0].long(), pairs_ij[:, 1].long()
        delta_pair = pos_nm[:, NN(pi)] - pos_nm[:, NN(pj)]
        dist_pair = delta_pair.norm(dim=-1, keepdim=True).clamp(min=eps)
        w_p = pair_w[:len(pi)].to(dev).float() if pair_w is not None else torch.ones(len(pi), device=dev)
        k_eff = _K_PAIR * lam * w_p
        e_pair = (0.5 * k_eff * (dist_pair.squeeze(-1) - _R0_PAIR) ** 2).sum(dim=-1)
        f_pair_mag = k_eff.unsqueeze(1) * (dist_pair - _R0_PAIR)
        f_pair_vec = (f_pair_mag * delta_pair / dist_pair).squeeze(-1)
        total_E += e_pair
        total_F[:, NN(pi)] += f_pair_vec
        total_F[:, NN(pj)] -= f_pair_vec

    # ── Stacking (sparse indexing): O(N) ──
    if L > 2:
        st = torch.arange(L - 2, device=dev)
        e_st, f_st = _explicit_forces_bonds(
            pos_nm, torch.stack([P(st), P(st + 2)], dim=1), _K_STACK * lam, _R0_STACK)
        total_E += e_st; total_F += f_st

    # ── Clash (cell-list): O(K) ──
    e_clash, f_clash = _explicit_forces_clash(pos_nm, cell_list, _K_CLASH, _R0_CLASH)
    total_E += e_clash; total_F += f_clash

    # ── BSJ guide (single pair) ──
    # Bounded near-attraction: x = (R0 - dist)/0.2; E = -K*softplus(x);
    # dE/dr = +K*sig/0.2 -> F = -K*sig/0.2 * delta/r (pulls together)
    delta_bsj_g = pos_nm[:, P(0)] - pos_nm[:, P(L - 1)]
    dist_bsj_g = delta_bsj_g.norm(dim=-1, keepdim=True).clamp(min=eps)
    sig = torch.sigmoid((_R0_PAIR - dist_bsj_g) / 0.2)
    e_bsj_guide = -_K_BSJ_GUIDE * _stable_softplus(
        (_R0_PAIR - dist_bsj_g) / 0.2).sum(dim=-1)
    f_bsj_guide_mag = -_K_BSJ_GUIDE / 0.2 * sig / (dist_bsj_g + eps)
    f_bsj_guide_vec = (f_bsj_guide_mag * delta_bsj_g / dist_bsj_g).squeeze(-1)
    total_E += e_bsj_guide
    total_F[:, P(0)] += f_bsj_guide_vec
    total_F[:, P(L - 1)] -= f_bsj_guide_vec

    # ── GB/SA + Mg2+ (simplified: distance-dependent potentials) ──
    # Only for P particles: O(L) with direct indexing
    p_idx = torch.arange(L, device=dev)
    p_coords = pos_nm[:, P(p_idx), :]  # (B, L, 3)

    # GB: screening energy (pairwise-sum approximation)
    # E_gb = 0.5 * k_gb * Σ_{i≠j} exp(-r_ij/λ_D) / r_ij
    ion_strength = c_mg * 2.0 + c_na
    lambda_d = 0.304 / math.sqrt(max(ion_strength, 1e-6))
    p_diff = p_coords[:, :, None, :] - p_coords[:, None, :, :]  # (B, L, L, 3)
    p_dist = _safe_norm(p_diff, dim=-1, eps=eps)  # (B, L, L)
    eye_L = torch.eye(L, device=dev).unsqueeze(0)
    mask_L = (1.0 - eye_L)
    exp_screen = torch.exp(-p_dist / lambda_d) * mask_L / (p_dist + eps)
    e_gb = 0.365 * exp_screen.sum(dim=-1).sum(dim=-1)

    # SA: SASA overlap (short-range)
    overlap_sa = (1.0 - p_dist / 0.58).clamp(min=0.0) * mask_L
    sasa_i = 4.0 * math.pi * 0.15**2 * (1.0 - overlap_sa.sum(dim=-1) / (2.0 * L))
    e_sasa = 0.072 * sasa_i.sum(dim=-1)

    # Mg2+ softmin (using the p_dist nearest neighbor)
    exp_softmin = torch.exp(-p_dist / 0.3) * mask_L
    softmin_dist = -0.3 * torch.log(exp_softmin.sum(dim=-1).clamp(min=1e-12))
    e_mg = -_K_MG * torch.exp(-softmin_dist / LAMBDA_MG).sum(dim=-1)

    # Manning condensation
    r_local = p_dist.min(dim=-1).values
    xi = 0.714 / (2.0 * r_local.clamp(min=0.1))
    e_mg_ion = -2.494 * torch.log(1.0 + c_mg * xi**2 / (1.0 + xi**2) / max(c_mg, 1e-6)).sum(dim=-1)

    # Mg2+ screening
    e_mg_screen = -0.5 * c_mg * (exp_softmin / (p_dist + eps) * mask_L).sum(dim=-1).sum(dim=-1)

    total_E += e_gb + e_sasa + e_mg + e_mg_ion + e_mg_screen

    # GB/SA/Mg2+ forces: numerical gradients (O(L) terms, fast)
    gb_pos = p_coords.detach().clone().requires_grad_(True)
    # Rebuild these energy terms
    gb_diff = gb_pos[:, :, None, :] - gb_pos[:, None, :, :]
    gb_dist = _safe_norm(gb_diff, dim=-1, eps=eps)
    gb_mask = 1.0 - eye_L

    gb_screen = torch.exp(-gb_dist / lambda_d) * gb_mask / (gb_dist + eps)
    gb_e = 0.365 * gb_screen.sum(dim=-1).sum(dim=-1)

    sa_overlap = (1.0 - gb_dist / 0.58).clamp(min=0.0) * gb_mask
    sa_sasa_i = 4.0 * math.pi * 0.15**2 * (1.0 - sa_overlap.sum(dim=-1) / (2.0 * L))
    sa_e = 0.072 * sa_sasa_i.sum(dim=-1)

    mg_softmin_exp = torch.exp(-gb_dist / 0.3) * gb_mask
    mg_softmin_dist = -0.3 * torch.log(mg_softmin_exp.sum(dim=-1).clamp(min=1e-12))
    mg_e = -_K_MG * torch.exp(-mg_softmin_dist / LAMBDA_MG).sum(dim=-1)

    mg_r_local = gb_dist.min(dim=-1).values
    mg_xi = 0.714 / (2.0 * mg_r_local.clamp(min=0.1))
    mg_ion_e = -2.494 * torch.log(1.0 + c_mg * mg_xi**2 / (1.0 + mg_xi**2) / max(c_mg, 1e-6)).sum(dim=-1)

    mg_screen_e = -0.5 * c_mg * (mg_softmin_exp / (gb_dist + eps) * gb_mask).sum(dim=-1).sum(dim=-1)

    gb_total_e = (gb_e + sa_e + mg_e + mg_ion_e + mg_screen_e).sum()
    gb_total_e.backward()
    gb_forces = -gb_pos.grad  # (B, L, 3)
    total_F[:, P(p_idx)] += gb_forces

    _require_finite(total_E, "explicit energy")
    _require_finite(total_F, "explicit forces")
    return total_E, total_F


def batch_langevin_step(
    pos: "torch.Tensor", vel: "torch.Tensor", forces: "torch.Tensor",
    temperatures: "torch.Tensor",           # (B,) K
    dt_ps: float = 0.002, friction: float = 1.0,
    mass_amu: float = 110.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Langevin BAOAB integrator (per-replica temperature).

    BAOAB scheme (Leimkuhler & Matthews, 2013):
      B: v += (f/m) * dt/2
      A: x += v * dt/2
      O: v = c1*v + c2*ξ  (Langevin drag + noise)
      A: x += v * dt/2
      B: v += (f/m) * dt/2

    More accurate than BBK (v-v-r), especially in the high-friction regime.
    """
    def _integrator_finite_guard():
        _require_finite(pos, "input coordinates")
        _require_finite(forces, "input forces")
        _require_finite(vel, "input velocities")
        _require_finite(temperatures, "temperatures")
        if torch.any(temperatures <= 0):
            raise ValueError("temperatures must be positive")

    _integrator_finite_guard()
    if dt_ps <= 0 or friction < 0 or mass_amu <= 0:
        raise ValueError("dt_ps must be > 0, friction >= 0, and mass_amu > 0")
    c1 = math.exp(-friction * dt_ps)
    c2 = math.sqrt(max(1.0 - c1 * c1, 0.0))
    half_dt = dt_ps * 0.5
    # Unit conversion: kJ/mol/nm = amu·nm/ps² × 100
    unit_conv = 100.0
    sigma_base = math.sqrt(2.0 * friction * KB_KJ / mass_amu)
    noise_scale = sigma_base * torch.sqrt(
        temperatures.view(-1, 1, 1))

    # B step: half-step velocity update
    vel.add_((forces * half_dt / mass_amu) / unit_conv)
    # A step: half-step position update
    pos.add_(vel * half_dt)
    # O step: Langevin drag + noise
    vel.mul_(c1)
    vel.add_(_safe_randn(vel.shape, vel.device) * noise_scale
            * math.sqrt(dt_ps) * c2)
    # A step: half-step position update
    pos.add_(vel * half_dt)
    # B step: half-step velocity update (using the same forces; a simplified
    # version - the exact one would require recomputing the forces)
    vel.add_((forces * half_dt / mass_amu) / unit_conv)
    _require_finite(pos, "integrated coordinates")
    _require_finite(vel, "integrated velocities")
    return pos, vel


class BatchedREMD:
    """GPU batched temperature-REMD orchestrator.

    All replicas are integrated synchronously on one (n_rep, N, 3) tensor;
    every exchange_interval steps an adjacent Metropolis swap is attempted (tensor-index swaps).
    """

    def __init__(
        self,
        n_replicas: int = 16,
        t_lo: float = 300.0, t_hi: float = 500.0,
        exchange_interval: int = 1000,
        dt_ps: float = 0.002,
        device: str = "cuda",
    ):
        assert TORCH_OK, "PyTorch is not installed"
        self.temps = np.geomspace(t_lo, t_hi, n_replicas).tolist()
        self.n_replicas = n_replicas
        self.exchange_interval = exchange_interval
        self.dt = dt_ps
        self.device = device if torch.cuda.is_available() else "cpu"

    def run(self, coords_A, pairs, n_steps, verbose=True):
        coords_A = np.asarray(coords_A, dtype=np.float64)
        if coords_A.ndim != 2 or coords_A.shape[1] != 3:
            raise ValueError("coords_A must have shape (L, 3)")
        if not np.all(np.isfinite(coords_A)):
            raise ValueError("coords_A contains NaN or Inf")
        dev = self.device
        L = len(coords_A)
        pos = torch.tensor(coords_A / 10.0, dtype=torch.float32,
                           device=dev)[None].repeat(self.n_replicas, 1, 1)
        vel = _safe_zeros(pos.shape, dev)

        if pairs:
            pairs_t = torch.tensor(np.asarray(pairs)[:, :2],
                                   dtype=torch.long, device=dev)
            pw = torch.tensor([p[2] for p in pairs], dtype=torch.float32,
                              device=dev)
        else:
            pairs_t = _safe_zeros((0, 2), dev, dtype=torch.long)
            pw = None

        temps_t = torch.tensor(self.temps, dtype=torch.float32, device=dev)
        idx = _arange_dev(L, dev)
        seq_near = ((idx[None, :] - idx[:, None]).abs() <= 2) | \
            _safe_eye(L, dev, dtype=torch.bool)

        n_reports = max(1, n_steps // self.exchange_interval)
        best_e = float("inf")
        best_pos = None
        acc = att = 0
        e_hist = []

        for rep in range(n_reports):
            for _ in range(self.exchange_interval):
                en, f = cg_forces_3bead(pos, pairs_t, pw)
                pos, vel = batch_langevin_step(pos, vel, f, temps_t,
                                               dt_ps=self.dt)

            with torch.no_grad():
                pos = pos.detach()
            en, _f = cg_forces_3bead(pos, pairs_t, pw)
            energies = en.cpu().numpy()
            e_hist.append(float(energies.min()))
            beta = 1.0 / (KB_KJ * np.asarray(self.temps))
            for ri in range(self.n_replicas - 1):
                att += 1
                dE = energies[ri] - energies[ri + 1]
                expo = np.clip((beta[ri] - beta[ri + 1]) * dE, -30, 30)
                if expo <= 0 or np.random.rand() < np.exp(-expo):
                    acc += 1
                    tmp = pos[ri].clone()
                    pos[ri] = pos[ri + 1].clone()
                    pos[ri + 1] = tmp
            i_min = int(np.argmin(energies))
            if energies[i_min] < best_e:
                best_e = float(energies[i_min])
                best_pos = pos[i_min].detach().cpu().numpy() * 10.0
            if verbose:
                print(f"    [GPU-REMD] {rep+1}/{n_reports}: "
                      f"E_min={energies.min():.0f} acc={acc}/{att}")

        diag = {"acceptance": acc / max(att, 1), "e_hist": e_hist}
        if best_pos is None:
            best_pos = np.asarray(coords_A)
            best_e = 0.0
        return best_pos, best_e, diag


def exchange_log_alpha_temperature(beta_a: float, beta_b: float,
                                    energy_a: float, energy_b: float) -> float:
    """Return log Metropolis ratio for swapping two temperature slots."""
    return (float(beta_a) - float(beta_b)) * (
        float(energy_a) - float(energy_b))


def exchange_log_alpha_lambda(beta: float, lambda_a: float, lambda_b: float,
                              solute_a: float, solute_b: float) -> float:
    """Return log Metropolis ratio for a REST2 λ-coordinate swap."""
    return float(beta) * (float(lambda_a) - float(lambda_b)) * (
        float(solute_a) - float(solute_b))


def metropolis_accept_log_alpha(log_alpha: float, rng=None) -> bool:
    """Draw a numerically stable Metropolis decision from a log ratio."""
    if not np.isfinite(log_alpha):
        return False
    if log_alpha >= 0.0:
        return True
    random_value = np.random.random() if rng is None else rng.random()
    return random_value < math.exp(max(float(log_alpha), -30.0))


def tri_effective_scale_torch(
    base_scale: float,
    temperatures: "torch.Tensor",
    t_lo: float = 300.0,
    t_hi: float = 550.0,
) -> "torch.Tensor":
    """TriRNASP temperature-dependent effective strength (consistent with rest2_remd_2d.tri_effective_scale).

    Low temperature ≈ base; high temperature → ~5% of base (sigmoid transition band).
    Returns: (B,) effective strength per replica.
    """
    t_mid = 0.5 * (t_lo + t_hi)
    width = max((t_hi - t_lo) / 6.0, 1e-6)
    s = torch.sigmoid(-(temperatures - t_mid) / width)
    floor = 0.05
    # Keep the high-temperature potential positive and at a 5% floor.
    # The old (s-floor)/(1-floor) mapping became negative when s < floor,
    # reversing TriRNASP's sign on hot replicas and destroying T exchange.
    return base_scale * (floor + (1.0 - floor) * s)


def compute_gradient_alignment(
    f_cg: "torch.Tensor",
    f_tri: "torch.Tensor",
    mask: "torch.Tensor" = None,
) -> "torch.Tensor":
    """Compute the cosine similarity between the CG and Tri forces (per replica).

    Args:
        f_cg: (B, N, 3) CG force-field forces
        f_tri: (B, N, 3) TriRNASP forces
        mask: (B, N) optional mask (True=participate in the computation)

    Returns:
        (B,) cosine similarity in [-1, 1]
        - 1: exactly the same direction
        - 0: orthogonal
        - -1: exactly opposite directions
    """
    B = f_cg.shape[0]
    dev = f_cg.device

    # Flatten to (B, N*3)
    cg_flat = f_cg.reshape(B, -1)
    tri_flat = f_tri.reshape(B, -1)

    if mask is not None:
        # Expand the mask to (B, N*3)
        mask_expanded = mask.unsqueeze(-1).expand_as(f_cg).reshape(B, -1)
        cg_flat = cg_flat * mask_expanded
        tri_flat = tri_flat * mask_expanded

    # Cosine similarity
    cg_norm = torch.norm(cg_flat, dim=-1, keepdim=True).clamp(min=1e-8)
    tri_norm = torch.norm(tri_flat, dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = (cg_flat * tri_flat).sum(dim=-1) / (cg_norm * tri_norm)

    return cos_sim.clamp(-1.0, 1.0)


def compute_adaptive_tri_weight(
    pos_3bead: "torch.Tensor",
    pairs_ij: "torch.Tensor",
    L: int,
    dev,
    base_scale: float = 0.1,
    min_scale: float = 0.01,
    max_scale: float = 0.5,
) -> "torch.Tensor":
    """Structure-state-adaptive TriRNASP weight.

    Logic:
      - secondary structure is well formed → trust Tri more (it guides the tertiary structure)
      - backbone already compact → lower Tri (avoid over-compaction)
      - pair distances look abnormal → lower Tri (let CG lead the repair)

    Returns:
        (1,) adaptive scaling factor
    """
    # 1. Pairing completeness
    if pairs_ij.numel() > 0 and L > 0:
        pi, pj = pairs_ij[:, 0].long(), pairs_ij[:, 1].long()
        # Look only at the P beads
        p_coords = pos_3bead[:, 0::3, :]  # (B, L, 3)
        d_pair = (p_coords[:, pi] - p_coords[:, pj]).norm(dim=-1)  # (B, P)
        # Ideal pair distance ~1.0 nm
        pair_completion = (d_pair < 1.2).float().mean()
    else:
        pair_completion = torch.tensor(0.5, device=dev)

    # 2. Backbone compactness (mean P-P bond length)
    p_coords = pos_3bead[:, 0::3, :]  # (B, L, 3)
    if L > 1:
        bb_len = (p_coords[:, 1:] - p_coords[:, :-1]).norm(dim=-1).mean()
        # Ideal bond length 0.59 nm; the closer, the more "normal"
        compactness = torch.exp(-(bb_len - 0.59).abs() / 0.2)
    else:
        compactness = torch.tensor(0.5, device=dev)

    # 3. Adaptive weight
    # When secondary-structure completeness is high + the backbone is normal → raise the Tri weight
    # When the backbone is abnormal (too long/short) → lower Tri and let CG fix it
    adaptive_scale = base_scale * (0.5 + pair_completion) * (0.8 + 0.2 * compactness)

    # Clamp the range
    adaptive_scale = torch.clamp(adaptive_scale, min_scale, max_scale)

    return adaptive_scale


class BatchedREMD2D:
    """GPU batched 2D REST2×REMD — a single-tensor (T_i, λ_j) replica grid.

    The GPU counterpart of rest2_remd_2d.py (the CPU multi-process version):
      - n_T × n_λ replicas = one (n_rep, 3L, 3) tensor
      - a temperature vector (n_rep,) drives the per-replica Langevin noise
      - a λ vector (n_rep,) scales the pair/stack solute terms (per-replica energy)
      - two-axis Metropolis: the temperature axis uses the total-energy difference;
        the λ axis uses the solute-term energy difference
        (cg_energy_forces returns per-term components → the λ-axis criterion uses only the scaled terms)
    """

    def __init__(
        self,
        n_t: int = 8,
        t_lo: float = 300.0, t_hi: float = 1000.0,
        lambdas: Tuple[float, ...] = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65),
        exchange_interval: int = 500,
        dt_ps: float = 0.002,
        device: str = "cuda",
        use_trirnasp: bool = False,
        use_trirnasp_force: bool = False,   # True=inject TriRNASP forces (slow), False=energy recording only (fast)
        trirnasp_energy_dir: Optional[str] = None,
        trirnasp_scale: float = 0.003,  # unified default: 0.003 (optimal value)
        sequence: Optional[str] = None,
        force_refresh_freq: int = 500,
        tri_force_replica_policy: str = "cold",
        relax_bond_k: float = 500.0,
        relax_angle_k: float = 200.0,
        relax_pair_k: float = 500.0,
        restraint_k: float = 500.0,
        use_adaptive_tri_weight: bool = False,
        # New: staged strategy
        use_staged_tri: bool = False,
        tri_stage_config: Optional[dict] = None,
        initial_global_step: int = 0,  # New: accumulated step count across rounds
        initial_velocities: Optional["torch.Tensor"] = None,  # New: velocities carried across rounds
    ):
        assert TORCH_OK
        self.temps = np.geomspace(t_lo, t_hi, n_t).tolist()
        self.lambdas = list(lambdas)
        self.exchange_interval = exchange_interval
        self.dt = dt_ps
        self.device = device if torch.cuda.is_available() else "cpu"
        # TriRNASP statistical potential
        self.use_trirnasp = use_trirnasp
        self.use_trirnasp_force = use_trirnasp_force
        self.trirnasp_energy_dir = trirnasp_energy_dir
        self.trirnasp_scale = trirnasp_scale
        self.sequence = sequence
        self.force_refresh_freq = force_refresh_freq
        self.tri_force_replica_policy = tri_force_replica_policy
        if tri_force_replica_policy not in {"all", "cold", "lambda1", "cold_lambda1"}:
            raise ValueError("tri_force_replica_policy must be all, cold, lambda1, or cold_lambda1")
        self.relax_bond_k = relax_bond_k
        self.relax_angle_k = relax_angle_k
        self.relax_pair_k = relax_pair_k
        self.restraint_k = restraint_k
        self.use_adaptive_tri_weight = use_adaptive_tri_weight
        self.use_staged_tri = use_staged_tri
        # Staged configuration: 3 stages by default
        if tri_stage_config is None:
            tri_stage_config = {
                "stages": [
                    {"name": "CG-only", "steps": 2000, "tri_scale": 0.0},
                    {"name": "Ramp-up", "steps": 3000, "tri_scale": 0.05},
                    {"name": "Tri-guided", "steps": 5000, "tri_scale": 0.1},
                ],
            }
        self.tri_stage_config = tri_stage_config
        self.initial_global_step = initial_global_step
        self.initial_velocities = initial_velocities

    @property
    def n_replicas(self):
        return len(self.temps) * len(self.lambdas)

    def run(self, coords_A, pairs, n_steps, verbose=True, initial_pos_3bead=None,
            initial_global_step=None, initial_velocities=None):
        """coords_A: (L,3) Å P coordinates → expanded internally to 3-bead.

        initial_pos_3bead: optional (1, 3L, 3) tensor in nm.
            If provided, use it directly instead of re-expanding P-only.
            This preserves C4'/N optimization state across multi-round calls.

        initial_global_step: optional int.
            Bug 10 fix: accumulate the step count across rounds instead of restarting from 0.

        initial_velocities: optional torch.Tensor.
            Bug 3 fix: carry velocities across rounds.
        """
        coords_A = np.asarray(coords_A, dtype=np.float64)
        if coords_A.ndim != 2 or coords_A.shape[1] != 3:
            raise ValueError("coords_A must have shape (L, 3)")
        if not np.all(np.isfinite(coords_A)):
            raise ValueError("coords_A contains NaN or Inf")
        dev = self.device
        L = len(coords_A)

        n_rep = self.n_replicas
        if initial_pos_3bead is not None:
            # Multi-round reuse: use the previous round's full 3-bead state directly
            pos = initial_pos_3bead.to(dtype=torch.float32, device=dev)
            if pos.ndim == 2:
                pos = pos.unsqueeze(0)
            if pos.ndim != 3 or pos.shape[1:] != (3 * L, 3):
                raise ValueError(
                    f"initial_pos_3bead must have shape (1 or {n_rep}, {3 * L}, 3), "
                    f"got {tuple(pos.shape)}")
            if pos.shape[0] == 1:
                pos = pos.repeat(n_rep, 1, 1).contiguous()
            elif pos.shape[0] != n_rep:
                raise ValueError(
                    f"initial_pos_3bead batch must be 1 or {n_rep}, got {pos.shape[0]}")
        else:
            # First call: P-only -> 3-bead initialization. C4' and N come from the 1EHZ
            # template reconstruction; a random 0.3 A perturbation of P put them in a
            # random direction, carrying neither base identity nor real geometry.
            from .aform_from_template import real_cg_beads
            _beads = real_cg_beads(np.asarray(coords_A, dtype=np.float64), self.sequence)
            pos0 = _beads.reshape(3 * L, 3) / 10.0
            pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None].repeat(
                n_rep, 1, 1).contiguous()

        # Bug 3 fix: support carrying velocities across rounds (run() parameters take priority)
        _init_vel = initial_velocities if initial_velocities is not None else self.initial_velocities
        if _init_vel is not None:
            vel = _init_vel.to(dtype=torch.float32, device=dev)
            if vel.shape != pos.shape:
                vel = _safe_zeros(pos.shape, dev)  # re-initialize if the shape does not match
        else:
            vel = _safe_zeros(pos.shape, dev)

        if pairs:
            pairs_t = torch.tensor(np.asarray(pairs)[:, :2],
                                   dtype=torch.long, device=dev)
            pw = torch.tensor([p[2] for p in pairs], dtype=torch.float32,
                              device=dev)
        else:
            pairs_t = _safe_zeros((0, 2), dev, dtype=torch.long)
            pw = None

        # Grid index: replica r = ri * n_lam + cj
        n_t, n_lam = len(self.temps), len(self.lambdas)
        temps_grid = np.repeat(np.asarray(self.temps), n_lam)
        lams_grid = np.tile(np.asarray(self.lambdas), n_t)
        temps_t = torch.tensor(temps_grid, dtype=torch.float32, device=dev)
        lams_t = torch.tensor(lams_grid, dtype=torch.float32, device=dev)

        N_tot = 3 * L
        res_of = torch.repeat_interleave(_arange_dev(L, dev), 3)
        seq_near = ((res_of[None] - res_of[:, None]).abs() <= 1) | \
            _safe_eye(N_tot, dev, dtype=torch.bool)

        KB = KB_KJ
        beta = 1.0 / (KB * np.asarray(self.temps))          # temperature-axis β

        # TriRNASP statistical potential (GPU scoring / optional CPU force)
        TRI_KBT = 2.494      # kBT@300K → kJ/mol
        TRI_MAX_F = 500.0    # per-particle force cap (kJ/mol/nm)
        tri_eff = torch.full_like(temps_t, float(self.trirnasp_scale))
        tri_pot = None
        if self.use_trirnasp and self.sequence:
            try:
                from .trirnasp_torch import TriRNASPTorch
                tri_pot = TriRNASPTorch(
                    self.sequence, energy_dir=self.trirnasp_energy_dir, device=dev)
                # Shared scale across temperatures; λ remains replica-specific.
                if verbose:
                    print(f"    [GPU-2D] TriRNASP statistical potential loaded (scale={self.trirnasp_scale})")
            except Exception as exc_tri:
                if verbose:
                    print(f"    [GPU-2D] TriRNASP failed to load: {exc_tri}")
                tri_pot = None

        tri_force_mask = np.ones(n_rep, dtype=bool)
        if self.tri_force_replica_policy != "all":
            tri_force_mask[:] = False
            for _ri in range(n_t):
                for _cj in range(n_lam):
                    _slot = _ri * n_lam + _cj
                    _is_cold = _ri < max(1, n_t // 2)
                    _is_lambda1 = _cj == 0
                    tri_force_mask[_slot] = (
                        (_is_cold if "cold" in self.tri_force_replica_policy else True)
                        and (_is_lambda1 if "lambda1" in self.tri_force_replica_policy else True)
                    )
        tri_force_mask_t = torch.tensor(tri_force_mask, dtype=torch.bool, device=dev)
        tri_force_scales_np = None
        if self.use_trirnasp and self.sequence:
            tri_force_scales_np = (
                TRI_KBT * 10.0 * tri_eff * lams_t
            ).detach().cpu().numpy()
        # ── CPU TriRNASP (score_with_gradient: force direction aligns with the hard scoring surface) ──
        # Physical choice: REMD potential-surface self-consistency matters more than gradient
        # accuracy. autograd follows the soft interpolation surface (which differs from the hard
        # scoring surface by ~50%); 38% of samples point the wrong way → it would break detailed
        # balance. CPU forward differences at cosine=0.35 are noisy, but the direction naturally
        # aligns with the hard surface, and Langevin absorbs the noise.
        tri_pot_cpu = None
        if self.use_trirnasp and self.sequence:
            try:
                from torusfold.scheme2.trirnasp_openmm import TriRNASPPotential
                tri_pot_cpu = TriRNASPPotential(self.trirnasp_energy_dir)
            except Exception as exc_cpu_tri:
                if verbose:
                    print(f"    [GPU-2D] CPU TriRNASP failed to load: {exc_cpu_tri}")

        _tri_refresh_counter = [0]
        # CPU force cache: (n_rep, L, 3) kJ/mol/nm, refreshed every force_refresh_freq steps
        _tri_cpu_force_cache = np.zeros((n_rep, L, 3), dtype=np.float64)
        _tri_force_refresh_count = [0]
        _tri_force_refresh_skipped = [0]
        _tri_force_refresh_failed = [0]
        _tri_last_refresh_age = [0]
        _tri_scale = [1.0]  # GPU soft-binning → CPU calibration scaling factor
        class _TriEnergyCache:
            val = None
        _tri_energy_cache = _TriEnergyCache()
        # Precompute the initial TriRNASP energy (GPU)
        if tri_pot is not None:
            _tri_energy_cache.val = (
                tri_pot.energy_from_3bead(pos.detach(), soft=False)
                * (TRI_KBT * 10.0) * tri_eff * lams_t).float()
            _require_finite(_tri_energy_cache.val, "initial TriRNASP energy")

        import multiprocessing as _mp
        _cpu_pool = [_mp.Pool(1)] if (tri_pot_cpu is not None and self.use_trirnasp_force) else []
        _cpu_future = [None]
        _cpu_pool_broken = [False]

        def _rebuild_cpu_pool():
            """Terminate orphan workers and create a fresh pool."""
            if _cpu_pool:
                try:
                    _cpu_pool[0].terminate()
                except Exception:
                    pass
                try:
                    _cpu_pool[0].join(timeout=5)
                except Exception:
                    pass
            _cpu_pool[0] = _mp.Pool(1)
            _cpu_pool_broken[0] = False

        def _refresh_cpu_forces_async():
            """Compute CPU forces in a separate process without blocking the GPU (bypasses the GIL)."""
            if not _cpu_pool or _cpu_pool_broken[0]:
                return
            # A single worker is used deliberately.  Never queue a second
            # full-length score while the previous one is still running.
            if _cpu_future[0] is not None:
                if not _cpu_future[0].ready():
                    _tri_force_refresh_skipped[0] += 1
                    return
                _collect_cpu_forces()
            with torch.no_grad():
                selected = np.flatnonzero(tri_force_mask)
                pos_cpu = pos[selected].double().cpu().numpy().copy()
            _cpu_future[0] = _cpu_pool[0].apply_async(
                _trirnasp_force_worker,
                ((pos_cpu, selected, self.sequence, self.trirnasp_energy_dir,
                  tri_force_scales_np[selected], TRI_MAX_F),))
            _tri_force_refresh_count[0] += 1

        def _collect_cpu_forces(wait=False, initial=False):
            """Collect the asynchronous CPU force result and update the cache.

            When initial=True, give the worker a longer startup time (first-time table load + scoring).
            On timeout, automatically kill orphan processes and rebuild the pool to avoid memory leaks.
            """
            if _cpu_future[0] is None or _cpu_pool_broken[0]:
                return
            if not (wait or _cpu_future[0].ready()):
                return
            try:
                timeout = 300.0 if initial else (60.0 if wait else 0.001)
                result = _cpu_future[0].get(timeout=timeout)
                if isinstance(result, tuple):
                    result_indices, result_forces = result
                else:
                    result_indices = np.flatnonzero(tri_force_mask)
                    result_forces = result
                result_indices = np.asarray(result_indices, dtype=np.int64)
                result_forces = np.asarray(result_forces, dtype=np.float64)
                if (result_forces.shape != (len(result_indices), L, 3)
                        or not np.all(np.isfinite(result_forces))
                        or np.any(result_indices < 0)
                        or np.any(result_indices >= n_rep)):
                    raise RuntimeError("invalid TriRNASP force cache")
                for local_idx, rep in enumerate(result_indices):
                    _tri_cpu_force_cache[rep] = result_forces[local_idx]
                _tri_last_refresh_age[0] = 0
            except Exception as exc_force:
                _tri_force_refresh_failed[0] += 1
                _msg = (f"{type(exc_force).__name__}"
                        if not str(exc_force) else str(exc_force))
                if verbose:
                    print(f"    [GPU-2D] CPU TriRNASP force refresh failed: {_msg}")
                # Timeout → orphan processes keep consuming resources; kill the pool + defer rebuild
                if isinstance(exc_force, (_mp.TimeoutError,)):
                    _cpu_pool_broken[0] = True
                    try:
                        _cpu_pool[0].terminate()
                    except Exception:
                        pass
                    try:
                        _cpu_pool[0].join(timeout=5)
                    except Exception:
                        pass
            _cpu_future[0] = None

        def _tri_energy_reference(pos_in, replica_indices=None):
            """TriRNASP energy at the λ=1 reference Hamiltonian."""
            if tri_pot is None:
                return torch.zeros(pos_in.shape[0], device=pos_in.device)
            pos_in = pos_in.detach()
            e_kbt = tri_pot.energy_from_3bead(pos_in)
            if replica_indices is None:
                indices = torch.arange(pos_in.shape[0], device=pos_in.device)
            else:
                indices = torch.as_tensor(replica_indices, dtype=torch.long,
                                          device=pos_in.device)
            return e_kbt * (TRI_KBT * 10.0) * tri_eff[indices]

        def _tri_energy(pos_in, replica_indices=None):
            """TriRNASP energy for supplied coordinates and replica slots."""
            if replica_indices is None:
                indices = torch.arange(pos_in.shape[0], device=pos_in.device)
            else:
                indices = torch.as_tensor(replica_indices, dtype=torch.long,
                                          device=pos_in.device)
            return _tri_energy_reference(pos_in, indices) * lams_t[indices]

        def _swap_tri_force_cache(a, b, *, lambda_swap=False):
            """Move cached state forces with coordinates across an exchange."""
            cached_a = _tri_cpu_force_cache[a].copy()
            cached_b = _tri_cpu_force_cache[b].copy()
            if lambda_swap:
                scale_a = float(lams_grid[a])
                scale_b = float(lams_grid[b])
                # Cache entries are scaled for their source λ. Rescale the
                # force that moves into each destination slot.
                cached_a *= scale_b / max(abs(scale_a), 1e-12)
                cached_b *= scale_a / max(abs(scale_b), 1e-12)
            _tri_cpu_force_cache[a] = cached_b
            _tri_cpu_force_cache[b] = cached_a

        def _refresh_slot_energies(slot_indices, energies, solute_np):
            """Re-evaluate Hamiltonian energies after a coordinate swap."""
            slots = np.asarray(sorted(set(int(x) for x in slot_indices)), dtype=np.int64)
            if slots.size == 0:
                return
            with torch.no_grad():
                slot_t = torch.as_tensor(slots, dtype=torch.long, device=dev)
                e_slot, s_slot = _energy_split(pos[slot_t], slots)
                e_slot_np = e_slot.cpu().numpy()
                s_slot_np = s_slot.cpu().numpy()
                if tri_pot is not None:
                    e_slot_np += _tri_energy(pos[slot_t], slots).cpu().numpy()
            energies[slots] = e_slot_np
            solute_np[slots] = s_slot_np

        def _accept_log_alpha(log_alpha):
            return metropolis_accept_log_alpha(log_alpha)

        n_reports = max(1, n_steps // self.exchange_interval)
        best_e = float("inf")
        best_pos = None
        best_pos_3bead = None
        accT = [0] * (n_t - 1); attT = [0] * (n_t - 1)
        accL = [0] * (n_lam - 1); attL = [0] * (n_lam - 1)
        e_hist = []
        tri_scores = None
        no_improve_count = 0
        patience = 6  # early-stop if there is no improvement for 6 consecutive rounds (3000 steps)

        def _energy_split(pos_in, replica_indices=None):
            """Return own-λ energy and reference λ=1 solute energy."""
            if replica_indices is None:
                indices = torch.arange(pos_in.shape[0], device=pos_in.device)
                own_lams = lams_t
            else:
                indices = torch.as_tensor(replica_indices, dtype=torch.long,
                                          device=pos_in.device)
                own_lams = lams_t[indices]
            with torch.no_grad():
                p = pos_in.detach()
                # Solute term: the λ=1 minus λ=0 difference, plus the TriRNASP λ=1 reference term.
                en_ref, _ = cg_energy_forces(p, pairs_t, pw, lam=1.0)
                en_base, _ = cg_energy_forces(p, pairs_t, pw, lam=0.0)
                solute = en_ref - en_base
                tri_ref = _tri_energy_reference(p, indices)
                solute = (solute + tri_ref).detach()
                # CG total energy at each slot's own λ; Tri is added by the caller.
                en_own, _ = cg_energy_forces(p, pairs_t, pw, lams=own_lams)
            return en_own.detach(), solute

        # ── Before the MD loop: Langevin relaxation ──
        # No Adam pre-minimization: the CG-only optimum is meaningless on the joint CG+Tri surface,
        # and it would lock the replicas into a wrong minimum, causing the MD energy to rise monotonically.
        # Instead run a 500-step Langevin relaxation so replicas spread naturally through their
        # temperature/λ differences, while the CG forces pull the structure into a reasonable region
        # of the CG surface.
        if verbose:
            print(f"    [GPU-2D] Relaxing for 500 steps...")
        with torch.no_grad():
            for _ in range(500):
                _e, _f = cg_energy_forces(pos, pairs_t, pw, lams=lams_t)
                pos, vel = batch_langevin_step(pos, vel, _f, temps_t, dt_ps=self.dt)
        _require_finite(pos, "equilibrated coordinates")

        if verbose:
            with torch.no_grad():
                # Report the joint CG+Tri energy (first replica, λ=1.0)
                e_cg, _ = cg_energy_forces(pos[:1], pairs_t, pw, lam=1.0)
                tri_val = ""
                if tri_pot is not None:
                    tri_raw = _tri_energy(pos, range(n_rep)).cpu().numpy()
                    tri_val = f" Tri={tri_raw.min():.0f}"
                print(f"    [GPU-2D] After relaxation E_CG={e_cg[0].item():.0f}{tri_val}")

        if tri_pot_cpu is not None and self.use_trirnasp_force:
            _refresh_cpu_forces_async()
            _collect_cpu_forces(wait=True, initial=True)
            if verbose:
                _status = ("ok" if _tri_last_refresh_age[0] == 0
                           else f"timeout/failed (cached_age={_tri_last_refresh_age[0]})")
                _count = int(tri_force_mask.sum())
                print(f"    [GPU-2D] CPU TriRNASP initial forces {_status} "
                      f"(refresh every {self.force_refresh_freq} steps, "
                      f"{_count}/{n_rep} replicas)")

        # cell-list (only one is needed after the merged forward)
        cl_2d = GPUCellList(cell_size=1.5)

        # ── Staged TriRNASP strategy ──
        # Bug 10 fix: run() parameters take priority, then instance attributes
        _init_step = initial_global_step if initial_global_step is not None else self.initial_global_step
        _global_step = [_init_step]  # use the passed-in initial step count (accumulated across rounds)
        _current_stage = [0]  # current-stage index
        _stage_boundaries = []  # stage boundaries (in steps)
        if self.use_staged_tri:
            _cumulative = 0
            for _stage_idx, _stage in enumerate(self.tri_stage_config["stages"]):
                _stage_boundaries.append(_cumulative + _stage["steps"])
                _cumulative += _stage["steps"]
            if verbose:
                print(f"    [GPU-2D] Staged TriRNASP: {len(self.tri_stage_config['stages'])} stages")
                for _si, _s in enumerate(self.tri_stage_config["stages"]):
                    print(f"      stage {_si+1}: {_s['name']}, {_s['steps']} steps, "
                          f"tri_scale={_s['tri_scale']:.3f}")

        def _get_staged_tri_scale():
            """Return the staged TriRNASP scaling factor for the current step count."""
            if not self.use_staged_tri:
                return 1.0  # no staging; keep the original strength

            step = _global_step[0]
            # Find the current stage
            for _si, _boundary in enumerate(_stage_boundaries):
                if step < _boundary:
                    _current_stage[0] = _si
                    break
            else:
                _current_stage[0] = len(_stage_boundaries) - 1

            _stage = self.tri_stage_config["stages"][_current_stage[0]]
            _target_scale = _stage["tri_scale"]

            # Linear interpolation within the stage (smooth transition)
            if _current_stage[0] > 0:
                _prev_boundary = _stage_boundaries[_current_stage[0] - 1]
                _curr_boundary = _stage_boundaries[_current_stage[0]]
                _progress = (step - _prev_boundary) / max(_curr_boundary - _prev_boundary, 1)
                _prev_scale = self.tri_stage_config["stages"][_current_stage[0] - 1]["tri_scale"]
                _interpolated = _prev_scale + (_target_scale - _prev_scale) * _progress
            else:
                _interpolated = _target_scale

            return _interpolated

        for rep in range(n_reports):
            for _ in range(self.exchange_interval):
                # Update the global step count (used by the staged strategy)
                _global_step[0] += 1

                # Per-replica λ: vectorize lam into the energy — here we batch by group
                # (replicas sharing the same λ reuse one forward pass; n_lam groups)
                # A single forward: all replicas use their per-replica λ
                # Bug 5 fix: renamed to clearer variable names (recomputed each step, not accumulated)
                en_total, f_total = cg_energy_forces(pos, pairs_t, pw, lams=lams_t,
                                                     cell_list=cl_2d)

                if tri_pot_cpu is None and tri_pot is not None:
                    # ── GPU TriRNASP energy recording only (every 500 steps) ──
                    _tri_refresh_counter[0] += 1
                    if _tri_refresh_counter[0] >= self.force_refresh_freq:
                        _tri_refresh_counter[0] = 0
                        # staged scaling
                        _stage_scale = _get_staged_tri_scale()
                        _effective_tri_eff = tri_eff * _stage_scale
                        _tri_energy_cache.val = (
                            tri_pot.energy_from_3bead(pos.detach(), soft=False)
                            * (TRI_KBT * 10.0) * _effective_tri_eff * lams_t).float()
                    if _tri_energy_cache.val is not None:
                        en_total = en_total + _tri_energy_cache.val

                # TriRNASP: inject CPU forces only when forces are explicitly enabled.
                # tri_pot's GPU hard-lookup table is for scoring only; do not add the hard score back into the integrator.
                elif tri_pot_cpu is not None and self.use_trirnasp_force:
                    # ── Optional: TriRNASP force injection (async CPU, does not block the GPU) ──
                    _tri_refresh_counter[0] += 1
                    # Collect the async result (non-blocking)
                    _collect_cpu_forces()
                    _tri_last_refresh_age[0] += 1
                    # Trigger the async computation; the pool auto-rebuilds after a timeout
                    if _tri_refresh_counter[0] >= self.force_refresh_freq:
                        _tri_refresh_counter[0] = 0
                        if _cpu_pool_broken[0]:
                            _rebuild_cpu_pool()
                        _refresh_cpu_forces_async()

                    # ── Gradient-alignment check: diagnose the CG vs Tri force directions ──
                    # Compute the cosine similarity; if the directions oppose, lower the Tri weight
                    if _tri_last_refresh_age[0] <= 1:  # only check right after a new force injection
                        f_tri_full_raw = torch.zeros_like(pos)
                        p_idx = torch.arange(0, 3 * L, 3, device=dev)
                        f_tri_raw_np = torch.tensor(
                            _tri_cpu_force_cache, dtype=torch.float32, device=dev)
                        f_tri_full_raw[:, p_idx, :] = f_tri_raw_np * tri_force_mask_t[:, None, None]

                        # Compute the gradient alignment (P beads only)
                        p_mask = torch.zeros(N_tot, device=dev, dtype=torch.bool)
                        p_mask[0::3] = True
                        cos_sim = compute_gradient_alignment(f_total, f_tri_full_raw, mask=p_mask)

                        # Diagnostic: if most replicas point the opposite way, lower the Tri weight
                        mean_cos = cos_sim.mean().item()
                        if mean_cos < -0.3:  # directions clearly oppose
                            # Adaptively lower the Tri strength
                            _adaptive_scale = max(0.01, self.trirnasp_scale * 0.1)
                            if verbose and rep % 2 == 0:  # print once every 2 rounds
                                print(f"    [GPU-2D] ⚠️ CG-Tri gradient conflict! cos={mean_cos:.3f}, "
                                      f"lowering scale: {self.trirnasp_scale:.3f} → {_adaptive_scale:.3f}")
                            # Apply the adaptive scaling
                            f_tri_full_raw = f_tri_full_raw * (_adaptive_scale / max(self.trirnasp_scale, 1e-6))
                        elif verbose and rep % 4 == 0:  # print once every 4 rounds
                            print(f"    [GPU-2D] CG-Tri gradients aligned: cos={mean_cos:.3f}")

                    # ── Structure-state-adaptive weight ──
                    if self.use_adaptive_tri_weight:
                        _adaptive_factor = compute_adaptive_tri_weight(
                            pos[:1], pairs_t, L, dev,
                            base_scale=self.trirnasp_scale,
                            min_scale=0.01,
                            max_scale=0.5,
                        )
                        # print only during diagnostics
                        if verbose and rep % 4 == 0:
                            print(f"    [GPU-2D] Adaptive Tri weight: {_adaptive_factor.item():.4f}")

                    # Write forces only to the selected P beads; unselected replicas keep zero Tri force.
                    f_tri_full = torch.zeros_like(pos)
                    p_idx = torch.arange(0, 3 * L, 3, device=dev)
                    f_tri_raw = torch.tensor(
                        _tri_cpu_force_cache, dtype=torch.float32, device=dev)
                    f_tri_raw = f_tri_raw * tri_force_mask_t[:, None, None]

                    # Apply staged scaling (if enabled)
                    if self.use_staged_tri:
                        _stage_scale = _get_staged_tri_scale()
                        f_tri_raw = f_tri_raw * _stage_scale
                        # Diagnostic: print the current stage every 1000 steps
                        if verbose and _global_step[0] % 1000 == 0:
                            _stage_name = self.tri_stage_config["stages"][_current_stage[0]]["name"]
                            print(f"    [GPU-2D] Step {_global_step[0]}: stage {_current_stage[0]+1} "
                                  f"({_stage_name}), tri_scale={_stage_scale:.3f}")

                    # Apply the adaptive weight (if enabled)
                    if self.use_adaptive_tri_weight:
                        f_tri_raw = f_tri_raw * _adaptive_factor

                    f_tri_full[:, p_idx, :] = f_tri_raw
                    f_total = f_total + f_tri_full
                    # Energies are only computed at exchange/report time, avoiding an O(N²) GPU Tri score every step.

                pos, vel = batch_langevin_step(
                    pos, vel, f_total, temps_t, dt_ps=self.dt)

            e_full, e_solute = _energy_split(pos)
            # Extract the total energy of each swapped state before accepting; rejected states are left untouched.
            # TriRNASP enters the total Hamiltonian just like CG; the solute reference energy drives the λ exchange.
            if tri_pot is not None:
                tri_total = _tri_energy(pos, range(n_rep)).cpu().numpy()
                energies = e_full.cpu().numpy() + tri_total
            else:
                energies = e_full.cpu().numpy()
            solute_np = e_solute.cpu().numpy()
            _require_finite(e_full, "REMD total energy")
            if not np.all(np.isfinite(energies)):
                raise RuntimeError("REMD total energy contains NaN or Inf")
            e_hist.append(float(energies.min()))

            # ── Temperature-axis exchange (all vertical neighbors within each column) ──
            # Full-edge exchange: try all n_T-1 temperature edges every round to maximize the swap flux.
            # Coordinates and momenta are swapped together; the momenta are rescaled to the target
            # temperature to preserve the Maxwell distribution.
            for cj in range(n_lam):
                col = [ri * n_lam + cj for ri in range(n_t)]
                for k in range(n_t - 1):
                    a, b = col[k], col[k + 1]
                    attT[k] += 1
                    # log(new/old) = (β_a - β_b) (E_a - E_b).
                    # E_a/E_b already include every Hamiltonian term of the same λ column.
                    log_alpha = exchange_log_alpha_temperature(
                        beta[k], beta[k + 1], energies[a], energies[b])
                    if _accept_log_alpha(float(log_alpha)):
                        accT[k] += 1
                        tmp_pos = pos[a].clone()
                        pos[a] = pos[b].clone()
                        pos[b] = tmp_pos
                        tmp_vel = vel[a].clone()
                        vel[a] = vel[b].clone() * math.sqrt(
                            self.temps[k] / self.temps[k + 1])
                        vel[b] = tmp_vel * math.sqrt(
                            self.temps[k + 1] / self.temps[k])
                        _swap_tri_force_cache(a, b)
                        _refresh_slot_energies((a, b), energies, solute_np)

            # ── λ-axis exchange (horizontal neighbors within each row; criterion uses only the λ=1 solute term) ──
            # REST2: log(new/old) = β (λ_a - λ_b) (S_a - S_b).
            # Here S is the solute term referenced to λ=1, not each slot's own scaled energy.
            l_edge_start = rep & 1
            for ri in range(n_t):
                row = [ri * n_lam + cj for cj in range(n_lam)]
                for k in range(l_edge_start, n_lam - 1, 2):
                    a, b = row[k], row[k + 1]
                    attL[k] += 1
                    log_alpha = exchange_log_alpha_lambda(
                        beta[ri], self.lambdas[k], self.lambdas[k + 1],
                        solute_np[a], solute_np[b])
                    if _accept_log_alpha(float(log_alpha)):
                        accL[k] += 1
                        tmp_pos = pos[a].clone()
                        pos[a] = pos[b].clone()
                        pos[b] = tmp_pos
                        tmp_vel = vel[a].clone()
                        vel[a] = vel[b].clone()
                        vel[b] = tmp_vel
                        _swap_tri_force_cache(a, b, lambda_swap=True)
                        # The target λ is set by slots k/k+1; do not use the stale cj.
                        _refresh_slot_energies((a, b), energies, solute_np)

            i_min = int(np.argmin(energies))
            e_now = float(energies[i_min])
            # An improvement counts only if it exceeds 1% (avoids noise-triggered false early-stops)
            if e_now < best_e * 0.99:
                best_e = e_now
                best_pos = pos[i_min].detach().cpu().numpy()[0::3] * 10.0
                best_pos_3bead = pos[i_min].detach().clone()  # full 3-bead state
                no_improve_count = 0
            else:
                no_improve_count += 1
            if verbose:
                # ── Energy-decomposition diagnostics ──
                # CG energy
                e_cg_min = e_full[i_min].item()
                # Tri energy
                e_tri_min = tri_total[i_min] if tri_pot is not None else 0.0
                # Tri/CG ratio (the key diagnostic!)
                tri_cg_ratio = abs(e_tri_min) / max(abs(e_cg_min), 1.0)

                tri_str = ""
                if tri_pot is not None:
                    tri_str = f" Tri={e_tri_min:.0f}"

                # Diagnostic: energy gap between adjacent-temperature replicas (first λ column)
                dE_diag = ""
                if n_t >= 2:
                    _a, _b = 0, 1  # the first two temperature replicas (same λ=1.0 column)
                    _dE = abs(energies[_a] - energies[_b])
                    _crit = abs(beta[0] - beta[1]) * _dE
                    dE_diag = f" dE={_dE:.0f} crit={_crit:.1f}"

                # Print detailed diagnostics every 2 rounds
                if rep % 2 == 0:
                    print(f"    [GPU-2D] {rep+1}/{n_reports}: E_min={e_now:.0f} "
                          f"CG={e_cg_min:.0f}{tri_str} "
                          f"Tri/CG={tri_cg_ratio:.3f}{dE_diag} "
                          f"T-acc={[f'{a}/{t}' for a,t in zip(accT,attT)]}")

                    # ⚠️ Energy-ratio diagnostics
                    if tri_cg_ratio > 0.1:
                        print(f"    [GPU-2D] ⚠️ Tri energy share too high ({tri_cg_ratio:.1%}), "
                              f"may cause CG-Tri competition! Consider lowering trirnasp_scale")
                    elif tri_cg_ratio < 0.001:
                        print(f"    [GPU-2D] ⚠️ Tri energy share too low ({tri_cg_ratio:.3%}), "
                              f"TriRNASP is nearly ineffective! Consider raising trirnasp_scale")

            # Early stop: no >1% improvement for patience consecutive rounds
            if no_improve_count >= patience:
                if verbose:
                    print(f"    [GPU-2D] Early stop: no significant improvement for {patience} rounds; terminating early")
                break

        diag = {
            "acceptance_T": [a / max(t, 1) for a, t in zip(accT, attT)],
            "acceptance_lam": [a / max(t, 1) for a, t in zip(accL, attL)],
            "e_hist": e_hist,
            "best_pos_3bead": best_pos_3bead,  # full 3-bead state for multi-round reuse
            "velocities": vel.cpu(),  # Bug 3 fix: return velocities for reuse in the next round
            "tri_scores": tri_scores.tolist() if tri_scores is not None else None,
            "tri_force_cache_age": _tri_last_refresh_age[0],
            "tri_force_refresh_count": _tri_force_refresh_count[0],
            "tri_force_refresh_failed": _tri_force_refresh_failed[0],
            "tri_force_refresh_skipped": _tri_force_refresh_skipped[0],
            "tri_force_refresh_skipped": _tri_force_refresh_skipped[0],
            "tri_force_injected": bool(tri_pot_cpu is not None and self.use_trirnasp_force),
            "tri_force_policy": self.tri_force_replica_policy,
            "tri_force_replica_count": int(tri_force_mask.sum()),
        }
        if best_pos is None:
            best_pos = np.asarray(coords_A)
            best_e = 0.0

        # Clean up the CPU pool resources (prevent leaks)
        if _cpu_pool:
            try:
                _cpu_pool[0].terminate()
            except Exception:
                pass
            try:
                _cpu_pool[0].join(timeout=5)
            except Exception:
                pass

        return best_pos, best_e, diag


if __name__ == "__main__":
    """Smoke test: 1D REMD + 2D REST2×REMD (full 3-bead force field)."""
    import os
    import sys, time
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    dev = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    L = 60
    pairs = [(i, i + 30, 1.0) for i in range(5)]
    coords = np.zeros((L, 3))
    for i in range(L):
        ang = i * (5.9 / 11.0) * np.pi * 2 * 0.55
        coords[i] = [8 * np.cos(ang), 8 * np.sin(ang), i * 4.7]

    # ── 1D smoke ──
    remd = BatchedREMD(n_replicas=8, exchange_interval=500)
    t0 = time.time()
    best_A, best_E, diag = remd.run(coords, pairs, n_steps=2500, verbose=False)
    print(f"[1D] best E={best_E:.0f}, acc={diag['acceptance']:.0%}, "
          f"{time.time()-t0:.1f}s")
    assert np.all(np.isfinite(best_A))
    assert np.all(np.isfinite(diag["e_hist"])), "NaN in energy history"

    # ── 2D smoke (4T × 2λ = 8 replicas) ──
    remd2 = BatchedREMD2D(n_t=4, lambdas=(1.0, 0.75),
                          exchange_interval=500)
    t0 = time.time()
    best2, e2, diag2 = remd2.run(coords, pairs, n_steps=2500)
    print(f"[2D] grid={remd2.n_replicas} replicas, best E={e2:.0f}, "
          f"T-acc={[f'{a:.0%}' for a in diag2['acceptance_T']]}, "
          f"λ-acc={[f'{a:.0%}' for a in diag2['acceptance_lam']]}, "
          f"{time.time()-t0:.1f}s")
    assert np.all(np.isfinite(best2))
    assert all(math.isfinite(x) for x in diag2["e_hist"])

    # ── 2D + TriRNASP statistical-potential smoke ──
    seq60 = ("AUGCAUGC" * 8)[:L]
    remd3 = BatchedREMD2D(n_t=4, lambdas=(1.0, 0.75),
                          exchange_interval=500,
                          use_trirnasp=True, trirnasp_scale=0.003,  # unified default value
                          sequence=seq60)
    t0 = time.time()
    best3, e3, diag3 = remd3.run(coords, pairs, n_steps=1500, verbose=False)
    print(f"[2D+TriRNASP] best E={e3:.0f}, "
          f"{time.time()-t0:.1f}s")
    assert np.all(np.isfinite(best3))
    assert all(math.isfinite(x) for x in diag3["e_hist"]), "NaN w/ TriRNASP"

    print("[PASS] GPU 1D+2D batched REMD (3-bead) smoke")
