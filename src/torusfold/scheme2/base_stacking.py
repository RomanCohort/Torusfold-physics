"""A base-level stacking term, built from the measured target rather than from taste.

WHAT IT IS. For every consecutive pair (i, i+1), with the base-plane normals recovered from the beads by
the rigid template map (base_frames, one pooled triple, 6.6 deg off the template's own planes):

    E = -eps * SUM_i  Gd(d_i) * Gr(rise_i) * Gt(theta_i)

    Gd = exp(-((d - d0)/sd)^2)             d    = |N_{i+1} - N_i|
    Gr = exp(-((rise - r0)/sr)^2)          rise = (N_{i+1} - N_i) . n_mean
    Gt = exp(-(theta/theta_c)^2)           theta = angle(n_i, n_{i+1})  (sign-free by construction)

and every parameter is a measurement, not a choice. From 20 crystal fragments, taken through the SAME
bead-to-plane map the term uses (scripts/measure_base_coords.py --planes, 1625 consecutive pairs):

    d     0.570 +- 0.198 nm
    rise  0.328 +- 0.188 nm      (5th percentile +0.06 nm: the target is ONE-SIDED)
    theta 24.5 +- 22.2 deg       (median 15.9)

WHY A WELL IN THREE FACTORS. Each factor is the marginal the crystals actually show, so the term cannot be
"tuned" away from the target without moving a measured number. The rise well is the load-bearing one:
measured on the field's own sampler (findings Part 15) the sampled rise has a 5th percentile at -0.43 nm
against the target's +0.02, i.e. the model has no preference for one base lying OVER its neighbour rather
than beside or under it -- and that asymmetry IS stacking. The distance well supplies the attraction the
sampler does not have (sampled d 0.703 against 0.570 nm), and the theta factor is the orientation coupling
nothing in the field provides (sampled 49.4 against 24.5 deg on the same definition).

WHAT IT IS NOT. Not a fitted potential: eps is a strength (kJ/mol) to be scanned, and the three wells are
the target marginals. A fit that moves the wells away from those numbers would be fitting the model's own
coupling, which is the failure mode Part 8 documented for the trace coordinates.

Units: pos_nm is (B, 3L, 3) in nm and the coefficient triple is scaled for nm (base_frames.pooled_coef(10)).
Forces come from autograd, like the other injected potentials.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from .base_frames import pooled_coef

# The measured target, in nm and degrees (scripts/measure_base_coords.py --planes 20).
D_REF, D_SIG = 0.570, 0.198
RISE_REF, RISE_SIG = 0.328, 0.188
THETA_C = 25.0
THETA_C_BROAD = 60.0         # orientation scale for the "sum" form; a modelling choice, see below
EPS_DEFAULT = 5.0            # kJ/mol; kBT = 2.494 kJ/mol at 300 K


def base_normals(pos_nm: torch.Tensor, coef_nm: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B, 3L, 3) beads in nm -> (B, L, 3) unit base-plane normals, beads ordered (P, C4', N)."""
    B, N, _ = pos_nm.shape
    L = N // 3
    p = pos_nm[:, 0::3, :]
    c4 = pos_nm[:, 1::3, :]
    nn = pos_nm[:, 2::3, :]
    e1 = c4 - p
    e2 = nn - p
    e3 = torch.cross(e1, e2, dim=-1)
    if coef_nm is None:
        coef_nm = torch.tensor(pooled_coef(10.0), dtype=pos_nm.dtype, device=pos_nm.device)
    n = coef_nm[0] * e1 + coef_nm[1] * e2 + coef_nm[2] * e3
    nrm = torch.linalg.norm(n, dim=-1, keepdim=True).clamp_min(1e-12)
    return n / nrm


def stack_energy(pos_nm: torch.Tensor, eps: float = EPS_DEFAULT,
                 coef_nm: Optional[torch.Tensor] = None,
                 d_ref: float = D_REF, d_sig: float = D_SIG,
                 rise_ref: float = RISE_REF, rise_sig: float = RISE_SIG,
                 theta_c: float = THETA_C) -> torch.Tensor:
    """Per-batch stacking energy (B,) in kJ/mol, differentiable in pos_nm."""
    B, N, _ = pos_nm.shape
    L = N // 3
    if L < 2:
        return torch.zeros(B, dtype=pos_nm.dtype, device=pos_nm.device)
    n = base_normals(pos_nm, coef_nm)                      # (B, L, 3)
    nb = pos_nm[:, 2::3, :]                                # (B, L, 3) the N9/N1 bead
    ni, nj = n[:, :-1, :], n[:, 1:, :]
    nb_i, nb_j = nb[:, :-1, :], nb[:, 1:, :]
    # Sign alignment before the sum, so theta is geometry and not the map's arbitrary normal sign.
    flip = (ni * nj).sum(-1, keepdim=True) < 0
    nj = torch.where(flip, -nj, nj)
    nm = ni + nj
    nrm = torch.linalg.norm(nm, dim=-1, keepdim=True)
    nm = torch.where(nrm > 1e-9, nm / nrm.clamp_min(1e-12), ni)

    dc = nb_j - nb_i
    d = torch.linalg.norm(dc, dim=-1)
    rise = (dc * nm).sum(-1)
    cos_theta = (ni * nj).sum(-1).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_theta)                        # radians, 0..pi

    g_d = torch.exp(-((d - d_ref) / d_sig) ** 2)
    g_r = torch.exp(-((rise - rise_ref) / rise_sig) ** 2)
    g_t = torch.exp(-(theta / math.radians(theta_c)) ** 2)
    return -eps * (g_d * g_r * g_t).sum(dim=-1)


def stack_energy_penalty(pos_nm: torch.Tensor, eps: float = EPS_DEFAULT,
                         coef_nm: Optional[torch.Tensor] = None,
                         d_ref: float = D_REF, d_sig: float = D_SIG,
                         rise_ref: float = RISE_REF, rise_sig: float = RISE_SIG,
                         theta_c: float = THETA_C) -> torch.Tensor:
    """The same three measured wells, combined so that eps IS the strength: E = eps * (1 - Gd*Gr*Gt).

    WHY THE COMBINATION HAD TO CHANGE, measured. The reward form above multiplies three factors that are
    each <= 1, so its value collapses exactly where the system needs to be moved: at the sampled theta of
    51.4 degrees the orientation factor with a 25 degree scale is 0.016, and a nominal eps of 10 kJ/mol
    acted as 0.07 kJ/mol per pair -- against the ~6 kJ/mol a 20 degree base reorientation costs
    (scripts/measure_base_rotation_cost.py). A scan at eps = 4 and 10 moved the base-level marginals by
    10-20 percent of what was needed and left the rise one-sidedness untouched.

    One minus the same product is a BOUNDED PENALTY: it is ~1 (i.e. eps, the whole strength) where the
    geometry is unstacked, ~0 where it matches the target, and its gradient does not vanish anywhere in
    between -- at d = 0.75 nm it is 0.56 with a slope of 0.8 per nm, against 0.44 and 0.4 for the reward.
    The wells, their centres and their widths are unchanged: they are the measured target marginals, and
    this function exists only to give them a scale that means what it says.
    """
    B, N, _ = pos_nm.shape
    L = N // 3
    if L < 2:
        return torch.zeros(B, dtype=pos_nm.dtype, device=pos_nm.device)
    n = base_normals(pos_nm, coef_nm)
    nb = pos_nm[:, 2::3, :]
    ni, nj = n[:, :-1, :], n[:, 1:, :]
    nb_i, nb_j = nb[:, :-1, :], nb[:, 1:, :]
    flip = (ni * nj).sum(-1, keepdim=True) < 0
    nj = torch.where(flip, -nj, nj)
    nm = ni + nj
    nrm = torch.linalg.norm(nm, dim=-1, keepdim=True)
    nm = torch.where(nrm > 1e-9, nm / nrm.clamp_min(1e-12), ni)
    dc = nb_j - nb_i
    d = torch.linalg.norm(dc, dim=-1)
    rise = (dc * nm).sum(-1)
    theta = torch.arccos((ni * nj).sum(-1).clamp(-1.0, 1.0))
    f = (torch.exp(-((d - d_ref) / d_sig) ** 2)
         * torch.exp(-((rise - rise_ref) / rise_sig) ** 2)
         * torch.exp(-(theta / math.radians(theta_c)) ** 2))
    return eps * (1.0 - f).sum(dim=-1)


def stack_energy_sum(pos_nm: torch.Tensor, eps: float = EPS_DEFAULT,
                     coef_nm: Optional[torch.Tensor] = None,
                     d_ref: float = D_REF, d_sig: float = D_SIG,
                     rise_ref: float = RISE_REF, rise_sig: float = RISE_SIG,
                     theta_c: float = THETA_C_BROAD,
                     w_d: float = 1.0, w_r: float = 1.0, w_t: float = 1.0) -> torch.Tensor:
    """SEPARATE penalties, one per coordinate: E = eps * SUM_i [ (1-Gd) + (1-Gr) + (1-Gt) ].

    WHY THIS AND NOT "1 MINUS THE PRODUCT", measured the hard way. The previous attempt combined the same
    three factors as eps * (1 - Gd*Gr*Gt) and was expected to act at scale eps; it is ALGEBRAICALLY THE SAME
    FORCES as the reward form, because SUM(1 - f) = N - SUM(f) differs from -SUM(f) by a constant. Caught
    by the sampler: the two forms produced bit-identical trajectories (same bead coordinates to five
    decimals) while both differed from no term at all. Identical output is a measurement.

    What actually limits the product form is the GRADIENT, not the value: with a 25 degree orientation
    scale, f_theta at the sampled theta of 51 degrees is 0.016, and its slope is f*2*theta/theta_c^2, about
    0.0026 per degree -- so the pull is ~1.5 percent of what a 20 degree reorientation needs, no matter
    what eps is. A sum of independent penalties does not have that property: each coordinate's slope is
    full-strength wherever its own well is missed, even when the other two are satisfied.

    The theta scale is widened to THETA_C_BROAD = 60 degrees for the same reason: a factor that is 0.016 at
    the configurations that need repair cannot pull them in. The wells' centres stay at the measured target
    values; only the width of the orientation factor is a modelling choice, and it is recorded as one.
    """
    B, N, _ = pos_nm.shape
    L = N // 3
    if L < 2:
        return torch.zeros(B, dtype=pos_nm.dtype, device=pos_nm.device)
    n = base_normals(pos_nm, coef_nm)
    nb = pos_nm[:, 2::3, :]
    ni, nj = n[:, :-1, :], n[:, 1:, :]
    nb_i, nb_j = nb[:, :-1, :], nb[:, 1:, :]
    flip = (ni * nj).sum(-1, keepdim=True) < 0
    nj = torch.where(flip, -nj, nj)
    nm = ni + nj
    nrm = torch.linalg.norm(nm, dim=-1, keepdim=True)
    nm = torch.where(nrm > 1e-9, nm / nrm.clamp_min(1e-12), ni)
    dc = nb_j - nb_i
    d = torch.linalg.norm(dc, dim=-1)
    rise = (dc * nm).sum(-1)
    theta = torch.arccos((ni * nj).sum(-1).clamp(-1.0, 1.0))
    g_d = torch.exp(-((d - d_ref) / d_sig) ** 2)
    g_r = torch.exp(-((rise - rise_ref) / rise_sig) ** 2)
    g_t = torch.exp(-(theta / math.radians(theta_c)) ** 2)
    # PER-COORDINATE WEIGHTS, added because the equal-weight scan showed they should not be equal: at
    # eps = 20 the distance well got the MEAN right (0.750 -> 0.621 against 0.570) while squeezing the WIDTH
    # to sd 0.122 against the target's 0.198, so the distance's total variation got worse while the other
    # two coordinates' improved. The rise and the orientation are the ones the model cannot express at all
    # today (no preference for over vs under, no coupling at all); the base-base distance at least has the
    # pair and link network pulling on it. So the distance penalty is the one to discount.
    return eps * (w_d * (1.0 - g_d) + w_r * (1.0 - g_r) + w_t * (1.0 - g_t)).sum(dim=-1)


def make_base_stack_potential(eps: float = EPS_DEFAULT, coef_nm: Optional[np.ndarray] = None,
                              form: str = "reward", **kw) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """A cg_energy_forces injection: pos_nm -> (energy (B,), forces (B, 3L, 3)).

    Autograd, like the other injected potentials: the term is O(L) with a handful of elementwise ops, so
    the backward pass is cheap, and writing the analytic gradient by hand would be a second place for the
    term to be wrong.

    form="reward" is the original eps * Gd * Gr * Gt; form="penalty" is eps * (1 - Gd * Gr * Gt), which is
    the same three wells with eps as a scale that means what it says. Both are kept: the first one's failure
    is a measurement, not a mistake to be erased.
    """
    coef = None if coef_nm is None else torch.tensor(np.asarray(coef_nm, dtype=np.float64))
    _energy = {"reward": stack_energy, "penalty": stack_energy_penalty,
               "sum": stack_energy_sum}[form]

    def _pot(pos_nm: torch.Tensor):
        with torch.enable_grad():
            p = pos_nm.detach().clone().requires_grad_(True)
            e = _energy(p, eps=eps, coef_nm=coef, **kw)
            g, = torch.autograd.grad(e.sum(), p, create_graph=False)
        return e.detach(), (-g).detach()

    return _pot


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 2, 8
    pos = torch.randn(B, 3 * L, 3, dtype=torch.float64) * 0.2
    pos[:, 0::3, :] = torch.arange(L, dtype=torch.float64).reshape(1, L, 1) * torch.tensor([0.55, 0.0, 0.0])
    pot = make_base_stack_potential(eps=5.0)
    e, f = pot(pos)
    print("E:", [round(float(x), 4) for x in e], " max|F|:", float(f.abs().max()))
    # finite-difference check on one coordinate
    i = 0
    k = (0, 1, 2)
    d = 1e-6
    p2 = pos.clone(); p2[k] += d
    p1 = pos.clone(); p1[k] -= d
    num = float((stack_energy(p2).sum() - stack_energy(p1).sum()) / (2 * d))
    print("dE/dx analytic %.6f   finite difference %.6f" % (float(-f[k]), num))
