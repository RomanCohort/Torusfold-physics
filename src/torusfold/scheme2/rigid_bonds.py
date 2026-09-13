"""
rigid_bonds.py — SHAKE/RATTLE distance constraints for the batched Langevin integrator.

Why this exists
---------------
Two of the three intra-residue distances in the 3-bead model have no calibratable spring
constant. Measured 2026-09-13 (memory/spring-criterion-resolution-confounded.md):
K_INTRA_PC = 20752.7 and K_INTRA_CN = 36399.2 correspond to sigma = 0.13 A and 0.08 A, which sit
BELOW the 0.1-0.3 A coordinate-error floor of the crystallographic database they were fitted on.
They measure how tightly refinement restrained those bonds, not physics, and no dataset can
calibrate them better -- the noise is larger than the signal.

A constraint needs no constant. Holding |x_i - x_j| at the deposited mean removes the width from
the model entirely, which is the honest description of a bond whose thermal width is
unresolvable at the resolution of any structure we have.

This module is deliberately free of any field knowledge: it takes bead index pairs and target
lengths and knows nothing about P, C4', N9/N1, or torch_cgsim. torch_cgsim imports THIS, not the
other way round, so the solver can be tested against hand-built geometries without a force field
and cannot drift when the field is retuned.

Algorithm
---------
Positions: M-SHAKE, i.e. Newton on the Lagrange multipliers of a whole constraint component at
once rather than Gauss-Seidel over one constraint at a time. For a component with n_c
constraints the system is

    g_c(x) = |x_i - x_j| - sigma_c
    G = dg/dx,   J = G M^-1 G^T
    solve  J lambda = g,   x <- x - M^-1 G^T lambda

and the step is repeated to convergence. The textbook sequential form loops in Python over every
constraint -- 2L = 4026 iterations per step at L = 2013, on a hot path -- while the dense form
solves every component in one batched call.

Velocities: RATTLE, which for the linear constraint Gv = 0 is EXACT IN ONE SOLVE. That is not an
approximation that happens to be good; the constraint is linear in v, so the Newton step lands
on the constraint manifold and the residual afterwards is roundoff.
tests/test_bond_constraints.py asserts that at machine precision rather than "to a tolerance",
because the distinction is what makes one call per step sufficient.

Constraints are partitioned into components by "shares a bead": P-C4' and C4'-N of one residue
land in the same component because they share C4'. J is assembled by scattering the unit vectors
onto the component's bead slots, never by writing out the 2x2 by hand, so a future constraint
that spans two residues needs no edit here.

The assembly is the load-bearing part of the SHAKE step and the easiest thing to get silently
wrong: a hard-coded 2x2 that omits the shared-bead coupling still converges to SOMETHING (it is
then Jacobi iteration, not Newton), and the lengths it produces are close enough that only a
convergence-rate check notices. test_the_assembled_jacobian_matches_a_dense_difference compares
against a finite-difference G for exactly that reason.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


# ── dtype-aware tolerances ─────────────────────────────────────────────────────────────────
# The production path is float32 and a dtype-blind tolerance raises on every step. Positions in
# this codebase are O(1-10) nm, so float32 carries ~1e-6 nm of absolute rounding on a difference
# of two of them; relative to a 0.39 nm bond that is ~5e-6. float64 carries ~1e-15, i.e. nine
# decades lower, so a fixed 1e-9 would be above the achievable floor for float32 AND uselessly
# loose for float64. Hence a RELATIVE criterion -- the residual is compared against sigma_c --
# with one default per dtype, each roughly two decades above that dtype's own floor.
_TOL_REL: Dict[torch.dtype, float] = {
    torch.float32: 1e-4,
    torch.float64: 1e-9,
}
_MAX_ITER = 60

# A distance of exactly zero has no unit vector. It is a singular configuration, not a physical
# one (two beads cannot occupy the same nm-scale point without the pair term exploding), so the
# clamp only has to keep the arithmetic finite long enough for the convergence check to report
# the real problem.
_TINY = 1e-12

# det(J) below this fraction of |ad| + |bc| counts as a dependent constraint pair. The intra-
# residue motif sits at (4 - (u0.u1)^2)/(4 + (u0.u1)^2) >= 3/5 = 0.6, i.e. twelve orders of
# magnitude clear of this. The cut is set for the general n_c = 2 case, not for the one shipped.
_SINGULAR_FRAC = 1e-9


class DistanceConstraints:
    """A rigid distance network, solved by batched M-SHAKE (positions) and RATTLE (velocities).

    pairs    : (C, 2) integer tensor of BEAD indices, not residue indices. For the 3-bead model
               a residue's P-C4' and C4'-N constraints are beads (3r, 3r+1) and (3r+1, 3r+2).
    targets  : (C,) target lengths sigma_c, same order, same units as the positions.
    mass_amu : the uniform bead mass the integrator uses. J = G M^-1 G^T needs it, and passing a
               mass that disagrees with batch_langevin_step's makes the SHAKE correction use a
               slightly wrong metric -- which converges to the wrong length at O(1) accuracy
               rather than failing loudly. Keep the two in step.

    ridge    : optional Tikhonov term added to J as a multiple of its own mean diagonal. Zero by
               default. It exists for constraint sets where a component CAN become dependent --
               n_c >= 3 with a degenerate arrangement, or two constraints between the same bead
               pair. The shipped intra-residue network cannot reach one (see _solve: its 2x2 has
               det >= 3/m^2 for every geometry), so for this model ridge is untested insurance
               rather than a knob anyone should turn.
    """

    def __init__(
        self,
        pairs,
        targets,
        mass_amu: float = 110.0,
        tol_rel: Optional[float] = None,
        max_iter: int = _MAX_ITER,
        ridge: float = 0.0,
        name: str = "constraints",
    ) -> None:
        if not torch.is_tensor(pairs):
            pairs = torch.as_tensor(pairs, dtype=torch.long)
        pairs = pairs.to(dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must be a (C, 2) tensor of bead indices, "
                             f"got {tuple(pairs.shape)}")
        if pairs.shape[0] == 0:
            raise ValueError("a DistanceConstraints with no constraints is a no-op; pass None to "
                             "batch_langevin_step instead, so the unconstrained path stays the "
                             "one with no extra ops in it")
        if bool((pairs < 0).any()):
            raise ValueError("pairs contains a negative bead index")
        if bool((pairs[:, 0] == pairs[:, 1]).any()):
            raise ValueError("a constraint cannot join a bead to itself; that is a degenerate "
                             "zero-length constraint, not a typo worth guessing at")

        tgt = torch.as_tensor(targets, dtype=torch.float64).reshape(-1)
        if tgt.numel() != pairs.shape[0]:
            raise ValueError(f"targets has {tgt.numel()} entries for {pairs.shape[0]} "
                             f"constraints")
        if not bool(torch.isfinite(tgt).all()):
            raise ValueError("targets contains NaN or Inf")
        if bool((tgt <= 0).any()):
            raise ValueError("every target length must be > 0 nm")
        if mass_amu <= 0:
            raise ValueError(f"mass_amu must be > 0, got {mass_amu}")
        if tol_rel is not None and not (0.0 < tol_rel < 1.0):
            raise ValueError(f"tol_rel must be in (0, 1), got {tol_rel}")

        self.pairs = pairs.cpu()
        self.targets = tgt
        self.mass_amu = float(mass_amu)
        self.tol_rel = tol_rel
        self.max_iter = int(max_iter)
        self.ridge = float(ridge)
        self.name = str(name)

        (bead_idx, loc_i, loc_j, sigma,
         comp_of_con, slot_of_con) = self._components(pairs, tgt)
        self.bead_idx = bead_idx          # (K, n_b) global bead indices
        self.loc_i = loc_i                # (K, n_c) local slot of the first endpoint
        self.loc_j = loc_j                # (K, n_c) local slot of the second endpoint
        self.sigma = sigma                # (K, n_c) float64
        self._comp_of_con = comp_of_con   # (C,) which component a constraint went into
        self._slot_of_con = slot_of_con   # (C,) its slot inside that component
        self.n_components = int(bead_idx.shape[0])
        self.n_constraints = int(pairs.shape[0])
        self.n_local_beads = int(bead_idx.shape[1])
        self._per_device: Dict[Tuple[str, str], Tuple[torch.Tensor, ...]] = {}
        self.last_shake_iters = 0

    # ── construction ───────────────────────────────────────────────────────────────────────
    @staticmethod
    def _components(pairs: torch.Tensor, tgt: torch.Tensor):
        """Union-find the bead graph, then index each component for a batched dense solve.

        Every component must have the same (n_b, n_c) for the batched tensor shape to exist.
        That holds for the intra-residue network (L components, 3 beads and 2 constraints each)
        and is asserted rather than assumed: a ragged set would need padding, and padding with
        fake beads is how a constraint solver comes to depend on a geometry that is not there.
        """
        n_beads = int(pairs.max()) + 1
        parent = list(range(n_beads))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for c in range(pairs.shape[0]):
            ra, rb = find(int(pairs[c, 0])), find(int(pairs[c, 1]))
            if ra != rb:
                parent[rb] = ra

        groups: Dict[int, List[int]] = {}
        for c in range(pairs.shape[0]):
            groups.setdefault(find(int(pairs[c, 0])), []).append(c)

        shapes = set()
        for cons in groups.values():
            beads = {int(b) for c in cons for b in (pairs[c, 0], pairs[c, 1])}
            shapes.add((len(beads), len(cons)))
        if len(shapes) != 1:
            raise NotImplementedError(
                f"constraint components are not all the same shape: {sorted(shapes)}. This "
                f"solver batches one dense (n_c, n_c) system per component and needs a uniform "
                f"shape; a ragged set would have to be padded, and padded beads are constraints "
                f"on a geometry that does not exist.")

        n_b, n_c = shapes.pop()
        K = len(groups)
        bead_idx = torch.zeros((K, n_b), dtype=torch.long)
        loc_i = torch.zeros((K, n_c), dtype=torch.long)
        loc_j = torch.zeros((K, n_c), dtype=torch.long)
        sigma = torch.zeros((K, n_c), dtype=torch.float64)
        comp_of_con = torch.zeros(pairs.shape[0], dtype=torch.long)
        slot_of_con = torch.zeros(pairs.shape[0], dtype=torch.long)

        # Sorted by root so the component order is a deterministic function of the input, not of
        # dict insertion -- otherwise two constructions of the same graph could pair a component
        # with a different slice of the batch and nothing would ever raise.
        for k, root in enumerate(sorted(groups)):
            cons = sorted(groups[root])
            beads = sorted({int(b) for c in cons for b in (pairs[c, 0], pairs[c, 1])})
            slot = {b: s for s, b in enumerate(beads)}
            bead_idx[k] = torch.tensor(beads, dtype=torch.long)
            for m, c in enumerate(cons):
                loc_i[k, m] = slot[int(pairs[c, 0])]
                loc_j[k, m] = slot[int(pairs[c, 1])]
                sigma[k, m] = tgt[c]
                comp_of_con[c] = k
                slot_of_con[c] = m
        return bead_idx, loc_i, loc_j, sigma, comp_of_con, slot_of_con

    def _idx(self, device, dtype):
        """Index tensors and sigma on the batch's device and dtype. Cached per (device, dtype)."""
        key = (str(device), str(dtype))
        got = self._per_device.get(key)
        if got is None:
            got = (self.bead_idx.to(device), self.loc_i.to(device), self.loc_j.to(device),
                   self.sigma.to(device=device, dtype=dtype))
            self._per_device[key] = got
        return got

    # ── the equations ──────────────────────────────────────────────────────────────────────
    def _local(self, pos: torch.Tensor, bead_idx: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) positions -> (B, K, n_b, 3) component-local beads.

        Flattened to a 1-D index on purpose. `pos[:, bead_idx]` with a (K, n_b) index looks like
        it should work and does not: the advanced index sits at dim 2 while dims 0 and 1 are
        slices, so it broadcasts AGAINST them and returns (B, K, K, n_b, 3). Nothing raises at
        the indexing -- the failure surfaces much later as a shape mismatch, or not at all if
        the downstream reduce happens to be tolerant. One flat index into dim 1 has no such
        ambiguity.
        """
        B = pos.shape[0]
        return pos[:, bead_idx.reshape(-1)].reshape(
            B, self.n_components, self.n_local_beads, 3)

    @staticmethod
    def _pick(x: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
        """(B, K, n_b, 3) and (K, n_c) local slots -> (B, K, n_c, 3).

        gather rather than `x[:, :, loc]`, for the reason in _local.
        """
        idx = loc.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], loc.shape[0], loc.shape[1], 3)
        return torch.gather(x, 2, idx)

    def _geometry(self, x_loc, loc_i, loc_j, sigma):
        """Component-local positions -> (r, u, G, g).

        sigma is passed in rather than read off self: it is the DEVICE- AND DTYPE-RESOLVED
        copy (see _idx), and subtracting the float64 attribute from a float32 r promotes the
        residual to float64 -- which then fails to solve against a float32 J, with an error
        message about dtypes that says nothing about the real cause.
        """
        d = self._pick(x_loc, loc_i) - self._pick(x_loc, loc_j)        # (B, K, n_c, 3)
        r = torch.linalg.vector_norm(d, dim=-1).clamp_min(_TINY)       # (B, K, n_c)
        u = d / r.unsqueeze(-1)
        G = self._assembled_G(u, loc_i, loc_j)
        return r, u, G, r - sigma

    def _assembled_G(self, u: torch.Tensor, loc_i: torch.Tensor, loc_j: torch.Tensor):
        """(B, K, n_c, 3) unit vectors -> (B, K, n_c, n_b*3) constraint Jacobian rows.

        Assembled by scattering, so the shared-bead coupling is whatever the topology says it is.
        Writing the 2x2 out by hand for the intra-residue case would be correct today and would
        silently degrade to Jacobi iteration -- which still converges, just not quadratically --
        the day a constraint spans two residues.
        """
        B, K, n_c, _ = u.shape
        cols = torch.arange(3, device=u.device, dtype=torch.long)
        G = torch.zeros(B, K, n_c, self.n_local_beads * 3, dtype=u.dtype, device=u.device)
        for sign, loc in ((1.0, loc_i), (-1.0, loc_j)):
            idx = (loc[..., None] * 3 + cols).unsqueeze(0).expand(B, K, n_c, 3)
            G.scatter_add_(3, idx, sign * u)
        return G

    def _solve(self, J: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Solve J lambda = rhs for every component at once.

        WHY THE 2x2 IS INVERTED IN CLOSED FORM. It is 6x faster: measured at B=8, K=2013 on this
        machine, torch.linalg.solve on the (8, 2013, 2, 2) batch costs 853 us and the explicit
        determinant formula 137 us, against a whole unconstrained step of ~1500 us. CPU batched
        LU has a per-call cost that a 2x2 cannot amortise.

        This is NOT the "write the 2x2 out by hand" the module docstring warns against. J is
        assembled from the topology as always; this only inverts whatever came out, reading
        a, b, c, d OFF that matrix. Change the topology and J changes with it. Change n_c and the
        fast path stops firing -- guarded below -- and the general solve takes over.

        It also buys a real singularity test that torch.linalg.solve does not have. A 2x2 LU
        raises only on an exactly-zero pivot and otherwise returns garbage for a near-singular
        matrix without a word; the determinant is right there to compare against the term
        magnitudes. For this model's motif that test never fires, and the reason is worth
        stating: with u0 and u1 the two unit bond vectors, J = (1/m)[[2, -u0.u1],[-u0.u1, 2]] so
        det = (4 - (u0.u1)^2)/m^2 >= 3/m^2. It is uniformly well conditioned, and a collinear
        P-C4'-N cannot make it singular however far the chain is stretched.
        """
        n_c = J.shape[-1]
        if self.ridge > 0.0:
            diag = J.diagonal(dim1=-2, dim2=-1)
            scale = diag.mean(dim=-1, keepdim=True).clamp_min(_TINY)
            eye = torch.eye(n_c, dtype=J.dtype, device=J.device)
            J = J + (self.ridge * scale).unsqueeze(-1) * eye

        if n_c == 2:
            a, b = J[..., 0, 0], J[..., 0, 1]
            c, d = J[..., 1, 0], J[..., 1, 1]
            det = a * d - b * c
            scale = (a * d).abs() + (b * c).abs()
            bad = (det.abs() <= _SINGULAR_FRAC * scale) | ~torch.isfinite(det)
            if bool(bad.any()):
                k, e = divmod(int(bad.reshape(-1).nonzero()[0]), J.shape[1])
                raise RuntimeError(
                    f"[{self.name}] singular 2x2 constraint system at batch {k}, component {e}: "
                    f"det = {float(det[k, e]):.3e} against a scale of {float(scale[k, e]):.3e}. "
                    f"Two constraints in this component have become linearly dependent. "
                    f"ridge=<small> trades an approximate projection for a defined answer; it "
                    f"does not make the geometry well posed.")
            return torch.stack([(d * rhs[..., 0] - b * rhs[..., 1]) / det,
                                (a * rhs[..., 1] - c * rhs[..., 0]) / det], dim=-1)

        try:
            lam = torch.linalg.solve(J, rhs.unsqueeze(-1)).squeeze(-1)
        except Exception as exc:                                       # noqa: BLE001
            raise RuntimeError(
                f"[{self.name}] the {n_c}x{n_c} Lagrange-multiplier system did not solve. "
                f"Original error: {exc}") from None
        if not bool(torch.isfinite(lam).all()):
            raise RuntimeError(
                f"[{self.name}] non-finite Lagrange multipliers from torch.linalg.solve; the "
                f"constraint Jacobian is singular on at least one component")
        return lam

    # ── public API ─────────────────────────────────────────────────────────────────────────
    def residual(self, pos: torch.Tensor) -> torch.Tensor:
        """(B, C) signed g_c = r_c - sigma_c, in the ORDER THE CONSTRAINTS WERE SUPPLIED.

        The solver's internal layout groups constraints by component (that is what makes the
        batched solve possible), so a residual read off the internal tensors would come back in a
        different order than the caller's `targets`. Scattering back through the recorded
        (component, slot) of each constraint is what keeps "constraint 7" meaning the same thing
        to the caller and to the solver.
        """
        bead_idx, loc_i, loc_j, sigma = self._idx(pos.device, pos.dtype)
        _r, _u, _G, g = self._geometry(self._local(pos, bead_idx), loc_i, loc_j, sigma)
        # flat_k and flat_m are adjacent 1-D advanced indices at dims 1 and 2 of g, so together
        # they broadcast to (C,) and land at position 1 -> (B, C). Unlike the `x[:, :, loc]`
        # case above, there are no intervening slices for them to broadcast against.
        flat_k = self._comp_of_con.to(pos.device)
        flat_m = self._slot_of_con.to(pos.device)
        return g[:, flat_k, flat_m]

    def shake(self, pos: torch.Tensor) -> torch.Tensor:
        """Project positions onto the constraint manifold. Returns a NEW tensor.

        Newton on the multipliers of a whole component at once (M-SHAKE), iterated to a RELATIVE
        tolerance: |g_c| <= tol_rel * sigma_c for every constraint. Raises on non-convergence
        with the worst residual and the iteration count -- a silently capped iteration is a bond
        that drifts, and drift over 1e5 steps is a different molecule.
        """
        bead_idx, loc_i, loc_j, sigma = self._idx(pos.device, pos.dtype)
        tol_rel = (self.tol_rel if self.tol_rel is not None
                   else _TOL_REL.get(pos.dtype, _TOL_REL[torch.float64]))

        x = pos[:, bead_idx]                                    # (B, K, n_b, 3)
        inv_m = 1.0 / self.mass_amu
        worst, iters = None, 0
        for it in range(self.max_iter):
            _r, _u, G, g = self._geometry(x, loc_i, loc_j, sigma)
            # detach: torch_cgsim's minimiser path calls shake on a requires_grad tensor, and the
            # convergence test is a diagnostic, not part of the expression anything differentiates.
            worst = (g.abs() / sigma).amax().detach()
            iters = it
            if float(worst) <= tol_rel:
                break
            J = (G @ G.transpose(-1, -2)) * inv_m
            lam = self._solve(J, g)
            dx = -inv_m * torch.einsum("bkcd,bkc->bkd", G, lam)
            x = x + dx.view(x.shape)
        else:
            raise RuntimeError(
                f"[{self.name}] SHAKE did not converge in {self.max_iter} iterations; worst "
                f"relative residual {float(worst):.3e} against a tolerance of {tol_rel:.1e} "
                f"(dtype {pos.dtype}). Raising the iteration cap will not fix a system that is "
                f"not converging -- check that the target lengths are reachable and that "
                f"mass_amu matches the integrator's.")

        self.last_shake_iters = iters + 1
        out = pos.clone()
        out[:, bead_idx.reshape(-1)] = x.reshape(pos.shape[0], -1, 3)
        _require_finite(out, "SHAKE-corrected coordinates", self.name)
        return out

    def rattle(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """Project velocities onto Gv = 0 at the supplied (already SHAKE-corrected) positions.

        ONE solve, and that is exact rather than adequate: the velocity constraint is linear in
        v, so Newton's first step lands on the manifold and what is left afterwards is roundoff.
        Solving at the pre-SHAKE positions instead would leave a residual proportional to the
        position correction, i.e. the scheme would be a slightly wrong projection every step.

        The returned velocity satisfies d/dt|x_i - x_j| = 0 at the returned positions. The next
        step's opening B half-kick perturbs it again, which is why the position projection is
        what actually holds the bond -- RATTLE's job is to stop the thermostat pumping energy
        into the constrained directions, where the potential cannot take it back out.
        """
        bead_idx, loc_i, loc_j, sigma = self._idx(pos.device, pos.dtype)
        x = self._local(pos, bead_idx)
        v = self._local(vel, bead_idx)
        _r, _u, G, _g = self._geometry(x, loc_i, loc_j, sigma)
        J = (G @ G.transpose(-1, -2)) * (1.0 / self.mass_amu)
        # G's trailing axis is the FLATTENED bead axis (n_b*3), so v has to be flattened the same
        # way: (B, K, n_b, 3) -> (B, K, n_b*3). Reshaping to (B, K*n_b, 3) looks equivalent and
        # is not -- it interleaves the two axes differently.
        gdot = torch.einsum("bkcd,bkd->bkc", G, v.reshape(v.shape[0], v.shape[1], -1))
        mu = self._solve(J, gdot)
        dv = -(1.0 / self.mass_amu) * torch.einsum("bkcd,bkc->bkd", G, mu)

        out = vel.clone()
        out[:, bead_idx.reshape(-1)] = (v + dv.view(v.shape)).reshape(pos.shape[0], -1, 3)
        _require_finite(out, "RATTLE-corrected velocities", self.name)
        return out

    def velocity_residual(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """Max |Gv| per batch element, (B,), after RATTLE. Roundoff, not a tolerance."""
        bead_idx, loc_i, loc_j, _sigma = self._idx(pos.device, pos.dtype)
        _r, _u, G, _g = self._geometry(self._local(pos, bead_idx), loc_i, loc_j, _sigma)
        v = self._local(vel, bead_idx)
        gdot = torch.einsum("bkcd,bkd->bkc", G, v.reshape(v.shape[0], v.shape[1], -1))
        return gdot.abs().amax(dim=(1, 2))

    def describe(self) -> str:
        """One provenance line. A constrained run and an unconstrained one print IDENTICAL
        constants, so this is the only thing in a log that says the model changed."""
        uniq = sorted({round(float(t), 6) for t in self.targets})
        body = ", ".join(f"{u:g}" for u in uniq[:4]) + ("..." if len(uniq) > 4 else "")
        return (f"{self.name}: {self.n_constraints} distance constraints in "
                f"{self.n_components} components, targets [{body}] nm, "
                f"mass {self.mass_amu:g} Da")


def _require_finite(value: torch.Tensor, what: str, who: str = "constraints") -> None:
    """Local rather than imported from torch_cgsim: the dependency runs one way only."""
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"[{who}] non-finite {what}")
