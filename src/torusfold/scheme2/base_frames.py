"""Base-plane normals as a fixed function of the three CG beads.

WHY THIS FILE EXISTS. A CG potential can only be a function of the beads. The model carries three per
residue -- P, C4', and N9 (purine) or N1 (pyrimidine) -- and the base PLANE is not determined by them:
it takes the rigid 1EHZ template to say where the ring sits relative to those three points. Measured on 20
crystal fragments (findings Part 15), the raw triangle normal P-C4'-N is 20.2 degrees (median 16.2) off the
true base-plane normal, which is why a term written on the triangle normal carries a systematic.

THE MAP IS LINEAR AND EXACT. The template is rigid, so for a residue the base-plane normal is a fixed
linear combination of the three template vectors

    e1 = C4' - P,   e2 = N - P,   e3 = e1 x e2

-- a basis, since the triangle is non-degenerate -- and those coefficients are a property of the template
alone. At run time the same combination is evaluated on the SAMPLED beads, which is exactly the rotation
the three-point Kabsch in aform_from_template applies to the whole residue: no per-step reconstruction, no
per-step SVD, a few cross products and dot products per residue, and differentiable like every other term
in the field.

WHAT IS APPROXIMATED, and it is measured rather than hidden: the sampler does not carry base identity
(load_structures returns name, pairs and pos -- no sequence), so the coefficients here can be either
per-base or pooled. __main__ reports the plane-normal error each choice costs on the template itself.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .aform_from_template import _load_templates

# Ring atoms whose plane IS the base plane: the purine bicycle (9 atoms) and the pyrimidine ring (6).
RING_ATOMS = {
    "A": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"),
    "G": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"),
    "C": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "U": ("N1", "C2", "N3", "C4", "C5", "C6"),
}
GLYCOLYSIS = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}

_coeffs: Dict[str, Tuple[np.ndarray, float]] = {}


def _plane_normal(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - c)
    n = vt[2]
    return n / np.linalg.norm(n)


def coeffs(pooled: bool = False) -> Dict[str, Tuple[np.ndarray, float]]:
    """Per-base (alpha, beta, gamma) and the template's own residual, or one pooled triple.

    The returned (coef, resid_deg): coef maps (e1, e2, e3) to the plane normal, resid_deg is the angle
    between that linear image and the template's own ring-atom plane normal, i.e. the exactness of the map.
    """
    key = "pooled" if pooled else "per-base"
    if key in _coeffs:
        return _coeffs[key]
    tmpl = _load_templates()
    out: Dict[str, Tuple[np.ndarray, float]] = {}
    allc = []
    for base in "AUGC":
        names = tmpl[base]["names"]
        xyz = np.asarray(tmpl[base]["coords"], dtype=np.float64)
        idx = {n: i for i, n in enumerate(names)}
        P = xyz[idx["P"]]
        C4 = xyz[idx["C4'"]]
        N = xyz[idx[GLYCOLYSIS[base]]]
        ring = xyz[[idx[n] for n in RING_ATOMS[base]]]
        n_true = _plane_normal(ring)
        e1, e2 = C4 - P, N - P
        e3 = np.cross(e1, e2)
        M = np.stack([e1, e2, e3], axis=1)              # columns are the basis vectors
        coef = np.linalg.solve(M, n_true)
        # SIGN CONVENTION. An SVD plane normal has an arbitrary sign, so the raw solve gives G and C
        # coefficients of the opposite handedness to A and U (measured: the pooled triple then misses the
        # plane by 77.5 degrees). Fix it once, here: the normal points along +e3 = (C4'-P) x (N-P). Every
        # downstream angle (theta, twist, rise) is then sign-free, and the pooled row below becomes
        # meaningful instead of meaningless.
        if coef[2] < 0.0:
            coef = -coef
        n_lin = M @ coef
        n_lin = n_lin / np.linalg.norm(n_lin)
        resid = float(np.degrees(np.arccos(max(-1.0, min(1.0, abs(float(n_lin @ n_true)))))))
        out[base] = (coef / np.linalg.norm(coef), resid)
        allc.append(coef / np.linalg.norm(coef))
    if pooled:
        c = np.mean(allc, axis=0)
        c = c / np.linalg.norm(c)
        res = []
        for base in "AUGC":
            names = tmpl[base]["names"]
            xyz = np.asarray(tmpl[base]["coords"], dtype=np.float64)
            idx = {n: i for i, n in enumerate(names)}
            P = xyz[idx["P"]]
            e1, e2 = xyz[idx["C4'"]] - P, xyz[idx[GLYCOLYSIS[base]]] - P
            e3 = np.cross(e1, e2)
            n_true = _plane_normal(xyz[[idx[n] for n in RING_ATOMS[base]]])
            n_lin = (np.stack([e1, e2, e3], axis=1) @ c)
            n_lin = n_lin / np.linalg.norm(n_lin)
            res.append(float(np.degrees(np.arccos(max(-1.0, min(1.0, abs(float(n_lin @ n_true))))))))
        out["_pooled"] = (c, float(np.mean(res)))
        # How much identity actually costs: the purine triple applied to every base.
        cp = out["A"][0] if "A" in out else allc[0]
        resA = []
        for base in "AUGC":
            names = tmpl[base]["names"]
            xyz = np.asarray(tmpl[base]["coords"], dtype=np.float64)
            idx = {n: i for i, n in enumerate(names)}
            P = xyz[idx["P"]]
            e1, e2 = xyz[idx["C4'"]] - P, xyz[idx[GLYCOLYSIS[base]]] - P
            e3 = np.cross(e1, e2)
            n_true = _plane_normal(xyz[[idx[n] for n in RING_ATOMS[base]]])
            n_lin = np.stack([e1, e2, e3], axis=1) @ cp
            n_lin = n_lin / np.linalg.norm(n_lin)
            resA.append(float(np.degrees(np.arccos(max(-1.0, min(1.0, abs(float(n_lin @ n_true))))))))
        out["_purine_for_all"] = (cp, float(np.mean(resA)))
    _coeffs[key] = out
    return out


def pooled_coef(unit_scale: float = 1.0) -> np.ndarray:
    """The pooled triple, rescaled for bead coordinates measured in (template unit / unit_scale).

    THE SCALE MATTERS and it is not cosmetic. The map is alpha*e1 + beta*e2 + gamma*(e1 x e2): the first
    two terms scale with length and the cross product with length SQUARED, so a caller working in nm while
    the coefficients were fitted in Angstrom (the template's unit) gets a normal that is 60 degrees wrong
    rather than a normalization away from right. Measured, on 20 crystal fragments: feeding nm beads gave
    60.9 degrees implied-vs-ring, feeding Angstrom beads 17.6.

    unit_scale = 10.0 returns the triple to use when the beads are in nm.
    """
    c = np.asarray(coeffs(pooled=True)["_pooled"][0], dtype=np.float64).copy()
    c[0] *= unit_scale
    c[1] *= unit_scale
    c[2] *= unit_scale ** 2
    return c


def _pooled_coef_template_units() -> np.ndarray:
    """The one (alpha, beta, gamma) triple for every base: 6.573 degrees off the template's own plane.

    Base identity is not available to the sampler (load_structures returns name, pairs, pos), and it turns
    out not to be needed: with the sign convention above, the four bases' normals in the bead basis differ
    by 6.6 degrees on average, so one triple covers all of them. Carrying a per-residue coefficient array
    through the loop's task tuple would cost more than it buys.
    """
    return np.asarray(coeffs(pooled=True)["_pooled"][0], dtype=np.float64)


def normals_np(beads: np.ndarray, coef: np.ndarray = None, unit_scale: float = 1.0) -> np.ndarray:
    """(L, 3, 3) beads -> (L, 3) unit base-plane normals. The numpy twin of the torch path used in the
    energy term, kept here so the analysis scripts and the potential cannot drift apart.

    unit_scale must match the units of beads (10.0 for nm, 1.0 for Angstrom); see pooled_coef.
    """
    if coef is None:
        coef = pooled_coef(unit_scale)
    b = np.asarray(beads, dtype=np.float64)
    e1 = b[:, 1, :] - b[:, 0, :]
    e2 = b[:, 2, :] - b[:, 0, :]
    e3 = np.cross(e1, e2)
    n = coef[0] * e1 + coef[1] * e2 + coef[2] * e3
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(nn > 1e-12, nn, 1.0)


if __name__ == "__main__":
    per = coeffs()
    print("%-6s %-34s %10s" % ("base", "(alpha, beta, gamma)", "residual"))
    for b in "AUGC":
        c, r = per[b]
        print("%-6s (%9.5f, %9.5f, %9.5f)  %8.3f deg" % (b, c[0], c[1], c[2], r))
    pooled = coeffs(pooled=True)["_pooled"]
    c, r = pooled
    print("%-6s (%9.5f, %9.5f, %9.5f)  %8.3f deg  <- one triple for every base" % ("pooled", c[0], c[1], c[2], r))
    print("\n(That residual is the exactness of the map on the TEMPLATE: the linear image of the beads"
          "\n reproduces the ring plane to this angle. The pooled row is the price of the sampler not"
          "\n carrying base identity.)")
