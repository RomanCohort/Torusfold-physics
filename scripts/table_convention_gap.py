"""The table convention gap, computed from the stored tables in under a second.

scripts/fix_table_interpolation_convention.py measures this by re-fitting the tables from 126
chains, which takes tens of seconds and is why the number had no regression test. But the gap is
a property of (U, binw) as STORED, and results/boltzmann_tables_clean.npz holds exactly that, so
it can be recomputed from the npz without touching the database.

This module is the fast half: load, integrate, return the numbers. The slow script remains the
authority for how the tables were fitted.
"""
from pathlib import Path

import numpy as np

NPZ = Path(__file__).resolve().parent.parent / "results" / "boltzmann_tables_clean.npz"
KBT = 2.494
CORE = 1e-4
TABLED = ("angle", "dihedral", "stack")


def seg_integral(ua, ub, d, kbt):
    """Integral of exp(-U/kbt) over a segment of length d with U linear from ua to ub."""
    ua = np.asarray(ua, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    du = ub - ua
    ea = np.exp(-ua / kbt)
    eb = np.exp(-ub / kbt)
    flat = np.abs(du) < 1e-12
    safe = np.where(flat, 1.0, du)
    return np.where(flat, d * ea, d * kbt / safe * (ea - eb))


def bin_masses(u, binw, kbt=KBT):
    """Mass of every bin under the piecewise-linear u, matching _sample's clamping exactly.

    boltzmann_bonded._sample clamps i0 to [0, len-2] and f to [0, 1], so the first bin's left half
    is flat at u[0] and the last bin's right half is flat at u[-1]. Reproduced rather than
    idealised, because those two flats are what the sampler does.
    """
    n = len(u)
    u_left = np.empty(n)
    u_left[0] = u[0]
    u_left[1:] = 0.5 * (u[:-1] + u[1:])
    u_right = np.empty(n)
    u_right[-1] = u[-1]
    u_right[:-1] = 0.5 * (u[:-1] + u[1:])
    return (seg_integral(u_left, u, 0.5 * binw, kbt)
            + seg_integral(u, u_right, 0.5 * binw, kbt))


def gaps(coord):
    """(core max, core probability-weighted rms) of kBT*ln(P_model/P_target), in kJ/mol."""
    z = np.load(NPZ)
    u = np.asarray(z[f"{coord}__U"], dtype=np.float64)
    binw = float(z[f"{coord}__binw"])
    p_target = np.exp(-(u - u.min()) / KBT)
    p_target = p_target / p_target.sum()
    m = bin_masses(u, binw)
    p_model = np.maximum(m / m.sum(), 1e-300)
    d = KBT * np.log(np.maximum(p_target, 1e-300) / p_model)
    core = p_target >= CORE
    w = p_target[core]
    return float(np.abs(d[core]).max()), float((np.abs(d[core]) * w).sum() / w.sum())
