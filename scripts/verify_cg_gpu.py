"""Verify the tabulated CG field on the GPU: same structure, cpu against cuda.

Run with any interpreter that has torch:
    C:/ana/envs/comfyui/python.exe scripts/verify_cg_gpu.py    (ROCm build, cuda.is_available() True)
    C:/ana/envs/circrna3d/python.exe scripts/verify_cg_gpu.py  (cpu build, verifies only the cpu half)

WHAT IT CHECKS. The production tables (results/production_tables.npz, composed by
scripts/build_production_tables.py) are installed through cg_potentials.build_potential_kwargs -- the
same call the pipeline now makes when TORUSFOLD_CG_TABLES is set -- and cg_energy_forces is evaluated
on the identical geometry on cpu and on the GPU. Before this change the table potentials refused a
non-cpu pos outright (force_reference's "left until something needs it"), so the ratio being finite at
all is the point; the numbers say whether the device path is faithful.

Tolerance: cross-device float64 reductions reorder, so the two are compared with a relative bound
rather than bit-equality. The cpu-vs-cpu pair at the top is the zero the comparison is read against.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))

import cg_potentials as P              # noqa: E402
from torusfold.scheme2 import torch_cgsim as C   # noqa: E402

TABLE = REPO / "results" / "production_tables.npz"
REL_TOL = 1e-9


def build(L=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    pos = torch.zeros(1, 3 * L, 3, dtype=torch.float64)
    step = 0.59
    for i in range(L):
        x = step * i
        pos[0, 3 * i] = torch.tensor([x, 0.0, 0.0], dtype=torch.float64)
        pos[0, 3 * i + 1] = torch.tensor([x + 0.12, 0.32, 0.0], dtype=torch.float64)
        pos[0, 3 * i + 2] = torch.tensor([x + 0.19, 0.58, 0.13], dtype=torch.float64)
    pos = pos + 0.01 * torch.randn(pos.shape, generator=g, dtype=torch.float64)
    ij = torch.tensor([[i, i + 1] for i in range(L - 1)], dtype=torch.long)
    pw = torch.ones(len(ij), dtype=torch.float32)
    return pos, ij, pw


def main():
    print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
    pots, pot_kw = P.build_potential_kwargs(TABLE)
    print("potentials from", TABLE.name, "->", [(c, s if isinstance(s, str) else s[0])
                                                for c, s, _ in pots])
    pos, ij, pw = build()
    with torch.no_grad():
        e1, f1 = C.cg_energy_forces(pos, ij, pw, **pot_kw)
        e2, f2 = C.cg_energy_forces(pos.clone(), ij, pw, **pot_kw)
    de = float((e1 - e2).abs().max())
    print(f"cpu repeat:        dE = {de:.3e}  (the zero this comparison is read against)")
    print(f"cpu: E = {float(e1.sum()):.12e}  |F| = {float(f1.norm()):.12e}")
    if not torch.cuda.is_available():
        print("no CUDA device: cuda half skipped (this is the cpu build of torch)")
        return 0
    pos_g = pos.to("cuda")
    ij_g, pw_g = ij.to("cuda"), pw.to("cuda")
    with torch.no_grad():
        eg, fg = C.cg_energy_forces(pos_g, ij_g, pw_g, **pot_kw)
    eg_cpu, fg_cpu = eg.detach().cpu(), fg.detach().cpu()
    dE = float((eg_cpu - e1).abs().max())
    relE = dE / max(abs(float(e1.sum())), 1e-30)
    dF = float((fg_cpu - f1).abs().max())
    relF = dF / max(float(f1.abs().max()), 1e-30)
    print(f"cuda: E = {float(eg_cpu.sum()):.12e}  |F| = {float(fg_cpu.norm()):.12e}")
    print(f"cpu vs cuda:       dE = {dE:.3e} (rel {relE:.3e})   dF = {dF:.3e} (rel {relF:.3e})")
    ok = relE < REL_TOL and relF < REL_TOL
    print("VERDICT:", "the tabulated field runs on the GPU and agrees with cpu" if ok
          else "MISMATCH beyond the tolerance -- do not ship this")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
