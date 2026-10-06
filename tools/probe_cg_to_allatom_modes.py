"""Does CG_to_allatom produce base ring atoms when fed a 3-bead PDB?

The mechanism this tests, assembled from the code rather than guessed:

  * openmm_gpu_refiner._write_allatom_pdb() writes a PDB whose residue names are
    ADE / GUA / CYT / URA and whose comment says CG_to_allatom.exe requires them.
  * isrnacirc_wrapper.build_cg_pdb() (line 121) uses the SAME ADE/GUA/CYT/URA map,
    and isrnacirc_wrapper.cg_to_allatom() then runs
        CG_to_allatom.exe <in.pdb> <out.pdb> <coeff_dir>
  * isrnaclong._write_coords_pdb() (line 3387) ALSO uses ADE/GUA/CYT/URA but emits
    a single P per residue.

Two products come out of the same binary depending on the input:
    remd_r0.pdb from a 3-bead input  -> 21.6 atoms/residue, base ring atoms present
    cg2aa/*.pdb  from a P-only input -> 12.0 atoms/residue, backbone only

This script runs the P-only case and the 3-bead case through the same binary, on
the same coordinates, and reports the atom sets. Read-only with respect to the
toolset: nothing is copied, renamed or configured.
"""
from __future__ import annotations

import collections
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
TOOLS = REPO / "_isrnacirc_ascii"
EXE = TOOLS / "bin" / "CG_to_allatom.exe"
COEFF = TOOLS / "Data" / "data" / "IsRNA2"
RUN = REPO / "results" / "immuno_full"

RING_N = {"N1", "N2", "N3", "N4", "N6", "N7", "N9", "O2", "O4", "O6"}
BASE_MAP = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}

print("=" * 78)
print("toolset")
print("=" * 78)
print(f"  exe   {EXE}  ({EXE.stat().st_size/1024:.1f} KB)")
print(f"  coeff {COEFF}  ({len(list(COEFF.iterdir()))} files)")

# --- the sequence and the CG coordinates -----------------------------------
src = "".join((REPO / "artifacts" / "2013nt" / "sequence.txt").read_text(
    encoding="utf-8").split()).upper()
seq = "".join(c for c in src if c in "ACGU")

cg = RUN / "cg2aa" / "seg_0_cg.pdb"
lines = cg.read_text(encoding="utf-8", errors="replace").splitlines()
atoms = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
P = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in atoms]
L = len(P)
seq = seq[:L]
print(f"  input {cg.name}: {L} P atoms, sequence {len(seq)} nt")
print(f"  input residue name as written: {atoms[0][17:20].strip()!r}")


def write_variant(path: pathlib.Path, n_beads: int) -> pathlib.Path:
    """Write the same P trace with 1 or 3 beads per residue.

    3-bead uses the P, C4' and N1/N9 slots the model itself uses
    (torch_cgsim.py:1297-1299: P=3i, C4=3i+1, NN=3i+2). C4 and NN are placed
    along/against the backbone step, which is a reconstruction -- the point of
    this probe is the ATOM SET that comes back, not the geometry.
    """
    names = ["P", "C4'", "N9"] if n_beads == 3 else ["P"]
    out = []
    serial = 1
    for i in range(L):
        px, py, pz = P[i]
        if i + 1 < L:
            dx, dy, dz = (P[i + 1][0] - px, P[i + 1][1] - py, P[i + 1][2] - pz)
        else:
            dx, dy, dz = (P[0][0] - px, P[0][1] - py, P[0][2] - pz)
        n = (dx * dx + dy * dy + dz * dz) ** 0.5 or 1.0
        ux, uy, uz = dx / n, dy / n, dz / n
        base = seq[i]
        # N1 for pyrimidines, N9 for purines
        nn_name = "N9" if base in "AG" else "N1"
        coords3 = [
            (px, py, pz),
            (px + 2.8 * ux, py + 2.8 * uy, pz + 2.8 * uz),
            (px - 2.8 * ux, py - 2.8 * uy, pz - 2.8 * uz),
        ]
        for k in range(n_beads):
            x, y, z = coords3[k]
            nm = names[k]
            if k == 2:
                nm = nn_name
            out.append(
                f"ATOM  {serial:5d} {nm:<4s} {BASE_MAP[base]:>3s} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {nm[0]:>2s}"
            )
            serial += 1
    out.append("END")
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    return path


def run_exe(inp: pathlib.Path, outp: pathlib.Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PATH"] = str(EXE.parent) + os.pathsep + env.get("PATH", "")
    r = subprocess.run([str(EXE), str(inp), str(outp), str(COEFF)],
                       capture_output=True, text=True, timeout=900,
                       env=env, cwd=str(EXE.parent))
    return r.returncode, r.stdout or "", r.stderr or ""


def profile(path: pathlib.Path):
    a = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
         if l.startswith(("ATOM", "HETATM"))]
    res: dict = collections.OrderedDict()
    names = set()
    resnames = set()
    for l in a:
        res.setdefault((l[21], l[22:27]), l[17:20].strip())
        names.add(l[12:16].strip())
        resnames.add(l[17:20].strip())
    return a, res, names, resnames


work = REPO / "results" / "immuno_full" / "_cg2aa_probe"
work.mkdir(parents=True, exist_ok=True)

for n_beads in (1, 3):
    inp = write_variant(work / f"in_{n_beads}bead.pdb", n_beads)
    outp = work / f"out_{n_beads}bead.pdb"
    if outp.exists():
        outp.unlink()
    print()
    print("=" * 78)
    print(f"{n_beads} bead(s) per residue")
    print("=" * 78)
    print(f"  input  {inp.name}: {inp.read_text(encoding='utf-8').count(chr(10))-1} atoms")
    ret, so, se = run_exe(inp, outp)
    print(f"  exit   {ret}")
    if so.strip():
        print(f"  stdout {so.strip()[:200]}")
    if se.strip():
        print(f"  stderr {se.strip()[:200]}")
    if not outp.is_file():
        print("  NO OUTPUT")
        continue
    a, res, names, resnames = profile(outp)
    print(f"  output {outp.name}: {len(a)} atoms, {len(res)} res, "
          f"{len(a)/max(1,len(res)):.1f} atoms/res")
    print(f"  residue names : {sorted(resnames)}")
    print(f"  atom names    : {sorted(names)}")
    ring = sorted(names & RING_N)
    print(f"  RING ATOMS    : {ring if ring else 'none'}")
    print(f"  => pair cross-check possible: {'YES' if ring else 'NO'}")
