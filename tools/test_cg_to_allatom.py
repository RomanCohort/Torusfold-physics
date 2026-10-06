"""Test CG_to_allatom on the P-only structure our run produced.

This is the step that decides whether per-residue accessibility is computable at
all on this machine. Everything the pipeline writes in this repository is
one-phosphorus-per-residue, so without a working CG->all-atom conversion the
immune fingerprint can have pairing (from the checkpoint) but not accessibility.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
TOOLS = REPO / "_isrnacirc_ascii"

os.environ["ISRNACIRC_BIN_DIR"] = str(TOOLS / "bin")
os.environ["ISRNACIRC_ROOT"] = str(TOOLS)
os.environ["CG_TO_ALLATOM_COEFF"] = str(TOOLS / "Data" / "data" / "IsRNA2")

sys.path.insert(0, str(REPO / "src"))

cg = REPO / "results" / "immuno_run" / "cg2aa" / "seg_0_cg.pdb"
out = REPO / "results" / "immuno_run" / "cg2aa" / "_test_aa.pdb"

print(f"toolset : {TOOLS}")
print(f"bin dir : {os.environ['ISRNACIRC_BIN_DIR']}")
print(f"coeff   : {os.environ['CG_TO_ALLATOM_COEFF']}")
print(f"input   : {cg}  ({cg.stat().st_size} bytes)")
print()

seq_path = REPO / "results" / "immuno_run" / "_ss.txt"
# recover the sequence the run used
src = "".join((REPO / "artifacts" / "2013nt" / "sequence.txt").read_text(
    encoding="utf-8").split()).upper()
seq = "".join(c for c in src if c in "ACGU")[:200]
print(f"sequence: {len(seq)} nt")

try:
    from torusfold.scheme2.isrnacirc_wrapper import cg_to_allatom, _CG_TO_AA_EXE
    print(f"wrapper resolved exe: {_CG_TO_AA_EXE}")
    print(f"  exists: {os.path.isfile(_CG_TO_AA_EXE)}")
    print()
    if not os.path.isfile(_CG_TO_AA_EXE):
        print("ABORT: wrapper did not resolve the binary")
        sys.exit(2)
    cg_to_allatom(str(cg), str(out), seq)
    print(f"OK -> {out}  ({out.stat().st_size} bytes)")
except Exception as e:  # noqa: BLE001
    print(f"wrapper FAILED: {type(e).__name__}: {e}")
    print()
    print("--- trying the binary directly, to separate a wrapper problem")
    print("    from a binary problem ---")
    exe = TOOLS / "bin" / "CG_to_allatom.exe"
    cmd = f'"{exe}" "{cg}" "{out}"'
    print(f"cmd: {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(TOOLS / "bin"))
    print(f"returncode: {r.returncode}")
    print(f"stdout: {r.stdout[:600]!r}")
    print(f"stderr: {r.stderr[:600]!r}")
    sys.exit(1)

# If we got here, report what came out.
import collections  # noqa: PLC0415

lines = out.read_text(encoding="utf-8", errors="replace").splitlines()
atoms = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
res = collections.OrderedDict()
for l in atoms:
    res.setdefault((l[21], l[22:27]), l[17:20].strip())
print()
print(f"  atoms    : {len(atoms)}")
print(f"  residues : {len(res)}")
print(f"  atoms/res: {len(atoms) / max(1, len(res)):.1f}")
names = collections.Counter(l[12:16].strip() for l in atoms)
c1_atom = "C1" + "'"
print(f"  atom names: {dict(names.most_common(8))}")
print(f"  has C1'   : {c1_atom in names}")
print(f"  first: {atoms[0][:72]!r}")
