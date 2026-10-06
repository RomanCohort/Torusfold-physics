"""Run the full isrnaclong pipeline once, scaled down, and report what lands on disk.

WHY THIS EXISTS. The immune fingerprint needs two different things that live in
two different places:

  * base pairing  -- written into `_checkpoint.json` as `pairs` / `ss_consensus` /
    `stem_blocks`, and into `ckpt_bpp.npy`. Measured on the saved `output_web`
    run, which is a 10 nt smoke test whose sequence folds to nothing, so its
    checkpoint carries `pairs: []` and a 10x10 all-zero bpp matrix. That proves
    the fields are written but not that they carry data.
  * all-atom coordinates -- needed for per-residue accessibility. `_write_coords_pdb`
    emits ONE PHOSPHORUS PER RESIDUE in a file named `.pdb`, so most of the PDBs
    the pipeline writes are coarse-grained despite the extension. The delivered
    2013 nt artifact has 21.3 atoms per residue, i.e. it is a different, all-atom
    product.

This driver runs the real entry point -- `isrnaclong_pipeline` -- with the
production arguments from `run_2013nt.py` cut down so it finishes, then prints the
tree and dumps the checkpoint keys. It does not touch server output or any other
session's directories.

Usage:
    python tools/run_pipeline_with_checkpoint.py --length 200 --out results/immuno_run
    python tools/run_pipeline_with_checkpoint.py --length 200 --full   # production args
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

# The CG->all-atom binary cannot be reached through its original path (it is
# GBK-encoded and that path contains non-ASCII characters), so a copy sits in
# _isrnacirc_ascii/. Without these three the pipeline silently produces
# one-phosphorus-per-residue PDBs and there is no per-residue accessibility.
_TOOLS = REPO / "_isrnacirc_ascii"
if _TOOLS.is_dir():
    os.environ.setdefault("ISRNACIRC_BIN_DIR", str(_TOOLS / "bin"))
    os.environ.setdefault("ISRNACIRC_ROOT", str(_TOOLS))
    os.environ.setdefault("CG_TO_ALLATOM_COEFF", str(_TOOLS / "Data" / "data" / "IsRNA2"))

os.environ.setdefault("OPENMM_CPU_THREADS", "32")


def _patch_openmm_no_opencl() -> None:
    """Force OpenMM off OpenCL; the Windows LLVM JIT fails with
    "Can't get available size". Same patch run_2013nt.py applies, and it has to
    happen before openmm is imported."""
    try:
        import openmm as _mm
    except ImportError:
        return
    _orig = _mm.Platform.getPlatformByName

    def _safe_get(name):
        if name in ("OpenCL", "CUDA"):
            raise RuntimeError(f"Disabled: {name}")
        return _orig(name)

    _mm.Platform.getPlatformByName = staticmethod(_safe_get)


_patch_openmm_no_opencl()


def build_sequence(length: int, source: pathlib.Path) -> str:
    """Prefix of the delivered mature circRNA, or a self-complementary stand-in."""
    if source.is_file():
        seq = "".join(source.read_text(encoding="utf-8").split()).upper()
        seq = "".join(c for c in seq if c in "ACGU")
        if len(seq) >= length:
            return seq[:length]
        print(f"  source sequence is only {len(seq)} nt; using it whole")
        return seq
    # No source: a sequence that definitely folds, so the checkpoint cannot come
    # back empty and the run is a real test of the pairing path.
    stem = "GGGCUUCGGCCC"          # 12 nt, self-complementary-ish hairpin stem
    loop = "UUUCGAAA"
    unit = stem + loop
    return (unit * (length // len(unit) + 1))[:length]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200,
                    help="how many nt to run; must be <= --max-seg-len for one segment")
    ap.add_argument("--out", default="results/immuno_run")
    ap.add_argument("--source", default=str(REPO / "artifacts" / "2013nt" / "sequence.txt"))
    ap.add_argument("--full", action="store_true",
                    help="production arguments from run_2013nt.py (hours, not minutes)")
    ap.add_argument("--max-seg-len", type=int, default=200)
    ap.add_argument("--metad-n-steps", type=int, default=2000,
                    help="metadynamics steps; run_2013nt.py left the 200000 default "
                         "in place, which dominates wall clock for a quantity nothing "
                         "downstream reads")
    args = ap.parse_args()

    seq = build_sequence(args.length, pathlib.Path(args.source))
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"sequence : {len(seq)} nt  ({seq[:40]}...)")
    print(f"output   : {out}")
    print()

    # Secondary structure, same route run_2013nt.py uses: ViennaRNA MFE. Cached
    # next to the run so a re-run does not depend on ViennaRNA being importable.
    #
    # CIRCULAR MODE IS NOT OPTIONAL HERE. The target is a circRNA, and the
    # default linear fold cannot pair the two ends across the back-splice
    # junction. The first scaled run used the linear default by accident, which
    # is the wrong secondary structure for the molecule being modelled.
    ss_path = out / "_ss.txt"
    if ss_path.is_file():
        ss = ss_path.read_text(encoding="utf-8").strip()
        print(f"structure: reused from {ss_path.name}  pairs={ss.count('(')}")
    else:
        import RNA  # noqa: PLC0415

        md = RNA.md()
        md.circ = 1
        fc = RNA.fold_compound(seq, md)
        ss, mfe = fc.mfe()
        ss_path.write_text(ss)
        print(f"structure: ViennaRNA MFE={mfe:.1f} (circular)  pairs={ss.count('(')}")

    if len(seq) != len(ss):
        print(f"ABORT: sequence {len(seq)} != structure {len(ss)}")
        return 2

    from torusfold.scheme2.isrnaclong import isrnaclong_pipeline  # noqa: PLC0415

    if args.full:
        # run_2013nt.py's call, with two deliberate changes, both named here so
        # nobody has to diff them:
        #   metad_n_steps   200000 -> args.metad_n_steps. Not in run_2013nt.py's
        #                   list, so it silently took the 200000 default; on a
        #                   200 nt chain that is most of the wall clock for a
        #                   quantity nothing downstream reads.
        #   use_rhofold     True -> False. RhoFold needs its own weights and
        #                   database; the scaled run demonstrated it loads (its
        #                   warning appeared) but the database path is outside
        #                   this repository and nothing here depends on it.
        # Everything else is identical.
        kwargs = dict(
            max_seg_len=args.max_seg_len, overlap=30, n_relax_rounds=20,
            use_rl_relax=True, use_rl_mcts=True, rl_n_simulations=50,
            n_rest2_replicas=16, rest2_nsteps=100000, md_step_scale=0.5, nrep=16,
            platform="auto", use_rhofold=False, use_pyrosetta=True, use_ppr=True,
            ppr_max_rounds=5, n_candidates=1, use_msa=True,
            metad_n_steps=args.metad_n_steps,
            rfam_dir=str(REPO / "msa_work"), rfam_cm=os.environ.get("RFAM_CM"),
            msa_blocks=None, resume=True, verbose=True,
        )
    else:
        # Cut down so it finishes, but keep every STAGE switched on: the point is
        # to exercise the code paths that write the checkpoint, not to sample well.
        kwargs = dict(
            max_seg_len=args.max_seg_len, overlap=20, n_relax_rounds=2,
            use_rl_relax=False, use_rl_mcts=False, rl_n_simulations=2,
            n_rest2_replicas=2, rest2_nsteps=500, md_step_scale=0.05, nrep=1,
            platform="auto", use_rhofold=False, use_pyrosetta=False, use_ppr=True,
            ppr_max_rounds=1, n_candidates=1, use_msa=False,
            rfam_dir="", rfam_cm="", msa_blocks=None, resume=False, verbose=True,
        )

    print()
    print("arguments:")
    for k, v in kwargs.items():
        print(f"  {k:<20} {v}")
    print()

    t0 = time.time()
    result = isrnaclong_pipeline(
        sequence=seq, secondary_structure=ss, output_dir=str(out), **kwargs
    )
    dt = time.time() - t0

    print()
    print("=" * 74)
    print(f"finished in {dt:.0f} s ({dt/60:.1f} min)")
    print("=" * 74)
    for attr in ("n_segments", "pair_rate", "cross_segment_ok_rate", "energy_cg"):
        if hasattr(result, attr):
            print(f"  {attr:<24} {getattr(result, attr)}")

    print()
    print("--- files written ---")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    for p in files:
        print(f"  {p.stat().st_size/1024:9.1f} KB  {p.relative_to(out)}")
    print(f"  total {len(files)} files, "
          f"{sum(p.stat().st_size for p in files)/1024/1024:.1f} MB")

    print()
    print("--- checkpoint ---")
    ck_path = out / "_checkpoint.json"
    if not ck_path.is_file():
        print("  NO CHECKPOINT WRITTEN -- that is the thing this run was testing")
        return 1
    ck = json.loads(ck_path.read_text(encoding="utf-8"))
    print(f"  {ck_path.name}: {len(ck)} keys")
    for k in ("level", "pairs", "far_pairs", "stem_blocks", "ss_consensus",
              "bpp_high", "bpp_mid", "pair_rate", "_seq_len", "n_segments"):
        if k not in ck:
            print(f"    {k:<16} (absent)")
            continue
        v = ck[k]
        if isinstance(v, list):
            print(f"    {k:<16} list[{len(v)}]  {str(v[:3])[:70]}")
        elif isinstance(v, str):
            print(f"    {k:<16} {len(v)} chars  {v[:60]}")
        else:
            print(f"    {k:<16} {v}")

    bpp = out / "ckpt_bpp.npy"
    if bpp.is_file():
        import numpy as np  # noqa: PLC0415

        a = np.load(bpp)
        nz = int((a > 0).sum())
        print(f"  ckpt_bpp.npy: shape {a.shape}  non-zero {nz}  "
              f"max {a.max():.4f}  sum {a.sum():.2f}")

    print()
    print("--- PDB atom counts (is this all-atom or coarse-grained?) ---")
    import collections  # noqa: PLC0415

    for p in sorted(out.rglob("*.pdb")):
        atoms = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.startswith(("ATOM", "HETATM"))]
        if not atoms:
            continue
        res = collections.OrderedDict()
        for l in atoms:
            res.setdefault((l[21], l[22:27]), l[17:20].strip())
        print(f"  {p.relative_to(out)!s:<44} {len(atoms):6d} atoms  "
              f"{len(res):5d} res  {len(atoms)/max(1,len(res)):.1f} atoms/res")
    return 0


if __name__ == "__main__":
    sys.exit(main())
