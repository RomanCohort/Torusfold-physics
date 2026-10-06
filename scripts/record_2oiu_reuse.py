# -*- coding: utf-8 -*-
"""A checkable reuse record: 2OIU through the shipped CG stage, from the deposit, one command.

WHY THIS EXISTS. Everything in this project's force-field work runs on 2OIU, the one experimentally
resolved circular RNA in the tree, and the reuse record had two entries and a gap: 2OIU, and a 2,013 nt
artifact that is DECODED rather than re-run (artifacts/2013nt/provenance.json says "is_a_rerun": false), so
a third party can verify its hashes but cannot reproduce it. A community-reuse page should carry at least
one record a reader can EXECUTE and CHECK. This is that record, and it is on 2OIU for a reason worth
stating plainly: 2OIU is the only circRNA here with a structure to be right about, so the record can be
checked against something the reader did not get from us.

WHAT IT DOES.

    python scripts/record_2oiu_reuse.py --spec results/plan_c/_2oiu_input.json
    python scripts/record_2oiu_reuse.py --verify-only

The run: takes the deposited P trace as the start, switches the production stage to the calibrated
protocol through the five environment variables of findings Part 29 (no edited call site), runs the
shipped torch refiner with verbose=False -- the path a predictor uses -- then measures the product with
the same instruments the findings use and writes everything to artifacts/reuse_demo/2oiu/:

    record.json      every number, every hash, the exact command and environment, the sampled bead frame
    2oiu_reuse.pdb   the all-atom product (the thing to look at)
    2oiu_reuse_cg.pdb the CG P trace the run ended on
    input_p.pdb      the P-only start the refiner was given
    run_output.txt   the run's own stdout, including the lines saying what the environment changed

--verify-only re-derives every product-side number from the committed files and compares, and it needs no
GPU, no torch and no ViennaRNA -- only numpy and the two small PDBs. That is the point of the record: the
reader who cannot reproduce the run can still check that the artifacts are what the record says they are,
and a reader who can reproduce the run has a hash to compare against.

UNITS. Everything measured is Angstrom. The pipeline's own coordinates are nm on the way in and out of
torch_gpu_refine; the conversions are at the call sites and flagged there, because getting one wrong is
how this project's first product scoreboard reported an unstacked product (findings Part 25).
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))

import measure_base_stacking as M                                          # noqa: E402
# numpy-only, and needed by BOTH modes: the run measures the two attribution rows with it, and
# --verify-only re-derives them. It was a function-local import first, which is how run() came to call a
# name that did not exist in its own scope.
from torusfold.scheme2.aform_from_template import (reconstruct_all_atom,   # noqa: E402
                                                   reconstruct_all_atom_from_beads,
                                                   repair_base_placement, write_allatom_pdb)

DEPOSIT = REPO / "artifacts" / "2oiu" / "2OIU.pdb"
SPEC = REPO / "results" / "plan_c" / "_2oiu_input.json"
OUT_DEFAULT = REPO / "artifacts" / "reuse_demo" / "2oiu"
OUT_REPAIR = REPO / "artifacts" / "reuse_demo" / "2oiu_repair"
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
# The five switches of Part 29, plus the seed. Values are the ones the record was produced with; a reader
# changing any of them is running a different experiment, which is why they are in the record verbatim.
ENV_ROUTE = {
    "TORUSFOLD_CG_TABLES": "results/production_tables.npz",
    "TORUSFOLD_BASE_STACK": "16.6:0,1,0.19",
    "TORUSFOLD_REFINE_MODE": "refine",
    "TORUSFOLD_REFINE_STEPS": "1000",
    "TORUSFOLD_BEAD_SOURCE": "artifacts/2oiu/2OIU.pdb",
    "TORUSFOLD_SEED": "20261005",
}
CONTACT_CUTOFF = 3.6
# The source files whose bytes decide what this run produces. A record is only as reproducible as the tree
# it came from, and at the time of writing three of these were DIRTY -- uncommitted edits in a shared
# working tree -- which is exactly the fact a reader comparing hashes needs before blaming their GPU. So
# the record carries the fingerprint and says which entries are not in a commit.
TREE_FILES = (
    "src/torusfold/scheme2/torch_gpu_refine.py",
    "src/torusfold/scheme2/torch_cgsim.py",
    "src/torusfold/scheme2/aform_from_template.py",
    "src/torusfold/scheme2/openmm_gpu_refiner.py",
    "src/torusfold/scheme2/base_frames.py",
    "src/torusfold/scheme2/base_stacking.py",
    "scripts/cg_potentials.py",
    "scripts/ibi_core.py",
    "scripts/measure_base_stacking.py",
)


# ---------------------------------------------------------------- small machinery

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path):
    """A repository-relative POSIX path when the file is inside the tree, else an absolute POSIX one.

    POSIX because this record is meant to be read on other machines: a Windows-built record with
    backslashes does not resolve on Linux, and an absolute path from the machine that produced it is not
    useful to anyone else. The first version stored both, which is how it was noticed.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def _abs(rel):
    """Inverse of _rel, tolerant of the backslashes an earlier record wrote."""
    p = Path(str(rel).replace("\\", "/"))
    return p if p.is_absolute() else REPO / p


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8", newline="\n")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:                                                      # noqa: BLE001
        return None


def tree_fingerprint():
    """(files, uncommitted) -- sha256 of every file whose bytes decide the run, and which are not committed."""
    files = []
    for rel in TREE_FILES:
        p = REPO / rel
        if not p.exists():
            files.append({"path": rel, "sha256": None, "committed": False, "note": "absent"})
            continue
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=str(REPO),
                                    capture_output=True, text=True).stdout.strip())
        files.append({"path": rel, "sha256": sha256_file(p), "committed": not dirty})
    return files, [f["path"] for f in files if not f["committed"]]


def deposit_reference():
    """The deposit as the reference row: sequence, geometric WC pairs, its own beads, its own rings."""
    seq, residues, p_crystal, chain = M.parse_pdb(str(DEPOSIT))
    beads = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]] for r in residues])
    planes = [M.plane(r) for r in residues]
    pairs = M.wc_pairs(seq, residues, planes)
    return {"seq": seq, "chain": chain, "residues": residues, "p": p_crystal, "beads": beads,
            "pairs": pairs, "planes": planes}


# ---------------------------------------------------------------- the instruments

def linear_rows(beads_A, n):
    """base_dist / base_rise / base_cos through the FIXED LINEAR MAP the field itself scores on.

    This is base_frames.normals_np, the same map the base-level term and every base number in Parts 15-24
    use. It is a linearisation of the rigid body about the template geometry, which is why the record also
    carries the ring-atom row below -- the two differ away from the deposit by a measured amount, and
    reporting only one of them would make the gap look like a disagreement between experiments.
    """
    from torusfold.scheme2 import base_frames as BF
    nrm = BF.normals_np(beads_A, BF.pooled_coef(1.0))
    d, rise, cos = [], [], []
    for i in range(n - 1):
        a, b = nrm[i].copy(), nrm[i + 1].copy()
        if float(np.dot(a, b)) < 0:
            b = -b
        m = a + b
        nm = float(np.linalg.norm(m))
        m = m / nm if nm > 1e-9 else a
        dc = beads_A[i + 1, 2] - beads_A[i, 2]
        d.append(float(np.linalg.norm(dc)))
        rise.append(float(np.dot(dc, m)))
        cos.append(float(np.dot(a, b)))
    return np.array(d), np.array(rise), np.array(cos)


def ring_rows(beads_A, residues, seq):
    """The same three quantities measured on the base RING PLANES of the structure itself.

    The beads are used for one thing only: the sign of an SVD normal is arbitrary, so it is flipped to
    point along +e3 = (C4'-P) x (N-P), which is the convention the map uses. Without that flip the mean
    rise over pairs is a sum of mixed signs and reads ~0 (measured on the deposit: +0.086 +- 0.997 A
    against the map's +3.186).
    """
    L = len(seq)
    planes = [M.plane(r) for r in residues]
    # THE SIGN CONVENTION GOES FIRST, for every plane, BEFORE any mean normal or cosine exists. Flipping
    # inside the pairwise loop is what this did first: it aligned b to a's OLD (arbitrary) sign and only
    # then reversed a, so both the mean rise and the mean cosine came out as sums of mixed signs -- on the
    # deposit that read +0.056 A and 0.158 against the map's +3.19 A and 0.901. The reference row is the
    # only reason it was visible, which is why the record carries one.
    signed = []
    for i, pl in enumerate(planes):
        if pl is None:
            signed.append(None)
            continue
        c, n, v = pl
        e3 = np.cross(beads_A[i, 1] - beads_A[i, 0], beads_A[i, 2] - beads_A[i, 0])
        n = np.asarray(n, float)
        signed.append((c, -n if float(np.dot(n, e3)) < 0 else n, v))
    d, rise, cos = [], [], []
    for i in range(L - 1):
        if signed[i] is None or signed[i + 1] is None:
            continue
        _ci, a, _vi = signed[i]
        _cj, b, _vj = signed[i + 1]
        if float(np.dot(a, b)) < 0:
            b = -b
        m = a + b
        nm = float(np.linalg.norm(m))
        m = m / nm if nm > 1e-9 else a
        gi, gj = GLY.get(seq[i]), GLY.get(seq[i + 1])
        if gi not in residues[i]["atoms"] or gj not in residues[i + 1]["atoms"]:
            continue
        # The SAME points the map measures -- N9/N1 -- not the ring centroids, which sit in the middle of
        # the ring and read 2.62 A where the beads read 5.31 on the deposit.
        dc = np.asarray(residues[i + 1]["atoms"][gj], float) - np.asarray(residues[i]["atoms"][gi], float)
        d.append(float(np.linalg.norm(dc)))
        rise.append(float(np.dot(dc, m)))
        cos.append(float(a @ b))
    return np.array(d), np.array(rise), np.array(cos)


def contacts(residues, seq, pairs, cutoff=CONTACT_CUTOFF):
    """How many of the DEPOSIT's WC pairs have all their key contacts inside the cutoff, here.

    The criterion is diagnose_wc_contacts.py's, and the pair list is the deposit's own (12 pairs on 2OIU),
    so '12/12' means 'every pair the experiment has is still in contact' rather than 'the predictor found
    pairs and then satisfied them'.
    """
    ok, worst = 0, []
    for (i, j) in pairs:
        trip = M.WC_TRIPLES.get((seq[i], seq[j])) or M.WC_TRIPLES.get((seq[j], seq[i]))
        if trip is None:
            continue
        flip = (seq[i], seq[j]) not in M.WC_TRIPLES
        ai, aj = residues[i]["atoms"], residues[j]["atoms"]
        ds = []
        for a, b in trip:
            na, nb = (b, a) if flip else (a, b)
            if na not in ai or nb not in aj:
                ds.append(float("nan"))
                continue
            ds.append(float(np.linalg.norm(np.asarray(ai[na], float) - np.asarray(aj[nb], float))))
        worst.append(float(np.nanmax(ds)) if ds else float("nan"))
        if ds and np.nanmax(ds) <= cutoff:
            ok += 1
    return ok, len(worst), worst


def stacking(residues, seq, pairs, planes=None):
    """The stacking instrument's own summary over helical steps."""
    planes = planes if planes is not None else [M.plane(r) for r in residues]
    rows, skipped = M.measure(seq, residues, pairs, planes)
    if not rows:
        return {"stacked_percent": None, "n_steps": 0, "rise_stacked": None, "theta_deg": None}
    st = [r for r in rows if r["stacked"]]
    return {"stacked_percent": 100.0 * len(st) / len(rows), "n_steps": len(rows),
            "rise_stacked": float(np.mean([r["rise"] for r in st])) if st else None,
            # Both definitions, because the findings and this record quote different ones and the
            # difference (3.29 vs 3.42 A on the same run) is a definition, not a disagreement.
            "rise_all_helical_mean": float(np.mean([r["rise"] for r in rows])),
            "rise_all_helical_sd": float(np.std([r["rise"] for r in rows])),
            "theta_deg": float(np.mean([r["theta"] for r in rows])),
            "skipped": skipped}


def side_report(beads_A, residues, seq, pairs, planes=None):
    """One row of the scoreboard: everything the record claims about a structure, from its own files."""
    d_lin, r_lin, c_lin = linear_rows(beads_A, len(seq))
    d_rng, r_rng, c_rng = ring_rows(beads_A, residues, seq)
    ok, total, worst = contacts(residues, seq, pairs)
    out = {
        "base_dist_nm_linear_map": float(d_lin.mean()) / 10.0,
        "base_rise_nm_linear_map": float(r_lin.mean()) / 10.0,
        "base_cos_linear_map": float(c_lin.mean()),
        "base_dist_nm_ring_atoms": float(d_rng.mean()) / 10.0 if len(d_rng) else None,
        "base_rise_nm_ring_atoms": float(r_rng.mean()) / 10.0 if len(r_rng) else None,
        "base_cos_ring_atoms": float(c_rng.mean()) if len(c_rng) else None,
        "wc_contacts": "%d/%d" % (ok, total),
        "wc_contact_worst_A": sorted(round(w, 3) for w in worst if np.isfinite(w)),
    }
    out.update(stacking(residues, seq, pairs, planes))
    return out


def beads_of(residues):
    """(L, 3, 3) Angstrom P / C4' / N9-N1 of a structure's own atoms."""
    return np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]] for r in residues])


def parse_side(path, seq_expected):
    """(seq, residues, beads in A, P trace in A) of a structure file, sequence checked not assumed."""
    seq, residues, p, chain = M.parse_pdb(str(path))
    if seq is None:
        raise SystemExit("%s: no RNA chain with >= 10 residues was parsed" % path)
    if seq_expected is not None and seq != seq_expected:
        raise SystemExit("%s: sequence differs from the deposit's (%d vs %d nt)"
                         % (path, len(seq), len(seq_expected)))
    beads = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]] for r in residues])
    return seq, residues, beads, p


# ---------------------------------------------------------------- the run

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def read_spec(spec_path, deposit_seq):
    """(ss, pairs, seq) from the precomputed spec, or from ViennaRNA when no spec is given."""
    if spec_path is not None:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        # Both spellings: the pipeline's own spec uses "seq"/"ss", the copy this script commits next to the
        # product uses the longer names, and either should be accepted as --spec.
        ss = spec.get("ss") or spec.get("secondary_structure")
        seq = spec.get("seq") or spec.get("sequence")
        return ss, [(int(i), int(j), float(w)) for i, j, w in spec["pairs"]], seq, \
            {"route": "precomputed spec", "path": _rel(spec_path), "sha256": sha256_file(spec_path)}
    try:
        import ViennaRNA
    except Exception as exc:                                               # noqa: BLE001
        raise SystemExit("no --spec given and ViennaRNA is not importable here (%s).\n"
                         "The GPU environment on this machine has no ViennaRNA, which is why the record "
                         "was produced from results/plan_c/_2oiu_input.json -- the same call "
                         "scripts/prep_2oiu_input.py makes. Pass --spec, or run where ViennaRNA is "
                         "installed." % exc)
    fc = ViennaRNA.fold_compound(deposit_seq)
    ss = fc.mfe()[0]
    fc.pf()
    bpp = fc.bpp()
    pairs = [(i - 1, j - 1, float(bpp[i][j])) for i in range(1, len(deposit_seq) + 1)
             for j in range(i + 1, len(deposit_seq) + 1) if bpp[i][j] > 0.1]
    try:
        import importlib.metadata as md
        vna_version = md.version("ViennaRNA")
    except Exception:                                                      # noqa: BLE001
        vna_version = "version unknown"
    return ss, pairs, deposit_seq, {"route": "ViennaRNA %s" % vna_version,
                                    "path": None, "sha256": None}


def _find_tables(explicit, out):
    """The CG tables the run needs, from wherever they can be found.

    results/ is git-ignored here, so a fresh clone has no production_tables.npz; the record commits a
    byte-identical copy next to the product, and this is the order it looks in. The first version only
    looked beside the output directory, which failed the moment the reader re-ran into a different output
    directory -- which is exactly what re-running means.
    """
    name = Path(ENV_ROUTE["TORUSFOLD_CG_TABLES"]).name
    dirs = []
    if explicit:
        e = Path(explicit)
        dirs.append(e.parent if e.is_file() or not e.exists() else e)
    dirs.append(REPO / "results")
    dirs.append(Path(out))
    dirs.extend(sorted((REPO / "artifacts" / "reuse_demo").glob("*/")))
    # BOTH FILES, in one directory: the refiner takes the dynamics from production_tables.npz and its
    # reference binning grids from refit_smooth5_with_base.npz, and it looks for the second one NEXT TO
    # whichever TORUSFOLD_CG_TABLES it was given (_refine_langevin). A directory with only the first is
    # therefore not a usable answer, which is how the first version of this search failed.
    for d in dirs:
        p = Path(d) / name
        if p.exists() and (Path(d) / "refit_smooth5_with_base.npz").exists():
            return p
    return None


def _execute(start, out_dir, seq, ss, name, steps, stream, calls):
    """One call to the shipped stage, with everything it prints going to the given stream."""
    from torusfold.scheme2.torch_gpu_refine import torch_gpu_refine
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(Tee(stream, buf)):
        pdb_out, energy, diag = torch_gpu_refine(
            str(start), str(out_dir), seq, ss, name=name, verbose=False, refine_steps=steps,
            on_report=(lambda *a, **k: calls.append(a)) if calls is not None else None)
        print("returned: %s  E=%.1f" % (Path(pdb_out).name, float(energy)))
    # The text comes back too: the repair reports how many bases it moved on stdout and nowhere else, and
    # the record has to carry that number for --verify-only to check the repair it is asked to reproduce.
    return pdb_out, float(energy), diag, time.time() - t0, buf.getvalue()


def run(args):
    ref = deposit_reference()
    seq, L = ref["seq"], len(ref["seq"])
    ss, pairs_w, spec_seq, spec_info = read_spec(args.spec, seq)
    if spec_seq != seq:
        raise SystemExit("the spec's sequence (%d nt) is not the deposit's (%d nt)" % (len(spec_seq), L))
    # Absolute, because the record stores paths relative to the repository root and a relative --out was
    # how the first --verify-only run died: Path("artifacts/...").relative_to(REPO) is a ValueError.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # The start: the deposited P trace, written P-only at the same fixed columns the loader reads.
    p_crystal = np.asarray(ref["p"], dtype=float)
    start = out / "input_p.pdb"
    with start.open("w", encoding="utf-8", newline="\n") as fh:
        for i in range(L):
            x, y, z = p_crystal[i]
            if max(abs(x), abs(y), abs(z)) >= 1000.0:
                raise SystemExit("coordinate %.1f A does not fit PDB columns 31-54 (8.3f); a longer chain "
                                 "needs a different start format, not a wider field" % max(abs(x), abs(y), abs(z)))
            fh.write("ATOM  %5d  P   RA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           P\n"
                     % (i + 1, i + 1, x, y, z))
        fh.write("END\n")
    bsj_A = float(np.linalg.norm(p_crystal[0] - p_crystal[-1]))

    env = dict(ENV_ROUTE)
    env["TORUSFOLD_REFINE_STEPS"] = str(args.steps)
    if args.repair:
        env["TORUSFOLD_HBOND_REPAIR"] = "1"

    # THE FIELD HAS TO TRAVEL WITH THE RECORD. results/ is git-ignored in this repository, so a reader who
    # clones it has no production_tables.npz at all -- and a record whose command cannot be run from the
    # repository it ships in is not a record. The tables are 26 KB, so a byte-identical copy goes beside
    # the product and the run falls back to that copy when the source is missing.
    tables_src = _find_tables(args.tables, out)
    if tables_src is None:
        raise SystemExit(
            "no CG tables found. The run needs production_tables.npz AND refit_smooth5_with_base.npz in one "
            "directory (the refiner takes the dynamics from the first and its reference grids from the "
            "second). Looked in results/, the output directory and artifacts/reuse_demo/*/. Pass --tables "
            "<production_tables.npz>, or put a copy where the record expects it.")
    tables_copy = out / tables_src.name
    refit_copy = out / "refit_smooth5_with_base.npz"
    for src, dst in ((tables_src, tables_copy),
                     (tables_src.parent / "refit_smooth5_with_base.npz", refit_copy)):
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    if tables_src.resolve() != (REPO / ENV_ROUTE["TORUSFOLD_CG_TABLES"]).resolve():
        env["TORUSFOLD_CG_TABLES"] = _rel(tables_copy)
        print("  NOTE: running from the CG tables at %s" % _rel(tables_copy))
    tables_info = {"run_used": env["TORUSFOLD_CG_TABLES"], "found_at": _rel(tables_src),
                   "production_tables": {"path": _rel(tables_copy), "sha256": sha256_file(tables_copy)},
                   "refit_smooth5_with_base": {"path": _rel(refit_copy),
                                               "sha256": sha256_file(refit_copy)}}

    for k, v in env.items():
        os.environ[k] = v
    resolved = {k: str((REPO / v).resolve()) if not Path(v).is_absolute() else v
                for k, v in env.items() if k in ("TORUSFOLD_CG_TABLES", "TORUSFOLD_BEAD_SOURCE")}

    print("running the shipped stage: L=%d, %d spec pairs, %d deposit WC pairs, start BSJ %.3f A"
          % (L, len(pairs_w), len(ref["pairs"]), bsj_A), flush=True)
    import torch
    calls = []
    # Not ".log": the repository ignores *.log, and a record whose transcript is silently missing from a
    # clone is not a record.
    log = out / "run_output.txt"
    with log.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("command: %s\n" % " ".join([sys.executable, "scripts/record_2oiu_reuse.py"] + sys.argv[1:]))
        fh.write("environment: %s\n" % json.dumps(env, sort_keys=True))
        pdb_out, energy, diag, wall, text = _execute(start, out, seq, ss, args.name, args.steps,
                                                     Tee(sys.stdout, fh), calls)
    print("  E=%.1f in %.0f s -> %s" % (energy, wall, Path(pdb_out).name), flush=True)

    # DOES THE SEED ACTUALLY PIN THE RUN? A record that says "seed 20261005" is only worth the answer to
    # that question, and the shipped log says as much ("this fixes the noise stream, not the kernels").
    # Repeats go to a scratch directory so the committed record stays clean while carrying the evidence.
    repeats = []
    main_sha = sha256_file(pdb_out)
    for k in range(args.repeat):
        rdir = REPO / "results" / "plan_c" / "_reuse_repeat" / ("repeat%d" % k)
        rdir.mkdir(parents=True, exist_ok=True)
        r_pdb, r_e, _d, r_wall, _t = _execute(start, rdir, seq, ss, args.name + "_r%d" % k, args.steps,
                                              io.StringIO(), None)
        r_sha = sha256_file(r_pdb)
        repeats.append({"sha256": r_sha, "energy_kJ_per_mol": r_e, "wall_seconds": r_wall,
                        "identical_to_the_recorded_product": r_sha == main_sha})
        print("  repeat %d: %s %s (%.0f s)" % (k, r_sha[:16],
              "identical" if r_sha == main_sha else "DIFFERENT FROM THE RECORDED PRODUCT", r_wall))

    prod_seq, prod_res, prod_beads, prod_p = parse_side(pdb_out, seq)
    # THE SAMPLED FRAME, not the P trace: the base level is scored on the beads the run actually moved.
    beads_A = diag.get("beads") if isinstance(diag, dict) else None
    if beads_A is None:
        # A CLEAN CHECKOUT LANDS HERE, and it must not refuse. At the time of writing, the sampled frame
        # leaving the refiner in diag["beads"] was an UNCOMMITTED change in a shared working tree, so the
        # committed pipeline returns a diag without it. The product is the same either way -- this script
        # does not feed the sampler -- and the frame the product was actually built from is recoverable
        # from the product's own P/C4'/N atoms, which is what parse_side already read. Which route was
        # taken is a field in the record, and --verify-only checks the one the record claims.
        beads_A = prod_beads
        bead_frame_source = ("read from the product's own P/C4'/N atoms, because the run's diag carried no "
                            "sampled frame (uncommitted at the time of this record)")
        print("  NOTE: " + bead_frame_source)
    else:
        bead_frame_source = "diag['beads'] -- the frame the run sampled"
    beads_A = np.asarray(beads_A, dtype=float).reshape(L, 3, 3)
    # What the repair did, straight out of the run's own log, and the pair list it was given. The pair list
    # is the DOT-BRACKET's, because that is what the refiner derives internally (_dotbracket_to_pairs) and
    # passes to the repair -- the spec's bpp pairs only guide far pairs, and none were added here.
    # torch_gpu_refine imports this from openmm_gpu_refiner (there is no such name in torch_gpu_refine
    # itself -- the first version of this line imported it from there and died after the run, which is a
    # three-minute way to learn where a function lives).
    from torusfold.scheme2.openmm_gpu_refiner import _dotbracket_to_pairs
    db_pairs = [[int(i), int(j)] for i, j, _w in _dotbracket_to_pairs(ss)]
    _m = re.search(r"repair \(chi\): (\d+) bases rotated", text)
    bases_rotated = int(_m.group(1)) if _m else 0

    # The pair list the run consumed, committed beside the product. The source spec lives in results/,
    # which is not in the repository, so without this copy a reader who wants to repeat the prediction has
    # to fold the sequence themselves -- possible, and checked by --check-spec, but not the point of a
    # record that claims to be executable.
    spec_copy = out / "spec.json"
    write_json(spec_copy, {
        "sequence": seq, "secondary_structure": ss,
        "pairs": [[int(i), int(j), float(w)] for i, j, w in pairs_w],
        "note": "sequence, dot-bracket and base-pair probabilities with bpp > 0.1 as folded by ViennaRNA "
                "(scripts/prep_2oiu_input.py). --check-spec re-folds the deposit and compares.",
    })

    # THE REFERENCE ROWS THAT ATTRIBUTE THE GAP. A product's numbers mean nothing on their own: the
    # ordinary question is "how much of this is the sampler and how much is the reconstruction?", and two
    # ceilings answer it. Row 2 is the shipped reconstruction handed the DEPOSIT's own bead frame; row 3 is
    # the shipped reconstruction handed the deposit's own P TRACE, which is the best any P-trace-based tool
    # can do with zero sampling error. Both are deterministic and --verify-only re-derives them, so
    # publishing them costs nothing but honesty -- and without them a reader cannot tell the two apart.
    _dep_beads = beads_of(ref["residues"])
    _dep_recon_res = M.structure_to_residues(reconstruct_all_atom_from_beads(_dep_beads, seq), seq)
    _ptr_recon_res = M.structure_to_residues(
        reconstruct_all_atom(np.asarray(ref["p"], dtype=float), seq,
                             pairs=[(int(i), int(j)) for i, j in ref["pairs"]]), seq)
    tree_files, tree_dirty = tree_fingerprint()

    record = {
        "what": "2OIU through the shipped CG stage, from the deposit, with the calibrated protocol",
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                 "executable": sys.executable,
                 "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "command": " ".join(["python", "scripts/record_2oiu_reuse.py"] + sys.argv[1:]),
        "environment": env, "environment_resolved": resolved,
        "inputs": {
            "deposit": {"path": _rel(DEPOSIT), "sha256": sha256_file(DEPOSIT)},
            "start_pdb": {"path": _rel(start), "sha256": sha256_file(start)},
            "spec": spec_info,
            "spec_copy": {"path": _rel(spec_copy), "sha256": sha256_file(spec_copy)},
            "cg_tables": tables_info,
        },
        "bead_frame_source": bead_frame_source,
        # WHAT TREE THIS CAME FROM. Not decoration: the fingerprint is what lets a reader who gets a
        # different product find out it is a different source tree rather than guess.
        "tree": {"files": tree_files, "uncommitted_files": tree_dirty},
        "tree_note": ("a file listed in uncommitted_files was not in any commit when this record was "
                      "made; a run from the commit alone is a DIFFERENT code path and will not reproduce "
                      "these bytes"),
        "circrna": {"name": "2OIU", "length": L, "sequence": seq, "secondary_structure": ss,
                    "spec_pairs": len(pairs_w), "deposit_wc_pairs": len(ref["pairs"]),
                    "deposit_bsj_A": bsj_A},
        "determinism": {"repeats": repeats,
                        "the_seed_reproduces_the_product_byte_for_byte":
                            bool(repeats) and all(r["identical_to_the_recorded_product"] for r in repeats)},
        "run": {"wall_seconds": wall, "energy_kJ_per_mol": float(energy),
                "product_atoms": int(sum(len(r["atoms"]) for r in prod_res)),
                "on_report_calls": len(calls), "bases_rotated_by_the_repair": bases_rotated},
        "products": {"allatom": {"path": _rel(pdb_out), "sha256": sha256_file(pdb_out)},
                     "cg_trace": {"path": _rel(out / (args.name + "_cg.pdb")),
                                  "sha256": sha256_file(out / (args.name + "_cg.pdb"))}},
        "crystal_pair_list": [[int(i), int(j)] for i, j in ref["pairs"]],
        "dotbracket_pair_list": db_pairs,
        "sampled_beads_angstrom": beads_A.reshape(L, 9).tolist(),
        "measured": {
            "deposit": side_report(_dep_beads, ref["residues"], seq, ref["pairs"], ref["planes"]),
            "deposit_bead_frame_reconstruction": side_report(beads_of(_dep_recon_res), _dep_recon_res,
                                                             seq, ref["pairs"]),
            "deposit_ptrace_reconstruction": side_report(beads_of(_ptr_recon_res), _ptr_recon_res,
                                                         seq, ref["pairs"]),
            "product": side_report(beads_A, prod_res, seq, ref["pairs"]),
        },
    }
    record["measured"]["deposit"]["trace_rmsd_to_deposit_A"] = 0.0
    record["measured"]["product"]["trace_rmsd_to_deposit_A"] = float(
        M.kabsch_rmsd(np.asarray(prod_p, dtype=float), p_crystal))
    p_only = np.asarray(prod_p, dtype=float)
    record["measured"]["product"]["closure_A"] = float(np.linalg.norm(p_only[0] - p_only[-1]))
    record["measured"]["product"]["radius_of_gyration_A"] = float(
        np.sqrt(((p_only - p_only.mean(0)) ** 2).sum(1).mean()))
    write_json(out / "record.json", record)
    print("\ntree: %d files fingerprinted, %d uncommitted%s"
          % (len(tree_files), len(tree_dirty), (" (" + ", ".join(Path(p).name for p in tree_dirty) + ")")
             if tree_dirty else ""), flush=True)

    cols = (("deposit", record["measured"]["deposit"]),
            ("recon(beads)", record["measured"]["deposit_bead_frame_reconstruction"]),
            ("recon(Ptrace)", record["measured"]["deposit_ptrace_reconstruction"]),
            ("PRODUCT", record["measured"]["product"]))
    print("\n%-26s" % "" + "".join("%14s" % c[0] for c in cols))
    for key in ("base_dist_nm_linear_map", "base_rise_nm_linear_map", "base_cos_linear_map",
                "base_dist_nm_ring_atoms", "base_rise_nm_ring_atoms", "base_cos_ring_atoms",
                "stacked_percent", "rise_stacked", "theta_deg", "trace_rmsd_to_deposit_A", "wc_contacts"):
        row = "%-26s" % key
        for _tag, col in cols:
            # .get, because the two attribution rows have no trace deviation of their own: they are built
            # FROM the deposit, so the quantity would be a tautology rather than a measurement.
            v = col.get(key)
            row += ("%14.3f" % v) if isinstance(v, float) else ("%14s" % (v if v is not None else "-"))
        print(row)
    print("\n%s atoms in %.0f s | %s | wrote %s"
          % (record["run"]["product_atoms"], wall, record["products"]["allatom"]["sha256"][:16],
             (out / "record.json").relative_to(REPO)))
    return 0


# ---------------------------------------------------------------- the check

def verify(args):
    out = Path(args.out).resolve()
    rec = json.loads((out / "record.json").read_text(encoding="utf-8"))
    seq = rec["circrna"]["sequence"]
    L = len(seq)
    pairs = [(i, j) for i, j in rec["crystal_pair_list"]]
    beads_A = np.asarray(rec["sampled_beads_angstrom"], dtype=float).reshape(L, 3, 3)

    checks = []
    for key, rel in (("deposit", rec["inputs"]["deposit"]["path"]),
                     ("start_pdb", rec["inputs"]["start_pdb"]["path"]),
                     ("product", rec["products"]["allatom"]["path"]),
                     ("cg_trace", rec["products"]["cg_trace"]["path"])):
        path = _abs(rel)
        want = (rec["inputs"]["deposit"] if key == "deposit" else
                rec["inputs"]["start_pdb"] if key == "start_pdb" else
                rec["products"]["allatom"] if key == "product" else rec["products"]["cg_trace"])["sha256"]
        checks.append(("sha256 %s" % key, path.exists() and sha256_file(path) == want,
                       "%s..." % want[:16]))

    ref = deposit_reference()
    if ref["seq"] != seq:
        raise SystemExit("the deposit now reads %d nt, the record says %d" % (len(ref["seq"]), L))
    prod_seq, prod_res, prod_beads, prod_p = parse_side(_abs(rec["products"]["allatom"]["path"]), seq)

    # THE STRONGEST FORM OF THE BINDING CHECK: rebuild the product from the record's own bead frame with
    # the same code and compare BYTES. Not the anchor atoms -- a least-squares template fit onto three
    # points does not put the template's own P/C4'/N back onto them (measured on this record: 0.76 A max,
    # which is the reconstruction's documented residual, findings Part 25) -- but every atom of the file,
    # which is what the record's sha256 covers.
    if str(rec.get("bead_frame_source", "")).startswith("diag"):
        st = reconstruct_all_atom_from_beads(beads_A, seq)
        if rec["environment"].get("TORUSFOLD_HBOND_REPAIR") == "1":
            n_rot = repair_base_placement(st, seq,
                                          [(int(i), int(j)) for i, j in rec["dotbracket_pair_list"]])
            want_rot = rec["run"].get("bases_rotated_by_the_repair")
            checks.append(("the repair rotates the recorded number of bases", n_rot == want_rot,
                           "recomputed %d, record %s" % (n_rot, want_rot)))
        tmp = out / "_verify_rebuild.pdb"
        write_allatom_pdb(st, str(tmp))
        rebuilt = sha256_file(tmp)
        tmp.unlink()
        checks.append(("rebuilding from the recorded bead frame reproduces the product byte for byte",
                       rebuilt == rec["products"]["allatom"]["sha256"], rebuilt[:16]))
    else:
        # The record says its bead frame was READ FROM the product, so the check that binds them is that
        # the committed file still reads back to the same three atoms per residue.
        dev = float(np.max(np.abs(np.asarray(prod_beads, dtype=float) - beads_A)))
        checks.append(("the product still reads back the recorded bead frame (max dev)",
                       dev <= 1e-9 * max(1.0, float(np.abs(beads_A).max())), "%.3e A" % dev))

    def check_row(tag, want, got):
        for key, g in got.items():
            w = want.get(key)
            if isinstance(w, float) and isinstance(g, float):
                checks.append(("%s %s" % (tag, key), abs(g - w) <= 1e-9 * max(1.0, abs(w)),
                               "record %.6f, recomputed %.6f" % (w, g)))
            elif key == "wc_contact_worst_A":
                checks.append(("%s %s" % (tag, key),
                               len(g) == len(w) and all(abs(a - b) < 1e-9 for a, b in zip(g, w)),
                               "%d distances" % len(g)))
            elif w is not None:
                checks.append(("%s %s" % (tag, key), g == w, "record %s" % w))

    fresh = side_report(beads_A, prod_res, seq, pairs)
    fresh["trace_rmsd_to_deposit_A"] = float(M.kabsch_rmsd(np.asarray(prod_p, dtype=float), ref["p"]))
    check_row("product", rec["measured"]["product"], fresh)

    # The three reference rows, re-derived the same way. They are deterministic and depend only on the
    # deposit and the pair list, so a record whose attribution rows do not come back is a record whose
    # explanation of its own numbers is wrong.
    dep_beads = beads_of(ref["residues"])
    check_row("deposit", rec["measured"]["deposit"],
              side_report(dep_beads, ref["residues"], seq, pairs, ref["planes"]))
    dep_recon = M.structure_to_residues(reconstruct_all_atom_from_beads(dep_beads, seq), seq)
    check_row("deposit_bead_frame_reconstruction", rec["measured"]["deposit_bead_frame_reconstruction"],
              side_report(beads_of(dep_recon), dep_recon, seq, pairs))
    ptr_recon = M.structure_to_residues(
        reconstruct_all_atom(np.asarray(ref["p"], dtype=float), seq, pairs=[(int(i), int(j)) for i, j in pairs]),
        seq)
    check_row("deposit_ptrace_reconstruction", rec["measured"]["deposit_ptrace_reconstruction"],
              side_report(beads_of(ptr_recon), ptr_recon, seq, pairs))

    print("verifying %s (no GPU, no torch, no ViennaRNA)\n" % (out / "record.json").relative_to(REPO))
    bad = 0
    for name, ok, detail in checks:
        if not ok:
            bad += 1
        print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    print("\n%d checks, %d failed" % (len(checks), bad))
    if bad:
        print("THE RECORD DOES NOT VERIFY. Do not quote its numbers.")
        return 1
    print("record verified: the committed files are the ones the record describes, and every number in it "
          "re-derives from them.")
    return 0


def check_spec(args):
    """Prove the pair list the run consumed is ViennaRNA's output on THIS sequence.

    The GPU environment on this machine has no ViennaRNA, so the recorded run was fed a precomputed spec --
    which is only honest if the spec is reproducible. This mode re-folds the deposit's sequence where
    ViennaRNA IS installed and compares the dot-bracket, the pair indices and every base-pair probability
    against the file, so the record's input is checkable rather than asserted.
    """
    ref = deposit_reference()
    path = Path(args.spec)
    spec = json.loads(path.read_text(encoding="utf-8"))
    try:
        import ViennaRNA
    except Exception as exc:                                               # noqa: BLE001
        raise SystemExit("ViennaRNA is not importable here (%s); this mode needs it, and the environment "
                         "that can fold is usually not the one with the GPU" % exc)
    fc = ViennaRNA.fold_compound(ref["seq"])
    ss = fc.mfe()[0]
    fc.pf()
    bpp = fc.bpp()
    pairs = [(i - 1, j - 1, float(bpp[i][j])) for i in range(1, len(ref["seq"]) + 1)
             for j in range(i + 1, len(ref["seq"]) + 1) if bpp[i][j] > 0.1]
    have = [(int(i), int(j), float(w)) for i, j, w in spec["pairs"]]
    # Both spellings, like read_spec: the pipeline's spec says "seq"/"ss", the committed copy says
    # "sequence"/"secondary_structure".
    spec_seq = spec.get("seq") or spec.get("sequence")
    spec_ss = spec.get("ss") or spec.get("secondary_structure")
    checks = [("the spec's sequence is the deposit's", spec_seq == ref["seq"], "%d nt" % len(ref["seq"])),
              ("the dot-bracket reproduces", spec_ss == ss, ss),
              ("the pair list reproduces", [(i, j) for i, j, _ in have] == [(i, j) for i, j, _ in pairs],
               "%d pairs" % len(pairs))]
    if len(have) == len(pairs):
        wmax = max(abs(a[2] - b[2]) for a, b in zip(have, pairs))
        checks.append(("every base-pair probability reproduces", wmax == 0.0, "max |d bpp| = %.1e" % wmax))
    # ViennaRNA exposes no __version__ (the obvious getattr returns "?"), so ask the distribution.
    try:
        import importlib.metadata as md
        vna_version = md.version("ViennaRNA")
    except Exception:                                                      # noqa: BLE001
        vna_version = "version unknown"
    print("checking %s against ViennaRNA %s" % (path, vna_version))
    bad = sum(1 for _n, ok, _d in checks if not ok)
    for name, ok, detail in checks:
        print("  [%s] %-44s %s" % ("PASS" if ok else "FAIL", name, detail))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default=str(SPEC),
                    help="precomputed seq/ss/pairs JSON (default: results/plan_c/_2oiu_input.json, which a "
                         "GPU environment without ViennaRNA needs); pass --spec '' to fold with ViennaRNA")
    ap.add_argument("--steps", type=int, default=1000, help="TORUSFOLD_REFINE_STEPS")
    ap.add_argument("--repair", action="store_true", help="also switch on TORUSFOLD_HBOND_REPAIR=1")
    ap.add_argument("--name", default="2oiu_reuse")
    ap.add_argument("--tables", default=None,
                    help="production_tables.npz to run with, when results/ is not present")
    ap.add_argument("--repeat", type=int, default=0,
                    help="re-run the same protocol N more times into a scratch directory and record "
                         "whether the product reproduces byte for byte")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-only", action="store_true", dest="verify_only")
    ap.add_argument("--check-spec", action="store_true", dest="check_spec",
                    help="re-fold with ViennaRNA and compare against --spec (needs ViennaRNA, no GPU)")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(OUT_REPAIR if args.repair else OUT_DEFAULT)
    if args.spec == "":
        args.spec = None
    if args.check_spec:
        return check_spec(args)
    return verify(args) if args.verify_only else run(args)


if __name__ == "__main__":
    sys.exit(main())
