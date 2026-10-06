"""Report a pipeline run's outputs: checkpoint, all-atom products, and the
cross-check between the pairing it predicted and the pairing its coordinates form.

Use after tools/run_pipeline_with_checkpoint.py has finished (or on any run
directory), so the answer to "did this produce what the immune fingerprint needs"
is one command rather than a session of digging.

Usage:
    python tools/report_run.py results/immuno_full
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from torusfold.immuno.checkpoint_pairs import load_checkpoint_pairing  # noqa: E402
from torusfold.immuno.pair_graph_from_coords import build_pair_graph  # noqa: E402


def atom_profile(p: pathlib.Path) -> tuple[int, int, float]:
    atoms = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines()
             if l.startswith(("ATOM", "HETATM"))]
    res: dict = collections.OrderedDict()
    for l in atoms:
        res.setdefault((l[21], l[22:27]), l[17:20].strip())
    return len(atoms), len(res), len(atoms) / max(1, len(res))


def main() -> int:
    run = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "immuno_full"
    if not run.is_absolute():
        run = REPO / run
    if not run.is_dir():
        print(f"no such run directory: {run}")
        return 2

    print("=" * 78)
    print(f"run: {run.relative_to(REPO)}")
    print("=" * 78)

    log = run / "_run.log"
    if log.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
        print("\nlast log lines:")
        for l in tail:
            print(f"  {l[:110]}")
    pid_file = run / "_run.pid"
    if pid_file.is_file():
        pid = pid_file.read_text().strip()
        alive = pathlib.Path(f"/proc/{pid}").exists() or _pid_alive(pid)
        print(f"\nrunner pid {pid}: {'ALIVE' if alive else 'not running'}")

    # -- 1. checkpoint ------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. checkpoint  (the pairing the pipeline decided)")
    print("=" * 78)
    ck = None
    try:
        ck = load_checkpoint_pairing(run)
        print(json.dumps(ck.summary(), indent=2, ensure_ascii=False))
        print(f"\n  longest helix : {max(ck.helix_lengths() or [0])} bp "
              f"(PKR needs ~30 bp)")
    except Exception as e:  # noqa: BLE001
        print(f"  unavailable: {type(e).__name__}: {e}")

    # -- 2. what the run wrote ---------------------------------------------
    print("\n" + "=" * 78)
    print("2. structures written  (is any of it all-atom?)")
    print("=" * 78)
    pdbs = sorted(run.rglob("*.pdb"))
    if not pdbs:
        print("  no PDB files")
    best = None
    for p in pdbs:
        try:
            n_at, n_res, ratio = atom_profile(p)
        except Exception:  # noqa: BLE001
            continue
        if n_at == 0:
            continue
        tag = "ALL-ATOM" if ratio > 5 else ("P-only" if ratio < 2 else "partial")
        print(f"  {tag:<9} {ratio:5.1f} atoms/res  {n_at:6d} atoms  "
              f"{p.relative_to(run)}")
        if ratio > 5 and (best is None or n_at > best[1]):
            best = (p, n_at)

    # -- 3. the cross-check -------------------------------------------------
    print("\n" + "=" * 78)
    print("3. does the structure actually form the pairing it predicted?")
    print("=" * 78)
    if ck is None:
        print("  skipped: no checkpoint")
    else:
        # Every all-atom file gets checked, because the run writes several and
        # they are not equivalent -- cg2aa/merged_aa.pdb is a per-segment product
        # while final_allatom.pdb is the end of the chain.
        candidates = sorted(p for p in pdbs if atom_profile(p)[2] > 5)
        if not candidates:
            print("  skipped: no all-atom structure in this run")
        for path in candidates:
            g = build_pair_graph(path.read_text(encoding="utf-8"), is_circular=True)
            ck_set = {(min(i, j), max(i, j)) for i, j, _ in ck.pairs}
            co_set = {(min(p.key, p.partner), max(p.key, p.partner)) for p in g.pairs}
            both = ck_set & co_set
            reasons = collections.Counter(r["reason"].split(" ")[0] for r in g.rejected)
            print(f"\n  --- {path.relative_to(run)}")
            print(f"      residues {len(g.residues)}   helix runs {len(g.helix_runs())}")
            print(f"      checkpoint pairs {len(ck_set)}   coordinate pairs {len(co_set)}"
                  f"   in both {len(both)}")
            print(f"      candidates considered {len(g.rejected)}   "
                  f"rejection reasons {dict(reasons)}")
            # "missing" is not a geometry verdict: it means the file lacks the
            # base ring atoms the pair test needs, so nothing could be decided.
            if reasons.get("missing", 0) == len(g.rejected) and g.rejected:
                print("      NOTE: every candidate failed for MISSING ATOMS, not for")
                print("      geometry. This file has no base ring atoms, so the pair")
                print("      graph cannot be computed on it at all -- it is not")
                print("      evidence that no pairs exist.")
            idx = sorted(g.residues)
            if idx:
                coords_ = np.vstack([g.residues[i].c1p for i in idx])
                from scipy.spatial import cKDTree  # noqa: PLC0415

                d, _ = cKDTree(coords_).query(coords_, k=2)
                nn = d[:, 1]
                print(f"      nearest C1'--C1'  min {nn.min():.2f}  "
                      f"median {np.median(nn):.2f}  max {nn.max():.2f} A")
                print(f"      a Watson-Crick pair needs ~10.4 A; the largest nearest-"
                      f"neighbour distance is {'BELOW' if nn.max() < 10.4 else 'above'} that")
                if nn.min() < 3.0:
                    print(f"      WARNING: a {nn.min():.2f} A C1'--C1' contact is"
                          f" physically impossible (shorter than a C-C bond)")

    # -- verdict ------------------------------------------------------------
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    has_pairs = ck is not None and bool(ck.pairs or ck.stem_blocks)
    has_aa = best is not None
    print(f"  checkpoint pairing present      : {'YES' if has_pairs else 'NO'}")
    print(f"  all-atom structure present      : {'YES' if has_aa else 'NO'}")
    print()
    print("  Both halves exist as FILES, which is what makes the fingerprint")
    print("  computable in principle. Whether it should be computed is a separate")
    print("  question -- see section 3. If no predicted pair is formed by the")
    print("  coordinates, the pairing half describes an intent the structure does")
    print("  not realise, and any fingerprint built on it inherits that gap.")
    print("=" * 78)
    return 0


def _pid_alive(pid: str) -> bool:
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=30).stdout
        return pid in out
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())
